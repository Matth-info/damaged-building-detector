from __future__ import annotations

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained Scale-MAE models."""

import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import einops
import torch
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor, nn

_ori_img_size = 224

from src.models.foundational.backbones.scale_mae import _ori_img_size
from src.models.foundational.utils.download import download_file
from src.models.foundational.utils.pos_embed import get_2d_sincos_pos_embed_with_resolution
from src.models.foundational.utils.seg_heads import Seg_Conv_Decoder

if TYPE_CHECKING:
    from collections import OrderedDict


class ScaleMAE(VisionTransformer):
    """Custom Vision Transformer for Scale-MAE with GSD positional embeddings. (TorchGeo Implementation).

    This is a ViT encoder only model of the Scale-MAE architecture with GSD positional embeddings.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2212.14532
    """

    def __init__(self, res: float = 1.0, *args: Any, **kwargs: Any) -> None:
        """Initialize a new ScaleMAE model.

        Args:
            res: Spatial resolution of the image in meters.
            *args: Additional arguments to
                pass to :class:`timm.models.vision_transformer.VisionTransformer`.
            **kwargs: Additional keyword arguments to
                pass to :class:`timm.models.vision_transformer.VisionTransformer`.
        """
        super().__init__(*args, **kwargs)

        self.res = res

        # Scale MAE uses resolution specific positional embeddings
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False

    def _pos_embed(self, x: Tensor) -> Tensor:
        """Apply GSD positional embeddings to the input tensor."""
        res = torch.tensor(self.res, dtype=x.dtype, device=x.device)
        res = res.repeat(x.shape[0])
        pos_embed = (
            get_2d_sincos_pos_embed_with_resolution(
                self.embed_dim,
                int(self.patch_embed.num_patches**0.5),
                res,
                cls_token=True,
            )
            .to(x.dtype)
            .to(x.device)
        )
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + pos_embed
        return self.pos_drop(x)


def interpolate_pos_embed(
    model: ScaleMAE, state_dict: OrderedDict[str, Tensor]
) -> OrderedDict[str, Tensor]:
    """Interpolate the positional embeddings if image size is different than pretrained image size.

    Args:
        model: ScaleMAE model.
        state_dict: Pretrained model state dict.

    Returns:
        state_dict: State dict with interpolated positional embeddings.
    """
    pos_embed_checkpoint = state_dict["pos_embed"]
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = 0
    if model.pos_embed is not None:
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches**0.5)
    # class_token and dist_token are kept unchanged
    if orig_size != new_size:
        print(
            f"Interpolating positional embeddings from {orig_size}x{orig_size} to {new_size}x{new_size}"
        )
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(
            0, 3, 1, 2
        )
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        state_dict["pos_embed"] = new_pos_embed

    return state_dict


class ScaleMaeSeg(nn.Module):
    """ScaleMAE for semantic segmentation task."""

    def __init__(
        self,
        img_size: tuple[int, int] | None = None,
        head_mode: Literal["diff", "conc"] = "diff",
        *,
        freeze_backbone: bool = True,
        num_classes: int = 1,
        decoder_type: Literal[None, "res_block"] = None,
        depth: int = 4,
        res: float = 1.0,
        checkpoint_folder: str | None = None,
    ) -> None:
        """Initialize the Scale MAE Segmentation model.

        Args:
            img_size (tuple[int, int] | None, optional): Input image size as (height, width). If None, uses default size. Defaults to None.
            head_mode (Literal["diff", "conc"], optional): Mode for the segmentation head, either "diff" or "conc". Defaults to "diff".
            freeze_backbone (bool, optional): If True, freezes the encoder (backbone) parameters during training. Defaults to True.
            num_classes (int, optional): Number of output segmentation classes. Defaults to 1.
            decoder_type (Literal[None, "res_block"], optional): Type of decoder block to use. If "res_block", uses residual blocks. Defaults to None.
            depth (int, optional): Number of decoder layers. Defaults to 4.
            res (float, optional): Resolution scaling factor for the encoder. Defaults to 1.0.
            checkpoint_folder (str, optional): Directory to store or load model checkpoints. Defaults to ".".

        Raises:
            FileNotFoundError: If the pretrained weights cannot be found or downloaded.
            RuntimeError: If loading the encoder state dict fails.

        Example:
                model = ScaleMaeSeg(
                    img_size=(224, 224),
                    head_mode="diff",
                    num_classes=5,
                    decoder_type="res_block",
                    res=0.3,
                    checkpoint_folder="..",
                    depth=4,
                )
        """
        super().__init__()

        self.mode = head_mode
        if checkpoint_folder:
            weights_dir = Path(checkpoint_folder) / "models" / "scale_mae"
            weights_dir.mkdir(parents=True, exist_ok=True)
            weights_path = weights_dir / "vit_large_patch16_224_fmow_rgb_scalemae.pth"

            download_file(
                url="https://huggingface.co/torchgeo/vit_large_patch16_224_fmow_rgb_scalemae/resolve/main/vit_large_patch16_224_fmow_rgb_scalemae-98ed9821.pth?download=true",
                filename=weights_path,
                retries=3,
                always_download=False,
            )
        else:
            weights_path = None

        encoder = ScaleMAE(
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            res=res,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        if Path.is_file(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
            logging.info("Encoder Weights have been downloaded and correctly loaded.")
            if img_size is not None and img_size != _ori_img_size:
                state_dict = interpolate_pos_embed(encoder, state_dict)
                logging.info("Interpolate Positional Embedding for input size compatibility.")

            encoder.load_state_dict(state_dict, strict=False)

        self.encoder = encoder

        if freeze_backbone:
            # Freeze the encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            # UnFreeze the encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = True

        # estimate encoder output weight and width
        self.H_in, self.W_in = self.encoder.patch_embed.dynamic_feat_size(img_size)

        self.decoder = Seg_Conv_Decoder(
            in_channels=self.encoder.embed_dim,
            out_channels=num_classes,
            input_size=(self.H_in, self.W_in),
            target_size=img_size,
            head_type=head_mode,
            depth=depth,
            up_block=decoder_type,
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        feat_1 = self.encoder.forward_features(x1)[:, 1:, :]  # remove class token
        feat_2 = self.encoder.forward_features(x2)[:, 1:, :]  # remove class token

        # reshape
        feat_1 = einops.rearrange(feat_1, "b (h w) d -> b d h w", h=self.H_in, w=self.W_in)
        feat_2 = einops.rearrange(feat_2, "b (h w) d -> b d h w", h=self.H_in, w=self.W_in)

        return self.decoder(feat_1, feat_2)
