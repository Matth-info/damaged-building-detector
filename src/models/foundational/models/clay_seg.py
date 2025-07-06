from __future__ import annotations

"""Clay Segmentor for semantic segmentation tasks.

Attribution:
Decoder from Segformer: Simple and Efficient Design for Semantic Segmentation
with Transformers
Paper URL: https://arxiv.org/abs/2105.15203
"""  # noqa: N999
import logging
import re
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from src.models.foundational.backbones.clay import ClayEncoder
from src.models.foundational.utils.download import download_file

hidden_dim = 512
c_out = 64


class FCNHead(nn.Module):
    """Fully Convolutional Network Head."""

    def __init__(
        self,
        enc_dim: int,
        encoder_patch_size: int,
        num_classes: int,
        channel_out: int = 64,
        hidden_dim: int = 512,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the FCNHead.

        Args:
            enc_dim (int): Number of input channels from the encoder.
            encoder_patch_size (int): Patch size used by the encoder.
            num_classes (int): Number of output classes.
            channel_out (int, optional): Number of output channels after pixel shuffle. Defaults to 64.
            hidden_dim (int, optional): Hidden dimension size. Defaults to 512.
        """
        # Define layers after the encoder
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(enc_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv_ps = nn.Conv2d(
            hidden_dim,
            channel_out * encoder_patch_size * encoder_patch_size,
            kernel_size=3,
            padding=1,
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=encoder_patch_size)
        self.conv_out = nn.Conv2d(channel_out, num_classes, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Pass through convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv_ps(x)  # [b, c_out * r^2, h', w']
        # Upsample using PixelShuffle
        x = self.pixel_shuffle(x)  # [b, c_out, h_in, w_in]

        # Final convolution to get desired output channels
        return self.conv_out(x)  # [b, num_outputs, h_in, w_in]


class SegmentEncoder(ClayEncoder):
    """SegmentEncoder extends ClayEncoder for use in semantic segmentation tasks."""

    def __init__(
        self,
        mask_ratio: float,
        patch_size: int,
        *,
        shuffle: bool,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_ratio: float,
        ckpt_path: str | None = None,
    ) -> None:
        """Initialize the Clay_Seg model with the specified configuration.

        Args:
            mask_ratio (float): The ratio of patches to mask during training.
            patch_size (int): The size of each image patch.
            shuffle (bool): Whether to shuffle the patches.
            dim (int): The dimensionality of the model embeddings.
            depth (int): The number of transformer layers.
            heads (int): The number of attention heads in each transformer layer.
            dim_head (int): The dimensionality of each attention head.
            mlp_ratio (float): The ratio for the hidden dimension in the MLP block.
            ckpt_path (str | None, optional): Path to a checkpoint file to load model weights from. Defaults to None.

        Notes:
            - Sets the device to CUDA if available, otherwise uses CPU.
            - Loads model weights from the specified checkpoint if provided.
        """
        super().__init__(
            mask_ratio,
            patch_size,
            shuffle,
            dim,
            depth,
            heads,
            dim_head,
            mlp_ratio,
        )

        # Set device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # Load model from checkpoint if provided
        self.load_from_ckpt(ckpt_path)

    def load_from_ckpt(self, ckpt_path: str | None = None) -> None:
        """Load the model's state from a checkpoint file.

        Args:
            ckpt_path (str): The path to the checkpoint file.
        """
        if ckpt_path:
            # Load checkpoint
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state_dict = ckpt.get("state_dict")

            # Prepare new state dict with the desired subset and naming
            new_state_dict = {
                re.sub(r"^model\.encoder\.", "", name): param
                for name, param in state_dict.items()
                if name.startswith("model.encoder")
            }

            _ = self.load_state_dict(new_state_dict, strict=False)
            logging.info("Pre-trained Weights from Clay FM encoder have been successful loaded.")
            # Freeze the loaded parameters
            for name, param in self.named_parameters():
                if name in new_state_dict:
                    param.requires_grad = False
            logging.info("SegmentEncoder weights have been frozen.")

    def forward(self, datacube: dict) -> torch.Tensor:
        """Forward pass of the SegmentEncoder.

        Args:
            datacube (dict): A dictionary containing the input datacube and
                meta information like time, latlon, gsd & wavelenths.

        Returns:
            list: A list of feature maps extracted from the datacube.
        """
        cube, time, latlon, gsd, waves = (
            datacube["pixels"],  # [b c h w]
            datacube["time"],  # [b 2]
            datacube["latlon"],  # [b 2]
            datacube["gsd"],  # 1
            datacube["waves"],  # [N]
        )

        b, c, h, w = cube.shape

        # Patchify and create embeddings per patch
        patches, waves_encoded = self.to_patch_embed(cube, waves)  # [b L d]
        patches = self.add_encodings(patches, time, latlon, gsd)  # [b L d]

        # Add class tokens
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)  # [b 1 d]
        patches = torch.cat((cls_tokens, patches), dim=1)  # [b (1 + L) d]

        patches = self.transformer(patches)
        return patches[:, 1:, :]  # [b L d]


class ClaySegmentor(nn.Module):
    """Clay Segmentor class that combines the ClayEncoder with FPN layers for semanticsegmentation.

    Attributes:
        num_classes (int): Number of output classes for segmentation.
        feature_maps (list): Indices of layers to be used for generating feature maps.
        ckpt_path (str): Path to the checkpoint file.
    """

    def __init__(
        self,
        num_classes: int,
        version: Literal["1", "1.5", None] = None,
        *,
        freeze_backbone: bool = True,
        checkpoint_folder: str | None = None,
    ) -> None:
        """Initialize Clay Segmentor."""
        # Default values are for the clay mae base model.
        super().__init__()
        weights_dir = Path(checkpoint_folder) / "models" / "clay"
        weights_dir.mkdir(parents=True, exist_ok=True)

        if checkpoint_folder:
            if version == "1.5":
                logging.info("Downloading Clay v1.5 Encoder.")
                weights_path = weights_dir / "clay-v1.5.ckpt"
                download_file(
                    "https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt?download=true",
                    weights_path,
                    always_download=False,
                )
            elif version == "1":
                logging.info("Downloading Clay v1 Encoder.")
                weights_path = weights_dir / "clay-v1-base.ckpt"
                download_file(
                    "https://huggingface.co/made-with-clay/Clay/resolve/main/v1/clay-v1-base.ckpt?download=true",
                    weights_path,
                    always_download=False,
                )
            else:
                logging.info("Encoder Random weight initialization.")

        encoder = SegmentEncoder(
            mask_ratio=0.0,
            patch_size=8,
            shuffle=False,
            dim=1024,
            depth=24,
            heads=16,
            dim_head=64,
            mlp_ratio=4.0,
            ckpt_path=weights_path,
        )
        if freeze_backbone:
            # Freeze the encoder parameters
            for param in encoder.parameters():
                param.requires_grad = False
        else:
            # Freeze the encoder parameters
            for param in encoder.parameters():
                param.requires_grad = True

        if Path.is_file(weights_path):
            encoder.load_from_ckpt(weights_path)

        self.encoder = encoder

        # Define the layer head
        self.head = FCNHead(
            enc_dim=self.encoder.dim,
            encoder_patch_size=self.encoder.patch_size,
            num_classes=num_classes,
            hidden_dim=512,
            channel_out=64,
        )

    def forward(self, datacube: dict) -> torch.Tensor:
        """Forward pass of the Segmentor.

        Args:
            datacube (dict): A dictionary containing the input datacube and meta information like time, latlon, gsd & wavelenths.

        Returns:
            torch.Tensor: The segmentation logits.
        """
        cube = datacube["pixels"]  # [b c h_in w_in]
        b, c, h_in, w_in = cube.shape

        # Get embeddings from the encoder
        patches = self.encoder(datacube)  # [b, L, d]

        # Reshape embeddings to [b, d, h', w']
        h_patches = h_in // self.encoder.patch_size
        w_patches = w_in // self.encoder.patch_size
        x = rearrange(patches, "b (h w) d -> b d h w", h=h_patches, w=w_patches)

        return self.head(x)
