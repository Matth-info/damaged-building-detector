# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
from functools import partial

import torch
from einops import rearrange
from timm.models.vision_transformer import Block, PatchEmbed
from torch import Tensor, nn

from src.models.foundational.utils.pos_embed import (
    get_2d_sincos_pos_embed,
    get_2d_sincos_pos_embed_with_resolution,
)
from src.models.foundational.utils.scale_layers import (
    Block as GPTBlock,
)
from src.models.foundational.utils.scale_layers import (
    FCNHead,
    FPNHead,
    MAEDecoder,
)

_ori_img_size = 224


class PatchEmbedUnSafe(PatchEmbed):
    """Image to Patch Embedding."""

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.proj(x).flatten(2).transpose(1, 2)


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone.

    Args:
        img_size (int): Input image size (height and width).
        patch_size (int): Patch size for patch embedding.
        in_chans (int): Number of input channels.
        embed_dim (int): Embedding dimension for encoder.
        depth (int): Number of encoder transformer blocks.
        num_heads (int): Number of attention heads in encoder.
        decoder_embed_dim (int): Embedding dimension for decoder.
        decoder_depth (int): Number of decoder transformer blocks.
        decoder_num_heads (int): Number of attention heads in decoder.
        decoder_aux_loss_layers (int): Number of auxiliary loss layers in decoder.
        mlp_ratio (float): MLP hidden dimension ratio.
        norm_layer (Type[nn.Module]): Normalization layer.
        norm_pix_loss (bool): If True, normalize pixel loss.
        use_mask_token (bool): If True, use mask token in decoder.
        project_pos_emb (bool): If True, project positional embeddings in decoder.
        loss_masking (bool): If True, apply masking to loss computation.
        self_attention (bool): If True, use self-attention in encoder.
        absolute_scale (bool): If True, use absolute scale for positional embeddings.
        target_size (list[int]): List of target sizes for multi-scale decoding.
        fixed_output_size (Optional[int]): Fixed output size for decoding.
        fcn_dim (int): Dimension for FCN head.
        fcn_layers (int): Number of layers in FCN head.
        independent_fcn_head (bool): If True, use independent FCN heads for multi-scale.
        use_l1_loss (bool): If True, use L1 loss for high-frequency reconstruction.
        l1_loss_weight (float): Weight for L1 loss.
        band_config (list[int]): Band configuration for multi-scale (e.g., [14, 224]).
        progressive (bool): If True, use progressive FPN head.

    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        decoder_aux_loss_layers: int = 0,
        mlp_ratio: float = 4.0,
        norm_layer: type = nn.LayerNorm,
        *,
        norm_pix_loss: bool = False,
        use_mask_token: bool = False,
        project_pos_emb: bool = False,
        loss_masking: bool = True,
        self_attention: bool = False,
        absolute_scale: bool = False,
        target_size: list[int] = [],
        fixed_output_size: int | None = None,
        fcn_dim: int = 256,
        fcn_layers: int = 3,
        independent_fcn_head: bool = False,
        use_l1_loss: bool = False,
        l1_loss_weight: float = 1.0,
        band_config: tuple[int, ...] = (14, 224),
        progressive: bool = False,
    ):
        """Initialize a Masked Autoencoder with VisionTransformer backbone model."""
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        if len(band_config) != 2:
            raise KeyError("2 elements int band config are required")
        self.use_l1_loss = use_l1_loss
        self.l1_loss_weight = l1_loss_weight
        self.band_config = list(band_config)
        self.patch_size = patch_size
        if fixed_output_size % patch_size != 0:
            msg = "Fixed Output size must be divisible by patch size, here we got {fixed_output_size} and  {patch_size}."
            raise ValueError(msg)
        self.fixed_output_size = fixed_output_size // patch_size
        self.multiscale = len(target_size) > 1
        self.patch_embed = PatchEmbedUnSafe(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.self_attention = self_attention
        self.absolute_scale = absolute_scale
        self.target_size = target_size
        self.independent_fcn_head = independent_fcn_head
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token_decoder = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.use_mask_token = use_mask_token
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding
        self.project_pos_emb = project_pos_emb
        if project_pos_emb:
            self.pos_emb_projection = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=True)
        self.fpn = FPNHead(decoder_embed_dim, share_weights=progressive)
        if independent_fcn_head:
            self.fcn_high = FCNHead(decoder_embed_dim, fcn_dim, fcn_layers, 3)
            self.fcn_low = FCNHead(decoder_embed_dim, fcn_dim, fcn_layers, 3)
        else:
            self.fcn = FCNHead(decoder_embed_dim, fcn_dim, fcn_layers, 3)
        # Depending on the mode of decoding we are using, the decoder architecture is different
        if self.multiscale:
            self.decoder_blocks = nn.ModuleList(
                [
                    GPTBlock(
                        decoder_embed_dim,
                        decoder_num_heads,
                        mlp_ratio,
                        qkv_bias=True,
                        norm_layer=norm_layer,
                    )
                    for _ in range(decoder_depth)
                ]
            )
        else:
            self.decoder_blocks = nn.ModuleList(
                [
                    Block(
                        decoder_embed_dim,
                        decoder_num_heads,
                        mlp_ratio,
                        qkv_bias=True,
                        norm_layer=norm_layer,
                    )
                    for _ in range(decoder_depth)
                ]
            )

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.norm_pix_loss = norm_pix_loss
        self.loss_masking = loss_masking

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize (and freeze) pos_embed by sin-cos embedding."""
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        if self.use_mask_token:
            torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs: Tensor) -> Tensor:
        """Transform a batch of images into a batch of set of flatten patches.

        Args:
            imgs: (b, 3, H, W)

        Returns:
            x: (b, le, patch_size**2 *3) where le : number of patches per image.
        """
        p = self.patch_embed.patch_size[0]
        b, c, h, w = imgs.shape
        if h != w or h % p != 0:
            raise ValueError("Image dimensions are incompatible with patches size.")

        return rearrange(imgs, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)

    def unpatchify(self, x: Tensor) -> Tensor:
        """Reconstruct images from flattened patches.

        Args:
            x: (n, le, patch_size**2 * 3)

        Returns:
            imgs: (n, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        n, le, _ = x.shape
        h = w = int(le**0.5)
        if h * w != le:
            raise ValueError("Input does not have a square number of patches")

        return rearrange(x, "n (h w) (p1 p2 c) -> n c (h p1) (w p2)", h=h, w=w, p1=p, p2=p, c=3)

    def upsample_decoder(self, x: Tensor, target_dim: int) -> Tensor:
        """Upsample the input tensor `x` to match the target spatial dimension `target_dim`.

        Args:
            x (Tensor): Input tensor of shape (n, le, num_patches**2, decoder_embed_dim).
            target_dim (int): The target spatial dimension to upsample to (decoder_num_patches).

        Returns:
        Tensor: Upsampled and reshaped tensor of shape (n, decoder_num_patches**2, decoder_embed_dim).
        """
        p = target_dim
        x = x.unsqueeze(dim=1)
        n, _, l_low, _ = x.shape
        l_low_dim = int(l_low**0.5)
        x = torch.nn.functional.interpolate(
            input=x.reshape(n, 1, l_low_dim, l_low_dim, self.decoder_embed_dim),
            size=(p, p, self.decoder_embed_dim),
            mode="nearest",
        ).view(n, 1, p**2, self.decoder_embed_dim)
        return x.squeeze(dim=1)

    def find_closest_multiple(self, target_resolution: float) -> int:
        """Find the closest multiple."""
        n = target_resolution + self.patch_embed.patch_size[0] / 2
        n = n - (n % self.patch_embed.patch_size[0])
        return int(n)

    def plot_decoder_vector(self, x: Tensor) -> Tensor:
        """Plot Decoder Vector."""
        b, total_patches, _ = x.shape
        num_patches_per_axis = int(total_patches**0.5)
        patch_size = self.patch_embed.patch_size[0]
        embed_dim = self.decoder_embed_dim

        output_raster = torch.zeros(b, num_patches_per_axis * embed_dim, num_patches_per_axis)

        data = x.reshape(b, num_patches_per_axis, num_patches_per_axis, embed_dim)  # 4, 7, 7, 512

        data = data.permute(0, 3, 1, 2)  # 4, 512, 7, 7

        for img in range(b):
            for i in range(embed_dim):
                output_raster[
                    img, i * num_patches_per_axis : (i + 1) * num_patches_per_axis, :
                ] = data[img, i, :, :]

        return output_raster

    def random_masking(self, x: Tensor, mask_ratio: float) -> Tensor:
        """Randomly masks a portion of the input sequence for each sample in the batch.

        Performs per-sample random masking by shuffling the sequence using random noise,
        then selecting a subset to keep and generating a corresponding binary mask.

        Args:
            x (Tensor): Input tensor of shape [n, le, d], where n is the batch size,
                le is the sequence length, and d is the feature dimension.
            mask_ratio (float): Fraction of the sequence to mask (between 0 and 1).

        Returns:
            x_masked (Tensor): Masked input tensor of shape [n, le * (1 - mask_ratio), d].
            mask (Tensor): Binary mask tensor of shape [n, le], where 0 indicates kept tokens \
                and 1 indicates masked tokens, restored to the original order.
            ids_restore (Tensor): Indices to restore the original sequence order, of shape [n, le].
        """
        n, le, d = x.shape  # batch, length, dim
        len_keep = int(le * (1 - mask_ratio))

        noise = torch.rand(n, le, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset of patches
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([n, le], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(
        self, x: Tensor, mask_ratio: float = 0.0, input_res: float | None = None
    ) -> Tensor:
        """Encoder Forward pass."""
        # embed patches
        _, _, h, w = x.shape
        x = self.patch_embed(x)
        input_res = input_res.cpu()

        num_patches = int(
            (h * w) / (self.patch_embed.patch_size[0] * self.patch_embed.patch_size[1])
        )
        pos_embed = get_2d_sincos_pos_embed_with_resolution(
            x.shape[-1],
            int(num_patches**0.5),
            input_res,
            cls_token=True,
            device=x.device,
        )

        x = x + pos_embed[:, 1:, :]  # add positional encoding w/o class token pos encoding

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Added back to the mask token in decoder for decoding modes != "demasking"
        pos_embed_encoder = get_2d_sincos_pos_embed_with_resolution(
            self.decoder_embed_dim,
            int(num_patches**0.5),
            input_res,
            cls_token=True,
            device=x.device,
        )

        return x, mask, ids_restore, pos_embed_encoder

    def forward_decoder(
        self,
        x: Tensor,
        ids_restore: Tensor | None = None,
        target_res: float | None = None,
        target_dim: int | None = None,
        pos_embed_encoder: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Decoder forward pass.

        Args:
            x: decoder input tokens (n, le, d)
            ids_restore: indices to restore masked tokens (n, L_full)
            target_res: target resolution (H, W)
            target_dim: output dimension (e.g., image size)
            pos_embed_encoder: positional embeddings from encoder
            mask: binary mask used during encoding
        """
        ids_restore = ids_restore if ids_restore is not None else []
        target_res = target_res if target_res is not None else [14, 14]  # fallback
        if target_dim is None:
            raise AttributeError("target_dim must be provided")

        # Decoder input embedding
        x = self.decoder_embed(x)  # (n, le, d)

        # Prepare positional embedding
        pos_embed = get_2d_sincos_pos_embed_with_resolution(
            embed_dim=x.shape[-1],
            img_size=target_dim,
            grid_size=target_res,
            cls_token=True,
            device=x.device,
        )

        if ids_restore is not None and len(ids_restore) > 0:
            # Add mask tokens to sequence
            b, l_visible, d = x.shape
            num_patches = ids_restore.shape[1]
            num_mask = num_patches + 1 - l_visible  # +1 for cls token

            mask_tokens = self.mask_token.repeat(b, num_mask, 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # exclude cls token
            x_ = torch.gather(
                x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, d)
            )  # restore
            x = torch.cat([x[:, :1, :], x_], dim=1)  # add cls token back

        if pos_embed_encoder is not None:
            x = x + pos_embed_encoder

        # Optionally remove masked tokens
        if not self.use_mask_token and mask is not None:
            num_masked = (mask == 0).sum(-1).min().item()
            mask_idx = torch.argsort(mask, dim=-1, descending=True)[:, :num_masked]
            batch_indices = (
                torch.arange(x.size(0), device=x.device).unsqueeze(1).expand(-1, num_masked)
            )
            x = x[batch_indices, mask_idx]

        # Project positional embeddings if needed
        pos_embed_raw = pos_embed
        if self.project_pos_emb:
            pos_embed = self.pos_emb_projection(pos_embed)

        # Drop cls token before decoding
        x = x[:, 1:, :]  # (n, le, d)

        # Decode through transformer blocks
        n, le, d = x.shape
        p = int(le**0.5)
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)
        x = x.view(n, p, p, d).permute(0, 3, 1, 2).contiguous()  # (b, C, H, W)

        # Decode to FPN features
        x = self.fpn(x)
        if self.independent_fcn_head:
            x = [self.fcn_high([x[0]])[0], self.fcn_low([x[1]])[0]]
        else:
            x = self.fcn(x)

        return x, pos_embed_raw, None

    def split_pred(
        self, target_dim: int, pred: tuple[Tensor, Tensor], mean: Tensor, var: Tensor
    ) -> Tensor:
        """Split and processes prediction tensors into high, low, and combined resolutions, then patchifies each.

        Args:
            target_dim (int): The target dimension for processing (not used directly in this function).
            pred (Tuple[Tensor, Tensor]): A tuple containing the high-resolution and low-resolution prediction tensors.
            mean (Tensor): The mean tensor (not used directly in this function).
            var (Tensor): The variance tensor (not used directly in this function).

        Returns:
            List[Tensor]: A list containing the patchified high-resolution, low-resolution, and combined prediction tensors.
        """
        pred_high, pred_low = pred
        pred_all = (
            nn.functional.interpolate(
                nn.functional.interpolate(
                    pred_low, (self.band_config[0], self.band_config[0]), mode="area"
                ),
                pred_high.shape[-2:],
                mode="bilinear",
            )
            + pred_high
        )
        return [self.patchify(x) for x in [pred_high, pred_low, pred_all]]

    @classmethod
    def random_crop(
        cls, seq: Tensor, target_size: int, *, cls_token: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        """Randomly crops a sequence to the target size, optionally preserving the class token.

        Args:
            seq (Tensor): Input sequence tensor of shape (n, le, d).
            target_size (int): Target size for cropping.
            cls_token (bool, optional): Whether to preserve the class token. Defaults to False.

        Returns:
            Tuple[Tensor, Optional[Tensor]]: Cropped sequence and mask tensor (or None if no cropping).
        """
        # seq:
        if cls_token:
            seq, cls_tk = seq[:, 1:], seq[:, :1]
        n, le, _ = seq.shape
        dim = int(le**0.5)
        assert dim**2 == le
        if dim <= target_size:
            mask = None
        else:
            x0 = torch.randint(0, dim - target_size, (n,))  # n
            x1 = x0 + target_size
            y0 = torch.randint(0, dim - target_size, (n,))  # n
            y1 = y0 + target_size
            base = torch.zeros(n, dim, dim, 2)
            arr = torch.arange(dim)  # dim
            base[..., 1] += arr.view(1, dim, 1)  # y = h
            base[..., 0] += arr.view(1, 1, dim)  # x = w
            # base now is a grid
            xx = base[..., 0]
            yy = base[..., 1]
            mask = ((xx >= x0.view(n, 1, 1)) & (xx < x1.view(n, 1, 1))) & (
                (yy >= y0.view(n, 1, 1)) & (yy < y1.view(n, 1, 1))
            )  # n x dim x dim
            mask = mask.view(n, dim**2).long()  # n X le
            mask = torch.argsort(mask, dim=-1, descending=True)  # n X le
            mask = mask[:, : target_size**2]  # n X L_tgt
            mask, _ = torch.sort(mask, dim=-1)
            seq = cls.subsample(seq, mask)
        if cls_token:
            seq = torch.cat([cls_tk, seq], dim=1)
        return seq, mask

    @staticmethod
    def subsample(seq: Tensor, mask: Tensor) -> Tensor:
        """Subsamples the input sequence using the provided mask indices.

        Args:
            seq (Tensor): Input tensor of shape (n, le, d).
            mask (Tensor): Mask tensor of indices to select, shape (n, l_mask).

        Returns:
            Tensor: Subsampled tensor of shape (n, l_mask, d).
        """
        if mask is None:
            return seq
        n, le = seq.shape[:2]
        _, l_mask = mask.shape
        x_arr = torch.arange(n).view(n, 1).repeat(1, l_mask)
        return seq[x_arr, mask]

    def set_fix_decoding_size(self, fixed_output_size: int | list) -> None:
        """Set the fixed output size for decoding, ensuring it is a multiple of patch size.

        Args:
            fixed_output_size (int or list): The desired fixed output size.
        """
        if isinstance(fixed_output_size, list):
            fixed_output_size = fixed_output_size[0]
        if fixed_output_size % self.patch_size != 0:
            raise ValueError("fixed_output_size must be a multiple of patch_size")
        self.fixed_output_size = fixed_output_size // self.patch_size

    def build_input_sequence(
        self, x: Tensor, base_res: int, base_dim: int, pos_emb_base: Tensor
    ) -> tuple[Tensor, list[int], Tensor, list[Tensor | None]]:
        """Build the input sequence for the decoder, including positional embeddings and attention masks.

        Args:
            x (Tensor): Input tensor of shape (batch, length, dim).
            base_res (int): Base resolution.
            base_dim (int): Base dimension.
            pos_emb_base (Tensor): Base positional embedding.

        Returns:
            Tuple containing:
                - Concatenated input tensor,
                - List of positional embedding lengths,
                - Attention mask tensor,
                - List of mask indices.
        """
        p = self.patch_embed.patch_size[0]
        _, l_x, _ = x.shape
        _, length_pos_embed, _ = pos_emb_base.shape
        mask_tokens = self.mask_token_decoder.repeat(x.shape[0], length_pos_embed, 1)
        mask_tokens[:, :1] = x[:, :1]  # copy class token
        mask_tokens += pos_emb_base
        if self.fixed_output_size > 0:
            mask_tokens, mask = self.random_crop(
                mask_tokens, self.fixed_output_size, cls_token=True
            )
            _, length_pos_embed, _ = mask_tokens.shape
        else:
            mask = None
        new_x = [x, mask_tokens]  # first decoding has cls token

        atten_mask = [
            torch.ones((l_x + length_pos_embed, l_x + length_pos_embed), device=x.device)
        ]
        length_pos_embeds = [length_pos_embed]
        ids = [mask]
        target_sizes = [x for x in self.target_size if x != max(self.target_size)]
        for d in target_sizes:
            d = d // p
            pos_emb = get_2d_sincos_pos_embed_with_resolution(
                x.shape[-1], d, base_res * d / base_dim, cls_token=True, device=x.device
            )
            _, length_pos_embed, _ = pos_emb.shape
            mask_tokens = self.mask_token_decoder.repeat(x.shape[0], length_pos_embed, 1)
            mask_tokens += pos_emb
            mask_tokens = mask_tokens[:, 1:]
            length_pos_embed = length_pos_embed - 1

            if self.fixed_output_size > 0:
                mask_tokens, mask = self.random_crop(mask_tokens, self.fixed_output_size)
                _, length_pos_embed, _ = mask_tokens.shape
            else:
                mask = None
            new_x.append(mask_tokens)
            length_pos_embeds.append(length_pos_embed)
            ids.append(mask)
            atten_mask.append(torch.ones((length_pos_embed, length_pos_embed), device=x.device))

        x = torch.cat(new_x, dim=1)
        atten_mask = torch.block_diag(*atten_mask)  # le X le
        atten_mask[:l_x] = 1
        atten_mask[:, :l_x] = 1
        atten_mask = 1 - atten_mask  # 0 no mask, 1 mask
        atten_mask[atten_mask == 1] = float("-inf")
        return x, length_pos_embeds, atten_mask, ids

    def forward_loss(
        self,
        imgs: Tensor,
        pred: tuple[Tensor, Tensor],
        mask: Tensor,
        target_dim: int,
        ids: list[int],
    ) -> tuple[Tensor, int, int]:
        """Compute the combined loss for the model's forward pass, including both L2 and L1 (or L2) losses at different scales.

        Args:
            imgs (Tensor): Input images of shape [n, 3, H, W].
            pred (Tuple[Tensor, Tensor]): Tuple containing predictions at high and low resolutions.
                - pred_high: Predicted high-frequency components.
                - pred_low: Predicted low-frequency components.
            mask (Tensor): Mask tensor of shape [n, le], where 0 indicates keep and 1 indicates remove.
            target_dim (Any): Unused parameter, kept for compatibility.
            ids (Any): Unused parameter, kept for compatibility.

        Returns:
            Tuple[Tensor, int, int]:
                - Combined loss (sum of L1/L2 and L2 losses) as a scalar tensor.
                - 0 (placeholder).
                - 1 (placeholder).

        Notes:
            - The function computes losses at two scales: low-frequency (L2 loss) and high-frequency (L1 or L2 loss depending on configuration).
            - Masks are interpolated to match the prediction shapes.
            - Losses are normalized by the sum of the mask to avoid scale issues.
        """
        p = self.patch_embed.patch_size[0]
        dim1, dim2 = self.band_config  # 14,224
        pred_high, pred_low = pred
        n, _, _, _ = imgs.shape
        if dim2 != 224:
            target_low = nn.functional.interpolate(imgs, pred_low.shape[-2:], mode="area")
        else:
            target_low = nn.functional.interpolate(
                nn.functional.interpolate(imgs, (dim2, dim2), mode="area"),
                pred_low.shape[-2:],
                mode="area",
            )
        target_high = imgs - nn.functional.interpolate(
            nn.functional.interpolate(imgs, (dim1, dim1), mode="area"),
            pred_high.shape[-2:],
            mode="bilinear",
        )
        n, l_low = mask.shape
        l_low_dim = int(l_low**0.5)
        mask = mask.reshape(n, 1, l_low_dim, l_low_dim)
        mask_low = torch.nn.functional.interpolate(mask, pred_low.shape[-2:])
        mask_high = torch.nn.functional.interpolate(mask, pred_high.shape[-2:])
        loss_l2 = mask_low * (target_low - pred_low) ** 2
        loss_l2 = loss_l2.sum() / (mask_low.sum() + 1e-9)

        if self.use_l1_loss:
            loss_l1 = mask_high * torch.abs(target_high - pred_high) * self.l1_loss_weight
        else:
            loss_l1 = mask_high * (target_high - pred_high) ** 2
        loss_l1 = loss_l1.sum() / (mask_high.sum() + 1e-9)
        return loss_l1 + loss_l2, 0, 1

    def set_target_size(self, target_size: int) -> None:
        """Set target size."""
        self.target_size = target_size
        self.multiscale = len(target_size) > 1

    def forward(
        self,
        imgs: Tensor,
        targets: Tensor = None,
        mask_ratio: float = 0.75,
        *,
        knn_feats: bool = False,
        input_res: Tensor | None = None,
        target_res: Tensor | None = None,
        source_size: int | None = None,
    ) -> Tensor:
        """Forward pass for the MaskedAutoencoderViT model.

        Args:
            imgs (Tensor): Input images of shape (batch, channels, height, width).
            targets (Tensor, optional): Target images for reconstruction.
            mask_ratio (float, optional): Ratio of patches to mask. Default is 0.75.
            knn_feats (bool, optional): If True, returns only the CLS token features for kNN. Default is False.
            input_res (Tensor, optional): Input resolution tensor.
            target_res (Tensor, optional): Target resolution tensor.
            source_size (int, optional): If provided, resizes the input and target images to this size.

        Returns:
            tuple: (loss, pred, mask, mean, var, pos_embed_encoder, pos_embed_decoder, imgs)
                - loss: Computed loss value.
                - pred: Model predictions.
                - mask: Mask used for patch masking.
                - mean: Placeholder (0).
                - var: Placeholder (1).
                - pos_embed_encoder: Positional embeddings from encoder.
                - pos_embed_decoder: Positional embeddings from decoder.
                - imgs: Possibly resized input images.
        """
        # Handle resizing if source_size is provided
        if source_size is not None:
            input_size = imgs.shape[2]
            if source_size % self.patch_size != 0:
                raise ValueError("Source size must be a valid multiple of patch size")
            if source_size > input_size:
                raise ValueError("Source size must be no greater than image size")
            if source_size < input_size:
                imgs = nn.functional.interpolate(imgs, (source_size, source_size), mode="area")
                input_res = input_res * (input_size / source_size)
                target_size = targets.shape[2]
                target_size_new = int(target_size * (source_size / input_size))
                targets = nn.functional.interpolate(
                    imgs, (target_size_new, target_size_new), mode="area"
                )
                target_res = target_res * (target_size / target_size_new)

        if self.absolute_scale:
            input_res = torch.ones_like(input_res).to(input_res.device)

        if knn_feats:
            latent, mask, ids_restore, _ = self.forward_encoder(imgs, 0.0, input_res)
            return latent[:, 0, :]  # take cls token

        if self.absolute_scale and target_res is not None:
            target_res = torch.ones_like(target_res).to(target_res.device)

        latent, mask, ids_restore, pos_embed_encoder = self.forward_encoder(
            imgs, mask_ratio, input_res
        )

        p = self.patch_embed.patch_size[0]
        target_dim = targets.shape[2] // p
        pred, pos_embed_decoder, ids = self.forward_decoder(
            latent,
            ids_restore=ids_restore,
            target_res=target_res,
            target_dim=target_dim,
            pos_embed_encoder=pos_embed_encoder,
            mask=mask,
        )  # [N_layers_decoder, le,n, p*p*3]
        loss, mean, var = self.forward_loss(targets, pred, mask, target_dim, ids)
        pred = self.split_pred(target_dim, pred, mean, var)
        return (loss, pred, mask, mean, var, pos_embed_encoder, pos_embed_decoder, imgs)


def mae_vit_large_patch16_dec512d8b(**kwargs):
    return MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        # decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


"""# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
"""
