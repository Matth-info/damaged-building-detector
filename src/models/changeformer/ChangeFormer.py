import math

import warnings
from functools import partial

import torch
import torch.nn.functional
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn

from .help_funcs import conv_diff, resize, make_prediction

from .ChangeFormerBaseNetworks import (
    ConvLayer,
    ResidualBlock,
    UpsampleConvLayer,
    OverlapPatchEmbed,
    MLP,
    Block,
)


class EncoderTransformer(nn.Module):
    """Encoder Transformer (V3).

    Args:
        img_size (int): Input image size. Default is 256.
        patch_size (int): Patch size for patch embedding. Default is 3.
        in_chans (int): Number of input channels. Default is 3.
        num_classes (int): Number of output classes. Default is 2.
        embed_dims (list[int] or tuple[int]): Embedding dimensions for each stage. Default is [32, 64, 128, 256].
        num_heads (list[int] or tuple[int]): Number of attention heads for each stage. Default is [2, 2, 4, 8].
        mlp_ratios (list[int] or tuple[int]): MLP ratios for each stage. Default is [4, 4, 4, 4].
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default is True.
        qk_scale (float or None): Override default qk scale of head_dim ** -0.5 if set. Default is None.
        drop_rate (float): Dropout rate. Default is 0.0.
        attn_drop_rate (float): Attention dropout rate. Default is 0.0.
        drop_path_rate (float): Stochastic depth rate. Default is 0.0.
        norm_layer (nn.Module): Normalization layer. Default is nn.LayerNorm.
        depths (list[int] or tuple[int]): Number of blocks for each stage. Default is [3, 3, 6, 18].
        sr_ratios (list[int] or tuple[int]): Spatial reduction ratios for each stage. Default is [8, 4, 2, 1].
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 3,
        in_chans: int = 3,
        num_classes: int = 2,
        embed_dims: tuple[int, ...] = (32, 64, 128, 256),
        num_heads: tuple[int, ...] = (2, 2, 4, 8),
        mlp_ratios: tuple[int, ...] = (4, 4, 4, 4),
        qkv_bias: bool = True,
        qk_scale: float = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type = nn.LayerNorm,
        depths: tuple[int, ...] = (3, 3, 6, 18),
        sr_ratios: tuple[int, ...] = (8, 4, 2, 1),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = list(depths)
        self.embed_dims = list(embed_dims)

        # patch embedding definitions
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=patch_size,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=patch_size,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16,
            patch_size=patch_size,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3],
        )

        # Stage-1 (x1/4 scale)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios[0],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
                for i in range(depths[0])
            ],
        )
        self.norm1 = norm_layer(embed_dims[0])

        # Stage-2 (x1/8 scale)
        cur += depths[0]
        self.block2 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[1],
                )
                for i in range(depths[1])
            ],
        )
        self.norm2 = norm_layer(embed_dims[1])

        # Stage-3 (x1/16 scale)
        cur += depths[1]
        self.block3 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratios[2],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[2],
                )
                for i in range(depths[2])
            ],
        )
        self.norm3 = norm_layer(embed_dims[2])

        # Stage-4 (x1/32 scale)
        cur += depths[2]
        self.block4 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratios[3],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[3],
                )
                for i in range(depths[3])
            ],
        )
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def _forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x1, H1, W1 = self.patch_embed1(x)
        for _i, blk in enumerate(self.block1):
            x1 = blk(x1, H1, W1)
        x1 = self.norm1(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # stage 2
        x1, H1, W1 = self.patch_embed2(x1)
        for _i, blk in enumerate(self.block2):
            x1 = blk(x1, H1, W1)
        x1 = self.norm2(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # stage 3
        x1, H1, W1 = self.patch_embed3(x1)
        for _i, blk in enumerate(self.block3):
            x1 = blk(x1, H1, W1)
        x1 = self.norm3(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # stage 4
        x1, H1, W1 = self.patch_embed4(x1)
        for _i, blk in enumerate(self.block4):
            x1 = blk(x1, H1, W1)
        x1 = self.norm4(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)
        return outs

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        x = self._forward_features(x)
        return x


class DecoderTransformer(nn.Module):
    """Transformer Decoder (V3)."""

    def __init__(
        self,
        input_transform: str = "multiple_select",
        in_index: tuple[int, ...] = (0, 1, 2, 3),
        align_corners: bool = True,
        in_channels: tuple[int, ...] = (32, 64, 128, 256),
        embedding_dim: int = 64,
        output_nc: int = 2,
        decoder_softmax: bool = False,
        feature_strides: tuple[int, ...] = (2, 4, 8, 16),
    ) -> None:
        """Initialize the DecoderTransformer with decoder settings and layers."""
        super().__init__()
        # assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]

        # settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index = list(in_index)
        self.align_corners = align_corners
        self.in_channels = list(in_channels)
        self.embedding_dim = embedding_dim
        self.output_nc = output_nc
        (
            c1_in_channels,
            c2_in_channels,
            c3_in_channels,
            c4_in_channels,
        ) = self.in_channels

        # MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        # convolutional Difference Modules
        self.diff_c4 = conv_diff(
            in_channels=2 * self.embedding_dim,
            out_channels=self.embedding_dim,
        )
        self.diff_c3 = conv_diff(
            in_channels=2 * self.embedding_dim,
            out_channels=self.embedding_dim,
        )
        self.diff_c2 = conv_diff(
            in_channels=2 * self.embedding_dim,
            out_channels=self.embedding_dim,
        )
        self.diff_c1 = conv_diff(
            in_channels=2 * self.embedding_dim,
            out_channels=self.embedding_dim,
        )

        # taking outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(
            in_channels=self.embedding_dim,
            out_channels=self.output_nc,
        )
        self.make_pred_c3 = make_prediction(
            in_channels=self.embedding_dim,
            out_channels=self.output_nc,
        )
        self.make_pred_c2 = make_prediction(
            in_channels=self.embedding_dim,
            out_channels=self.output_nc,
        )
        self.make_pred_c1 = make_prediction(
            in_channels=self.embedding_dim,
            out_channels=self.output_nc,
        )

        # Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(
                in_channels=self.embedding_dim * len(in_channels),
                out_channels=self.embedding_dim,
                kernel_size=1,
            ),
            nn.BatchNorm2d(self.embedding_dim),
        )

        # Final predction head
        self.convd2x = UpsampleConvLayer(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=4,
            stride=2,
        )
        self.dense_2x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.convd1x = UpsampleConvLayer(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=4,
            stride=2,
        )
        self.dense_1x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(
            self.embedding_dim,
            self.output_nc,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Final activation
        self.output_softmax = decoder_softmax
        self.active = nn.Sigmoid()

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
        ----
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
        -------
            Tensor: The transformed inputs

        """
        if self.input_transform == "resize_concat":
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2) -> torch.Tensor:
        """Forward pass."""
        # Transforming encoder features (select layers)
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

        # img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2

        # MLP decoder on C1-C4 #
        n, _, h, w = c4_1.shape

        outputs = []
        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(c4_1).permute(0, 2, 1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0, 2, 1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4 = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))
        p_c4 = self.make_pred_c4(_c4)
        outputs.append(p_c4)
        _c4_up = resize(_c4, size=c1_2.size()[2:], mode="bilinear", align_corners=False)

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0, 2, 1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0, 2, 1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + F.interpolate(
            _c4,
            scale_factor=2,
            mode="bilinear",
        )
        p_c3 = self.make_pred_c3(_c3)
        outputs.append(p_c3)
        _c3_up = resize(_c3, size=c1_2.size()[2:], mode="bilinear", align_corners=False)

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0, 2, 1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0, 2, 1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2 = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1)) + F.interpolate(
            _c3,
            scale_factor=2,
            mode="bilinear",
        )
        p_c2 = self.make_pred_c2(_c2)
        outputs.append(p_c2)
        _c2_up = resize(_c2, size=c1_2.size()[2:], mode="bilinear", align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0, 2, 1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0, 2, 1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1 = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1)) + F.interpolate(
            _c2,
            scale_factor=2,
            mode="bilinear",
        )
        p_c1 = self.make_pred_c1(_c1)
        outputs.append(p_c1)

        # Linear Fusion of difference image from all scales
        _c = self.linear_fuse(torch.cat((_c4_up, _c3_up, _c2_up, _c1), dim=1))

        # #Dropout
        # if dropout_ratio > 0:
        #     self.dropout = nn.Dropout2d(dropout_ratio)
        # else:
        #     self.dropout = None

        # Upsampling x2 (x1/2 scale)
        x = self.convd2x(_c)
        # Residual block
        x = self.dense_2x(x)
        # Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        # Residual block
        x = self.dense_1x(x)

        # Final prediction
        cp = self.change_probability(x)

        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs


# ChangeFormer:
class ChangeFormer(nn.Module):
    """ChangeFormer (V6)."""

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 2,
        decoder_softmax: bool = False,
        embed_dim: int = 256,
        **kwargs,
    ) -> None:
        """Initialize ChangeFormer model with encoder and decoder transformers."""
        super().__init__()
        # Transformer Encoder
        self.embed_dims = [64, 128, 320, 512]
        self.depths = [3, 3, 4, 3]  # [3, 3, 6, 18, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1

        self.Tenc_x2 = EncoderTransformer(
            img_size=256,
            patch_size=7,
            in_chans=input_nc,
            num_classes=output_nc,
            embed_dims=self.embed_dims,
            num_heads=[1, 2, 4, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            qk_scale=None,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop,
            drop_path_rate=self.drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=self.depths,
            sr_ratios=[8, 4, 2, 1],
        )

        # Transformer Decoder
        self.TDec_x2 = DecoderTransformer(
            input_transform="multiple_select",
            in_index=[0, 1, 2, 3],
            align_corners=False,
            in_channels=self.embed_dims,
            embedding_dim=self.embedding_dim,
            output_nc=output_nc,
            decoder_softmax=decoder_softmax,
            feature_strides=[2, 4, 8, 16],
        )

    def forward(self, x1, x2) -> torch.Tensor:
        """Forward pass."""
        [fx1, fx2] = [self.Tenc_x2(x1), self.Tenc_x2(x2)]

        cp = self.TDec_x2(fx1, fx2)[-1]

        # # Save to mat
        # save_to_mat(x1, x2, fx1, fx2, cp, "ChangeFormerV4")

        # exit()
        return cp

    @torch.no_grad()
    def predict(self, x1, x2):
        """Prediction pass."""
        self.forward(x1, x2)
