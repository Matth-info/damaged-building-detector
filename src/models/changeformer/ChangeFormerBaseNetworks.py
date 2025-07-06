from math import sqrt

import torch
from torch import nn
from torch.nn import init
from timm.layers import DropPath, to_2tuple, trunc_normal_
import math

from .help_funcs import resize


class ConvBlock(torch.nn.Module):
    """Convolutional block."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        activation: str = "prelu",
        norm: str | None = None,
    ) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            input_size,
            output_size,
            kernel_size,
            stride,
            padding,
            bias=bias,
        )

        self.norm = norm
        if self.norm == "batch":
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == "instance":
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == "relu":
            self.act = torch.nn.ReLU(True)
        elif self.activation == "prelu":
            self.act = torch.nn.PReLU()
        elif self.activation == "lrelu":
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == "tanh":
            self.act = torch.nn.Tanh()
        elif self.activation == "sigmoid":
            self.act = torch.nn.Sigmoid()

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation != "no":
            return self.act(out)
        return out


class DeconvBlock(torch.nn.Module):
    """Deconvolution block."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        bias: bool = True,
        activation: str = "prelu",
        norm: str | None = None,
    ) -> None:
        super().__init__()
        self.deconv = torch.nn.ConvTranspose2d(
            input_size,
            output_size,
            kernel_size,
            stride,
            padding,
            bias=bias,
        )

        self.norm = norm
        if self.norm == "batch":
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == "instance":
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == "relu":
            self.act = torch.nn.ReLU(True)
        elif self.activation == "prelu":
            self.act = torch.nn.PReLU()
        elif self.activation == "lrelu":
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == "tanh":
            self.act = torch.nn.Tanh()
        elif self.activation == "sigmoid":
            self.act = torch.nn.Sigmoid()

    def forward(self, x) -> torch.Tensor:
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        return out


class ConvLayer(nn.Module):
    """Convolutional Layer."""

    def __init__(  # noqa: D107
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int
    ) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        out = self.conv2d(x)
        return out


class UpsampleConvLayer(torch.nn.Module):
    """Upsample Convolutional layer (with ConvTranspose)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:  # noqa: D107
        super().__init__()
        self.conv2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=1,
        )

    def forward(self, x) -> torch.Tensor:  # noqa: D102
        out = self.conv2d(x)
        return out


class ResidualBlock(torch.nn.Module):
    """Residual Double Convolution Block."""

    def __init__(self, channels: int) -> None:  # noqa: D107
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out


class OverlapPatchEmbed(nn.Module):
    """Overlap Patch Embedding with a Convolutional Layer."""

    def __init__(  # noqa: D107
        self,
        img_size: int = 224,
        patch_size: int = 7,
        stride: int = 4,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m) -> None:
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

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        # pdb.set_trace()
        x = self.proj(x)  # shape [B, embed_dim, H, W]
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # shape: [B, H*W, embed_dim]
        x = self.norm(x)  # shape: [B, H*W, embed_dim]

        return x, H, W


class Attention(nn.Module):
    """Attention layer."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        """Initialize an Attention layer."""
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m) -> None:
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

    def forward(self, x, H, W) -> torch.Tensor:
        """Forward pass."""
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = (
                self.kv(x_)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.kv(x)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


#### Transformer Decoder ####
class MLP(nn.Module):
    """Linear Embedding."""

    def __init__(self, input_dim=2048, embed_dim=768) -> None:  # noqa: D107
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x) -> torch.Tensor:  # noqa: D102
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class DWConv(nn.Module):
    """Convolution layer with reshape before and after Convolution layer."""

    def __init__(self, dim=768):  # noqa: D107
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W) -> torch.Tensor:  # noqa: D102
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    """Linear Embedding for Transformer Block."""

    def __init__(  # noqa: D107
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m) -> None:
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

    def forward(self, x, H, W) -> torch.Tensor:  # noqa: D102
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer block."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ) -> None:
        """Initialize a Transformer block."""
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m) -> None:
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

    def forward(self, x, H, W) -> torch.Tensor:
        """Forward pass."""
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x
