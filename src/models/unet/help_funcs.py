from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor


class DoubleConv(nn.Module):
    """Unet Double Convolution Layer : (convolution => [BN] => ReLU) * 2."""

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: int | None = None
    ) -> None:
        """Unet Double Convolution Layer .

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            mid_channels (int | None, optional): number of intermediate channels. Defaults to None.
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling Layer with maxpool then double conv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize a Downscaling Layer.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling layer then double conv."""

    def __init__(self, in_channels: int, out_channels: int, *, bilinear: bool = True) -> None:
        """_summary_.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            bilinear (bool, optional): Choose bilinear upsample. Defaults to True.
        """
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Forward pass."""
        x1 = self.up(x1)
        # input is BCHW
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x) -> torch.Tensor:  # noqa: D102
        return self.conv(x)
