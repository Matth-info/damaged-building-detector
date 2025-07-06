from __future__ import annotations
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


#### Layers
class Norm2D(nn.Module):
    """A normalization layer for 2D inputs.

    This class implements a 2D normalization layer using Layer Normalization.
    It is designed to normalize 2D inputs (e.g., images or feature maps in a
    convolutional neural network).

    Attributes:
        ln (nn.LayerNorm): The layer normalization component.

    Args:
        embed_dim (int): The number of features of the input tensor (i.e., the number of
            channels in the case of images).

    Methods:
        forward: Applies normalization to the input tensor.
    """

    def __init__(self, embed_dim: int) -> None:
        """Initialize the Norm2D module.

        Args:
            embed_dim (int): The number of features of the input tensor.
        """
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the normalization process to the input tensor.

        Args:
            x (torch.Tensor): A 4D input tensor with shape
                (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The normalized tensor, having the same shape as the input.
        """
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2).contiguous()


class ResidualUpsamplingBlock(nn.Module):
    """Residual Upsampling Block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize Residual Residual Upsampling Block."""
        super().__init__()

        # Main upsampling path
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        # Shortcut (skip connection) - Uses a 1x1 convolution to match dimensions
        self.skip_connection = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=1, stride=2, output_padding=1
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = self.skip_connection(x)  # Skip connection
        out = self.upsample(x)  # Main path
        out += identity  # Add residual connection
        return self.relu(out)  # Apply ReLU


class UpscalingBlock(nn.Module):
    """Upscaling block.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.

    Returns:
        An upscaling block configured to upscale spatially.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Upscaling block.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.

        Returns:
            An upscaling block configured to upscale spatially.
        """
        super().__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.layer(x)


class Seg_Conv_Decoder(nn.Module):
    """Semantic Segmentation Head with a simple or residual convolutional head build on top the last encoder layer outputs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_size: tuple[int, int] = (14, 14),
        target_size: tuple[int, int] = (224, 224),
        embed_dims: list[int] | None = None,
        head_type: Literal["conc", "diff"] = "diff",
        depth: int = 4,
        up_block: Literal["res_block", None] = None,
    ) -> None:
        """Semantic Segmentation Head with a simple or residual convolutional head build on top the last encoder layer outputs.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            input_size (tuple[int,int], optional): size of the input image. Defaults to (14,14).
            target_size (tuple[int,int], optional): size of the target image. Defaults to (224,224).
            head_type (Literal["conc", "diff"], optional): type of head to use. Defaults to "diff".
            depth (int, optional): depth of the decoder. Defaults to 4.
            up_block (Literal["res_block", None], optional): type of upsampling block to use. Defaults to None.
        """
        super().__init__()
        self.embed_dims = (
            embed_dims
            if embed_dims
            else [int(in_channels)] + [int(in_channels / 2**i) for i in range(1, depth)]
        )
        self.input_size = input_size
        self.target_size = target_size
        self.head_type = head_type
        self.num_classes = out_channels

        self.up_block = ResidualUpsamplingBlock if up_block == "res_block" else UpscalingBlock
        self.depth = depth
        self.decoder = nn.Sequential(
            *[
                self.up_block(in_channels=self.embed_dims[i], out_channels=self.embed_dims[i + 1])
                for i in range(len(self.embed_dims) - 1)
            ]
        )
        self.final_layer = nn.Conv2d(
            kernel_size=1, in_channels=self.embed_dims[-1], out_channels=self.num_classes
        )

        if self.head_type == "conc":
            self.embed_dims[0] = self.embed_dims[0] * 2

    def forward_siamese(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass for siamese input."""
        feat_1 = self.decoder(x1)
        feat_2 = self.decoder(x2)

        if self.head_type == "diff":
            features = torch.abs(feat_1 - feat_2)
        elif self.head_type == "conc":
            features = torch.cat([feat_1, feat_2], dim=1)

        features = F.interpolate(
            features, size=self.target_size, mode="bilinear", align_corners=False
        )  # ensure outputs have the same size than input images
        return self.final_layer(features)  # ensure outputs have num_classes channels

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for single input."""
        feat = self.decoder(x)
        feat = F.interpolate(
            feat, size=self.target_size, mode="bilinear", align_corners=False
        )  # ensure outputs have the same size than input images
        return self.final_layer(feat)  # ensure outputs have num_classes channels

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for the decoder.

        If two inputs are provided, uses the siamese forward.
        If one input is provided, uses the single forward.
        """
        if len(inputs) == 2:
            return self.forward_siamese(inputs[0], inputs[1])
        elif len(inputs) == 1:
            return self.forward_single(inputs[0])
        else:
            raise ValueError("Seg_Conv_Decoder.forward expects 1 or 2 input tensors.")
