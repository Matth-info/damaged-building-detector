import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import logging
from typing import Any

from .help_funcs import DoubleConv, Down, OutConv, Up


class UNet(nn.Module):
    """Unet model."""

    def __init__(
        self, n_channels: int = 3, n_classes: int = 2, *, bilinear: bool = False, **kwargs: Any
    ) -> None:
        """Original Unet implementation with 4 downscaling 4 upscaling layers.

        Args:
            n_channels (int): number of input channels (e.g RGB image requires 3 n_channels)
            n_classes (int): number of classes in the semantic segmentation task.
            bilinear (bool, optional): choose bilinear upscaling (reduce the number of weights in the model.). Defaults to False.
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

    def use_checkpointing(self) -> None:
        """Apply checkpointing to blocks of layers for memory efficiency."""
        self.inc = torch.utils.checkpoint.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint.checkpoint(self.outc)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> None:
        """Inference method."""
        outputs = self.forward(x)
        return torch.argmax(outputs, dim=1).cpu().numpy()

    def save(self, checkpoint_path: str) -> None:
        """Saves the model's state_dict to the specified path."""
        torch.save(self.state_dict(), checkpoint_path)
        logging.info("Model checkpoint saved to %s", checkpoint_path)

    def load(self, checkpoint_path: str, device: str) -> None:
        """Loads the model's state_dict from the specified path."""
        # Load the state_dict from the checkpoint
        state_dict = torch.load(checkpoint_path, map_location=device)
        # Load the state_dict into the model
        self.load_state_dict(state_dict)
        logging.info("Model checkpoint loaded from %s", checkpoint_path)
