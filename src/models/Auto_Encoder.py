import torch
from torch import nn


class AutoEncoder(nn.Module):
    """Simple Convolutional-based AutoEncoder.

    Args:
        num_input_channel (int): Number of input channels.
        base_channel_size (int): Number of hidden channels.
        kwargs: Other key-words arguments.
    """

    def __init__(  # noqa: D107
        self,
        num_input_channel: int = 3,
        base_channel_size: int = 32,  # Reduced the base channel size to reduce parameters
        **kwargs,
    ) -> None:
        super().__init__()

        self.num_input_channel = num_input_channel
        self.act_fn = nn.ReLU(inplace=True)
        c_hid = base_channel_size  # Base channel size (e.g., 32 for reduced size)

        # Encoding block
        self.features = nn.Sequential(
            # Conv layer 1: Downsample 4x / 512 => 128
            nn.Conv2d(self.num_input_channel, c_hid, kernel_size=8, stride=4, padding=2),
            self.act_fn,
            # Conv layer 2: Downsample 4x / 128 => 32
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=4, stride=4, padding=1),
            self.act_fn,
            # Conv layer 3: Downsample 2x / 32 => 16
            nn.Conv2d(2 * c_hid, 4 * c_hid, kernel_size=4, stride=2, padding=1),
            self.act_fn,
        )

        # Decoding block
        self.upsample = nn.Sequential(
            # ConvTranspose layer 1: Upsample 2x / 16 => 32
            nn.ConvTranspose2d(4 * c_hid, 2 * c_hid, kernel_size=4, stride=2, padding=1),
            self.act_fn,
            # ConvTranspose layer 2: Upsample 4x / 32 => 128
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=8, stride=4, padding=2),
            self.act_fn,
            # Final ConvTranspose layer: Upsample 4x / 128 => 512 and reduce to input channels
            nn.ConvTranspose2d(c_hid, self.num_input_channel, kernel_size=8, stride=4, padding=2),
            nn.Sigmoid(),  # Normalize output to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.features(x)
        return self.upsample(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """Predict pass."""
        x = self.features(x)
        return self.upsample(x)

    def save(self, file_path: str):
        """Saving model."""
        torch.save(self.state_dict(), file_path)
        print(f"{self.__class__} Model saved to {file_path}")

    def load(self, file_path: str):
        """Load model checkpoint."""
        # Load the state_dict into the model
        self.load_state_dict(torch.load(file_path, weights_only=True))
        print(f"{self.__class__} Model loaded from {file_path}")
        return self
