import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, num_input_channel: int, base_channel_size: int, latent_dim: int, act_fn: object):
        super(AutoEncoder, self).__init__()

        self.num_input_channel = num_input_channel
        self.act_fn = act_fn
        c_hid = base_channel_size  # Base channel size (e.g., 64)

        # Encoding block
        self.features = nn.Sequential(
            nn.Conv2d(self.num_input_channel, c_hid, kernel_size=8, stride=4, padding=2),  # Downsample 4x
            self.act_fn,
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=4, stride=2, padding=1),  # Downsample 2x
            self.act_fn,
            nn.Conv2d(2 * c_hid, 4 * c_hid, kernel_size=4, stride=2, padding=1),  # Downsample 2x
            self.act_fn,
            nn.Flatten(),
            nn.Linear(4 * c_hid * 4 * 4, latent_dim),  # Adjust dimensions for 4x4 spatial size
        )

        # Decoding block
        self.upsample = nn.Sequential(
            nn.Linear(latent_dim, 4 * c_hid * 4 * 4),  # Map back to convolutional dimensions
            nn.Unflatten(1, (4 * c_hid, 4, 4)),  # Match ConvTranspose input shape
            nn.ConvTranspose2d(4 * c_hid, 2 * c_hid, kernel_size=4, stride=2, padding=1),  # Upsample 2x
            act_fn,
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=4, stride=2, padding=1),  # Upsample 2x
            act_fn,
            nn.ConvTranspose2d(c_hid, self.num_input_channel, kernel_size=8, stride=4, padding=2),  # Upsample 4x
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

    def forward(self, x):
        x = self.features(x)
        x = self.upsample(x)
        return x

    @torch.no_grad()
    def predict(self, x):
        x = self.features(x)
        x = self.upsample(x)
        return x

    @torch.no_grad()
    def encoder(self, x):
        return self.features(x)