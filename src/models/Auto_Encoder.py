import torch
import torch.nn as nn

"""class AutoEncoder(nn.Module):
    def __init__(
        self,
        num_input_channel=3,
        base_channel_size=64,
        latent_dim=256,
        act_fn=nn.ReLU(inplace=True),
    ):
        super(AutoEncoder, self).__init__()

        self.num_input_channel = num_input_channel
        self.act_fn = act_fn
        c_hid = base_channel_size  # Base channel size (e.g., 64)

        # Encoding block
        self.features = nn.Sequential(
            nn.Conv2d(
                self.num_input_channel, c_hid, kernel_size=4, stride=2, padding=1
            ),  # Downsample 2x / 512 => 256
            self.act_fn,
            nn.Conv2d(
                c_hid, 2 * c_hid, kernel_size=4, stride=2, padding=1
            ),  # Downsample 2x / 256 => 128
            self.act_fn,
            nn.Conv2d(
                2 * c_hid, 4 * c_hid, kernel_size=4, stride=2, padding=1
            ),  # Downsample 2x / 128 => 64
            self.act_fn,
            nn.Conv2d(
                4 * c_hid, 8 * c_hid, kernel_size=4, stride=2, padding=1
            ),  # Downsample 2x / 64 => 32
            self.act_fn,
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsampling x2 / 32 => 16
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(8 * c_hid * 16 * 16, latent_dim),  # Compress to latent_dim
        )

        # Decoding block
        self.upsample = nn.Sequential(
            nn.Linear(
                latent_dim, 8 * c_hid * 16 * 16
            ),  # Map back to convolutional dimensions
            nn.Unflatten(1, (8 * c_hid, 16, 16)),  # Match ConvTranspose input shape
            nn.ConvTranspose2d(
                8 * c_hid, 8 * c_hid, kernel_size=4, stride=2, padding=1
            ),  # Upsample 2x 16 => 32
            act_fn,
            nn.ConvTranspose2d(
                8 * c_hid, 4 * c_hid, kernel_size=4, stride=2, padding=1
            ),  # Upsample 2x / 32 => 64
            act_fn,
            nn.ConvTranspose2d(
                4 * c_hid, 2 * c_hid, kernel_size=4, stride=2, padding=1
            ),  # Upsample 2x / 64 => 128
            act_fn,
            nn.ConvTranspose2d(
                2 * c_hid, c_hid, kernel_size=4, stride=2, padding=1
            ),  # Upsample 2x / 128 => 256
            act_fn,
            nn.ConvTranspose2d(
                c_hid, self.num_input_channel, kernel_size=4, stride=2, padding=1
            ),  # Upsample 2x / 256 => 512
            nn.Sigmoid(),  # Normalize output to [0, 1]
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

"""
class AutoEncoder(nn.Module):
    def __init__(
        self,
        num_input_channel=3,
        base_channel_size=32,  # Reduced the base channel size to reduce parameters
        act_fn=nn.ReLU(inplace=True),
    ):
        super(AutoEncoder, self).__init__()

        self.num_input_channel = num_input_channel
        self.act_fn = act_fn
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

    def forward(self, x):
        x = self.features(x)
        x = self.upsample(x)
        return x

    @torch.no_grad()
    def predict(self, x):
        x = self.features(x)
        x = self.upsample(x)
        return x

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)
        print(f"{self.__class__} Model saved to {file_path}")
        
    def load(self, file_path):
        # Load the state_dict into the model
        self.load_state_dict(torch.load(file_path, weights_only=True))
        print(f"{self.__class__} Model loaded from {file_path}")
        return self
        


