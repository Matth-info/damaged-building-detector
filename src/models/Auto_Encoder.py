import torch
import torch.nn as nn

class Auto_encoder(nn.Module):
    def __init__(self):
        super(Auto_encoder, self).__init__()

        # Encoding block
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 256, kernel_size=2, padding=2),
            nn.ReLU(inplace=True)
        )
        
        # Decoding block
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 192, kernel_size=2, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 64, kernel_size=2, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=8, stride=4, padding=2),
            nn.Sigmoid()  # Use Sigmoid for normalized outputs
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
    
