# PyTorch imports
import torch
import torch.nn as nn
# Vision-related imports
from torchvision.models import (
    resnet18,
    resnet34, 
    resnet50, 
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights
)

def choose_resnet(model_name="resnet18", pretrained=True):

    if model_name=="resnet18":
        return resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    elif model_name=="resnet34":
        return resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
    elif model_name=="resnet50":
        return resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    else:
        raise ModuleNotFoundError


class ResNet_UNET(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=2,
        backbone_name="resnet18",
        pretrained=True,
        freeze_backbone=True,
    ):
        #super(ResNet_UNET, self).__init__()
        super().__init__()

        # Modify first layer of ResNet34 to accept custom number of channels
        base_model = choose_resnet(model_name=backbone_name, pretrained=pretrained)  # Change this line
       
        self.base_layers = list(base_model.children())
        self.freeze_backbone(freeze_backbone)

        # Define the Unet Head/Neck
        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

        self.upconv4 = self.expand_block(512, 256)
        self.upconv3 = self.expand_block(256 * 2, 128)
        self.upconv2 = self.expand_block(128 * 2, 64)
        self.upconv1 = self.expand_block(64 * 2, 64)
        self.upconv0 = self.expand_block(64 * 2, out_channels)

    def forward(self, x):

        # Contracting Path
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        # Expansive Path
        upconv4 = self.upconv4(layer4)
        upconv3 = self.upconv3(torch.cat([upconv4, layer3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, layer2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, layer1], 1))
        upconv0 = self.upconv0(torch.cat([upconv1, layer0], 1))

        return upconv0

    def expand_block(self, in_channels, out_channels):
        expand = nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )
        return expand

    def freeze_backbone(self, freeze_backbone):
        if freeze_backbone:
            for layer in self.base_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    @torch.no_grad()
    def predict(self, x):
        """Inference method"""
        outputs = self.forward(x)
        return torch.argmax(outputs, dim=1).cpu().numpy()

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)
        print(f"{self.__class__.__name__} Model saved to {file_path}")

    def load(self, file_path):
        # Load the state_dict into the model
        self.load_state_dict(torch.load(file_path))
        print(f"{self.__class__.__name__} Model loaded from {file_path}")
        return self
