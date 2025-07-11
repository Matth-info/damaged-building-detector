# Utils function for Siamese ResNet Unet
import torch
from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)


def choose_resnet(model_name="resnet18", pretrained=True):
    if model_name == "resnet18":
        filters = [64, 64, 128, 256, 512]
        return filters, resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    if model_name == "resnet34":
        filters = [64, 64, 128, 256, 512]
        return filters, resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
    if model_name == "resnet50":
        filters = [64, 256, 512, 1024, 2048]
        return filters, resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    raise ModuleNotFoundError


class DecoderBlock(nn.Module):
    """Decoder Block
    Source : https://github.com/TripleCoenzyme/ResNet50-Unet/blob/master/model.py
    """

    def __init__(self, in_channels, mid_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.upsample = nn.ConvTranspose2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.upsample(x)
        x = self.norm2(x)
        x = self.relu2(x)
        return x
