# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Vision-related imports
from torchvision.models import resnet18, resnet34, ResNet18_Weights

class ResNet_UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, pretrained=ResNet18_Weights.DEFAULT, freeze_backbone=True):
        super(ResNet_UNET, self).__init__()
        
        # Modify first layer of ResNet34 to accept custom number of channels
        base_model = resnet18(weights=pretrained)  # Change this line
        base_model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.base_layers = list(base_model.children())
        self.freeze_backbone(freeze_backbone)

        # Define the Unet Head/Neck
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        
        self.upconv4 = self.expand_block(512, 256)
        self.upconv3 = self.expand_block(256*2, 128)
        self.upconv2 = self.expand_block(128*2, 64)
        self.upconv1 = self.expand_block(64*2, 64)
        self.upconv0 = self.expand_block(64*2, out_channels)
        

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
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )
        return expand
    
    def freeze_backbone(self, freeze_backbone):
        if freeze_backbone:
            for l in self.base_layers:
                for param in l.parameters():
                    param.requires_grad = False

    @torch.no_grad()
    def predict(self, x):
        """ Inference method """
        if self.training:
            self.eval()
        outputs = self.forward(x)
        
        return torch.argmax(outputs, dim=1).cpu().numpy()