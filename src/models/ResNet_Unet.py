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
        filters = [64, 64, 128, 256, 512]
        return filters, resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    elif model_name=="resnet34":
        filters = [64, 64, 128, 256, 512]
        return filters,  resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
    elif model_name=="resnet50":
        filters = [64, 256, 512, 1024, 2048] 
        return filters , resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    else:
        raise ModuleNotFoundError


class DecoderBlock(nn.Module):
    """
    Decoder Block  
    Source : https://github.com/TripleCoenzyme/ResNet50-Unet/blob/master/model.py
    """
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
   
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.norm1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.upsample = nn.ConvTranspose2d(
            in_channels=mid_channels, out_channels=out_channels,
            kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
      
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.upsample(x)
        x = self.norm2(x)
        x = self.relu2(x)
        return x


class ResNet_UNET(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=2,
        backbone_name="resnet18",
        pretrained=True,
        freeze_backbone=True,
    ):
        super().__init__()

        # Modify first layer of ResNet34 to accept custom number of channels
        filters, resnet = choose_resnet(model_name=backbone_name, pretrained=pretrained)  # Change this line
       
        # Modify input channels if not 3
        if in_channels != 3:
            self.firstconv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )
            resnet.conv1 = self.firstconv  # Replace original ResNet conv1
        else:
            self.firstconv = resnet.conv1
        
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        # Extract the 3 first Blocks from ResNet 
        self.encoder1 = resnet.layer1 # 2 ResNet Basic Blocks : in (filter[0] -> filter[1])
        self.encoder2 = resnet.layer2 # 2 Basic Blocks : (filter[1] -> filter[2])
        self.encoder3 = resnet.layer3 # 2 Basic Blocks : (filter[2] -> filter[3])
        
        self.center = DecoderBlock(in_channels=filters[3], mid_channels=filters[3]*4, out_channels=filters[3])
        
        self.decoder1 = DecoderBlock(in_channels=filters[3]+filters[2], mid_channels=filters[2]*4, out_channels=filters[2])
        self.decoder2 = DecoderBlock(in_channels=filters[2]+filters[1], mid_channels=filters[1]*4, out_channels=filters[1])
        self.decoder3 = DecoderBlock(in_channels=filters[1]+filters[0], mid_channels=filters[0]*4, out_channels=filters[0])
        
        self.final = nn.Sequential(
                nn.Conv2d(in_channels=filters[0], out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32), 
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1),
                nn.ReLU(inplace=True)
                )
        
        if freeze_backbone:
            self.freeze_backbone(in_channels)

    def freeze_backbone(self, in_channels):
        """
        Freezes the weights of the ResNet backbone layers to prevent them from updating during training.
        """
        list_layers = [self.firstbn, self.encoder1, self.encoder2, self.encoder3]
        if in_channels == 3:
            list_layers = [self.firstconv] + list_layers

        for layer in list_layers:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x): 
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x) 

        x_ = self.firstmaxpool(x)

        e1 = self.encoder1(x_)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        center = self.center(e3)

        d2 = self.decoder1(torch.cat([center, e2], dim=1))
        d3 = self.decoder2(torch.cat([d2, e1], dim=1))
        d4 = self.decoder3(torch.cat([d3, x], dim=1))

        return self.final(d4)

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
