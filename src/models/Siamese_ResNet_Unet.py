import torch
import torch.nn as nn
from torchvision.models import (
    resnet18,
    resnet34, 
    resnet50, 
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights
)


### Inpired by FC-Siam-diff. from FULLY CONVOLUTIONAL SIAMESE NETWORKS FORCHANGE DETECTION by  Rodrigo Caye Daudt, Bertrand Le Saux, Alexandre Boulch
def choose_resnet(model_name="resnet18", pretrained=True):
    if model_name == "resnet18":
        filters = [64, 64, 128, 256, 512]
        return filters, resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    elif model_name == "resnet34":
        filters = [64, 64, 128, 256, 512]
        return filters, resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
    elif model_name == "resnet50":
        filters = [64, 256, 512, 1024, 2048]
        return filters, resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    else:
        raise ModuleNotFoundError

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.upsample = nn.ConvTranspose2d(
            in_channels=mid_channels, out_channels=out_channels,
            kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
        )
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

class SiameseResNetUNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=2,
        backbone_name="resnet18",
        pretrained=True,
        freeze_backbone=True,
        mode="diff"
    ):
        super().__init__()
        assert mode in ["diff", "conc"], "Mode must be either 'diff' or 'conc'."
        self.mode = mode
        self.filters, resnet = choose_resnet(model_name=backbone_name, pretrained=pretrained)
        self.in_channels = in_channels
        # Shared Encoder
        if self.in_channels != 3:
            self.firstconv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )
        else:
            self.firstconv = resnet.conv1

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        if self.mode == "diff":
            # Center and Decoder
            self.center = DecoderBlock(in_channels=self.filters[3], mid_channels=self.filters[3]*4, out_channels=self.filters[3])
            self.decoder1 = DecoderBlock(in_channels=self.filters[3]+self.filters[2], mid_channels=self.filters[2]*4, out_channels=self.filters[2])
            self.decoder2 = DecoderBlock(in_channels=self.filters[2]+self.filters[1], mid_channels=self.filters[1]*4, out_channels=self.filters[1])
            self.decoder3 = DecoderBlock(in_channels=self.filters[1]+self.filters[0], mid_channels=self.filters[0]*4, out_channels=self.filters[0])

        elif self.mode == "conc":
            self.center = DecoderBlock(in_channels=self.filters[3]*2, mid_channels=self.filters[3]*4, out_channels=self.filters[3])
            self.decoder1 = DecoderBlock(in_channels=self.filters[3]+self.filters[2]*2, mid_channels=self.filters[2]*4, out_channels=self.filters[2])
            self.decoder2 = DecoderBlock(in_channels=self.filters[2]+self.filters[1]*2, mid_channels=self.filters[1]*4, out_channels=self.filters[1])
            self.decoder3 = DecoderBlock(in_channels=self.filters[1]+self.filters[0]*2, mid_channels=self.filters[0]*4, out_channels=self.filters[0])

        self.final = nn.Sequential(
            nn.Conv2d(in_channels=self.filters[0], out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        list_layers = [self.firstbn, self.encoder1, self.encoder2, self.encoder3]
        if self.in_channels == 3:
            list_layers = [self.firstconv] + list_layers

        for layer in list_layers:
            for param in layer.parameters():
                param.requires_grad = False

    def forward_branch(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)
        e1 = self.encoder1(x_)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        return x, e1, e2, e3

    def forward(self, x1, x2):
        # Process both inputs through the shared encoder
        x1, e1_1, e2_1, e3_1 = self.forward_branch(x1)
        x2, e1_2, e2_2, e3_2 = self.forward_branch(x2)

        if self.mode == "diff":
            x = torch.abs(x1 - x2)
            e1 = torch.abs(e1_1 - e1_2)
            e2 = torch.abs(e2_1 - e2_2)
            e3 = torch.abs(e3_1 - e3_2)

            # Combine features (e.g., via concatenation)
            center = self.center(e3)

            d2 = self.decoder1(torch.cat([center, e2], dim=1))
            d3 = self.decoder2(torch.cat([d2, e1], dim=1))
            d4 = self.decoder3(torch.cat([d3, x], dim=1))

        elif self.mode == "conc":
            # Concatenate corresponding feature maps from both branches
            x = torch.cat([x1, x2], dim=1)
            e1 = torch.cat([e1_1, e1_2], dim=1)
            e2 = torch.cat([e2_1, e2_2], dim=1)
            e3 = torch.cat([e3_1, e3_2], dim=1)

            # Combine features
            center = self.center(e3)

            d2 = self.decoder1(torch.cat([center, e2], dim=1))
            d3 = self.decoder2(torch.cat([d2, e1], dim=1))
            d4 = self.decoder3(torch.cat([d3, x], dim=1))
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Choose 'diff' or 'conc'.")
        return self.final(d4)

    @torch.no_grad()
    def predict(self, x1, x2):
        outputs = self.forward(x1, x2)
        return torch.argmax(outputs, dim=1).cpu().numpy()

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)
        print(f"{self.__class__.__name__} Model saved to {file_path}")

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))
        print(f"{self.__class__.__name__} Model loaded from {file_path}")
        return self
