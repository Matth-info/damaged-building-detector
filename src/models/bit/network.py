import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)

from .help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d


class ResNetBackbone(nn.Module):
    def __init__(
        self,
        resnet_stages_num=5,
        backbone="resnet18",
        pretrained=True,
        if_upsample_2x=True,
    ):
        super().__init__()
        expand = 1
        if backbone == "resnet18":
            self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        elif backbone == "resnet34":
            self.resnet = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        elif backbone == "resnet50":
            self.resnet = resnet50(
                weights=ResNet50_Weights.DEFAULT if pretrained else None,
                replace_stride_with_dilation=[False, True, True],
            )
            expand = 4
        else:
            raise NotImplementedError("Unsupported ResNet backbone")

        self.resnet_stages_num = resnet_stages_num
        self.if_upsample_2x = if_upsample_2x

        # Define the output channels based on the selected stage
        stage_to_channels = {3: 128, 4: 256, 5: 512}
        self.out_channels = stage_to_channels.get(resnet_stages_num, None)
        if self.out_channels is None:
            raise ValueError("Invalid resnet_stages_num, should be 3, 4, or 5")
        self.out_channels *= expand

        self.conv_pred = nn.Conv2d(self.out_channels, 32, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)  # 1/4 scale
        x = self.resnet.layer2(x)  # 1/8 scale

        if self.resnet_stages_num > 3:
            x = self.resnet.layer3(x)  # 1/16 scale

        if self.resnet_stages_num == 5:
            x = self.resnet.layer4(x)  # 1/32 scale

        if self.if_upsample_2x:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        return self.conv_pred(x)


class BiT(nn.Module):
    def __init__(
        self,
        input_nc=3,
        output_nc=2,
        with_pos="learned",
        resnet_stages_num=5,
        token_len=4,
        token_trans=True,
        enc_depth=1,
        dec_depth=1,
        dim_head=64,
        decoder_dim_head=64,
        tokenizer=True,
        if_upsample_2x=True,
        pool_mode="max",
        pool_size=2,
        backbone="resnet18",
        decoder_softmax=True,
        with_decoder_pos=None,
        with_decoder=True,
        **kwargs
    ):
        super().__init__()

        self.backbone = ResNetBackbone(
            resnet_stages_num=resnet_stages_num,
            backbone=backbone,
            if_upsample_2x=if_upsample_2x,
        )

        self.token_len = token_len
        self.tokenizer = tokenizer
        self.token_trans = token_trans
        self.with_decoder = with_decoder

        dim = 32
        mlp_dim = 2 * dim

        # Tokenization layer
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1, bias=False)

        if not self.tokenizer:
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        # Positional embeddings
        self.with_pos = with_pos
        if with_pos == "learned":
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len * 2, dim))

        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == "learned":
            decoder_pos_size = 256 // 4
            self.pos_embedding_decoder = nn.Parameter(
                torch.randn(1, dim, decoder_pos_size, decoder_pos_size)
            )

        # Transformer layers
        self.transformer = Transformer(
            dim=dim,
            depth=enc_depth,
            heads=8,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=0,
        )
        self.transformer_decoder = TransformerDecoder(
            dim=dim,
            depth=dec_depth,
            heads=8,
            dim_head=decoder_dim_head,
            mlp_dim=mlp_dim,
            dropout=0,
            softmax=decoder_softmax,
        )

        # Final classification layer
        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)

    def _forward_tokens(self, x):
        if self.tokenizer:
            b, c, h, w = x.shape
            spatial_attention = torch.softmax(
                self.conv_a(x).flatten(2), dim=-1
            )  # B, token_len, H*W
            return torch.einsum("b t n, b c n -> b t c", spatial_attention, x.flatten(2))
        else:
            if self.pool_mode == "max":
                x = F.adaptive_max_pool2d(x, (self.pooling_size, self.pooling_size))
            elif self.pool_mode == "ave":
                x = F.adaptive_avg_pool2d(x, (self.pooling_size, self.pooling_size))
            return rearrange(x, "b c h w -> b (h w) c")

    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        return self.transformer(x)

    def _forward_decoder(self, x, tokens):
        b, c, h, w = x.shape
        if self.with_decoder_pos in ["fix", "learned"]:
            x += self.pos_embedding_decoder
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.transformer_decoder(x, tokens)
        return rearrange(x, "b (h w) c -> b c h w", h=h)

    def forward(self, x1, x2):
        x1, x2 = self.backbone(x1), self.backbone(x2)

        token1, token2 = self._forward_tokens(x1), self._forward_tokens(x2)

        if self.token_trans:
            tokens = self._forward_transformer(torch.cat([token1, token2], dim=1))
            token1, token2 = tokens.chunk(2, dim=1)

        x1 = self._forward_decoder(x1, token1) if self.with_decoder else x1
        x2 = self._forward_decoder(x2, token2) if self.with_decoder else x2

        x = torch.abs(x1 - x2)
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        x = self.classifier(x)
        # **Final upsampling to ensure x2 scaling**
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return x

    @torch.no_grad()
    def predict(self, x1, x2):
        self.forward(x1, x2)
