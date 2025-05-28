from typing import List

import torch
import torchvision
from torch import Tensor
from torch.nn import Identity, Module, ModuleList

from .layers import MixingBlock, MixingMaskAttentionBlock, PixelwiseLinear, UpMask


class TinyCD(Module):
    def __init__(
        self,
        bkbn_name="efficientnet_b4",
        pretrained=True,
        output_layer_bkbn="3",
        out_channels=2,
        freeze_backbone=False,
        **kwargs
    ):
        super().__init__()

        # Load the pretrained backbone according to parameters:
        self._backbone = _get_backbone(bkbn_name, pretrained, output_layer_bkbn, freeze_backbone)

        # Initialize mixing blocks:
        self._first_mix = MixingMaskAttentionBlock(6, 3, [3, 10, 5], [10, 5, 1])
        self._mixing_mask = ModuleList(
            [
                MixingMaskAttentionBlock(48, 24, [24, 12, 6], [12, 6, 1]),
                MixingMaskAttentionBlock(64, 32, [32, 16, 8], [16, 8, 1]),
                MixingBlock(112, 56),
            ]
        )

        # Initialize Upsampling blocks:
        self._up = ModuleList(
            [
                UpMask(2, 56, 64),
                UpMask(2, 64, 64),
                UpMask(2, 64, 32),
            ]
        )

        # Final classification layer:
        self._classify = PixelwiseLinear([32, 16, 8], [16, 8, out_channels], Identity())

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        features = self._encode(x1, x2)
        latents = self._decode(features)
        return self._classify(latents)

    @torch.no_grad()
    def predict(self, x1, x2):
        self.forward(x1, x2)

    def _encode(self, x1, x2) -> List[Tensor]:
        features = [self._first_mix(x1, x2)]
        for num, layer in enumerate(self._backbone):
            x1, x2 = layer(x1), layer(x2)
            if num != 0:
                features.append(self._mixing_mask[num - 1](x1, x2))
        return features

    def _decode(self, features) -> Tensor:
        upping = features[-1]
        for i, j in enumerate(range(-2, -5, -1)):
            upping = self._up[i](upping, features[j])
        return upping


def _get_backbone(bkbn_name, pretrained, output_layer_bkbn, freeze_backbone) -> ModuleList:
    # The whole model:
    entire_model = getattr(torchvision.models, bkbn_name)(
        weights="DEFAULT" if pretrained else None
    ).features

    # Slicing it:
    derived_model = ModuleList([])
    for name, layer in entire_model.named_children():
        derived_model.append(layer)
        if name == output_layer_bkbn:
            break

    # Freezing the backbone weights:
    if freeze_backbone:
        for param in derived_model.parameters():
            param.requires_grad = False
    return derived_model
