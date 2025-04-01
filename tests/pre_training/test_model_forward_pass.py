# tests\test_model_forward_pass.py

import pytest
import torch

from src.models import (
    ResNet_UNET, TinyCD, SiameseResNetUNet, BiT, ChangeFormer
)

@pytest.mark.parametrize("model_class, kwargs, input_shape, expected_shape", [
    (ResNet_UNET, {"in_channels": 3, "out_channels": 2, "backbone_name": "resnet18", "pretrained": False}, (4, 3, 224, 224), (4, 2, 224, 224)),
    (SiameseResNetUNet, {"in_channels": 3, "out_channels": 2, "backbone_name": "resnet18", "pretrained": False, "mode": "conc"}, (4, 3, 224, 224), (4, 2, 224, 224)),
    (TinyCD, {"bkbn_name": "efficientnet_b4", "pretrained": False, "output_layer_bkbn": "3", "out_channels": 2, "freeze_backbone": False}, (4, 3, 224, 224), (4, 2, 224, 224)),
    (BiT, {"input_nc": 3, "output_nc": 2, "with_pos": "learned", "resnet_stages_num": 4, "backbone": "resnet18"}, (4, 3, 224, 224), (4, 2, 224, 224)),
    (ChangeFormer, {"input_nc": 3, "output_nc": 2, "decoder_softmax": False, "embed_dim": 256}, (4, 3, 224, 224), (4, 2, 224, 224))
])
def test_model_forward_pass(model_class, kwargs, input_shape, expected_shape):
    model = model_class(**kwargs).eval()
    
    # Handle single-input vs. siamese models
    if model_class in [SiameseResNetUNet, BiT, TinyCD, ChangeFormer]:
        inputs = torch.randn(*input_shape)
        outputs = model(x1=inputs, x2=inputs)
    else:
        inputs = torch.randn(*input_shape)
        outputs = model(inputs)
    
    assert outputs.shape == expected_shape, f"Expected {expected_shape}, but got {outputs.shape}"