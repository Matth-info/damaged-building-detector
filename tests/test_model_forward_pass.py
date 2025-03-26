# tests/test_data_loader.py
import unittest
import pytest

import torch

from src.models import ResNet_UNET, Tiny_CD, SiameseResNetUNet, Segformer, Unet

def test_ResNet_UNET_forward_pass():
    model = ResNet_UNET(in_channels=3, out_channels=2, backbone_name="resnet18", pretrained=False)
    inputs = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    outputs = model(inputs)
    assert outputs.shape == (4, 2, 224, 224), "Output shape is incorrect!"

def test_SiameseResNetUNet_forward_pass():
    model = SiameseResNetUNet(in_channels=3, out_channels=2, backbone_name="resnet18", pretrained=False, mode="conc")
    pre_inputs = torch.randn(4, 3, 224, 224)
    post_inputs = torch.randn(4, 3, 224, 224)
    outputs = model(x1=pre_inputs, x2=post_inputs)
    assert outputs.shape == (4, 2, 224, 224), "Output shape is incorrect!"

def test_TinyCD_forward_pass():
    model = Tiny_CD(bkbn_name="efficientnet_b4", pretrained=False, output_layer_bkbn="3", out_channels=2,freeze_backbone=False)
    pre_inputs = torch.randn(4, 3, 224, 224)
    post_inputs = torch.randn(4, 3, 224, 224)
    outputs = model(ref=pre_inputs, test=post_inputs)
    assert outputs.shape == (4, 2, 224, 224), "Output shape is incorrect!"

def test_Unet_forward_pass():
    model = Unet(n_channels=3, n_classes=2, bilinear=False)
    inputs = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    outputs = model(inputs)
    assert outputs.shape == (4, 2, 224, 224), "Output shape is incorrect!"


