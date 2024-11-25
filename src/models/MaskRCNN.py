import torch
import torchvision
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class Maskrcnn(nn.Module):
    def __init__(self, num_classes: int, hidden_layer_dim = 256, pretrained: bool = True):
        """
        Wrapper for Mask R-CNN instance segmentation model.

        Parameters:
            num_classes (int): Number of classes for the instance segmentation task, including the background.
            pretrained (bool): Whether to load the model with pre-trained weights on COCO.
        """
        super().__init__()
        # Load the pre-trained Mask R-CNN model
        weights = "DEFAULT" if pretrained else None
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)

        # Replace the box predictor for classification
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Replace the mask predictor for segmentation
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = hidden_layer_dim
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )

    def forward(self, images, targets=None):
        """
        Forward pass for the model.

        Parameters:
            images (list[Tensor]): List of input images as tensors.
            targets (list[dict], optional): List of target dictionaries with 'boxes', 'labels', and 'masks'.

        Returns:
            dict or list[dict]: If training, returns losses; if evaluating, returns predictions.
        """
        return self.model(images, targets)

    @torch.no_grad()
    def predict(self, images):
        return self.model(images)
    
    def save(self, filepath: str):
        """
        Save the model's state dictionary and configuration.

        Parameters:
            filepath (str): Path to save the model file.
        """
        torch.save({
            'state_dict': self.state_dict(),
            'num_classes': self.model.roi_heads.box_predictor.cls_score.out_features,
        }, filepath)
        print(f"Model saved to {filepath}")

# Example usage:
# num_classes = 3  # Including the background class
# model = InstanceSegmentationModel(num_classes=num_classes)
# outputs = model(images, targets)  # During training
# predictions = model(images)  # During inference
