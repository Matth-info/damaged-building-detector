import numpy as np
import torch
import torchvision
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class Maskrcnn(nn.Module):
    def __init__(self, num_classes: int, hidden_layer_dim=256, pretrained: bool = True):
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
    def predict_sem_seg(self, images, score_threshold: float = 0.5, mask_threshold: float = 0.5):
        """
        Predict semantic segmentation masks while resolving overlapping regions.

        Args:
            images (Tensor): Batch of input images.
            score_threshold (float): Minimum confidence score to consider a prediction.
            mask_threshold (float): Threshold for binarizing instance masks.

        Returns:
            Tensor of semantic segmentation masks for the batch.
        """
        preds = self.model(images)
        # Model returns List[Dict] with keys: boxes, labels, scores, and masks.
        semantic_masks = []

        # Iterate over predictions for each image in the batch
        for pred in preds:
            # Apply score threshold to filter instances
            valid_mask = pred["scores"] > score_threshold
            pred_masks = pred["masks"][valid_mask]  # Filter masks
            pred_labels = pred["labels"][valid_mask]  # Filter labels
            pred_scores = pred["scores"][valid_mask]  # Filter scores

            if len(pred_masks) > 0:
                binarized_masks = (pred_masks > mask_threshold).squeeze(1)  # Shape: [N, H, W]

                # Create a score map and a label map
                score_map = torch.zeros_like(
                    binarized_masks[0], dtype=torch.float32
                )  # Shape: [H, W]
                label_map = torch.zeros_like(
                    binarized_masks[0], dtype=torch.int64
                )  # Shape: [H, W]

                # Iterate through each instance
                for mask, label, score in zip(binarized_masks, pred_labels, pred_scores):
                    # Update the label map only where the current instance has higher confidence
                    update_mask = mask & (score > score_map)
                    score_map[update_mask] = score
                    label_map[update_mask] = label

                semantic_mask = label_map.cpu().numpy()
            else:
                # Handle case where no valid predictions are present
                semantic_mask = torch.zeros_like(images[0, 0]).cpu().numpy()

            semantic_masks.append(semantic_mask)
        semantic_masks = np.array(semantic_masks)

        return torch.from_numpy(semantic_masks).to(dtype=torch.int64, device="cuda")

    @torch.no_grad()
    def predict(self, images):
        return self.model(images)

    def save(self, filepath: str):
        """
        Save the model's state dictionary and configuration.

        Parameters:
            filepath (str): Path to save the model file.
        """
        torch.save(
            {
                "state_dict": self.state_dict(),
                "num_classes": self.model.roi_heads.box_predictor.cls_score.out_features,
            },
            filepath,
        )
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """
        Load the model's state dictionary and configuration.

        Parameters:
            filepath (str): Path to the saved model file.
        """
        checkpoint = torch.load(filepath, map_location="cpu")  # Adjust map_location if needed
        self.load_state_dict(checkpoint["state_dict"])

        # Reconfigure the model's box predictor if num_classes differs
        num_classes = checkpoint["num_classes"]
        if num_classes != self.model.roi_heads.box_predictor.cls_score.out_features:
            print(f"Adjusting the box predictor for {num_classes} classes.")
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        print(f"Model loaded from {filepath}")


# Example usage:
# num_classes = 3  # Including the background class
# model = InstanceSegmentationModel(num_classes=num_classes)
# outputs = model(images, targets)  # During training
# predictions = model(images)  # During inference
