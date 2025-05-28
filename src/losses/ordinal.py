import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


# This file sum up some Ordinal Losses to better exploit the ranking of certains order-related labels like damage assessment labels
class Weighted_Categorical_CrossEntropy(_Loss):
    def __init__(self, reduction="none"):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        # y_true shape (B, H, W)
        # y_pred shape (B, C, H, W)
        # Convert one-hot encoded true labels to class indices
        # true_indices = torch.argmax(y_true, dim=1)
        true_indices = y_true
        pred_indices = torch.argmax(y_pred, dim=1)

        # Compute the weights
        num_classes = y_pred.size(1)
        ordinal_weights = torch.abs(true_indices - pred_indices).float() / (num_classes - 1)

        # Calculate categorical cross-entropy loss
        cross_entropy_loss = F.cross_entropy(y_pred, true_indices, reduction=self.reduction)

        # Apply weights
        weighted_loss = (1.0 + ordinal_weights) * cross_entropy_loss
        return weighted_loss.mean()


class Ordinal_CrossEntropy(_Loss):
    def __init__(self, reduction="sum", num_classes: int = 2):
        super().__init__()
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: raw logits of shape (B, C, H, W)
            y_true: class indices of shape (B, H, W)
        Returns:
            Scalar ordinal BCE loss
        """
        # Convert y_true to ordinal labels
        ordinal_y_true = convert_to_ordinal_labels(y_true, self.num_classes)  # shape: (B, C, H, W)

        # Sigmoid output for binary ordinal classification
        ordinal_y_pred = torch.sigmoid(y_pred)  # shape: (B, C, H, W)

        # Compute binary cross entropy per ordinal bin
        mse = F.mse_loss(ordinal_y_pred, ordinal_y_true, reduction="none")  # (B, C, H, W)

        if self.reduction == "mean":
            return mse.mean(axis=1)
        elif self.reduction == "sum":
            return mse.sum(axis=1)
        else:  # 'none'
            return mse


def preds_with_ordinal_outputs(y_pred: torch.Tensor) -> torch.Tensor:
    """
    Converts ordinal predictions (logits) to class labels.

    Args:
        y_pred: Tensor of shape (B, C, H, W) or (B, C)

    Returns:
        Tensor of class predictions: (B, H, W) or (B,)
    """
    probs = torch.sigmoid(y_pred)
    binary_preds = (probs > 0.5).float()
    ordinal_preds = binary_preds.cumprod(dim=1).sum(dim=1)
    return ordinal_preds.long()


class O2_Loss(nn.Module):
    """
    O2 Loss for Semantic Segmentation:
    Enforces unimodal softmax distributions for each pixel in ordinal segmentation tasks.
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Raw logits of shape (B, C, H, W)
            y_true: Ground truth class indices of shape (B, H, W)
        Returns:
            Scalar loss promoting unimodal class distributions per pixel
        """
        B, C, H, W = y_pred.shape
        y_probs = F.softmax(y_pred, dim=1)  # shape: (B, C, H, W)

        total_loss = 0.0
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    probs = y_probs[b, :, h, w]  # shape: (C,)
                    true_class = y_true[b, h, w]
                    total_loss += self._unimodal_penalty(probs, true_class)

        # Normalize by number of pixels
        return total_loss / (B * H * W)

    def _unimodal_penalty(self, probs: torch.Tensor, true_class: int) -> torch.Tensor:
        """
        Compute unimodal penalty for one pixel.
        """
        penalty = 0.0

        # Before true class: penalize dips (should increase)
        for k in range(1, true_class + 1):
            penalty += F.relu(probs[k - 1] - probs[k])

        # After true class: penalize rises (should decrease)
        for k in range(true_class + 1, len(probs)):
            penalty += F.relu(probs[k] - probs[k - 1])

        return penalty


def convert_to_ordinal_labels(y_true: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Converts class index labels into ordinal labels for semantic segmentation.

    Args:
        y_true (torch.Tensor): Class indices, shape (B, H, W)
        num_classes (int) : number of classes

    Returns:
        torch.Tensor: Ordinal encoded labels, shape (B, num_classes, H, W)
    """
    B, H, W = y_true.shape

    # Create ordinal labels by comparing class indices with range vector
    class_range = torch.arange(num_classes, device=y_true.device).view(1, num_classes, 1, 1)
    y_true_exp = y_true.unsqueeze(1)  # shape: (B, 1, H, W)

    ordinal_labels = (y_true_exp >= class_range).float()  # shape: (B, C, H, W)

    return ordinal_labels
