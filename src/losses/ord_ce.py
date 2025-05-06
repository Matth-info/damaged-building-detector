import torch
import torch.nn.functional as F


def weighted_categorical_crossentropy(y_true, y_pred):
    # y_true shape (B, H, W)
    # y_pred shape (B, C, H, W)
    # Convert one-hot encoded true labels to class indices
    # true_indices = torch.argmax(y_true, dim=1)
    true_indices = y_true
    pred_indices = torch.argmax(y_pred, dim=1)

    # Compute the weights
    num_classes = y_pred.size(1)
    weights = torch.abs(true_indices - pred_indices).float() / (num_classes - 1)

    # Calculate categorical cross-entropy loss
    cross_entropy_loss = F.cross_entropy(y_pred, true_indices, reduction="none")

    # Apply weights
    weighted_loss = (1.0 + weights) * cross_entropy_loss
    return weighted_loss.mean()
