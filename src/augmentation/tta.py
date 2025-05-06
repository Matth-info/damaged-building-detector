from typing import Callable, List

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2


def reverse_augmentation(predictions: np.array, augmentation: A.VerticalFlip | A.HorizontalFlip):
    """
    Reverse specific augmentations applied to images to make predictions compatible.

    Args:
        predictions: Model predictions (e.g., logits, probabilities, or masks).
        augmentation: The Albumentations transform applied to the image.

    Returns:
        The reversed predictions, if necessary.
    """
    if isinstance(augmentation, A.HorizontalFlip):
        # Horizontal flip: reverse the flip operation on predictions (masks).
        return np.flip(predictions, axis=-1)  # Flip along width
    elif isinstance(augmentation, A.VerticalFlip):
        # Vertical flip: reverse the flip operation on predictions (masks).
        return np.flip(predictions, axis=-2)  # Flip along height
    # Add cases for other augmentations if needed, such as rotation.
    return predictions  # Default to no reversal.


def augmentation_test_time(
    model: nn.Module,
    images: torch.Tensor,
    list_augmentations: List[Callable],
    aggregation: str = "mean",
    device: str = "cuda",
) -> torch.Tensor:
    """
    Perform test-time augmentation (TTA) on a model and aggregate predictions.

    Args:
        model: The trained model to evaluate.
        batch: A batch of inputs, including images and optionally masks.
        list_augmentations: A list of Albumentations transformations to apply.
        aggregation: Method to aggregate predictions, "mean" or "max".
        image_tag: Key in the batch dictionary that corresponds to images.
        device: Device to run the model on ('cuda' or 'cpu').

    Returns:
        Aggregated predictions for the batch.
    """
    model.eval()
    with torch.no_grad():
        inputs = images.to(device)  # Images
        batch_predictions = []

        # Apply each augmentation and get predictions
        for aug in list_augmentations:
            augmented_inputs = [
                aug(image=x.transpose(1, 2, 0))["image"] for x in inputs.cpu().numpy()
            ]
            augmented_inputs = torch.stack(
                [ToTensorV2()(image=x)["image"] for x in augmented_inputs]
            ).to(device)

            predictions = model(
                augmented_inputs
            )  # Forward pass for predictions (e.g., logits or masks)

            # Reverse augmentation
            predictions = np.array(
                [reverse_augmentation(pred.cpu().numpy(), aug) for pred in predictions]
            )
            batch_predictions.append(predictions)

        # add not augmented prediction
        preds = model(inputs).cpu().numpy()
        batch_predictions.append(preds)

        # Aggregate predictions for this batch
        batch_predictions = np.array(
            batch_predictions
        )  # Shape: [num_augmentations, batch_size, H, W]

        if aggregation == "mean":
            aggregated = np.mean(batch_predictions, axis=0)
        elif aggregation == "max":
            aggregated = np.max(batch_predictions, axis=0)
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")

        return torch.from_numpy(aggregated).to(device)


def augmentation_test_time_siamese(
    model: nn.Module,
    images_1: torch.Tensor,
    images_2: torch.Tensor,
    list_augmentations: List[Callable],
    aggregation: str = "mean",
    device: str = "cuda",
) -> torch.Tensor:
    """
    Perform test-time augmentation (TTA) for a Siamese model and aggregate predictions.

    Args:
        model (nn.Module): The trained Siamese model to evaluate.
        images_1 (torch.Tensor): Batch of first input images (e.g., pre-disaster images).
        images_2 (torch.Tensor): Batch of second input images (e.g., post-disaster images).
        list_augmentations (List[Callable]): List of Albumentations transformations to apply.
        aggregation (str): Method to aggregate predictions, "mean" or "max".
        device (str): Device to run the model on ("cuda" or "cpu").

    Returns:
        torch.Tensor: Aggregated predictions for the batch.
    """
    model.eval()
    with torch.no_grad():
        inputs_1 = images_1.to(device)  # Pre-disaster images
        inputs_2 = images_2.to(device)  # Post-disaster images
        batch_predictions = []

        # Apply each augmentation and get predictions
        for aug in list_augmentations:
            # Augment both images
            augmented_inputs_1 = torch.stack(
                [
                    ToTensorV2()(image=aug(image=x.transpose(1, 2, 0))["image"])["image"]
                    for x in inputs_1.cpu().numpy()
                ]
            ).to(device)

            augmented_inputs_2 = torch.stack(
                [
                    ToTensorV2()(image=aug(image=x.transpose(1, 2, 0))["image"])["image"]
                    for x in inputs_2.cpu().numpy()
                ]
            ).to(device)

            # Forward pass for predictions
            predictions = model(augmented_inputs_1, augmented_inputs_2)  # Siamese forward pass

            # Reverse augmentation
            reversed_predictions = np.array(
                [reverse_augmentation(pred.cpu().numpy(), aug) for pred in predictions]
            )
            batch_predictions.append(reversed_predictions)

        # Add non-augmented predictions
        preds = model(inputs_1, inputs_2).cpu().numpy()
        batch_predictions.append(preds)

        # Aggregate predictions for this batch
        batch_predictions = np.array(
            batch_predictions
        )  # Shape: [num_augmentations, batch_size, H, W]

        if aggregation == "mean":
            aggregated = np.mean(batch_predictions, axis=0)
        elif aggregation == "max":
            aggregated = np.max(batch_predictions, axis=0)
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")

        return torch.from_numpy(aggregated).to(device)
