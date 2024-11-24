import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch
import numpy as np

### Augmentation Test Time

def reverse_augmentation(predictions, augmentation):
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

def augmentation_test_time(model: nn.Module, batch, list_augmentations, aggregation="mean", image_tag="image", device="cuda"):
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
        inputs = batch[image_tag].to(device)  # Images
        batch_predictions = []

        # Apply each augmentation and get predictions
        for aug in list_augmentations:
            augmented_inputs = [aug(image=x)["image"] for x in inputs.cpu().numpy()]
            augmented_inputs = torch.stack([ToTensorV2()(image=x)["image"] for x in augmented_inputs]).to(device)
            predictions = model(augmented_inputs)  # Forward pass for predictions (e.g., logits or masks)

            # Reverse augmentation
            predictions = np.array([reverse_augmentation(pred.cpu().numpy(), aug) for pred in predictions])
            batch_predictions.append(predictions)


        # add not augmented prediction 
        preds = model(inputs).cpu().numpy()
        batch_predictions.append(preds)

        # Aggregate predictions for this batch
        batch_predictions = np.array(batch_predictions)  # Shape: [num_augmentations, batch_size, H, W]
        
        if aggregation == "mean":
            aggregated = np.mean(batch_predictions, axis=0)
        elif aggregation == "max":
            aggregated = np.max(batch_predictions, axis=0)
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")

        return aggregated