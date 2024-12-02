import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch
import numpy as np

### Augmentation Test Time

__all__ = ["augmentation_test_time",
            'get_train_augmentation_pipeline',
            'get_val_augmentation_pipeline', 
            'get_train_autoencoder_augmentation_pipeline'
            'get_val_autoencoder_augmentation_pipeline'
            ]

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

def augmentation_test_time(model: nn.Module, images , list_augmentations, aggregation="mean", device="cuda"):
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
            augmented_inputs = [aug(image=x.transpose(1,2,0))["image"] for x in inputs.cpu().numpy()]
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

        return torch.from_numpy(aggregated).to(device)
    

#### Building Segmentation Model Augmentation Pipeline #####
def get_train_augmentation_pipeline(image_size=(512, 512), max_pixel_value=1, mean=(0.349, 0.354, 0.268), std=(0.114, 0.102, 0.094)):
    transform = A.Compose([
        # Resize images and masks
        A.Resize(image_size[0], image_size[1], p=1.0),  # Ensure both image and mask are resized
        # Normalize images
        A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value, p=1.0),
        # Random horizontal flip
        A.HorizontalFlip(p=0.5),
        # Random vertical flip
        A.VerticalFlip(p=0.5),
        # Random rotation
        A.RandomRotate90(p=0.5),
        # Switch x and y axis 
        A.Transpose(p=0.5), 
        # Random scale and aspect ratio change
        A.RandomSizedCrop(min_max_height=(image_size[0], image_size[1]), height=image_size[0], width=image_size[1], p=0.2),
        # Random brightness and contrast adjustments
        A.RandomBrightnessContrast(p=0.2),
        # Random saturation adjustment
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.2),
        # Convert to tensor (works for both image and mask)
        ToTensorV2()
    ])
    return transform

def get_val_augmentation_pipeline(image_size=(512, 512), max_pixel_value=1, mean=(0.349, 0.354, 0.268), std=(0.114, 0.102, 0.094)):
    transform = A.Compose([
        # Resize images and masks
        A.Resize(image_size[0], image_size[1], p=1.0),  # Ensure both image and mask are resized
        A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value, p=1.0),
        ToTensorV2()
    ])
    return transform

# xDB Tier3 Mean: [0.34944121 0.35439991 0.26813794], Std: [0.11447578 0.10222107 0.09438808]

###### AutoEncoder Augmentation Pipeline #######
def get_train_autoencoder_augmentation_pipeline(image_size=(512, 512)):
    return A.Compose(
        [
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),  # Random horizontal flip with 50% probability
            A.VerticalFlip(p=0.5),    # Random vertical flip with 50% probability
            A.RandomRotate90(p=0.5),  # Random 90 degree rotation with 50% probability
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=0
            ),  # Random shift, scale, and rotation with fill at borders
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),  # Adjust color properties
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),  # Adjust brightness and contrast
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),  # Apply a Gaussian blur
            A.GaussNoise(var_limit=(10, 50), p=0.3),  # Add Gaussian noise
            ToTensorV2(),  # Convert to PyTorch tensors
        ], is_check_shapes=True
        )
def get_val_autoencoder_augmentation_pipeline(image_size=(512, 512)):
    return  A.Compose(
            [
            A.Resize(image_size[0], image_size[1]),
            ToTensorV2(),
            ], is_check_shapes=True
        )