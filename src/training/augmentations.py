import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.composition import OneOf
import torch.nn as nn
import torch
import numpy as np
from typing import List, Callable

### Augmentation Test Time

__all__ = ["augmentation_test_time",
           "augmentation_test_time_siamese", 
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

def augmentation_test_time(model: nn.Module,
                            images: torch.Tensor, 
                            list_augmentations: List[Callable],
                            aggregation: str = "mean",
                            device: str = "cuda"
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
    
def augmentation_test_time_siamese(
    model: nn.Module,
    images_1: torch.Tensor,
    images_2: torch.Tensor,
    list_augmentations: List[Callable],
    aggregation: str = "mean",
    device: str = "cuda"
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
            augmented_inputs_1 = torch.stack([
                ToTensorV2()(image=aug(image=x.transpose(1, 2, 0))["image"])["image"]
                for x in inputs_1.cpu().numpy()
            ]).to(device)

            augmented_inputs_2 = torch.stack([
                ToTensorV2()(image=aug(image=x.transpose(1, 2, 0))["image"])["image"]
                for x in inputs_2.cpu().numpy()
            ]).to(device)

            # Forward pass for predictions
            predictions = model(augmented_inputs_1, augmented_inputs_2)  # Siamese forward pass

            # Reverse augmentation
            reversed_predictions = np.array([
                reverse_augmentation(pred.cpu().numpy(), aug)
                for pred in predictions
            ])
            batch_predictions.append(reversed_predictions)

        # Add non-augmented predictions
        preds = model(inputs_1, inputs_2).cpu().numpy()
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
def get_train_augmentation_pipeline(image_size=(512, 512), max_pixel_value=1, mean=None, std=None):
    transform = A.Compose([
            # Resize images and masks
            A.RandomResizedCrop(height=image_size[0], width=image_size[1], scale=(0.7, 1), p=1.0),  # Ensure both image and mask are resized
            # Normalize images
            A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value, p=1.0) if mean and std else A.NoOp(),
            # Scale (+-10%) and rotation (+-10 degrees)
            A.ShiftScaleRotate(shift_limit=0, scale_limit=0.1, rotate_limit=10, p=0.5), 
            # Mask dropout
            #A.CoarseDropout(max_holes=4, max_height=64, max_width=64, p=0.5),  
            OneOf([
                A.GridDistortion(p=0.5),

                A.HorizontalFlip(p=0.5),
                # Random vertical flip
                A.VerticalFlip(p=0.5),
                # Random rotation
                A.RandomRotate90(p=0.5),
                # Switch x and y axis 
                A.Transpose(p=0.5), 
            ], p = 1), 
            OneOf([
                A.RandomBrightnessContrast(p=0.5) # Random brightness and contrast change
            ], p=1),
            OneOf([
                A.HueSaturationValue(p=0.5),       # HSV adjustment
                A.RGBShift(p=0.5),                 # RGB adjustment
            ], p=1),
            ToTensorV2()
        ], additional_targets= {
                'post_image' : 'image',
                'post_mask': 'mask'
        })
    return transform

def get_val_augmentation_pipeline(image_size=None, max_pixel_value=1, mean=None, std=None):
    transform = A.Compose([
        A.Resize(image_size[0], image_size[1]) if image_size is not None else A.NoOp(),
        A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value, p=1.0) if mean and std else A.NoOp(),
        ToTensorV2()
    ], additional_targets={
                'post_image' : 'image',
                'post_mask': 'mask'
        }
    )
    return transform

# xDB Tier3 Mean: [0.349 0.354 0.268], Std: [0.114 0.102 0.094]
# ImageNet Mean: [0.485, 0.456, 0.406], Std = [0.229, 0.224, 0.225]
# Levir-CD Mean: [0.387 , 0.382, 0.325], Std = [0.158 ,0.150, 0.138]

###### AutoEncoder Augmentation Pipeline #######
def get_train_autoencoder_augmentation_pipeline(image_size=None):
    return A.Compose(
        [
            A.Resize(image_size[0], image_size[1]) if image_size is not None else A.NoOp(),
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

