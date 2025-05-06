from typing import List, Callable

import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.composition import OneOf

import numpy as np


def load_augmentation_pipeline(config_path="augmentation_config.yaml"):
    return A.load(config_path, data_format="yaml")


def save_augmentation_pipeline(transform: A.Compose = None, output_path: str = "augmentation_config.yaml"):
    A.save(transform, output_path, data_format="yaml")


#### Building Segmentation Model Augmentation Pipeline #####
def get_train_augmentation_pipeline(image_size=(256, 256), max_pixel_value=1, mean=None, std=None):
    transform = A.Compose(
        [
            # Resize images and masks
            A.RandomResizedCrop(height=image_size[0], width=image_size[1], scale=(0.8, 1), p=1.0)
            if image_size is not None
            else A.NoOp(),  # Ensure both image and mask are resized
            # Normalize images
            A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value, p=1.0) if mean and std else A.NoOp(),
            # Scale (+-10%) and rotation (+-10 degrees)
            A.ShiftScaleRotate(shift_limit=0, scale_limit=0.1, rotate_limit=10, p=0.5),
            # Mask dropout
            # A.CoarseDropout(max_holes=4, max_height=64, max_width=64, p=0.5),
            OneOf(
                [
                    A.HorizontalFlip(p=0.5),
                    # Random vertical flip
                    A.VerticalFlip(p=0.5),
                    # Random rotation
                    A.RandomRotate90(p=0.5),
                    # Switch x and y axis
                    A.Transpose(p=0.5),
                ],
                p=1,
            ),
            OneOf(
                [A.RandomBrightnessContrast(p=0.5)],  # Random brightness and contrast change
                p=1,
            ),
            OneOf(
                [
                    A.HueSaturationValue(p=0.5),  # HSV adjustment
                    A.RGBShift(p=0.5),  # RGB adjustment
                ],
                p=1,
            ),
            ToTensorV2(),
        ],
        additional_targets={"post_image": "image", "post_mask": "mask"},
    )
    return transform


def get_val_augmentation_pipeline(image_size=None, max_pixel_value=1, mean=None, std=None):
    transform = A.Compose(
        [
            A.Resize(image_size[0], image_size[1]) if image_size is not None else A.NoOp(),
            A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value, p=1.0) if mean and std else A.NoOp(),
            ToTensorV2(),
        ],
        additional_targets={"post_image": "image", "post_mask": "mask"},
    )
    return transform


###### AutoEncoder Augmentation Pipeline #######
def get_train_autoencoder_augmentation_pipeline(image_size=None):
    return A.Compose(
        [
            A.Resize(image_size[0], image_size[1]) if image_size is not None else A.NoOp(),
            A.HorizontalFlip(p=0.5),  # Random horizontal flip with 50% probability
            A.VerticalFlip(p=0.5),  # Random vertical flip with 50% probability
            A.RandomRotate90(p=0.5),  # Random 90 degree rotation with 50% probability
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=0
            ),  # Random shift, scale, and rotation with fill at borders
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),  # Adjust color properties
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.5
            ),  # Adjust brightness and contrast
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),  # Apply a Gaussian blur
            A.GaussNoise(var_limit=(10, 50), p=0.3),  # Add Gaussian noise
            ToTensorV2(),  # Convert to PyTorch tensors
        ],
        is_check_shapes=True,
    )


def get_val_autoencoder_augmentation_pipeline(image_size=(512, 512)):
    return A.Compose(
        [
            A.Resize(image_size[0], image_size[1]),
            ToTensorV2(),
        ],
        is_check_shapes=True,
    )
