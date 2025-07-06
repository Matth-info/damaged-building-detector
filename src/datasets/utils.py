# Utils files for Dataset definitions
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from typing import Literal

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    "tif",
    "tiff",
]


def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized targets in a batch.

    Required for Instance Segmentation Dataloader

    Parameters
    ----------
        batch (list): List of (image, target) tuples.

    Returns:
    -------
        Tuple: (images, targets)

    """
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

    # Stack images into a single batch tensor
    images = torch.stack(images, dim=0)

    # Return images and list of targets
    return images, targets


def is_image_file(filepath):
    """Check if a Path object or string has an image file extension."""
    return filepath.suffix.lower() in IMG_EXTENSIONS


def train_val_test_split(list_labels: list, val_size: float, test_size: float, type: str) -> list:
    """Splits a list of labels into training, validation, and test sets based on specified proportions.

    Args:
    ----
        list_labels (list): List of labels to be split.
        val_size (float): Proportion of the dataset to allocate for validation (e.g., 0.2 for 20%).
        test_size (float): Proportion of the dataset to allocate for testing (e.g., 0.1 for 10%).
        type (str): Specifies which subset to return ('train', 'val', or 'test').

    Raises:
    ------
        ValueError: If the `type` argument is not one of 'train', 'val', or 'test'.

    Returns:
    -------
        list: A subset of labels corresponding to the specified `type`.

    Example:
    -------
        >>> labels = ["label1", "label2", "label3", "label4", "label5"]
        >>> train_labels = train_val_test_split(labels, val_size=0.2, test_size=0.2, type="train")
        >>> print(train_labels)
        ['label1', 'label2', 'label3']

    """
    total_size = len(list_labels)
    test_size = int(total_size * test_size)
    val_size = int(total_size * val_size)
    train_size = total_size - test_size - val_size

    indices = np.random.permutation(total_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    if type == "train":
        list_labels = [list_labels[i] for i in train_indices]
    elif type == "val":
        list_labels = [list_labels[i] for i in val_indices]
    elif type == "test":
        list_labels = [list_labels[i] for i in test_indices]
    else:
        raise ValueError("Unknown dataset type. Use 'train', 'val', or 'test'.")

    print(f"Loaded {len(list_labels)} {type} labels.")
    return list_labels
