from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, List, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .base import ChangeDetectionDataset
from .utils import is_image_file

if TYPE_CHECKING:
    from src.augmentation import Augmentation_pipeline


class LevirCDataset(ChangeDetectionDataset):
    """LEVIR-CD Dataset for change detection.

    Inspired by the original Levir-cd dataset definition : https://github.com/justchenhao/STANet/blob/master/data/changedetection_dataset.py
    The dataset structure is assumed to be:

    dataroot:
        ├── A (contains images before change)
        ├── B (contains images after change)
        ├── label (contains change detection labels)

    Args:
    ----
        origin_dir (str): Root directory for dataset (where A, B, and label are located).
        dataset_type (str): Dataset type - "train", "val", or "test".
        transform (callable, optional): Optional transform to be applied to the images and labels.

    """

    MEAN: ClassVar[list[float]] = [0.387, 0.382, 0.325]
    STD: ClassVar[list[float]] = [0.158, 0.150, 0.138]

    def __init__(
        self,
        origin_dir: str,
        dataset_type: Literal["train", "val", "test", "infer"] = "train",
        transform: Augmentation_pipeline | None = None,
        n_classes: int = 2,
        **kwargs,
    ) -> None:
        """Initialize LevirCD Dataset.

        Args:
            origin_dir (str): Origin directory.
            dataset_type (str, optional): Dataset type, choose between ['train', 'val', 'test', 'infer']. Defaults to "train".
            transform (Augmentation_pipeline | None, optional): Augmentation pipeline. Defaults to None.
            n_classes (int, optional): number of classes. Defaults to 2.
            **kwargs: other key arguments.
        """
        super().__init__(origin_dir, dataset_type, transform, n_classes)
        # Folder names

        self.dataset_type = dataset_type
        self.transform = transform
        root_dir = Path(origin_dir) / self.dataset_type

        # Set the path to each folder
        self.A_paths = sorted([x for x in (root_dir / "A").glob("*.*") if is_image_file(x)])
        self.B_paths = sorted([x for x in (root_dir / "B").glob("*.*") if is_image_file(x)])
        self.L_paths = sorted([x for x in (root_dir / "label").glob("*.*") if is_image_file(x)])

        logging.info("Loaded %d %s samples.", len(self), self.dataset_type)

    def __len__(self) -> int:
        """Return the number of image pairs."""
        return len(self.A_paths)

    def __getitem__(self, index) -> dict:
        """Retrieve one sample of A, B, and label (if available)."""
        # Load image pairs (A, B)
        img_a = Image.open(self.A_paths[index]).convert("RGB")  # Image before change
        img_b = Image.open(self.B_paths[index]).convert("RGB")  # Image after change
        label_img = Image.open(self.L_paths[index]).convert("L")  # Label mask (grayscale)

        # Convert images to numpy arrays (Albumentations expects this)
        pre_image = (np.array(img_a) / 255.0).astype(np.float32)
        post_image = (np.array(img_b) / 255.0).astype(np.float32)
        mask = np.array(label_img).astype(np.uint8)  # Convert to binary mask (0 or 1)
        mask = np.where(mask == 255, 1, 0)

        # Apply transformations if available
        if self.transform:
            transformed = self.transform(image=pre_image, mask=mask, post_image=post_image)
            pre_image = transformed["image"]
            post_image = transformed["post_image"]
            mask = transformed["mask"]
        else:
            pre_image = torch.from_numpy(pre_image).permute(2, 0, 1).float()
            post_image = torch.from_numpy(post_image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).long()

        return {"pre_image": pre_image, "post_image": post_image, "mask": mask}

    def display_data(self, list_indices: list[int]) -> None:
        """Display the pre-change, post-change, and mask images for a list of indices.

        Args:
        ----
            list_indices: list[int]: List of indices to display data for.

        """
        # Set up the plot
        num_images = len(list_indices)
        fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))

        # Ensure axes is 2D for easy indexing (in case there's only one image)
        if num_images == 1:
            axes = np.expand_dims(axes, axis=0)

        # Loop over each index in the provided list
        for i, index in enumerate(list_indices):
            data = self[index]

            pre_image = data["pre_image"].numpy().transpose(1, 2, 0)  # Convert back to HWC
            post_image = data["post_image"].numpy().transpose(1, 2, 0)  # Convert back to HWC
            mask = data["mask"].numpy() if "mask" in data else None

            # Pre-change image
            axes[i, 0].imshow(pre_image)
            axes[i, 0].set_title("Pre-Change Image")
            axes[i, 0].axis("off")

            # Post-change image
            axes[i, 1].imshow(post_image)
            axes[i, 1].set_title("Post-Change Image")
            axes[i, 1].axis("off")

            #  mask (change detection mask)
            if mask is not None:
                axes[i, 2].imshow(mask, cmap="gray")
                axes[i, 2].set_title("Mask (Change Detection)")
            else:
                axes[i, 2].axis("off")

        plt.tight_layout()
        plt.show()

    def compute_statistics(self) -> None:
        """Compute Mean and Standard Deviation and update statistics about the dataset."""
        total_images = len(self)
        mean_sum = np.zeros(3)
        std_sum = np.zeros(3)

        for i in range(total_images):
            data = self[i]
            pre_image = data["pre_image"].numpy().transpose(1, 2, 0)  # Convert to HWC
            post_image = data["post_image"].numpy().transpose(1, 2, 0)  # Convert to HWC

            mean_sum += pre_image.mean(axis=(0, 1))
            mean_sum += post_image.mean(axis=(0, 1))

            std_sum += pre_image.std(axis=(0, 1))
            std_sum += post_image.std(axis=(0, 1))

        mean = mean_sum / (2 * total_images)
        std = std_sum / (2 * total_images)

        logging.info("Dataset Mean: %f ", mean)
        logging.info("Dataset Std Dev: %f", std)
        self.MEAN = mean
        self.STD = std
        logging.info("MEAN and STD Levir-cd Dataset have been updated")
