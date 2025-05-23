import os
from pathlib import Path
from typing import Dict, List, Optional

import albumentations as A
import numpy as np
import rasterio
import torch

from .base import Segmentation_Dataset


class OpenCities_Building_Dataset(Segmentation_Dataset):
    MEAN = None
    STD = None

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform: Optional[A.Compose] = None,
        filter_invalid_image=True,
        **kwargs,
    ):
        """
        Initializes the OpenCities dataset with images and corresponding masks.

        Args:
            images_dir (str): Directory containing the image files.
            masks_dir (str): Directory containing the mask files.
            transform (Optional[A.Compose]): Albumentations transformation pipeline.

        Checklist:
            dataset creation : Done
            dataset iteration : Done
            data augmentation with albumentation : Done
            display data : Done
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.filenames = [Path(x) for x in self.images_dir.glob("*.tif")]
        self.nb_input_channel = 3

        if filter_invalid_image:
            self.remove_invalid_sample()

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.filenames)

    def extract_filename(self, filepath: Path) -> str:
        # Split the string at '.tif' and return the part before it
        return filepath.stem

    def remove_invalid_sample(self):
        """
        Removes invalid samples (image and mask) from the dataset.
        Some images might be corrupted and cannot be opened by rasterio.
        """
        import logging

        from tqdm import tqdm

        valid_filenames = []  # Store valid filenames

        for image_path in tqdm(self.filenames, desc="Removing Invalid Samples"):
            filename = self.extract_filename(image_path)
            filename_mask = f"{filename}_mask.tif"
            mask_path = os.path.join(self.masks_dir, filename_mask)

            # Try to load the image and the corresponding mask
            try:
                image = self.read_image(image_path)
                mask = self.read_mask(mask_path)
                # If no exception, keep this sample
                valid_filenames.append(image_path)
            except (rasterio.errors.RasterioIOError, FileNotFoundError) as e:
                # Log the error and skip this sample
                logging.warning(
                    f"Error opening image or mask for {image_path}. Skipping this file."
                )

        # Update the dataset with only valid filenames
        self.filenames = valid_filenames

    def __getitem__(self, idx: int):
        """Fetches and returns a single dataset sample with optional transformations."""
        image_path = self.filenames[idx]
        filename = self.extract_filename(
            image_path
        )  # Convert Path to string before extracting filename
        filename_mask = f"{filename}_mask.tif"
        mask_path = os.path.join(self.masks_dir, filename_mask)
        image = self.read_image(image_path)
        mask = self.read_mask(mask_path)

        # Apply transformations if specified
        if self.transform is not None:
            # Albumentation expect (H, W, C) images
            transformed = self.transform(image=image.transpose(1, 2, 0), mask=mask)
            image = transformed["image"].to(torch.float32) / 255
            mask = transformed["mask"].to(torch.int64)
        else:
            image = torch.tensor(image, dtype=torch.float32) / 255
            mask = torch.tensor(mask, dtype=torch.int64)

        return {"image": image, "mask": mask}  # {"image" : ..., "mask" : ...}

    def read_image(self, path: Path):
        """Reads and returns the image from a given path."""
        with rasterio.open(path) as f:
            image = f.read()  # Read the image as a multi-band array
        image = image[: self.nb_input_channel]  # (C, H, W)
        return image

    def read_mask(self, path: Path):
        """Reads and returns the mask from a given path."""
        with rasterio.open(path) as f:
            mask = f.read(1)  # Read only the first band (grayscale mask)
        return np.where(mask == 255, 1, 0)  # binary mask 0 : no building and 1 : building

    def read_image_profile(self, id: str):
        """Reads the image profile (metadata) for a given image."""
        path = self.images_dir / id
        with rasterio.open(path) as f:
            return f.profile

    def display_data(self, list_indices: List[int]) -> None:
        import matplotlib.pyplot as plt

        num_samples = len(list_indices)

        # Set up a subplot grid dynamically based on the number of samples
        fig, ax = plt.subplots(num_samples, 2, figsize=(15, 5 * num_samples))

        # Handle cases with a single sample by wrapping axes in a list
        if num_samples == 1:
            ax = [ax]

        for i, idx in enumerate(list_indices):
            sample = self.__getitem__(idx)
            image = sample["image"]
            # Convert from (C, H, W) to (H, W, C) for compatibility reasons

            # Check if the image is a torch tensor
            if isinstance(image, torch.Tensor):
                # Convert (C, H, W) to (H, W, C) for PyTorch tensor
                image = image.permute(1, 2, 0).numpy()  # Convert to NumPy array
            elif isinstance(image, np.ndarray):
                # If it's already a numpy array, no need to permute
                if image.ndim == 3 and image.shape[0] == 3:  # (C, H, W)
                    image = image.transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)

            mask = np.where(sample["mask"] == 1, 255, 0)  # For compatibility and readability

            # Display the image and the corresponding mask
            ax[i][0].imshow(image)
            ax[i][0].set_title(f"Image {self.filenames[idx].name}")
            ax[i][1].imshow(image, alpha=0.5)
            ax[i][1].imshow(mask.squeeze(), alpha=0.5)
            ax[i][1].set_title(f"Mask {self.filenames[idx].name}")

            # Hide axes for cleaner display
            for j in range(2):
                ax[i][j].axis("off")

        plt.tight_layout()
        plt.show()
