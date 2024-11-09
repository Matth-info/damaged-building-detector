import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional
from torch.utils.data import Dataset
import albumentations as A  # Import alias for Albumentations

class Puerto_Rico_Building_Dataset(Dataset):
    def __init__(self, base_dir: str, pre_disaster_dir: str, post_disaster_dir: str, mask_dir: str, 
                 transform: Optional[A.Compose] = None, extension: str = "jpg"):
        """
        Initializes the dataset with directories for pre- and post-disaster images and masks.
        
        Args:
            base_dir (str): Base directory path containing all subdirectories.
            pre_disaster_dir (str): Directory with pre-disaster images.
            post_disaster_dir (str): Directory with post-disaster images.
            mask_dir (str): Directory with segmentation masks.
            transform (Optional[A.Compose]): Albumentations transformation pipeline.
            extension (str): File extension for images (default is 'jpg').
        """
        self.base_dir = Path(base_dir)
        self.pre_disaster_dir = self.base_dir / pre_disaster_dir
        self.post_disaster_dir = self.base_dir / post_disaster_dir
        self.mask_dir = self.base_dir / mask_dir
        self.transform = transform
        self.extension = extension
        self.filename_pattern = re.compile(r"tile_\d+_\d+")  # Regex to match filenames in the format `tile_i_j`

        # Gather filenames that have images in all directories (pre, post, and mask)
        self.image_filenames = self._find_image_filenames()

    def _find_image_filenames(self) -> List[str]:
        """
        Finds and returns a list of filenames common to pre, post, and mask directories.
        
        Returns:
            List[str]: Sorted list of filenames (without extensions) common to all directories.
        """
        # Collect file stems (base names without extensions)
        pre_files = set(f.stem for f in self.pre_disaster_dir.glob(f"*.{self.extension}"))
        post_files = set(f.stem for f in self.post_disaster_dir.glob(f"*.{self.extension}"))
        mask_files = set(f.stem.replace("_mask", "") for f in self.mask_dir.glob("*_mask.jpg"))
        
        # Find common filenames among all sets
        common_tiles = pre_files & post_files & mask_files
        return sorted(common_tiles)

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.image_filenames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Fetches and returns a single dataset sample with optional transformations.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing pre- and post-disaster images and mask.
        """
        # Retrieve the filename based on the index
        image_name = self.image_filenames[idx]
        
        # Load images and mask, converting to appropriate modes (RGB for images, grayscale for mask)
        pre_image = np.array(Image.open(self.pre_disaster_dir / f"{image_name}.{self.extension}").convert("RGB")).astype(np.float32)
        post_image = np.array(Image.open(self.post_disaster_dir / f"{image_name}.{self.extension}").convert("RGB")).astype(np.float32)
        mask_image = np.array(Image.open(self.mask_dir / f"{image_name}_mask.jpg").convert("L"))
        
        # Apply transformations if specified (Albumentations supports multiple inputs in the same pipeline)
        if self.transform:
            transformed = self.transform(image=pre_image, mask=mask_image)
            transformed_bis = self.transform(image=post_image)
            pre_image = transformed["image"]
            post_image = transformed_bis["image"]
            mask_image = transformed["mask"]
        
        if self.transform is None:
            # Convert images and mask to tensors with normalization for compatibility with PyTorch
            pre_image = torch.tensor(pre_image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            post_image = torch.tensor(post_image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            mask_image = torch.tensor(mask_image, dtype=torch.long)
        
        return {"pre_image": pre_image, "post_image": post_image, "mask": mask_image}

    def open_as_array(self, idx: int, invert: bool = False) -> tuple:
        """
        Opens and returns pre- and post-disaster images as numpy arrays with optional inversion.
        
        Args:
            idx (int): Index of the sample to open.
            invert (bool): If True, the images are transposed to channel-first format.
        
        Returns:
            tuple: Pre- and post-disaster images as numpy arrays.
        """
        image_name = self.image_filenames[idx]
        pre_image = np.array(Image.open(self.pre_disaster_dir / f"{image_name}.{self.extension}").convert("RGB")) / 255.0
        post_image = np.array(Image.open(self.post_disaster_dir / f"{image_name}.{self.extension}").convert("RGB")) / 255.0

        if invert:
            # Transpose to channel-first format if required
            pre_image = pre_image.transpose((2, 0, 1))
            post_image = post_image.transpose((2, 0, 1))

        return pre_image, post_image
    
    def open_mask(self, idx: int) -> np.ndarray:
        """
        Opens and returns the mask image as a numpy array.
        
        Args:
            idx (int): Index of the mask to open.
        
        Returns:
            np.ndarray: Grayscale mask image.
        """
        image_name = self.image_filenames[idx]
        return np.array(Image.open(self.mask_dir / f"{image_name}_mask.jpg"))

    def display_data(self, list_indices: List[int]) -> None:
        """
        Displays pre- and post-disaster images alongside the corresponding mask.
        
        Args:
            list_indices (List[int]): List of sample indices to display.
        """
        num_samples = len(list_indices)
        
        # Set up a subplot grid dynamically based on the number of samples
        fig, ax = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

        # Handle cases with a single sample by wrapping axes in a list
        if num_samples == 1:
            ax = [ax]
        
        for i, idx in enumerate(list_indices):
            pre_image, post_image = self.open_as_array(idx)
            mask_image = self.open_mask(idx)

            # Display each of the images in the appropriate subplot
            ax[i][0].imshow(pre_image)
            ax[i][0].set_title(f'Pre Event {self.image_filenames[idx]}')
            ax[i][1].imshow(post_image)
            ax[i][1].set_title(f'Post Event {self.image_filenames[idx]}')
            ax[i][2].imshow(mask_image, cmap="gray")
            ax[i][2].set_title(f'Ground Truth {self.image_filenames[idx]}')
            
            # Hide axes for cleaner display
            for j in range(3):
                ax[i][j].axis("off")

        plt.tight_layout()
        plt.show()
