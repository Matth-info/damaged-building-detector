# Dataset folder keep track of the custom pytorch dataset that have been used to load, preprocess data according to the source dataset and the model specificity
import os 
import torch 
import albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd 
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt

class Cloud_DrivenData_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        x_paths: pd.DataFrame,
        bands: List[str],
        y_paths: Optional[pd.DataFrame] = None,
        transform: Optional[A.Compose] = None,
        pytorch: bool = True
    ):
        """
        Args:
            x_paths (pd.DataFrame): DataFrame containing file paths to the input image channels.
            bands (List[str]): List of band names to load.
            y_paths (Optional[pd.DataFrame]): DataFrame containing file paths for corresponding labels (masks).
            transform (Optional[A.Compose]): Albumentations transformations to apply to images and masks.
            pytorch (bool): Flag to indicate if the data should be in PyTorch's channel-first format (default is True).
        
        Note : The expected bands are B02 : Blue , B03 : Green, B04 : Red, B08 : nir (optional)
        RGB format is [B04, B03, B02]
        
        """
        super().__init__()
        self.data = x_paths
        self.label = y_paths
        self.bands = bands
        self.transform = transform
        self.pytorch = pytorch

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data)

    def load_channel(self, filepath: str) -> np.ndarray:
        """
        Loads a single image channel from the provided file path.
        
        Args:
            filepath (str): Path to the image file.

        Returns:
            np.ndarray: Loaded image as a NumPy array.
        """
        return np.array(Image.open(filepath))

    def open_mask(self, idx: int) -> np.ndarray:
        """
        Loads the mask from the provided file path.
        
        Args:
            idx (int): Index in self.label list.

        Returns:
            np.ndarray: Mask as a NumPy array (should already be in 0-1 range).
        """
        filepath = self.label.loc[idx]["label_path"]
        return self.load_channel(filepath)

    def open_as_array(self, idx: int) -> np.ndarray:
        """
        Loads the image channels for the sample at the given index and stacks them into a single array.
        
        Args:
            idx (int): Index of the sample in the DataFrame.        
        Returns:
            np.ndarray: Stacked image channels as a NumPy array, normalized to [0, 1].
        """
        # Load the channels based on the band names and stack them
        band_arrs = [self.load_channel(self.data.loc[idx][f"{band}_path"]) for band in self.bands]
        x_arr = np.stack(band_arrs, axis=-1)

        # Normalize the array (divide by max value to scale between 0 and 1)
        x_arr = (x_arr - x_arr.min()) / (x_arr.max() - x_arr.min()) 
        return x_arr

    def true_color_img(self, idx):
        visible_bands = ["B04", "B03", "B02"] #RGB
        band_arrs = [self.load_channel(self.data.loc[idx][f"{band}_path"]) for band in visible_bands]
        x_arr = np.stack(band_arrs, axis=-1)
        # Normalize the array (divide by max value to scale between 0 and 1)
        x_arr = (x_arr - x_arr.min()) / (x_arr.max() - x_arr.min()) 
        return x_arr 

    def __getitem__(self, idx: int) -> torch.Tensor:
        """ 
        Retrieves a single sample (image and label) from the dataset, applying transformations if specified.
        
        Args:
            idx (int): Index of the sample in the dataset.
        
        Returns:
            tuple: A tuple containing the image tensor and the mask tensor (if labels exist).
        """
        # Load image channels and optional mask
        x = self.open_as_array(idx).astype(np.float32) # numpy format (H,W,C)
        y = None
        if self.label is not None:
            y = self.open_mask(idx)

        # Apply transformations if provided : format (H,W,C)
        if self.transform:
            if y is not None:
                augmented = self.transform(image=x, mask=y)
                x, y = augmented['image'], augmented['mask']
            else:
                augmented = self.transform(image=x)
                x = augmented['image']
        
        # Convert to PyTorch tensors and normalize
        if self.transform is None: # numpy to torch tensor
            x = x.transpose((2, 0, 1))
            x = torch.tensor(x, dtype=torch.float32)
            if y is not None:
                y = torch.tensor(y, dtype=torch.int64)

        # Return image and mask (or just image if no mask)
        return {"image": x, "mask": y} if y is not None else x

    def __repr__(self) -> str:
        """
        String representation of the dataset class, showing the number of samples.
        
        Returns:
            str: Dataset class representation.
        """
        return f'Dataset class with {len(self)} samples'

    def display_data(self, list_indices: List[int]) -> None:
        """
        Displays a grid of images and their corresponding masks for a given list of sample indices.
        
        Args:
            list_indices (List[int]): List of indices to display.
        """
        num_samples = len(list_indices)
        rows = (num_samples + 1) // 2  # Calculate the number of rows for the subplot grid
        fig, ax = plt.subplots(rows, 2, figsize=(15, 5 * rows))

        # Handle cases where there is only one sample by making ax iterable
        if num_samples == 1:
            ax = [ax]

        for i, idx in enumerate(list_indices):
            # Load image and mask data
            x = self.true_color_img(idx)
            mask = self.open_mask(idx) if self.label is not None else None

            # Display image and mask
            ax[i][0].imshow(x)
            ax[i][0].set_title(f'Sample {idx + 1}')
            ax[i][0].axis('off')

            if mask is not None:
                ax[i][1].imshow(mask)
                ax[i][1].set_title(f'Ground truth {idx + 1}')
                ax[i][1].axis('off')
            else:
                ax[i][1].axis('off')

        # Adjust layout for better spacing
        plt.tight_layout()
        plt.show()
