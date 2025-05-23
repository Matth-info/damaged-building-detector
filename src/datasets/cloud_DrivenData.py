# Dataset folder keep track of the custom pytorch dataset that have been used to load, preprocess data according to the source dataset and the model specificity
from pathlib import Path
from typing import List, Optional

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from .base import Cloud_Dataset

__all__ = ["prepare_cloud_segmentation_data"]


class Cloud_DrivenData_Dataset(Cloud_Dataset):
    def __init__(
        self,
        x_paths: pd.DataFrame,
        bands: List[str] = ["B04", "B03", "B02"],
        y_paths: Optional[pd.DataFrame] = None,
        transform: Optional[A.Compose] = None,
        **kwargs,
    ):
        """
        Args:
            x_paths (pd.DataFrame): DataFrame containing file paths to the input image channels.
            bands (List[str]): List of band names to load.
            y_paths (Optional[pd.DataFrame]): DataFrame containing file paths for corresponding labels (masks).
            transform (Optional[A.Compose]): Albumentations transformations to apply to images and masks.

        Note : The expected bands are B02 : Blue , B03 : Green, B04 : Red, B08 : nir (optional)
        RGB format is [B04, B03, B02]

        """
        super().__init__()
        self.data = x_paths
        self.label = y_paths
        self.bands = bands
        self.transform = transform

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
        visible_bands = ["B04", "B03", "B02"]  # RGB
        band_arrs = [
            self.load_channel(self.data.loc[idx][f"{band}_path"]) for band in visible_bands
        ]
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
        x = self.open_as_array(idx).astype(np.float32)  # numpy format (H,W,C)
        y = None
        if self.label is not None:
            y = self.open_mask(idx)

        # Apply transformations if provided : format (H,W,C)
        if self.transform:
            if y is not None:
                augmented = self.transform(image=x, mask=y)
                x, y = augmented["image"], augmented["mask"]
            else:
                augmented = self.transform(image=x)
                x = augmented["image"]

        # Convert to PyTorch tensors and normalize
        if self.transform is None:  # numpy to torch tensor
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
        return f"Dataset class with {len(self)} samples"

    def display_data(self, list_indices: List[int]) -> None:
        """
        Displays a grid of images and their corresponding masks for a given list of sample indices.

        Args:
            list_indices (List[int]): List of indices to display.
        """
        num_samples = len(list_indices)
        rows = num_samples  # Calculate the number of rows for the subplot grid
        cols = 2  # Fixed number of columns: image and mask

        fig, ax = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        ax = ax.reshape(-1, cols) if rows > 1 else [ax]  # Ensure `ax` is a list of lists

        for i, idx in enumerate(list_indices):
            row = i

            # Load image and mask data
            x = self.true_color_img(idx)
            mask = self.open_mask(idx) if self.label is not None else None

            # Display image
            ax[row][0].imshow(x)
            ax[row][0].set_title(f"Sample {idx + 1}")
            ax[row][0].axis("off")

            # Display mask + image
            if mask is not None:
                ax[row][1].imshow(x, alpha=0.5)
                ax[row][1].imshow(mask, alpha=0.5)
                ax[row][1].set_title(f"Ground truth {idx + 1}")
                ax[row][1].axis("off")
            else:
                ax[row][1].axis("off")

        # Hide unused subplots if any
        for i in range(num_samples, rows * cols):
            ax[i // cols][i % cols].axis("off")

        # Adjust layout for better spacing
        plt.tight_layout()
        plt.show()


def add_paths(
    df: pd.DataFrame,
    feature_dir: Path,
    label_dir: Path = None,
    bands: list = ["B04", "B03", "B02"],
) -> pd.DataFrame:
    """
    Adds file paths for each band and label to the dataframe based on chip_id.

    Args:
        df (pd.DataFrame): DataFrame containing chip_id (e.g., image identifiers).
        feature_dir (Path): Directory where feature TIF images are stored.
        label_dir (Path, optional): Directory where label TIF images are stored. Defaults to None.
        bands (list): List of band names (e.g., ["B02", "B03", ...]). Defaults to BANDS.

    Returns:
        pd.DataFrame: Updated dataframe with new columns for each band path and label path.

    Adds the following columns to the dataframe:
        - "{band}_path" for each band image.
        - "label_path" for the label image, if `label_dir` is provided.
        - "has_{band}_path" boolean column indicating if the feature file exists.
        - "has_image_channels" boolean column indicating if all feature band files exist.
        - "has_label_path" boolean column indicating if the label file exists (if `label_dir` is provided).
        - "accessible" boolean column indicating if all image channels and label file exist.

    Ex: train_meta = add_paths(train_meta, TRAIN_FEATURES, TRAIN_LABELS)
    """
    # Ensure feature_dir and label_dir are Path objects
    feature_dir = Path(feature_dir)
    if label_dir is not None:
        label_dir = Path(label_dir)

    selected_columns = ["chip_id", "location", "datetime", "cloudpath"]

    # Initialize columns to track file existence for each band
    for band in bands:
        df[f"{band}_path"] = feature_dir / df["chip_id"] / f"{band}.tif"
        # Check if the band file exists and add a boolean column
        df[f"has_{band}_path"] = df[f"{band}_path"].apply(lambda x: x.exists())
        selected_columns.append(f"{band}_path")

    # Add "has_image_channels" to check if all bands exist
    df["has_image_channels"] = df[[f"has_{band}_path" for band in bands]].all(axis=1)
    # Add label path and check existence if label_dir is provided
    if label_dir:
        df["label_path"] = label_dir / (df["chip_id"] + ".tif")
        # Check if the label file exists and add a boolean column
        df["has_label_path"] = df["label_path"].apply(lambda x: x.exists())
        selected_columns.append("label_path")

    # Add "accessible" column to check if all bands and label file exist
    df["accessible"] = df["has_image_channels"] & df["has_label_path"]

    return df[df["accessible"]][selected_columns]


def prepare_cloud_segmentation_data(
    folder_path: str = "../data/Cloud_DrivenData/final/public",
    train_share: float = 0.8,
    seed: int = 42,
):
    """
    Data processing function to create training and validation datasets
    from the DrivenData Cloud Segmentation Challenge dataset.

    Args:
        folder_path (str): Path to the main dataset directory. Defaults to "../data/Cloud_DrivenData/final/public".
        train_share (float): Proportion of data to use for training (0 < train_share < 1). Defaults to 0.8.
        seed (int): Define the seed

    Returns:
        tuple: Four dataframes - train_x, train_y, val_x, val_y
               - train_x: Training features dataframe
               - train_y: Training labels dataframe
               - val_x: Validation features dataframe
               - val_y: Validation labels dataframe
    """
    import random
    from pathlib import Path

    import pandas as pd

    random.seed(seed)

    # Set up paths and constants
    DATA_DIR = Path(folder_path).resolve()
    TRAIN_FEATURES = DATA_DIR / "train_features"
    TRAIN_LABELS = DATA_DIR / "train_labels"
    TRAIN_META_FILE = DATA_DIR / "train_metadata.csv"
    BANDS = ["B04", "B03", "B02"]  # Bands to use; B08 can be added if needed

    # Ensure required directories and files exist
    assert TRAIN_FEATURES.exists(), f"Train features directory not found: {TRAIN_FEATURES}"
    assert TRAIN_META_FILE.exists(), f"Metadata file not found: {TRAIN_META_FILE}"

    # Load metadata
    train_meta = pd.read_csv(TRAIN_META_FILE)

    # Add paths for feature and label files
    train_meta = add_paths(train_meta, TRAIN_FEATURES, TRAIN_LABELS)

    # Compute validation share
    val_share = 1 - train_share

    # Split chip IDs into training and validation sets
    chip_ids = train_meta.chip_id.unique().tolist()
    val_chip_ids = random.sample(chip_ids, round(len(chip_ids) * val_share))

    # Mask for validation chips
    val_mask = train_meta.chip_id.isin(val_chip_ids)
    val = train_meta[val_mask].copy().reset_index(drop=True)
    train = train_meta[~val_mask].copy().reset_index(drop=True)

    # Separate features and labels
    feature_cols = ["chip_id"] + [f"{band}_path" for band in BANDS]
    train_x = train[feature_cols].copy()  # Training features
    train_y = train[["chip_id", "label_path"]].copy()  # Training labels
    val_x = val[feature_cols].copy()  # Validation features
    val_y = val[["chip_id", "label_path"]].copy()  # Validation labels

    return train_x, train_y, val_x, val_y
