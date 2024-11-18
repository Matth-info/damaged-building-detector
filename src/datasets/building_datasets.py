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
import rasterio
import re
import glob
import csv 
from tqdm import tqdm

class Puerto_Rico_Building_Dataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        pre_disaster_dir: str,
        post_disaster_dir: str,
        mask_dir: str,
        transform: Optional[A.Compose] = None,
        extension: str = "jpg",
        cloud_filter_params=None,
        preprocessing_mode: Optional[str] = "none",  # 'none', 'online', or 'offline'
        filtered_list_path: Optional[str] = None,  # Path to save/load filtered filenames
    ):
        """
        Initializes the dataset with directories for pre- and post-disaster images and masks.

        Args:
            base_dir (str): Base directory path containing all subdirectories.
            pre_disaster_dir (str): Directory with pre-disaster images.
            post_disaster_dir (str): Directory with post-disaster images.
            mask_dir (str): Directory with segmentation masks.
            transform (Optional[A.Compose]): Albumentations transformation pipeline.
            extension (str): File extension for images (default is 'jpg').
            cloud_filter_params (dict): Parameters for the cloud filter model.
            preprocessing_mode (str): Mode for filtering ('none', 'online', 'offline').
            filtered_list_path (str): Path to save/load filtered filenames for offline mode.
        """
        self.base_dir = Path(base_dir)
        self.pre_disaster_dir = self.base_dir / pre_disaster_dir
        self.post_disaster_dir = self.base_dir / post_disaster_dir
        self.mask_dir = self.base_dir / mask_dir
        self.transform = transform
        self.extension = extension
        self.filename_pattern = re.compile(r"tile_\d+_\d+")
        self.cloud_filter_params = cloud_filter_params
        self.preprocessing_mode = preprocessing_mode
        self.filtered_list_path = filtered_list_path

        # Gather initial filenames
        self.image_filenames = self._find_image_filenames()

        # Load cloud filter if available
        self.cloud_filter_load()

        # Apply offline preprocessing if specified
        if self.preprocessing_mode == "offline":
            self.image_filenames = self._offline_filter_images()

        
    def cloud_filter_load(self):
        """Loads the cloud filter model if specified."""
        if self.cloud_filter_params is not None:
            model_class = self.cloud_filter_params.get("model_class")
            device = self.cloud_filter_params.get("device")
            file_path = self.cloud_filter_params.get("file_path")

            self.cloud_filter = model_class.load(file_path=file_path).eval()  # Evaluation mode
            self.cloud_filter.to(device)
            print(f"A {model_class.__class__.__name__} model has been loaded from {file_path} on {device}.")
        else:
            self.cloud_filter = None

    def cloud_filter_batch(self, batch) -> List[bool]:
        """
        Filters a batch of images based on cloud reconstruction loss.

        Args:
            batch (dict): A dictionary containing "pre_image" and "post_image".
                        Each key maps to a tensor of shape [batch_size, channels, height, width].

        Returns:
            dict: A dictionary containing filtered "image_pre", "image_post", and a "mask" indicating non-cloudy images.
        """
        # Extract parameters
        loss_fn = self.cloud_filter_params.get("loss")(reduction="none")  # Use non-aggregated loss
        threshold = self.cloud_filter_params.get("threshold")  # Loss threshold for filtering
        device = self.cloud_filter_params.get('device')  # Device (e.g., 'cuda' or 'cpu')

        # Move images to device
        pre_images = batch["pre_image"].to(device)
        post_images = batch["post_image"].to(device)

        with torch.no_grad():
            # Reconstruct images
            reconstructed_pre_images = self.cloud_filter.predict(pre_images)
            reconstructed_post_images = self.cloud_filter.predict(post_images)

            # Compute pixel-wise losses
            loss_pre = loss_fn(reconstructed_pre_images, pre_images).mean(dim=[1, 2, 3])  # Per-image loss
            loss_post = loss_fn(reconstructed_post_images, post_images).mean(dim=[1, 2, 3])  # Per-image loss

            # Determine cloudiness based on threshold
            is_cloudy = (loss_pre < threshold) | (loss_post < threshold)  # Cloudy if either loss less than a threshold
            not_cloudy = ~is_cloudy  # Non-cloudy mask

            # Return results
            return not_cloudy.cpu().numpy()

    def _offline_filter_images(self) -> List[str]:
        """Applies offline cloud filtering and saves filtered filenames."""
        
        if self.filtered_list_path and os.path.exists(self.filtered_list_path):
            # Load precomputed filtered list from CSV file
            print(f"Loading filtered filenames from {self.filtered_list_path}...")
            with open(self.filtered_list_path, "r", newline='') as f:
                reader = csv.reader(f)
                return [row[0] for row in reader]  # Each row contains one filename

        print("Applying offline filtering...")

        # Prepare a DataLoader for batch processing
        batch_size = self.cloud_filter_params.get("batch_size", 32)
        data_loader = torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=False)

        filtered_filenames = []
        with tqdm(data_loader, desc=f"Offline Filtering", unit="batch") as t:
            for batch_num, batch in enumerate(t):
                batch_filter = self.cloud_filter_batch(batch)
                for idx, keep in enumerate(batch_filter):
                    if keep:
                        image_id = batch_num * batch_size + idx 
                        filtered_filenames.append(self.image_filenames[image_id])
            # Save filtered filenames
        if self.filtered_list_path:
            with open(self.filtered_list_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                # Write each filename in a new row
                for filename in filtered_filenames:
                    writer.writerow([filename])
        # save the filtered_filenames in a txt files
        return filtered_filenames 

    def _find_image_filenames(self) -> List[str]:
        """
        Finds and returns a list of filenames common to pre, post, and mask directories.

        Returns:
            List[str]: Sorted list of filenames (without extensions) common to all directories.
        """
        # Collect file stems (base names without extensions)
        pre_files = set(
            f.stem for f in self.pre_disaster_dir.glob(f"*.{self.extension}")
        )
        post_files = set(
            f.stem for f in self.post_disaster_dir.glob(f"*.{self.extension}")
        )
        mask_files = set(
            f.stem.replace("_mask", "") for f in self.mask_dir.glob(f"*_mask.{self.extension}")
        )

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
        pre_image = np.array(
            Image.open(
                self.pre_disaster_dir / f"{image_name}.{self.extension}"
            ).convert("RGB")
        ).astype(np.float32)

        post_image = np.array(
            Image.open(
                self.post_disaster_dir / f"{image_name}.{self.extension}"
            ).convert("RGB")
        ).astype(np.float32)

        mask_image = np.array(
            Image.open(self.mask_dir / f"{image_name}_mask.{self.extension}").convert("L")
        )

        # Apply transformations if specified (Albumentations supports multiple inputs in the same pipeline)
        if self.transform:
            transformed = self.transform(image=pre_image, mask=mask_image)
            transformed_bis = self.transform(image=post_image)
            pre_image = transformed["image"]
            post_image = transformed_bis["image"]
            mask_image = transformed["mask"]
        else:
            # Convert images and mask to tensors with normalization for compatibility with PyTorch
            pre_image = (torch.tensor(pre_image, dtype=torch.float32).permute(2, 0, 1) / 255.0)
            post_image = (torch.tensor(post_image, dtype=torch.float32).permute(2, 0, 1) / 255.0)
            mask_image = torch.tensor(mask_image, dtype=torch.long)

        # Online filtering
        if self.preprocessing_mode == "online" and self.cloud_filter:
            batch = {"pre_image": pre_image.unsqueeze(0), "post_image": post_image.unsqueeze(0)}
            not_cloudy = self.cloud_filter_batch(batch)[0] # First (and only) sample in batch
            if not not_cloudy:
                raise IndexError(f"Filtered out image: {image_name}")

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
        pre_image = (
            np.array(
                Image.open(
                    self.pre_disaster_dir / f"{image_name}.{self.extension}"
                ).convert("RGB")
            )
            / 255.0
        )
        post_image = (
            np.array(
                Image.open(
                    self.post_disaster_dir / f"{image_name}.{self.extension}"
                ).convert("RGB")
            )
            / 255.0
        )

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
        return np.array(Image.open(self.mask_dir / f"{image_name}_mask.{self.extension}"))

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
            ax[i][0].set_title(f"Pre Event {self.image_filenames[idx]}")
            ax[i][1].imshow(post_image)
            ax[i][1].set_title(f"Post Event {self.image_filenames[idx]}")
            ax[i][2].imshow(mask_image, cmap="gray")
            ax[i][2].set_title(f"Ground Truth {self.image_filenames[idx]}")

            # Hide axes for cleaner display
            for j in range(3):
                ax[i][j].axis("off")

        plt.tight_layout()
        plt.show()


class OpenCities_Building_Dataset(Dataset):
    def __init__(
        self, images_dir: str, masks_dir: str, transform: Optional[A.Compose] = None
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

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.filenames)

    def extract_filename(self, filepath: Path) -> str:
        # Split the string at '.tif' and return the part before it
        return filepath.stem

    def __getitem__(self, idx: int):
        """Fetches and returns a single dataset sample with optional transformations."""
        image_path = self.filenames[idx]
        filename = self.extract_filename(
            image_path
        )  # Convert Path to string before extracting filename
        filename_mask = f"{filename}_mask.tif"
        mask_path = os.path.join(self.masks_dir, filename_mask)

        # Load the image and the corresponding mask
        sample = {
            "image": self.read_image(image_path),
            "mask": self.read_mask(mask_path),
        }

        # Apply transformations if specified
        if self.transform is not None:
            # Albumentation expect (H, W, C) images
            transformed = self.transform(
                image=sample["image"].transpose(1, 2, 0), mask=sample["mask"]
            )
            sample = transformed
        return sample  # {"image" : ..., "mask" : ...}

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
        return np.where(
            mask == 255, 1, 0
        )  # binary mask 0 : no building and 1 : building

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
                    image = image.transpose(
                        1, 2, 0
                    )  # Convert from (C, H, W) to (H, W, C)

            mask = np.where(
                sample["mask"] == 1, 255, 0
            )  # For compatibility and readibility

            # Display the image and the corresponding mask
            ax[i][0].imshow(image)
            ax[i][0].set_title(f"Image {self.filenames[idx].name}")
            ax[i][1].imshow(image)
            ax[i][1].imshow(mask.squeeze(), alpha=0.5)
            ax[i][1].set_title(f"Mask {self.filenames[idx].name}")

            # Hide axes for cleaner display
            for j in range(2):
                ax[i][j].axis("off")

        plt.tight_layout()
        plt.show()
