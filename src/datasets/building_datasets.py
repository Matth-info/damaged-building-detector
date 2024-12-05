import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pathlib import Path
from typing import List, Dict, Optional
from torch.utils.data import Dataset
from shapely import wkt
import albumentations as A  # Import alias for Albumentations
import rasterio
import csv 
from tqdm import tqdm
import json
import numpy as np

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
        if self.cloud_filter_params is not None: 
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

        mask_image = (np.array(
            Image.open(self.mask_dir / f"{image_name}_mask.{self.extension}").convert("L") 
        ) > 0).astype(np.uint8)

        # Apply transformations if specified (Albumentations supports multiple inputs in the same pipeline)
        if self.transform:
            transformed = self.transform(image=pre_image / 255.0, mask=mask_image)
            transformed_bis = self.transform(image=post_image / 255.0)
            pre_image = transformed["image"].to(torch.float32) 
            post_image = transformed_bis["image"].to(torch.float32)
            mask_image = transformed["mask"].to(torch.long)
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
                # Replace with tensors of zeros
                h, w = pre_image.shape[1], pre_image.shape[2]  # Height and width of the images
                pre_image = torch.zeros((3, h, w), dtype=torch.float32)  # Zero tensor for pre-image
                post_image = torch.zeros((3, h, w), dtype=torch.float32)  # Zero tensor for post-image
                mask_image = torch.zeros((h, w), dtype=torch.long)  # Zero tensor for mask

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
        self, images_dir: str, 
        masks_dir: str, 
        transform: Optional[A.Compose] = None,
        filter_invalid_image=True
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
        from tqdm import tqdm
        import logging

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
                logging.warning(f"Error opening image or mask for {image_path}. Skipping this file.")

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
            transformed = self.transform(
                image=image.transpose(1, 2, 0), mask=mask
            )
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
            ax[i][1].imshow(image, alpha=0.5)
            ax[i][1].imshow(mask.squeeze(), alpha=0.5)
            ax[i][1].set_title(f"Mask {self.filenames[idx].name}")

            # Hide axes for cleaner display
            for j in range(2):
                ax[i][j].axis("off")

        plt.tight_layout()
        plt.show()




# Color codes for polygons
damage_dict = {
    "no-damage": (0, 255, 0, 50), # Green
    "minor-damage": (0, 0, 255, 50), # Blue
    "major-damage": (255, 69, 0, 50), # Red-Green
    "destroyed": (255, 0, 0, 50), # Red
    "un-classified": (255, 255, 255, 50)
}

class xDB_Damaged_Building(Dataset):
    def __init__(self, 
                 origin_dir: str,
                 mode = "building",
                 time = "pre",
                 transform: Optional[A.Compose] = None,
                 type: str = "train",
                 val_ratio=0.1, 
                 test_ratio=0.1, 
                 seed: int = 42
                 ):
        
        assert type in ["train", "val", "test"], "Dataset must be 'train','val' or 'test'"
        self.type = type
        np.random.seed(seed=seed)
        self.label_dir = Path(origin_dir) / "labels"
        assert mode in ["building", "damage"], "Mode must be 'building' or 'damage'."
        self.mode = mode 
        self.time = time
        self.list_labels = [str(x) for x in self.label_dir.rglob(pattern=f'*{self.time}_*.json')]
        self.transform = transform

        self.val_ratio=val_ratio
        self.test_ratio=test_ratio

        self._split() # perform a split datases train, val, test 
    
    def _split(self):
        # Calculate the size of each split
        total_size = len(self.list_labels)
        test_size = int(total_size * self.test_ratio)
        val_size = int(total_size * self.val_ratio)
        train_size = total_size - test_size - val_size

        # Shuffle indices for random splits
        indices = np.random.permutation(total_size)

        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Now, depending on self.type, choose which split to use
        if self.type == "train":
            self.list_labels = [self.list_labels[i] for i in train_indices]
        elif self.type == "val":
            self.list_labels = [self.list_labels[i] for i in val_indices]
        elif self.type == "test":
            self.list_labels = [self.list_labels[i] for i in test_indices]
        else:
            raise ValueError("Unknown dataset type. Use 'train', 'val', or 'test'.")

        # Now self.list_labels will contain the appropriate split based on self.type
        print(f"Loaded {len(self.list_labels)} {self.type} labels.")

    def __getitem__(self, index) -> Dict[str,torch.tensor]:
        """
        Retrieve an image and its corresponding mask for semantic segmentation.

        Parameters:
            index (int): Index of the image-mask pair.

        Returns:
            dict: A dictionary containing:
                - 'image': Transformed image as a torch tensor.
                - 'mask': Corresponding mask as a torch tensor.
        """
        # Get the JSON path for the given index
        json_path = self.list_labels[index]
        
        # Load the image
        image = self.get_image(json_path)
        
        # Load the mask
        mask = self.get_mask(json_path, mode=self.mode)
        
        # Apply transformations if defined
        if self.transform:
            # Compose image and mask into a single dict for joint transformation
            image = np.float32(np.array(image)) / 255.0 # Normalize to [0,1]
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"].float() # image is converted into torch.float32
            mask = transformed["mask"].long()  # mask is converted into torch.int64
        else:
            # Convert image and mask to tensors directly if no transform
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]
            mask = torch.from_numpy(mask).long() # mask is converted into torch.int64

        return {"image": image, "mask": mask}
    
    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.list_labels)
    
    def get_image(self, json_path) -> Image:
        """
        Open image file based on the json_path label file.
        
        Image path example: ../images/joplin-tornado_00000000_post_disaster.png
        Label path example: ../labels/joplin-tornado_00000000_post_disaster.json
        
        Parameters:
            json_path (str): Path to the label JSON file.
            time (str): "pre" for pre-disaster image, "post" for post-disaster image.
        
        Returns:
            PIL.Image.Image: The corresponding image.
        
        Raises:
            FileNotFoundError: If the image file is not found.
        """

        # Adjust path for "pre" time if needed
        if self.time == 'pre':
            json_path = json_path.replace('post', 'pre')

        # Replace 'labels' with 'images' and '.json' with '.png'
        img_path = json_path.replace('labels', 'images').replace('.json', '.png')
        # Open and return the image
        return Image.open(img_path)
    
    def get_damage_type(self, properties) -> str:
        if 'subtype' in properties:
            return properties['subtype']
        else:
            return 'no-damage'
    
    def get_mask(self, label_path, mode="building") -> np.ndarray:
        """
        Build a mask from a label file.

        Parameters:
            label_path (str): Path to the JSON label file.
            image_size (tuple): Size of the output mask (height, width).
            mode (str): "building" for binary mask, "damage" for multiclass mask.

        Returns:
            np.ndarray: Mask as a numpy array.
        """        
        # Load JSON file
        with open(label_path) as json_file:
            image_json = json.load(json_file)
        
        # Extract building polygons and their damage types
        metadata = image_json["metadata"]
        image_height , image_width = metadata["height"], metadata["width"]
        coords = image_json['features']["xy"]
        wkt_polygons = []

        for coord in coords:
            damage = self.get_damage_type(coord['properties'])  # Get damage type
            wkt_polygons.append((damage, coord['wkt']))
        
        # Convert WKT to Shapely polygons
        polygons = []
        for damage, swkt in wkt_polygons:
            polygons.append((damage, wkt.loads(swkt)))

        # Initialize the mask
        mask = Image.new('L', (image_height , image_width), 0)  # 'L' mode for grayscale, initialized to 0
        draw = ImageDraw.Draw(mask)

        # Define damage classes (used in "damage" mode)
        damage_classes = {
            "no-damage": 1,
            "minor-damage": 2,
            "major-damage": 3,
            "destroyed": 4
        }
        # Draw polygons on the mask
        for damage, polygon in polygons:
            if polygon.is_valid:  # Ensure the polygon is valid
                x, y = polygon.exterior.coords.xy
                coords = list(zip(x, y))
                if mode == "building":
                    draw.polygon(coords, fill=1)  # All buildings get a value of 1
                elif mode == "damage":
                    damage_value = damage_classes.get(damage, 0)  # Default to 0 for unknown
                    draw.polygon(coords, fill=damage_value)

        # Convert mask to numpy array
        return np.array(mask)
    
    def annotate_img(self, draw, coords):
        wkt_polygons = []
        for coord in coords:
            damage = self.get_damage_type(coord['properties'])
            wkt_polygons.append((damage, coord['wkt']))

        polygons = []

        for damage, swkt in wkt_polygons:
            polygons.append((damage, wkt.loads(swkt)))

        for damage, polygon in polygons:
            x,y = polygon.exterior.coords.xy
            coords = list(zip(x,y))
            draw.polygon(coords, damage_dict[damage])

        del draw

    def display_img(self, idx, time='pre', annotated=True):
        json_path = self.list_labels[idx]
        if time=='pre':
            json_path = json_path.replace('post', 'pre')
        
        img_path = json_path.replace('labels', 'images').replace('json','png')

        with open(json_path) as json_file:
            image_json = json.load(json_file)

        # Read Image 
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img, 'RGBA')

        if annotated:
            self.annotate_img(draw=draw,coords=image_json['features']['xy'])
        return img 

    def extract_metadata(self, idx, time="pre"):
        label_path = self.list_labels[idx]

        if time=='pre':
            label_path = label_path.replace('post', 'pre')
        elif time=="post":
            label_path = label_path.replace('pre', 'post')

        # Load JSON file
        with open(label_path) as json_file:
            image_json = json.load(json_file)
        
        return image_json["metadata"]
    
    def display_data(self, list_ids: List[int], time="pre", annotated=True, cols=3):
        """
        Display a list of images with or without annotations.

        Parameters:
            list_ids (List[int]): List of indices of images to display.
            annotated (bool): If True, overlay annotations on the images.
            cols (int): Number of columns in the grid for displaying images.
        """
        # Number of images
        num_images = len(list_ids)
        
        # Determine grid size
        rows = (num_images + cols - 1) // cols  # Ceiling division
        
        # Set up the matplotlib figure
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten()  # Flatten the axes array for easier indexing

        for i, idx in enumerate(list_ids):
            # Get the image
            img = self.display_img(idx,time=time, annotated=annotated)
            metadata = self.extract_metadata(idx)
            disaster_name, disaster_type = metadata["disaster"], metadata["disaster_type"]
            
            # Display the image
            axes[i].imshow(img)
            axes[i].set_title(f"Image {idx} / {disaster_name} / {disaster_type}")
            axes[i].axis("off")  # Hide axis ticks

        # Hide any remaining axes if the grid is larger than the number of images
        for j in range(num_images, len(axes)):
            axes[j].axis("off")
        
        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()

    def plot_image(self, idx, save = False):
        # read images
        img_A = self.display_img(idx, time='pre', annotated=False)
        img_B = self.display_img(idx, time='post', annotated=False)
        img_C = self.display_img(idx, time='pre', annotated=True)
        img_D = self.display_img(idx, time='post', annotated=True)

        # display images
        fig, ax = plt.subplots(2,2)
        fig.set_size_inches(30, 30)
        TITLE_FONT_SIZE = 24
        ax[0][0].imshow(img_A)
        ax[0][0].set_title('Pre Diaster Image (Not Annotated)', fontsize=TITLE_FONT_SIZE)
        ax[0][1].imshow(img_B)
        ax[0][1].set_title('Post Diaster Image (Not Annotated)', fontsize=TITLE_FONT_SIZE)
        ax[1][0].imshow(img_C)
        ax[1][0].set_title('Pre Diaster Image (Annotated)', fontsize=TITLE_FONT_SIZE)
        ax[1][1].imshow(img_D)
        ax[1][1].set_title('Post Diaster Image (Annotated)', fontsize=TITLE_FONT_SIZE)
        if save:
            plt.savefig('split_image.png', dpi = 100)
        plt.show()



###### Siamese Datasets ######
class xDB_Siamese_Dataset(Dataset):
    def __init__(self, 
                 origin_dir: str,
                 mode="damage",
                 transform: Optional[A.Compose] = None,
                 type: str = "train",
                 val_ratio = 0.1, 
                 test_ratio = 0.1, 
                 seed: int = 42
                 ):
        assert type in ["train", "val", "test"], "Dataset must be 'train', 'val' or 'test'"
        self.type = type
        np.random.seed(seed=seed)
        self.label_dir = Path(origin_dir) / "labels"
        assert mode in ["building", "damage"], "Mode must be 'building' or 'damage'."
        self.mode = mode
        self.list_labels = [str(x) for x in self.label_dir.rglob(pattern=f'*post_*.json')]
        self.transform = transform

        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self._split()  # Perform a split for train, val, and test datasets

    def _split(self):
        total_size = len(self.list_labels)
        test_size = int(total_size * self.test_ratio)
        val_size = int(total_size * self.val_ratio)
        train_size = total_size - test_size - val_size

        indices = np.random.permutation(total_size)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        if self.type == "train":
            self.list_labels = [self.list_labels[i] for i in train_indices]
        elif self.type == "val":
            self.list_labels = [self.list_labels[i] for i in val_indices]
        elif self.type == "test":
            self.list_labels = [self.list_labels[i] for i in test_indices]
        else:
            raise ValueError("Unknown dataset type. Use 'train', 'val', or 'test'.")

        print(f"Loaded {len(self.list_labels)} {self.type} labels.")

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        json_path = self.list_labels[index]

        # Load pre and post images
        pre_image = self.get_image(json_path, time="pre")
        post_image = self.get_image(json_path, time="post")

        # Load masks
        pre_mask = self.get_mask(json_path, time="pre", mode=self.mode)
        post_mask = self.get_mask(json_path, time="post", mode=self.mode)

        # Apply transformations if defined
        if self.transform:
            pre_image = np.float32(np.array(pre_image)) / 255.0
            post_image = np.float32(np.array(post_image)) / 255.0

            transformed_pre = self.transform(image=pre_image, mask=pre_mask)
            transformed_post = self.transform(image=post_image, mask=post_mask)

            pre_image = transformed_pre["image"].float()
            pre_mask = transformed_pre["mask"].long()

            post_image = transformed_post["image"].float()
            post_mask = transformed_post["mask"].long()
        else:
            pre_image = torch.from_numpy(np.array(pre_image)).permute(2, 0, 1).float() / 255.0
            pre_mask = torch.from_numpy(pre_mask).long()

            post_image = torch.from_numpy(np.array(post_image)).permute(2, 0, 1).float() / 255.0
            post_mask = torch.from_numpy(post_mask).long()

        return {"pre_image": pre_image, "post_image": post_image, "pre_mask": pre_mask, "post_mask": post_mask}

    def __len__(self) -> int:
        return len(self.list_labels)

    def get_image(self, json_path, time="pre") -> Image:
        if time == 'pre':
            json_path = json_path.replace('post', 'pre')

        img_path = json_path.replace('labels', 'images').replace('.json', '.png')
        return Image.open(img_path)

    def get_damage_type(self, properties) -> str:
        if 'subtype' in properties:
            return properties['subtype']
        else:
            return 'no-damage'

    def get_mask(self, label_path, time="post", mode="building") -> np.ndarray:
        if time == 'pre':
            label_path = label_path.replace('post', 'pre')

        with open(label_path) as json_file:
            image_json = json.load(json_file)

        metadata = image_json["metadata"]
        image_height, image_width = metadata["height"], metadata["width"]
        coords = image_json['features']["xy"]
        wkt_polygons = []

        for coord in coords:
            damage = self.get_damage_type(coord['properties'])
            wkt_polygons.append((damage, coord['wkt']))

        polygons = []
        for damage, swkt in wkt_polygons:
            polygons.append((damage, wkt.loads(swkt)))

        mask = Image.new('L', (image_height, image_width), 0)
        draw = ImageDraw.Draw(mask)

        damage_classes = {
            "no-damage": 1,
            "minor-damage": 2,
            "major-damage": 3,
            "destroyed": 4
        }

        for damage, polygon in polygons:
            if polygon.is_valid:
                x, y = polygon.exterior.coords.xy
                coords = list(zip(x, y))
                if mode == "building":
                    draw.polygon(coords, fill=1)
                elif mode == "damage":
                    damage_value = damage_classes.get(damage, 0)
                    draw.polygon(coords, fill=damage_value)

        return np.array(mask)

        
    def display_data(self, list_ids: List[int], annotated=True, cols=2):
        """
        Display a list of images with or without annotations.

        Parameters:
            list_ids (List[int]): List of indices of images to display.
            annotated (bool): If True, overlay annotations on the images.
            cols (int): Number of columns in the grid for displaying images.
        """

        from matplotlib.patches import Patch

        if cols <= 0:
            raise ValueError("Number of columns (cols) must be a positive integer.")

        # Number of images
        num_images = len(list_ids)
        if num_images == 0:
            print("No images to display.")
            return

        # Determine grid size
        rows = num_images # Ceiling division
        # Set up the matplotlib figure
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = axes.flatten() if rows > 1 or cols > 1 else [axes]  # Handle single image case
        # Damage classes and their corresponding colors
        if self.mode == "damage":
            damage_classes = {
                "no-damage": 1,
                "minor-damage": 2,
                "major-damage": 3,
                "destroyed": 4
            }
            damage_colors = {
                    0: (0, 0, 0, 0),  # Transparent background for class 0
                    1: (0, 1, 0, 1),  # Green with some transparency for "no-damage"
                    2: (1, 1, 0, 1),  # Yellow with some transparency for "minor-damage"
                    3: (1, 0.5, 0, 1),  # Orange with some transparency for "major-damage"
                    4: (1, 0, 0, 1)   # Red with some transparency for "destroyed"
                }
            legend_elements = [
                    Patch(facecolor=damage_colors[val], edgecolor=None, label=key)
                    for key, val in damage_classes.items()
                    ]
            cmap=plt.cm.colors.ListedColormap([damage_colors[val] for val in sorted(damage_colors.keys())])
        
        elif self.mode == "building":
            # Legend components
            legend_elements = [
                Patch(facecolor='red', edgecolor=None, label='building')
                ]
            cmap = "Reds"
            
        for i, idx in enumerate(list_ids):
            if idx < 0 or idx >= len(self):  # Validate index
                print(f"Index {idx} is out of bounds. Skipping...")
                continue

            # Get the image data
            data = self.__getitem__(idx)
            pre_image, pre_mask = data["pre_image"].permute(1, 2, 0).numpy(), data["pre_mask"].numpy()
            post_image, post_mask = data["post_image"].permute(1, 2, 0).numpy(), data["post_mask"].numpy()
          
            # Display pre-disaster image
            axes[2 * i].imshow(pre_image)
            if annotated:
                color_map_pre_image = plt.cm.colors.ListedColormap([(0, 0, 0, 0),(0, 1, 0, 1)])
                axes[2 * i].imshow(pre_mask, alpha=0.3, cmap = cmap if self.mode == "building" else color_map_pre_image) 
            axes[2 * i].set_title(f"Image {idx}: Pre-disaster")
            axes[2 * i].axis("off")

            # Display post-disaster image
            axes[2 * i + 1].imshow(post_image)
            if annotated:
                axes[2 * i + 1].imshow(post_mask, alpha=0.3, cmap=cmap)
            axes[2 * i + 1].set_title(f"Image {idx}: Post-disaster")
            axes[2 * i + 1].axis("off")

        # Hide unused axes
        for j in range(2 * num_images, len(axes)):
            axes[j].axis("off")

        # Add legend
        plt.figlegend(handles=legend_elements, loc='lower right', ncol=1, frameon=False, fontsize=10)
        
        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()