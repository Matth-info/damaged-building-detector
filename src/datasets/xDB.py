from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np

# torch imports
import torch
from PIL import Image, ImageDraw
from shapely import wkt
from torchvision import transforms, tv_tensors
from torchvision.ops.boxes import masks_to_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.v2 import functional as F
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from src.datasets.utils import train_val_test_split

from .base import ChangeDetectionDataset, InstanceSegmentationDataset, SegmentationDataset

if TYPE_CHECKING:
    from src.augmentation import Augmentation_pipeline

# Color codes for polygons
damage_dict = {
    "no-damage": (0, 255, 0, 50),  # Green
    "minor-damage": (0, 0, 255, 50),  # Blue
    "major-damage": (255, 69, 0, 50),  # Red-Green
    "destroyed": (255, 0, 0, 50),  # Red
    "un-classified": (255, 255, 255, 50),
}


# Segmentation Dataset #
class xDBDamagedBuilding(SegmentationDataset):
    """xDB Damage Building Dataset. From xDB Dataset, a semantic segmentation task dataset for single image damage building detection."""

    MEAN: ClassVar[list[float]] = [0.349, 0.354, 0.268]
    STD: ClassVar[list[float]] = [0.114, 0.102, 0.094]

    def __init__(
        self,
        origin_dir: str,
        mode: Literal["building", "damage"] = "building",
        time: Literal["pre", "post"] = "pre",
        transform: Augmentation_pipeline | None = None,
        dataset_type: str = "train",
        n_classes: int | None = None,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        **kwargs: object,
    ) -> None:
        """Initialize the xDBDamagedBuilding dataset.

        Args:
        origin_dir : str
            Directory containing the dataset.
        mode : str, optional
            "building" or "damage" to determine the dataset_type of masks to load.
        time : str, optional
            Temporal filter for the dataset (e.g., "pre" or "post").
        transform : Augmentation_pipeline | None, optional
            Optional transformation to apply on images and masks.
        dataset_type : str, optional
            "train", "val", or "test" to determine the dataset split.
        n_classes : int, optional
            Number of classes for segmentation.
        val_ratio : float, optional
            Fraction of the dataset to use for validation.
        test_ratio : float, optional
            Fraction of the dataset to use for testing.
        seed : int, optional
            Random seed for reproducibility.
        **kwargs:
            Additional keyword arguments.
        """
        super().__init__(
            origin_dir=origin_dir,
            dataset_type=dataset_type,
            transform=transform,
            n_classes=n_classes,
        )
        np.random.default_rng(seed=seed)

        self.label_dir = Path(origin_dir) / "labels"
        self.mode = mode
        self.time = time
        self.list_labels = [str(x) for x in self.label_dir.rglob(pattern=f"*{self.time}_*.json")]
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.n_classes = 2 if mode == "building" else 5

        self._split()  # perform a split datases train, val, test

    def _split(self) -> None:
        self.list_labels = train_val_test_split(
            self.list_labels,
            val_size=self.val_ratio,
            test_size=self.test_ratio,
            dataset_type=self.dataset_type,
        )

    def __getitem__(self, index: int) -> dict[str, torch.tensor]:
        """Retrieve an image and its corresponding mask for semantic segmentation.

        Args:
            index (int): Index of the image-mask pair.

        Returns:
            dict: A dictionary containing:
                   'image': Transformed image as a torch tensor.
                   'mask': Corresponding mask as a torch tensor.

        """
        # Get the JSON path for the given index
        json_path = self.list_labels[index]

        # Load the image
        image = self._get_image(json_path)

        # Load the mask
        mask = self._get_mask(json_path, mode=self.mode)

        # Apply transformations if defined
        if self.transform:
            # Compose image and mask into a single dict for joint transformation
            image = np.float32(np.array(image)) / 255.0  # Normalize to [0,1]
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"].float()  # image is converted into torch.float32
            mask = transformed["mask"].long()  # mask is converted into torch.int64
        else:
            # Convert image and mask to tensors directly if no transform
            image = (
                torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            )  # Normalize to [0, 1]
            mask = torch.from_numpy(mask).long()  # mask is converted into torch.int64

        return {"image": image, "mask": mask}

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.list_labels)

    def _get_image(self, json_path: str) -> Image:
        """Open image file based on the json_path label file.

        Image path example: ../images/joplin-tornado_00000000_post_disaster.png
        Label path example: ../labels/joplin-tornado_00000000_post_disaster.json

        Args:
            json_path (str): Path to the label JSON file.
            time (str): "pre" for pre-disaster image, "post" for post-disaster image.

        Returns:
            PIL.Image.Image: The corresponding image.

        Raises:
        ------
            FileNotFoundError: If the image file is not found.

        """
        # Adjust path for "pre" time if needed
        if self.time == "pre":
            json_path = json_path.replace("post", "pre")

        # Replace 'labels' with 'images' and '.json' with '.png'
        img_path = json_path.replace("labels", "images").replace(".json", ".png")
        # Open and return the image
        return Image.open(img_path)

    def _get_damage_type(self, properties: dict) -> str:
        if "subtype" in properties:
            return properties["subtype"]
        return "no-damage"

    def _get_mask(self, label_path: str, mode="building") -> np.ndarray:
        """Build a mask from a label file.

        Args:
            label_path (str): Path to the JSON label file.
            image_size (tuple): Size of the output mask (height, width).
            mode (str): "building" for binary mask, "damage" for multiclass mask.

        Returns:
            np.ndarray: Mask as a numpy array.

        """
        # Load JSON file
        with Path.open(label_path) as json_file:
            image_json = json.load(json_file)

        # Extract building polygons and their damage types
        metadata = image_json["metadata"]
        image_height, image_width = metadata["height"], metadata["width"]
        coords = image_json["features"]["xy"]
        wkt_polygons = []

        for coord in coords:
            damage = self._get_damage_type(coord["properties"])  # Get damage dataset_type
            wkt_polygons.append((damage, coord["wkt"]))

        # Convert WKT to Shapely polygons
        polygons = []
        for damage, swkt in wkt_polygons:
            polygons.append((damage, wkt.loads(swkt)))

        # Initialize the mask
        mask = Image.new(
            "L",
            (image_height, image_width),
            0,
        )  # 'L' mode for grayscale, initialized to 0
        draw = ImageDraw.Draw(mask)

        # Define damage classes (used in "damage" mode)
        damage_classes = {
            "no-damage": 1,
            "minor-damage": 2,
            "major-damage": 3,
            "destroyed": 4,
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

    def _annotate_img(self, draw: ImageDraw, coords: dict) -> None:
        wkt_polygons = []
        for coord in coords:
            damage = self._get_damage_type(coord["properties"])
            wkt_polygons.append((damage, coord["wkt"]))

        polygons = []

        for damage, swkt in wkt_polygons:
            polygons.append((damage, wkt.loads(swkt)))

        for damage, polygon in polygons:
            x, y = polygon.exterior.coords.xy
            coords = list(zip(x, y))
            draw.polygon(coords, damage_dict[damage])

        del draw

    def display_img(self, idx: int, time: str = "pre", annotated: bool = True) -> Image.Image:
        """Display an image with optional annotation overlays.

        Args:
            idx (int): Index of the image to display.
            time (str, optional): "pre" or "post" to select the temporal version of the image (default is "pre").
            annotated (bool, optional): If True, overlays annotations on the image (default is True).

        Returns:
            out (PIL.Image.Image) : The image with or without annotations.
        """
        json_path = self.list_labels[idx]
        if time == "pre":
            json_path = json_path.replace("post", "pre")

        img_path = json_path.replace("labels", "images").replace("json", "png")

        with Path.open(json_path) as json_file:
            image_json = json.load(json_file)

        # Read Image
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img, "RGBA")

        if annotated:
            self._annotate_img(draw=draw, coords=image_json["features"]["xy"])
        return img

    def extract_metadata(self, idx: int, time: str = "pre"):
        """Extract metadata.

        Args:
            idx (int): image index in Dataset
            time (str, optional): period to extract metadata. Defaults to "pre".

        Returns:
            dict: Extracted corresponding metadata
        """
        label_path = self.list_labels[idx]

        if time == "pre":
            label_path = label_path.replace("post", "pre")
        elif time == "post":
            label_path = label_path.replace("pre", "post")

        # Load JSON file
        with Path.open(label_path) as json_file:
            image_json = json.load(json_file)

        return image_json["metadata"]

    def display_data(
        self, list_ids: list[int], time: str = "pre", annotated: bool = True, cols: int = 3
    ) -> None:
        """Display a list of images with or without annotations.

        Args:
            list_ids (list[int]): List of indices of images to display.
            time (str): Choose Timestamp ("pre" or "post").
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
            img = self.display_img(idx, time=time, annotated=annotated)
            metadata = self.extract_metadata(idx)
            disaster_name, disaster_type = (
                metadata["disaster"],
                metadata["disaster_type"],
            )

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

    def plot_image(self, idx: int, save: bool = False):
        """Plot a sample with its pre, post disaster images and its mask labels.

        Args:
            idx (_type_): _description_
            save (bool, optional): _description_. Defaults to False.
        """
        # read images
        img_a = self.display_img(idx, time="pre", annotated=False)
        img_b = self.display_img(idx, time="post", annotated=False)
        img_c = self.display_img(idx, time="pre", annotated=True)
        img_d = self.display_img(idx, time="post", annotated=True)

        # display images
        fig, ax = plt.subplots(2, 2)
        fig.set_size_inches(30, 30)
        title_font_size = 24
        ax[0][0].imshow(img_a)
        ax[0][0].set_title("Pre Disaster Image (Not Annotated)", fontsize=title_font_size)
        ax[0][1].imshow(img_b)
        ax[0][1].set_title("Post Disaster Image (Not Annotated)", fontsize=title_font_size)
        ax[1][0].imshow(img_c)
        ax[1][0].set_title("Pre Disaster Image (Annotated)", fontsize=title_font_size)
        ax[1][1].imshow(img_d)
        ax[1][1].set_title("Post Disaster Image (Annotated)", fontsize=title_font_size)
        if save:
            plt.savefig("split_image.png", dpi=100)
        plt.show()


# Change Detection Datasets #
class xDB_Siamese_Dataset(ChangeDetectionDataset):
    """xDB Siamese Damage Building Dataset. From xDB Dataset, a semantic segmentation task dataset for bi temporal damage building detections."""

    MEAN: ClassVar[list[float]] = [0.349, 0.354, 0.268]
    STD: ClassVar[list[float]] = [0.114, 0.102, 0.094]

    def __init__(
        self,
        origin_dir: str,
        mode: Literal[
            "building", "full_damage", "simple_damage", "change_detection"
        ] = "change_detection",
        transform: Augmentation_pipeline = None,
        dataset_type: str = "train",
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        n_classes: int | None = None,
        **kwargs,
    ) -> None:
        """Initialize the xDB_Siamese_Dataset for change detection tasks.

        Args:
            origin_dir (str): Directory containing the dataset.
            mode (Literal["building", "full_damage", "simple_damage", "change_detection"], optional):
                Determines the type of mask to load. Defaults to "change_detection".
            transform (Augmentation_pipeline, optional): Optional transformation to apply on images and masks.
            dataset_type (str, optional): "train", "val", or "test" to determine the dataset split.
            val_ratio (float, optional): Fraction of the dataset to use for validation.
            test_ratio (float, optional): Fraction of the dataset to use for testing.
            seed (int, optional): Random seed for reproducibility.
            n_classes (int, optional): Number of classes for segmentation.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            origin_dir, dataset_type=dataset_type, transform=transform, n_classes=n_classes
        )
        np.random.default_rng(seed=seed)
        self.label_dir = Path(origin_dir) / "labels"
        self.mode = mode
        self.list_labels = [str(x) for x in self.label_dir.rglob(pattern="*post_*.json")]
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.n_classes = (
            len(set(self._get_classes(self.mode).values())) + 1
        )  # +1 for background class

        self._split()  # Perform a split for train, val, and test datasets

    def _split(self) -> None:
        self.list_labels = train_val_test_split(
            self.list_labels,
            val_size=self.val_ratio,
            test_size=self.test_ratio,
            dataset_type=self.dataset_type,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # noqa: D105
        json_path = self.list_labels[index]

        # Load pre and post images
        pre_image = self._get_image(json_path, time="pre")
        post_image = self._get_image(json_path, time="post")

        # Load masks
        pre_mask = self._get_mask(json_path, time="pre", mode=self.mode)
        post_mask = self._get_mask(json_path, time="post", mode=self.mode)

        # Apply transformations if defined
        if self.transform:
            pre_image = np.float32(np.array(pre_image)) / 255.0
            post_image = np.float32(np.array(post_image)) / 255.0

            transformed = self.transform(
                image=pre_image,
                mask=pre_mask,
                post_image=post_image,
                post_mask=post_mask,
            )
            pre_image = transformed["image"].float()
            pre_mask = transformed["mask"].long()
            post_image = transformed["post_image"].float()
            post_mask = transformed["post_mask"].long()

        else:
            pre_image = torch.from_numpy(np.array(pre_image)).permute(2, 0, 1).float() / 255.0
            pre_mask = torch.from_numpy(pre_mask).long()
            post_image = torch.from_numpy(np.array(post_image)).permute(2, 0, 1).float() / 255.0
            post_mask = torch.from_numpy(post_mask).long()

        return {
            "pre_image": pre_image,
            "post_image": post_image,
            "pre_mask": pre_mask,
            "post_mask": post_mask,
        }

    def __len__(self) -> int:  # noqa: D105
        return len(self.list_labels)

    def _get_image(self, json_path: str, time: Literal["pre", "post"] = "pre") -> Image:
        if time == "pre":
            json_path = json_path.replace("post", "pre")

        img_path = json_path.replace("labels", "images").replace(".json", ".png")
        return Image.open(img_path)

    def _get_damage_type(self, properties: dict) -> str:
        if "subtype" in properties:
            return properties["subtype"]
        return "no-damage"

    def _get_mask(
        self,
        label_path: str,
        time: Literal["pre", "post"] = "post",
        mode: Literal["building", "full_damage", "simple_damage", "change_detection"] = "building",
    ) -> np.ndarray:
        if time == "pre":
            label_path = label_path.replace("post", "pre")

        with Path.open(label_path) as json_file:
            image_json = json.load(json_file)

        metadata = image_json["metadata"]
        image_height, image_width = metadata["height"], metadata["width"]
        coords = image_json["features"]["xy"]
        wkt_polygons = []

        for coord in coords:
            damage = self._get_damage_type(coord["properties"])
            wkt_polygons.append((damage, coord["wkt"]))

        polygons = []
        for damage, swkt in wkt_polygons:
            polygons.append((damage, wkt.loads(swkt)))

        mask = Image.new("L", (image_height, image_width), 0)
        draw = ImageDraw.Draw(mask)

        damage_classes = self._get_classes(mode)

        for damage, polygon in polygons:
            if polygon.is_valid:
                x, y = polygon.exterior.coords.xy
                coords = list(zip(x, y))
                if mode is not None:
                    damage_value = damage_classes.get(damage, 0)
                    draw.polygon(coords, fill=damage_value)

        return np.array(mask)

    def _get_classes(
        self, mode: Literal["building", "full_damage", "simple_damage", "change_detection"]
    ) -> dict:
        if mode == "full_damage":
            # every damage classes are kept
            return {
                "no-damage": 1,
                "minor-damage": 2,
                "major-damage": 3,
                "destroyed": 4,
            }
        if mode == "simple_damage":
            # gather classes into 2 damage level classification
            return {
                "no-damage": 1,
                "minor-damage": 1,
                "major-damage": 2,
                "destroyed": 2,
            }
        if mode == "change_detection":
            # only consider major damages as damages leading to changes for a binary classication task
            return {
                "no-damage": 0,
                "minor-damage": 0,
                "major-damage": 1,
                "destroyed": 1,
            }
        if mode == "building":
            # gather every classes into a building class for a building footprint detections
            return {
                "no-damage": 1,
                "minor-damage": 1,
                "major-damage": 1,
                "destroyed": 1,
            }
        return None

    def display_data(
        self,
        list_ids: list[int],
        display_mode: Literal["annotated", "plain"] = "annotated",
        cols: int = 2,
    ) -> None:
        """Display a list of images with or without annotations.

        Args:
            list_ids (list[int]): List of indices of images to display.
            display_mode (Literal["annotated", "plain"]): "plain" for no annotations, "annotated" to overlay annotations.
            cols (int): Number of columns in the grid for displaying images.

        """
        if cols <= 0:
            raise ValueError("Number of columns (cols) must be a positive integer.")

        num_images = len(list_ids)
        if num_images == 0:
            logging.info("No images to display.")
            return

        rows = num_images
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

        damage_colors = {
            0: (0, 0, 0),
            1: (0, 1, 0),
            2: (1, 1, 0),
            3: (1, 0.5, 0),
            4: (1, 0, 0),
        }
        cmap = plt.cm.colors.ListedColormap(
            [damage_colors[val] for val in sorted(damage_colors.keys())],
        )

        for i, idx in enumerate(list_ids):
            if idx < 0 or idx >= len(self):
                logging.info("Index %d is out of bounds. Skipping...", idx)
                continue

            data = self.__getitem__(idx)
            pre_image, pre_mask = (
                data["pre_image"].permute(1, 2, 0).numpy(),
                data["pre_mask"].numpy(),
            )
            post_image, post_mask = (
                data["post_image"].permute(1, 2, 0).numpy(),
                data["post_mask"].numpy(),
            )

            # Display pre-disaster image
            axes[2 * i].imshow(pre_image)
            if display_mode == "annotated":
                axes[2 * i].imshow(pre_mask, alpha=0.5, cmap=cmap)
            axes[2 * i].set_title(f"Image {idx}: Pre-disaster")
            axes[2 * i].axis("off")

            # Display post-disaster image
            axes[2 * i + 1].imshow(post_image)
            if display_mode == "annotated":
                axes[2 * i + 1].imshow(post_mask, alpha=0.5, cmap=cmap)
            axes[2 * i + 1].set_title(f"Image {idx}: Post-disaster")
            axes[2 * i + 1].axis("off")

        for j in range(2 * num_images, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()


class xDB_Instance_Building(InstanceSegmentationDataset):
    """Dataset for instance segmentation tasks with building or damage masks.

    This dataset supports both "instance" and "semantic" segmentation tasks for buildings or damage classes.
    It loads images and their corresponding masks, performs train/val/test splits, and provides data in a format
    suitable for PyTorch models.

    Args:
        origin_dir (str): Directory containing the dataset.
        mode (str): "building" or "damage" to determine the type of masks to load.
        time (str): Temporal filter for the dataset (e.g., "pre" or "post").
        transform (callable | None): Optional transformation to apply on images and targets.
        dataset_type (str): "train", "val", or "test" to determine the dataset split.
        val_ratio (float): Fraction of the dataset to use for validation.
        test_ratio (float): Fraction of the dataset to use for testing.
        task (str): "instance" or "semantic" segmentation.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple[tv_tensors.Image, dict[str, torch.Tensor]]: Image and target dictionary for segmentation.

    """

    MEAN: ClassVar[list[float]] = [0.349, 0.354, 0.268]
    STD: ClassVar[list[float]] = [0.114, 0.102, 0.094]

    def __init__(
        self,
        origin_dir: str,
        mode: Literal["building", "damage"],
        time: Literal["pre", "post"],
        transform: callable | None = None,
        dataset_type: Literal["train", "val", "test"] = "train",
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        task: Literal["instance", "semantic"] = "instance",
        seed: int = 42,
    ) -> None:
        """Initialize the xDB_Instance_Building dataset for instance or semantic segmentation.

        Args:
            origin_dir (str): Directory containing the dataset.
            mode (Literal["building", "damage"]): Type of masks to load.
            time (Literal["pre", "post"]): Temporal filter for the dataset.
            transform (callable | None): Optional transformation to apply on images and targets.
            dataset_type (Literal["train", "val", "test"]): Dataset split type.
            val_ratio (float): Fraction of the dataset to use for validation.
            test_ratio (float): Fraction of the dataset to use for testing.
            task (Literal["instance", "semantic"]): Segmentation task type.
            seed (int): Random seed for reproducibility.
        """
        self.dataset_type = dataset_type
        self.mode = mode
        self.time = time
        self.transform = transform
        self.label_dir = Path(origin_dir) / "labels"
        self.list_labels = [str(x) for x in self.label_dir.rglob(f"*{self.time}_*.json")]

        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        np.random.default_rng(seed)  # Ensure reproducibility
        self._split()  # Perform train/val/test split
        self.task = task

    def _split(self) -> None:
        """Split the dataset into train, validation, and test sets."""
        self.list_labels = train_val_test_split(
            self.list_labels,
            val_size=self.val_ratio,
            test_size=self.test_ratio,
            dataset_type=self.dataset_type,
        )

    def __getitem__(self, index: int) -> tuple[tv_tensors.Image, dict[str, torch.Tensor]]:
        """Retrieve an image and its corresponding mask for instance segmentation.

        Args:
            index (int): Index of the sample.

        Returns:
            out (tuple[Image, dict[str, torch.Tensor]]):
                A tuple containing:
                    - img (tv_tensors.Image): Transformed image as a torch tensor.
                    - target (dict[str, torch.Tensor]): A dictionary with the following keys:
                        - "boxes": Bounding boxes for each instance.
                        - "masks": Instance masks.
                        - "labels": Labels for each instance (1 for all in this case).
                        - "image_id": Identifier for the image.
                        - "area": Areas of the bounding boxes.
        """
        # Get the JSON path for the given index
        json_path = self.list_labels[index]

        # Load the image
        image = self._get_image(json_path)
        h, w = image.size[-2:]

        # Load the mask
        binary_mask = self._get_mask(json_path, mode=self.mode)

        # From binary mask to instance mask
        num_objs, mask = cv2.connectedComponents(binary_mask)

        # instances are encoded as different colors
        obj_ids = np.unique(mask)

        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        """
        # Exclude the background (first ID is background)
        masks = torch.from_numpy(mask).unsqueeze(0) == torch.arange(1, num_objs, dtype=torch.uint8).view(-1, 1, 1)
        masks = masks.any(dim=0)
        """
        if len(obj_ids) > 0:
            # split the color-encoded mask into a set
            # of binary masks
            masks = torch.from_numpy(mask == obj_ids[:, None, None]).to(dtype=torch.uint8)
            # Get bounding boxes for each mask
            boxes = masks_to_boxes(masks)
            # Create labels (1 for all instances)
            labels = torch.ones((num_objs,), dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, h, w), dtype=torch.uint8)

        # Wrap image into torchvision tv_tensors
        img = tv_tensors.Image(image)

        # Create the target dictionary
        if self.task == "instance":
            target = {
                "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(h, w)),
                "masks": tv_tensors.Mask(masks),
                "labels": labels,
                "image_id": index,
            }
        elif self.task == "semantic":
            target = {
                "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(h, w)),
                "masks": tv_tensors.Mask(binary_mask),
                "labels": labels,
                "image_id": index,
            }
        target = self._filter_invalid_boxes(target, width=w, height=h)

        # Apply transformations if provided
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.list_labels)

    def _filter_invalid_boxes(self, target: dict, width: int = 512, height: int = 512) -> dict:
        if self.task == "instance":
            boxes = target["boxes"]
            masks = target["masks"]
            labels = target["labels"]

            # Keep boxes where x_max > x_min and y_max > y_min
            valid_boxes, valid_masks, valid_labels = [], [], []

            for box, mask, label in zip(boxes, masks, labels):
                if box[2] > box[0] and box[3] > box[1]:  # Ensure valid dimensions
                    valid_boxes.append(box.tolist())  # Convert to list explicitly
                    valid_masks.append(mask.tolist() if isinstance(mask, torch.Tensor) else mask)
                    valid_labels.append(label)

            if len(valid_boxes) == 0:
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target["labels"] = torch.zeros((0,), dtype=torch.int64)
                target["masks"] = torch.zeros((0, height, width), dtype=torch.uint8)
            else:
                target["boxes"] = torch.tensor(valid_boxes, dtype=torch.float32)
                target["masks"] = torch.tensor(
                    valid_masks,
                    dtype=torch.float32,
                )  # Ensure masks have appropriate dtype
                target["labels"] = torch.tensor(
                    valid_labels,
                    dtype=torch.long,
                )  # Labels are typically integers

        return target

    def _get_image(self, json_path: str) -> Image:
        """Open image file based on the json_path label file.

        Image path example: ../images/joplin-tornado_00000000_post_disaster.png
        Label path example: ../labels/joplin-tornado_00000000_post_disaster.json

        Args:
        json_path : str
            Path to the label JSON file.

        Returns:
            PIL.Image.Image: The corresponding image.

        Raises:
        ------
            FileNotFoundError: If the image file is not found.

        """
        # Adjust path for "pre" time if needed
        if self.time == "pre":
            json_path = json_path.replace("post", "pre")

        # Replace 'labels' with 'images' and '.json' with '.png'
        img_path = json_path.replace("labels", "images").replace(".json", ".png")
        # Open and return the image
        return Image.open(img_path)

    def _get_damage_type(self, properties: dict) -> str:
        if "subtype" in properties:
            return properties["subtype"]
        return "no-damage"

    def _get_mask(self, label_path: str, mode: str = "building") -> np.ndarray:
        """Build a mask from a label file.

        Args:
            label_path (str): Path to the JSON label file.
            mode (str): "building" for binary mask, "damage" for multiclass mask.

        Returns:
            np.ndarray: Mask as a numpy array.

        """
        # Load JSON file
        with Path.open(label_path) as json_file:
            image_json = json.load(json_file)

        # Extract building polygons and their damage types
        metadata = image_json["metadata"]
        image_height, image_width = metadata["height"], metadata["width"]
        coords = image_json["features"]["xy"]
        wkt_polygons = []

        for coord in coords:
            damage = self._get_damage_type(coord["properties"])  # Get damage dataset_type
            wkt_polygons.append((damage, coord["wkt"]))

        # Convert WKT to Shapely polygons
        polygons = []
        for damage, swkt in wkt_polygons:
            polygons.append((damage, wkt.loads(swkt)))

        # Initialize the mask
        mask = Image.new(
            "L",
            (image_height, image_width),
            0,
        )  # 'L' mode for grayscale, initialized to 0
        draw = ImageDraw.Draw(mask)

        # Define damage classes (used in "damage" mode)
        damage_classes = {
            "no-damage": 1,
            "minor-damage": 2,
            "major-damage": 3,
            "destroyed": 4,
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

    def display_sample(self, idx: int, threshold: float = 0.5) -> None:
        """Display an image with its segmentation masks, bounding boxes, and labels.

        Args:
        idx : int
            Index of the sample to display.
        threshold : float, optional
            Threshold value to binarize the masks for display (default is 0.5).

        """
        img, target = self.__getitem__(idx)

        # Convert image to PIL for consistent handling
        pil_image = to_pil_image(img)
        tensor_image = transforms.PILToTensor()(pil_image)

        # Extract data from the target dictionary
        masks = target["masks"]  # Segmentation masks as a tensor
        boxes = target["boxes"]  # Bounding boxes as a tensor
        labels = target["labels"].tolist()  # Convert labels to a list for easier manipulation

        # Prepare label names (if class_names are available)
        if hasattr(self, "class_names"):
            label_names = [self.class_names[label] for label in labels]
        else:
            label_names = [str(label) for label in labels]

        # Define colors for each label (assign one color per unique class)
        unique_labels = list(set(labels))
        int_colors = [(i * 30 % 256, i * 60 % 256, i * 90 % 256) for i in unique_labels]
        label_to_color = dict(zip(unique_labels, int_colors))

        colors = [label_to_color[label] for label in labels]

        # Annotate the image with segmentation masks
        annotated_tensor = draw_segmentation_masks(
            image=tensor_image,
            masks=(masks > threshold),  # Ensure masks are binary
            alpha=0.3,
            colors=colors,
        )

        # Annotate the image with bounding boxes and labels
        annotated_tensor = draw_bounding_boxes(
            image=annotated_tensor,
            boxes=boxes,
            labels=label_names,
            colors=colors,
            font_size=12,
        )

        # Convert annotated tensor back to PIL image for display
        annotated_image = to_pil_image(annotated_tensor)

        # Display the image
        plt.figure(figsize=(10, 10))
        plt.imshow(annotated_image)
        plt.axis("off")
        plt.title(f"Sample {idx}    {len(labels)} Objects")
        plt.show()

    """def display_sample(self, idx: int, show_labels: bool = True):
        Displays an image with its bounding boxes, masks, and optionally labels.

        Args::
            idx (int): Index of the sample to display.
            show_labels (bool): Whether to display labels with bounding boxes.
        img, target = self.__getitem__(idx)

        # Convert tv_tensors.Image to PIL image
        pil_image = to_pil_image(img)

        # Extract data from the target dictionary
        boxes = target["boxes"].numpy()  # Convert bounding boxes to numpy
        masks = target["masks"].numpy()  # Convert masks to numpy
        labels = target["labels"].numpy()  # Convert labels to numpy

        # Set up the figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(pil_image)
        ax.axis("off")

        # Overlay masks
        alpha = 0.3 / masks.shape[0]

        for mask_idx in range(masks.shape[0]):
            mask = masks[mask_idx]
            ax.imshow(mask, alpha=alpha, cmap='jet')

        # Overlay bounding boxes and labels
        for box, label in zip(boxes, labels):
            x_min, y_min, x_max, y_max = box
            width = x_max    x_min
            height = y_max    y_min
            rect = Rectangle((x_min, y_min), width, height, edgecolor='lime', facecolor='none', linewidth=1)
            ax.add_patch(rect)

            if show_labels:
                ax.text(
                    x_min, y_min    5,  # Position text slightly above the box
                    str(label),  # Convert label to string for display
                    fontsize=5,
                    color='white',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
                )

        # Title and display
        ax.set_title(f"Sample {idx}    {len(boxes)} Objects")
        plt.show()"""
