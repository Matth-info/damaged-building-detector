import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.v2 import functional as F
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision import transforms

from PIL import Image, ImageDraw
from shapely import wkt
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt



### Datasets used for Instance Segmentation Task (not Semantic Segmentation Task)

class xDB_Instance_Building(Dataset):
    def __init__(self, 
                 origin_dir: str,
                 mode: str = "building",
                 time: str = "pre",
                 transform: Optional[callable] = None,
                 type: str = "train",
                 val_ratio: float = 0.1, 
                 test_ratio: float = 0.1,
                 seed: int = 42):
        """
        Dataset for instance segmentation tasks with building or damage masks.

        Parameters:
        - origin_dir: Directory containing the dataset.
        - mode: "building" or "damage" to determine the type of masks to load.
        - time: Temporal filter for the dataset (e.g., "pre" or "post").
        - transform: Optional transformation to apply on images and targets.
        - type: "train", "val", or "test" to determine the dataset split.
        - val_ratio: Fraction of the dataset to use for validation.
        - test_ratio: Fraction of the dataset to use for testing.
        - seed: Random seed for reproducibility.
        """
        assert type in ["train", "val", "test"], "Dataset type must be 'train', 'val', or 'test'."
        assert mode in ["building", "damage"], "Mode must be 'building' or 'damage'."

        self.type = type
        self.mode = mode
        self.time = time
        self.transform = transform
        self.label_dir = Path(origin_dir) / "labels"
        self.list_labels = [str(x) for x in self.label_dir.rglob(f"*{self.time}_*.json")]

        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        np.random.seed(seed)  # Ensure reproducibility
        self._split()  # Perform train/val/test split

    def _split(self):
        """Splits the dataset into train, validation, and test sets."""
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

    def __getitem__(self, idx: int) -> Tuple[tv_tensors.Image, Dict[str, torch.Tensor]]:
        """
        Retrieves an image and its corresponding mask for instance segmentation.

        Parameters:
        - index: Index of the sample.

        Returns:
        - img: Transformed image as a torch tensor.
        - target: A dictionary with the following keys:
            - "boxes": Bounding boxes for each instance.
            - "masks": Instance masks.
            - "labels": Labels for each instance (1 for all in this case).
            - "image_id": Identifier for the image.
            - "area": Areas of the bounding boxes.
        """
        # Get the JSON path for the given index
        json_path = self.list_labels[idx]

        # Load the image
        image = self.get_image(json_path)
        H, W = image.size[-2:]

        # Load the mask
        binary_mask = self.get_mask(json_path, mode=self.mode)

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
        if len(obj_ids)> 0:
            # split the color-encoded mask into a set
            # of binary masks
            masks = torch.from_numpy((mask == obj_ids[:, None, None])).to(dtype=torch.uint8)
            # Get bounding boxes for each mask
            boxes = masks_to_boxes(masks)
            # Create labels (1 for all instances)
            labels = torch.ones((num_objs,), dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, H, W), dtype=torch.uint8)

        # Wrap image into torchvision tv_tensors
        img = tv_tensors.Image(image)

        # Create the target dictionary
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(H,W)),
            "masks": tv_tensors.Mask(masks),
            "labels": labels,
            "image_id": idx
        }

        # Apply transformations if provided
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, self._filter_invalid_boxes(target, width=W, height=H)

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.list_labels)
    
    def _filter_invalid_boxes(self, target, width=512, height=512):
        boxes = target['boxes']
        masks = target['masks']
        labels = target['labels']

        # Keep boxes where x_max > x_min and y_max > y_min
        valid_boxes, valid_masks, valid_labels = [], [], []

        for box, mask, label in zip(boxes, masks, labels):
            if box[2] > box[0] and box[3] > box[1]:  # Ensure valid dimensions
                valid_boxes.append(box.tolist())  # Convert to list explicitly
                valid_masks.append(mask.tolist() if isinstance(mask, torch.Tensor) else mask)
                valid_labels.append(label)

        if len(valid_boxes) == 0:
            
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
            target['masks'] = torch.zeros((0, height, width), dtype=torch.uint8)
        
        else:

            target['boxes'] = torch.tensor(valid_boxes, dtype=torch.float32)
            target['masks'] = torch.tensor(valid_masks, dtype=torch.float32)  # Ensure masks have appropriate dtype
            target['labels'] = torch.tensor(valid_labels, dtype=torch.long)  # Labels are typically integers

        return target


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
        
   

    def display_sample(self, idx: int):
        """
        Displays an image with its segmentation masks, bounding boxes, and labels.

        Parameters:
            idx (int): Index of the sample to display.
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
        label_to_color = {label: color for label, color in zip(unique_labels, int_colors)}

        colors = [label_to_color[label] for label in labels]

        # Annotate the image with segmentation masks
        annotated_tensor = draw_segmentation_masks(
            image=tensor_image,
            masks=masks > 0.5,  # Ensure masks are binary
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
        plt.title(f"Sample {idx} - {len(labels)} Objects")
        plt.show()
        

    """def display_sample(self, idx: int, show_labels: bool = True):
        Displays an image with its bounding boxes, masks, and optionally labels.

        Parameters:
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
            width = x_max - x_min
            height = y_max - y_min
            rect = Rectangle((x_min, y_min), width, height, edgecolor='lime', facecolor='none', linewidth=1)
            ax.add_patch(rect)
            
            if show_labels:
                ax.text(
                    x_min, y_min - 5,  # Position text slightly above the box
                    str(label),  # Convert label to string for display
                    fontsize=5,
                    color='white',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
                )

        # Title and display
        ax.set_title(f"Sample {idx} - {len(boxes)} Objects")
        plt.show()"""