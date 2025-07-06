from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.utils import read_tiff_pillow

# Utility functions for Semantic Segmentation Conformal Prediction
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


def one_hot_encoding_of_2d_int_mask(mask: torch.Tensor, n_labels: int) -> torch.Tensor:
    """Convert a 2D integer mask into a one-hot encoded tensor.

    Args:
    ----
        mask (torch.Tensor): Semantic segmentation mask with shape (H, W).
        n_labels (int): Total number of possible labels.

    Returns:
    -------
        torch.Tensor: One-hot encoded mask with shape (n_labels, H, W).

    """
    mask_one_hot = torch.nn.functional.one_hot(mask, num_classes=n_labels)
    return torch.movedim(mask_one_hot, 2, 0)


def one_hot_encoding_of_gt(gt: torch.Tensor, n_labels: int) -> torch.Tensor:
    """One-hot encoding wrapper of ground truth masks.

    Args:
    ----
        gt (torch.Tensor): Ground truth mask.
        n_labels (int): Total number of possible labels.

    Returns:
    -------
        torch.Tensor: One-hot encoded ground truth mask.

    """
    return one_hot_encoding_of_2d_int_mask(gt, n_labels)


def check_risk_upper_bound(alpha_risk: float, b_loss_upper_bound: float, n_calibs: int) -> None:
    """Validate that the number of calibration points is sufficient.

    Args:
    ----
        alpha_risk (float): Risk level.
        b_loss_upper_bound (float): Upper bound of the loss.
        n_calibs (int): Number of calibration points.

    Raises:
    ------
        ValueError: If the number of calibration points is insufficient.

    """
    min_calib = np.floor((b_loss_upper_bound - alpha_risk) / alpha_risk).astype(int) + 1
    if n_calibs < min_calib:
        msg = f"Insufficient calibration points: {n_calibs}. Minimum required: {min_calib}."
        raise ValueError(
            msg,
        )


def compute_risk_bound(
    risk_level: float,
    risk_upper_bound_b: float,
    num_calib_samples: int,
) -> float:
    """Compute the risk bound based on calibration samples.

    Args:
    ----
        risk_level (float): Desired risk level.
        risk_upper_bound_b (float): Upper bound of the risk.
        num_calib_samples (int): Number of calibration samples.

    Returns:
    -------
        float: Computed risk bound.

    """
    check_risk_upper_bound(risk_level, risk_upper_bound_b, num_calib_samples)
    return risk_level - (risk_upper_bound_b - risk_level) / num_calib_samples


def compute_activable_pixels(one_hot_semantic_mask: torch.Tensor, n_labs: int) -> int:
    """Compute the number of activable pixels in the semantic mask.

    Args:
    ----
        one_hot_semantic_mask (torch.Tensor): One-hot encoded ground truth mask.
        n_labs (int): Number of classes in the dataset.

    Returns:
    -------
        int: Number of activable pixels.

    """
    return one_hot_semantic_mask.shape[1] * one_hot_semantic_mask.shape[2]
    # All pixels are activable since there are no ignored pixels.


def split_dataset_idxs(len_dataset: int, n_calib: int, random_seed: int | None = None):
    """Split dataset indices into calibration and test sets.

    Args:
    ----
        len_dataset (int): Total number of samples in the dataset.
        n_calib (int): Number of calibration samples.
        random_seed (int | None): Seed for random shuffling.

    Returns:
    -------
        tuple[list[int], list[int]]: Calibration and test indices.

    Raises:
    ------
        ValueError: If n_calib exceeds the dataset size.

    """
    if n_calib > len_dataset:
        msg = f"n_calib [{n_calib}] must be less than dataset size [{len_dataset}]"
        raise ValueError(msg)

    idxs = list(range(len_dataset))

    if random_seed:
        random.seed(random_seed)

    random.shuffle(idxs)

    cal_idx = idxs[:n_calib]
    test_idx = idxs[n_calib:]

    if len(set(cal_idx).intersection(test_idx)) == 0:
        raise ValueError("Calibration and test sets are not disjoint.")

    return cal_idx, test_idx


def is_semantic_mask_in_multimask(
    one_hot_semantic_mask: torch.Tensor,
    one_hot_multimask: torch.Tensor,
) -> torch.Tensor:
    """Check if a semantic mask is covered by a multimask.

    Args:
    ----
        one_hot_semantic_mask (torch.Tensor): One-hot encoded semantic mask.
        one_hot_multimask (torch.Tensor): One-hot encoded multimask.

    Returns:
    -------
        torch.Tensor: Coverage tensor indicating covered pixels.

    """
    return torch.mul(one_hot_semantic_mask, one_hot_multimask)


def read_image_into_pytorch(image_path: str) -> torch.Tensor:
    """Read an image file into a PyTorch tensor.

    Args:
    ----
        image_path (str): Path to the image file.

    Returns:
    -------
        torch.Tensor: Image tensor with shape (1, C, H, W).

    """
    image_path = Path(image_path)
    if image_path.suffix == ".tif":
        image = read_tiff_pillow(image_path)
    else:
        image = Image.open(image_path).convert("RGB")

    image = np.array(image).astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)  # Convert to (C, H, W)
    return torch.tensor([image], dtype=torch.float32)


def recode_extra_classes(
    semantic_map: torch.Tensor,
    good_labels: Sequence[int],
    extra_labels: Sequence[int | None],
    *,
    force_recode: bool = False,
) -> torch.Tensor:
    """Recodes extra labels in a semantic map to new sequential labels.

    Args:
    ----
        semantic_map (torch.Tensor): Input semantic map with shape (H, W).
        good_labels (Sequence[int]): List of valid labels.
        extra_labels (Sequence[int | None]): List of extra labels to recode.
        force_recode (bool): Force recoding even if extra labels overlap with good labels.

    Returns:
    -------
        torch.Tensor: Recoded semantic map.

    Raises:
    ------
        ValueError: If input types or dimensions are invalid.

    """
    if not isinstance(semantic_map, torch.Tensor) or semantic_map.ndim != 2:
        raise ValueError("semantic_map must be a 2D torch tensor.")

    if not isinstance(good_labels, list) or not isinstance(extra_labels, list):
        raise TypeError("good_labels and extra_labels must be lists.")

    intersection = set(good_labels).intersection(extra_labels)
    if intersection and not force_recode:
        msg = f"Extra labels overlap with good labels: {intersection}"
        raise ValueError(msg)

    extra_labels = sorted(extra_labels)
    max_good_label = max(good_labels)
    recoded_semantic_map = torch.clone(semantic_map)

    for i, extra_label in enumerate(extra_labels):
        new_label = max_good_label + i + 1
        recoded_semantic_map[recoded_semantic_map == extra_label] = new_label

    return recoded_semantic_map


class TrivialDataset(Dataset):
    """A trivial dataset for testing purposes. Outputs random images and masks."""

    def __init__(self, num_samples: int, image_size: tuple[int, int], n_classes: int) -> None:  # noqa: D107
        self.num_samples = num_samples
        self.image_size = image_size
        self.n_classes = n_classes
        self.name = "TrivialDataset"

    def __len__(self):  # noqa: D105
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str : torch.Tensor]:  # noqa: D105
        image = torch.rand(3, *self.image_size)  # Random RGB image
        mask = torch.randint(0, self.n_classes, self.image_size)  # Random mask
        return {"image": image, "mask": mask}


# Assets for testing COSE
class TrivialModel(torch.nn.Module):
    """A trivial model for testing purposes. Outputs random logits."""

    def __init__(self, num_classes: int) -> None:
        """Initialize a Trivial Random model.

        Args:
            num_classes (int): number of classes.

        """
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Method.

        Args:
            x (torch.Tensor): A torch tensor

        Returns:
            torch.Tensor: A random logit tensor with num_classes classes.

        """
        batch_size, _, height, width = x.shape
        return torch.rand(batch_size, self.num_classes, height, width)  # Random logits


# Utils functions
def draw_random_color_circle(
    image_size: tuple[int] = (224, 224),
    radius: int = 50,
    center: tuple[int] | None = None,
    num_classes: int = 3,
):
    """Draws a randomly colored circle with a corresponding class label mask.

    Args:
    ----
        image_size (tuple): Image size (H, W).
        radius (int): Radius of the circle.
        center (tuple or None): Circle center. If None, placed at center.
        num_classes (int): Number of total classes.

    Returns:
    -------
        image_tensor (torch.Tensor): Tensor of shape (3, H, W), normalized to [0, 1].
        label_tensor (torch.Tensor): Tensor of shape (H, W), with values in [0, num_classes - 1].

    """
    if center is None:
        center = (image_size[1] // 2, image_size[0] // 2)

    # Randomly choose a class ID and assign a color
    class_id = rng.randint(0, num_classes)
    color_map = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 128, 128),  # Gray
        (255, 128, 0),  # Orange
    ]
    color = color_map[class_id % len(color_map)]

    # Create black image and draw colored circle
    image = np.zeros((*image_size, 3), dtype=np.uint8)
    label = np.zeros(image_size, dtype=np.uint8)

    cv2.circle(image, center, radius, color, thickness=-1)
    cv2.circle(label, center, radius, class_id, thickness=-1)

    # Convert to torch tensors
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # (3, H, W)
    label_tensor = torch.tensor(label, dtype=torch.long)  # (H, W)

    return image_tensor, label_tensor


class RoundDataset(Dataset):
    """A dataset that generates images with random colored circles and corresponding masks.

    Attributes:
    ----------
    length : int
        Number of samples in the dataset.
    center_list : list[tuple[int, int]]
        List of circle centers for each sample.
    radius_list : list[int]
        List of circle radii for each sample.
    name : str
        Name of the dataset.
    image_label_pairs : list[tuple[torch.Tensor, torch.Tensor]]
        List of (image, label) pairs.
    is_siamese : bool
        Whether to return siamese (pre/post) images.
    n_classes : int
        Number of classes.

    """

    def __init__(  # noqa: D107
        self,
        length: int = 64,
        image_size: int = 224,
        num_classes: int = 3,
        *,
        is_siamese: bool = True,
    ):
        self.length = length
        self.center_list = [
            (rng.randint(0, image_size), rng.randint(0, image_size)) for _ in range(length)
        ]
        self.radius_list = [rng.randint(10, image_size // 2) for _ in range(length)]
        self.name = "RoundDataset"
        self.image_label_pairs = [
            draw_random_color_circle((image_size, image_size), radius, center, num_classes)
            for radius, center in zip(self.radius_list, self.center_list)
        ]
        self.is_siamese = is_siamese
        self.n_classes = num_classes

    def __len__(self) -> int:
        """Return the dataset length."""
        return self.length

    def __getitem__(self, index: int) -> dict:
        """Return a sample with an image and its mask."""
        image, label = self.image_label_pairs[index]
        if self.is_siamese:
            return {"pre_image": image, "post_image": image, "post_mask": label}
        return {"image": image, "mask": label}
