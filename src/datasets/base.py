from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import torch
from torch.utils.data import Dataset

from src.augmentation import Augmentation_pipeline


class BuildingDataset(Dataset):
    """General Class for Building-Related PyTorch Dataset.

    Attributes:
    ----------
        origin_dir (Path): Path to the dataset folder.
        dataset_type (str): Dataset type ('train', 'val', 'test', 'infer').
        transform (Augmentation_pipeline | None): Custom augmentation pipeline.
        n_classes (int): Number of different classes.

    """

    MEAN = None
    STD = None

    def __init__(
        self,
        origin_dir: str | Path,
        dataset_type: Literal["train", "val", "test", "infer"],
        transform: Augmentation_pipeline | None = None,
        n_classes: int | None = None,
    ) -> None:
        """Initialize the BuildingDataset class.

        Args:
        ----
            origin_dir (str | Path): Path to the dataset folder.
            dataset_type (str): Dataset type ('train', 'val', 'test', 'infer').
            transform (Augmentation_pipeline | None): Custom augmentation pipeline.
            n_classes (int | None): Number of different classes.

        """
        self.origin_dir = Path(origin_dir)
        self.dataset_type = dataset_type
        self.transform = transform
        self.n_classes = n_classes

    def __repr__(self) -> str:
        """Return a string representation of the dataset.

        Returns:
        -------
            str: String representation of the dataset.

        """
        return (
            f"Dataset class: {self.__class__.__name__} / "
            f"Type: {self.dataset_type} / "
            f"Number of samples: {len(self)} / "
            f"Number of classes: {self.n_classes}"
        )

    def display_data(self, list_indices: list[int]) -> None:
        """Display data for the given indices.

        Args:
        ----
            list_indices (list[int]): List of indices to display.

        """
        raise NotImplementedError("Subclasses must implement this method.")

    def display_img(self, idx: int, **kwargs: object) -> None:
        """Display an image for the given index.

        Args:
        ----
            idx (int): Index of the image to display.
            **kwargs: Additional keyword arguments for display customization.

        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Retrieve an item from the dataset.

        Args:
        ----
            index (int): Index of the item to retrieve.

        Returns:
        -------
            dict[str, torch.Tensor]: Dictionary containing the data.

        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
        -------
            int: Number of samples in the dataset.

        """
        raise NotImplementedError("Subclasses must implement this method.")


class CloudDataset(Dataset):
    """General Class for Cloud-Related PyTorch Dataset.

    Attributes:
    ----------
        bands (list[str]): List of bands used in the dataset.

    """

    def __init__(self, bands: list[str]) -> None:
        """Initialize the CloudDataset class.

        Args:
        ----
            bands (list[str]): List of bands used in the dataset.

        """
        super().__init__()
        self.bands = bands


class SegmentationDataset(BuildingDataset):
    """Dataset class for segmentation tasks."""

    def __init__(
        self,
        origin_dir: str | Path,
        dataset_type: str | None = None,
        transform: Augmentation_pipeline | None = None,
        n_classes: int | None = None,
    ) -> None:
        """Initialize the SegmentationDataset class.

        Args:
            origin_dir (str | Path): Path to the dataset folder.
            dataset_type (str | None): Dataset type ('train', 'val', 'test', 'infer').
            transform (Augmentation_pipeline | None): Custom augmentation pipeline.
            n_classes (int | None): Number of different classes.

        """
        super().__init__(origin_dir, dataset_type, transform, n_classes)


class ChangeDetectionDataset(BuildingDataset):
    """Dataset class for change detection tasks."""

    def __init__(
        self,
        origin_dir: str | Path,
        dataset_type: str | None = None,
        transform: Augmentation_pipeline | None = None,
        n_classes: int | None = None,
    ) -> None:
        """Initialize the ChangeDetectionDataset class.

        Args:
            origin_dir (str | Path): Path to the dataset folder.
            dataset_type (str | None): Dataset type ('train', 'val', 'test', 'infer').
            transform (Augmentation_pipeline | None): Custom augmentation pipeline.
            n_classes (int | None): Number of different classes.

        """
        super().__init__(origin_dir, dataset_type, transform, n_classes)


class InstanceSegmentationDataset(BuildingDataset):
    """Dataset class for instance segmentation tasks."""

    def __init__(
        self,
        origin_dir: str | Path,
        dataset_type: str | None = None,
        transform: Augmentation_pipeline | None = None,
        n_classes: int | None = None,
    ) -> None:
        """Initialize the InstanceSegmentationDataset class.

        Args:
            origin_dir (str | Path): Path to the dataset folder.
            dataset_type (str | None): Dataset type ('train', 'val', 'test', 'infer').
            transform (Augmentation_pipeline | None): Custom augmentation pipeline.
            n_classes (int | None): Number of different classes.

        """
        super().__init__(origin_dir, dataset_type, transform, n_classes)
