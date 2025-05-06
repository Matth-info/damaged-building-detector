from pathlib import Path
from typing import Dict, List, Optional

import albumentations as A
import torch
from torch.utils.data import Dataset


class Building_Dataset(Dataset):
    """General Class for Building Related Pytorch Dataset"""

    MEAN = None
    STD = None

    def __init__(self, origin_dir: str, type: str = None, transform: Optional[A.Compose] = None):
        self.origin_dir = Path(origin_dir)

        assert type in [
            "train",
            "val",
            "test",
            "infer",
        ], "Dataset must be 'train','val', 'test' or 'infer"
        self.type = type
        self.transform = transform

    def __repr__(self):
        return f"Dataset class : {self.__class__.__name__()} / Type : {self.type} / Number of samples : {len(self)}"

    def display_data(self, list_indices: List[int]) -> None:
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement this method")

    def display_img(self, idx, **kwars) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def __getitem__(self, index) -> Dict[str, torch.tensor]:
        raise NotImplementedError("Pytorch Dataset Subclasses must implement this method")

    def __len__(self) -> int:
        raise NotImplementedError("Pytorch Dataset Subclasses must implement this method")


class Cloud_Dataset(Dataset):
    """General Class for Cloud Related Pytorch Dataset"""

    def __init__(self, bands: List[str]):
        super().__init__()
        self.bands = bands


class Segmentation_Dataset(Building_Dataset):
    def __init__(self, origin_dir, type=None, transform=None):
        super().__init__(origin_dir, type, transform)


class Change_Detection_Dataset(Building_Dataset):
    def __init__(self, origin_dir, type=None, transform=None):
        super().__init__(origin_dir, type, transform)


class Instance_Segmentation_Dataset(Building_Dataset):
    def __init__(self, origin_dir, type=None, transform=None):
        super().__init__(origin_dir, type, transform)
