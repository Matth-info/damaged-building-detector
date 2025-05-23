from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.augmentation import Augmentation_pipeline
from src.data.utils import read_tiff_rasterio
from src.datasets.base import Segmentation_Dataset


class Dataset_Inference_Siamese(Segmentation_Dataset):
    """
    Siamese dataset class for inference on pre- and post-disaster GeoTIFF tiles.

    Args:
        origin_dir (str): Root directory containing both image subdirectories.
        pre_disaster_dir (str): Subdirectory name for pre-disaster image tiles.
        post_disaster_dir (Optional[str]): Subdirectory name for post-disaster image tiles. If None, post-disaster tiles are replaced with zeros.
        transform (Optional[Augmentation_pipeline]): Transformation pipeline to apply to both images.
        extension (str): Image file extension (default: 'tif').

    Expected folder structure:
        origin_dir/
            └── pre_disaster_dir/
            └── post_disaster_dir/

    Returns:
        dict:
            "pre_image" (torch.Tensor): Pre-disaster image tensor (C x H x W).
            "post_image" (torch.Tensor): Post-disaster image tensor (C x H x W).
            "filename" (str): Filename without extension.
            "profile" (dict): Rasterio metadata profile.
    """

    def __init__(
        self,
        origin_dir: str,
        pre_disaster_dir: str,
        post_disaster_dir: Optional[str] = None,
        transform: Optional[Augmentation_pipeline] = None,
        extension: str = "tif",
    ):
        super().__init__(origin_dir=origin_dir, type="infer", transform=transform)
        self.extension = extension.lower()
        self.pre_disaster_dir = self.origin_dir / pre_disaster_dir
        self.post_disaster_dir = self.origin_dir / post_disaster_dir if post_disaster_dir else None

        self.image_filenames = self._filter_filenames()

    def _filter_filenames(self):
        pre_files = {f.stem for f in self.pre_disaster_dir.glob(f"*.{self.extension}")}
        if self.post_disaster_dir:
            post_files = {f.stem for f in self.post_disaster_dir.glob(f"*.{self.extension}")}
            return sorted(pre_files & post_files)
        return sorted(pre_files)

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, index: int):
        filename = self.image_filenames[index]

        # Load pre-disaster image
        pre_path = self.pre_disaster_dir / f"{filename}.{self.extension}"
        pre_image, profile, _ = read_tiff_rasterio(pre_path, bands=[1, 2, 3], with_profile=True)
        pre_image = np.array(pre_image).astype(np.float32) / 255.0

        # Load post-disaster image or use zeros if not provided
        if self.post_disaster_dir:
            post_path = self.post_disaster_dir / f"{filename}.{self.extension}"
            post_image, _, _ = read_tiff_rasterio(post_path, bands=[1, 2, 3])
            post_image = np.array(post_image).astype(np.float32) / 255.0
        else:
            post_image = np.zeros_like(pre_image)

        # Apply transformations if provided
        if self.transform:
            transformed = self.transform(image=pre_image, post_image=post_image)
            pre_image = transformed["image"].to(torch.float32)
            post_image = transformed["post_image"].to(torch.float32)
        else:
            pre_image = torch.tensor(pre_image, dtype=torch.float32).permute(2, 0, 1)
            post_image = torch.tensor(post_image, dtype=torch.float32).permute(2, 0, 1)

        return {
            "pre_image": pre_image,
            "post_image": post_image,
            "filename": filename,
            "profile": dict(profile),
        }


class Dataset_Inference(Segmentation_Dataset):
    """
    Dataset class for inference on segmented GeoTIFF tiles generated from large images.

    Args:
        origin_dir (str): Directory containing the input image tiles.
        transform (Optional[Augmentation_pipeline]): Transformation pipeline to apply to the images.
        extension (str): File extension of the image tiles (default: 'tif').

    Expected folder structure:
        origin_dir/
            └── *.tif

    Returns:
        dict:
            "image" (torch.Tensor): Transformed image tensor (C x H x W).
            "filename" (str): Filename (without extension).
            "profile" (dict): Rasterio profile metadata.
    """

    def __init__(
        self,
        origin_dir: str,
        transform: Optional[Augmentation_pipeline] = None,
        extension: str = "tif",
    ):
        super().__init__(origin_dir=origin_dir, type="infer", transform=transform)
        self.extension = extension.lower()
        self.image_dir = Path(self.origin_dir)
        self.image_filenames = self._filter_filenames()

    def _filter_filenames(self):
        return sorted([f.stem for f in self.image_dir.glob(f"*.{self.extension}")])

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, index: int):
        filename = self.image_filenames[index]
        path = self.image_dir / f"{filename}.{self.extension}"

        image, profile, _ = read_tiff_rasterio(path, bands=[1, 2, 3], with_profile=True)
        image = np.array(image).astype(np.float32) / 255.0

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"].to(torch.float32)
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        return {"image": image, "filename": filename, "profile": dict(profile)}
