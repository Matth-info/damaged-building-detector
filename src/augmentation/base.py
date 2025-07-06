from __future__ import annotations

import os
import pathlib
from pathlib import Path
from typing import NoReturn, Optional

import albumentations as alb
import matplotlib.pyplot as plt
import numpy as np
from albumentations.core.composition import OneOf
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import v2

from src.data import renormalize_image

TARGET_SIZE = (256, 256)


class Augmentation_pipeline:  # noqa: D101, N801
    def __init__(
        self,
        transform: alb.Compose = None,
        image_size: tuple = TARGET_SIZE,
        mean: list | None = None,
        std: list | None = None,
        mode: str = "train",
        package: str = "albumentations",
    ) -> None:
        """Initialize the Augmentation pipeline.

        Args:
        ----
            transform (alb.Compose): Albumentations transformation pipeline.
            image_size (tuple): Size of the images.
            mean (list): Mean values for normalization.
            std (list): Standard deviation values for normalization.
            mode (str): Augmentation mode (btw 'train','val', 'test' or 'infer')
            package (str): The augmentation package to use ('albumentations' or 'torchvision').

        """
        if package not in [
            "albumentations",
            "torchvision",
        ]:
            msg = "Package must be 'albumentations' or 'torchvision'"
            raise ValueError(msg)
        if mode not in [
            "train",
            "val",
            "test",
            "infer",
        ]:
            msg = "Dataset must be 'train','val', 'test' or 'infer'"
            raise ValueError(msg)
        self.mode = mode
        self.package = package
        self.max_pixel_value = 1
        self.image_size = image_size
        self.mean = mean
        self.std = std

        if not transform:
            self.pipeline = self._build_augmentation_pipeline()
        else:
            self.pipeline = transform

    def __repr__(self) -> str:
        """Return a string representation of the augmentation pipeline."""
        return f"{self.mode} {self.package} augmentation pipeline \n {self.pipeline}"

    def __call__(self, **kwargs: dict) -> alb.Compose:
        """Apply the augmentation pipeline to the provided inputs."""
        return self.pipeline(**kwargs)

    def _build_augmentation_pipeline(self) -> alb.Compose:
        if self.package == "albumentations":
            return self._build_albumentations_pipeline()  # return alb ablumentation pipeline
        if self.package == "torchvision":
            return self._build_torchvision_pipeline()  # return alb pytorch pipeline
        return None

    def _build_torchvision_pipeline(self) -> NoReturn:
        raise NotImplementedError

    def _build_albumentations_pipeline(self) -> alb.Compose:
        transforms = []

        if self.mode == "train":
            if self.image_size is not None:
                transforms.append(
                    alb.RandomResizedCrop(
                        size=(self.image_size[0], self.image_size[1]),
                        scale=(0.8, 1),
                        p=1.0,
                    ),
                )
        elif self.mode == "infer":
            if self.image_size is not None:
                transforms.append(
                    alb.Resize(height=self.image_size[0], width=self.image_size[1], p=1.0),
                )
        elif self.mode in ("val", "test") and self.image_size is not None:
            # Validation or Test mode
            transforms.append(
                alb.CenterCrop(height=self.image_size[0], width=self.image_size[1], p=1.0),
            )

        if self.mode == "train":
            transforms.extend(
                [
                    OneOf(
                        [
                            alb.Affine(keep_ratio=True, scale=1, rotate=(1, 1), p=0.5),
                            alb.HorizontalFlip(p=0.5),
                            alb.VerticalFlip(p=0.5),
                            alb.RandomRotate90(p=0.5),
                            alb.Transpose(p=0.5),
                        ],
                        p=1,
                    ),  # Spatial Transformation
                    OneOf(
                        [
                            alb.RandomBrightnessContrast(p=0.5),
                            alb.HueSaturationValue(p=0.5),
                            alb.RGBShift(p=0.5),
                        ],
                        p=1,
                    ),  # Color Transformation
                ],
            )

        if self.mean and self.std:
            transforms.append(
                alb.Normalize(
                    mean=self.mean,
                    std=self.std,
                    max_pixel_value=self.max_pixel_value,
                    p=1.0,
                ),
            )

        transforms.append(ToTensorV2())

        return alb.Compose(
            transforms=transforms,
            additional_targets={"post_image": "image", "post_mask": "mask"},
        )

    def _build_config_filename(self) -> str:
        return f"{self.mode}_augmentation_config_{self.package}"

    def save_pipeline(self, data_format: str = "json", folder_path: str | None = None) -> None:
        """Save the augmentation pipeline configuration to a file.

        Args:
            data_format (str): The format to save the pipeline ('json' or 'yaml').
            folder_path (str | None): The directory where the pipeline configuration will be saved.

        Returns:
            str or None: The file path if saved, otherwise None.

        """
        Path(folder_path).mkdir(parents=True)
        if self.package == "albumentations":
            filepath = Path(folder_path) / self._build_config_filename()
            alb.save(
                self.pipeline,
                filepath_or_buffer=f"{filepath}.{data_format}",
                data_format=data_format,
            )
            return filepath
        return None

    @classmethod
    def load_pipeline(cls, filepath: str | None = None) -> Augmentation_pipeline:
        """Load an augmentation pipeline from a configuration file.

        Args:
            filepath (str | None): Path to the configuration file.

        Returns:
            Augmentation_pipeline: An instance of Augmentation_pipeline initialized from the file.

        """
        path = pathlib.Path(filepath)
        data_format = path.suffix[1:]
        filename = path.stem

        if data_format not in ["yaml", "json"]:
            msg = "Only 'json' and 'yaml' formats are supported."
            raise ValueError(msg)

        filename_components = filename.split("_")
        mode_, package = filename_components[0], filename_components[-1]

        pipeline = alb.load(filepath, data_format=data_format)
        pipeline_dict = pipeline.to_dict()
        fields = _extract_fields(pipeline_dict)

        return cls(
            transform=pipeline,
            mode=mode_,
            package=package,
            mean=fields.get("means")[0] if fields.get("means") else None,
            std=fields.get("stds")[0] if fields.get("stds") else None,
            image_size=fields.get("sizes")[0] if fields.get("sizes") else TARGET_SIZE,
        )

    def debug(
        self,
        image_1: np.ndarray = None,
        image_2: np.ndarray = None,
        mask: np.ndarray = None,
        n_samples: int = 10,
    ) -> None:
        """Visualize the effect of the augmentation pipeline by applying it to sample images and masks.

        Args:
            image_1 (np.ndarray): The first input image.
            image_2 (np.ndarray, optional): The second input image for post-processing visualization.
            mask (np.ndarray, optional): The ground truth mask.
            n_samples (int): Number of samples to visualize.

        """
        cols = 3 if image_2 is not None else 2
        fig, ax = plt.subplots(nrows=n_samples, ncols=cols, figsize=(10, 24))
        ax = np.atleast_2d(ax)  # ensures 2D indexing even when n_samples == 1

        for i in range(n_samples):
            results = self.pipeline(image=image_1, post_image=image_2, mask=mask)

            image = np.transpose(
                renormalize_image(
                    results["image"],
                    mean=self.mean,
                    std=self.std,
                    device="cpu",
                ).numpy(),
                (1, 2, 0),
            )

            vis_image_2 = None
            if image_2 is not None:
                vis_image_2 = np.transpose(
                    renormalize_image(
                        results["post_image"],
                        mean=self.mean,
                        std=self.std,
                        device="cpu",
                    ).numpy(),
                    (1, 2, 0),
                )

            vis_mask = results["mask"].numpy()

            ax[i, 0].imshow(image, interpolation="nearest")
            ax[i, 0].set_title("Image 1")
            ax[i, 0].set_axis_off()

            ax[i, 1].imshow(vis_mask, interpolation="nearest")
            ax[i, 1].set_title("Ground truth mask")
            ax[i, 1].set_axis_off()

            if vis_image_2 is not None:
                ax[i, 2].imshow(vis_image_2, interpolation="nearest")
                ax[i, 2].set_title("Image 2")
                ax[i, 2].set_axis_off()

        plt.tight_layout()
        plt.show()


def _extract_fields(config_dict: dict) -> dict:
    _config_dict = config_dict.get("transform", config_dict)

    result = {
        "sizes": [],
        "means": [],
        "stds": [],
    }

    def _recursive_search(d: dict) -> None:
        if isinstance(d, dict):
            # Size extraction
            if "size" in d:
                result["sizes"].append(d["size"])
            elif "height" in d and "width" in d:
                result["sizes"].append((d["height"], d["width"]))

            # Mean, std, and max_pixel_value extraction
            if "mean" in d:
                result["means"].append(d["mean"])
            if "std" in d:
                result["stds"].append(d["std"])

            # Recurse into any sub-transforms
            for key in ["transforms", "children"]:
                if key in d and isinstance(d[key], list):
                    for sub_transform in d[key]:
                        _recursive_search(sub_transform)

    _recursive_search(_config_dict)
    return result
