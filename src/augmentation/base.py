import os
import pathlib

import matplotlib.pyplot as plt

import albumentations as A
from torchvision.transforms import v2
from albumentations.core.composition import OneOf
from albumentations.pytorch import ToTensorV2
import numpy as np

from src.data import renormalize_image

TARGET_SIZE = (256, 256)


class Augmentation_pipeline:
    def __init__(
        self,
        transform: A.Compose = None,
        image_size: tuple = TARGET_SIZE,
        mean: list = None,
        std: list = None,
        mode: str = "train",
        package: str = "albumentations",
    ):
        """
        Initialize the Augmentation pipeline.

        Args:
            transform (A.Compose): Albumentations transformation pipeline.
            image_size (tuple): Size of the images.
            mean (list): Mean values for normalization.
            std (list): Standard deviation values for normalization.
            mode (str) : Augmentation mode (btw 'train','val', 'test' or 'infer')
        """
        assert package in [
            "albumentations",
            "torchvision",
        ], "Package must be 'albumentations' or 'torchvision'"
        assert mode in [
            "train",
            "val",
            "test",
            "infer",
        ], "Dataset must be 'train','val', 'test' or 'infer'"
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

    def __repr__(self):
        return f"{self.mode} {self.package} augmentation pipeline \n {self.pipeline}"

    def __call__(self, **kwargs):
        return self.pipeline(**kwargs)

    def _build_augmentation_pipeline(self):
        if self.package == "albumentations":
            return self._build_albumentations_pipeline()  # return a ablumentation pipeline
        elif self.package == "torchvision":
            return self._build_torchvision_pipeline()  # return a pytorch pipeline
        else:
            return None

    def _build_torchvision_pipeline(self):
        raise NotImplementedError

    def _build_albumentations_pipeline(self):
        transforms = []

        if self.mode == "train":
            if self.image_size is not None:
                transforms.append(
                    A.RandomResizedCrop(
                        size=(self.image_size[0], self.image_size[1]),
                        scale=(0.8, 1),
                        p=1.0,
                    )
                )
        elif self.mode == "infer":
            if self.image_size is not None:
                transforms.append(A.Resize(height=self.image_size[0], width=self.image_size[1], p=1.0))
        else:
            if self.image_size is not None:
                transforms.append(A.CenterCrop(height=self.image_size[0], width=self.image_size[1], p=1.0))

        if self.mode == "train":
            transforms.extend(
                [
                    OneOf(
                        [
                            A.Affine(keep_ratio=True, scale=1, rotate=(1, 1), p=0.5),
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                            A.RandomRotate90(p=0.5),
                            A.Transpose(p=0.5),
                        ],
                        p=1,
                    ),  # Spatial Transformation
                    OneOf(
                        [
                            A.RandomBrightnessContrast(p=0.5),
                            A.HueSaturationValue(p=0.5),
                            A.RGBShift(p=0.5),
                        ],
                        p=1,
                    ),  # Color Transformation
                ]
            )

        if self.mean and self.std:
            transforms.append(
                A.Normalize(
                    mean=self.mean,
                    std=self.std,
                    max_pixel_value=self.max_pixel_value,
                    p=1.0,
                )
            )

        transforms.append(ToTensorV2())

        return A.Compose(
            transforms=transforms,
            additional_targets={"post_image": "image", "post_mask": "mask"},
        )

    def _build_config_filename(self):
        return f"{self.mode}_augmentation_config_{self.package}"

    def save_pipeline(self, data_format: str = "json", folder_path: str = None) -> None:
        os.makedirs(name=folder_path, exist_ok=True)
        if self.package == "albumentations":
            filepath = os.path.join(folder_path, self._build_config_filename())
            A.save(
                self.pipeline,
                filepath_or_buffer=f"{filepath}.{data_format}",
                data_format=data_format,
            )

    @classmethod
    def load_pipeline(cls, filepath: str = None):
        path = pathlib.Path(filepath)
        data_format = path.suffix[1:]
        filename = path.stem

        assert data_format in [
            "yaml",
            "json",
        ], "Only 'json' and 'yaml' formats are supported."

        filename_components = filename.split("_")
        mode_, package = filename_components[0], filename_components[-1]

        pipeline = A.load(filepath, data_format=data_format)
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
    ):
        cols = 3 if image_2 is not None else 2
        fig, ax = plt.subplots(nrows=n_samples, ncols=cols, figsize=(10, 24))
        ax = np.atleast_2d(ax)  # ensures 2D indexing even when n_samples == 1

        for i in range(n_samples):
            results = self.pipeline(image=image_1, post_image=image_2, mask=mask)

            image = np.transpose(
                renormalize_image(results["image"], mean=self.mean, std=self.std, device="cpu").numpy(),
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


def _extract_fields(config_dict: dict):

    _config_dict = config_dict["transform"] if "transform" in config_dict.keys() else config_dict

    result = {
        "sizes": [],
        "means": [],
        "stds": [],
    }

    def _recursive_search(d):
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
