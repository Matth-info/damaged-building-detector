#  pytest .\test_dataset_loading.py -v
import pytest
from src.datasets import (
    Cloud_DrivenData_Dataset,
    Puerto_Rico_Building_Dataset,
    OpenCities_Building_Dataset,
    xDB_Damaged_Building,
    xDB_Instance_Building,
    xDB_Siamese_Dataset,
    Levir_cd_dataset,
    Dataset_Inference,
)

data_folder_path = ".../data/data_samples"


@pytest.mark.parametrize(
    "dataset_cls, kwargs, expected_shapes",
    [
        (
            OpenCities_Building_Dataset,
            {
                "images_dir": f"{data_folder_path}/Open-cities/training_data/images",
                "masks_dir": f"{data_folder_path}/Open-cities/training_data/masks",
                "transform": None,
                "filter_invalid_image": False,
            },
            [(3, 1024, 1024), (1024, 1024)],
        ),
        (
            xDB_Damaged_Building,
            {
                "origin_dir": f"{data_folder_path}/xDB/tier3",
                "mode": "building",
                "time": "pre",
                "transform": None,
                "type": "train",
                "val_ratio": 0.1,
                "test_ratio": 0.1,
            },
            [(3, 1024, 1024), (1024, 1024)],
        ),
    ],
)
def test_load_standard_datasets(dataset_cls, kwargs, expected_shapes):
    """Test loading for Puerto Rico, OpenCities, and xDB datasets."""
    dataset = dataset_cls(**kwargs)
    image_shape, mask_shape = expected_shapes

    assert len(dataset) >= 1, f"{dataset_cls.__name__} Dataset Loading has failed!"
    sample = dataset[0]

    if isinstance(sample, dict):  # Some datasets return dict with images/masks
        image_shape, mask_shape = expected_shapes
        assert "image" in sample and "mask" in sample, f"{dataset_cls.__name__} sample structure incorrect!"
        assert sample["image"].shape == image_shape, f"Expected shape {image_shape}, but got {sample['image'].shape}"
        assert sample["mask"].shape == mask_shape, f"Expected shape {mask_shape}, but got {sample['mask'].shape}"
    else:
        assert sample.shape == expected_shapes[0], f"Expected shape {expected_shapes[0]}, but got {sample.shape}"


# siamese dataset
@pytest.mark.parametrize(
    "dataset_cls, kwargs, expected_shapes",
    [
        (
            Puerto_Rico_Building_Dataset,
            {
                "base_dir": f"{data_folder_path}/Puerto_Rico_dataset",
                "pre_disaster_dir": "Pre_Event_Grids_In_TIFF",
                "post_disaster_dir": "Post_Event_Grids_In_TIFF",
                "mask_dir": "Pre_Event_Grids_In_TIFF_mask",
                "transform": None,
                "extension": "tif",
            },
            [(3, 512, 512), (512, 512)],
        ),
        (
            Levir_cd_dataset,
            {
                "origin_dir": f"{data_folder_path}/Levir-cd",
                "transform": None,
                "type": "test",
            },
            [(3, 1024, 1024), (1024, 1024)],
        ),
        (
            xDB_Siamese_Dataset,
            {
                "origin_dir": f"{data_folder_path}/xDB/tier3",
                "mode": "building",
                "transform": None,
                "type": "train",
                "val_ratio": 0,
                "test_ratio": 0,
                "seed": 42,
            },
            [(3, 1024, 1024), (1024, 1024)],
        ),
        (
            Dataset_Inference,
            {
                "base_dir": f"{data_folder_path}/Puerto_Rico_dataset",
                "pre_disaster_dir": "Pre_Event_Grids_In_TIFF",
                "post_disaster_dir": "Post_Event_Grids_In_TIFF",
                "transform": None,
                "extension": "tif",
            },
            [(3, 512, 512), (512, 512)],
        ),
    ],
)
def test_load_siamese_datasets(dataset_cls, kwargs, expected_shapes):
    """Test loading for Siamese-type datasets (Levir CD, xDB Siamese)."""
    dataset = dataset_cls(**kwargs)
    image_shape, mask_shape = expected_shapes
    assert len(dataset) >= 1, f"{dataset_cls.__name__} Dataset Loading has failed!"

    sample = dataset[0]
    assert isinstance(sample, dict), "Sample should be a dictionary!"
    assert all(
        key in sample for key in ["pre_image", "post_image"]
    ), f"{dataset_cls.__name__} sample missing required keys!"

    assert (
        sample["pre_image"].shape == image_shape
    ), f"Expected shape {image_shape}, but got {sample['pre_image'].shape}"
    assert (
        sample["post_image"].shape == image_shape
    ), f"Expected shape {image_shape}, but got {sample['post_image'].shape}"

    if "pre_mask" in sample.keys():
        assert (
            sample["pre_mask"].shape == mask_shape
        ), f"Expected shape {mask_shape}, but got {sample['pre_mask'].shape}"
    if "post_mask" in sample.keys():
        assert (
            sample["post_mask"].shape == mask_shape
        ), f"Expected shape {mask_shape}, but got {sample['post_mask'].shape}"
    if "mask" in sample.keys():
        assert sample["mask"].shape == mask_shape, f"Expected shape {mask_shape}, but got {sample['mask'].shape}"
