import os
from typing import Tuple

import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.profiles import Profile
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.utils import save_numpy_rasterGeoTiff
from src.utils.visualization import COLOR_DICT, DEFAULT_MAPPING, apply_color_map


def custom_infer_collate_siamese(batch: list[dict]):
    """A Custom Pytorch Collate function handling the aggregation of filenames (for naming prediction) and share the GeoTiff profile
        to the predictions files.

    Args:
        batch (list[dict]): Pytorch collate function take a list of dictionary as input.

    Returns:
        _type_: Dict[str, list[Torch.Tensor|str|Profile]]
    """
    pre_images = torch.stack([item["pre_image"] for item in batch])
    post_images = torch.stack([item["post_image"] for item in batch])
    filenames = [item["filename"] for item in batch]
    profiles = [item["profile"] for item in batch]
    return {
        "pre_image": pre_images,
        "post_image": post_images,
        "filename": filenames,
        "profile": profiles,
    }


def custom_infer_collate(batch: list[dict]):
    images = torch.stack([item["image"] for item in batch])
    filenames = [item["filename"] for item in batch]
    profiles = [item["profile"] for item in batch]
    return {"image": images, "filename": filenames, "profile": profiles}


def save_batch_prediction(
    predictions: torch.Tensor,
    filenames: list[str],
    profiles: list[Profile] = None,
    pred_folder_path: str = None,
    mode: str = "png",
    color_map: dict[int, Tuple[int, int, int, int]] = DEFAULT_MAPPING,
):
    """
    Save predictions as GeoTIFFs or PNGs/NPY depending on mode.

    Args:
        predictions (torch.Tensor): Batch of predicted masks [B, H, W].
        filenames (list[str]): List of output filenames (no extension).
        profiles (list[dict]): List of GeoTIFF profiles (one per image).
        pred_folder_path (str): Output directory.
        mode (str): File format: 'tif', 'npy', or 'png'.
        color_map (dict): Optional color mapping for PNGs (class -> RGB).
    """

    mask_predictions = 255.0 * apply_color_map(
        predictions, color_map, with_transparency=True
    ).permute(0, 2, 3, 1).numpy().astype(
        np.uint8
    )  # RGB batch (B, C, H, W) -> (B, H, W, C)

    for i, (pred, basename) in enumerate(zip(mask_predictions, filenames)):
        file_path = os.path.join(pred_folder_path, f"{basename}.{mode.lower()}")

        if mode == "tif":
            profile = rasterio.profiles.Profile(profiles[i]).copy()
            profile.update({"count": 4, "dtype": "uint8", "compress": "deflate"})

            save_numpy_rasterGeoTiff(data=pred, output_file=file_path, profile=profile)

        elif mode == "png":
            pred_img = Image.fromarray(pred, mode="RGBA")
            pred_img.save(file_path)
        elif mode == "npy":
            np.save(file_path, pred)
        else:
            raise ValueError("Unsupported mode. Use 'tif', 'png', or 'npy'.")


def batch_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str,
    siamese: bool = False,
    pred_folder_path: str = None,
    color_mode: str = "default",
    save: bool = True,
):
    """
    Perform inference on a dataset using a PyTorch model.

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): DataLoader with inference data.
        device (torch.device | str): CUDA or CPU.
        siamese (bool): If model takes both pre/post images.
        pred_folder_path (str): Folder to save prediction outputs.
        color_mode (str) : choose btw 'default', 'building', 'cloud', 'damage'
        save (bool) : choose to save prediction into pred_folder_path folder
    """
    if save:
        os.makedirs(pred_folder_path, exist_ok=True)
    # Torch Compile for speedup computation
    model.to(device)
    model.eval()
    with torch.no_grad():
        with tqdm(dataloader, desc="Inference Running", unit="batch") as t:
            for batch in t:
                filenames, profiles = batch["filename"], batch["profile"]
                if siamese:
                    pre_images = batch["pre_image"].to(device)
                    post_images = batch["post_image"].to(device)
                    outputs = model(pre_images, post_images)
                else:
                    images = batch["image"].to(device)
                    outputs = model(images)

                preds = torch.argmax(outputs, dim=1).cpu()

                if save:
                    save_batch_prediction(
                        predictions=preds,
                        filenames=filenames,
                        profiles=profiles,
                        pred_folder_path=pred_folder_path,
                        mode="tif",
                        color_map=COLOR_DICT.get(color_mode, "default"),
                    )
