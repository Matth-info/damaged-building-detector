from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import rasterio
import shapely
import torch
from PIL import Image
from rasterio.features import shapes
from rasterio.warp import transform_geom
from tqdm import tqdm

if TYPE_CHECKING:
    from rasterio.transform import AffineTransformer


def renormalize_image(
    image: torch.Tensor | np.ndarray,
    mean: list[float] | tuple[float],
    std: list[float] | tuple[float],
    device: str = "cpu",
) -> torch.Tensor | np.ndarray:
    """Renormalizes an image tensor or NumPy array by reversing the normalization process.

    Args:
    ----
        image (torch.Tensor or np.ndarray): Normalized image (C, H, W) with values in range ~N(0,1).
        mean (list or tuple): Mean values used for normalization (per channel).
        std (list or tuple): Standard deviation values used for normalization (per channel).
        device (str): Device on which the computation is performed, either 'cpu' or 'cuda'.

    Returns:
        torch.Tensor or np.ndarray: Renormalized image with pixel values in range [0, 1].

    """
    if isinstance(image, torch.Tensor):
        mean = torch.tensor(mean).view(-1, 1, 1).to(device)  # Shape (C, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1).to(device)  # Shape (C, 1, 1)
        renormalized_image = image * std + mean
        return torch.clamp(renormalized_image, 0, 1)  # Keep values in [0, 1]

    if isinstance(image, np.ndarray):
        mean = np.array(mean).reshape(-1, 1, 1)  # Shape (C, 1, 1)
        std = np.array(std).reshape(-1, 1, 1)  # Shape (C, 1, 1)
        renormalized_image = image * std + mean
        return np.clip(renormalized_image, 0, 1)  # Keep values in [0, 1]

    raise TypeError("Input should be a torch.Tensor or a np.ndarray")


# Manipulate TIF images
def extract_title_ij(filename: str) -> tuple[int, int]:
    """Extract coordinate ids (i,j) from tile_i_j.tif filename."""
    pattern = r"tile_(\d+)_(\d+).tif"
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def read_tiff_pillow(filepath: str, mode: str) -> np.ndarray:
    """Read a tif image with PIL package and return a numpy array."""
    return np.array(Image.open(filepath).convert(mode))


def read_tiff_rasterio(
    file_path: str,
    bands: list | None = None,
    *,
    with_profile: bool = False,
    with_transform: bool = False,
) -> tuple[np.ndarray, dict | None, AffineTransformer | None]:
    """Read a GeoTIFF image using rasterio and return it as a NumPy array, optionally with metadata.

    Args:
    ----
        file_path (str): Path to the GeoTIFF file.
        bands (list, optional): list of band indices to read. Defaults to [1, 2, 3].
        with_profile (bool, optional): If True, returns the raster profile. Defaults to False.
        with_transform (bool, optional): If True, returns the affine transform. Defaults to False.

    Returns:
    -------
        tuple: (image_array, profile, transform)
            image_array (np.ndarray): Image data as a NumPy array with shape (height, width, bands).
            profile (dict or None): Raster profile dictionary if with_profile is True, else None.
            transform (Affine or None): Affine transform if with_transform is True, else None.

    """
    if bands is None:
        bands = [1, 2, 3]
    with rasterio.open(file_path) as src:
        image_array = src.read(bands)
        profile = src.profile
        transform = src.transform
    return (
        image_array.transpose(1, 2, 0),
        profile if with_profile else None,
        transform if with_transform else None,
    )


def save_rastergeotiff(data: np.ndarray, output_file: str, profile: dict) -> None:
    """Save GeoTiff images (No axis swap required)."""
    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(data)


def save_numpy_rastergeotiff(data: np.ndarray, output_file: str, profile: dict) -> None:
    """Save Numpy array as GeoTiff Images."""
    # Transpose Numpy array
    # GeoTiff expect (bands, height, width) not (height, width,  bands)
    data = np.moveaxis(data, -1, 0)
    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(data)


def extract_coor_from_tiff_image(filepath: str) -> tuple[float, float, float, float] | None:
    """Extract the bounding box (in WGS84) from a GeoTIFF image.

    Returns:
    -------
        Tuple of (left, bottom, right, top) or None if failed.

    """
    bounds = None
    try:
        with rasterio.open(filepath) as dataset:
            mask = dataset.dataset_mask()
            for geom, _ in shapes(mask, transform=dataset.transform):
                _geom = transform_geom(dataset.crs, "EPSG:4326", geom, precision=6)
                bounds = shapely.geometry.shape(_geom).bounds
    except Exception as e:
        logging.exception("Error processing %s", filepath)
        return None  # (minx, miny, maxx, maxy)
    else:
        return bounds


def extract_coordinates_parallel(
    filepaths: list[str],
    folder_path: list[str],
    max_workers: int = 8,
) -> list[tuple[float, float, float, float]] | None:
    """Parallel wrapper for extracting coordinates from multiple GeoTIFF files.

    Args:
    ----
        filepaths (list[str]): list of paths to GeoTIFF files.
        folder_path (str): Folder path
        max_workers (int): Number of parallel threads.

    Returns:
    -------
        list of coordinate tuples (left, bottom, right, top).

    """
    results = []
    filepaths = [Path(folder_path) / filepath for filepath in filepaths]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = list(
            tqdm(
                executor.map(extract_coor_from_tiff_image, filepaths),
                total=len(filepaths),
                desc="Extracting coordinates",
            ),
        )
        results.extend(futures)
    return results


def filter_files_by_bounds(
    filepaths: list[str],
    coordinates: list[tuple[float, float, float, float]],
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> list[str]:
    """Filter GeoTIFF file paths based on whether their bounds fall within a given bounding box.

    Args:
    ----
        filepaths (list[str]): list of file paths.
        coordinates (list[Tuple]): Corresponding list of bounding boxes (left, bottom, right, top).
        lon_min (float): Minimum longitude (left).
        lon_max (float): Maximum longitude (right).
        lat_min (float): Minimum latitude (bottom).
        lat_max (float): Maximum latitude (top).

    Returns:
    -------
        list[str]: Filtered list of file paths.

    """
    filtered_files = []
    filtered_coordinates = []

    for filepath, bounds in zip(filepaths, coordinates):
        if bounds is None:
            continue  # skip failed coordinate extraction

        left, bottom, right, top = bounds

        if (left >= lon_min) and (right <= lon_max) and (bottom >= lat_min) and (top <= lat_max):
            filtered_files.append(filepath)
            filtered_coordinates.append(bounds)

    return filtered_files, filtered_coordinates
