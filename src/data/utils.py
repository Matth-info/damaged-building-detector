import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
import rasterio
import shapely
import torch
from PIL import Image
from rasterio.features import shapes
from rasterio.warp import transform_geom
from tqdm import tqdm


def renormalize_image(
    image: torch.Tensor | np.ndarray,
    mean: list[float] | tuple[float],
    std: list[float] | tuple[float],
    device: str = "cpu",
):
    """
    Renormalizes an image tensor or NumPy array by reversing the normalization process.

    Args:
        image (torch.Tensor or np.ndarray): Normalized image (C, H, W) with values in range ~N(0,1).
        mean (list or tuple): Mean values used for normalization (per channel).
        std (list or tuple): Standard deviation values used for normalization (per channel).
        device (str): 'cpu' or 'cuda'
    Returns:
        torch.Tensor or np.ndarray: Renormalized image with pixel values in range [0, 1].
    """
    if isinstance(image, torch.Tensor):
        mean = torch.tensor(mean).view(-1, 1, 1).to(device)  # Shape (C, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1).to(device)  # Shape (C, 1, 1)
        renormalized_image = image * std + mean
        return torch.clamp(renormalized_image, 0, 1)  # Keep values in [0, 1]

    elif isinstance(image, np.ndarray):
        mean = np.array(mean).reshape(-1, 1, 1)  # Shape (C, 1, 1)
        std = np.array(std).reshape(-1, 1, 1)  # Shape (C, 1, 1)
        renormalized_image = image * std + mean
        return np.clip(renormalized_image, 0, 1)  # Keep values in [0, 1]

    else:
        raise TypeError("Input should be a torch.Tensor or a np.ndarray")


# Manipulate TIF images
def extract_title_ij(filename: str) -> Tuple[int, int]:
    pattern = r"tile_(\d+)_(\d+).tif"
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def read_tiff_pillow(filepath: str, mode: str):
    return np.array(Image.open(filepath).convert(mode))


def read_tiff_rasterio(file_path, bands: list = [1, 2, 3], with_profile: bool = False):
    """
    Read a GeoTIFF image and converted into a Numpy array
    """
    with rasterio.open(file_path) as src:
        image_array = src.read(bands)
        profile = src.profile
    return image_array.transpose(1, 2, 0), profile if with_profile else None


def save_rasterGeoTiff(data, output_file, profile):
    """Save GeoTiff images"""
    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(data)


def save_numpy_rasterGeoTiff(data: np.ndarray, output_file, profile):
    """Save Numpy array as GeoTiff Images"""
    # Transpose Numpy array
    # GeoTiff expect (bands, height, width) not (height, width,  bands)
    data = np.moveaxis(data, -1, 0)
    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(data)


def extract_coor_from_tiff_image(filepath: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Extract the bounding box (in WGS84) from a GeoTIFF image.

    Returns:
        Tuple of (left, bottom, right, top) or None if failed.
    """
    try:
        with rasterio.open(filepath) as dataset:
            mask = dataset.dataset_mask()
            for geom, _ in shapes(mask, transform=dataset.transform):
                geom = transform_geom(dataset.crs, "EPSG:4326", geom, precision=6)
                bounds = shapely.geometry.shape(geom).bounds
                return bounds  # (minx, miny, maxx, maxy)
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


def extract_coordinates_parallel(
    filepaths: List[str], folder_path: list[str], max_workers: int = 8
) -> List[Optional[Tuple[float, float, float, float]]]:
    """
    Parallel wrapper for extracting coordinates from multiple GeoTIFF files.

    Args:
        filepaths (List[str]): List of paths to GeoTIFF files.
        folder_path (str): Folder path
        max_workers (int): Number of parallel threads.

    Returns:
        List of coordinate tuples (left, bottom, right, top).
    """
    results = []
    filepaths = [os.path.join(folder_path, filepath) for filepath in filepaths]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = list(
            tqdm(
                executor.map(extract_coor_from_tiff_image, filepaths),
                total=len(filepaths),
                desc="Extracting coordinates",
            )
        )
        results.extend(futures)
    return results


def filter_files_by_bounds(
    filepaths: List[str],
    coordinates: List[Tuple[float, float, float, float]],
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> List[str]:
    """
    Filters GeoTIFF file paths based on whether their bounds fall within a given bounding box.

    Args:
        filepaths (List[str]): List of file paths.
        coordinates (List[Tuple]): Corresponding list of bounding boxes (left, bottom, right, top).
        lon_min (float): Minimum longitude (left).
        lon_max (float): Maximum longitude (right).
        lat_min (float): Minimum latitude (bottom).
        lat_max (float): Maximum latitude (top).

    Returns:
        List[str]: Filtered list of file paths.
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
