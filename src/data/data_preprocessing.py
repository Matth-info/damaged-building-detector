import logging
import os
from pathlib import Path

import rasterio
from rasterio.windows import Window
from tqdm import tqdm

from src.data.utils import save_rastergeotiff

_logger = logging.getLogger(__name__)


def generate_tiles(input_file: str, output_dir: str, grid_x: int, grid_y: int) -> None:
    """Split a large GeoTIFF into smaller tiles and save them individually using rasterio (Sequential Implementation).

    Args:
    ----
        input_file (str): Path to the input GeoTIFF file.
        output_dir (str): Directory to save the output tiles.
        grid_x (int): Width of each tile in pixels.
        grid_y (int): Height of each tile in pixels.

    """
    subfolder_name = Path(input_file).stem  # extract the filename
    output_dir = Path(output_dir) / subfolder_name
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_file) as src:
        width = src.width
        height = src.height
        num_bands = src.count
        dtype = src.dtypes[0]
        profile = src.profile.copy()

        transform = src.transform
        crs = src.crs

        num_tiles_x = width // grid_x
        num_tiles_y = height // grid_y

        _logger.info("Total number of tiles: %d", num_tiles_x * num_tiles_y)

        for i in tqdm(range(num_tiles_x), desc="Processing tiles (X)"):
            for j in range(num_tiles_y):
                x_offset = i * grid_x
                y_offset = j * grid_y

                tile_width = min(grid_x, width - x_offset)
                tile_height = min(grid_y, height - y_offset)

                window = Window(x_offset, y_offset, tile_width, tile_height)
                tile_data = src.read(window=window)

                # Update transform for this tile
                tile_transform = src.window_transform(window)

                # Update profile for tile
                tile_profile = profile.copy()
                tile_profile.update(
                    {
                        "height": tile_height,
                        "width": tile_width,
                        "transform": tile_transform,
                        "compress": "deflate",
                        "tiled": True,
                        "predictor": 2,
                    },
                )

                output_file = Path(output_dir) / f"tile_{i}_{j}.tif"

                # Save Raster Tile in GeoTIFF format
                save_rastergeotiff(tile_data, output_file, tile_profile)

    _logger.info("Tiles generation completed.")


from concurrent.futures import ThreadPoolExecutor


def process_tile(
    i: int, j: int, grid_x: int, grid_y: int, src_profile: dict, input_file: str, output_dir: str
) -> None:
    """Process and saves a single tile from the input GeoTIFF file.

    Args:
        i (int): Tile index along the x-axis.
        j (int): Tile index along the y-axis.
        grid_x (int): Width of each tile in pixels.
        grid_y (int): Height of each tile in pixels.
        src_profile (dict): Rasterio profile dictionary for the source image.
        input_file (str): Path to the input GeoTIFF file.
        output_dir (str): Directory to save the output tile.

    """
    with rasterio.open(input_file) as src:
        width = src.width
        height = src.height

        x_offset = i * grid_x
        y_offset = j * grid_y

        tile_width = min(grid_x, width - x_offset)
        tile_height = min(grid_y, height - y_offset)

        window = Window(x_offset, y_offset, tile_width, tile_height)
        tile_data = src.read(window=window)

        tile_transform = src.window_transform(window)

        tile_profile = src_profile.copy()
        tile_profile.update(
            {
                "height": tile_height,
                "width": tile_width,
                "transform": tile_transform,
                "compress": "deflate",
                "tiled": True,
                "predictor": 2,
            },
        )

        output_file = Path(output_dir) / f"tile_{i}_{j}.tif"
        save_rastergeotiff(tile_data, output_file, tile_profile)


def generate_tiles_parallel(
    input_file: str,
    output_dir: str,
    grid_x: int,
    grid_y: int,
    max_workers: int = 4,
) -> None:
    """Split a large GeoTIFF into smaller tiles and save them individually using parallel processing.

    Args:
        input_file (str): Path to the input GeoTIFF file.
        output_dir (str): Directory to save the output tiles.
        grid_x (int): Width of each tile in pixels.
        grid_y (int): Height of each tile in pixels.
        max_workers (int, optional): Maximum number of parallel workers. Defaults to 4.

    """
    subfolder_name = Path(input_file).stem
    output_dir = Path(output_dir) / subfolder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_file) as src:
        width = src.width
        height = src.height
        profile = src.profile.copy()

        num_tiles_x = width // grid_x
        num_tiles_y = height // grid_y

        _logger.info("Total number of tiles: %d", num_tiles_x * num_tiles_y)
        tasks = [(i, j) for i in range(num_tiles_x) for j in range(num_tiles_y)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(
                    lambda ij: process_tile(
                        *ij,
                        grid_x,
                        grid_y,
                        profile,
                        input_file=input_file,
                        output_dir=output_dir,
                    ),
                    tasks,
                ),
                total=len(tasks),
                desc="Processing tiles (Parallel)",
            ),
        )

    _logger.info("Tiles generation completed (parallel).")
