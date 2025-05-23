import os
import shutil
import tempfile

import pytest
import rasterio

from src.data import generate_tiles

# Sample data directory (adjust as needed)
data_folder_path = "./data/data_samples"


# Parameterized test cases with input file and tile size
@pytest.mark.parametrize(
    "input_file, grid_x, grid_y",
    [
        (
            os.path.join(
                data_folder_path, "Puerto_Rico_dataset/Post_Event_Grids_In_TIFF/tile_0_107.tif"
            ),
            256,
            256,
        ),
        (
            os.path.join(
                data_folder_path, "Puerto_Rico_dataset/Pre_Event_Grids_In_TIFF/tile_0_107.tif"
            ),
            256,
            256,
        ),
    ],
)
def test_generate_tiles_tiff(input_file: str, grid_x: int, grid_y: int):
    # Create a temporary directory to store generated tiles
    output_dir = tempfile.mkdtemp()

    try:
        # Open the input raster file to get dimensions
        with rasterio.open(input_file) as dataset:
            width = dataset.width
            height = dataset.height

        # Calculate expected number of tiles
        expected_tiles_x = width // grid_x
        expected_tiles_y = height // grid_y
        expected_generated_tiles = expected_tiles_x * expected_tiles_y

        # Run the tile generation
        generate_tiles(input_file, output_dir, grid_x, grid_y)

        # Count the number of files generated
        actual_generated_tiles = len([f for f in os.listdir(output_dir) if f.endswith(".tif")])

        # Assertion with helpful message
        assert actual_generated_tiles == expected_generated_tiles, (
            f"Expected {expected_generated_tiles} tiles, "
            f"but found {actual_generated_tiles} in {output_dir}"
        )

    finally:
        # Clean up the temporary directory
        shutil.rmtree(output_dir)
