import os
import re
import time

from qgis import processing

i_min, i_max = 100, 110  # 100, 200
j_min, j_max = 100, 110  # 100, 200
folder_path = "C:/Users/besni/Documents/GitHub/damaged-building-detector/data/predictions"


def extract_tile_ij(filename: str):
    """
    Parse a filename of the form 'tile_i_j.tif' and return integers (i, j).
    If the pattern doesn't match, returns (None, None).
    """
    pattern = r"^tile_(\d+)_(\d+)\.tif$"  # regex: start, 'tile_', digits, '_', digits, '.tif', end
    match = re.search(pattern, filename)
    if match:
        # match.group(1) is the i index, group(2) is the j index
        return int(match.group(1)), int(match.group(2))
    return None, None


def filter_list(filename: str):
    """
    Determine whether a given filename should be included in the list.
    Returns False if indices are missing or out of the specified bounds.
    """
    i, j = extract_tile_ij(filename)
    if i is None or j is None:
        return False  # filename didn’t match the expected pattern
    # only include tiles within the specified i/j ranges
    return (i_min <= i <= i_max) and (j_min <= j <= j_max)


def build_list_file(folder: str):
    """
    Scan the given folder, filter TIFFs by tile indices, and return
    a list of absolute paths to the matching files.
    """
    absolute_path = os.path.abspath(folder)
    # list all entries in the folder, keep only those passing filter_list
    file_list = [
        os.path.join(absolute_path, f) for f in os.listdir(absolute_path) if filter_list(f)
    ]
    return file_list


# Get your file list
file_list = build_list_file(folder_path)
print(f"{len(file_list)} have been found")

folder_output_path = (
    "C:/Users/besni/Documents/GitHub/damaged-building-detector/data/processed_data"
)

# Set your output VRT path
output_vrt = os.path.join(os.path.abspath(folder_path), "predictions.vrt")

# Run gdalbuildvrt via QGIS Processing Framework
# Build a Virtual dataset (Virtual Raster) from a list of files (allow to merge the files from the file_list)
# (vrt file)
t0 = time.time()
processing.run(
    "gdal:buildvirtualraster",
    {
        "INPUT": file_list,
        "RESOLUTION": 0,  # 0: highest resolution
        "SEPARATE": False,
        "PROJ_DIFFERENCE": False,
        "ADD_ALPHA": False,
        "ASSIGN_CRS": None,
        "RESAMPLING": 0,
        "OUTPUT": output_vrt,
    },
)
t1 = time.time()
print("✅ VRT created at:", output_vrt)
print(f"Virtual Raster has been successfully computed in {t1 - t0}s")

# Build Pyramids / Overviews (ovr file)
input = output_vrt
resampling = 0
clean = True

processing.run(
    "gdal:overviews",
    {
        "INPUT": input,
        "CLEAN": clean,
        "LEVELS": "",
        "RESAMPLING": resampling,
        "FORMAT": 2,
        "EXTRA": "",
    },
)

print(f"🧱 Overviews successfully built in {time.time() - t1}s")
