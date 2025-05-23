import logging
import os
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, Tuple
from typing import Dict

import geopandas as gpd
import mercantile
import numpy as np
import pandas as pd
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape
from shapely.geometry.polygon import Polygon
from tqdm import tqdm

from src.data.utils import extract_coor_from_tiff_image, read_tiff_rasterio
from src.utils.visualization import COLOR_DICT, apply_inverse_color_map

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# Utils function to transform predicted segmentation mask into geo-referenced blob
# Use Vector Data instead of Raster Data improve imformation display.


def extract_overall_vector_data(file_path: str, mode: str, min_area: int = 80):
    """
    Extract georeferenced polygons from a semantic segmentation prediction mask (all classes merged)

    Args:
        file_path (str): Path to the RGB mask image.
        mode (str): Mode name corresponding to COLOR_DICT.
        min_area (int): Minimum polygon area (pixels) to keep.

    Returns:
        List[shapely.geometry.Polygon]: Extracted polygons.
    """
    color_dict = COLOR_DICT[mode]

    # Load RGB mask
    color_mask, _, transform = read_tiff_rasterio(
        file_path, bands=[1, 2, 3], with_profile=False, with_transform=True
    )

    # Convert colorized mask to 2D label mask
    mask = apply_inverse_color_map(color_mask, color_dict)

    # Create binary mask (everything > 0 becomes 1)
    binary_mask = (mask > 0).astype("uint8")

    # Label connected components
    labeled_mask, _ = ndimage.label(binary_mask)

    # Extract polygons using rasterio
    polygons = []
    for geom, value in shapes(labeled_mask, mask=binary_mask, transform=transform):
        if value == 0:
            continue
        polygon = shape(geom)
        if polygon.area >= min_area:
            polygons.append(polygon)

    return polygons


def extract_overall_vector_data_by_class(file_path: str, mode: str, min_area: int = 80):
    """
    Extract georeferenced polygons per class from a semantic segmentation prediction mask.

    Args:
        file_path (str): Path to the RGB mask image.
        mode (str): Mode name corresponding to COLOR_DICT.
        min_area (int): Minimum polygon area (pixels) to keep.

    Returns:
        List[Tuple[Polygon, int]]: List of polygons with class labels.
    """
    color_dict = COLOR_DICT[mode]

    color_mask, _, transform = read_tiff_rasterio(
        file_path, bands=[1, 2, 3], with_profile=False, with_transform=True
    )
    mask = apply_inverse_color_map(color_mask, color_dict)

    class_labels = np.unique(mask)
    polygons_by_class = []

    for class_label in class_labels:
        if class_label == 0:
            continue  # skip background
        binary_mask = (mask == class_label).astype("uint8")
        labeled_mask, _ = ndimage.label(binary_mask)

        for geom, value in shapes(labeled_mask, mask=binary_mask, transform=transform):
            if value == 0:
                continue
            polygon = shape(geom)
            if polygon.area >= min_area:
                polygons_by_class.append((polygon, int(class_label)))

    return polygons_by_class


def extract_vector_data_by_class(file_path: str, mode: str, min_area: int = 80):
    """
    Match general polygons to their most overlapping class-specific polygons.

    Args:
        file_path (str): Path to the RGB mask.
        mode (str): COLOR_DICT key for class colors.
        min_area (int): Minimum polygon area to keep.

    Returns:
        List[Tuple[Polygon, int]]: Polygons with associated class.
    """
    polygons = extract_overall_vector_data(file_path, mode, min_area)
    polygons_by_class = extract_overall_vector_data_by_class(file_path, mode, min_area)

    polygons_max_overlap = [0.0] * len(polygons)
    polygons_max_overlap_class = [None] * len(polygons)

    for polygon_i, polygon in enumerate(polygons):
        for shape_j, shape_class in polygons_by_class:
            inter_area = polygon.intersection(shape_j).area
            if inter_area > polygons_max_overlap[polygon_i]:
                polygons_max_overlap[polygon_i] = inter_area
                polygons_max_overlap_class[polygon_i] = shape_class

    return [(polygons[i], polygons_max_overlap_class[i]) for i in range(len(polygons))]


# Parallelize Implementation
def _process_single_mask(file_path: str, mode: str, min_area: int) -> Dict[str, list]:
    """
    Extract polygons from one mask and save to temporary GeoPackage.
    """
    try:
        polygons_with_class = extract_vector_data_by_class(file_path, mode, min_area)
        filename = Path(file_path).name

        if not polygons_with_class:
            return None  # skip empty

        polygons = [poly for poly, _ in polygons_with_class]
        classes = [cls for _, cls in polygons_with_class]
        source_filename = [filename for _ in range(len(polygons_with_class))]

        return {"class": classes, "geometry": polygons, "source": source_filename}
    except Exception as e:
        logging.info(f"❌ Error processing {file_path}: {e}")
        traceback.logging.info_exc()
        return None


def process_masks_parallel(
    folder_path: str,
    mode: str,
    output_path: str = "merged_output.gpkg",
    layer_name: str = "polygons",
    file_suffix: str = ".tif",
    min_area: int = 80,
    num_workers: int = 8,
):
    """
    Parallel batch processor to convert all masks in a folder to one GeoPackage layer. (Parallel Execution)

    Args:
        folder_path (str): Path to the folder containing masks.
        mode (str): Mode name corresponding to COLOR_DICT.
        output_path (str): Path to save the merged GeoPackage.
        layer_name (str): Layer name in the GeoPackage.
        file_suffix (str): File suffix to filter files in the folder.
        min_area (int): Minimum polygon area (pixels) to keep.
        num_workers (int): Number of parallel workers.
    """

    all_files = sorted(
        [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(file_suffix)]
    )

    if not all_files:
        logging.info("⚠️ No matching .tif files found.")
        return None
    else:
        _, profile, _ = read_tiff_rasterio(all_files[0], with_profile=True)
        crs = profile["crs"]
        logging.info(
            f"🚀 Processing georeferenced {len(all_files)} masks in parallel (CRS : {crs}) over {num_workers} workers ..."
        )

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_process_single_mask, file_path, mode, min_area)
            for i, file_path in enumerate(all_files)
        ]

        temp_outputs = []
        for future in tqdm(futures):
            result = future.result()
            if result:
                temp_outputs.append(result)

        # Merge all temp GPKG files
        logging.info(f"📦 Merging {len(temp_outputs)} temporary files into final GeoPackage...")
        temp_gdf = pd.concat([gpd.GeoDataFrame(data, crs=crs) for data in temp_outputs])
        merged_gdf = gpd.GeoDataFrame(temp_gdf).to_crs("EPSG:4326")
        merged_gdf["id"] = pd.Series(range(len(merged_gdf)))
        merged_gdf.to_file(
            output_path, layer=layer_name, driver="GPKG", use_arrow=True, index=False
        )
        logging.info(f"\n✅ Done: {len(merged_gdf)} polygons saved to {output_path}")


# Match footlogging.info predictions to Microsoft Global footlogging.infos.
# functions inspired by https://github.com/microsoft/GlobalMLBuildingFootlogging.infos/blob/main/examples/example_building_footlogging.infos.ipynb
def find_quad_keys(base_image: str) -> Dict[str, list | Polygon]:
    """Using mercantile package functions, find the tiles intersecting the area of interest (AOI)

    Args:
        base_image (str): _description_

    Returns:
        _type_: _description_
    """
    if Path(base_image).ext == "tif":
        left, bottom, right, top = extract_coor_from_tiff_image(base_image)  # EPSG:4326

    # longitude , latitude
    # Define the Area Of Interest (AOI)
    aoi_geom = {
        "coordinates": [[[left, top], [left, bottom], [right, bottom], [right, top]]],
        "type": "Polygon",
    }
    aoi_shape = shape(aoi_geom)
    minx, miny, maxx, maxy = aoi_shape.bounds

    quad_keys = set()
    for tile in list(mercantile.tiles(minx, miny, maxx, maxy, zooms=9)):
        quad_keys.add(mercantile.quadkey(tile))
    quad_keys = list(quad_keys)
    logging.info(f"The input area spans {len(quad_keys)} tiles: {quad_keys}")

    return {"quad_keys": quad_keys, "aoi_shape": aoi_shape}


GDP_DRIVER_MAPPING = {
    "geojson": {"driver": "GeoJSON"},
    "gpkg": {"driver": "GPKG", "layer": "polygons"},
}


def dowload_footprints_aoi(
    quad_keys: list[str],
    output_filename: str = "building_footprints",
    output_format: str = "geojson",
    aoi_shape: Polygon = None,
):
    # downloaded Opensource dataset links to footprint files
    df = pd.read_csv(
        "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv",
        dtype=str,
    )  # features : Location	QuadKey	Url	Size UploadDat

    idx = 0
    combined_gdf = gpd.GeoDataFrame()
    option_saving = GDP_DRIVER_MAPPING[output_format]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download the GeoJSON files for each tile that intersects the input geometry
        tmp_fns = []
        for quad_key in tqdm(quad_keys):
            rows = df[df["QuadKey"] == quad_key]
            if rows.shape[0] == 1:
                url = rows.iloc[0]["Url"]

                df2 = pd.read_json(url, lines=True)
                df2["geometry"] = df2["geometry"].apply(shape)

                gdf = gpd.GeoDataFrame(df2, crs=4326)
                fn = os.path.join(tmpdir, f"{quad_key}.{output_format}")
                tmp_fns.append(fn)
                if not os.path.exists(fn):
                    gdf.to_file(fn, **option_saving)
            elif rows.shape[0] > 1:
                raise ValueError(f"Multiple rows found for QuadKey: {quad_key}")
            else:
                raise ValueError(f"QuadKey is not found in dataset: {quad_key}")

        # Merge the GeoJSON files into a single file
        for i, fn in enumerate(tmp_fns):
            gdf = gpd.read_file(fn)  # Read each file into a GeoDataFrame
            gdf = gdf[gdf.geometry.within(aoi_shape)]  # Filter geometries within the AOI
            gdf["id"] = range(idx, idx + len(gdf))  # Update 'id' based on idx
            idx += len(gdf)
            combined_gdf = pd.concat([combined_gdf, gdf], ignore_index=True)

        combined_gdf = combined_gdf.to_crs("EPSG:4326")
        combined_gdf.to_file(f"{output_filename}.{output_format}", **option_saving)
        logging.info(
            f"Footprint file has been successfully created and stored at {output_filename}.{output_format}"
        )


def merge_predictions_with_footprints(
    predictions_file: str, footprints_file: str, max_distance: int = 5, projected_crs: int = 3857
):
    """
    Merge local damage predictions with Microsoft building footprints using nearest spatial join.

    Parameters:
    - predictions_file: str, path to the GeoJSON or shapefile with building damage predictions.
    - footprints_file: str, path to the Microsoft building footprints.
    - max_distance: float, maximum distance in meters to search for the nearest match (default: 5 meters).
    - projected_crs: int, projected CRS to better handle local distance assessment (default: 3857 Mercator)

    Returns:
    - GeoDataFrame with merged class and geometry, including area features.
    """
    # Load and reproject to GPS system
    logging.info("Loading data...")
    predictions = gpd.read_file(predictions_file).to_crs(epsg=4326)
    footprints = gpd.read_file(footprints_file).to_crs(epsg=4326)

    logging.info(f"Number of predictions: {len(predictions)}")
    logging.info(f"Number of footprints: {len(footprints)}")

    # Project to EPSG:3857 for accurate spatial operations (meters)
    predictions = predictions.to_crs(epsg=projected_crs)
    footprints = footprints.to_crs(epsg=projected_crs)

    # Spatial join using nearest match
    # Nearest match is useful in post-catastrophe settings where building geometry might not align exactly
    logging.info("Performing nearest spatial join...")
    merged_data = gpd.sjoin_nearest(
        footprints, predictions, how="left", max_distance=max_distance, distance_col="distance"
    )

    # Assign class = 0 if no prediction was found
    if "class" not in merged_data.columns:
        logging.warning("'class' column not found in predictions. Defaulting all to 0.")
        merged_data["class"] = 0
    else:
        merged_data["class"] = merged_data["class"].fillna(0).astype(int)

    # Add building area in square meters
    merged_data["area_m2"] = merged_data.geometry.area

    # Keep only relevant columns
    merged_data = merged_data[["class", "geometry", "area_m2"]].copy()

    # Reproject back to EPSG:4326 for mapping
    merged_data = merged_data.to_crs(epsg=4326)

    logging.info(f"Number of merged footprints: {len(merged_data)}")
    return merged_data
