from __future__ import annotations

import logging
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import branca.colormap as cm
import dash
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
from folium.plugins import HeatMap
from mpl_toolkits.basemap import Basemap
from PIL import Image
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import box, shape
from tqdm import tqdm

from src.data.utils import (
    extract_coordinates_parallel,
    extract_title_ij,
    filter_files_by_bounds,
    read_tiff_rasterio,
)
from src.utils.visualization import (
    COLOR_DICT,
    add_image_transparency,
    apply_inverse_color_map,
)

CLASS_MAPPING = {
    "0": "no-damage",
    "1": "minor-damage",
    "2": "major-damage",
    "3": "destroyed",
    "4": "not classified",
}

COLOR_MAPPING = {
    "no-damage": "#2ECC71",
    "minor-damage": "#F1C40F",
    "major-damage": "#E67E22",
    "destroyed": "#E74C3C",
    "not classified": "#95A5A6",
}

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.ERROR)


def filter_tile_list(path: str, i_min: int, i_max: int, j_min: int, j_max: int) -> bool:
    """Filter a tile list based on max and min index value."""
    i_str, j_str = extract_title_ij(path)
    i, j = int(i_str), int(j_str)
    return i_min <= i <= i_max and j_min <= j <= j_max


def get_base_img(
    each_file: str,
    mode: str,
    pre_image_folder_path: str,
    post_image_folder_path: str,
    h: int,
    w: int,
) -> np.ndarray:
    """Load a base image tile in either 'pre' or 'post' mode from the specified folder paths.

    Parameters
    ----------
    each_file : str
        The filename of the image tile.
    mode : str
        Either 'pre' or 'post' to specify which folder to load from.
    pre_image_folder_path : str
        Path to the folder containing pre-event images.
    post_image_folder_path : str
        Path to the folder containing post-event images.
    h : int
        Height of the image tile.
    w : int
        Width of the image tile.

    Returns:
    -------
    np.ndarray
        The loaded image as a NumPy array, or a zero array if loading fails.

    """
    if mode == "pre":
        img_path = Path(pre_image_folder_path) / each_file
    elif mode == "post":
        img_path = Path(post_image_folder_path) / each_file
    else:
        raise ValueError("Choose a valid mode: 'pre' or 'post'.")
    try:
        base_img, _, _ = read_tiff_rasterio(img_path, bands=[1, 2, 3], with_profile=False)
    except Exception as e:
        _logger.exception("Failed to read %s", img_path)
        return np.zeros((h, w, 3), dtype=np.uint8)
    else:
        return base_img


def overlay_prediction(base_img: np.ndarray, each_file: str, pred_folder_path: str) -> np.ndarray:
    """Overlay a prediction image onto a base image if the prediction file exists.

    Parameters
    ----------
    base_img : np.ndarray
        The base image as a NumPy array.
    each_file : str
        The filename of the image tile.
    pred_folder_path : str
        Path to the folder containing prediction images.

    Returns:
    -------
    np.ndarray
        The resulting image with the prediction overlay, or the base image if overlay fails or does not exist.

    """
    if not pred_folder_path:
        return base_img
    pred_path = Path(pred_folder_path) / each_file
    if Path(pred_path).exists():
        try:
            pred_img, _, _ = read_tiff_rasterio(
                pred_path,
                bands=[1, 2, 3, 4],
                with_profile=False,
            )
            pred_img_pil = Image.fromarray(pred_img).convert("RGBA")
            base_img_pil = Image.fromarray(base_img).convert("RGBA")
            blended = Image.alpha_composite(base_img_pil, pred_img_pil)
            return np.array(blended.convert("RGB"))
        except Exception as e:
            _logger.exception("Failed to overlay prediction")
            return base_img
    else:
        return base_img


def process_and_fill_image(
    file_list: list,
    i_min: int,
    j_min: int,
    h: int,
    w: int,
    n_rows: int,
    n_cols: int,
    mode: str,
    pre_image_folder_path: str,
    post_image_folder_path: str,
    pred_folder_path: str,
) -> np.ndarray:
    """Process a list of image tiles, overlay predictions if available, and fill a full image grid.

    Parameters
    ----------
    file_list : list
        List of image tile filenames.
    i_min, j_min : int
        Minimum i and j indices for tile placement.
    h, w : int
        Height and width of each tile.
    n_rows, n_cols : int
        Number of rows and columns in the output image grid.
    mode : str
        Mode for selecting base images ('pre' or 'post').
    pre_image_folder_path, post_image_folder_path : str
        Paths to pre- and post-event image folders.
    pred_folder_path : str
        Path to prediction images folder.

    Returns:
    -------
    np.ndarray
        The concatenated image grid as a NumPy array.

    """
    full_image = np.zeros((n_rows * h, n_cols * w, 3), dtype=np.uint8)
    for each_file in tqdm(file_list, total=len(file_list), desc="Image Processing"):
        if not each_file.lower().endswith(".tif"):
            continue

        i_str, j_str = extract_title_ij(each_file)
        if i_str is None or j_str is None:
            continue

        i, j = int(i_str), int(j_str)
        row = abs(i - i_min)
        col = abs(j - j_min)

        y1, y2 = row * h, (row + 1) * h
        x1, x2 = col * w, (col + 1) * w

        base_img = get_base_img(
            each_file, mode, pre_image_folder_path, post_image_folder_path, h, w
        )
        blended = overlay_prediction(base_img, each_file, pred_folder_path)
        full_image[x1:x2, y1:y2, :] = blended
    return full_image


def display_tiles_by_indices(
    left_top: tuple[int, int],
    right_bottom: tuple[int, int],
    pre_image_folder_path: str | None = None,
    post_image_folder_path: str | None = None,
    pred_folder_path: str | None = None,
    image_size: tuple[int, int] = (256, 256),
    mode: str = "pre",
) -> np.ndarray:
    """Efficiently concatenate tiles into a single image using only (i, j) indices.

    Handles optional post and prediction overlay with RGBA blending.

    Example:
    -------
    display_tiles_by_indices(
        left_top=(50, 50),
        right_bottom=(100, 100),
        pre_image_folder_path=pre_image_folder,
        post_image_folder_path=post_image_folder,
        pred_folder_path=pred_folder,
        image_size=(256, 256),
        mode="post",
    )

    """
    i_min, j_min = left_top
    i_max, j_max = right_bottom
    h, w = image_size

    n_rows = abs(i_max - i_min) + 1
    n_cols = abs(j_max - j_min) + 1

    if not Path.exists(pre_image_folder_path):
        raise FileNotFoundError("Pre-image folder not found: %s", pre_image_folder_path)

    file_list = list(
        filter(
            lambda path: filter_tile_list(path, i_min, i_max, j_min, j_max),
            os.listdir(pre_image_folder_path),
        ),
    )
    _logger.info("Visualize %d images", len(file_list))

    full_image = process_and_fill_image(
        file_list,
        i_min,
        j_min,
        h,
        w,
        n_rows,
        n_cols,
        mode,
        pre_image_folder_path,
        post_image_folder_path,
        pred_folder_path,
    )

    _logger.info(
        "A %d image with overlaying predictions have been generated", full_image.shape[:2]
    )
    plt.figure(figsize=(10, 10), dpi=100)
    plt.imshow(full_image)
    plt.title(f"Concatenated Tiles with {mode}-disaster Overlay")
    plt.axis("off")
    plt.show()

    return full_image


"""
if __name__ == '__main__':

    base_path = Path("./data/processed_data")
    pre_image_folder = base_path / "Pre_Event_San_Juan"
    post_image_folder = base_path / "Post_Event_San_Juan"
    pred_folder = Path("./outputs/predictions_damages")

    display_tiles_by_indices(
            left_top=(100, 100),
            right_bottom=(150, 150),
            pre_image_folder_path=pre_image_folder,
            post_image_folder_path=post_image_folder,
            pred_folder_path=pred_folder,
            image_size=(256, 256),
            mode="post",
        )
"""


# Display Predictions in a given location with folium
def create_folium_damage_map(
    geojson_path: str,
    left_top: tuple[float, float],
    right_bottom: tuple[float, float],
    output_html: str = "damage_map.html",
) -> None:
    """Create an interactive damage map using folium, filtered by a bounding box.

    Parameters
    ----------
    geojson_path : str
        Path to the GeoJSON file.
    left_top : tuple[float, float]
        (lon, lat) of the top-left corner of the bounding box.
    right_bottom : tuple[float, float]
        (lon, lat) of the bottom-right corner of the bounding box.
    output_html : str, optional
        Name of the output HTML file (default is "damage_map.html").

    Example:
        create_folium_damage_map(
            geojson_path = "../outputs/classified_building_footprints.geojson",
            left_top=(-66.195, 18.33),
            right_bottom=(-66.19, 18.32),
            output_html="damage_map.html"
        )

    """
    _logger.info("Loading GeoJSON...")
    gdf = gpd.read_file(geojson_path)

    # Ensure data is in WGS84
    gdf = gdf.to_crs("EPSG:4326")

    # Optional: Drop any previous centroid columns if they exist
    gdf.drop(columns=[c for c in ["centroid_x", "centroid_y"] if c in gdf.columns])

    # Define bounding box
    minx, maxy = left_top
    maxx, miny = right_bottom
    bbox = box(minx, miny, maxx, maxy)

    # Filter by bbox
    gdf = gdf[gdf.intersects(bbox)]
    _logger.info("Filtered to %d features within bounding box.", len(gdf))

    # Check class column
    if "class" in gdf.columns:
        new_var = "'class' column not found in the GeoJSON."
        raise (new_var)

    # Compute centroids safely using projected CRS
    gdf_projected = gdf.to_crs(epsg=3857)
    centroids = gdf_projected.centroid.to_crs(epsg=4326)
    gdf["centroid_lat"] = centroids.y
    gdf["centroid_lon"] = centroids.x

    # Map class labels
    gdf["class_label"] = gdf["class"].astype(str).map(CLASS_MAPPING)

    # Prepare HeatMap data
    heat_data = gdf[["centroid_lat", "centroid_lon"]].dropna().to_numpy().tolist()

    # Create map centered on mean coordinates
    center = [gdf["centroid_lat"].mean(), gdf["centroid_lon"].mean()]
    folium_map = folium.Map(location=center, zoom_start=15, tiles=None)

    # Add base layers
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(folium_map)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Satellite",
        overlay=False,
    ).add_to(folium_map)

    # Style function
    def style_function(feature: dict) -> dict[str, Any]:
        label = feature["properties"].get("class_label", "not classified")
        return {
            "fillColor": COLOR_MAPPING.get(label, "#CCCCCC"),
            "color": "black",
            "weight": 0.5,
            "fillOpacity": 0.7,
        }

    # Add choropleth layer
    choropleth_group = folium.FeatureGroup(name="Damage Classes")
    folium.GeoJson(
        gdf,
        name="Building Footprints",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["class_label", "area_m2"],
            aliases=["Damage", "Area (m²)"],
        ),
    ).add_to(choropleth_group)
    choropleth_group.add_to(folium_map)

    # Add heatmap
    heatmap_group = folium.FeatureGroup(name="Damage Density Heatmap")
    HeatMap(heat_data, radius=10, blur=15, max_zoom=18).add_to(heatmap_group)
    heatmap_group.add_to(folium_map)

    # Legend
    colormap = cm.StepColormap(
        colors=[COLOR_MAPPING[k] for k in CLASS_MAPPING.values()],
        vmin=0,
        vmax=4,
        index=[0, 1, 2, 3, 4, 5],
        caption="Damage Classification",
    )
    folium_map.add_child(colormap)

    # Controls
    folium.LayerControl().add_to(folium_map)

    # Save
    folium_map.save(output_html)
    logging.info("Interactive map saved to %s", output_html)
