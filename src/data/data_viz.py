import logging
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Tuple

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
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


def display_tiles_by_indices(
    left_top: Tuple[int, int],
    right_bottom: Tuple[int, int],
    pre_image_folder_path: str = None,
    post_image_folder_path: str = None,
    pred_folder_path: str = None,
    image_size: Tuple[int, int] = (256, 256),
    mode: str = "pre",
):
    """
    Efficiently concatenate tiles into a single image using only (i, j) indices.
    Handles optional post and prediction overlay with RGBA blending.

    Example:
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

    def filter_list(path: str):
        i_str, j_str = extract_title_ij(path)
        i, j = int(i_str), int(j_str)
        return i_min <= i <= i_max and j_min <= j <= j_max

    full_image = np.zeros((n_rows * h, n_cols * w, 3), dtype=np.uint8)

    if not os.path.exists(pre_image_folder_path):
        raise FileNotFoundError(f"Pre-image folder not found: {pre_image_folder_path}")

    file_list = list(filter(lambda path: filter_list(path), os.listdir(pre_image_folder_path)))
    logger.info(f"Visualize {len(file_list)} images")

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

        # --- Base image ---
        base_img = np.zeros((h, w, 3), dtype=np.uint8)
        if mode == "pre":
            img_path = os.path.join(pre_image_folder_path, each_file)
        elif mode == "post":
            img_path = os.path.join(post_image_folder_path, each_file)
        else:
            raise ValueError("Choose a valid mode: 'pre' or 'post'.")

        try:
            base_img, _, _ = read_tiff_rasterio(img_path, bands=[1, 2, 3], with_profile=False)
        except Exception as e:
            logger.error(f"Failed to read {img_path}: {e}")
            continue

        # --- Overlay mask (RGBA TIF) ---
        if pred_folder_path:
            pred_path = os.path.join(pred_folder_path, each_file)
            if os.path.exists(pred_path):
                try:
                    pred_img, _, _ = read_tiff_rasterio(
                        pred_path, bands=[1, 2, 3, 4], with_profile=False
                    )
                    pred_img_pil = Image.fromarray(pred_img).convert("RGBA")
                    base_img_pil = Image.fromarray(base_img).convert("RGBA")
                    # Blended base image and prediction mask
                    blended = Image.alpha_composite(base_img_pil, pred_img_pil)
                    blended = np.array(blended.convert("RGB"))  # remove A layer (useless)
                except Exception as e:
                    logger.error(f"Failed to overlay prediction: {e}")
                    blended = base_img
            else:
                blended = base_img
        else:
            blended = base_img

        full_image[x1:x2, y1:y2, :] = blended

    logger.info(f"A {full_image.shape[:2]} image with overlaying predictions have been generated")
    # --- Display the result ---
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
    left_top: Tuple[float, float],
    right_bottom: Tuple[float, float],
    output_html: str = "damage_map.html",
):
    """
    Creates an interactive damage map using folium, filtered by a bounding box.

    Parameters:
    - geojson_path: str, path to the GeoJSON file
    - left_top: (lon, lat) of the top-left corner of the bounding box
    - right_bottom: (lon, lat) of the bottom-right corner of the bounding box
    - output_html: str, name of the output HTML file

    Example:
        create_folium_damage_map(
            geojson_path = "../outputs/classified_building_footprints.geojson",
            left_top=(-66.195, 18.33),
            right_bottom=(-66.19, 18.32),
            output_html="damage_map.html"
        )
    """

    print("Loading GeoJSON...")
    gdf = gpd.read_file(geojson_path)

    # Ensure data is in WGS84
    gdf = gdf.to_crs("EPSG:4326")

    # Optional: Drop any previous centroid columns if they exist
    gdf.drop(columns=[c for c in ["centroid_x", "centroid_y"] if c in gdf.columns], inplace=True)

    # Define bounding box
    minx, maxy = left_top
    maxx, miny = right_bottom
    bbox = box(minx, miny, maxx, maxy)

    # Filter by bbox
    gdf = gdf[gdf.intersects(bbox)]
    print(f"Filtered to {len(gdf)} features within bounding box.")

    # Check class column
    assert "class" in gdf.columns, "'class' column not found in the GeoJSON."

    # Compute centroids safely using projected CRS
    gdf_projected = gdf.to_crs(epsg=3857)
    centroids = gdf_projected.centroid.to_crs(epsg=4326)
    gdf["centroid_lat"] = centroids.y
    gdf["centroid_lon"] = centroids.x

    # Map class labels
    gdf["class_label"] = gdf["class"].astype(str).map(CLASS_MAPPING)

    # Prepare HeatMap data
    heat_data = gdf[["centroid_lat", "centroid_lon"]].dropna().values.tolist()

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
    def style_function(feature):
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
            fields=["class_label", "area_m2"], aliases=["Damage", "Area (m²)"]
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
    print(f"Interactive map saved to {output_html}")
