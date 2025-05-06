import os
from typing import Tuple, List
import logging
import re
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import folium
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


from src.utils.visualization import add_image_transparency, make_background_transparent

from src.data.utils import read_tiff_rasterio, extract_title_ij, extract_coordinates_parallel, filter_files_by_bounds

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


def display_tiles_by_coordinates_folium(
    left_top: Tuple[float, float],
    right_bottom: Tuple[float, float],
    pre_image_folder_path: str = None,
    post_image_folder_path: str = None,
    pred_folder_path: str = None,
):
    """
    Display pre/post-event images and prediction overlays within given geospatial coordinates.
    """
    lat_max, lon_min = left_top  # top-left corner (latitude, longitude)
    lat_min, lon_max = right_bottom  # bottom-right corner (latitude, longitude)

    lat = (lat_max + lat_min) / 2
    long = (lon_max + lon_min) / 2

    m = folium.Map(location=[lat, long], zoom_start=20)

    pre_group = folium.FeatureGroup(name="Pre-Event Images", show=False)
    post_group = folium.FeatureGroup(name="Post-Event Images", show=False)
    mask_group = folium.FeatureGroup(name="Predictions", show=False)

    # Create a Rectangle to localize the Images
    ls = folium.Rectangle(
        bounds=[[lat_max, lon_min], [lat_min, lon_max]],
        color="red",
        line_join="mitter",
        dash_array="5, 10",
    )
    ls.add_to(m)

    file_list = os.listdir(pre_image_folder_path)

    coordinates = extract_coordinates_parallel(filepaths=file_list, folder_path=pre_image_folder_path, max_workers=8)
    file_list, coordinates = filter_files_by_bounds(
        filepaths=file_list, coordinates=coordinates, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max
    )

    for each_file, extent in tqdm(zip(file_list, coordinates), desc="Processing Images", total=len(file_list)):
        if each_file.lower().endswith(".tif"):
            pre_path = os.path.join(pre_image_folder_path, each_file)

            left, bottom, right, top = extent  # long_min, lat_min, long_max, lat_max
            # PRE-EVENT IMAGE
            pre_img, _ = read_tiff_rasterio(pre_path)
            # ax.imshow(pre_img, extent=(left, right, bottom, top), alpha=1.0)
            folium.raster_layers.ImageOverlay(
                bounds=[[bottom, left], [top, right]],  # folium latitude first
                image=pre_img,
                opacity=1.0,
                control=True,
                tooltip=f"Pre: {each_file}",
                interactive=True,
                overlay=True,
                zindex=1,
            ).add_to(pre_group)

            # Post-event image
            post_path = os.path.join(post_image_folder_path, each_file)
            if os.path.exists(post_path):
                post_img, _ = read_tiff_rasterio(post_path)
                folium.raster_layers.ImageOverlay(
                    bounds=[[bottom, left], [top, right]],
                    image=post_img,
                    opacity=1.0,
                    interactive=True,
                    tooltip=f"Post: {each_file}",
                    control=True,
                    overlay=True,
                    zindex=1,
                ).add_to(post_group)
                # ax.imshow(post_img, extent=(left, right, bottom, top), alpha=0.3)

            # Prediction mask
            pred_path = os.path.join(pred_folder_path, each_file.replace(".tif", ".png"))
            if os.path.exists(pred_path):
                pred_img = np.array(Image.open(pred_path).convert("RGBA"))
                pred_img = add_image_transparency(mask=pred_img)
                folium.raster_layers.ImageOverlay(
                    bounds=[[bottom, left], [top, right]],
                    image=pred_img,
                    opacity=1.0,
                    tooltip=f"Pred: {each_file}",
                    control=True,
                    overlay=True,
                ).add_to(mask_group)

    pre_group.add_to(m)
    post_group.add_to(m)
    mask_group.add_to(m)
    folium.LayerControl().add_to(m)
    m.save("tiles_map.html")
    logging.info("Map has been saved as tiles_map.html ✅")


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
            base_img, _ = read_tiff_rasterio(img_path, bands=[1, 2, 3], with_profile=False)
        except Exception as e:
            logger.error(f"Failed to read {img_path}: {e}")
            continue

        # --- Overlay mask (RGBA TIF) ---
        if pred_folder_path:
            pred_path = os.path.join(pred_folder_path, each_file)
            if os.path.exists(pred_path):
                try:
                    pred_img, _ = read_tiff_rasterio(pred_path, bands=[1, 2, 3, 4], with_profile=False)
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

    # --- Display the result ---
    plt.figure(figsize=(10, 10), dpi=100)
    plt.imshow(full_image)
    plt.title("Concatenated Tiles with Overlay")
    plt.axis("off")
    plt.show()

    return full_image


base_path = Path("./data/processed_data")
pre_image_folder = base_path / "Pre_Event_San_Juan"
post_image_folder = base_path / "Post_Event_San_Juan"
pred_folder = "./data/predictions"

display_tiles_by_indices(
    left_top=(20, 20),
    right_bottom=(50, 50),
    pre_image_folder_path=pre_image_folder,
    post_image_folder_path=post_image_folder,
    pred_folder_path=pred_folder,
    image_size=(256, 256),
    mode="post",
)

"""
# Example with coordinates and folium maps
tiles_by_coordinates_folium(
    left_top=(18.33, -66.195),       # latitude, longitude (top-left corner)
    right_bottom=(18.32, -66.19),   # latitude, longitude (bottom-right corner)
    pre_image_folder_path=pre_image_folder,
    post_image_folder_path=post_image_folder,
    pred_folder_path=pred_folder
)
"""
