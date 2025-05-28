# Utils files for Dataset definitions
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

__all__ = [
    "custom_collate_fn",
    "IMG_EXTENSIONS",
    "is_image_file",
    "split_and_save_images",
]

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    "tif",
    "tiff",
]


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized targets in a batch.
    Required for Instance Segmentation Dataloader

    Parameters:
        batch (list): List of (image, target) tuples.

    Returns:
        Tuple: (images, targets)
    """
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

    # Stack images into a single batch tensor
    images = torch.stack(images, dim=0)

    # Return images and list of targets
    return images, targets


def is_image_file(filepath):
    """Check if a Path object or string has an image file extension."""
    return filepath.suffix.lower() in IMG_EXTENSIONS


def split_and_save_images(
    input_dir: str,
    output_dir: str,
    patch_size: int = 256,
    images_folder_names: list[str] = ["A", "B"],
    label_folder_name: list[str] = "label",
):
    """
    Splits all images in the input directory into non-overlapping patches and saves them to the output directory.

    Args:
        input_dir (str): Path to the root directory containing 'A', 'B', and 'label' folders.
        output_dir (str): Path to the directory where patches will be saved.
        patch_size (int): Size of the patches (default: 256).
        image_format (str): Format to save patches, e.g., 'png' or 'jpg'.
        folder_names (list): List of folder names to process (default: ['A', 'B', 'label']).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    folder_names = images_folder_names + [label_folder_name]
    image_counter = 0
    crop_counter = 0

    # Create output directories for patches
    for folder in folder_names:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    for folder in folder_names:
        input_folder = input_dir / folder
        output_folder = output_dir / folder

        # Process each image in the folder
        for img_path in tqdm(input_folder.glob("*.*")):
            if is_image_file(img_path):
                image_counter += 1
                img = Image.open(img_path)
                img = img.convert("RGB") if folder != label_folder_name else img.convert("L")

                img_name = img_path.stem  # Get the image name without extension
                img_ext = img_path.suffix
                width, height = img.size

                # Split into patches
                for y in range(0, height, patch_size):
                    for x in range(0, width, patch_size):
                        patch = img.crop((x, y, x + patch_size, y + patch_size))

                        # Ensure patches are exactly patch_size x patch_size
                        if patch.size != (patch_size, patch_size):
                            continue

                        # Save patch
                        patch_filename = f"{img_name}_{y}_{x}.{img_ext}"
                        patch.save(output_folder / patch_filename)
                        crop_counter += 1

    print(
        f"{image_counter} images ({width}, {height}) have been created into {crop_counter} patches of size ({patch_size}, {patch_size}) and saved at {output_dir}"
    )
