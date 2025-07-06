# Visualization utils functions
from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from src.data import renormalize_image

DEFAULT_MAPPING = {
    0: (255, 255, 255, 0),
    1: (0, 255, 0, 255),  # Green
    2: (0, 0, 255, 255),  # Blue
    3: (255, 69, 0, 255),  # Red-Green
    4: (255, 0, 0, 255),  # Red
}

CLOUD_MAPPING = {
    0: (255, 255, 255, 0),  # White (No-Cloud)
    1: (127, 0, 255, 255),  # Purple (Cloud)
}

BUILDING_MAPPING = {
    0: (255, 255, 255, 0),  # White (No-Building)
    1: (0, 0, 255, 255),  # Blue (Building)
}

DAMAGE_MAPPING = {
    0: (255, 255, 255, 0),  # White (Background)
    1: (0, 255, 0, 255),  # Green (No-damaged)
    2: (255, 255, 0, 255),  # Yellow (Minor-damaged)
    3: (255, 69, 0, 255),  # Orange (major-damaged)
    4: (255, 0, 0, 255),  # Red (destroyed)
}

COLOR_DICT = {
    "default": DEFAULT_MAPPING,
    "cloud": CLOUD_MAPPING,
    "building": BUILDING_MAPPING,
    "damage": DAMAGE_MAPPING,
}

RGB_CHANNEL_SIZE = 3
RGBA_CHANNEL_SIZE = 4


# Function to apply colors to masks (labels and predictions)
def apply_color_map(
    mask: torch.Tensor | np.ndarray,
    color_dict: dict[int, tuple] = DEFAULT_MAPPING,
    *,
    with_transparency: bool = False,
) -> Tensor:
    """Applies a color map (color_dict) to a mask tensor.

    Args:
    ----
        mask (tensor): Mask tensor (predictions or labels). shape (N, H, W)
        color_dict (dict): A dictionary mapping class labels to RGB colors.
        with_transparency (bool) : Add a transparency channel (4th one) or not

    Returns:
    -------
        color_mask (tensor): The colored mask tensor.

    """
    batch_size, height, width = mask.shape
    nb_channels = 3
    if with_transparency:
        nb_channels = RGBA_CHANNEL_SIZE
        color_mask = torch.zeros(
            (batch_size, RGBA_CHANNEL_SIZE, height, width), dtype=torch.float32
        )
    else:
        color_mask = torch.zeros((batch_size, 3, height, width), dtype=torch.float32)

    # Apply the color mapping for each class label in the mask

    for label, color in color_dict.items():
        binary_mask = (mask == label).float()
        # Apply the color to the binary mask
        for c in range(nb_channels):
            color_mask[:, c, :, :] += binary_mask * (color[c] / 255.0)

    return color_mask


def apply_inverse_color_map(
    color_mask: np.ndarray,
    color_dict: dict[int, tuple] = DEFAULT_MAPPING,
) -> np.ndarray:
    """Converts a colorized mask (RGB or RGBA) back into a label mask.

    Args:
    ----
        color_mask (np.ndarray): The colorized mask, shape (H, W, C) where C is 3 (RGB) or 4 (RGBA).
        color_dict (dict): A dictionary mapping class labels to RGB(A) colors.

    Returns:
    -------
        np.ndarray: The label mask, shape (H, W), where each pixel contains the class label.

    """
    # Ensure the input is a NumPy array
    if isinstance(color_mask, torch.Tensor):
        color_mask = color_mask.cpu().numpy()

    # Remove the alpha channel if present
    if color_mask.shape[-1] == RGBA_CHANNEL_SIZE:  # RGBA
        color_mask = color_mask[..., :3]  # Keep only RGB

    # Initialize the label mask
    height, width, _ = color_mask.shape
    label_mask = np.zeros((height, width), dtype=np.int32)

    # Iterate through the color dictionary and map colors to labels
    for label, color in color_dict.items():
        # Convert the color to a NumPy array for comparison
        color_array = np.array(color[:3])  # Use only RGB values
        # Create a binary mask where the color matches
        match = np.all(color_mask == color_array, axis=-1)
        # Assign the label to the corresponding pixels
        label_mask[match] = label

    return label_mask


def add_image_transparency(mask: np.ndarray) -> np.ndarray:
    """Add a Transparency Layer to Images, making white pixels transparent from RGB to RGBA."""
    # Initialize with zeros, then copy RGB from mask and set alpha appropriately
    new_mask = np.zeros((mask.shape[0], mask.shape[1], RGBA_CHANNEL_SIZE), dtype=np.uint8)
    # Copy RGB channels
    new_mask[..., :3] = mask
    # Create transparency mask
    white_pixels = np.all(mask == [255, 255, 255], axis=-1)
    # Set alpha to 0 (transparent) where white, 255 (opaque) elsewhere
    new_mask[..., 3] = np.where(white_pixels, 0, 255)
    return new_mask


def make_background_transparent(mask: np.ndarray) -> np.ndarray:
    """Create a Matplotlib Alpha to make background white-pixels transparent."""
    # Create a transparent alpha channel (initially all 255, meaning fully opaque)
    alpha_mask = np.ones((mask.shape[0], mask.shape[1], RGBA_CHANNEL_SIZE), dtype=np.uint8)

    # Identify where the pixels are white (i.e., [255, 255, 255])
    white_pixels = np.all(mask == [255, 255, 255], axis=-1)

    # Set the alpha channel to 0 for white pixels (making them transparent)
    alpha_mask[white_pixels] = [0, 0, 0, 0]  # RGBA with 0 alpha (transparent)

    # The rest of the image keeps its alpha at 255 (fully opaque)
    alpha_mask[~white_pixels, 3] = 255  # Set alpha to 255 for non-white pixels

    return alpha_mask


def display_semantic_predictions_batch(
    images: Tensor | np.ndarray,
    mask_predictions: Tensor | np.ndarray,
    mask_labels: Tensor | np.ndarray,
    normalized: dict | None = None,
    folder_path: str | None = None,
) -> None:
    """Display a batch of images alongside their predicted masks and ground truth masks and optionally saves the images in a folder.

    Args:
    ----
        images (torch.Tensor or numpy.ndarray): Batch of input images, shape (N, C, H, W) or (N, H, W, C).
        mask_predictions (torch.Tensor or numpy.ndarray): Batch of predicted masks, shape (N, H, W) or (N, H, W, C).
        mask_labels (torch.Tensor or numpy.ndarray): Batch of ground truth masks, shape (N, H, W) or (N, H, W, C).
        normalized (dict): Dictionary containing "mean" and "std" for unnormalization.
        folder_path (str): Path to the folder where images will be saved. If None, images are not saved.

    """
    if normalized is None:
        bool_normalized = False
    else:
        bool_normalized = True
        mean = normalized.get("mean")
        std = normalized.get("std")

    # Convert tensors to numpy arrays if needed
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    if isinstance(mask_predictions, torch.Tensor):
        mask_predictions = mask_predictions.detach().cpu().numpy()
    if isinstance(mask_labels, torch.Tensor):
        mask_labels = mask_labels.detach().cpu().numpy()

    batch_size = images.shape[0]  # Number of images in the batch

    # Create the folder if folder_path is provided
    if folder_path is not None:
        Path.mkdir(folder_path, exist_ok=True)

    for i in range(batch_size):
        image = images[i]
        mask_prediction = mask_predictions[i]
        mask_label = mask_labels[i]

        # Handle channel-first images (C, H, W) and unnormalize
        if image.ndim == RGB_CHANNEL_SIZE and image.shape[0] == RGB_CHANNEL_SIZE:  # RGB images
            image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, C)
            image = (
                renormalize_image(image, std, mean) if bool_normalized else image
            )  # Unnormalize and clip to [0, 1]
        elif image.ndim == RGB_CHANNEL_SIZE and image.shape[0] == 1:  # Grayscale
            image = image[0]  # Remove channel dimension

        # Create the plot
        plt.figure(figsize=(12, 4))

        # Show the input image
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title("Input Image")

        # Show the predicted mask
        plt.subplot(1, 3, 2)
        plt.imshow(image, alpha=0.5)
        plt.imshow(mask_prediction, cmap="jet", interpolation="none", alpha=0.5)
        plt.axis("off")
        plt.title("Predicted Mask")

        # Show the ground truth mask
        plt.subplot(1, 3, 3)
        plt.imshow(image, alpha=0.5)
        plt.imshow(mask_label, cmap="jet", interpolation="none", alpha=0.5)
        plt.axis("off")
        plt.title("Ground Truth Mask")

        plt.tight_layout()

        # Save the plot if folder_path is provided
        if folder_path is not None:
            file_path = Path(folder_path) / f"sample_{i + 1}.png"
            plt.savefig(file_path)
            logging.info("Saved: %s", file_path)

        plt.show()


def display_instance_predictions_batch(
    images: Tensor,
    mask_predictions: list[dict],
    score_threshold: float = 0.6,
    max_images: int = 5,
    display: list[str] | None = None,
) -> None:
    """Display a batch of images with their instance segmentation predictions, including bounding boxes and/or masks.

    Args:
    ----
        images (Tensor): Batch of input images, shape (N, C, H, W).
        mask_predictions (list[dict]): List of prediction dicts with keys "boxes", "labels", "scores", "masks".
        score_threshold (float): Minimum score for a prediction to be displayed.
        max_images (int): Maximum number of images to display.
        display (list[str] | None): List containing "boxes" and/or "masks" to specify what to display.
    """
    if display is None:
        logging.info(
            "Default Display mode is set : visualizing bounding boxes and semantic masks."
        )
        display = ["boxes", "masks"]

    # Loop through the data loader (for visualization, we can stop after a few images)

    # Make predictions
    predictions = mask_predictions[
        :max_images
    ]  # Predictions are a list of dicts with "boxes", "labels", "scores", "masks"

    for i in range(len(predictions)):
        pred = predictions[i]
        image = images[i]

        # Normalize and convert to uint8
        output_image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(
            torch.uint8,
        )

        # Filter out low-confidence predictions
        mask = pred["scores"] > score_threshold
        pred_boxes = pred["boxes"][mask]
        pred_labels = [f"{score:.3f}" for score in pred["scores"][mask]]
        pred_masks = pred["masks"][mask]

        # Draw bounding boxes
        if "boxes" in display:
            output_image = draw_bounding_boxes(
                output_image,
                pred_boxes.long(),
                labels=pred_labels,
                colors="red",
            )

        # Draw segmentation masks
        if "masks" in display and pred_masks.numel() > 0:  # Ensure there are masks to draw
            masks = (pred_masks > 0.5).squeeze(1)  # Binarize the masks  # noqa: PLR2004
            output_image = draw_segmentation_masks(
                output_image,
                masks,
                alpha=0.5,
                colors="blue",
            )

        # Move the output image to the CPU and convert to NumPy array for plotting
        output_image = output_image.cpu()

        # Plot the image
        plt.figure(figsize=(12, 4))
        plt.imshow(output_image.permute(1, 2, 0))
        plt.axis("off")
        plt.show()


def display_semantic_is_siamese_predictions_batch(
    pre_images: Tensor | np.ndarray,
    post_images: Tensor | np.ndarray,
    mask_predictions: Tensor | np.ndarray,
    mask_labels: Tensor | np.ndarray,
    normalized: dict | None = None,
    folder_path: str | None = None,
) -> None:
    """Displays a batch of pre-event and post-event images alongside their predicted masks and ground truth masks, and optionally saves the images in a folder.

    Args:
    ----
        pre_images (torch.Tensor or numpy.ndarray): Batch of pre-event images, shape (N, C, H, W) or (N, H, W, C).
        post_images (torch.Tensor or numpy.ndarray): Batch of post-event images, shape (N, C, H, W) or (N, H, W, C).
        mask_predictions (torch.Tensor or numpy.ndarray): Batch of predicted masks, shape (N, H, W) or (N, H, W, C).
        mask_labels (torch.Tensor or numpy.ndarray): Batch of ground truth masks, shape (N, H, W) or (N, H, W, C).
        normalized (dict): Dictionary containing "mean" and "std" for unnormalization.
        folder_path (str): Path to the folder where images will be saved. If None, images are not saved.

    """
    # Determine normalization parameters
    bool_normalized = normalized is not None
    mean, std = (
        normalized.get("mean"),
        normalized.get("std")
        if bool_normalized
        else (
            0,
            1,
        ),
    )

    # Convert tensors to numpy arrays if needed
    if isinstance(pre_images, torch.Tensor):
        pre_images = pre_images.detach().cpu().numpy()
        post_images = post_images.detach().cpu().numpy()
    if isinstance(mask_predictions, torch.Tensor):
        mask_predictions = mask_predictions.detach().cpu().numpy()
    if isinstance(mask_labels, torch.Tensor):
        mask_labels = mask_labels.detach().cpu().numpy()

    batch_size = pre_images.shape[0]  # Number of samples in the batch

    # Create the output folder if folder_path is provided
    if folder_path is not None:
        Path.mkdir(folder_path, exist_ok=True)

    for i in range(batch_size):
        pre_image = pre_images[i]
        post_image = post_images[i]
        mask_prediction = mask_predictions[i]
        mask_label = mask_labels[i]

        # Handle channel-first images (C, H, W) and unnormalize if needed
        def process_image(image: Tensor | np.ndarray) -> Tensor | np.ndarray:
            if image.ndim == RGB_CHANNEL_SIZE and image.shape[0] == RGB_CHANNEL_SIZE:  # RGB
                image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, C)
                return renormalize_image(image, mean, std) if bool_normalized else image
            if image.ndim == RGB_CHANNEL_SIZE and image.shape[0] == 1:  # Grayscale
                return image[0]  # Remove channel dimension
            return image

        pre_image = process_image(pre_image)
        post_image = process_image(post_image)

        # Create the plot
        plt.figure(figsize=(16, 8))

        # Display pre-event image
        plt.subplot(1, 4, 1)
        plt.imshow(pre_image)
        plt.axis("off")
        plt.title("Pre-Event Image")

        # Display post-event image
        plt.subplot(1, 4, 2)
        plt.imshow(post_image)
        plt.axis("off")
        plt.title("Post-Event Image")

        # Show the predicted mask
        plt.subplot(1, 4, 3)
        plt.imshow(mask_prediction, cmap="jet", interpolation="none")
        plt.axis("off")
        plt.title("Predicted Mask")

        # Show the ground truth mask
        plt.subplot(1, 4, 4)
        plt.imshow(mask_label, cmap="jet", interpolation="none")
        plt.axis("off")
        plt.title("Ground Truth Mask")

        plt.tight_layout()

        # Save the plot if folder_path is provided
        if folder_path is not None:
            file_path = Path(folder_path) / f"sample_{i + 1}.png"
            plt.savefig(file_path)
            logging.info("Saved: %s", file_path)

        plt.show()
