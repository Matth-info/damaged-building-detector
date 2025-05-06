from tqdm import tqdm
from PIL import Image

import torch
from torchvision.transforms import v2
import albumentations as A

import numpy as np
import matplotlib.pyplot as plt


class Inference:
    def __init__(
        self,
        model: torch.nn.Module,
        pre_image: np.ndarray = None,
        post_image: np.ndarray = None,
        window_size: int = 512,
        stride: int = 100,
        device: str = "cuda",
        num_classes: int = 5,
        mode: str = "siamese",
        transform=None,
    ):
        """
        Initialize the Inference class.

        Args:
            model (torch.nn.Module): The trained semantic segmentation model.
            pre_image (Image): Pre-change image.
            post_image (Image): Post-change image.
            window_size (int): Size of the sliding window.
            num_classes (int): Number of classes
            stride (int): Stride for the sliding window.
            device (str): Device to run inference on ('cuda' or 'cpu').
            transform : Data Augmentation functions
        """
        assert mode in ["siamese", None]

        self.mode = mode
        self.model = model.to(device)
        self.pre_image = pre_image
        if self.mode == "siamese":
            self.post_image = post_image
        self.window_size = window_size
        self.stride = stride
        self.device = device
        self.num_classes = num_classes
        self.transform = transform

    def _sliding_window(self, image: np.ndarray):
        """
        Generator for sliding window patches.

        Args:
            image (np.ndarray): Input image as a NumPy array.

        Yields:
            tuple: (x, y, patch) where (x, y) is the top-left corner of the patch.
        """
        h, w, _ = image.shape
        for y in range(0, h - self.window_size + 1, self.stride):
            for x in range(0, w - self.window_size + 1, self.stride):
                patch = image[y : y + self.window_size, x : x + self.window_size]
                yield x, y, patch

    def _merge_patches(self, patches, image_shape):
        """
        Merge patches back into the full-size image.

        Args:
            patches (list): List of (x, y, patch_pred) tuples.
            image_shape (tuple): Shape of the original image (height, width).

        Returns:
            np.ndarray: Merged image with shape (5, height, width).
        """
        full_pred = np.zeros((self.num_classes, image_shape[0], image_shape[1]), dtype=np.float32)
        count_map = np.zeros((image_shape[0], image_shape[1]), dtype=np.float32)

        for x, y, patch_pred in patches:
            full_pred[:, y : y + self.window_size, x : x + self.window_size] += patch_pred
            count_map[y : y + self.window_size, x : x + self.window_size] += 1

        # Avoid division by zero
        count_map[count_map == 0] = 1
        for i in range(full_pred.shape[0]):
            full_pred[i] /= count_map

        return full_pred

    def infer(self):
        """
        Perform inference using the sliding window approach.

        Returns:
            np.ndarray: Full-size prediction with shape (5, height, width).
        """
        self.model.eval()

        pre_image_np = np.array(self.pre_image)
        if self.mode == "siamese":
            post_image_np = np.array(self.post_image)

        image_shape = pre_image_np.shape[:2]  # (height, width)
        patches = []

        with torch.no_grad():
            for x, y, patch in tqdm(self._sliding_window(pre_image_np)):
                pre_patch = self.transform(Image.fromarray(patch)).unsqueeze(0).to(self.device)
                if self.mode == "siamese":
                    post_patch = (
                        self.transform(
                            Image.fromarray(post_image_np[y : y + self.window_size, x : x + self.window_size])
                        )
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    patch_pred = self.model(x1=pre_patch, x2=post_patch).squeeze(0).cpu().numpy()
                else:
                    patch_pred = self.model(pre_patch).squeeze(0).cpu().numpy()

                patches.append((x, y, patch_pred))

        return self._merge_patches(patches, image_shape)


def merge_images(images):
    """
    Merge a list of images into a single large image by concatenating them horizontally.

    Args:
        images (list of Image): List of PIL Image objects to merge.

    Returns:
        Image: The merged image.
    """
    # Get the maximum height of all images
    max_height = max(image.height for image in images)

    # Resize all images to have the same height (optional)
    resized_images = []
    for image in images:
        resized_image = image.resize((int(image.width * max_height / image.height), max_height))
        resized_images.append(resized_image)

    # Concatenate images horizontally
    total_width = sum(image.width for image in resized_images)
    merged_image = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for image in resized_images:
        merged_image.paste(image, (x_offset, 0))
        x_offset += image.width

    return merged_image


def plot_results_building(image: Image, prediction: np.ndarray, color_dict: dict):
    """
    Plot the pre-change image, post-change image, and overlay the prediction.

    Args:
        image (Image): image.
        prediction (np.ndarray): Prediction array with shape (height, width).
    """
    # Create an empty array to store the color-mapped prediction mask
    height, width = prediction.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Map each pixel's label in prediction to a color
    for label, color in color_dict.items():
        colored_mask[prediction == label] = color[:3]  # Use only RGB

    # Plot the original image and prediction overlay
    plt.figure(figsize=(15, 5))

    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    # Plot the prediction overlay
    plt.imshow(image)
    plt.imshow(colored_mask, alpha=0.3)
    plt.title("Prediction Overlay")
    plt.axis("off")

    # Show the plot
    plt.tight_layout()
    plt.show()
