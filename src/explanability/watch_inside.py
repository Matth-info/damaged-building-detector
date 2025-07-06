from __future__ import annotations

import argparse
import glob
import logging
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.augmentation import Augmentation_pipeline
from src.data import renormalize_image
from src.datasets import DATASETS_MAP
from src.explanability.help_funcs import read_config
from src.models import MODELS_MAP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Dictionaries to store feature maps
feature_maps_pre = {}  # Features from encoder processing pre_image
feature_maps_post = {}  # Features from encoder processing post_image
feature_maps_decoder = {}  # Features from shared decoder layers


def hook_fn_encoder(name: str) -> callable:
    """Define Hook functions for layers processing pre_image and post image by encoder."""

    def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        if name not in feature_maps_pre:
            feature_maps_pre[name] = output.detach().cpu()  # Store feature map
        else:
            feature_maps_post[name] = output.detach().cpu()  # Store feature map

    return hook


def hook_fn_decoder(name) -> callable:
    """Define Hook function for shared decoder layers."""

    def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        feature_maps_decoder[name] = output.detach().cpu()  # Store feature map

    return hook


def register_hooks(model) -> None:
    """Register hooks on the Siamese network to extract feature maps separately for pre_image, post_image, and the shared decoder.

    Args:
     model (torch.nn.Module): Siamese network model

    """
    for name, layer in model.named_modules():
        # Encoder part (before fusion)
        if "encoder" in name and isinstance(layer, nn.Conv2d):
            layer.register_forward_hook(hook_fn_encoder(name))

        # Decoder part (after fusion)
        if "decoder" in name and isinstance(layer, nn.Conv2d):
            layer.register_forward_hook(hook_fn_decoder(name))


def visualize_feature_maps(
    x1: torch.Tensor,
    x2: torch.Tensor,
    preds: torch.Tensor,
    ground_truth: torch.Tensor,
    feature_maps_pre: dict,
    feature_maps_post: dict,
    feature_maps_decoder: dict,
    k: int = 10,
    num_cols: int = 6,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    output_path: str | None = None,
) -> None:
    """Visualize input images, predictions, ground truth, and randomly selected feature maps.

    Args:
     x1 (torch.Tensor): Before-change image (1, 3, H, W)
     x2 (torch.Tensor): After-change image (1, 3, H, W)
     preds (torch.Tensor): Model predictions
     ground_truth (torch.Tensor): Ground truth mask (1, H, W)
     feature_maps_pre (dict): Feature maps from the pre-image encoder
     feature_maps_post (dict): Feature maps from the post-image encoder
     feature_maps_decoder (dict): Feature maps from the decoder
     k (int): Number of random feature maps to show
     num_cols (int): Number of columns in the grid layout
     mean (list): Mean for denormalization
     std (list): Standard deviation for denormalization
     output_path (str): output filepath (default is None which does not save the figures.)

    """
    if set(feature_maps_pre.keys()) != set(feature_maps_post.keys()):
        raise KeyError("Feature maps mismatch between pre and post encoders")

    # Select k random layers from pre and corresponding post
    feature_maps_pre_list = list(feature_maps_pre.items())
    selected_pre_layers_ids = sorted(
        random.sample(list(range(len(feature_maps_pre_list))), min(k, len(feature_maps_pre_list))),
    )
    selected_pre_layers = [feature_maps_pre_list[i] for i in selected_pre_layers_ids]
    selected_post_layers = [
        (layer_name, feature_maps_post[layer_name]) for layer_name, _ in selected_pre_layers
    ]

    num_cols = max(4, min(k, len(feature_maps_pre_list)))
    num_rows = 4  # Fixed structure: input row + pre + post + decoder
    plt.figure(figsize=(num_cols * 1.5, num_rows * 1.5))

    def tensor_to_image(tensor: torch.Tensor) -> torch.Tensor:
        tensor = renormalize_image(tensor, mean, std, device)
        return tensor.squeeze().permute(1, 2, 0).cpu().numpy()

    def tensor_to_mask(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.squeeze().cpu().numpy()

    def process_feature_maps(feature_maps: torch.Tensor) -> torch.Tensor:
        feature_maps = feature_maps.mean(dim=1).squeeze().cpu().numpy()
        return (feature_maps - feature_maps.min()) / (
            feature_maps.max() - feature_maps.min() + 1e-6
        )

    # Plot row 1: Input Images, Predictions, Ground Truth
    plt.subplot(num_rows, num_cols, 1)
    plt.imshow(tensor_to_image(x1))
    plt.axis("off")
    plt.title("Before Change")

    plt.subplot(num_rows, num_cols, 2)
    plt.imshow(tensor_to_image(x2))
    plt.axis("off")
    plt.title("After Change")

    plt.subplot(num_rows, num_cols, 3)
    plt.imshow(tensor_to_mask(ground_truth), cmap="gray")
    plt.axis("off")
    plt.title("Ground Truth")

    plt.subplot(num_rows, num_cols, 4)
    plt.imshow(tensor_to_mask(preds), cmap="gray")
    plt.axis("off")
    plt.title("Prediction")

    # Plot row 2: Pre-image encoder feature maps
    for i, (layer_name, feature_map) in enumerate(selected_pre_layers):
        plt.subplot(num_rows, num_cols, num_cols + i + 1)
        plt.imshow(process_feature_maps(feature_map), cmap="jet")
        plt.axis("off")
        plt.title(f"Pre-{layer_name}")

    # Plot row 3: Post-image encoder feature maps
    for i, (layer_name, feature_map) in enumerate(selected_post_layers):
        axes = plt.subplot(num_rows, num_cols, 2 * num_cols + i + 1)
        plt.imshow(process_feature_maps(feature_map), cmap="jet")
        plt.axis("off")
        axes.set_title(f"Post-{layer_name}", rotation=45)

    # Plot row 4: Decoder feature maps
    for i, (layer_name, feature_map) in enumerate(feature_maps_decoder.items()):
        axes = plt.subplot(num_rows, num_cols, 3 * num_cols + i + 1)
        plt.imshow(process_feature_maps(feature_map), cmap="jet")
        plt.axis("off")
        axes.title(f"Decoder-{layer_name}", rotation=45)

    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()


def main() -> None:
    """Whatch Inside a CNN based model."""
    parser = argparse.ArgumentParser(
        prog="WatchInside",
        description="Plot Internal Layer representation",
    )
    parser.add_argument("--config_path", default=None, type=str)
    args = parser.parse_args()
    config_dict = read_config(args.config_path)

    # Load model
    model_class = MODELS_MAP.get(config_dict.get("model_class", None), None)
    model_params = config_dict.get("model_params", None)
    checkpoint_path = (
        f"{config_dict.get('checkpoint_folder')}/{config_dict.get('checkpoint_name')}"
    )
    model = model_class(**model_params)
    if Path.exists(checkpoint_path):
        model = model.load(file_path=checkpoint_path).eval()
    else:
        logging.info("Checkpoint path does not exist, initialize new model weights")

    # Load data
    dataset_class = DATASETS_MAP.get(config_dict.get("dataset_name", None))
    dataset_path = f"{config_dict.get('data_folder')}/{config_dict.get('dataset_path')}"
    mean, std = dataset_class.MEAN, dataset_class.STD

    image_size = config_dict.get("image_size")
    transform = Augmentation_pipeline(
        image_size=(image_size, image_size),
        mean=mean,
        std=std,
        mode="test",
    )
    dataset = dataset_class(origin_dir=dataset_path, type="test", transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Define Models Hooks
    register_hooks(model)

    model = model.to(device)

    for batch in dataloader:
        pre_image, post_image, mask = (
            batch["pre_image"].to(device, dtype=torch.float32),
            batch["post_image"].to(device, dtype=torch.float32),
            batch["mask"],
        )
        preds = model.predict(x1=pre_image, x2=post_image)

        break

    output_folder = config_dict.get("output_folder")
    if output_folder:
        count = len(list(Path(output_folder).glob("viz_*.png")))
        output_path = f"{output_folder}/viz_{count}.png"

    visualize_feature_maps(
        x1=pre_image,
        x2=post_image,
        preds=preds,
        ground_truth=mask,
        feature_maps_pre=feature_maps_pre,
        feature_maps_post=feature_maps_post,
        feature_maps_decoder=feature_maps_decoder,
        mean=mean,
        std=std,
        k=6,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
