from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.explanability.cose.attribution_methods import choose_attribution_method

if TYPE_CHECKING:
    from torch.utils.data import Dataset


@torch.no_grad()
def plot_overlay(
    idx: int,
    dataset: Dataset,
    model: torch.nn.Module,
    alpha_nominal: float,
    threshold: float,
    mode: str = "LAC",
    colormap: str = "turbo",
    device: str = "cpu",
    figsize: tuple[int, int] = (12, 6),
) -> None:
    """Plot the attribution map over the input image.

    Args:
    ----
        idx (int): Index of the sample in the dataset.
        dataset (Dataset): Dataset containing the samples.
        model (torch.nn.Module): PyTorch model for predictions.
        alpha_nominal (float): Nominal risk level.
        threshold (float): Threshold value for attribution.
        mode (str): Attribution mode ("LAC" or "APS").
        colormap (str): Colormap for the attribution map.
        device (str): Device to run the model on ("cpu" or "cuda").
        figsize (tuple[int, int]): Figure size for the plot.

    Returns:
    -------
        None: Displays the overlay of the attribution map on the input image.

    """
    model.eval()
    attribution_method = choose_attribution_method(name=mode)

    # Get inputs and predictions
    inputs = dataset[idx]
    images = inputs["image"].unsqueeze(0).to(device)  # shape (1, C, H, W)
    image_np = images.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
    logits = model(images)
    logits = logits.squeeze(0)  # shape (C, H, W)
    probs = torch.nn.functional.softmax(logits, dim=0)

    # Compute attribution
    one_hot_attribution = attribution_method(
        threshold,
        probs,
        dataset.n_classes,
        always_include_top1=True,
    )
    multimask = one_hot_attribution.sum(dim=0)  # count the size of predicted set for each pixel

    # Plot overlay
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.imshow(image_np / image_np.max())  # Normalize image for display
    ax.imshow(multimask.cpu().numpy(), cmap=colormap, alpha=0.5)  # Overlay attribution map
    ax.set_title(
        f"Overlay Attribution Map ({mode}). Risk level = {alpha_nominal:.2f}, Threshold = {threshold:.5f}",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(["Input Image", "Attribution Map"], loc="upper right")  # Add legend
    plt.show()
    torch.cuda.empty_cache()


@torch.no_grad()
def plot_heatmap(
    idx: int,
    dataset: Dataset,
    model: torch.nn.Module,
    alpha_nominal: float,
    threshold: float,
    mode: str = "LAC",
    class_labels: list[tuple[int, str | None]] | None = None,
    figsize: tuple[int, int] = (12, 6),
    colormap: str = "turbo",
    binclasses_cmap: str = "Purples",
    device: str = "cpu",
    *,
    plot_ticks: bool = True,
    plot_classes: bool = False,
) -> None:
    """Plot a heatmap for model predictions and optionally binary masks for specific classes.

    Args:
    ----
        idx (int): Index of the sample in the dataset.
        dataset (Dataset): Dataset containing the samples.
        model (torch.nn.Module): PyTorch model for predictions.
        alpha_nominal (float): Nominal risk level.
        threshold (float): Threshold value for attribution.
        mode (str): Attribution mode ("LAC" or "APS").
        class_labels (list[tuple[int, str | None]]): List of tuples containing class indices and names.
        plot_classes (bool): Whether to plot binary masks for specific classes.
        figsize (tuple[int, int]): Figure size for the plot.
        plot_ticks (bool): Whether to display axis ticks.
        colormap (str): Colormap for the heatmap.
        binclasses_cmap (str): Colormap for binary class masks.
        device (str): Device to run the model on ("cpu" or "cuda").

    Returns:
    -------
        None: Displays the heatmap and optional binary masks.

    """
    model.eval()
    attribution_method = choose_attribution_method(name=mode)

    # Get inputs and predictions
    inputs = dataset[idx]
    images = inputs["image"].unsqueeze(0).to(device)  # shape (1, C, H, W)
    gt_mask = inputs["mask"].long()
    logits = model(images)
    logits = logits.squeeze(0)  # shape (C, H, W)
    probs = torch.nn.functional.softmax(logits, dim=0)

    # Compute attribution
    one_hot_attribution = attribution_method(
        threshold,
        probs,
        dataset.n_classes,
        always_include_top1=True,
    )

    # Create figure
    fig, axes = plt.subplots(
        nrows=1 + (len(class_labels) if plot_classes and class_labels else 0),
        ncols=1,
        figsize=figsize,
        constrained_layout=True,
    )

    # Plot binary masks for specific classes if requested
    if plot_classes and class_labels:
        n_cols = min(5, len(class_labels))
        n_rows = (len(class_labels) + n_cols - 1) // n_cols
        fig_masks, axes_masks = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(figsize[0], figsize[1] * n_rows),
            constrained_layout=True,
        )
        if n_rows == 1:
            axes_masks = axes_masks.reshape(-1)  # Ensure axes_masks is always a list

        for i, (lab, name) in enumerate(class_labels):
            ax = axes_masks[i]
            ax.set_title(f"Class: {name}")
            ax.imshow(
                one_hot_attribution[lab].cpu().numpy(),
                cmap=binclasses_cmap,
            )
            selclass = np.where(gt_mask.cpu().numpy() == lab, 1, 0)
            ax.imshow(selclass, cmap=binclasses_cmap, alpha=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(["Attribution", "Ground Truth"], loc="upper right")

        # Hide unused axes
        for j in range(len(class_labels), len(axes_masks)):
            axes_masks[j].axis("off")

        plt.show()

    # Compute and plot the multimask heatmap
    multimask = one_hot_attribution.sum(dim=0)  # count the size of predicted set for each pixel
    activations = multimask.sum() / (one_hot_attribution.shape[1] * one_hot_attribution.shape[2])

    fig_heatmap, ax_heatmap = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax_heatmap.imshow(
        multimask.cpu().numpy(),
        cmap=colormap,
        vmin=1,
        vmax=dataset.n_classes,
    )
    if not plot_ticks:
        ax_heatmap.set_xticks([])
        ax_heatmap.set_yticks([])

    title = (
        f"Heatmap ({mode}). Risk level = {alpha_nominal:.2f}, "
        f"Threshold = {threshold:.5f}, Activations: {activations * 100:.2f}%"
    )
    ax_heatmap.set_title(title)
    fig_heatmap.colorbar(im, ax=ax_heatmap, orientation="vertical", fraction=0.046, pad=0.04)

    plt.show()
    torch.cuda.empty_cache()
