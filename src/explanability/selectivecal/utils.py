from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributions as td
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn


def initialization(m: torch.Module) -> None:
    """Initialize kernel weights with Gaussian distributions."""
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def _calculate_ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """Calculate the Expected Calibration Error (ECE) of a classification model.

    ECE measures the difference between predicted confidence and actual accuracy across bins of confidence scores.
    The function divides the confidence outputs into equally-sized interval bins, computes the average confidence
    and accuracy in each bin, and returns a weighted average of their absolute differences.

    Args:
    ----
        logits (torch.Tensor): The raw, unnormalized model outputs (logits) of shape (N, num_classes).
        labels (torch.Tensor): The ground truth labels of shape (N,).
        n_bins (int, optional): Number of bins to use for calibration. Default is 10.

    Returns:
    -------
        float: The expected calibration error (ECE) as a single scalar value.

    References:
    ----------
        Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
        "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI. 2015.

    """
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()


def make_model_diagrams(
    outputs: list,
    labels: list,
    n_bins: int = 10,
    filename: str = "reliability_diagram.png",
    info: str | None = None,
    folderpath: str = "./outputs",
) -> None:
    """Generate and displays a reliability diagram for a classification model's outputs, and computes the Expected Calibration Error (ECE).

    Args:
    ----
        outputs (torch.Tensor): Tensor of shape (n, num_classes) containing the raw outputs (logits) from the model's final linear layer (not softmax probabilities).
        labels (torch.Tensor): Tensor of shape (n,) containing the ground truth class labels.
        n_bins (int, optional): Number of bins to use for the reliability diagram. Default is 10.
        filename (str, optional): filename
        info (str, optional): add extra info in the plot name
        folderpath (str, optional): folder path

    Returns:
    -------
        float: The computed Expected Calibration Error (ECE) for the given outputs and labels.

    Side Effects:
    -----
        - Saves the reliability diagram as 'reliability_diagram.png' in the current working directory.
        - Displays the reliability diagram using matplotlib.

    Notes:
    -----
        - The reliability diagram visualizes the relationship between model confidence and accuracy.
        - The function expects raw logits as input, not softmax probabilities.
        - The ECE value is also displayed on the plot.

    """
    Path.mkdir(folderpath, exist_ok=True)
    filename = Path(folderpath) / filename
    softmaxes = torch.nn.functional.softmax(outputs, 1)
    confidences, predictions = softmaxes.max(1)
    accuracies = torch.eq(predictions, labels)

    # Reliability diagram
    bins = torch.linspace(0, 1, n_bins + 1)
    width = 1.0 / n_bins
    bin_centers = np.linspace(0, 1.0 - width, n_bins) + width / 2
    bin_indices = [
        confidences.ge(bin_lower) * confidences.lt(bin_upper)
        for bin_lower, bin_upper in zip(bins[:-1], bins[1:])
    ]

    bin_corrects = np.array(
        [torch.mean(accuracies[bin_index].float()) for bin_index in bin_indices],
    )
    bin_scores = np.array(
        [torch.mean(confidences[bin_index].float()) for bin_index in bin_indices],
    )
    bin_corrects = np.nan_to_num(bin_corrects)
    bin_scores = np.nan_to_num(bin_scores)

    plt.figure(0, figsize=(8, 8))
    gap = np.array(bin_scores - bin_corrects)

    confs = plt.bar(bin_centers, bin_corrects, color=[0, 0, 1], width=width, ec="black")
    bin_corrects = np.nan_to_num(np.array(list(bin_corrects)))
    gaps = plt.bar(
        bin_centers,
        gap,
        bottom=bin_corrects,
        color=[1, 0.7, 0.7],
        alpha=0.5,
        width=width,
        hatch="//",
        edgecolor="r",
    )

    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.legend([confs, gaps], ["Accuracy", "Gap"], loc="upper left", fontsize="x-large")

    ece = _calculate_ece(outputs, labels)

    # Clean up
    bbox_props = {"boxstyle": "square", "fc": "lightgrey", "ec": "gray", "lw": 1.5}
    plt.text(
        0.17,
        0.82,
        f"ECE: {ece:.4f}",
        ha="center",
        va="center",
        size=20,
        weight="normal",
        bbox=bbox_props,
    )

    plt.title(f"Reliability Diagram {info}", size=22)
    plt.ylabel("Accuracy", size=18)
    plt.xlabel("Confidence", size=18)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(filename)
    logging.info("Reliability Diagram has been successfully saved in %s", filename)
