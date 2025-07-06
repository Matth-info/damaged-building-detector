from __future__ import annotations

import logging
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm

if TYPE_CHECKING:
    from torch.utils.data import Dataset


def define_class_weights(
    dataset: Dataset,
    mask_key: str = "post_mask",
    subset_size: int | None = None,
    seed: int | None = None,
) -> dict:
    """Define classweights for a segmentation dataset to address class imbalance.

    Args:
    ----
        dataset: A segmentation dataset where each sample includes an image and its corresponding mask.
        mask_key: Key to access the mask in the dataset sample (default: "post_mask").
        subset_size: Number of random samples to use for estimating class weights. If None, uses the full dataset.
        seed: Seed for random number generation.

    Returns:
        class_weights: Inversely proportional class weights for imbalance dataset.

    """
    if seed is not None:
        random.seed(seed)
        rng = np.random.default_rng(seed)

    # Sample a subset of indices if required
    if subset_size is not None:
        sampled_indices = rng.choice(
            len(dataset), size=min(subset_size, len(dataset)), replace=False
        )
    else:
        sampled_indices = range(len(dataset))

    # Initialize a counter for pixel-level class frequencies
    class_counts = Counter()

    # Loop through the sampled subset of the dataset to count class frequencies in masks
    for i in tqdm(sampled_indices, desc="Counting class frequencies"):
        mask = dataset[i][mask_key]
        mask_flat = mask.flatten().numpy()  # Flatten the mask to count pixel-level classes
        class_counts.update(mask_flat)

    # Convert class counts to weights (inverse frequency)
    total_pixels = sum(class_counts.values())
    return {cls: total_pixels / (count + 1e-6) for cls, count in class_counts.items()}


# Utils for dealing with class imbalanced datasets
def define_weighted_random_sampler(
    dataset: Dataset,
    subset_size: int | None,
    seed: int | None,
    mask_key: str = "post_mask",
    num_workers: int = 16,
) -> WeightedRandomSampler:
    """Define a WeightedRandomSampler for a segmentation dataset to address class imbalance.

    Args:
    ----
        dataset: Dataset where each sample includes a mask under mask_key.
        mask_key: Key to access the mask in the dataset sample.
        subset_size: Optional number of samples to estimate class weights.
        seed: Optional random seed.
        num_workers: Number of worker threads to use for parallel sample weight computation.

    Returns:
    -------
        sampler: A WeightedRandomSampler for balanced class sampling.
        class_weights: Dict of class weights.

    """
    class_weights = define_class_weights(dataset, mask_key, subset_size, seed)
    logging.info("Class weights : %s", class_weights)

    # Precompute weights with multiprocessing
    def _compute_sample_weight(i: int):  # noqa: ANN202
        mask = dataset[i][mask_key]
        mask_flat = mask.flatten().numpy()
        unique, counts = np.unique(mask_flat, return_counts=True)
        pixel_weights = np.array([class_weights[cls] for cls in unique])
        return np.dot(counts, pixel_weights) / counts.sum()

    # Use ThreadPoolExecutor to parallelize sample weight computation
    with ThreadPoolExecutor(
        max_workers=num_workers if num_workers else os.cpu_count(),
    ) as executor:
        sample_weights = list(
            tqdm(
                executor.map(_compute_sample_weight, range(len(dataset))),
                total=len(dataset),
                desc="Assigning sample weights",
            ),
        )

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(dataset), replacement=True
    )
    return sampler, class_weights
