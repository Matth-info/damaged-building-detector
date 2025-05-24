# Core Python imports
import logging
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# Third-party libraries
import numpy as np

# PyTorch imports
import torch
import torch.distributed as dist
import torch.nn as nn

# PyTorch Vision imports
import torchvision
from prettytable import PrettyTable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.utils import save_image
from tqdm import tqdm

from src.utils.visualization import apply_color_map

__all__ = [
    "define_weighted_random_sampler",
    "define_class_weights",
    "save_checkpoint",
    "load_checkpoint",
    "initialize_optimizer_scheduler",
]


def initialize_optimizer_scheduler(
    model, optimizer=None, scheduler=None, optimizer_params=None, scheduler_params=None
):
    """
    Initializes the optimizer and learning rate scheduler.

    Args:
        model (torch.nn.Module): The model whose parameters require optimization.
        optimizer (type or torch.optim.Optimizer, optional): Optimizer class (e.g., torch.optim.AdamW) or an initialized optimizer.
            Defaults to None, which uses AdamW with a default learning rate.
        scheduler (type or torch.optim.lr_scheduler._LRScheduler, optional): Scheduler class or an initialized scheduler.
            Defaults to None, which uses StepLR with a step size of 10 and gamma of 0.1.
        optimizer_params (dict, optional): Parameters for the optimizer. Ignored if `optimizer` is already initialized.
            Defaults to None.
        scheduler_params (dict, optional): Parameters for the scheduler. Ignored if `scheduler` is already initialized.
            Defaults to None.

    Returns:
        tuple: A tuple containing the initialized optimizer and scheduler.
    """
    # Default optimizer parameters
    optimizer_params = optimizer_params or {"lr": 1e-4}
    scheduler_params = scheduler_params or {"step_size": 10, "gamma": 0.1}

    # Initialize the optimizer
    if optimizer is None:
        optimizer_ft = torch.optim.AdamW(
            params=filter(lambda p: p.requires_grad, model.parameters()),
            **optimizer_params,
        )
    elif isinstance(optimizer, type):  # If optimizer is a class, initialize it
        optimizer_ft = optimizer(
            params=filter(lambda p: p.requires_grad, model.parameters()),
            **optimizer_params,
        )
    else:  # If optimizer is already initialized
        optimizer_ft = optimizer

    # Initialize the scheduler
    if scheduler is None:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, **scheduler_params)
    elif isinstance(scheduler, type):  # If scheduler is a class, initialize it
        lr_scheduler = scheduler(optimizer=optimizer_ft, **scheduler_params)
    else:  # If scheduler is already initialized
        lr_scheduler = scheduler

    return optimizer_ft, lr_scheduler


def save_checkpoint(epoch, model, optimizer, scheduler, model_dir):
    """
    Save a checkpoint for the model, optimizer, and scheduler.

    Args:
        epoch (int): Current epoch number.
        model (torch.nn.Module): The model being trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        model_dir (str): Directory to save the checkpoint.
    """

    # Ensure checkpoint directory exists
    os.makedirs(model_dir, exist_ok=True)
    # Save periodic checkpoints
    checkpoint_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch}.pth")
    checkpoint_data = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    torch.save(checkpoint_data, checkpoint_path)
    logging.info(f"Checkpoint saved at {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Load a checkpoint for the model, optimizer, and scheduler.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to restore the state to.
        optimizer (torch.optim.Optimizer, optional): The optimizer to restore the state to.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to restore the state to.

    Returns:
        int: The epoch to resume training from.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Model state loaded from {checkpoint_path}")

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        print("Optimizer state loaded.")

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        print("Scheduler state loaded.")

    return checkpoint["epoch"]


def save_model(model, ckpt_path="./models", name="model"):
    path = os.path.join(ckpt_path, f"{name}.pth")
    torch.save(model.state_dict(), path, _use_new_zipfile_serialization=False)


def load_model(model, ckpt_path):
    state = torch.load(ckpt_path)
    model.load_state_dict(state)
    return model


def display_metrics(metrics, phase):
    """
    Display metrics in a tabular format.

    Args:
        phase (str): The phase of training (e.g., 'training', 'validation').
        metrics (dict): Dictionary containing metric names and their values.
    """
    print(f"\nMetrics ({phase.capitalize()} Phase): \n")
    table = PrettyTable()
    table.field_names = ["Metric", "Value"]

    for metric, value in metrics.items():
        # Convert the value to a float if it's a numpy float
        if hasattr(value, "item"):
            value = value.item()
        table.add_row([metric, f"{value:.4f}"])

    print(table)


# Utils for dealing with class imbalanced datasets
def define_weighted_random_sampler(dataset, mask_key="post_mask", subset_size=None, seed: int = None):
    """
    Define a WeightedRandomSampler for a segmentation dataset to address class imbalance.

    Args:
        dataset: Dataset where each sample includes a mask under mask_key.
        mask_key: Key to access the mask in the dataset sample.
        subset_size: Optional number of samples to estimate class weights.
        seed: Optional random seed.

    Returns:
        sampler: A WeightedRandomSampler for balanced class sampling.
        class_weights: Dict of class weights.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Sample a subset of indices if required
    if subset_size is not None:
        sampled_indices = random.sample(range(len(dataset)), min(subset_size, len(dataset)))
    else:
        sampled_indices = range(len(dataset))

    # Count class frequencies across selected samples
    class_counts = Counter()
    for i in tqdm(sampled_indices, desc="Counting class frequencies"):
        mask = dataset[i][mask_key]
        mask_flat = mask.flatten().numpy()
        class_counts.update(mask_flat)

    # Calculate inverse frequency weights
    total_pixels = sum(class_counts.values())
    class_weights = {
        cls: np.round(total_pixels / (count + 1e-6), 3) for cls, count in class_counts.items()
    }

    # Precompute weights with multiprocessing
    def compute_sample_weight(i):
        mask = dataset[i][mask_key]
        mask_flat = mask.flatten().numpy()
        unique, counts = np.unique(mask_flat, return_counts=True)
        pixel_weights = np.array([class_weights[cls] for cls in unique])
        return np.dot(counts, pixel_weights) / counts.sum()

    # Use ThreadPoolExecutor to parallelize sample weight computation
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        sample_weights = list(tqdm(
            executor.map(compute_sample_weight, range(len(dataset))),
            total=len(dataset),
            desc="Assigning sample weights"
        ))

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)
    return sampler, class_weights


def define_class_weights(dataset, mask_key="post_mask", subset_size=None, seed: int = None):
    """
    Define classweights for a segmentation dataset to address class imbalance.

    Args:
        dataset: A segmentation dataset where each sample includes an image and its corresponding mask.
        mask_key: Key to access the mask in the dataset sample (default: "post_mask").
        subset_size: Number of random samples to use for estimating class weights. If None, uses the full dataset.
        seed : seed number
    Returns:
        sampler: A WeightedRandomSampler for balanced class sampling.
        class_weights: Inversely proportional class weights for imbalance dataset.
    """
    if seed is not None:
        random.seed(seed)
    # Determine subset of dataset to analyze (optional)
    if subset_size is not None:
        sampled_indices = random.sample(range(len(dataset)), min(len(dataset), subset_size))
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
    class_weights = {cls: total_pixels / (count + 1e-6) for cls, count in class_counts.items()}
    return class_weights


# Utils Functions for Fine Tuning Mask-R-CNN #

__all__ = ["reduce_dict"]


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
