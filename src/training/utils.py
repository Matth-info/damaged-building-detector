# Core Python imports
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

# Third-party libraries
# PyTorch imports
import torch
import torch.distributed as dist

# PyTorch Vision imports
from prettytable import PrettyTable
from torch import nn

if TYPE_CHECKING:
    import numpy as np
    from torch.optim import Optimizer, lr_scheduler


def initialize_optimizer_scheduler(
    model: nn.Module,
    optimizer: Optimizer | None,
    scheduler: lr_scheduler.LRScheduler | None,
    optimizer_params: dict | None,
    scheduler_params: dict | None,
) -> tuple[Optimizer, lr_scheduler.LRScheduler]:
    """Initialize the optimizer and learning rate scheduler.

    Args:
    ----
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
    -------
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


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer | None,
    scheduler: lr_scheduler.LRScheduler | None,
    model_dir: str,
    experiment_name: str,
) -> None:
    """Save a checkpoint for the model, optimizer, and scheduler.

    Args:
    ----
        epoch (int): Current epoch number.
        model (torch.nn.Module): The model being trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        model_dir (str): Directory to save the checkpoint.
        experiment_name (str): Experiment name

    """
    checkpoint_dir = Path(model_dir) / experiment_name
    # Ensure checkpoint directory exists
    Path.mkdir(checkpoint_dir, exist_ok=True)
    # Save periodic checkpoints
    checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pth"
    checkpoint_data = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
    }
    torch.save(checkpoint_data, checkpoint_path)
    logging.info("Checkpoint saved at %s", checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optimizer | None,
    scheduler: lr_scheduler.LRScheduler | None,
) -> int:
    """Load a checkpoint for the model, optimizer, and scheduler.

    Args:
    ----
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to restore the state to.
        optimizer (torch.optim.Optimizer, optional): The optimizer to restore the state to.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to restore the state to.

    Returns:
    -------
        int: The epoch to resume training from.

    """
    if not Path.exists(checkpoint_path):
        raise FileNotFoundError("Checkpoint file not found: %s", checkpoint_path)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    logging.info("Model state loaded from %d", checkpoint_path)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        logging.info("Optimizer state loaded.")

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        logging.info("Scheduler state loaded.")

    return checkpoint["epoch"]


def display_metrics(metrics: dict, phase: str) -> None:
    """Display metrics in a tabular format.

    Args:
    ----
        phase (str): The phase of training (e.g., 'training', 'validation').
        metrics (dict): Dictionary containing metric names and their values.

    Returns:
        None: This function prints the formatted confusion matrix to the console.

    """
    logging.info("\nMetrics (%s Phase): \n", phase.capitalize())
    table = PrettyTable()
    table.field_names = ["Metric", "Value"]

    for metric, value in metrics.items():
        # Convert the value to a float if it's a numpy float
        table.add_row([metric, f"{value.item() if hasattr(value, 'item') else value:.4f}"])

    logging.info(table)


def display_confusion_matrix(cm: np.ndarray, class_names: list[str], decimals: int) -> None:
    """Display a confusion matrix in a formatted table using PrettyTable.

    Args:
        cm (np.ndarray): The confusion matrix to display, typically a 2D array of shape (n_classes, n_classes).
        class_names (list[str]): List of class names corresponding to the rows and columns of the confusion matrix.
        decimals (int): Number of decimal places to round the matrix values for display.

    Returns:
        None: This function prints the formatted confusion matrix to the console.

    """
    table = PrettyTable()
    table.field_names = [r"True \ Pred", *class_names]

    for i, row in enumerate(cm):
        formatted_row = [round(x, decimals) for x in row]
        table.add_row([class_names[i], *formatted_row])
    logging.info(table)


# Utils Functions for Fine Tuning Mask-R-CNN #


def _is_dist_avail_and_initialized() -> bool:
    if not dist.is_available():
        return False
    return dist.is_initialized()


def _get_world_size() -> int:
    if not _is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def _reduce_dict(input_dict: dict, *, average: bool = True) -> dict:
    """Reduce the values in the dictionary from all processes so that all processes have the averaged results. Returns a dict with the same fields as input_dict, after reduction.

    Args:
    ----
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum

    """
    world_size = _get_world_size()
    if world_size < 2:  # noqa: PLR2004
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
        return dict(zip(names, values))
