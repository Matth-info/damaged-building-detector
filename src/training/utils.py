# Core Python imports
import os
import random
from collections import Counter
from prettytable import PrettyTable
from tqdm import tqdm
from typing import Union, Dict, Any
import logging

# PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

# PyTorch Vision imports
import torchvision
from torchvision.utils import save_image

# Third-party libraries
import numpy as np
import mlflow
from mlflow.models.signature import infer_signature

__all__ = [
    "apply_color_map", 
    "define_weighted_random_sampler", 
    "define_class_weights", 
    "custom_infer_signature",
    "save_checkpoint",
    "load_checkpoint",
    "initialize_optimizer_scheduler"
]

def initialize_optimizer_scheduler(model, 
                            optimizer=None, 
                            scheduler=None, 
                            optimizer_params=None, 
                            scheduler_params=None
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

def log_metrics(
    metrics: dict, step_number: int, phase: str = "Validation"
):
    """
    Logs each metric in the dictionary to TensorBoard.

    Parameters:
    - metrics: Dictionary of metric name and value pairs.
    - step_number: The current step number.
    - phase: 'Validation' or 'Training', used to distinguish metrics in TensorBoard.
    """
    mlflow.log_metrics({f"{phase}_{metric_name}": value for metric_name, value in metrics.items()}, step=step_number)

def log_loss(loss_value: float, 
    step_number: int, phase: str = "Validation"
):
    """
    Logs loss value to TensorBoard.

    Parameters:
    - loss_value: current loss value
    - step_number: The current step number.
    - phase: 'Validation' or 'Training', used to distinguish metrics in TensorBoard.
    """
    mlflow.log_metric(f"{phase}_loss", f"{loss_value:2f}", step=step_number)

"""
    def log_graph(model: torch.nn.Module, siamese: bool = False, device: str = "cuda", input_shape: tuple = (1, 3, 512, 512)):

    Logs the model graph to TensorBoard. Supports both single-input and siamese models.

    Args:
        writer (SummaryWriter): The TensorBoard SummaryWriter.
        model (torch.nn.Module): The model to log.
        siamese (bool): Flag to indicate if the model is Siamese.
        device (str): The device (cuda or cpu) where the model and tensors should be moved.
        input_shape (tuple): The shape of the input tensor (default: (1, 3, 512, 512)).
    model.to(device)  # Move model to the correct device
    
    # For Siamese models, we create two input tensors
    if siamese:
        x1 = torch.randn(*input_shape).to(device)
        x2 = torch.randn(*input_shape).to(device)
        writer.add_graph(model, (x1, x2))
    else:
        x1 = torch.randn(*input_shape).to(device)
        writer.add_graph(model, x1)"""

def log_model(model, artifact_path="best_model", signature=None, input_example=None):
    mlflow.pytorch.log_model(
            model, 
            artifact_path=artifact_path,
            signature=signature,
            input_example=input_example
        )
        
def custom_infer_signature(model, data_loader, siamese: bool = False, image_key: str = "image", mask_key: str = "mask", device: str = "cuda"):
    """
    Infers the MLflow signature for a given model using a single batch of data from the dataloader.

    Args:
        model (torch.nn.Module): The model for which to infer the signature.
        data_loader (DataLoader): DataLoader providing batches of input data.
        siamese (bool): Whether the model is a Siamese model (two inputs).
        image_key (str): Key in the batch for input images.
        mask_key (str): Key in the batch for target masks.
        device (str): Device to perform inference ('cuda' or 'cpu').

    Returns:
        signature: The inferred signature for the model.
    """
    # Get a single batch from the data loader
    batch = next(iter(data_loader))

    # Move the model to the specified device and set it to evaluation mode
    model.to(device)
    model.eval()

    with torch.no_grad():
        # Handle the inputs based on whether the model is Siamese or not
        if siamese:
            pre_image = batch["pre_image"].to(device)  # Move to device
            post_image = batch["post_image"].to(device)
            targets = batch[mask_key].to(device)

            # Perform inference
            predictions = model(pre_image, post_image)

            # Convert inputs and outputs to CPU and NumPy for MLflow
            pre_image = pre_image.cpu().numpy()
            post_image = post_image.cpu().numpy()
            predictions = predictions.cpu().numpy()

            # Prepare example inputs for MLflow signature
            example_inputs = {"pre_image": pre_image, "post_image": post_image}

        else:
            images = batch[image_key].to(device)  # Move to device
            targets = batch[mask_key].to(device)

            # Perform inference
            predictions = model(images)

            # Convert inputs and outputs to CPU and NumPy for MLflow
            images = images.cpu().numpy()
            predictions = predictions.cpu().numpy()

            # Prepare example inputs for MLflow signature
            example_inputs = {"images": images}

    # Infer the signature using example inputs and outputs
    signature = infer_signature(example_inputs, predictions)
    logging.info("Model Signature has been defined")
    return signature, example_inputs

# Function to apply colors to masks (labels and predictions)
def apply_color_map(mask, color_dict):
    """
    Applies a color map (color_dict) to a mask tensor.
    
    Args:
        mask (tensor): Mask tensor (predictions or labels). shape (N, H, W)
        color_dict (dict): A dictionary mapping class labels to RGB colors.
    
    Returns:
        color_mask (tensor): The colored mask tensor.
    """
    batch_size, height, width = mask.shape
    color_mask = torch.zeros((batch_size, 3, height, width), dtype=torch.float32)

    # Apply the color mapping for each class label in the mask

    for label, color in color_dict.items():
        binary_mask = (mask == label).float()
        color = color_dict[label] 
        # Apply the color to the binary mask
        color_mask[:, 0, :, :] += binary_mask * (color[0] / 255.0)  # Red channel
        color_mask[:, 1, :, :] += binary_mask * (color[1] / 255.0)  # Green channel
        color_mask[:, 2, :, :] += binary_mask * (color[2] / 255.0)  # Blue channel
    return color_mask

"""def log_images_to_tensorboard(
    model: torch.nn.Module,
    data_loader: DataLoader,
    writer: SummaryWriter,
    epoch: int,
    device: Union[str, torch.device],
    max_images: int = 4,
    image_key: str = "image",
    mask_key: str = "mask",
    siamese: bool = False, 
    color_dict = None
):
    Logs images, labels, and model predictions to TensorBoard.

    Args:
        model (torch.nn.Module): The model being evaluated.
        data_loader (DataLoader): DataLoader providing batches of data.
        writer (SummaryWriter): TensorBoard SummaryWriter instance.
        epoch (int): Current training epoch.
        device (Union[str, torch.device]): Device to perform computations on.
        max_images (int): Maximum number of images to log.
        image_key (str): Key in the batch dictionary for input images.
        mask_key (str): Key in the batch dictionary for target masks.
        siamese (bool): Whether the model is a Siamese model.
    model.eval()  # Set model to evaluation mode

    # Get a single batch from the data loader
    batch = next(iter(data_loader))
    if siamese:
        x1 = batch["pre_image"].to(device)  # Pre-disaster image
        x2 = batch["post_image"].to(device)  # Post-disaster image
        labels = batch[mask_key].to(device)  # Target mask
    else:
        x = batch[image_key].to(device)  # Single input image
        labels = batch[mask_key].to(device)  # Target mask

    # Generate predictions
    with torch.no_grad():
        if siamese:
            predictions = model(x1, x2)  # Forward pass for Siamese model
        else:
            predictions = model(x)  # Forward pass for single input model
        predictions = torch.argmax(predictions, dim=1)


    #Select a random subset of images
    batch_size = labels.size(0)
    num_images = min(max_images, batch_size)  # Ensure max_images doesn't exceed batch size
    indices = random.sample(range(batch_size), num_images)  # Randomly select indices
    
    # Move data back to CPU for visualization and limit the number of images
    if siamese:
        inputs_1 = x1.cpu().float()[indices]
        inputs_2 = x2.cpu().float()[indices]
    else:
        inputs = x.cpu().float()[indices]
    
    labels = labels.cpu().float()[indices]
    predictions = predictions.cpu().float()[indices]

    # Apply color map to labels and predictions if color_dict is provided
    if color_dict:
        colored_labels = apply_color_map(labels, color_dict)
        colored_predictions = apply_color_map(predictions, color_dict)
    else:
        # Default behavior: no color map (just grayscale)
        colored_labels = labels.unsqueeze(1).float() / 255.0  # Normalize to [0, 1] for grayscale
        colored_predictions = predictions.unsqueeze(1).float() / 255.0  # Normalize to [0, 1] for grayscale

    # Create grids for inputs, labels, and predictions
    if siamese:
        input_grid_1 = torchvision.utils.make_grid(inputs_1, normalize=True, scale_each=True)
        input_grid_2 = torchvision.utils.make_grid(inputs_2, normalize=True, scale_each=True)
        writer.add_image(tag="Inputs/Pre_Image", img_tensor=input_grid_1, global_step=epoch)
        writer.add_image(tag="Inputs/Post_Image", img_tensor=input_grid_2, global_step=epoch)
    else:
        input_grid = torchvision.utils.make_grid(inputs, normalize=False, scale_each=True)
        writer.add_image(tag="Inputs", img_tensor=input_grid, global_step=epoch)

    label_grid = torchvision.utils.make_grid(
        colored_labels, normalize=False, scale_each=True
    )
    pred_grid = torchvision.utils.make_grid(
        colored_predictions, normalize=False, scale_each=True
    )

    # Log labels and predictions
    writer.add_image(tag="Labels", img_tensor=label_grid, global_step=epoch)
    writer.add_image(tag="Predictions", img_tensor=pred_grid, global_step=epoch)

    model.train()  # Return to training mode if necessary
"""

def log_images_to_mlflow(
    model: torch.nn.Module,
    data_loader: DataLoader,
    epoch: int,
    device: Union[str, torch.device],
    max_images: int = 4,
    image_key: str = "image",
    mask_key: str = "mask",
    siamese: bool = False, 
    color_dict=None,
    log_dir: str = "mlflow_logs"
):
    """
    Logs images, labels, and model predictions to MLflow.

    Args:
        model (torch.nn.Module): The model being evaluated.
        data_loader (DataLoader): DataLoader providing batches of data.
        epoch (int): Current training epoch.
        device (Union[str, torch.device]): Device to perform computations on.
        max_images (int): Maximum number of images to log.
        image_key (str): Key in the batch dictionary for input images.
        mask_key (str): Key in the batch dictionary for target masks.
        siamese (bool): Whether the model is a Siamese model.
        color_dict (dict, optional): Dictionary for color mapping.
        log_dir (str): Directory to save images before logging to MLflow.
    """
    model.eval()  # Set model to evaluation mode

    # Get a single batch from the data loader
    batch = next(iter(data_loader))
    if siamese:
        x1 = batch["pre_image"].to(device)  # Pre-disaster image
        x2 = batch["post_image"].to(device)  # Post-disaster image
        labels = batch[mask_key].to(device)  # Target mask
    else:
        x = batch[image_key].to(device)  # Single input image
        labels = batch[mask_key].to(device)  # Target mask

    # Generate predictions
    with torch.no_grad():
        if siamese:
            predictions = model(x1, x2)  # Forward pass for Siamese model
        else:
            predictions = model(x)  # Forward pass for single input model
        predictions = torch.argmax(predictions, dim=1)

    # Select a random subset of images
    batch_size = labels.size(0)
    num_images = min(max_images, batch_size)  # Ensure max_images doesn't exceed batch size
    indices = random.sample(range(batch_size), num_images)  # Randomly select indices

    # Move data back to CPU for visualization and limit the number of images
    if siamese:
        inputs_1 = x1.cpu().float()[indices]
        inputs_2 = x2.cpu().float()[indices]
    else:
        inputs = x.cpu().float()[indices]

    labels = labels.cpu().float()[indices]
    predictions = predictions.cpu().float()[indices]

    # Apply color map to labels and predictions if color_dict is provided
    if color_dict:
        colored_labels = apply_color_map(labels, color_dict)
        colored_predictions = apply_color_map(predictions, color_dict)
    else:
        # Default behavior: no color map (just grayscale)
        colored_labels = labels.unsqueeze(1).float() / 255.0  # Normalize to [0, 1] for grayscale
        colored_predictions = predictions.unsqueeze(1).float() / 255.0  # Normalize to [0, 1] for grayscale

    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Save and log input images
    if siamese:
        pre_image_path = os.path.join(log_dir, f"pre_image_epoch_{epoch}.png")
        post_image_path = os.path.join(log_dir, f"post_image_epoch_{epoch}.png")
        save_image(inputs_1, pre_image_path, normalize=True, scale_each=True)
        save_image(inputs_2, post_image_path, normalize=True, scale_each=True)
        mlflow.log_artifact(pre_image_path, artifact_path="images")
        mlflow.log_artifact(post_image_path, artifact_path="images")
    else:
        input_image_path = os.path.join(log_dir, f"input_image_epoch_{epoch}.png")
        save_image(inputs, input_image_path, normalize=True, scale_each=True)
        mlflow.log_artifact(input_image_path, artifact_path="images")

    # Save and log labels and predictions
    label_image_path = os.path.join(log_dir, f"labels_epoch_{epoch}.png")
    pred_image_path = os.path.join(log_dir, f"predictions_epoch_{epoch}.png")
    save_image(colored_labels, label_image_path, normalize=False, scale_each=True)
    save_image(colored_predictions, pred_image_path, normalize=False, scale_each=True)
    mlflow.log_artifact(label_image_path, artifact_path="images")
    mlflow.log_artifact(pred_image_path, artifact_path="images")

    model.train()  # Return to training mode if necessary

def save_model(model, ckpt_path="./models", name="model"):
    path = os.path.join(ckpt_path, "{}.pth".format(name))
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
    print(f"\nMetrics ({phase.capitalize()} Phase):\n")
    table = PrettyTable()
    table.field_names = ["Metric", "Value"]

    for metric, value in metrics.items():
        # Convert the value to a float if it's a numpy float
        if hasattr(value, "item"):
            value = value.item()
        table.add_row([metric, f"{value:.4f}"])

    print(table)

############ Utils for dealing with class imbalanced datasets ########################
def define_weighted_random_sampler(dataset, mask_key="post_mask", subset_size=None, seed: int = None):
    """
    Define a WeightedRandomSampler for a segmentation dataset to address class imbalance.

    Args:
        dataset: A segmentation dataset where each sample includes an image and its corresponding mask.
        mask_key: Key to access the mask in the dataset sample (default: "post_mask").
        subset_size: Number of random samples to use for estimating class weights. If None, uses the full dataset.
        seed : seed number 
    Returns:
        sampler: A WeightedRandomSampler for balanced class sampling.
        class_weights: Inversely propotional class weights for imbalance dataset. 
    """
    if seed is not None : random.seed(seed) 
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

    # Assign a weight to each sample based on the class distribution in its mask
    sample_weights = []
    for i, input in tqdm(enumerate(dataset), desc="Assigning sample weights"):
        mask = input[mask_key]
        mask_flat = mask.flatten().numpy()
        unique, counts = np.unique(mask_flat, return_counts=True)
        pixel_weights = np.array([class_weights[cls] for cls in unique])
        sample_weight = np.dot(counts, pixel_weights) / counts.sum()
        sample_weights.append(sample_weight)

    # Create the WeightedRandomSampler
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
        class_weights: Inversely propotional class weights for imbalance dataset. 
    """
    if seed is not None : random.seed(seed) 
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

############ Utils Functions for Fine Tuning Mask-R-CNN ##############################

import math
import sys
import torch
import torch.distributed as dist

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
