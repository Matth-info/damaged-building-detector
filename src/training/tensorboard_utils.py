# Core Python imports
import random
from typing import Union

# Third-party libraries
import numpy as np

# PyTorch imports
import torch
import torch.nn as nn

# PyTorch Vision imports
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Local application/library imports
from src.utils.visualization import apply_color_map


def log_graph(
    writer: SummaryWriter,
    model: torch.nn.Module,
    siamese: bool = False,
    device: str = "cuda",
    input_shape: tuple = (1, 3, 512, 512),
):
    """
    Logs the model graph to TensorBoard. Supports both single-input and siamese models.

    Args:
        writer (SummaryWriter): The TensorBoard SummaryWriter.
        model (torch.nn.Module): The model to log.
        siamese (bool): Flag to indicate if the model is Siamese.
        device (str): The device (cuda or cpu) where the model and tensors should be moved.
        input_shape (tuple): The shape of the input tensor (default: (1, 3, 512, 512)).

    """
    model.to(device)  # Move model to the correct device

    # For Siamese models, we create two input tensors
    if siamese:
        x1 = torch.randn(*input_shape).to(device)
        x2 = torch.randn(*input_shape).to(device)
        writer.add_graph(model, (x1, x2))
    else:
        x1 = torch.randn(*input_shape).to(device)
        writer.add_graph(model, x1)


def log_images_to_tensorboard(
    model: torch.nn.Module,
    data_loader: DataLoader,
    writer: SummaryWriter,
    epoch: int,
    device: Union[str, torch.device],
    max_images: int = 4,
    image_key: str = "image",
    mask_key: str = "mask",
    siamese: bool = False,
    color_dict=None,
):
    """
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
        colored_predictions = (
            predictions.unsqueeze(1).float() / 255.0
        )  # Normalize to [0, 1] for grayscale

    # Create grids for inputs, labels, and predictions
    if siamese:
        input_grid_1 = torchvision.utils.make_grid(inputs_1, normalize=True, scale_each=True)
        input_grid_2 = torchvision.utils.make_grid(inputs_2, normalize=True, scale_each=True)
        writer.add_image(tag="Inputs/Pre_Image", img_tensor=input_grid_1, global_step=epoch)
        writer.add_image(tag="Inputs/Post_Image", img_tensor=input_grid_2, global_step=epoch)
    else:
        input_grid = torchvision.utils.make_grid(inputs, normalize=False, scale_each=True)
        writer.add_image(tag="Inputs", img_tensor=input_grid, global_step=epoch)

    label_grid = torchvision.utils.make_grid(colored_labels, normalize=False, scale_each=True)
    pred_grid = torchvision.utils.make_grid(colored_predictions, normalize=False, scale_each=True)

    # Log labels and predictions
    writer.add_image(tag="Labels", img_tensor=label_grid, global_step=epoch)
    writer.add_image(tag="Predictions", img_tensor=pred_grid, global_step=epoch)

    model.train()  # Return to training mode if necessary
