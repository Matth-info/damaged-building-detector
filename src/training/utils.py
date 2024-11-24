import torch
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from typing import Union
import os
from prettytable import PrettyTable


def log_metrics(
    writer: SummaryWriter, metrics: dict, step_number: int, phase: str = "Validation"
):
    """
    Logs each metric in the dictionary to TensorBoard.

    Parameters:
    - writer: The SummaryWriter instance.
    - metrics: Dictionary of metric name and value pairs.
    - step_number: The current step number.
    - phase: 'Validation' or 'Training', used to distinguish metrics in TensorBoard.
    """
    for metric_name, value in metrics.items():
        writer.add_scalar(f"{phase}/{metric_name}", value, step_number)

def log_loss(
    writer: SummaryWriter, loss_value: float, step_number: int, phase: str = "Validation"
):
    """
    Logs loss value to TensorBoard.

    Parameters:
    - writer: The SummaryWriter instance.
    - loss_value: current loss value
    - step_number: The current step number.
    - phase: 'Validation' or 'Training', used to distinguish metrics in TensorBoard.
    """
    writer.add_scalar(f"{phase}/Loss", loss_value, step_number)


import random
def log_images_to_tensorboard(
    model: torch.nn.Module,
    data_loader: DataLoader,
    writer: SummaryWriter,
    epoch: int,
    device: Union[str, torch.device],
    max_images: int = 4,
):
    model.eval()  # Set model to evaluation mode

    # Get a single batch from the data loader
    batch = next(iter(data_loader))
    inputs, labels = batch["image"].to(device), batch["mask"].to(device)

    # Generate predictions
    with torch.no_grad():
        predictions = model(inputs)  # (batch_size, nb_classes, Height, Width)
        predictions = torch.argmax(
            predictions, dim=1
        )  # Reduce to (batch_size, Height, Width)

    # Select a random subset of images
    num_images = min(max_images, inputs.size(0))  # Ensure max_images doesn't exceed batch size
    indices = random.sample(range(inputs.size(0)), num_images)  # Randomly select indices

    # Move data back to CPU for visualization and limit the number of images
    inputs = inputs.cpu().float()[indices]
    labels = labels.cpu().float()[indices]
    predictions = predictions.cpu().float()[indices]

    # Create grids for inputs, labels, and predictions
    input_grid = torchvision.utils.make_grid(inputs, normalize=True, scale_each=True)
    label_grid = torchvision.utils.make_grid(
        labels.unsqueeze(1), normalize=False, scale_each=True
    )
    pred_grid = torchvision.utils.make_grid(
        predictions.unsqueeze(1), normalize=False, scale_each=True
    )

    # Log grids to TensorBoard
    writer.add_image(tag="Inputs", img_tensor=input_grid, global_step=epoch)
    writer.add_image(tag="Labels", img_tensor=label_grid, global_step=epoch)
    writer.add_image(tag="Predictions", img_tensor=pred_grid, global_step=epoch)

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
