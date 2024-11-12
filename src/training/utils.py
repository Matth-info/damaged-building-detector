import torch
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from typing import Union


def log_metrics(writer: SummaryWriter, metrics: dict, epoch_number: int, phase: str = 'Validation'):
    """
    Logs each metric in the dictionary to TensorBoard.

    Parameters:
    - writer: The SummaryWriter instance.
    - metrics: Dictionary of metric name and value pairs.
    - epoch_number: The current epoch number.
    - phase: 'Validation' or 'Training', used to distinguish metrics in TensorBoard.
    """
    for metric_name, value in metrics.items():
        writer.add_scalar(f'{phase}/{metric_name}', value, epoch_number)


def log_images_to_tensorboard(
    model: torch.nn.Module,
    data_loader: DataLoader,
    writer: SummaryWriter,
    epoch: int,
    device: Union[str, torch.device],
    max_images: int = 4
):
    model.eval()  # Set model to evaluation mode

    # Get a single batch from the data loader
    batch = next(iter(data_loader))
    inputs, labels = batch["image"].to(device), batch["mask"].to(device)

    # Generate predictions
    with torch.no_grad():
        predictions = model(inputs)  # (batch_size, nb_classes, Height, Width)
        predictions = torch.argmax(predictions, dim=1)  # Reduce to (batch_size, Height, Width)

    # Move data back to CPU for visualization and limit the number of images
    inputs = inputs.cpu().float()[:max_images]
    labels = labels.cpu().float()[:max_images]
    predictions = predictions.cpu().float()[:max_images]

    # Create grids for inputs, labels, and predictions
    input_grid = torchvision.utils.make_grid(inputs, normalize=True, scale_each=True)
    label_grid = torchvision.utils.make_grid(labels.unsqueeze(1), normalize=False, scale_each=True)
    pred_grid = torchvision.utils.make_grid(predictions.unsqueeze(1), normalize=False, scale_each=True)

    # Log grids to TensorBoard
    writer.add_image(tag='Inputs', img_tensor=input_grid, global_step=epoch)
    writer.add_image(tag='Labels', img_tensor=label_grid, global_step=epoch)
    writer.add_image(tag='Predictions', img_tensor=pred_grid, global_step=epoch)

    model.train()  # Return to training mode if necessary