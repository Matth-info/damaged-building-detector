import torch
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

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
    model: nn.Module,
    data_loader: DataLoader,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
    max_images: int = 4
):
    model.eval()  # Set model to evaluation mode

    # Select a batch from the data loader
    inputs, labels = next(iter(data_loader))
    inputs, labels = inputs.to(device), labels.to(device)

    # Generate predictions
    with torch.no_grad():
        predictions = model(inputs) # (batch_size, nb_classes, Height, Width)
        predictions = torch.argmax(predictions, dim=1)  # Assuming segmentation with multiple classes (batch_size, 1, Height, Width)

    # Move the data back to CPU for visualization and take a subset of images
    inputs = inputs.cpu()[:max_images]
    labels = labels.cpu()[:max_images]
    predictions = predictions.cpu()[:max_images]

    # Create a grid for each of the input, label, and prediction images
    input_grid = torchvision.utils.make_grid(inputs, normalize=True, scale_each=True)
    label_grid = torchvision.utils.make_grid(labels.unsqueeze(1).float(), normalize=True, scale_each=True)
    pred_grid = torchvision.utils.make_grid(predictions.unsqueeze(1).float(), normalize=True, scale_each=True)

    # Log the grids to TensorBoard
    writer.add_image('Inputs', input_grid, global_step=epoch)
    writer.add_image('Labels', label_grid, global_step=epoch)
    writer.add_image('Predictions', pred_grid, global_step=epoch)

    model.train()  # Switch back to training mode if necessary