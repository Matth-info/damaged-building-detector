# Core Python imports
import json
import os
import random
from typing import Union

import mlflow

# PyTorch imports
import torch
import torch.nn as nn
from mlflow.models.signature import infer_signature

# PyTorch Vision imports
from torch.utils.data import DataLoader
from torchinfo import summary

from src.augmentation import Augmentation_pipeline
from src.utils.visualization import DEFAULT_MAPPING, apply_color_map


def log_metrics(metrics: dict, step_number: int, phase: str = "Validation"):
    """
    Logs each metric in the dictionary to MLFlow.

    Parameters:
    - metrics: Dictionary of metric name and value pairs.
    - step_number: The current step number.
    - phase: 'Validation' or 'Training', used to distinguish metrics in TensorBoard.
    """
    mlflow.log_metrics(
        {f"{phase}/{metric_name}": value for metric_name, value in metrics.items()},
        step=step_number,
    )


def log_loss(loss_value: float, step_number: int, phase: str = "Validation"):
    """
    Logs loss value to MLFlow.

    Parameters:
    - loss_value: current loss value
    - step_number: The current step number.
    - phase: 'Validation' or 'Training', used to distinguish metrics in TensorBoard.
    """
    mlflow.log_metric(f"{phase}/loss", f"{loss_value:2f}", step=step_number)


def log_model(model, artifact_path="best_model", signature=None, input_example=None):
    mlflow.pytorch.log_model(
        model,
        artifact_path=artifact_path,
        signature=signature,
        input_example=input_example,
    )


def log_tags(tags: dict[str]):
    mlflow.set_tags(tags)


def log_model_architecture(model: nn.Module):
    # Log model summary.
    with open("model_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(summary(model)))
    mlflow.log_artifact("model_summary.txt")
    os.remove("model_summary.txt")


def log_data_augmentation(aug_pipe: Augmentation_pipeline):
    file_path = aug_pipe.save(data_format="json", folder_path="./runs")
    mlflow.log_artifact(file_path)


def log_images(
    model: torch.nn.Module,
    data_loader: DataLoader,
    epoch: int,
    device: Union[str, torch.device],
    max_images: int = 4,
    image_key: str = "image",
    mask_key: str = "mask",
    siamese: bool = False,
    color_dict: dict = DEFAULT_MAPPING,
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

    # Save and log input images
    for i in range(len(labels)):
        if siamese:
            mlflow.log_image(
                image=inputs_1[i].numpy().transpose(1, 2, 0), key="pre_image", step=epoch
            )
            mlflow.log_image(
                image=inputs_2[i].numpy().transpose(1, 2, 0), key="post_image", step=epoch
            )
        else:
            mlflow.log_image(image=inputs[i].numpy().transpose(1, 2, 0), key="image", step=epoch)
        # Save and log labels and predictions
        mlflow.log_image(
            image=colored_labels[i].numpy().transpose(1, 2, 0), key="label", step=epoch
        )
        mlflow.log_image(
            image=colored_predictions[i].numpy().transpose(1, 2, 0), key="prediction", step=epoch
        )

    model.train()  # Return to training mode if necessary


def custom_infer_signature(
    model,
    data_loader,
    siamese: bool = False,
    image_key: str = "image",
    mask_key: str = "mask",
    device: str = "cuda",
):
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
    signature = infer_signature(model_input=example_inputs, model_output=predictions)
    return signature, example_inputs
