import os

import torch
from torch import nn
from pathlib import Path


def initialize_model(
    model_class: nn.Module,
    weights_path: str,
    kwargs: dict,
    device: str = "cpu",
    _logger=None,
):
    """Initialize a model (with its parameters) and load its weights from a specified path.

    Args:
    ----
        model_class (torch.nn.Module): The model class to initialize.
        weights_path (str): Path to the model's weights file.
        kwargs (dict): Model parameters.
        device (str): Device to load the model on ('cpu' or 'cuda').
        _logger (logging._logger, optional): _logger for error and info messages.

    Returns:
    -------
        torch.nn.Module: The initialized model with loaded weights on the specified device.

    """
    try:
        # Initialize the model
        model = model_class(**kwargs)
        model.to(device)

        # Validate weights path
        if weights_path and Path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
            message = f"Loaded weights from {weights_path} onto {device}."
        elif weights_path:
            message = f"Warning: Weights path '{weights_path}' does not exist. Model initialized with random weights."
        else:
            message = "No weights path provided. Model initialized with random weights."

        # Log or print the message
        if _logger:
            _logger.info(message)
        else:
            print(message)

        return model
    except Exception as e:
        error_message = f"Failed to correctly initialize the model: {e}"
        if _logger:
            _logger.error(error_message)
        else:
            print(error_message)
        raise
