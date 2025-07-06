# Core Python imports
from __future__ import annotations

import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import mlflow

# PyTorch imports
import torch
from mlflow.models.signature import ModelSignature, infer_signature
from torch import nn

# PyTorch Vision imports
from torchinfo import summary

from src.utils.visualization import DEFAULT_MAPPING, apply_color_map

if TYPE_CHECKING:
    import pandas as pd
    from mlflow.tracking import MlflowClient
    from torch.utils.data import DataLoader

    from src.augmentation import Augmentation_pipeline

from .logger import BaseLogger

_logger = logging.getLogger(__name__)


def custom_infer_signature(
    model: nn.Module,
    data_loader: DataLoader,
    image_key: str = "image",
    device: str = "cuda",
    *,
    is_siamese: bool = False,
) -> tuple[ModelSignature, dict[str, Any]]:
    """Infers the MLflow signature for a given model using a single batch of data from the dataloader.

    Args:
    ----
        model (torch.nn.Module): The model for which to infer the signature.
        data_loader (DataLoader): DataLoader providing batches of input data.
        is_siamese (bool): Whether the model is a Siamese model (two inputs).
        image_key (str): Key in the batch for input images.
        mask_key (str): Key in the batch for target masks.
        device (str): Device to perform inference ('cuda' or 'cpu').

    Returns:
    -------
        signature: The inferred signature for the model.

    """
    # Get a single batch from the data loader
    batch = next(iter(data_loader))

    # Move the model to the specified device and set it to evaluation mode
    model.to(device)
    model.eval()

    with torch.no_grad():
        # Handle the inputs based on whether the model is Siamese or not
        if is_siamese:
            pre_image = batch["pre_image"].to(device)  # Move to device
            post_image = batch["post_image"].to(device)

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


class MLflowLogger(BaseLogger):
    """Custom MLFlow logger."""

    def __init__(
        self,
        experiment_name: str = "logs",
        run_name: str | None = None,
        tracking_uri: str | None = os.getenv("MLFLOW_TRACKING_URI"),
        tags: dict[str, Any] | None = None,
        *,
        log_model: Literal[True, False, "all"] = False,
        artifact_location: str | None = "./mlruns",
        run_id: str | None = None,
        track_system_metrics: bool = True,
    ) -> None:
        """Initialize the MLflow utility class for experiment tracking.

        Args:
            experiment_name (str): Name of the MLflow experiment. Defaults to "logs".
            run_name (str | None): Optional name for the MLflow run.
            tracking_uri (str | None): URI for the MLflow tracking server. Defaults to the value of the "MLFLOW_TRACKING_URI" environment variable.
            tags (dict[str, Any] | None): Optional dictionary of tags to associate with the run.
            log_model (Literal[True, False, "all"]): Whether to log models. Can be True, False, or "all". Defaults to False.
            artifact_location (str | None): Optional location to store artifacts (default "./mlruns").
            run_id (str | None): Optional existing run ID to resume logging to.
            track_system_metrics (bool): Whether to track system metrics. Defaults to True.

        Raises:
            ImportError: If the mlflow package is not installed.

        """
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
        except ImportError:
            logging.exception("Mlflow package is not installed.")

        self._experiment_name = experiment_name
        self._experiment_id: str | None = None
        self._tracking_uri = tracking_uri
        self._run_name = run_name
        self._run_id = run_id
        self.tags = tags or {}
        self._log_model = log_model
        self._logged_model_time: dict[str, float] = {}
        self._artifact_location = artifact_location
        self._initialized = False
        self.track_system_metrics = track_system_metrics

        self._mlflow_client = MlflowClient(tracking_uri)

    def get_run_id(self) -> str:
        """Get run id."""
        return self._run_id

    def get_exp_name(self) -> str:
        """Get experiment name."""
        return self._experiment_name

    def initialize_experiment(
        self,
        model: torch.Module,
        dl: DataLoader,
        *,
        is_siamese: bool,
        image_key: str,
        mask_key: str,
        device: str,
        nb_epochs: int,
    ) -> None:
        """Initialize an MLflow experiment, set up tracking, and infer model signature.

        Args:
            model (torch.Module): The model to be tracked.
            dl (DataLoader): DataLoader for input data.
            is_siamese (bool): Whether the model is a Siamese model.
            image_key (str): Key for input images in the batch.
            mask_key (str): Key for target masks in the batch.
            device (str): Device to use for inference.
            nb_epochs (int): Number of training epochs.
        """
        if self._initialized:
            _logger.info("Experiment already initialized.")
            return

        # Setup tracking URI and experiment
        if self._tracking_uri:
            mlflow.set_tracking_uri(self._tracking_uri)

        # Check or create experiment
        expt = self._mlflow_client.get_experiment_by_name(self._experiment_name)
        if expt and expt.lifecycle_stage != "deleted":
            self._experiment_id = expt.experiment_id
        else:
            _logger.warning("Experiment '%s' not found. Creating it.", self._experiment_name)
            self._experiment_id = self._mlflow_client.create_experiment(
                name=self._experiment_name, artifact_location=self._artifact_location
            )

        mlflow.set_experiment(self._experiment_name)

        # System metrics
        if self.track_system_metrics:
            mlflow.enable_system_metrics_logging()

        # Run name and tags
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        # strategy to build a run name
        # model class / dataset /
        model_name = model.__class__.__name__
        dataset_name = dl.dataset.__class__.__name__
        self._run_name = self._run_name or f"{model_name}_{dataset_name}_{timestamp}"

        if self._run_name:
            from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

            if MLFLOW_RUN_NAME in self.tags:
                _logger.warning(
                    "Overwriting tag %s with run name '%s'", MLFLOW_RUN_NAME, self._run_name
                )
            self.tags[MLFLOW_RUN_NAME] = self._run_name

        # Prepare tags with resolver
        resolve_tags = _get_resolve_tags()
        resolved_tags = resolve_tags(self.tags)

        # Clean any existing run
        mlflow.end_run()

        # Create a new run
        run = self._mlflow_client.create_run(experiment_id=self._experiment_id, tags=resolved_tags)
        self._run_id = run.info.run_id

        _logger.info("Started MLflow run: %s", self._run_id)
        _logger.info("Training will run for %d epochs.", nb_epochs)

        # Optional: Infer model input/output signature
        try:
            self.signature, _ = custom_infer_signature(
                model,
                dl,
                is_siamese,
                image_key,
                mask_key,
                device,
            )
            _logger.info("Model signature inferred successfully.")
        except (RuntimeError, ValueError) as e:
            _logger.warning("Model signature inference failed: %s", e)

        self._initialized = True
        _logger.info("MLflow experiment initialized successfully.")

    def log_hyperparams(self, params: dict) -> None:
        """Log Hyperparameters into MLFLow."""
        from mlflow.entities import Param

        # flatten params
        params_list = [Param(key=k, value=str(v)[:250]) for k, v in params.items()]
        # Log in chunks of 100 parameters (the maximum allowed by MLflow).
        mlflow.log_params(params, run_id=self.get_run_id())

    def log_metrics(
        self, metrics: dict, step_number: int | None = None, phase: str = "Validation"
    ) -> None:
        """Log each metric in the dictionary to MLFlow.

        Args:
            metrics (dict): Dictionary of metric name and value pairs.
            step_number (int): mlflow.log_metric: The current step number.
            phase (str, optional): 'Validation' or 'Training', used to distinguish metrics in MLflow. Defaults to "Validation".

        """
        mlflow.log_metrics(
            {f"{phase}/{metric_name}": value for metric_name, value in metrics.items()},
            step=step_number,
        )

    def log_lr(self, lr_value: float, epoch: int) -> None:
        """Logging Learning Rate to MLFlow."""
        mlflow.log_metric("learning_rate", lr_value, step=epoch)

    def log_epoch_time(self, epoch_time: datetime, epoch: int) -> None:
        """Logging Epoch Time to MLFlow."""
        mlflow.log_metric("epoch_time", epoch_time, step=epoch)

    def log_loss(self, loss_value: float, step_number: int, phase: str = "Validation") -> None:
        """Log loss value to MLFlow.

        Parameters
        ----------
        - loss_value: current loss value
        - step_number: The current step number.
        - phase: 'Validation' or 'Training', used to distinguish metrics in TensorBoard.

        """
        mlflow.log_metric(f"{phase}/loss", loss_value, step=step_number)

    def log_model(
        self,
        model: nn.Module,
        artifact_path: str = "best_model",
    ) -> None:
        """Log a PyTorch model to MLflow.

        Parameters
        ----------
        model (torch.nn.Module): The PyTorch model to be logged.
        artifact_path (str, optional): The run-relative artifact path to save the model under. Defaults to "best_model".
        signature (mlflow.models.signature.ModelSignature, optional): Model signature describing input and output schema. Defaults to None.

        """
        mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path,
            signature=self.signature,
        )

    def log_tags(self, tags: dict[str]) -> None:
        """Log tags to MLflow.

        Args:
            tags (dict[str]): Tags to be logged

        Returns:
            None

        """
        mlflow.set_tags(tags)

    def log_model_architecture(self, model: nn.Module) -> None:
        """Log pytorch model architecture to MLflow.

        Args:
            model (nn.Module): Pytorch model to log its architecture

        """
        # Log model summary.
        with Path.open("model_summary.txt", "w", encoding="utf-8") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_summary.txt")
        Path("model_summary.txt").unlink()

    def log_data_augmentation(self, aug_pipe: Augmentation_pipeline) -> None:
        """Log Custom Data Augmentation pipeline config to MLflow.

        Args:
            aug_pipe (Augmentation_pipeline): Augmentation pipeline to log

        """
        file_path = aug_pipe.save(data_format="json", folder_path="./runs")
        mlflow.log_artifact(file_path)

    def log_images(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        epoch: int,
        device: str | torch.device,
        max_images: int = 4,
        image_key: str = "image",
        mask_key: str = "mask",
        *,
        is_siamese: bool = False,
        color_dict: dict = DEFAULT_MAPPING,
    ) -> None:
        """Log images, labels, and model predictions to MLflow.

        Args:
        ----
            model (torch.nn.Module): The model being evaluated.
            data_loader (DataLoader): DataLoader providing batches of data.
            epoch (int): Current training epoch.
            device (Union[str, torch.device]): Device to perform computations on.
            max_images (int): Maximum number of images to log.
            image_key (str): Key in the batch dictionary for input images.
            mask_key (str): Key in the batch dictionary for target masks.
            is_siamese (bool): Whether the model is a Siamese model.
            color_dict (dict, optional): Dictionary for color mapping.

        """
        model.eval()  # Set model to evaluation mode

        # Get a single batch from the data loader
        batch = next(iter(data_loader))
        if is_siamese:
            x1 = batch["pre_image"].to(device)  # Pre-disaster image
            x2 = batch["post_image"].to(device)  # Post-disaster image
            labels = batch[mask_key].to(device)  # Target mask
        else:
            x = batch[image_key].to(device)  # Single input image
            labels = batch[mask_key].to(device)  # Target mask

        # Generate predictions
        with torch.no_grad():
            predictions = (
                model(x1, x2) if is_siamese else model(x)
            )  # Forward pass for Siamese model
            predictions = torch.argmax(predictions, dim=1)

        # Select a random subset of images
        batch_size = labels.size(0)
        num_images = min(max_images, batch_size)  # Ensure max_images doesn't exceed batch size
        indices = random.sample(range(batch_size), num_images)  # Randomly select indices

        # Move data back to CPU for visualization and limit the number of images
        if is_siamese:
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
            colored_labels = (
                labels.unsqueeze(1).float() / 255.0
            )  # Normalize to [0, 1] for grayscale
            colored_predictions = (
                predictions.unsqueeze(1).float() / 255.0
            )  # Normalize to [0, 1] for grayscale

        # Save and log input images
        for i in range(len(labels)):
            if is_siamese:
                mlflow.log_image(
                    image=inputs_1[i].numpy().transpose(1, 2, 0),
                    key="pre_image",
                    step=epoch,
                )
                mlflow.log_image(
                    image=inputs_2[i].numpy().transpose(1, 2, 0),
                    key="post_image",
                    step=epoch,
                )
            else:
                mlflow.log_image(
                    image=inputs[i].numpy().transpose(1, 2, 0), key="image", step=epoch
                )
            # Save and log labels and predictions
            mlflow.log_image(
                image=colored_labels[i].numpy().transpose(1, 2, 0),
                key="label",
                step=epoch,
            )
            mlflow.log_image(
                image=colored_predictions[i].numpy().transpose(1, 2, 0),
                key="prediction",
                step=epoch,
            )

        model.train()  # Return to training mode if necessary

    def log_table(self, data: pd.DataFrame, artifact_file: str) -> None:
        """Logging dataframe into MLFlow."""
        mlflow.log_table(data, artifact_file=artifact_file)

    def log_artifacts(self, filepath: str) -> None:
        """Logging artifacts into MLFlow."""
        if Path(filepath).is_file():
            mlflow.log_artifact(filepath)

    def finalize(self, status: Literal["failed", "finished"] = "finished") -> None:
        """Terminate a MLFlow run."""
        if not self._initialized:
            return
        if status == "failed":
            status = "FAILED"
        elif status == "finished":
            status = "FINISHED"

        if self._mlflow_client.get_run(self.get_run_id()):
            self._mlflow_client.set_terminated(self.get_run_id(), status)
        mlflow.end_run()  # ensure the run is closed.


def _get_resolve_tags() -> Callable:
    from mlflow.tracking import context

    # before v1.1.0
    if hasattr(context, "resolve_tags"):
        from mlflow.tracking.context import resolve_tags
    # since v1.1.0
    elif hasattr(context, "registry"):
        from mlflow.tracking.context.registry import resolve_tags
    else:

        def resolve_tags(tags: dict):  # noqa: ANN202
            return tags

    return resolve_tags
