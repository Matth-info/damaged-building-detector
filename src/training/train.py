# utils import
from __future__ import annotations

import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

import albumentations as alb
import mlflow

# Do Not Forget to start your (local) MLFlow server (ex : mlflow server --host 127.0.0.1 --port 8080)
import torch
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torchinfo import summary
from tqdm import tqdm

from src.augmentation import augmentation_test_time, augmentation_test_time_is_siamese
from src.metrics import get_stats
from src.testing import model_evaluation
from src.training.utils import (
    display_metrics,
    initialize_optimizer_scheduler,
    load_checkpoint,
    save_checkpoint,
)
from src.utils import BaseLogger, get_logger
from src.utils.visualization import COLOR_DICT, DEFAULT_MAPPING

if TYPE_CHECKING:
    from torch.nn.modules.loss import _Loss
    from torch.utils.data import DataLoader

# Setup _logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
_logger = logging.getLogger(__name__)


class Trainer:
    """Trainer Module.

    This module provides a `Trainer` class to standardize and simplify the training, validation, and testing of PyTorch models.
    Inspired by PyTorch Lightning and Hugging Face's Trainer, it encapsulates common training workflows, including mixed precision
    training, gradient accumulation, learning rate scheduling, and early stopping.

    Key Features:
    - **Training and Validation**: Supports training and validation loops with customizable loss functions and metrics.
    - **Testing**: Includes a testing phase to evaluate the model on unseen data.
    - **Mixed Precision Training**: Supports mixed precision training using PyTorch's `torch.amp` module.
    - **Gradient Accumulation**: Allows training with gradient accumulation for large batch sizes.
    - **Learning Rate Scheduling**: Integrates learning rate schedulers for dynamic learning rate adjustments.
    - **Early Stopping**: Implements early stopping to terminate training when validation performance stops improving.
    - **Siamese Architectures**: Supports Siamese networks for tasks requiring paired inputs (e.g., change detection).
    - **Test-Time Augmentation (TTA)**: Includes test-time augmentation for improved evaluation performance.
    - **Logging and Checkpointing**: Logs metrics and saves model checkpoints during training and validation.
    - **MLFlow Integration**: Logs hyperparameters, metrics, and artifacts to MLFlow for experiment tracking.

    Classes:
        - Trainer: Encapsulates the training, validation, and testing workflows for PyTorch models.

    """

    def __init__(  # noqa: PLR0913
        self,
        model: nn.Module,
        train_dl: DataLoader,
        valid_dl: DataLoader,
        test_dl: DataLoader | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        optimizer_params: dict | None = None,
        scheduler_params: dict | None = None,
        loss_fn: _Loss = nn.CrossEntropyLoss,
        metrics: list[Callable] | None = None,
        nb_epochs: int = 50,
        tracking_uri: str = "http://127.0.0.1:8080",
        experiment_name: str = "experiment",
        log_dir: str = "runs",
        model_dir: str = "models",
        task: str = "semantic segmentation",
        logger_type: Literal["tensorboard", "mlflow"] = "mlflow",
        resume_path: str | None = None,
        early_stopping_params: dict[str, int] | None = None,
        image_key: str = "image",
        mask_key: str = "mask",
        checkpoint_interval: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_classes: int = 2,
        training_log_interval: int = 1,
        reduction: str = "weighted",
        class_weights: list[float] | None = None,
        class_names: list[str] | None = None,
        gradient_accumulation_steps: int = 1,
        *,
        verbose: bool = True,
        debug: bool = False,
        is_mixed_precision: bool = False,
        is_siamese: bool = False,
        tta: bool = False,
        enable_system_metrics: bool = False,
    ) -> None:
        """Initialize the Trainer class with model, data loaders, optimization, and training configuration.

        Args:
            model (nn.Module): The PyTorch model to train.
            train_dl (DataLoader): DataLoader for training data.
            valid_dl (DataLoader): DataLoader for validation data.
            test_dl (DataLoader | None): DataLoader for test data.
            optimizer (torch.optim.Optimizer | None): Optimizer instance.
            scheduler (torch.optim.lr_scheduler._LRScheduler | None): Learning rate scheduler.
            optimizer_params (dict | None): Parameters for the optimizer.
            scheduler_params (dict | None): Parameters for the scheduler.
            loss_fn (_Loss): Loss function.
            metrics (list[Callable] | None): List of metric functions.
            nb_epochs (int): Number of training epochs.
            experiment_name (str): Name of the experiment.
            log_dir (str): Directory for logs.
            model_dir (str): Directory for saving models.
            resume_path (str | None): Path to resume checkpoint.
            early_stopping_params (dict[str, int] | None): Early stopping configuration.
            image_key (str): Key for image in batch.
            mask_key (str): Key for mask in batch.
            checkpoint_interval (int): Interval for saving checkpoints.
            device (str): Device to use ('cuda' or 'cpu').
            num_classes (int): Number of classes.
            training_log_interval (int): Interval for logging during training.
            reduction (str): Reduction method for metrics.
            class_weights (list[float] | None): Class weights for loss/metrics.
            class_names (list[str] | None): Names of the classes.
            tracking_uri (str): MLflow tracking URI.
            task (str): Task description.
            gradient_accumulation_steps (int): Steps for gradient accumulation.
            verbose (bool): Verbosity flag.
            debug (bool): Debug flag.
            is_mixed_precision (bool): Use mixed precision training.
            is_siamese (bool): Use Siamese architecture.
            tta (bool): Use test-time augmentation.
            enable_system_metrics (bool): Enable system metrics logging.
            logger_type (str): Define Logger type.

        """
        self.model = model.to(device)
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl
        self.optimizer, self.scheduler = initialize_optimizer_scheduler(
            self.model,
            optimizer,
            scheduler,
            optimizer_params,
            scheduler_params,
        )
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.nb_epochs = nb_epochs
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.resume_path = resume_path
        self.early_stopping_params = early_stopping_params or {
            "patience": nb_epochs,
        }
        self.image_key = image_key
        self.mask_key = mask_key
        self.verbose = verbose
        self.checkpoint_interval = checkpoint_interval
        self.debug = debug
        self.device = device
        self.num_classes = num_classes
        self.training_log_interval = training_log_interval
        self.is_mixed_precision = is_mixed_precision
        if is_mixed_precision:
            self.scaler = GradScaler()
        self.reduction = reduction
        self.class_weights = class_weights
        self.class_names = class_names
        self.is_siamese = is_siamese
        self.tta = tta

        self.best_val_loss = float("inf")
        self.patience = self.early_stopping_params["patience"]
        self.trigger_times = 0
        self.start_epoch = 0
        self.mode = "multiclass"
        self.max_norm = 1.0
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.track_system_metrics = enable_system_metrics
        self.tracking_uri = tracking_uri
        self.task = task

        # initialize a local loss tracking
        self.history = {"train_loss": [], "val_loss": []}

        # Ensure directories exist
        Path(model_dir).mkdir(exist_ok=True)
        Path(log_dir).mkdir(exist_ok=True)

        # Resume from checkpoint if specified
        if self.resume_path:
            self.start_epoch = (
                load_checkpoint(
                    checkpoint_path=resume_path,
                    model=model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
                + 1
            )

        # Initialize logger
        self.logger = get_logger(logger_type=logger_type)(
            tracking_uri=tracking_uri,
            track_system_metrics=enable_system_metrics,
            experiment_name=experiment_name,
        )

    def training_step(
        self,
        batch: dict,
        step_number: int | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Perform a single training step on a batch of data, supporting Siamese architectures."""
        # Extract inputs and targets from the batch
        if self.is_siamese:
            x1 = batch["pre_image"].to(self.device, non_blocking=True, dtype=torch.float32)
            x2 = batch["post_image"].to(self.device, non_blocking=True, dtype=torch.float32)
        else:
            x = batch[self.image_key].to(self.device, non_blocking=True, dtype=torch.float32)

        y = batch[self.mask_key].to(self.device, non_blocking=True, dtype=torch.int64)

        self.optimizer.zero_grad()

        if self.is_mixed_precision:
            if self.scaler is None:
                raise RuntimeError("GradScaler must be provided for mixed precision training")
            with autocast(device_type=self.device, dtype=torch.float16):
                outputs = self.model(x1, x2) if self.is_siamese else self.model(x)
                loss = self.loss_fn(outputs, y)

            self.scaler.scale(loss).backward()

            if (step_number + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
        else:
            outputs = self.model(x1, x2) if self.is_siamese else self.model(x)
            loss = self.loss_fn(outputs, y) / self.gradient_accumulation_steps
            loss.backward()

            if (step_number + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        loss_value = loss.item()

        # Compute metrics
        metrics_step = {}
        with torch.no_grad():
            preds = outputs.argmax(dim=1).long()
            tp, fp, fn, tn = get_stats(
                output=preds,
                target=y,
                mode=self.mode,
                num_classes=self.num_classes,
            )

            for metric in self.metrics:
                metric_name = metric.__name__
                metrics_step[metric_name] = metric(
                    tp,
                    fp,
                    fn,
                    tn,
                    class_weights=self.class_weights,
                    reduction=self.reduction,
                )

        return loss_value, metrics_step

    def validation_step(self, batch: dict) -> tuple[float, dict[str, float]]:
        """Perform a single validation step on a batch of data."""
        if self.is_siamese:
            x1 = batch["pre_image"].to(self.device, non_blocking=True, dtype=torch.float32)
            x2 = batch["post_image"].to(self.device, non_blocking=True, dtype=torch.float32)
        else:
            x = batch[self.image_key].to(self.device, non_blocking=True, dtype=torch.float32)

        y = batch[self.mask_key].to(self.device, non_blocking=True, dtype=torch.int64)

        # Disable gradients for validation
        with torch.no_grad():
            # Mixed precision support
            if self.is_mixed_precision:
                with autocast(device_type=self.device, dtype=torch.float16):
                    outputs = self.model(x1, x2) if self.is_siamese else self.model(x)
                    vloss = self.loss_fn(outputs, y)
            else:
                outputs = self.model(x1, x2) if self.is_siamese else self.model(x)
                vloss = self.loss_fn(outputs, y)

        vloss_value = vloss.item()

        # Compute metrics
        metrics_step = {}
        with torch.no_grad():
            preds = outputs.argmax(dim=1).long() if len(outputs.shape) == 4 else outputs.long()
            tp, fp, fn, tn = get_stats(
                output=preds,
                target=y,
                mode=self.mode,
                num_classes=self.num_classes,
            )

            for metric in self.metrics:
                metric_name = metric.__name__
                metrics_step[metric_name] = metric(
                    tp,
                    fp,
                    fn,
                    tn,
                    class_weights=self.class_weights,
                    reduction=self.reduction,
                )

        return vloss_value, metrics_step

    def testing_step(self, batch: dict) -> tuple[float, dict[str, float]]:
        """Perform a single validation step on a batch of data."""
        if self.is_siamese:
            x1 = batch["pre_image"].to(self.device, non_blocking=True, dtype=torch.float32)
            x2 = batch["post_image"].to(self.device, non_blocking=True, dtype=torch.float32)
        else:
            x1 = batch[self.image_key].to(self.device, non_blocking=True, dtype=torch.float32)
            x2 = None

        y = batch[self.mask_key].to(self.device, non_blocking=True, dtype=torch.int64)

        # Disable gradients for validation
        with torch.no_grad():
            # Mixed precision support
            if self.is_mixed_precision:
                with autocast(device_type=self.device, dtype=torch.float16):
                    outputs = self._apply_tta_or_forward(x1, x2, self.tta, self.is_siamese)
                    tloss = self.loss_fn(outputs, y)
            else:
                outputs = self._apply_tta_or_forward(x1, x2, self.tta, self.is_siamese)
                tloss = self.loss_fn(outputs, y)

        tloss_value = tloss.item()

        # Compute metrics
        test_metrics_step = {}
        with torch.no_grad():
            expected_shape = 4
            preds = (
                outputs.argmax(dim=1).long()
                if len(outputs.shape) == expected_shape
                else outputs.long()
            )
            tp, fp, fn, tn = get_stats(
                output=preds,
                target=y,
                mode=self.mode,
                num_classes=self.num_classes,
            )

            for metric in self.metrics:
                metric_name = metric.__name__
                test_metrics_step[metric_name] = metric(
                    tp,
                    fp,
                    fn,
                    tn,
                    class_weights=self.class_weights,
                    reduction=self.reduction,
                )

        return tloss_value, test_metrics_step

    def _apply_tta_or_forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor = None,
        tta: bool = False,
        is_siamese: bool = False,
    ) -> torch.Tensor:
        """Handle TTA and forward pass."""
        if tta:
            if is_siamese:
                return augmentation_test_time_is_siamese(
                    model=self.model,
                    images_1=x1,
                    images_2=x2,
                    list_augmentations=[alb.HorizontalFlip(p=1.0), alb.VerticalFlip(p=1.0)],
                    aggregation="mean",
                    device=self.device,
                )
            return augmentation_test_time(
                model=self.model,
                images=x1,
                list_augmentations=[alb.HorizontalFlip(p=1.0), alb.VerticalFlip(p=1.0)],
                aggregation="mean",
                device=self.device,
            )
        return self.model(x1, x2) if is_siamese else self.model(x1)

    def training_epoch(self, epoch: int) -> tuple[float, dict[str, float]]:
        """Perform one epoch of training."""
        _logger.info("Epoch %d", epoch + 1)
        _logger.info("-" * 20)

        self.model.train()
        running_loss = 0.0
        total_metrics = {metric.__name__: 0.0 for metric in self.metrics}
        interval_samples = 0
        steps_per_epoch = math.ceil(len(self.train_dl.dataset) / self.train_dl.batch_size)
        step = 0

        with tqdm(self.train_dl, desc=f"Epoch {epoch + 1}", unit="batch") as t:
            for i, batch in enumerate(t):
                loss_t, metrics_step = self.training_step(batch=batch, step_number=i)

                batch_size = batch["pre_image" if self.is_siamese else self.image_key].size(0)
                running_loss += loss_t * batch_size
                interval_samples += batch_size

                for metric_name in total_metrics:
                    if metric_name in metrics_step:
                        total_metrics[metric_name] += metrics_step[metric_name] * batch_size

                if step % self.training_log_interval == 0:
                    step_metrics = {
                        name: value / max(1, interval_samples)
                        for name, value in total_metrics.items()
                    }
                    step_number = epoch * steps_per_epoch + step
                    self.logger.log_metrics(
                        metrics=step_metrics,
                        step_number=step_number,
                        phase="Training",
                    )
                    self.logger.log_loss(
                        loss_value=running_loss / max(1, interval_samples),
                        step_number=step_number,
                        phase="Training",
                    )

                t.set_postfix({"Loss": loss_t})
                step += 1

        epoch_loss = running_loss / len(self.train_dl.dataset)
        epoch_metrics = {
            name: value / len(self.train_dl.dataset) for name, value in total_metrics.items()
        }

        if self.verbose:
            _logger.info("Epoch %d Training completed. Loss: %.4f", epoch + 1, epoch_loss)
            display_metrics(metrics=epoch_metrics, phase="Training")

        if self.scheduler:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(epoch_loss)
            else:
                self.scheduler.step()

        self.history["train_loss"].append(epoch_loss)

        return epoch_loss, epoch_metrics

    def validation_epoch(self, epoch: int) -> tuple[float, dict[str, float]]:
        """Perform one epoch of validation."""
        _logger.info("Epoch %d - Validation Phase", epoch + 1)
        _logger.info("-" * 20)

        self.model.eval()
        running_loss = 0.0
        interval_samples = 0
        total_metrics = {metric.__name__: 0.0 for metric in self.metrics}
        steps_per_epoch = math.ceil(len(self.valid_dl.dataset) / self.valid_dl.batch_size)
        step = 0

        with (
            torch.no_grad(),
            tqdm(self.valid_dl, desc=f"Validation Epoch {epoch + 1}", unit="batch") as t,
        ):
            for batch in t:
                batch_size = batch["pre_image" if self.is_siamese else self.image_key].size(0)

                vloss, metrics_step = self.validation_step(batch=batch)

                running_loss += vloss * batch_size
                interval_samples += batch_size

                for metric_name in total_metrics:
                    if metric_name in metrics_step:
                        total_metrics[metric_name] += metrics_step[metric_name] * batch_size

                if step % self.training_log_interval == 0:
                    step_metrics = {
                        name: value / max(1, interval_samples)
                        for name, value in total_metrics.items()
                    }
                    step_number = epoch * steps_per_epoch + step
                    self.logger.log_metrics(
                        metrics=step_metrics,
                        step_number=step_number,
                        phase="Validation",
                    )
                    self.logger.log_loss(
                        loss_value=running_loss / max(1, interval_samples),
                        step_number=step_number,
                        phase="Validation",
                    )

                t.set_postfix({"Loss": vloss})
                step += 1

        epoch_vloss = running_loss / len(self.valid_dl.dataset)
        epoch_metrics = {
            name: total / len(self.valid_dl.dataset) for name, total in total_metrics.items()
        }

        if self.verbose:
            _logger.info("Epoch %d Validation completed. Loss: %.4f", epoch + 1, epoch_vloss)
            display_metrics(metrics=epoch_metrics, phase="Validation")

        self.history["val_loss"].append(epoch_vloss)

        return epoch_vloss, epoch_metrics

    def get_hyperparameters(self) -> dict:
        """Extract and return key hyperparameters for run comparison."""
        return {
            "model": self.model.__class__.__name__,
            "optimizer": self.optimizer.__class__.__name__ if self.optimizer else None,
            "optimizer_params": self.optimizer.defaults if self.optimizer else None,
            "scheduler": self.scheduler.__class__.__name__ if self.scheduler else None,
            "scheduler_params": self.scheduler.state_dict() if self.scheduler else None,
            "loss_fn": self.loss_fn.__class__.__name__,
            "nb_epochs": self.nb_epochs,
            "batch_size": self.train_dl.batch_size
            if hasattr(self.train_dl, "batch_size")
            else None,
            "early_stopping": self.early_stopping_params,
            "max_norm": self.max_norm,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "is_mixed_precision": self.is_mixed_precision,
            "class_weights": self.class_weights,
            "mode": self.mode,
            "is_siamese": self.is_siamese,
            "tta": self.tta,
            "device": self.device,
        }

    def handle_checkpointing(self, epoch: int, val_loss: float) -> None:
        """Handle saving checkpoints and best model."""
        if val_loss < self.best_val_loss:
            _logger.info("New best validation loss: %.4f, saving model...", val_loss)
            self.best_val_loss = val_loss
            self.trigger_times = 0
            torch.save(self.model.state_dict(), Path(self.model_dir) / "best_model.pth")
        else:
            self.trigger_times += 1

        if epoch % self.checkpoint_interval == 0:
            save_checkpoint(epoch, self.model, self.optimizer, self.scheduler, "./checkpoints")

    def testing(
        self,
    ) -> tuple[float, dict[str, float]]:
        """Evaluate the model on the test dataset and return loss and metrics."""
        _logger.info("Testing Phase")

        running_loss = 0.0
        interval_samples = 0
        total_metrics = {metric.__name__: 0.0 for metric in self.metrics}  # Initialize totals

        # Iterate through the test dataset
        with tqdm(self.test_dl, desc="Testing", unit="batch") as t:
            for batch in t:
                batch_size = batch["pre_image" if self.is_siamese else self.image_key].size(0)

                # Perform a validation step
                tloss, metrics_step = self.testing_step(batch=batch)

                # Accumulate test loss
                running_loss += tloss * batch_size
                interval_samples += batch_size

                for metric_name in total_metrics:
                    if metric_name in metrics_step:
                        total_metrics[metric_name] += metrics_step[metric_name] * batch_size

                # Update progress bar with running loss
                t.set_postfix({"Loss": tloss})

        # Calculate average loss and metrics for the entire test dataset
        epoch_tloss = running_loss / len(self.test_dl.dataset)
        test_metrics = {
            f"test_{name}": total / len(self.test_dl.dataset)
            for name, total in total_metrics.items()
        }

        return epoch_tloss, test_metrics

    def train(self) -> None:
        """Run the main training loop."""
        # Initialize Experiment
        self.logger.initialize_experiment(
            model=self.model,
            dl=self.train_dl,
            is_siamese=self.is_siamese,
            image_key=self.image_key,
            mask_key=self.mask_key,
            device=self.device,
            nb_epochs=self.nb_epochs,
        )

        try:
            _logger.info("Starting training for %d epochs.", self.nb_epochs)
            overall_start_time = time.time()
            self.logger.log_tags(tags={"task": self.task, "torch_version": torch.__version__})

            self.logger.log_hyperparams(self.get_hyperparameters())
            self.logger.log_model_architecture(self.model)
            _logger.info("Hyperparameters have been logged")

            # Training / Validation phases
            for epoch in range(self.start_epoch, self.nb_epochs):
                start_time = time.time()
                _logger.info("Epoch %d/%d", epoch + 1, self.nb_epochs)

                # Training Step
                train_loss, train_metrics = self.training_epoch(epoch)
                self.history["train_loss"].append(train_loss)

                # Validation Step
                val_loss, val_metrics = self.validation_epoch(epoch)
                self.history["val_loss"].append(val_loss)

                # Update Learning Rate
                if self.scheduler:
                    # lr_value
                    self.logger.log_lr(lr_value=self.scheduler.get_last_lr()[0], epoch=epoch)
                    self.scheduler.step()

                # Save best model & checkpoint
                self.handle_checkpointing(epoch, val_loss)

                # Log training and validation metrics
                steps_per_epoch = math.ceil(len(self.valid_dl.dataset) / self.valid_dl.batch_size)
                step_number = (epoch + 1) * steps_per_epoch
                self.logger.log_metrics(
                    metrics=train_metrics,
                    step_number=step_number,
                    phase="Training",
                )
                self.logger.log_metrics(
                    metrics=val_metrics,
                    step_number=step_number,
                    phase="Validation",
                )

                if epoch % self.training_log_interval == 0:
                    # Log sample images after validation epoch
                    self.logger.log_images(
                        model=self.model,
                        data_loader=self.valid_dl,
                        epoch=epoch,
                        device=self.device,
                        max_images=1,
                        image_key=self.image_key,
                        mask_key=self.mask_key,
                        is_siamese=self.is_siamese,
                        color_dict=DEFAULT_MAPPING,
                    )

                # Logging epoch duration
                epoch_time = time.time() - start_time
                self.logger.log_epoch_time("epoch_time", epoch_time, step=epoch)
                _logger.info("Epoch %d completed in %.2fs", epoch + 1, epoch_time)

                if self.verbose and self.debug:
                    _logger.info("Epoch %d took %.2f seconds", epoch + 1, epoch_time)
                    if torch.cuda.is_available():
                        _logger.info(
                            "GPU memory allocated: %.2f GB",
                            torch.cuda.memory_allocated() / 1024**3,
                        )

                # Early stopping
                if self.trigger_times >= self.patience:
                    _logger.info("Early stopping triggered. Training stopped.")
                    break

            _logger.info("Total training time: %.2fs", time.time() - overall_start_time)

            # log model
            self.logger.log_model(
                model=self.model,
                artifact_path="last_model",
            )

            _logger.info("Best model has been saved to MLflow")

            # Final Testing
            if self.test_dl:
                epoch_tloss, test_metrics = self.testing()
                self.logger.log_metrics(test_metrics, phase="Testing")

                # Testing phase
                test_metrics = model_evaluation(
                    model=self.model,
                    dataloader=self.test_dl,
                    num_classes=self.num_classes,
                    device=self.device,
                    class_names=self.class_names,
                    is_siamese=self.is_siamese,
                    image_key=self.image_key,
                    mask_key=self.mask_key,
                    average_mode="macro",
                    saving_method="mlflow",
                )
            # stop logger run
            self.logger.finalize(status="finished")
        except Exception as e:
            logging.exception(
                "Error during :\nRunning experiment=%s\nRun_id=%s",
                self.logger.get_exp_name(),
                self.logger.get_run_id(),
            )
            self.logger.finalize(status="failed")
