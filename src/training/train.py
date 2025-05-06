# utils import
import logging
import math
import os
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import albumentations as A
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.amp import GradScaler, autocast
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.augmentation import augmentation_test_time, augmentation_test_time_siamese
from src.metrics import compute_model_class_performance, get_stats

from .utils import (
    custom_infer_signature,
    display_metrics,
    initialize_optimizer_scheduler,
    load_checkpoint,
    load_model,
    log_images_to_mlflow,
    log_loss,
    log_metrics,
    log_model,
    save_checkpoint,
    save_model,
)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dl: DataLoader,
        valid_dl: DataLoader,
        test_dl: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        optimizer_params: dict = None,
        scheduler_params: dict = None,
        loss_fn: _Loss = nn.CrossEntropyLoss(),
        metrics: List[Callable] = [],
        nb_epochs: int = 50,
        experiment_name: str = "experiment",
        log_dir: str = "runs",
        model_dir: str = "models",
        resume_path: Optional[str] = None,
        early_stopping_params: Optional[Dict[str, int]] = None,
        image_key: str = "image",
        mask_key: str = "mask",
        verbose: bool = True,
        checkpoint_interval: int = 10,
        debug: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_classes: int = 2,
        training_log_interval: int = 1,
        is_mixed_precision: bool = False,
        reduction: str = "weighted",
        class_weights: List[float] = None,
        class_names: List[str] = None,
        siamese: bool = False,
        tta: bool = False,
    ):
        self.model = model.to(device)
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl
        self.optimizer, self.scheduler = initialize_optimizer_scheduler(
            self.model, optimizer, scheduler, optimizer_params, scheduler_params
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
            "trigger_times": 0,
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
        self.siamese = siamese
        self.tta = tta

        self.best_val_loss = float("inf")
        self.patience = self.early_stopping_params["patience"]
        self.trigger_times = 0
        self.start_epoch = 0
        self.mode = "multiclass"
        self.max_norm = 1.0
        self.gradient_accumulation_steps = 1
        self.track_system_metrics = False

        # initialize a local loss tracking
        self.history = {"train_loss": [], "val_loss": []}

        # Ensure directories exist
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

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

    def training_step(
        self, batch: dict, step_number: int = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Perform a single training step on a batch of data, supporting Siamese architectures.
        """
        # Extract inputs and targets from the batch
        if self.siamese:
            x1 = batch["pre_image"].to(self.device, non_blocking=True, dtype=torch.float32)
            x2 = batch["post_image"].to(self.device, non_blocking=True, dtype=torch.float32)
        else:
            x = batch[self.image_key].to(self.device, non_blocking=True, dtype=torch.float32)

        y = batch[self.mask_key].to(self.device, non_blocking=True, dtype=torch.int64)

        self.optimizer.zero_grad()

        if self.is_mixed_precision:
            assert (
                self.scaler is not None
            ), "GradScaler must be provided for mixed precision training"
            with autocast(device_type=self.device, dtype=torch.float16):
                outputs = self.model(x1, x2) if self.siamese else self.model(x)
                loss = self.loss_fn(outputs, y)

            self.scaler.scale(loss).backward()

            if (step_number + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
        else:
            outputs = self.model(x1, x2) if self.siamese else self.model(x)
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
                output=preds, target=y, mode=self.mode, num_classes=self.num_classes
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

    def validation_step(self, batch: dict) -> Tuple[float, Dict[str, float]]:
        """
        Perform a single validation step on a batch of data.
        """
        if self.siamese:
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
                    outputs = self._apply_tta_or_forward(x1, x2, x, self.tta, self.siamese)
                    vloss = self.loss_fn(outputs, y)
            else:
                outputs = self._apply_tta_or_forward(x1, x2, x, self.tta, self.siamese)
                vloss = self.loss_fn(outputs, y)

        vloss_value = vloss.item()

        # Compute metrics
        metrics_step = {}
        with torch.no_grad():
            preds = outputs.argmax(dim=1).long() if len(outputs.shape) == 4 else outputs.long()
            tp, fp, fn, tn = get_stats(
                output=preds, target=y, mode=self.mode, num_classes=self.num_classes
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

    def _apply_tta_or_forward(self, x1, x2, x, tta, siamese):
        """Helper function to handle TTA and forward pass"""
        if tta:
            if siamese:
                return augmentation_test_time_siamese(
                    model=self.model,
                    images_1=x1,
                    images_2=x2,
                    list_augmentations=[A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)],
                    aggregation="mean",
                    device=self.device,
                )
            return augmentation_test_time(
                model=self.model,
                images=x,
                list_augmentations=[A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)],
                aggregation="mean",
                device=self.device,
            )
        return self.model(x1, x2) if siamese else self.model(x)

    def training_epoch(self, epoch) -> Tuple[float, Dict[str, float]]:
        """
        Perform one epoch of training.
        """
        logging.info(f"Epoch {epoch + 1}")
        logging.info("-" * 20)

        self.model.train()
        running_loss = 0.0
        total_metrics = {metric.__name__: 0.0 for metric in self.metrics}
        interval_samples = 0
        steps_per_epoch = math.ceil(len(self.train_loader.dataset) / self.train_loader.batch_size)
        step = 0

        with tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", unit="batch") as t:
            for batch in t:
                loss_t, metrics_step = self.training_step(batch=batch)

                batch_size = batch[self.image_key].size(0)
                running_loss += loss_t * batch_size
                interval_samples += batch_size

                for metric_name in total_metrics.keys():
                    if metric_name in metrics_step:
                        total_metrics[metric_name] += metrics_step[metric_name] * batch_size

                if step % self.training_log_interval == 0:
                    step_metrics = {
                        name: value / max(1, interval_samples)
                        for name, value in total_metrics.items()
                    }
                    step_number = epoch * steps_per_epoch + step
                    log_metrics(metrics=step_metrics, step_number=step_number, phase="Training")
                    log_loss(
                        loss_value=running_loss / max(1, interval_samples),
                        step_number=step_number,
                        phase="Training",
                    )

                t.set_postfix({"Loss": loss_t})
                step += 1

        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_metrics = {
            name: value / len(self.train_loader.dataset) for name, value in total_metrics.items()
        }

        if self.verbose:
            logging.info(f"Epoch {epoch + 1} Training completed. Loss: {epoch_loss:.4f}")
            display_metrics(metrics=epoch_metrics, phase="Training")

        if self.scheduler:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(epoch_loss)
            else:
                self.scheduler.step()

        self.history["train_loss"].append(epoch_loss)

        return epoch_loss, epoch_metrics

    def validation_epoch(self, epoch) -> Tuple[float, Dict[str, float]]:
        """
        Perform one epoch of validation.
        """
        logging.info(f"Epoch {epoch + 1} - Validation Phase")
        logging.info("-" * 20)

        self.model.eval()
        running_loss = 0.0
        interval_samples = 0
        total_metrics = {metric.__name__: 0.0 for metric in self.metrics}
        steps_per_epoch = math.ceil(len(self.val_loader.dataset) / self.val_loader.batch_size)
        step = 0

        with torch.no_grad():
            with tqdm(self.val_loader, desc=f"Validation Epoch {epoch + 1}", unit="batch") as t:
                for batch in t:
                    batch_size = batch[self.image_key].size(0)

                    vloss, metrics_step = self.validation_step(batch=batch)

                    running_loss += vloss * batch_size
                    interval_samples += batch_size

                    for metric_name in total_metrics.keys():
                        if metric_name in metrics_step:
                            total_metrics[metric_name] += metrics_step[metric_name] * batch_size

                    if step % self.training_log_interval == 0:
                        step_metrics = {
                            name: value / max(1, interval_samples)
                            for name, value in total_metrics.items()
                        }
                        step_number = epoch * steps_per_epoch + step
                        log_metrics(
                            metrics=step_metrics,
                            step_number=step_number,
                            phase="Validation",
                        )
                        log_loss(
                            loss_value=running_loss / max(1, interval_samples),
                            step_number=step_number,
                            phase="Validation",
                        )

                    t.set_postfix({"Loss": vloss})
                    step += 1

        epoch_vloss = running_loss / len(self.val_loader.dataset)
        epoch_metrics = {
            name: total / len(self.val_loader.dataset) for name, total in total_metrics.items()
        }

        if self.verbose:
            logging.info(f"Epoch {epoch + 1} Validation completed. Loss: {epoch_vloss:.4f}")
            display_metrics(metrics=epoch_metrics, phase="Validation")

        self.history["val_loss"].append(epoch_vloss)

        return epoch_vloss, epoch_metrics

    def get_hyperparameters(self):
        """
        Extracts and returns key hyperparameters for run comparison.
        """
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
            "siamese": self.siamese,
            "tta": self.tta,
            "device": self.device,
        }

    def handle_checkpointing(self, epoch: int, val_loss: float):
        """Handles saving checkpoints and best model."""
        if val_loss < self.best_val_loss:
            logging.info(f"New best validation loss: {val_loss:.4f}, saving model...")
            self.best_val_loss = val_loss
            self.trigger_times = 0
            torch.save(self.model.state_dict(), os.path.join(self.model_dir, "best_model.pth"))
        else:
            self.trigger_times += 1

        if epoch % self.checkpoint_interval == 0:
            save_checkpoint(epoch, self.model, self.optimizer, self.scheduler, "./checkpoints")

    def train(self):
        """Main training loop."""

        # Connect MLFlow session to the local server
        mlflow.set_tracking_uri("http: //localhost: 5000")
        # Set Experiment name
        mlflow.set_experiment(self.experiment_name)

        if self.track_system_metrics:
            mlflow.enable_system_metrics_logging()  # track system performance

        # Create a directory for the experiment
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{self.experiment_name}_{timestamp}"

        logging.info(f"Starting training for {self.nb_epochs} epochs.")
        overall_start_time = time.time()

        with mlflow.start_run() as run:
            mlflow.set_tag("mlflow.runName", run_name)
            mlflow.log_params(self.get_hyperparameters())
            logging.info("Hyperparameters have been logged")

            # Training / Validation phases
            for epoch in range(self.start_epoch, self.nb_epochs):
                start_time = time.time()
                logging.info(f"Epoch {epoch + 1}/{self.nb_epochs}")

                # Training Step
                train_loss, train_metrics = self.training_epoch(epoch)

                # Validation Step
                val_loss, val_metrics = self.validation_epoch(epoch)

                if self.track_system_metrics:
                    self.track_system_metrics = False
                    mlflow.disable_system_metrics_logging()

                # Update Learning Rate
                if self.scheduler:
                    mlflow.log_metric(
                        "learning_rate", self.lr_scheduler.get_last_lr()[0], step=epoch
                    )
                    self.scheduler.step()

                # Save best model & checkpoint
                self.handle_checkpointing(epoch, val_loss)

                # Log training and validation metrics
                steps_per_epoch = math.ceil(len(self.valid_dl.dataset) / self.valid_dl.batch_size)
                step_number = (epoch + 1) * steps_per_epoch
                log_metrics(metrics=train_metrics, step_number=step_number, phase="Training")
                log_metrics(metrics=val_metrics, step_number=step_number, phase="Validation")

                # Log sample images after validation epoch
                log_images_to_mlflow(
                    model=self.model,
                    data_loader=self.valid_dl,
                    epoch=epoch,
                    device=self.device,
                    max_images=4,
                    image_key=self.image_key,
                    mask_key=self.mask_key,
                    siamese=self.siamese,
                    color_dict={
                        0: (0, 0, 0),  # Transparent background for class 0
                        1: (0, 255, 0),  # Green with some transparency for "no-damage"
                        2: (
                            255,
                            255,
                            0,
                        ),  # Yellow with some transparency for "minor-damage"
                        3: (
                            255,
                            126,
                            0,
                        ),  # Orange with some transparency for "major-damage"
                        4: (255, 0, 0),  # Red with some transparency for "destroyed"
                    },
                    log_dir="../runs/mlflow_logs",
                )

                # Logging epoch duration
                epoch_time = time.time() - start_time
                mlflow.log_metric("epoch_time", epoch_time, step=epoch)
                logging.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")

                if self.verbose and self.debug:
                    logging.info(f"Epoch {epoch + 1} took {epoch_time:.2f} seconds")
                    if torch.cuda.is_available():
                        logging.info(
                            f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
                        )

                # Early stopping
                if self.trigger_times >= self.patience:
                    logging.info("Early stopping triggered. Training stopped.")
                    break

        logging.info(f"Total training time: {time.time() - overall_start_time:.2f}s")

        # Final Testing
        if self.test_dl:
            epoch_tloss, test_metrics = self.testing()
            log_metrics(test_metrics, step_number=None, phase="Testing")

            # Testing phase
            compute_model_class_performance(
                model=self.model,
                dataloader=self.test_dl,
                num_classes=self.num_classes,
                device=self.device,
                class_names=self.class_names,
                siamese=self.siamese,
                image_key=self.image_key,
                mask_key=self.mask_key,
                average_mode="macro",
                mlflow_bool=True,
            )

    @classmethod
    def testing(
        self,
    ) -> Tuple[float, Dict[str, float]]:
        logging.info("Testing Phase")

        running_loss = 0.0
        interval_samples = 0
        total_metrics = {metric.__name__: 0.0 for metric in self.metrics}  # Initialize totals

        # Iterate through the test dataset
        with tqdm(self.test_dl, desc="Testing", unit="batch") as t:
            for batch in t:
                batch_size = (
                    batch[self.image_key].size(0)
                    if self.siamese
                    else batch[self.image_key].size(0)
                )

                # Perform a validation step
                tloss, metrics_step = self.validation_step(batch=batch)

                # Accumulate test loss
                running_loss += tloss * batch_size
                interval_samples += batch_size

                for metric_name in total_metrics.keys():
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
