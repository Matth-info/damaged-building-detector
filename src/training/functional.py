# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.nn.modules.loss import _Loss

# utils import
from datetime import datetime
import os
from typing import List, Callable, Dict, Optional, Tuple
from .utils import (
    log_metrics,
    log_loss,
    log_images_to_mlflow,
    log_model,
    save_model,
    load_model,
    display_metrics,
    custom_infer_signature,
    save_checkpoint,
    load_checkpoint,
    initialize_optimizer_scheduler
)

from metrics import compute_model_class_performance

from .augmentations import augmentation_test_time, augmentation_test_time_siamese
import albumentations as A

import time
import numpy as np
import math
import logging 
from tqdm import tqdm 

# from custom metrics and losses
from metrics import get_stats

# mlflow 
import mlflow

# Configure logging
logging.basicConfig(level=logging.INFO)


def training_step(
    model: nn.Module,
    batch: dict,
    loss_fn: _Loss,
    optimizer: optim.Optimizer,
    metrics: List[Callable],
    step_number: int,
    image_key: str = "image",
    mask_key: str = "mask",
    device: str = "cuda",
    is_mixed_precision: bool = False,
    scaler: torch.amp.GradScaler = None,
    mode: str = "multiclass",
    num_classes: int = 2,
    reduction: str = "weighted",
    class_weights: List[float] = [0.1, 0.9],
    max_norm: float = 1.0,
    siamese: bool = False
) -> Tuple[float, Dict[str, float]]:
    """
    Perform a single training step on a batch of data, supporting Siamese architectures.

    Args:
        model (nn.Module): The model being trained.
        batch (dict): The input batch with keys for pre- and post-disaster images.
        loss_fn (_Loss): The loss function to optimize.
        optimizer (optim.Optimizer): The optimizer for the model parameters.
        metrics (List[Callable]): List of metric functions to compute.
        step_number (int): Current training step number.
        image_key (str): Key in the batch dictionary for the input image.
        mask_key (str): Key in the batch dictionary for the target mask.
        device (str): The device to run the computation on ("cuda" or "cpu").
        is_mixed_precision (bool): Whether to use mixed precision.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision.
        mode (str): Metric computation mode, e.g., "multiclass".
        num_classes (int): Number of classes for metrics (if applicable).
        reduction (str): Reduction method for loss.
        max_norm (float): Max norm for gradient clipping.
        siamese (bool): Whether to perform Siamese training.

    Returns:
        Tuple[float, Dict[str, float]]: The loss value and a dictionary of computed metrics.
    """
    # Extract inputs and targets from the batch
    if siamese:
        x1 = batch["pre_image"].to(device, non_blocking=True, dtype=torch.float32)  # Pre-disaster image
        x2 = batch["post_image"].to(device, non_blocking=True, dtype=torch.float32)  # Post-disaster image
    else:
        x = batch[image_key].to(device, non_blocking=True, dtype=torch.float32)  # Single input image e.g., FloatTensor (B, C, H, W)
    
    y = batch[mask_key].to(device, non_blocking=True, dtype=torch.int64)  # Target mask e.g., LongTensor (B, H, W)

    optimizer.zero_grad()

    if is_mixed_precision:
        assert scaler is not None, "GradScaler must be provided for mixed precision training"
        with autocast(device_type=device, dtype=torch.float16):
            if siamese:
                outputs = model(x1, x2)  # Forward pass for Siamese
            else:
                outputs = model(x)  # Forward pass for standard model
            # Compute the loss and its gradients
            loss = loss_fn(outputs, y)
            # adjust the learning weights
        scaler.scale(loss).backward()  # compute gradient in float16 and multiple gradient by a scale factor
        # avoiding underflow = minor change because of weight format
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        if siamese:
            outputs = model(x1, x2)  # Forward pass for Siamese
        else:
            outputs = model(x)  # Forward pass for standard model

        loss = loss_fn(outputs, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()

    loss_value = loss.item()

    # Compute metrics (only at intervals)
    metrics_step = {}

    with torch.no_grad():
        # Get predictions from model outputs
        preds = outputs.argmax(dim=1).long()  # Shape: (B, H, W)

        # Calculate statistics for metrics (e.g., confusion matrix components)
        tp, fp, fn, tn = get_stats(
            output=preds, target=y, mode=mode, num_classes=num_classes
        )

        # Compute each metric
        for metric in metrics:
            metric_name = metric.__name__
            metrics_step[metric_name] = metric(
                tp, fp, fn, tn, class_weights=class_weights, reduction=reduction
            )

    return loss_value, metrics_step


def validation_step(
    model: nn.Module,
    batch: dict,
    loss_fn: _Loss,
    metrics: List[Callable],
    image_key: str = "image",
    mask_key: str = "mask",
    device: str = "cuda",
    is_mixed_precision: bool = False,
    mode: str = "multiclass",
    num_classes: int = 2,
    reduction: str = "weighted",
    class_weights: List[float] = None, 
    tta: bool = False,
    siamese: bool = False
) -> Tuple[float, Dict[str, float]]:
    """
    Perform a single validation step on a batch of data.

    Args:
        model (nn.Module): The model being validated.
        batch (dict): The input batch with keys `image_key` and `mask`.
        loss_fn (_Loss): The loss function used during validation.
        metrics (List[Callable]): List of metric functions to compute.
        image_key (str): Key in the batch dictionary for the input image.
        device (str): The device to run the computation on ("cuda" or "cpu").
        is_mixed_precision (bool): Whether to use mixed precision.
        mode (str): Metric computation mode, e.g., "multiclass".
        num_classes (int): Number of classes for metrics (if applicable).
        reduction (str): Reduction method for loss.
        siamese (bool): Whether to perform Siamese training.

    Returns:
        Tuple[float, Dict[str, float]]: Validation loss and a dictionary of computed metrics.
    """

    if siamese:
        x1 = batch["pre_image"].to(device, non_blocking=True, dtype=torch.float32)  # Pre-disaster image
        x2 = batch["post_image"].to(device, non_blocking=True, dtype=torch.float32)  # Post-disaster image
    else:
        x = batch[image_key].to(device, non_blocking=True, dtype=torch.float32)  # Single input image
    
    y = batch[mask_key].to(device, non_blocking=True, dtype=torch.int64)  # Target mask

    # Disable gradients for validation
    with torch.no_grad():
        # Mixed precision support
        if is_mixed_precision:
            with autocast(device_type=device, dtype=torch.float16):
                if tta:
                    if siamese:
                        outputs = augmentation_test_time_siamese(
                            model=model, 
                            images_1=x1, 
                            images_2=x2, 
                            list_augmentations=[
                                                A.HorizontalFlip(p=1.0),  # Horizontal flip
                                                A.VerticalFlip(p=1.0)    # Vertical flip
                                            ],
                            aggregation="mean", 
                            device=device
                            )
                    else:
                        outputs = augmentation_test_time(
                            model=model, 
                            images=x, 
                            list_augmentations=[
                                                A.HorizontalFlip(p=1.0),  # Horizontal flip
                                                A.VerticalFlip(p=1.0)    # Vertical flip
                                            ],
                            aggregation="mean", 
                            device=device
                        )
                else:
                    if siamese:
                        outputs = model(x1,x2)
                    else:
                        outputs = model(x)
                vloss = loss_fn(outputs, y)
        else:
            if tta:
                if siamese:
                    outputs = augmentation_test_time_siamese(
                        model=model, 
                        images_1=x1, 
                        images_2=x2, 
                        list_augmentations=[
                                            A.HorizontalFlip(p=1.0),  # Horizontal flip
                                            A.VerticalFlip(p=1.0)    # Vertical flip
                                        ],
                        aggregation="mean", 
                        device=device
                        )
                else:
                    outputs = augmentation_test_time(
                        model=model, 
                        images=x, 
                        list_augmentations=[
                                            A.HorizontalFlip(p=1.0),  # Horizontal flip
                                            A.VerticalFlip(p=1.0)    # Vertical flip
                                        ],
                        aggregation="mean", 
                        device=device
                    )
            else:
                if siamese:
                    outputs = model(x1,x2)
                else:
                    outputs = model(x)

            vloss = loss_fn(outputs, y)

    # Extract scalar loss value
    vloss_value = vloss.item()

    # Initialize metrics dictionary
    metrics_step = {}

    # Compute predictions and metrics
    with torch.no_grad():
        # Get predictions: Output shape from (B, C, H, W) => (B, H, W)
        if len(outputs.shape) == 4: 
            preds = outputs.argmax(dim=1).long()
        
        if len(outputs.shape) == 3:
            preds = outputs.long()

        # Compute confusion matrix components
        tp, fp, fn, tn = get_stats(
            output=preds, target=y, mode=mode, num_classes=num_classes,
        )

        # Calculate each metric
        for metric in metrics:
            metric_name = metric.__name__
            metric_value = metric(tp, fp, fn, tn, class_weights=class_weights, reduction=reduction)
            metrics_step[metric_name] = metric_value

    return vloss_value, metrics_step

def training_epoch(
    model: nn.Module,
    train_dl: DataLoader,
    loss_fn: _Loss,
    metrics: List[Callable],
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    epoch_number: int = 0,
    image_key: str = "image",
    mask_key: str = "mask",
    verbose: bool = True,
    num_classes: int = 2,
    training_log_interval: int = 10,
    is_mixed_precision: bool = False,
    reduction: str = "weighted",
    class_weights: List[float] = None,
    siamese: bool = False
):
    """
    Perform one epoch of training.

    Args:
        model (nn.Module): The model to train.
        train_dl (DataLoader): The DataLoader for training data.
        loss_fn (_Loss): The loss function.
        metrics (List[Callable]): A list of metrics to calculate during training.
        optimizer (Optimizer): Optimizer for weight updates.
        scheduler (Optional[_LRScheduler]): Learning rate scheduler (optional).
        epoch_number (int): The current epoch number (0-indexed).
        image_key (str): Key to access input images in the batch.
        verbose (bool): Whether to log detailed information (default: True).
        training_log_interval (int): Interval at which to log metrics.

    Returns:
        Tuple[float, Dict[str, float]]: Epoch loss and metric averages.
    """
    logging.info(f"Epoch {epoch_number + 1}")
    logging.info("-" * 20)

    # Set model to training mode
    model.train()

    # Initialize tracking variables
    running_loss = 0.0
    total_metrics = {metric.__name__: 0.0 for metric in metrics}
    interval_samples = 0
    steps_per_epoch = math.ceil(len(train_dl.dataset) / train_dl.batch_size)
    step = 0

    # Training loop with progress bar
    with tqdm(train_dl, desc=f"Epoch {epoch_number + 1}", unit="batch", disable=False) as t:
        for batch in t:
            # Perform a single training step
            loss_t, metrics_step = training_step(
                model=model,
                batch=batch,
                loss_fn=loss_fn,
                metrics=metrics,
                optimizer=optimizer,
                step_number=step,
                image_key=image_key,
                mask_key=mask_key,
                scaler=GradScaler(),
                is_mixed_precision=is_mixed_precision,
                num_classes=num_classes,
                reduction=reduction,
                class_weights=class_weights,
                siamese=siamese
            )

            # Update loss and metrics accumulators
            batch_size = batch[image_key].size(0)
            running_loss += loss_t * batch_size
            interval_samples += batch_size

            for metric_name in total_metrics.keys():
                if metric_name in metrics_step:
                    total_metrics[metric_name] += metrics_step[metric_name] * batch_size

            # Log intermediate results at intervals
            if step % training_log_interval == 0:
                step_metrics = {
                    name: value / max(1, interval_samples) for name, value in total_metrics.items()
                }
                step_number = epoch_number * steps_per_epoch + step
                log_metrics(metrics=step_metrics, step_number=step_number, phase="Training")
                log_loss(
                    loss_value=running_loss / max(1, interval_samples),
                    step_number=step_number,
                    phase="Training",
                )

            # Update the progress bar
            t.set_postfix({"Loss": loss_t})
            step += 1

    # Calculate epoch-level loss and metrics
    epoch_loss = running_loss / len(train_dl.dataset)
    epoch_metrics = {
        name: value / len(train_dl.dataset) for name, value in total_metrics.items()
    }

    # Log epoch summary
    if verbose:
        logging.info(f"Epoch {epoch_number + 1} Training completed. Loss: {epoch_loss:.4f}")
        display_metrics(metrics=epoch_metrics, phase="Training")

    # Adjust learning rate with the scheduler (if applicable)
    if scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_loss)
        else:
            scheduler.step()

    return epoch_loss, epoch_metrics

def validation_epoch(
    model: nn.Module,
    valid_dl: DataLoader,
    loss_fn: nn.Module,
    epoch_number: int,
    metrics: List[Callable] = None,
    image_key: str = "image",
    mask_key: str = "mask",
    num_classes: int = 2, 
    verbose: bool = True,  # Adding verbose flag to control logging
    training_log_interval: int = 10,  # Define default interval for logging,
    is_mixed_precision: bool = False,
    class_weights: List[float] = None,
    reduction: str = "weighted",
    tta: bool = False,
    siamese: bool = False
) -> Tuple[float, Dict[str, float]]:
    """
    Perform one epoch of validation.

    Args:
        model (nn.Module): The model to validate.
        valid_dl (DataLoader): The DataLoader for validation data.
        loss_fn (nn.Module): The loss function.
        epoch_number (int): The current epoch number (0-indexed).
        metrics (List[Callable]): A list of metrics to calculate during validation.
        image_key (str): Key to access input images in the batch.
        verbose (bool): Whether to log detailed information (default: True).
        training_log_interval (int): Interval at which to log metrics.

    Returns:
        Tuple[float, Dict[str, float]]: Epoch validation loss and metric averages.
    """
    logging.info(f"Epoch {epoch_number + 1} - Validation Phase")
    logging.info("-" * 20)

    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    interval_samples = 0
    total_metrics = {metric.__name__: 0.0 for metric in metrics}  # Initialize totals
    steps_per_epoch = math.ceil(len(valid_dl.dataset) / valid_dl.batch_size)
    step = 0

    # Iterate through the validation dataset
    with tqdm(valid_dl, desc=f"Validation Epoch {epoch_number + 1}", unit="batch", disable=False) as t:
        for batch in t:
            batch_size = batch[image_key].size(0)

            # Perform a validation step
            vloss, metrics_step = validation_step(
                model=model,
                batch=batch,
                loss_fn=loss_fn,
                metrics=metrics,
                image_key=image_key,
                mask_key=mask_key,
                num_classes=num_classes,
                is_mixed_precision=is_mixed_precision,
                class_weights=class_weights, 
                reduction=reduction,
                tta=tta,
                siamese=siamese
            )

            # Accumulate validation loss
            running_loss += vloss * batch_size
            interval_samples += batch_size

            for metric_name in total_metrics.keys():
                if metric_name in metrics_step:
                    total_metrics[metric_name] += metrics_step[metric_name] * batch_size

            # Accumulate metrics at defined intervals
            if step % training_log_interval == 0:
                # Log intermediate metrics
                step_metrics = {
                    name: value / max(1, interval_samples) for name, value in total_metrics.items()
                }
                step_number = epoch_number * steps_per_epoch + step
                log_metrics(metrics=step_metrics, step_number=step_number, phase="Validation")

                # Log intermediate loss
                log_loss(
                    loss_value=running_loss / max(1, interval_samples),
                    step_number=step_number,
                    phase="Validation",
                )

            # Update progress bar with running loss
            t.set_postfix({"Loss": vloss})
            step += 1

    # Calculate average loss and metrics for the entire validation dataset
    epoch_vloss = running_loss / len(valid_dl.dataset)
    epoch_metrics = {
        name: total / len(valid_dl.dataset) for name, total in total_metrics.items()
    }

    # Log final metrics for the validation phase
    if verbose:
        logging.info(f"Epoch {epoch_number + 1} Validation completed. Loss: {epoch_vloss:.4f}")
        display_metrics(metrics=epoch_metrics, phase="Validation")

    return epoch_vloss, epoch_metrics

def train(
    model: nn.Module,
    train_dl: DataLoader,
    valid_dl: DataLoader,
    test_dl: DataLoader,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    params_opt: Optional[Dict] = None,
    params_sc: Optional[Dict] = None,
    loss_fn: nn.Module = nn.CrossEntropyLoss(),
    metrics: List[Callable] = [],
    nb_epochs: int = 50,
    experiment_name="experiment",
    log_dir="runs",
    model_dir="models",
    resume_path: Optional[str] = None,
    early_stopping_params: Optional[Dict[str, int]] = None,
    image_key: str = "image",
    mask_key: str = "mask",
    verbose: bool = True,  # Adding verbose flag
    checkpoint_interval: int = 10,  # Add checkpoint interval parameter
    debug: bool = False,  # Add debug flag for memory logging, 
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
    # Connect MLFlow session to the local server
    mlflow.set_tracking_uri("http://localhost:5000")
    # Set Experiment name
    mlflow.set_experiment(experiment_name)

    # Create a directory for the experiment
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{experiment_name}_{timestamp}"

    log_dir = os.path.join(log_dir, experiment_name)
    logging.info(f"Experiment logs are recorded at {log_dir}")

    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if resume_path:
        epoch = load_checkpoint(checkpoint_path=resume_path, model=model, optimizer=optimizer, scheduler=scheduler)
        start_epoch = epoch + 1 
    else:
        start_epoch = 0

    # Handling None values
    params_opt = params_opt or {"lr": 1e-4}
    params_sc = params_sc or {}
    early_stopping_params = early_stopping_params or {"patience": nb_epochs, "trigger_times": 0}

    optimizer_ft, lr_scheduler = initialize_optimizer_scheduler(model, optimizer, scheduler, optimizer_params=params_opt, scheduler_params=params_sc)

    best_val_loss = 10e6
    overall_start_time = time.time()
    patience, trigger_times = early_stopping_params["patience"], early_stopping_params["trigger_times"]
    max_images = 1

    model.to(device)

    # Prepare hyperparameters for logging
    hyperparams = {
        "epochs": nb_epochs,
        "optimizer": type(optimizer_ft).__name__,
        "scheduler": type(lr_scheduler).__name__,
        "loss_function": repr(loss_fn),
        "metrics": [metric.__name__ for metric in metrics],
        "device": device,
        "num_classes": num_classes,
        "is_mixed_precision": is_mixed_precision,
        "reduction": reduction,
        "class_names": class_names, 
        "class_weights": class_weights,
        "siamese": siamese,
        "learning_rate": params_opt.get("lr", 1e-4),
        "optimizer_params": params_opt,
        "scheduler_params": params_sc,
        "early_stopping_params": early_stopping_params,
        "checkpoint_interval": checkpoint_interval,
        "training_log_interval": training_log_interval,
        "tta": tta
    }

    # Create model signature
    signature, input_example = custom_infer_signature(model,
        data_loader=test_dl,
        siamese=siamese,
        image_key=image_key,
        mask_key=mask_key,
        device=device
    )

    mlflow.enable_system_metrics_logging()
    system_log_bool = True
    with mlflow.start_run() as run: 
        mlflow.set_tag('mlflow.runName', run_name)
        # Log hyperparameters
        mlflow.log_params(hyperparams)
        logging.info("Hyperparameters have been logged")

        # Training / Validation phases 
        """
        for epoch in range(start_epoch, nb_epochs):
            start_time = time.time()

            # Train phase
            epoch_loss, epoch_metrics = training_epoch(
                model=model, 
                train_dl=train_dl, 
                loss_fn=loss_fn, 
                metrics=metrics, 
                optimizer=optimizer_ft, 
                scheduler=lr_scheduler,
                epoch_number=epoch, 
                image_key=image_key,
                mask_key=mask_key,
                num_classes=num_classes,
                verbose=verbose, 
                training_log_interval=training_log_interval,
                is_mixed_precision=is_mixed_precision,
                reduction=reduction,
                class_weights=class_weights, 
                siamese=siamese
            )

            # Validation phase
            epoch_vloss, epoch_vmetrics = validation_epoch(
                model=model, 
                valid_dl=valid_dl, 
                loss_fn=loss_fn, 
                epoch_number=epoch, 
                metrics=metrics,
                image_key=image_key, 
                mask_key=mask_key, 
                verbose=verbose,
                training_log_interval=training_log_interval,
                is_mixed_precision=is_mixed_precision,
                reduction=reduction,
                class_weights=class_weights,
                siamese=siamese,
                num_classes=num_classes
            )

            if system_log_bool:
                system_log_bool = False
                mlflow.disable_system_metrics_logging()

            # Scheduler step after training
            lr_epoch = lr_scheduler.get_last_lr()[0]
            mlflow.log_metric("learning_rate", lr_epoch, step=epoch)
            lr_scheduler.step()

            if verbose:
                logging.info(f"Training Loss: {epoch_loss} / Validation Loss: {epoch_vloss}")

            # Log training and validation metrics
            steps_per_epoch = math.ceil(len(valid_dl.dataset) / valid_dl.batch_size)
            step_number = (epoch + 1) * steps_per_epoch
            log_metrics(metrics=epoch_metrics, step_number=step_number, phase="Training")
            log_metrics(metrics=epoch_vmetrics, step_number=step_number, phase="Validation")
            
            # Log sample images after validation epoch
            log_images_to_mlflow(
                model=model,
                data_loader=valid_dl,
                epoch=epoch,
                device=device,
                max_images=max_images,
                image_key=image_key,
                mask_key=mask_key,
                siamese=siamese, 
                color_dict={
                    0: (0, 0, 0),  # Transparent background for class 0
                    1: (0, 255, 0),  # Green with some transparency for "no-damage"
                    2: (255, 255, 0),  # Yellow with some transparency for "minor-damage"
                    3: (255, 126, 0),  # Orange with some transparency for "major-damage"
                    4: (255, 0, 0)   # Red with some transparency for "destroyed"
                },
                log_dir="../runs/mlflow_logs"
            )

            epoch_time = time.time() - start_time
            mlflow.log_metric("epoch_time", epoch_time, step=epoch)
            if verbose and debug:
                logging.info(f"Epoch {epoch + 1} took {epoch_time:.2f} seconds")
                if torch.cuda.is_available():
                    logging.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

            # Save periodic checkpoints
            if epoch % checkpoint_interval == 0:
                save_checkpoint(epoch=epoch,
                                model=model,
                                optimizer=optimizer_ft,
                                scheduler=lr_scheduler,
                                model_dir="./checkpoints",
                                )

            # Save best model and early stopping logic
            if best_val_loss > epoch_vloss:
                logging.info(f"Saving best model with val loss : {epoch_vloss}")
                trigger_times = 0
                best_val_loss = epoch_vloss

                # save and log best models
                # save_model(model, ckpt_path=model_dir, name=f"{experiment_name}_{timestamp}_best_model")
                # log_model(model, artifact_path="best_model", signature=signature,input_example=input_example)
          
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    logging.info("Early stopping")
                    break

        total_time = time.time() - overall_start_time
        logging.info(f"Total training time: {total_time:.2f} seconds")
        """
        ### Testing phase 
        compute_model_class_performance(
                model=model,
                dataloader=test_dl,
                num_classes=num_classes,
                device=device,
                class_names=class_names, 
                siamese=siamese,
                image_key=image_key,
                mask_key=mask_key,
                average_mode="macro",
                tta=False,
                mlflow_bool=True
            )

def testing(
    model: nn.Module,
    test_dataloader: DataLoader,
    loss_fn: nn.Module,
    metrics: List[Callable] = [],
    image_key: str = "image",
    mask_key: str = "mask", 
    verbose: bool = True,  # Adding verbose flag to control logging
    is_mixed_precision: bool = False,
    num_classes: int = 2,
    reduction: str = "weighted",
    class_weights: List[float] = None, 
    tta: bool = True,
    siamese: bool = False
) -> Tuple[float, Dict[str, float]]:
    """
    Perform one epoch of validation.

    Args:
        model (nn.Module): The model to validate.
        test_dataloader (DataLoader): The DataLoader for Testing data.
        loss_fn (nn.Module): The loss function.
        epoch_number (int): The current epoch number (0-indexed).
        metrics (List[Callable]): A list of metrics to calculate during validation.
        image_key (str): Key to access input images in the batch.
        verbose (bool): Whether to log detailed information (default: True).
        training_log_interval (int): Interval at which to log metrics.
        siamese (bool) : Siamese Network 

    Returns:
        Tuple[float, Dict[str, float]]: Epoch validation loss and metric averages.
    """
    logging.info("Testing Phase")

    # Set model to evaluation mode
    if isinstance(model, nn.Module):
        model.eval()

    running_loss = 0.0
    interval_samples = 0
    total_metrics = {metric.__name__: 0.0 for metric in metrics}  # Initialize totals

    # Iterate through the validation dataset
    with tqdm(test_dataloader, desc=f"Testing", unit="batch") as t:
        for batch in t:
            if siamese:
                batch_size = batch[image_key].size(0)
            else:
                batch_size = batch["pre_image"].size(0)
            
            # Perform a validation step
            tloss, metrics_step = validation_step(
                model=model,
                batch=batch,
                loss_fn=loss_fn,
                metrics=metrics,
                image_key=image_key,
                mask_key=mask_key,
                num_classes=num_classes,
                is_mixed_precision=is_mixed_precision,
                reduction=reduction,
                class_weights=class_weights,
                tta=tta,
                siamese=siamese
            )

            # Accumulate validation loss
            running_loss += tloss * batch_size
            interval_samples += batch_size

            for metric_name in total_metrics.keys():
                if metric_name in metrics_step:
                    total_metrics[metric_name] += metrics_step[metric_name] * batch_size

            # Update progress bar with running loss
            t.set_postfix({"Loss": tloss})

    # Calculate average loss and metrics for the entire validation dataset
    epoch_tloss = running_loss / len(test_dataloader.dataset)
    test_metrics = {
        "test_" + name: total / len(test_dataloader.dataset) for name, total in total_metrics.items()
    }
    return epoch_tloss, test_metrics