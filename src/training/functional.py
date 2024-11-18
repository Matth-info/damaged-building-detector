# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torch.amp import autocast, GradScaler
from torch.nn.modules.loss import _Loss

# utils import
from datetime import datetime
import os
from typing import List, Callable, Dict, Optional, Tuple
from .utils import (
    log_metrics,
    log_images_to_tensorboard,
    save_model,
    load_model,
    display_metrics,
)
import time
import numpy as np

# from custom metrics and losses
from metrics import get_stats

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Training will be done on ", device)
verbose = True
mode = "multiclass"
reduction = "micro-imagewise"
print(f"Metric computations is based on {mode} mode")
is_mixed_precision = False
debug = False
print(f"Mixed Precision Training : {is_mixed_precision}")
training_log_interval = 10  # number of step between 2 training logs (avoid too much data transfer and metric computations)
num_classes = 2
scaler = GradScaler(device=device)


def training_step(
    model,
    batch,
    loss_fn: _Loss,
    optimizer: optim,
    metrics: List[Callable],
    step_number: int,
):
    x = batch["image"].to(torch.float16 if is_mixed_precision else torch.float32)
    y = batch["mask"].to(torch.float16 if is_mixed_precision else torch.float32)

    loss = 0.0

    if device == "cuda":
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

    optimizer.zero_grad()

    if is_mixed_precision:
        with autocast(device_type=device, dtype=torch.float16):
            outputs = model(x)
            # Compute the loss and its gradients
            loss = loss_fn(outputs, y)
            # adjust the learning weights
            scaler.scale(
                loss
            ).backward()  # compute gradient in float16 and multiple gradient by a scale factor
            # avoiding underflow = minor change because of weight format
            scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
    else:
        outputs = model(x)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

    loss = loss.item()
    metrics_step = {}
    if step_number % training_log_interval == 0:
        # Calculate each metric and accumulate the total
        # Output format from (B,N,H,W) => (B,H,W)
        out = outputs.long().argmax(dim=1)
        # Compute the Confusion matrix
        tp, fp, fn, tn = get_stats(
            output=out, target=y.long(), mode=mode, num_classes=num_classes
        )
        for metric in metrics:
            metric_name = metric.__name__
            metric_value = metric(
                tp, fp, fn, tn, class_weights=None, reduction=reduction
            )  # (N,C)
            metrics_step[metric_name] = metric_value

    return loss, metrics_step


def validation_step(model, batch, loss_fn, metrics):

    x = batch["image"].to(torch.float16 if is_mixed_precision else torch.float32)
    y = batch["mask"].to(torch.float16 if is_mixed_precision else torch.float32)

    batch_size = x.shape[0]

    if device == "cuda":
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

    with torch.no_grad():
        if is_mixed_precision:
            with autocast(device_type=device, dtype=torch.float16):
                # Forward pass
                outputs = model(x)
                vloss = loss_fn(outputs, y)
        else:
            outputs = model(x)
            vloss = loss_fn(outputs, y)

    vloss = vloss.item()
    # Calculate each metric and accumulate the total
    metrics_step = {}

    # model="multiclass"
    # Output format from (B,N,H,W) => (B,H,W)
    out = outputs.long().argmax(dim=1)
    # Compute the Confusion matrix
    tp, fp, fn, tn = get_stats(
        output=out, target=y.long(), mode=mode, num_classes=num_classes
    )
    for metric in metrics:
        metric_name = metric.__name__
        metric_value = metric(tp, fp, fn, tn, class_weights=None, reduction=reduction)
        metrics_step[metric_name] = metric_value

    return vloss, metrics_step


def training_epoch(
    model: nn.Module,
    train_dl: DataLoader,
    loss_fn: _Loss,
    metrics: List[Callable],
    optimizer: optim,
    scheduler: optim.lr_scheduler,
    epoch_number: int,
    writer: SummaryWriter,
):
    print(f"Epoch {epoch_number + 1}")
    print("-" * 10)

    model.train()
    running_loss = 0.0
    total_metrics = {metric.__name__: 0.0 for metric in metrics}
    interval_samples = 0

    for step, batch in enumerate(train_dl):
        loss_t, metrics_step = training_step(
            model=model,
            batch=batch,
            loss_fn=loss_fn,
            metrics=metrics,
            optimizer=optimizer,
            step_number=step,
        )
        # Accumulate loss
        batch_size = batch["image"].size(0)
        running_loss += loss_t * batch_size
        # accumulate training metrics
        if step % training_log_interval == 0:
            interval_samples += batch_size
            for metric_name in total_metrics.keys():
                total_metrics[metric_name] += metrics_step[metric_name] * batch_size

    # Calculate and return epoch loss
    epoch_loss = running_loss / len(train_dl.dataset)

    # Calculate and return epoch metric training average
    total_metrics = {
        name: value / interval_samples for name, value in total_metrics.items()
    }
    if verbose:
        print(f"Epoch {epoch_number + 1} Training completed. Loss: {epoch_loss:.4f}")
        display_metrics(metrics=total_metrics, phase="Training")
    return epoch_loss, total_metrics


def validation_epoch(
    model: nn.Module,
    valid_dl: DataLoader,
    loss_fn: nn.Module,
    epoch_number: int,
    metrics: List[Callable] = [],
) -> Tuple[int, Dict[str, float]]:

    running_loss = 0.0

    metric_totals = {
        metric.__name__: 0 for metric in metrics
    }  # Initialize totals for each metric

    model.eval()  # Set model to evaluation mode

    for step, batch in enumerate(valid_dl):
        batch_size = batch["image"].size(0)
        vloss, metrics_step = validation_step(
            model, batch, loss_fn=loss_fn, metrics=metrics
        )
        # Accumulate validation loss
        running_loss += vloss * batch_size
        # Accumulate metrics
        for metric in metrics:
            metric_totals[metric.__name__] += metrics_step[metric.__name__] * batch_size

    # Calculate average loss and metrics over the whole validation dataset
    epoch_vloss = running_loss / len(valid_dl.dataset)
    epoch_metrics = {
        name: total / len(valid_dl.dataset) for (name, total) in metric_totals.items()
    }

    if verbose:
        print(f"Epoch {epoch_number + 1} Validation completed. Loss: {epoch_vloss:.4f}")
        display_metrics(metrics=epoch_metrics, phase="Validation")

    return epoch_vloss, epoch_metrics


def train(
    model: nn.Module,
    train_dl: DataLoader,
    valid_dl: DataLoader,
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
    early_stopping_params: Optional[Dict[str, int]] = None,
):

    # Create a directory for the experiment
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
    print(f"Experiment logs are recoded at {log_dir}")

    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # handling None values
    if params_opt is None:
        params_opt = {"lr": 1e-4}
    if params_sc is None:
        params_sc = {}
    if early_stopping_params is None:
        early_stopping_params = {"patience": nb_epochs, "trigger_times": 0}

    if optimizer is None:
        optimizer_ft = torch.optim.AdamW(
            params=filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
        )
    else:
        optimizer_ft = optimizer

    if scheduler is None:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer_ft, step_size=10, gamma=0.1
        )
    else:
        lr_scheduler = scheduler(optimizer=optimizer_ft, **params_sc)

    # initialize some variables
    best_avg_metric = 0
    overall_start_time = time.time()
    patience, trigger_times = (
        early_stopping_params["patience"],
        early_stopping_params["trigger_times"],
    )
    max_images = 4
    log_interval = 2

    model.to(device)

    # Initialize TensorBoard writer
    with SummaryWriter(log_dir) as writer:
        for epoch in range(nb_epochs):
            start_time = time.time()
            epoch_loss, epoch_metrics = training_epoch(
                model,
                train_dl,
                loss_fn=loss_fn,
                metrics=metrics,
                optimizer=optimizer_ft,
                scheduler=lr_scheduler,
                epoch_number=epoch,
                writer=writer,
            )

            epoch_vloss, epoch_vmetrics = validation_epoch(
                model=model,
                valid_dl=valid_dl,
                loss_fn=loss_fn,
                metrics=metrics,
                epoch_number=epoch,
            )

            lr_scheduler.step()
            writer.add_scalar("learning rate", lr_scheduler.get_last_lr()[0], epoch + 1)

            if verbose:
                print(f"LOSS train {epoch_loss} valid {epoch_vloss}")

            # log training and validation losses
            writer.add_scalars(
                "Training vs. Validation Loss",
                {"Training": epoch_loss, "Validation": epoch_vloss},
                epoch + 1,
            )

            # log validation and training metrics
            log_metrics(
                writer, metrics=epoch_metrics, epoch_number=epoch, phase="training"
            )
            log_metrics(
                writer, metrics=epoch_vmetrics, epoch_number=epoch, phase="validation"
            )

            # log some sample images
            # After validation epoch
            if epoch % log_interval == 0:
                log_images_to_tensorboard(
                    model=model,
                    data_loader=valid_dl,
                    writer=writer,
                    epoch=epoch,
                    device=device,
                    max_images=max_images,
                )

            epoch_time = time.time() - start_time

            if verbose and debug:
                print(f"Epoch {epoch+1} took {epoch_time:.2f} seconds")
                print(
                    f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
                )

            # Save best model and early stopping
            avg_metric = np.mean([value for _, value in epoch_vmetrics.items()])
            if avg_metric > best_avg_metric:
                if verbose:
                    print("Saving best model")
                trigger_times = 0
                save_model(
                    model,
                    ckpt_path=model_dir,
                    name=f"{experiment_name}_{timestamp}_best_model",
                )
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print("Early stopping")
                    break

        total_time = time.time() - overall_start_time
        print(f"Total training time: {total_time:.2f} seconds")
