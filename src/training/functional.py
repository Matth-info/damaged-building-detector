

#PyTorch imports
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
from .utils import log_metrics, log_images_to_tensorboard
import time


# Loss import 


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Training will be done on ",device)
scaler = GradScaler()

def training_epoch(model: nn.Module, train_dl: DataLoader, loss_fn: _Loss, optimizer : optim , scheduler : optim.lr_scheduler,epoch_number:int, writer:SummaryWriter):
    print(f'Epoch {epoch_number + 1}')
    print('-' * 10)

    model.train()
    running_loss = 0.0

    for step, batch in enumerate(train_dl):
        x = batch["image"]
        y = batch["mask"]

        x = x.to(device, non_blocking=True)
        #y = y.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast(device_type=device):

            outputs = model(x)
            # Compute the loss and its gradients 
            loss = loss_fn(outputs, y)

        # adjust the learning weights 
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Accumulate loss
        running_loss += loss.item()

    # Calculate and return epoch loss
    epoch_loss = running_loss / len(train_dl) # len(dataloader) = number of batch 
    print(f'Epoch {epoch_number + 1} completed. Epoch loss: {epoch_loss:.4f}')
    return epoch_loss

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
    early_stopping_params: Optional[Dict[str, int]] = None
):
    
    #Create a directory for the experiment
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
    print(f'Experiment logs are recoded at {log_dir}')
    
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # handling None values
    if params_opt is None:
        params_opt = {'lr': 1e-4}
    if params_sc is None:
        params_sc = {}
    if early_stopping_params is None:
        early_stopping_params = {"patience": 5, "trigger_times": 0}

    if optimizer is None:
        optimizer_ft = torch.optim.AdamW(model.parameters(), lr=1e-4)
    else:
        optimizer_ft = optimizer(params=filter(lambda p: p.requires_grad, model.parameters()), **params_opt)

    if scheduler is None:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
    else:
        lr_scheduler = scheduler(optimizer=optimizer_ft,**params_sc)

    # initialize some variables
    best_vloss = float('inf')
    overall_start_time  = time.time()
    patience, trigger_times = early_stopping_params["patience"], early_stopping_params["trigger_times"]
    max_images = 4
    log_interval = 2
    
    # Initialize TensorBoard writer
    with SummaryWriter(log_dir) as writer:
        for epoch in range(nb_epochs):
            start_time = time.time()
            epoch_loss = training_epoch(model, train_dl, loss_fn=loss_fn, optimizer=optimizer_ft, scheduler=lr_scheduler, epoch_number=epoch, writer=writer)
            
            epoch_vloss, epoch_metrics  = validation_epoch(model=model,
                                                         valid_dl=valid_dl,
                                                         loss_fn=loss_fn,
                                                         metrics=metrics, 
                                                         epoch_number=epoch)
            
            lr_scheduler.step()
            print(f"LOSS train {epoch_loss} valid {epoch_vloss}")

            # log training and validation losses 
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : epoch_loss, 'Validation' : epoch_vloss }, epoch + 1)

            # log validation metrics
            log_metrics(writer, metrics=epoch_metrics, epoch_number=epoch, phase="validation")

            # log some sample images
            # After validation epoch
            if epoch % log_interval == 0:
                log_images_to_tensorboard(
                    model=model,
                    data_loader=valid_dl,
                    writer=writer,
                    epoch=epoch,
                    device=device,
                    max_images=max_images
                )
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1} took {epoch_time:.2f} seconds")

            if torch.cuda.is_available():
                print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            # Save best model and early stopping 
            if epoch_vloss < best_vloss:
                print("Saving best model")
                best_vloss = epoch_vloss
                trigger_times = 0
                torch.save(model.state_dict(), os.path.join(model_dir, f"{experiment_name}_{timestamp}_best_model.pth"))
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print("Early stopping")
                    break
        
        total_time = time.time() - overall_start_time
        print(f"Total training time: {total_time:.2f} seconds")

def validation_epoch(
    model: nn.Module,
    valid_dl: DataLoader,
    loss_fn: nn.Module,
    epoch_number: int,
    metrics: List[Callable] = [],
) -> Tuple[int, Dict[str, float]]:
    
    running_loss = 0.0
    metric_totals = {metric.__class__.__name__: 0.0 for metric in metrics}  # Initialize totals for each metric

    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        for step, batch in enumerate(valid_dl):
            x = batch["image"]
            y = batch["mask"]

            x = x.to(device, non_blocking=True)
            #y = y.to(device , non_blocking=True)

            # Forward pass
            outputs = model(x)
            
            # Calculate loss
            vloss = loss_fn(outputs, y)
            running_loss += vloss.item() * valid_dl.batch_size
            
            # Calculate each metric and accumulate the total
            for metric in metrics:
                metric_value = metric(outputs, y)
                metric_totals[metric.__class__.__name__] += metric_value * valid_dl.batch_size

    # Calculate average loss and metrics over the whole validation dataset
    epoch_vloss = running_loss / len(valid_dl.dataset)
    epoch_metrics = {name: float((total / len(valid_dl.dataset)).cpu().numpy()) for name, total in metric_totals.items()}

    print(f'Epoch {epoch_number + 1} validation completed. Loss: {epoch_vloss:.4f}, Metrics: {epoch_metrics}')
    return epoch_vloss, epoch_metrics