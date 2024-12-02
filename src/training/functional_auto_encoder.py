import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import MSELoss
from torch.amp import autocast, GradScaler
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from datetime import datetime
import os 
import numpy as np

__all__ = ['train','find_threshold']

def train(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset = None,
    num_epochs: int = 20,
    batch_size: int = 64,
    criterion: nn.Module = MSELoss(),
    learning_rate: float = 1e-3,
    log_dir: str = "../logs",
    model_dir: str = "../models",
    experiment_name: str = "experiment",
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    training_log_interval: int = 5,
    use_amp: bool = True,
    save_best_model: bool = True,
):
    """
    Train an autoencoder with TensorBoard logging and GPU optimizations.

    Args:
        model (nn.Module): The autoencoder model to train.
        train_dataset (Dataset): Dataset for training.
        val_dataset (Dataset, optional): Dataset for validation. Defaults to None.
        num_epochs (int, optional): Number of epochs. Defaults to 20.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 64.
        learning_rate (float, optional): Learning rate for optimizer. Defaults to 1e-3.
        criterion (nn.Module, optional): Loss function 
        log_dir (str, optional): Directory to save TensorBoard logs. Defaults to '../logs'.
        model_dir (str, optional): Directory to save models. Defaults to '../models'.
        experiment_name (str, optional): Name of the experiment for logging. Defaults to "experiment".
        device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda' if available.
        training_log_interval (int): Interval for logging training metrics.
        use_amp (bool): Enable mixed precision training to optimize performance on GPUs.
        save_best_model (bool): Save the best model based on validation loss.

    Returns:
        None
    """
    # Initialize paths and logging
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
    print(f"Experiment logs will be recorded at {log_dir}")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Set up device, model, optimizer, and loss
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler(enabled=use_amp)  # Mixed precision scaler
    writer = SummaryWriter(log_dir=log_dir)

    # Log hyperparameters
    writer.add_text("Hyperparameters", str({
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "use_amp": use_amp,
        "device": device
    }))

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8) if val_dataset else None

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as t:
            for step, batch in enumerate(t, start=1):
                inputs = batch["image"].to(device, non_blocking=True)
                optimizer.zero_grad()

                # Forward and backward passes
                with autocast(device_type="cuda", dtype=torch.float16 if use_amp else torch.float32):
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()
                if step % training_log_interval == 0:
                    writer.add_scalar("Loss/Train_Batch", loss.item(), epoch * len(train_loader) + step)
                t.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        writer.add_scalar("Loss/Train_Epoch", train_loss, epoch)

        # Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch["image"].to(device, non_blocking=True)
                    with autocast(device_type="cuda", dtype=torch.float16 if use_amp else torch.float32):
                        outputs = model(inputs)
                        loss = criterion(outputs, inputs)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f} / Validation Loss: {val_loss:.4f}")

            # Save the best model
            if save_best_model and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(model_dir, f"{experiment_name}_best.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved at {best_model_path}")

        # Log images every 3 epochs or at the last epoch
        if epoch % 3 == 0 or epoch == num_epochs - 1:
            writer.add_images("Reconstruction/Inputs", inputs[:8], epoch)
            writer.add_images("Reconstruction/Outputs", outputs[:8], epoch)

    # Final logging
    writer.close()
    print(f"Training complete. Logs saved to: {log_dir} and best model saved at : {best_model_path}")

    return best_model_path


def find_threshold(model, data, loss_fn, device, confidence_interval=0.95):
    """
    Find a reconstruction loss threshold based on the training data.

    Args:
        model (nn.Module): The autoencoder model.
        data (Dataset): The dataset to compute reconstruction loss.
        loss_fn (function): The loss function to compute reconstruction error.
        device (str): Device to run the model on ('cpu' or 'cuda').
        confidence_interval (float, optional): Confidence interval for threshold. Defaults to 0.95.

    Returns:
        threshold (float): The threshold value for reconstruction loss.
        mean_loss (float): The mean reconstruction loss.
        std_loss (float): The standard deviation of the reconstruction loss.
    """
    data_loader = DataLoader(data, batch_size=5, shuffle=True, num_workers=8)

    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Move model to the desired device

    all_losses = []  # To store all the individual losses

    with torch.no_grad():
        with tqdm(data_loader, desc=f"Threshold Computation", unit="batch") as t:
            for batch in t:
                inputs = batch["image"].to(device)
                outputs = model(inputs)
                
                # Compute the reconstruction loss (MSE)
                loss = loss_fn(outputs, inputs)
                all_losses.append(loss.item())

                t.set_postfix(loss=loss.item())  # Display batch loss

    # Convert list of losses to a NumPy array for easy manipulation
    all_losses = np.array(all_losses)
    
    # Calculate mean and standard deviation of the reconstruction loss
    mean_loss = np.mean(all_losses)
    std_loss = np.std(all_losses)
    
    # Calculate the threshold based on the confidence interval
    # For a 95% confidence interval, use 1.96 as k (the z-score for 95% confidence)
    z_score = 1.96 if confidence_interval == 0.95 else 1.64  # 1.64 for 90% confidence
    threshold = mean_loss + z_score * std_loss

    return threshold, mean_loss, std_loss
