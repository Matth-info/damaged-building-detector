import argparse
import os

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader

# Custom librairis
from datasets import Cloud_DrivenData_Dataset, prepare_cloud_segmentation_data
from models import AutoEncoder
from training.augmentations import (
    get_train_autoencoder_augmentation_pipeline,
    get_val_autoencoder_augmentation_pipeline,
)
from training.functional_auto_encoder import find_threshold, train


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an Auto-Encoder model for Cloud Coverage Detection"
    )

    # Experiment settings
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="Experiment",
        help="Name of the experiment for logging.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training and validation."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=30, help="Number of epochs for training."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--log_dir", type=str, default="./runs", help="Directory for storing logs."
    )
    parser.add_argument(
        "--model_dir", type=str, default="./models", help="Directory for saving trained models."
    )
    parser.add_argument(
        "--mixed_precision", action="store_true", help="Enable mixed-precision training."
    )
    parser.add_argument(
        "--origin_dir",
        type=str,
        default="./data/data_samples/Cloud_DrivenData/final/public",
        help="Local path to Cloud Dataset.",
    )
    return parser.parse_args()


def rename_model_file_with_threshold(model_path, threshold):
    """
    Rename a .pth file by appending the threshold value to the file name.

    Args:
        model_path (str): Path to the original .pth file.
        threshold (float): The reconstruction loss threshold to append.

    Returns:
        str: The new file path after renaming.
    """
    # Extract the directory and filename
    dir_name, file_name = os.path.split(model_path)
    file_base, file_ext = os.path.splitext(file_name)

    # Ensure it is a .pth file
    if file_ext != ".pth":
        raise ValueError(f"File {model_path} is not a .pth file.")

    # Construct the new filename
    new_file_name = f"{file_base}_threshold-{threshold:.6f}{file_ext}"
    new_file_path = os.path.join(dir_name, new_file_name)

    # Rename the file
    os.rename(model_path, new_file_path)
    print(f"Renamed file to: {new_file_path}")

    return new_file_path


def main():
    args = parse_args()

    origin_dir = args.origin_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare and split dataset
    train_x, train_y, val_x, val_y = prepare_cloud_segmentation_data(
        folder_path=origin_dir, train_share=0.8, seed=42
    )
    # Create Cloud Training and Validation Dataset
    data_train = Cloud_DrivenData_Dataset(
        x_paths=train_x,
        bands=["B04", "B03", "B02"],
        y_paths=train_y,
        transform=get_train_autoencoder_augmentation_pipeline(),
    )
    val_size = len(val_x) // 2
    data_val = Cloud_DrivenData_Dataset(
        x_paths=val_x[:val_size],
        bands=["B04", "B03", "B02"],
        y_paths=val_y[:val_size],
        transform=get_val_autoencoder_augmentation_pipeline(),
    )

    data_test = Cloud_DrivenData_Dataset(
        x_paths=val_x[val_size:],
        bands=["B04", "B03", "B02"],
        y_paths=val_y[val_size:],
        transform=get_val_autoencoder_augmentation_pipeline(),
    )

    print("Training Data Length : ", len(data_train))
    print("Validation Data Length : ", len(data_val))
    print("Evaluation Data Length : ", len(data_test))

    # Model Initialization
    model = AutoEncoder(
        num_input_channel=3,
        base_channel_size=64,
    )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    model.to(device)

    # Training
    torch.cuda.empty_cache()

    criterion = MSELoss()

    best_model_path = train(
        model,
        train_dataset=data_train,
        val_dataset=data_train,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        criterion=criterion,
        experiment_name=args.experiment_name,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        training_log_interval=5,
        device=device,
        use_amp=args.mixed_precision,
        save_best_model=True,
    )

    # Find a Threshold
    print("Find a Reconstruction loss threshold on Test Set")
    threshold, _, _ = find_threshold(
        model, data=data_train, loss_fn=criterion, device=device, confidence_interval=0.95
    )

    print(f"Test Reconstruction Loss threshold is : {threshold}")
    new_model_path = rename_model_file_with_threshold(best_model_path, threshold)
    print(f"New model path: {new_model_path}")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
