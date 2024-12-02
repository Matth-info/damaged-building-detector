import argparse
import os

import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss

# Custom libraries
from datasets import prepare_cloud_segmentation_data, Cloud_DrivenData_Dataset
from training.functional_auto_encoder import find_threshold
from training.augmentations import get_val_autoencoder_augmentation_pipeline
from models import AutoEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Test an Auto-Encoder model for Cloud Coverage Detection")
    
    # Experiment settings
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.pth file).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for the test dataset.")
    parser.add_argument("--origin_dir", type=str, default="../data/data_samples/Cloud_DrivenData/final/public", help="Path to the Cloud dataset.")
    return parser.parse_args()


def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Prepare the test dataset
    _, _, val_x, val_y = prepare_cloud_segmentation_data(folder_path=args.origin_dir, train_share=0.8, seed=42)

    data_test = Cloud_DrivenData_Dataset(
        x_paths=val_x,
        bands=["B04", "B03", "B02"],
        y_paths=val_y,
        transform=get_val_autoencoder_augmentation_pipeline(),
    )

    print("Evaluation Data Length:", len(data_test))

    # Model Initialization
    model = AutoEncoder(
        num_input_channel=3,
        base_channel_size=64,
    )

    # Model Loading 
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    model = model.load(args.model_path)
    model.to(device)
    print(f"Auto Encoder has been loaded from {args.model_path}")

    # Compute the threshold
    print("Finding Reconstruction Loss Threshold on Test Set")
    criterion = MSELoss()
    threshold, _, _ = find_threshold(
        model=model,
        data=data_test,
        loss_fn=criterion,
        device=device,
        confidence_interval=0.95
    )

    print(f"Reconstruction Loss Threshold: {threshold}")


if __name__ == "__main__":
    main()
