import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import numpy as np

# Custom libraries
from datasets import xDB_Damaged_Building, Puerto_Rico_Building_Dataset
from training import testing
from training.augmentations import get_val_augmentation_pipeline
from models import ResNet_UNET
from losses import DiceLoss, FocalLoss, Ensemble
from metrics import f1_score, iou_score, balanced_accuracy
from models import AutoEncoder
from utils import display_semantic_predictions_batch

import logging

def parse_args():
    """Parse command-line arguments for model testing."""
    parser = argparse.ArgumentParser(description="Test a ResNet-based U-Net model for building segmentation.")
    # Experiment settings
    parser.add_argument("--origin_dir", type=str, default="../data/xDB/tier3", help="Root directory of the dataset.")
    parser.add_argument("--dataset_name", type=str, default="xDB", help="Evaluate the model on the given dataset (choose between 'xDB' and 'puerto_rico')")
    parser.add_argument("--backbone", type=str, default="resnet18", help="ResNet backbone to use (e.g., resnet18, resnet34, resnet50).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for testing.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model file.")
    parser.add_argument("--tta", action="store_true", help="Enable Test Time Augmentation (TTA).")
    parser.add_argument("--mixed_precision", action="store_true", help="Enable mixed-precision testing.")
    parser.add_argument("--display_samples",type=int,default=0,help="Number of test samples to display predictions for. Set to 0 to disable."
) 
    return parser.parse_args()


def choose_test_dataset(dataset_name, origin_dir):
    if dataset_name == "xDB":
        # xView2 Dataset 
        return xDB_Damaged_Building(
            origin_dir=origin_dir,
            mode="building",
            time="pre",
            transform=get_val_augmentation_pipeline(image_size=(512, 512), max_pixel_value=1),
            type="test",
            val_ratio=0.1,
            test_ratio=0.1,
        ), 'image'
    elif dataset_name == "puerto_rico":
        # Puerto Rico EY Challenge Dataset
        # Remove the images covered with clouds. 

        cloud_filter_params = {
            "model_class": AutoEncoder(num_input_channel=3, base_channel_size=64), 
            "device": "cuda", 
            "file_path": "../models/AutoEncoder_Cloud_Detector_0.001297.pth",
            "threshold": 0.001297, 
            "loss": MSELoss,
            "batch_size": 32
            }

        return Puerto_Rico_Building_Dataset(
            base_dir=origin_dir,
            pre_disaster_dir="Pre_Event_Grids_In_TIFF",
            post_disaster_dir="Post_Event_Grids_In_TIFF",
            mask_dir="Pre_Event_Grids_In_TIFF_mask",
            transform=None,
            extension="tif",
            cloud_filter_params=cloud_filter_params,
            preprocessing_mode="offline",
            filtered_list_path=None
            ), 'pre_image'
    else:
        raise "Dataset Unavailable"


def main():
    args = parse_args()

    # Define and configure a logger
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    if not logger.hasHandlers():  # Prevent duplicate handlers
        logger.addHandler(console_handler)

    # Dataset selection
    data_test, image_tag = choose_test_dataset(dataset_name=args.dataset_name, origin_dir=args.origin_dir)

    # Model Initialization
    model = ResNet_UNET(
        in_channels=3,
        out_channels=2,
        backbone_name=args.backbone,
        pretrained=False
    )

    model.load_state_dict(torch.load(args.model_path))
    model.to("cuda")
    logger.info(f"Unet with {args.backbone} from {args.model_path} has been loaded for Testing on {args.dataset_name} Dataset")

    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    # Loss and Metrics
    criterion = Ensemble(
        list_losses=[DiceLoss(mode="multiclass"), FocalLoss(mode="multiclass")],
        weights=[1.0, 10.0]
    ).cuda()
    
    metrics = [balanced_accuracy, f1_score, iou_score]

    # DataLoader for Testing
    test_dl = DataLoader(data_test, batch_size=args.batch_size, shuffle=False)

    # Model Testing
    torch.cuda.empty_cache()
    epoch_tloss, test_metrics = testing(
        model=model,
        test_dataloader=test_dl,
        loss_fn=criterion,
        metrics=metrics,
        image_key=image_tag,
        mask_key="mask",
        verbose=True,
        is_mixed_precision=args.mixed_precision,
        num_classes=2,
        reduction="weighted",
        class_weights=[0.1, 0.9],
        tta=args.tta,
    )

    # Print Results
    logger.info(f"Test Time Augmentation: {args.tta}")
    logger.info(f"Testing Loss: {epoch_tloss}")
    for name, value in test_metrics.items():
        logger.info(f"{name}: {value.item()}")

    if args.display_samples > 0 :
        test_dl = DataLoader(data_test, batch_size=args.display_samples, shuffle=True)
        inputs = next(iter(test_dl))
        with torch.no_grad():
            images , masks = inputs[image_tag].to("cuda"), inputs["mask"].to("cuda")
            preds = model.predict(images)
            display_semantic_predictions_batch(
                                       images=images, 
                                       mask_predictions=preds,
                                       mask_labels=masks, 
                                       normalized = {
                                           "mean": np.array([0.349, 0.354, 0.268]), 
                                           "std": np.array([0.114, 0.102, 0.094])
                                            }
                                        folder_path="../outputs/plots"
                                       )

if __name__ == "__main__":
    main()
