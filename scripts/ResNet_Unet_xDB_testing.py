import argparse
import torch
from torch.utils.data import DataLoader

# Custom libraries
from datasets import xDB_Damaged_Building
from training import testing
from training.augmentations import get_val_augmentation_pipeline
from models import ResNet_UNET
from losses import DiceLoss, FocalLoss, Ensemble
from metrics import f1_score, iou_score, balanced_accuracy


def parse_args():
    """Parse command-line arguments for model testing."""
    parser = argparse.ArgumentParser(description="Test a ResNet-based U-Net model for building segmentation.")
    
    # Experiment settings
    parser.add_argument("--experiment_name", type=str, default="Experiment", help="Name of the experiment for logging.")
    parser.add_argument("--backbone", type=str, default="resnet18", help="ResNet backbone to use (e.g., resnet18, resnet34, resnet50).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for testing.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model file.")
    parser.add_argument("--tta", action="store_true", help="Enable Test Time Augmentation (TTA).")
    parser.add_argument("--mixed_precision", action="store_true", help="Enable mixed-precision testing.")
    parser.add_argument("--origin_dir", type=str, default="../data/xDB/tier3", help="Root directory of the dataset.")
    
    return parser.parse_args()


def main():
    args = parse_args()

    # Dataset Preparation
    data_test = xDB_Damaged_Building(
        origin_dir=args.origin_dir,
        mode="building",
        time="pre",
        transform=get_val_augmentation_pipeline(image_size=(512, 512), max_pixel_value=1),
        type="test",
        val_ratio=0.1,
        test_ratio=0.1,
    )

    # Model Initialization
    model = ResNet_UNET(
        in_channels=3,
        out_channels=2,
        backbone_name=args.backbone,
        pretrained=False
    )

    model.load_state_dict(torch.load(args.model_path))
    model.to("cuda")
    print(f"Unet with {args.backbone} from {args.model_path} has been loaded for Testing")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
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
        image_key="image",
        mask_key="mask",
        verbose=True,
        is_mixed_precision=args.mixed_precision,
        num_classes=2,
        reduction="weighted",
        class_weights=[0.1, 0.9],
        tta=args.tta,
    )

    # Print Results
    print("Test Time Augmentation:", args.tta)
    print("Testing Loss:", epoch_tloss)
    for name, value in test_metrics.items():
        print(f"{name}: {value.item()}")


if __name__ == "__main__":
    main()
