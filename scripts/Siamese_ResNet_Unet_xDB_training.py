import argparse
import json 
import logging
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


# Custom librairis
from datasets import xDB_Siamese_Dataset
from training import train, testing
from training.augmentations import (
    augmentation_test_time,
    get_train_augmentation_pipeline,
    get_val_augmentation_pipeline,
)
from models import SiameseResNetUNet
from losses import DiceLoss, FocalLoss, Ensemble
from metrics import f1_score, iou_score, balanced_accuracy, precision, recall


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet-based U-Net model for building segmentation.")
    
    # Experiment settings
    parser.add_argument("--experiment_name", type=str, default="Experiment", help="Name of the experiment for logging.")
    parser.add_argument("--backbone", type=str, default="resnet18", help="ResNet backbone to use (e.g., resnet18, resnet34, resnet50).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and validation.")
    parser.add_argument("--in_channels", type=int, default=3, help="Number of input channels")
    parser.add_argument("--out_channels", type=int, default=2, help="Number of output channels")
    parser.add_argument("--mode", type=str, default="full_damage", help="building, full_damage, simple_damage, change_detection")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--pretrained", action="store_true", help="Use a pretrained backbone.")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze the ResNet backbone during training.")
    parser.add_argument("--log_dir", type=str, default="../runs", help="Directory for storing logs.")
    parser.add_argument("--model_dir", type=str, default="../models", help="Directory for saving trained models.")
    parser.add_argument("--tta", action="store_true", help="Enable Test Time Augmentation (TTA).")
    parser.add_argument("--mixed_precision", action="store_true", help="Enable mixed-precision training.")
    parser.add_argument("--origin_dir", type=str, default="../data/xDB/tier3", help="Local path to xDB Dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Define seed")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Convert Namespace to dictionary
    args_dict = vars(args)
    for key, value in args_dict.items():
        logging.info(f"{key}: {value}")


    origin_dir = args.origin_dir

    # Data Preparation
    data_train = xDB_Siamese_Dataset(
        origin_dir=origin_dir,
        mode=args.mode,
        transform=get_train_augmentation_pipeline(image_size=(512, 512), max_pixel_value=1,  mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        type="train",
        val_ratio=0.1,
        test_ratio=0.1,
        seed=args.seed
    )

    data_val = xDB_Siamese_Dataset(
        origin_dir=origin_dir,
        mode=args.mode,
        transform=get_val_augmentation_pipeline(max_pixel_value=1, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        type="val",
        val_ratio=0.1,
        test_ratio=0.1,
        seed=args.seed
    )

    data_test = xDB_Siamese_Dataset(
        origin_dir=origin_dir,
        mode=args.mode,
        transform=get_val_augmentation_pipeline(max_pixel_value=1, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        type="test",
        val_ratio=0.1,
        test_ratio=0.1,
        seed=args.seed
    )

    # Model Initialization
    model = SiameseResNetUNet(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        backbone_name=args.backbone,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    model.to("cuda")

    # Dataloaders
    train_dl = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_dl = DataLoader(data_val, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Loss and Optimizer
    mode = "multiclass"
    reduction = "weighted"

    # building, full_damage, simple_damage, change_detection"
    if args.mode == "full_damage":
        class_weights = [1.0, 10.0, 30.0, 30.0, 30.0]
    elif args.mode == "simple_damage":
        class_weights = [1.0, 10.0, 30.0]
    elif args.mode == "change_detection":
        class_weights = [1.0, 10.0]

    
    optimizer = optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0001)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights), reduction='mean').to("cuda")

    # Metrics
    metrics = [balanced_accuracy, f1_score, iou_score, precision, recall]

    # Early Stopping
    early_stopping_params = {"patience": 10, "trigger_times": 3}

    # Training
    torch.cuda.empty_cache()

    train(
        model,
        train_dl=train_dl,
        valid_dl=val_dl,
        optimizer=optimizer,
        scheduler=scheduler,
        params_opt={},
        params_sc={},
        metrics=metrics,
        nb_epochs=args.num_epochs,
        num_classes=args.out_channels, 
        loss_fn=criterion,
        experiment_name=args.experiment_name,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        early_stopping_params=early_stopping_params,
        image_key="post_image",
        mask_key="post_mask",
        training_log_interval=2,
        verbose=True,
        is_mixed_precision=args.mixed_precision,
        reduction=reduction,
        class_weights=class_weights,
        siamese=True
    )

    # Testing
    test_dl = DataLoader(data_test, batch_size=5, shuffle=True)

    epoch_tloss, test_metrics = testing(
        model,
        test_dataloader=test_dl,
        loss_fn=criterion,
        metrics=metrics,
        image_key="post_image",
        mask_key="post_mask",
        verbose=True,
        is_mixed_precision=args.mixed_precision,
        num_classes=args.out_channels,
        reduction=reduction,
        class_weights=class_weights,
        tta=args.tta,
        siamese=True,
    )

    print("Test Time Augmentation:", args.tta)
    print("Testing Loss:", epoch_tloss)

    for name, value in test_metrics.items():
        print(f"{name} : {value.item()}")
    
    # Store test metrics
    test_results = {
        "Test Time Augmentation": args.tta,
        "Testing Loss": epoch_tloss,
        "Metrics": {name: value.item() for name, value in test_metrics.items()},
    }

    # File path for saving the results
    results_file = f"{args.experiment_name}_test_metrics.json"

    # Save to a JSON file
    with open(results_file, "w") as f:
        json.dump(test_results, f, indent=4)
    
    torch.cuda.empty_cache()
