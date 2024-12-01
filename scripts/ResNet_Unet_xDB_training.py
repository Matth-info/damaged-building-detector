import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


# Custom librairis
from datasets import xDB_Damaged_Building
from training import train, testing
from training.augmentations import (
    augmentation_test_time,
    get_train_augmentation_pipeline,
    get_val_augmentation_pipeline,
)
from models import ResNet_UNET
from losses import DiceLoss, FocalLoss, Ensemble
from metrics import f1_score, iou_score, balanced_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet-based U-Net model for building segmentation.")
    
    # Experiment settings
    parser.add_argument("--experiment_name", type=str, default="Experiment", help="Name of the experiment for logging.")
    parser.add_argument("--backbone", type=str, default="resnet18", help="ResNet backbone to use (e.g., resnet18, resnet34, resnet50).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and validation.")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs for training.")
    parser.add_argument("--learning_rate", type=float, default=0.00015, help="Learning rate for the optimizer.")
    parser.add_argument("--pretrained", action="store_true", help="Use a pretrained backbone.")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze the ResNet backbone during training.")
    parser.add_argument("--log_dir", type=str, default="../runs", help="Directory for storing logs.")
    parser.add_argument("--model_dir", type=str, default="../models", help="Directory for saving trained models.")
    parser.add_argument("--tta", action="store_true", help="Enable Test Time Augmentation (TTA).")
    parser.add_argument("--mixed_precision", action="store_true", help="Enable mixed-precision training.")
    parser.add_argument("--origin_dir", type=str, default="../data/xDB/tier3", help="Local path to xDB Dataset.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    origin_dir = args.origin_dir

    # Data Preparation
    data_train = xDB_Damaged_Building(
        origin_dir=origin_dir,
        mode="building",
        time="pre",
        transform=get_train_augmentation_pipeline(image_size=(512, 512), max_pixel_value=1),
        type="train",
        val_ratio=0.1,
        test_ratio=0.1,
    )

    data_val = xDB_Damaged_Building(
        origin_dir=origin_dir,
        mode="building",
        time="pre",
        transform=get_val_augmentation_pipeline(image_size=(512, 512), max_pixel_value=1),
        type="val",
        val_ratio=0.1,
        test_ratio=0.1,
    )

    data_test = xDB_Damaged_Building(
        origin_dir=origin_dir,
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
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    model.to("cuda")

    # Dataloaders
    train_dl = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_dl = DataLoader(data_val, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Loss and Optimizer
    mode = "multiclass"
    reduction = "weighted"
    class_weights = [0.1, 0.9]

    optimizer = optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 11, 17, 25, 33, 47, 50, 60, 70, 90, 110, 130, 150, 170, 180, 190], gamma=0.5)

    criterion = Ensemble(list_losses=[DiceLoss(mode=mode), FocalLoss(mode=mode)], weights=[1.0, 10.0]).cuda()

    # Metrics
    metrics = [balanced_accuracy, f1_score, iou_score]

    # Early Stopping
    early_stopping_params = {"patience": 5, "trigger_times": 0}

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
        loss_fn=criterion,
        experiment_name=args.experiment_name,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        early_stopping_params=early_stopping_params,
        image_key="image",
        training_log_interval=5,
        verbose=False,
        is_mixed_precision=args.mixed_precision,
        reduction=reduction,
        class_weights=class_weights,
    )

    # Testing
    test_dl = DataLoader(data_test, batch_size=5, shuffle=True)

    epoch_tloss, test_metrics = testing(
        model,
        test_dataloader=test_dl,
        loss_fn=criterion,
        metrics=metrics,
        image_key="image",
        mask_key="mask",
        verbose=True,
        is_mixed_precision=args.mixed_precision,
        num_classes=2,
        reduction=reduction,
        class_weights=class_weights,
        tta=args.tta,
    )
    print("Test Time Augmentation:", args.tta)
    print("Testing Loss:", epoch_tloss)

    for name, value in test_metrics.items():
        print(f"{name} : {value.item()}")
    
    torch.cuda.empty_cache()
