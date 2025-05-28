import argparse
import logging
import os

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

from src.augmentation import Augmentation_pipeline
from src.datasets import DATASETS_MAP
from src.losses import LOSSES_MAP
from src.metrics import METRICS_MAP
from src.models import MODELS_MAP
from src.training import (
    OPTIMIZER_MAP,
    SCHEDULER_MAP,
    Trainer,
    define_weighted_random_sampler,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(path="config.yaml"):
    logger.debug(f"Extracting Training Information from {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_augmentation_pipeline(aug_cfg: str = None, mode: str = None):
    config_path = aug_cfg.get("config_path")

    if config_path and os.path.isfile(config_path):
        aug_pipe = Augmentation_pipeline.load_pipeline(filepath=config_path)

    aug_pipe = Augmentation_pipeline(
        image_size=tuple(aug_cfg["image_size"]),
        mean=aug_cfg["normalize_mean"],
        std=aug_cfg["normalize_std"],
        mode=mode,
    )
    logger.debug("Augmentation Pipeline has been successfully loaded")
    return aug_pipe


def load_dataset(data_cfg: dict, aug_cfg: dict, type: str = "train"):
    # load augmentation pipeline
    transform = load_augmentation_pipeline(aug_cfg, mode=type)
    data_cls = DATASETS_MAP[data_cfg["dataset"]]
    dataset = data_cls(transform=transform, type=type, **data_cfg)
    logger.debug(f"Dataset in {type} mode has been successfully loaded")
    return dataset


def setup_optimizer_scheduler_from_config(model, cfg: dict):
    optimizer_cfg = cfg.get("optimizer", {})
    scheduler_cfg = cfg.get("scheduler", {})

    optimizer_class = OPTIMIZER_MAP.get(optimizer_cfg.get("name", "AdamW"))
    optimizer_params = optimizer_cfg.get("params", {})

    scheduler_class = SCHEDULER_MAP.get(scheduler_cfg.get("name", "StepLR"))
    scheduler_params = scheduler_cfg.get("params", {})

    logger.debug("Optimize and Scheduler have been successfully loaded")
    return optimizer_class, scheduler_class, optimizer_params, scheduler_params


def main():
    parser = argparse.ArgumentParser(description="Run Model Training and Evaluation")
    parser.add_argument("--config_path", type=str, help="Path to the training config file.")

    args = parser.parse_args()
    config = load_config(args.config_path)

    experiment_cfg = config["experiment"]
    data_cfg = config["data"]
    dl_cfg = config["data_loader"]
    aug_cfg = config["augmentation"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    testing_cfg = config["testing"]
    mlflow_cfg = config["mlflow"]

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Dataset
    train_dataset = load_dataset(data_cfg, aug_cfg, type="train")
    val_dataset = load_dataset(data_cfg, aug_cfg, type="val")
    test_dataset = load_dataset(data_cfg, aug_cfg, type="test")

    if dl_cfg.get("sampler") == "weighted" and not training_cfg.get("class_weights"):
        sampler, class_weights = define_weighted_random_sampler(
            train_dataset, mask_key=dl_cfg["mask_key"], subset_size=100, seed=42
        )
        class_weights = [class_weights[key] for key in class_weights.keys()]
    elif dl_cfg.get("sampler") == "weighted" and training_cfg.get("class_weights"):
        sampler, _ = define_weighted_random_sampler(
            train_dataset, mask_key=dl_cfg["mask_key"], subset_size=100, seed=42
        )
        class_weights = training_cfg["class_weights"]
    elif not dl_cfg.get("sampler") and training_cfg.get("class_weights"):
        sampler, class_weights = None, training_cfg.get("class_weights")
    else:
        sampler, class_weights = None, None

    # Initialize Dataloader
    train_dl = DataLoader(
        train_dataset,
        batch_size=dl_cfg["batch_size"],
        num_workers=dl_cfg["num_workers"],
        shuffle=False if sampler else True,
        sampler=sampler,
        pin_memory=True,
        drop_last=False,
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=dl_cfg["batch_size"],
        num_workers=dl_cfg["num_workers"],
        shuffle=False,
        pin_memory=True,
    )
    test_dl = DataLoader(
        test_dataset,
        batch_size=dl_cfg["batch_size"],
        num_workers=dl_cfg["num_workers"],
        shuffle=False,
    )

    # Load Model
    model = MODELS_MAP[model_cfg["name"]](**model_cfg).to(device)
    logger.debug("Model has been successfully initialized")

    # Load Metrics
    metrics = [METRICS_MAP[metric_name] for metric_name in testing_cfg["metrics"]]
    logger.debug("Metrics have been successfully initialized")

    # Define Criterion
    criterion = LOSSES_MAP[training_cfg["loss_fn"]](
        weight=torch.tensor(class_weights).float() if training_cfg["reduction"] == "weighted" else None
    ).to(device)
    logger.debug("Loss has been successfully initialized")
    # Extract optimizer and scheduler
    (
        optimizer_class,
        scheduler_class,
        optimizer_params,
        scheduler_params,
    ) = setup_optimizer_scheduler_from_config(model, training_cfg)

    # Define Trainer object
    trainer = Trainer(
        model=model,
        train_dl=train_dl,
        valid_dl=val_dl,
        test_dl=test_dl,
        optimizer=optimizer_class,
        scheduler=scheduler_class,
        optimizer_params=optimizer_params,
        scheduler_params=scheduler_params,
        nb_epochs=training_cfg["nb_epochs"],
        num_classes=model_cfg["out_channels"],
        loss_fn=criterion,
        metrics=metrics,
        mask_key=dl_cfg["mask_key"],
        image_key=dl_cfg["image_key"],
        experiment_name=experiment_cfg["name"],
        siamese=experiment_cfg["siamese"],
        is_mixed_precision=training_cfg["is_mixed_precision"],
        tta=testing_cfg["tta"],
        training_log_interval=experiment_cfg["training_log_interval"],
        device=device,
        debug=experiment_cfg["debug"],
        verbose=experiment_cfg["verbose"],
        checkpoint_interval=experiment_cfg["checkpoint_interval"],
        early_stopping_params=training_cfg["early_stopping"],
        reduction=training_cfg["reduction"],
        class_weights=training_cfg["class_weights"],
        class_names=testing_cfg["class_names"],
        enable_system_metrics=mlflow_cfg["enable_system_metrics"],
        tracking_uri=mlflow_cfg["tracking_uri"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"]
    )

    trainer.train()

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
