# scripts/batch_inference.py
# python .\scripts\batch_inference.py --base_dir data/processed_data --pre_disaster_dir  Pre_Event_San_Juan  --post_disaster_dir  Post_Event_San_Juan  --output_dir  data/predictions  --batch_size 16 --extension  tif --num_workers 8 --save
import argparse
import logging
import os

import torch
import yaml
from torch.utils.data import DataLoader

from src.augmentation import Augmentation_pipeline
from src.datasets import Dataset_Inference_Siamese, Levir_cd_dataset
from src.inference import batch_inference, custom_infer_collate_siamese
from src.models import SiameseResNetUNet

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model_checkpoint(
    model_path: str,
    in_channels: int = 3,
    out_channels: int = 2,
    backbone: str = "resnet18",
    mode: str = "conc",
):
    if not os.path.isfile(model_path):
        logger.error(f"Model checkpoint not found at {model_path}")
        raise FileNotFoundError(model_path)

    model = SiameseResNetUNet(in_channels, out_channels, backbone, pretrained=False, mode=mode)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def load_dataset(base_dir, pre_dir, post_dir, transform, extension):
    return Dataset_Inference_Siamese(
        origin_dir=base_dir,
        pre_disaster_dir=pre_dir,
        post_disaster_dir=post_dir,
        transform=transform,
        extension=extension,
    )


def load_augmentation_pipeline(aug_cfg: str = None):
    config_path = aug_cfg.get("config_path")

    if config_path and os.path.isfile(config_path):
        logger.info(f"Loading augmentation config from: {config_path}")
        return Augmentation_pipeline.load_pipeline(filepath=config_path)

    logger.info("Using default augmentation pipeline.")

    return Augmentation_pipeline(
        image_size=tuple(aug_cfg["image_size"]),
        mean=aug_cfg["normalize_mean"],
        std=aug_cfg["normalize_std"],
        mode=aug_cfg["mode"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run Siamese model inference on tiled image data."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/inference.yaml",
        help="Inference Config file path",
    )
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)
        model_cfg = config.get("model")
        data_cfg = config.get("data")
        infer_cfg = config.get("inference")
        aug_cfg = config.get("augmentation")

    # Setup device
    device = torch.device(
        "cuda"
        if model_cfg["device"] == "auto" and torch.cuda.is_available()
        else model_cfg["device"]
    )
    logger.info(f"Using device: {device}")

    # Load model
    model = load_model_checkpoint(
        model_path=model_cfg["path"],
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        backbone=model_cfg["backbone"],
        mode=model_cfg["mode"],
    )
    model.to(device)
    logger.info("Model loaded successfully.")

    # Augmentation

    transform = load_augmentation_pipeline(aug_cfg)

    # Load dataset & dataloader

    # Dataset & loader
    dataset = load_dataset(
        base_dir=data_cfg["base_dir"],
        pre_dir=data_cfg["pre_disaster_dir"],
        post_dir=data_cfg["post_disaster_dir"],
        transform=transform,
        extension=data_cfg["extension"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=infer_cfg["batch_size"],
        shuffle=False,
        num_workers=infer_cfg["num_workers"],
        collate_fn=custom_infer_collate_siamese,
    )

    logger.info(f"Loaded {len(dataset)} samples for inference.")

    # Run inference
    batch_inference(
        model=model,
        dataloader=dataloader,
        device=device,
        siamese=infer_cfg["siamese"],
        pred_folder_path=data_cfg["output_dir"],
        color_mode=infer_cfg["color_mode"],
        save=infer_cfg["save_outputs"],
    )

    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
