# run_mask_postprocessing.py
import argparse
import logging

import yaml

from src.data.data_postprocessing import process_masks_parallel

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path="config/inference.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Converting Prediction Masks into Vector objects")
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to the inference config file."
    )
    parser.add_argument("--output_name", type=str, default="vectorized_predictions.gpkg")
    args = parser.parse_args()

    config = load_config(args.config_path)
    config_data = config["data"]
    config_inference = config["inference"]

    # Extract necessary fields
    folder_path = config_data["output_dir"]
    mode = config_inference["color_mode"]
    file_suffix = f".{config_data['extension'].lstrip('.')}"
    num_workers = config_inference["num_workers"]

    # Optional/fixed values
    output_path = f"outputs/{args.output_name}"
    min_area = 80
    layer_name = "polygons"

    logger.info("Starting vector postprocessing from inference outputs...")

    process_masks_parallel(
        folder_path=folder_path,
        mode=mode,
        output_path=output_path,
        min_area=min_area,
        layer_name=layer_name,
        file_suffix=file_suffix,
        num_workers=num_workers,
    )
    logger.info("Vector postprocessing complete.")


if __name__ == "__main__":
    main()
