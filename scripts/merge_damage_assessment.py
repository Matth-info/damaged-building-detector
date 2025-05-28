import argparse
import logging
from pathlib import Path

from src.data.data_postprocessing import (
    GDP_DRIVER_MAPPING,
    dowload_footprints_aoi,
    find_quad_keys,
    merge_predictions_with_footprints,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run_pipeline(
    aoi_image_path: str, predictions_filepath: str, output_dir: str, output_format="geojson"
):
    # Create output path
    output_path = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / f"classified_building_footprints.{output_format}"
    footprint_file = output_path / f"building_footprints.{output_format}"
    driver_options = GDP_DRIVER_MAPPING[output_format]

    # 1. Get quad keys and download footprints
    logging.info("Finding quad keys and downloading footprints...")
    quad_keys, aoi_shape = find_quad_keys(base_image=aoi_image_path)
    dowload_footprints_aoi(quad_keys, footprint_file.stem, output_format, aoi_shape)

    # 2. Merge predictions
    logging.info("Merging predictions with footprints...")
    classified_footprints = merge_predictions_with_footprints(
        predictions_file=Path(predictions_filepath),
        footprints_file=footprint_file,
        max_distance=5,
        projected_crs=3857,
    )
    # 3. Save results
    classified_footprints.to_file(output_filename, **driver_options)
    logging.info(f"Saved merged GeoDataFrame to {output_filename}")

    return output_filename


def main():
    parser = argparse.ArgumentParser(
        description="Merge Damage assessment with reference footprints"
    )
    parser.add_argument("--aoi_image", type=str, required=True, help="Path to TIF image for AOI")
    parser.add_argument(
        "--predictions_filepath", type=str, required=True, help="Path to vector predictions file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Directory to save outputs"
    )
    parser.add_argument(
        "--format", type=str, default="geojson", choices=["geojson", "gpkg"], help="Output format"
    )

    args = parser.parse_args()
    run_pipeline(args.aoi_image, args.predictions_filepath, args.output_dir, args.format)


if __name__ == "__main__":
    main()
