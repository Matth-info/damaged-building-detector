import argparse
import logging

from src.data.data_preprocessing import generate_tiles, generate_tiles_parallel

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Split a large GeoTIFF into smaller tiles.")
    parser.add_argument("--input_file", type=str, help="Path to the input GeoTIFF file.")
    parser.add_argument("--output_dir", type=str, help="Directory where the tiles will be saved.")
    parser.add_argument("--grid_x", type=int, default=256, help="Tile width in pixels.")
    parser.add_argument("--grid_y", type=int, default=256, help="Tile height in pixels.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of Workers")
    args = parser.parse_args()

    generate_tiles_parallel(
        args.input_file, args.output_dir, args.grid_x, args.grid_y, max_workers=args.num_workers
    )
    # generate_tiles(args.input_file, args.output_dir, args.grid_x, args.grid_y)


if __name__ == "__main__":
    main()
