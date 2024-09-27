"""
This script processes YouTube videos into audio segments using a specified configuration file.

Usage:
$python synthesize.py --config <path_to_config_yaml>
"""

import argparse
import logging

from src.audio import synthesize
from src.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_args() -> argparse.Namespace:
    """Sets up the argument parser for the script."""
    parser = argparse.ArgumentParser(description="Process YouTube video to audio segments")
    parser.add_argument(
        "-c", "--config", required=False, help="Path to the configuration file",
        default="config.yaml"
    )
    return parser.parse_args()


def main():
    """ Entry point for data synthesis."""
    args = get_args()
    config = load_config(args.cf_path)
    synthesize(**config)


if __name__ == "__main__":
    main()
