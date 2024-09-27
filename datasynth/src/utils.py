import json
import logging
import os

import yaml


def load_config(config_path: str) -> dict:
    """Loads the configuration from a YAML or JSON file."""
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    ext = os.path.splitext(config_path)[-1].lower()

    try:
        with open(config_path, "r") as f:
            if ext == ".json":
                return json.load(f)
            elif ext == ".yaml" or ext == ".yml":
                return yaml.safe_load(f)
            else:
                logging.error(f"Unsupported config file format: {ext}")
                raise ValueError(f"Unsupported config file format: {ext}")
    except Exception as e:
        logging.error(f"Error loading configuration: {str(e)}")
        raise e


def delete_video_files(output_folder: str) -> None:
    """Deletes all files in the specified folder and removes the folder itself."""
    if os.path.exists(output_folder):
        for root, dirs, files in os.walk(output_folder):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                resp = input(f"Delete folder {os.path.join(root, dir)}? (y/n): ")
                if resp.lower() == "y":
                    os.rmdir(os.path.join(root, dir))
        os.rmdir(output_folder)
        logging.info(f"Deleted folder: {output_folder}")
    else:
        logging.warning(f"Output folder not found: {output_folder}")
