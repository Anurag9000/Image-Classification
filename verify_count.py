import json
import os
from pathlib import Path

import argparse

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Verify dataset counts.")
    parser.add_argument("--data", default="./data/Dataset_Final", help="Dataset directory")
    parser.add_argument("--json", default="./data/dataset_metadata.json", help="Metadata JSON file")
    return parser.parse_args()

def verify(data_dir, json_file):
    data_path = Path(data_dir)
    json_path = Path(json_file)

    # 1. Check Metadata Count
    if json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
            LOGGER.info(f"Metadata JSON Count: {len(data)}")
    else:
        LOGGER.warning(f"Metadata file not found: {json_path}")

    # 2. Recursive Scan
    LOGGER.info(f"Scanning {data_path} recursively...")
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    count = 0
    if data_path.exists():
        for root, dirs, files in os.walk(data_path):
            for f in files:
                if Path(f).suffix.lower() in image_extensions:
                    count += 1
        LOGGER.info(f"Recursive Scan Count: {count}")
    else:
        LOGGER.error(f"Dataset directory not found: {data_path}")

if __name__ == "__main__":
    args = parse_args()
    verify(args.data, args.json)
