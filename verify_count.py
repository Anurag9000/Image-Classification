import json
import os
from pathlib import Path

import argparse

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
            print(f"Metadata JSON Count: {len(data)}")
    else:
        print(f"Metadata file not found: {json_path}")

    # 2. Recursive Scan
    print(f"Scanning {data_path} recursively...")
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    count = 0
    if data_path.exists():
        for root, dirs, files in os.walk(data_path):
            for f in files:
                if Path(f).suffix.lower() in image_extensions:
                    count += 1
        print(f"Recursive Scan Count: {count}")
    else:
        print(f"Dataset directory not found: {data_path}")

if __name__ == "__main__":
    args = parse_args()
    verify(args.data, args.json)
