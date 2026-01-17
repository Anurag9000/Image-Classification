import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
METADATA_FILE = BASE_DIR / "data" / "dataset_metadata.json"
DATASET_DIR = BASE_DIR / "data" / "Dataset_Final"

def verify():
    # 1. Check Metadata Count
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r') as f:
            data = json.load(f)
            print(f"Metadata JSON Count: {len(data)}")
    else:
        print("Metadata file not found.")

    # 2. Recursive Scan
    print(f"Scanning {DATASET_DIR} recursively...")
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    count = 0
    for root, dirs, files in os.walk(DATASET_DIR):
        for f in files:
            if Path(f).suffix.lower() in image_extensions:
                count += 1
    
    print(f"Recursive Scan Count: {count}")

if __name__ == "__main__":
    verify()
