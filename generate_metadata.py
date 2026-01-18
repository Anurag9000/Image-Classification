import os
import json
import glob
from pathlib import Path

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

def generate_metadata(dataset_root, output_file):
    """
    Generates a metadata JSON file for an image classification dataset.
    
    Args:
        dataset_root (str): Path to the root folder of the dataset (containing class subfolders).
        output_file (str): Path to the output JSON file.
    """
    
    dataset_path = Path(dataset_root)
    if not dataset_path.exists():
        LOGGER.error(f"Dataset directory not found at {dataset_root}")
        return

    metadata = []
    
    # Valid image extensions
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp']
    
    LOGGER.info(f"Scanning dataset at: {dataset_path.absolute()}")

    # Iterate through class folders
    for class_dir in dataset_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            LOGGER.info(f"Found class: {class_name}")
            
            image_files = []
            for ext in image_extensions:
                # Case-insensitive search pattern for windows if needed, but glob is usually case-sensitive on linux
                # On Windows, glob itself is often case-insensitive depending on python version/OS settings.
                # To be safe and simple, we iterate.
                image_files.extend(list(class_dir.glob(ext)))
                image_files.extend(list(class_dir.glob(ext.upper()))) # Handle uppercase extensions just in case

            # Remove duplicates if case-insensitivity matched twice
            image_files = list(set(image_files))

            for img_path in image_files:
                # Store relative path to make it portable or absolute if preferred.
                # User asked for "file", relative is usually better for portability within project.
                # Let's verify what the user might expect. Relative to the script execution or dataset root?
                # Usually relative to the metadata file or project root is best. 
                # Let's store the absolute path for clarity now, or relative to the data folder.
                # Let's go with relative to the `data` folder (parent of Dataset_Final) generally?
                # Or just relative to the script location if script is at correct place.
                
                # Let's make paths relative to the directory containing the json file for maximizing portability
                # if the json is in `data/`, and images in `data/Dataset_Final/...`
                
                try:
                    rel_path = img_path.relative_to(Path(output_file).parent)
                    file_path_str = str(rel_path)
                except ValueError:
                    # Fallback to absolute strings if relative fails
                    file_path_str = str(img_path.absolute())

                metadata.append({
                    "file_path": file_path_str,
                    "label": class_name
                })
                
    LOGGER.info(f"Found {len(metadata)} total images.")
    
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    LOGGER.info(f"Metadata saved to {output_file}")

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate dataset metadata JSON.")
    parser.add_argument("--dataset-root", default="./data/Dataset_Final", help="Path to dataset root")
    parser.add_argument("--output", default="./data/dataset_metadata.json", help="Output JSON path")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Resolve paths
    dataset_root = Path(args.dataset_root).resolve()
    output_file = Path(args.output).resolve()
    
    LOGGER.info(f"Generating metadata...")
    LOGGER.info(f"Dataset: {dataset_root}")
    LOGGER.info(f"Output: {output_file}")
    
    generate_metadata(str(dataset_root), str(output_file))
