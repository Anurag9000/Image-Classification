
import os
import cv2
import pandas as pd
import numpy as np
import albumentations as A
import yaml
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from pipeline.files_dataset import CombinedFilesDataset
from pipeline.augmentations import get_heavy_transforms

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate V4 dataset with heavy augmentations.")
    parser.add_argument("--config", default="configs/garbage_edge.yaml", help="Path to config file")
    parser.add_argument("--output", default="./data/Dataset_V4", help="Output directory")
    parser.add_argument("--variations", type=int, default=50, help="Number of variations per training image")
    return parser.parse_args()

# --- Configuration ---
# Loaded from args in main()



def generate_v4(config_path, output_root, num_variations):
    LOGGER.info(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    root_dirs = cfg['dataset']['root_dirs']
    LOGGER.info(f"Source Roots: {root_dirs}")
    LOGGER.info(f"Target Root: {output_root}")
    LOGGER.info(f"Variations per Image: {num_variations}")

    # Load Source Dataset (Train split mainly, we can do valid/test too if needed but usually we only augment train)
    # Actually, for offline v4, we should probably merge EVERYTHING into a massive train folder 
    # and let the loader split it later, OR keep the split structure.
    # User asked for "entire folder with all images", implies a merged dataset.
    # Let's process 'train', 'valid', 'test' from sources and map them to v4 structure.
    
    splits = ['train', 'valid', 'test']
    transform = get_heavy_transforms()
    
    total_generated = 0

    for split in splits:
        LOGGER.info(f"\nProcessing Split: {split}")
        dataset = CombinedFilesDataset(root_dirs, split=split, transform=None) # Load raw first
        
        if len(dataset) == 0:
            LOGGER.info(f"Skipping empty split: {split}")
            continue

        # Create output directories
        split_out_dir = os.path.join(output_root, split)
        os.makedirs(split_out_dir, exist_ok=True)
        
        # Prepare CSV data for this split
        csv_data = [] # List of dicts {filename, CLASS1:0, CLASS2:1 ...}
        class_names = dataset.class_names
        LOGGER.info(f"Classes: {class_names}")

        for i in tqdm(range(len(dataset))):
            src_img_path, label_idx = dataset.samples[i]
            label_name = class_names[label_idx]
            
            # Read image explicitly to avoid dataset's internal transform logic
            image = cv2.imread(src_img_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Generate Variations
            # For validation/test, usually we DO NOT augment heavily. 
            # But user request "exhaustively all possible transforms... v4". 
            # Standard Practice: Only augment TRAIN. 
            # I will follow standard practice: Augment TRAIN x 25. Copy VALID/TEST x 1 (Original).
            
            num_vars = num_variations if split == 'train' else 1
            is_aug = (split == 'train')

            for v in range(num_vars):
                if is_aug:
                    # Apply transform
                    aug = transform(image=image)
                    img_out = aug['image']
                    suffix = f"_aug_{v}"
                else:
                    img_out = image
                    suffix = ""
                
                # Convert back to BGR for saving
                img_out_bgr = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
                
                # Filename: original_name_aug_0.jpg
                base_name = os.path.basename(src_img_path)
                name_part, ext = os.path.splitext(base_name)
                new_filename = f"{name_part}{suffix}{ext}"
                
                save_path = os.path.join(split_out_dir, new_filename)
                cv2.imwrite(save_path, img_out_bgr)
                
                # Add to CSV entry
                row = {'filename': new_filename}
                # One-hot
                for c in class_names:
                    row[c] = 1 if c == label_name else 0
                csv_data.append(row)
                total_generated += 1

        # Save _classes.csv for this split
        if csv_data:
            df = pd.DataFrame(csv_data)
            # Ensure column order matches standard
            cols = ['filename'] + class_names
            df = df[cols]
            csv_out_path = os.path.join(split_out_dir, '_classes.csv')
            df.to_csv(csv_out_path, index=False)

    LOGGER.info(f"V4 Generation Complete! Total Images: {total_generated}")

if __name__ == "__main__":
    args = parse_args()
    generate_v4(args.config, args.output, args.variations)
