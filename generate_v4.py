
import os
import cv2
import pandas as pd
import numpy as np
import albumentations as A
import yaml
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from pipeline.files_dataset import CombinedFilesDataset

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate V4 dataset with heavy augmentations.")
    parser.add_argument("--config", default="configs/garbage_edge.yaml", help="Path to config file")
    parser.add_argument("--output", default="./data/Dataset_V4", help="Output directory")
    parser.add_argument("--variations", type=int, default=50, help="Number of variations per training image")
    return parser.parse_args()

# --- Configuration ---
# Loaded from args in main()

def get_heavy_transforms():
    """
    Returns the heavy augmentation pipeline that was causing CPU bottlenecks.
    We apply this offline now.
    """
    return A.Compose([
        A.Resize(224, 224),
        
        # 1. GEOMETRIC TRANSFORMS (The "Shape" shifters)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5), # Garbage has no gravity orientation
        A.ShiftScaleRotate(shift_limit=0.3, scale_limit=0.3, rotate_limit=180, p=0.8),
        
        # 2. HEAVY MORPHOLOGICAL (The "Crushed/Mangled" look)
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=1.0, p=1.0),
            A.OpticalDistortion(distort_limit=1.0, shift_limit=1.0, p=1.0),
            A.Perspective(scale=(0.1, 0.2), p=1.0),
            A.PiecewiseAffine(scale=(0.03, 0.05), nb_rows=4, nb_cols=4, p=1.0), # New: Local warping
        ], p=0.9),

        # 3. PIXEL-LEVEL CORRUPTIONS (The "Camera/Sensor" flaws)
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0), # New: Sensor noise
            A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, p=1.0),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=1.0),
        ], p=0.7),

        # 4. BLUR & QUALITY (The "Bad Camera" look)
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=7, p=1.0),
            A.GaussianBlur(blur_limit=7, p=1.0),
            A.ImageCompression(quality_lower=50, quality_upper=100, p=1.0), # New: JPEG artifacts
            A.Downscale(scale_min=0.5, scale_max=0.9, p=1.0), # New: Low res simulation
        ], p=0.5),

        # 5. LIGHTING & COLOR (The "Environment" changes)
        A.OneOf([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0), # New: Adaptive Histogram
            A.ChannelShuffle(p=1.0), # New: Wrong color channels
            A.Solarize(threshold=128, p=1.0), # New: Extreme lighting
            A.ToGray(p=1.0), # New: Grayscale check
        ], p=0.8),
    ])

def generate_v4(config_path, output_root, num_variations):
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    root_dirs = cfg['dataset']['root_dirs']
    print(f"Source Roots: {root_dirs}")
    print(f"Target Root: {output_root}")
    print(f"Variations per Image: {num_variations}")

    # Load Source Dataset (Train split mainly, we can do valid/test too if needed but usually we only augment train)
    # Actually, for offline v4, we should probably merge EVERYTHING into a massive train folder 
    # and let the loader split it later, OR keep the split structure.
    # User asked for "entire folder with all images", implies a merged dataset.
    # Let's process 'train', 'valid', 'test' from sources and map them to v4 structure.
    
    splits = ['train', 'valid', 'test']
    transform = get_heavy_transforms()
    
    total_generated = 0

    for split in splits:
        print(f"\nProcessing Split: {split}")
        dataset = CombinedFilesDataset(root_dirs, split=split, transform=None) # Load raw first
        
        if len(dataset) == 0:
            print(f"Skipping empty split: {split}")
            continue

        # Create output directories
        split_out_dir = os.path.join(output_root, split)
        os.makedirs(split_out_dir, exist_ok=True)
        
        # Prepare CSV data for this split
        csv_data = [] # List of dicts {filename, CLASS1:0, CLASS2:1 ...}
        class_names = dataset.class_names
        print(f"Classes: {class_names}")

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

    print(f"V4 Generation Complete! Total Images: {total_generated}")

if __name__ == "__main__":
    args = parse_args()
    generate_v4(args.config, args.output, args.variations)
            print(f"Saved {len(df)} entries to {csv_out_path}")

    print(f"\nDone! Generated {total_generated} images in total at {OUTPUT_ROOT}")

if __name__ == "__main__":
    generate_v4()
