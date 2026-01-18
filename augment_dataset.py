import os
import cv2
import json
import numpy as np
import albumentations as A
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Augment dataset offline.")
    parser.add_argument("--input", default="./data/Dataset_Final", help="Input dataset directory")
    parser.add_argument("--output", default="./data/Dataset_Final_Aug", help="Output directory")
    parser.add_argument("--variations", type=int, default=10, help="Variations per image")
    return parser.parse_args()

# Configuration
# Loaded from args

def get_garbage_transforms():
    """
    Returns a heavy augmentation pipeline designed for garbage classification.
    Simulates:
    - Shape distortions (crushed cans, bent paper)
    - Environmental factors (dirty lens, lighting, shadows)
    - Camera quality issues (noise, blur, compression)
    """
    return A.Compose([
        A.Resize(224, 224), # Ensure consistent size
        
        # 1. GEOMETRIC (Shape variations) - High probability for garbage
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5), 
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=180, p=0.8),
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.5, p=1.0),
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1.0),
            A.Perspective(scale=(0.05, 0.1), p=1.0),
        ], p=0.7), # Increased from 0.6 to 0.7 for more shape variety

        # 2. PIXEL-LEVEL (Dirt, Noise)
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=1.0),
        ], p=0.5), # Increased from 0.4

        # 3. BLUR & QUALITY
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.ImageCompression(quality_lower=60, quality_upper=100, p=1.0),
        ], p=0.3),

        # 4. LIGHTING & COLOR
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            A.CLAHE(clip_limit=4.0, p=1.0),
            A.ToGray(p=0.1),
        ], p=0.6), # Increased from 0.5 to ensure lighting variety
    ])

def augment_dataset(input_dir, output_dir, num_variations, metadata_file):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist.")
        return

    print(f"Augmenting from: {input_path}")
    print(f"Saving to: {output_path}")
    print(f"Variations per image: {num_variations}")

    transform = get_garbage_transforms()
    metadata = []
    
    # Image extensions to look for
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    # Walk through input directory
    total_files = 0
    # First pass to count for tqdm
    for root, dirs, files in os.walk(input_path):
        for f in files:
            if Path(f).suffix.lower() in valid_exts:
                total_files += 1

    print(f"Found {total_files} original images.")
    
    pbar = tqdm(total=total_files, desc="Augmenting")

    for class_dir in input_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        output_class_dir = output_path / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() not in valid_exts:
                continue
                
            # Read Image
            image = cv2.imread(str(img_path))
            if image is None:
                pbar.update(1)
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Save Original
            original_dest = output_class_dir / img_path.name
            
            # Save the original (resized to match augmented if desired, or just saved)
            # Here we save it as BGR for consistency
            cv2.imwrite(str(original_dest), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # Recompute relative path for metadata
            # We'll use the output_path's parent or just the relative path to output_path
            rel_path = str(original_dest.relative_to(output_path.parent))
            
            metadata.append({
                "file_path": rel_path,
                "label": class_name,
                "type": "original"
            })

            # 2. GENERATE AUGMENTED
            for i in range(num_variations):
                try:
                    aug = transform(image=image)
                    aug_img = aug['image']
                    
                    aug_name = f"{img_path.stem}_aug_{i}{img_path.suffix}"
                    aug_dest = output_class_dir / aug_name
                    
                    cv2.imwrite(str(aug_dest), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                    
                    aug_rel_path = str(aug_dest.relative_to(output_path.parent))

                    metadata.append({
                        "file_path": aug_rel_path,
                        "label": class_name,
                        "type": "augmented"
                    })
                except Exception as e:
                    print(f"Error augmenting {img_path.name}: {e}")

            pbar.update(1)

    pbar.close()
    
    # Save Metadata
    m_file = Path(metadata_file)
    m_file.parent.mkdir(parents=True, exist_ok=True)
    with open(m_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Augmentation Complete. Metadata with {len(metadata)} entries saved to {m_file}")

if __name__ == "__main__":
    args = parse_args()
    meta_path = Path(args.output).parent / "dataset_aug_metadata.json"
    augment_dataset(args.input, args.output, args.variations, meta_path)
