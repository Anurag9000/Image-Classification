import os
import cv2
import json
import numpy as np
import albumentations as A
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
NUM_VARIATIONS = 5
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "Dataset_Final"
OUTPUT_DIR = DATA_DIR / "Dataset_Final_Aug"
METADATA_FILE = DATA_DIR / "dataset_aug_metadata.json"

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
        
        # 1. GEOMETRIC (Shape variations)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5), # Garbage creates no orientation bias usually
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=180, p=0.8),
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.5, p=1.0),
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1.0),
            A.Perspective(scale=(0.05, 0.1), p=1.0),
        ], p=0.6),

        # 2. PIXEL-LEVEL (Dirt, Noise)
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=1.0), # Simulate dirt/holes
        ], p=0.4),

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
            A.ToGray(p=0.1), # Occasional grayscale
        ], p=0.5),
    ])

def augment_dataset():
    if not INPUT_DIR.exists():
        print(f"Error: Input directory {INPUT_DIR} does not exist.")
        return

    print(f"Augmenting from: {INPUT_DIR}")
    print(f"Saving to: {OUTPUT_DIR}")
    print(f"Variations per image: {NUM_VARIATIONS}")

    transform = get_garbage_transforms()
    metadata = []
    
    # Image extensions to look for
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    # Walk through input directory
    total_files = 0
    # First pass to count for tqdm
    for root, dirs, files in os.walk(INPUT_DIR):
        for f in files:
            if Path(f).suffix.lower() in valid_exts:
                total_files += 1

    print(f"Found {total_files} original images.")
    
    pbar = tqdm(total=total_files, desc="Augmenting")

    for class_dir in INPUT_DIR.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        output_class_dir = OUTPUT_DIR / class_name
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

            # 1. PROCESS ORIGINAL (Simply Copy/Resize or just Copy? Plan said "Copy original")
            # But we might want resize consistency? Let's just save the original as best quality 
            # or apply only Resize. Let's keep original "Same as source" effectively by saving the read image.
            # Actually, to save space/time and consistency, let's just resize original too if we want 224x224.
            # But user said "Folder exactly as they are". Let's simply copy using CV2 to ensure format consistency.
            
            # Save Original
            original_dest = output_class_dir / img_path.name
            # For metadata path, we want relative to DATA_DIR potentially
            try:
                rel_path = original_dest.relative_to(DATA_DIR)
            except ValueError:
                rel_path = original_dest
            
            cv2.imwrite(str(original_dest), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            metadata.append({
                "file_path": str(rel_path),
                "label": class_name,
                "type": "original"
            })

            # 2. GENERATE AUGMENTED
            for i in range(NUM_VARIATIONS):
                try:
                    aug = transform(image=image)
                    aug_img = aug['image']
                    
                    aug_name = f"{img_path.stem}_aug_{i}{img_path.suffix}"
                    aug_dest = output_class_dir / aug_name
                    
                    cv2.imwrite(str(aug_dest), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                    
                    try:
                        aug_rel_path = aug_dest.relative_to(DATA_DIR)
                    except ValueError:
                        aug_rel_path = aug_dest

                    metadata.append({
                        "file_path": str(aug_rel_path),
                        "label": class_name,
                        "type": "augmented"
                    })
                except Exception as e:
                    print(f"Error augmenting {img_path.name}: {e}")

            pbar.update(1)

    pbar.close()
    
    # Save Metadata
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Augmentation Complete. Metadata with {len(metadata)} entries saved to {METADATA_FILE}")

if __name__ == "__main__":
    augment_dataset()
