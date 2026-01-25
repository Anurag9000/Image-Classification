import os
import cv2
import json
import numpy as np
import albumentations as A
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
import argparse
import logging
from pipeline.augmentations import get_advanced_transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Augment dataset offline.")
    parser.add_argument("--input", default="./data/Dataset_Final", help="Input dataset directory")
    parser.add_argument("--output", default="./data/Dataset_Final_Aug", help="Output directory")
    parser.add_argument("--variations", type=int, default=10, help="Variations per image")
    return parser.parse_args()

# Configuration
# Loaded from args



def augment_dataset(input_dir, output_dir, num_variations, metadata_file):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        LOGGER.error(f"Input directory {input_path} does not exist.")
        return

    LOGGER.info(f"Augmenting from: {input_path}")
    LOGGER.info(f"Saving to: {output_path}")
    LOGGER.info(f"Variations per image: {num_variations}")

    transform = get_advanced_transforms()
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

    LOGGER.info(f"Found {total_files} original images.")
    
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
                    LOGGER.error(f"Error augmenting {img_path.name}: {e}")

            pbar.update(1)

    pbar.close()
    
    # Save Metadata
    m_file = Path(metadata_file)
    m_file.parent.mkdir(parents=True, exist_ok=True)
    with open(m_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    LOGGER.info(f"Augmentation Complete. Metadata with {len(metadata)} entries saved to {m_file}")

if __name__ == "__main__":
    args = parse_args()
    meta_path = Path(args.output).parent / "dataset_aug_metadata.json"
    augment_dataset(args.input, args.output, args.variations, meta_path)
