import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from typing import List, Optional, Callable

class CombinedFilesDataset(Dataset):
    """
    A dataset that combines multiple root directories, each containing:
    - train/test/valid subfolders
    - _classes.csv inside those subfolders mapping filenames to one-hot labels.
    """
    def __init__(self, 
                 root_dirs: List[str], 
                 split: str = 'train', 
                 transform: Optional[A.Compose] = None,
                 class_names: List[str] = None):
        """
        Args:
            root_dirs: List of absolute paths to dataset roots (e.g. ['.../v1', '.../v2'])
            split: 'train', 'valid', or 'test'
            transform: Albumentations transform pipeline
            class_names: Expected list of class names (columns in CSV). 
                         If None, inferred from first CSV (but must be consistent).
        """
        self.root_dirs = root_dirs
        self.split = split
        self.transform = transform
        self.samples = []
        self.class_names = class_names

        for root in root_dirs:
            split_dir = os.path.join(root, split)
            csv_path = os.path.join(split_dir, '_classes.csv')
            
            if not os.path.exists(csv_path):
                print(f"Warning: {csv_path} not found. Skipping...")
                continue
                
            # Read CSV
            # Format: filename, CLASS1, CLASS2, ... (one-hot encoded usually 0/1)
            # Lines often have extra spaces, so strip whitespace from headers
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            
            # Filter for label columns (exclude 'filename')
            label_cols = [c for c in df.columns if c.lower() != 'filename']
            
            if self.class_names is None:
                self.class_names = label_cols
            else:
                # Ensure consistency
                if label_cols != self.class_names:
                    # Depending on strictness, we might raise error or just warn.
                    # For now assume they are consistent as per user context (both had 6 classes)
                    pass
            
            # Iterate rows
            for _, row in df.iterrows():
                fname = row['filename'].strip()
                img_path = os.path.join(split_dir, fname)
                
                # Get label index (argmax of one-hot)
                # Convert row part to float array first
                labels_one_hot = row[self.class_names].values.astype(float)
                label_idx = np.argmax(labels_one_hot)
                
                self.samples.append((img_path, int(label_idx)))

        print(f"Loaded {len(self.samples)} samples for split '{split}' from {len(root_dirs)} roots.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            # Handle missing image gracefully-ish (or error out)
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Default to basic tensor conversion if no transforms provided
            image = A.Compose([A.Resize(224, 224), ToTensorV2()])(image=image)['image']

        return image, label

def get_garbage_transforms(is_training: bool = True, img_size: int = 224):
    """
    Returns Albumentations transforms.
    Includes robust augmentations for training: Warp, Morph, etc.
    """
    if is_training:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
            
            # Advanced Augmentations requested by user
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1.0),
            ], p=0.3), # 30% chance to apply one of the warping/morphing effects

            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

def create_garbage_loader(
    root_dirs: List[str], 
    batch_size: int, 
    num_workers: int = 2, 
    val_split: float = 0.0 # Not used if we assume 'valid' folder exists, but kept for signature comp
):
    """
    Creates train/val/test loaders from the v1/v2 structure.
    """
    train_transform = get_garbage_transforms(is_training=True)
    val_transform = get_garbage_transforms(is_training=False)
    
    # We will use the explicit 'train' and 'valid' folders if they exist in all roots
    # Otherwise we might fall back to splitting 'train'
    
    # Check if 'valid' exists in the first root as a heuristic
    has_valid_folder = os.path.exists(os.path.join(root_dirs[0], 'valid'))
    
    if has_valid_folder:
        train_dataset = CombinedFilesDataset(root_dirs, split='train', transform=train_transform)
        val_dataset = CombinedFilesDataset(root_dirs, split='valid', transform=val_transform)
    else:
        # Fallback: Load 'train' and split it
        # Note: This is simplified. Ideally we'd support splitting here.
        # But given user data schema has 'valid' folders, we prioritize that.
        full_dataset = CombinedFilesDataset(root_dirs, split='train', transform=train_transform)
        # Simple split logic could go here if needed, but likely not needed for this user task
        # returning full dataset as train for now if no valid folder found (or handling downstream)
        train_dataset = full_dataset
        val_dataset = None 

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True
        )

    return train_loader, val_loader

def create_garbage_test_loader(root_dirs: List[str], batch_size: int, num_workers: int = 2):
    transform = get_garbage_transforms(is_training=False)
    dataset = CombinedFilesDataset(root_dirs, split='test', transform=transform)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
