import os
import cv2
import torch
import json
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from typing import List, Optional, Callable
from .augmentations import get_garbage_transforms
import logging

LOGGER = logging.getLogger(__name__)

class JsonDataset(Dataset):
    """
    Dataset that reads from a JSON metadata file.
    Expected JSON format: List of dicts with 'file_path' and 'label'.
    """
    def __init__(self, json_path: str | List[dict], root_dir: str, transform: Optional[A.Compose] = None):
        self.root_dir = root_dir
        self.transform = transform
        
        if isinstance(json_path, list):
            self.metadata = json_path
        else:
            with open(json_path, 'r') as f:
                self.metadata = json.load(f)
            
        # Create class mapping
        self.classes = sorted(list(set(item['label'] for item in self.metadata)))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        LOGGER.info(f"JsonDataset loaded {len(self.metadata)} images. Classes: {self.class_to_idx}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        # Allow absolute or relative paths
        if os.path.isabs(item['file_path']):
             img_path = item['file_path']
        else:
             img_path = os.path.join(self.root_dir, item['file_path'])
             
        label_str = item['label']
        label = self.class_to_idx[label_str]
        
        image = cv2.imread(img_path)
        if image is None:
            # Placeholder or skip? For training, raising error is better to catch issues.
            # But might be annoying if one image is corrupt.
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
             image = A.Compose([A.Resize(224, 224), ToTensorV2()])(image=image)['image']

        return image, label

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
                # print(f"Warning: {csv_path} not found. Skipping...")
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
                    # STRICT MODE: Raise error if classes differ between folders (e.g. train has 5 classes, val has 6)
                    raise ValueError(f"Inconsistent class names in {csv_path}. Expected {self.class_names}, got {label_cols}")
            
            # Iterate rows
            for _, row in df.iterrows():
                fname = row['filename'].strip()
                img_path = os.path.join(split_dir, fname)
                
                # Get label index (argmax of one-hot)
                # Convert row part to float array first
                labels_one_hot = row[self.class_names].values.astype(float)
                label_idx = np.argmax(labels_one_hot)
                
                self.samples.append((img_path, int(label_idx)))

        LOGGER.info(f"Loaded {len(self.samples)} samples for split '{split}' from {len(root_dirs)} roots.")

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



def create_garbage_loader(
    root_dirs: List[str], 
    batch_size: int, 
    num_workers: int = 2, 
    val_split: float = 0.0,
    test_split: float = 0.0,
    json_path: str = None,
    augment_online: bool = True
):
    """
    Creates train/val/test loaders.
    If json_path is provided, uses JsonDataset (single large dataset, split optional).
    Otherwise uses CombinedFilesDataset (folder structure with CSVs).
    """
    # If augment_online is False, use VALIDATION transforms (Resize+Norm) for training too.
    train_transform = get_garbage_transforms(is_training=augment_online)
    val_transform = get_garbage_transforms(is_training=False)
    
    test_loader = None
    
    if json_path:
        # JSON Mode
        LOGGER.info(f"Creating loader from JSON: {json_path}")
        # root_dirs[0] assumed to be data root if provided
        data_root = root_dirs[0] if root_dirs else "./data"
        
        # OPTIMIZATION: Load JSON once to save RAM (avoid 3x copies)
        with open(json_path, 'r') as f:
            full_metadata = json.load(f)
            
        full_len = len(full_metadata)
        LOGGER.info(f"Total images in JSON: {full_len}")

        total_len = len(full_metadata)
        val_len = int(total_len * val_split)
        test_len = int(total_len * test_split)
        train_len = total_len - val_len - test_len
        
        lengths = [train_len, val_len, test_len]
        # Ensure splits aren't empty if requested
        if test_split > 0 and test_len == 0: lengths = [train_len - 1, val_len, 1] 
        
        LOGGER.info(f"Splitting dataset: Train={lengths[0]}, Val={lengths[1]}, Test={lengths[2]}")
        
        # 1. Get indices
        indices = torch.randperm(total_len, generator=torch.Generator().manual_seed(42)).tolist()
        train_idx = indices[:train_len]
        val_idx = indices[train_len : train_len + val_len]
        test_idx = indices[train_len + val_len :]
        
        # 2. Extract Sub-Lists -> New JsonDataset instances
        # This is much cleaner and RAM efficient than Subset(JsonDataset()) which holds the whole list
        train_meta = [full_metadata[i] for i in train_idx]
        val_meta = [full_metadata[i] for i in val_idx]
        test_meta = [full_metadata[i] for i in test_idx]
        
        full_metadata = None # Free gigantic list from memory immediately

        train_dataset = JsonDataset(train_meta, data_root, transform=train_transform)
        val_dataset = JsonDataset(val_meta, data_root, transform=val_transform)
        test_dataset = JsonDataset(test_meta, data_root, transform=val_transform)
        
        LOGGER.info(f"Created RAM-Optimized Datasets: Train({len(train_dataset)}), Val({len(val_dataset)}), Test({len(test_dataset)})")
        
    else:
        # CSV/Folder Mode
        # Check if 'valid' exists in the first root as a heuristic
        has_valid_folder = os.path.exists(os.path.join(root_dirs[0], 'valid'))
        
        if has_valid_folder:
            train_dataset = CombinedFilesDataset(root_dirs, split='train', transform=train_transform)
            val_dataset = CombinedFilesDataset(root_dirs, split='valid', transform=val_transform)
            # Check for test folder
            if os.path.exists(os.path.join(root_dirs[0], 'test')):
                test_dataset = CombinedFilesDataset(root_dirs, split='test', transform=val_transform)
            else:
                test_dataset = None
        else:
            # Fallback: Load 'train' and split it
            full_dataset = CombinedFilesDataset(root_dirs, split='train', transform=train_transform)
            train_dataset = full_dataset
            val_dataset = None 
            test_dataset = None

    # Create WeightedRandomSampler for balanced training
    # 1. Gather all labels from the training dataset
    if json_path:
        # Optimization: train_dataset is now a standalone JsonDataset, not a Subset
        train_labels = [item['label'] for item in train_dataset.metadata]
        # Map string labels to indices
        train_labels_idx = [train_dataset.class_to_idx[l] for l in train_labels]
    else:
        # CombinedFilesDataset (list of tuples (path, label))
        train_labels_idx = [item[1] for item in train_dataset.samples]

    # 2. Compute Class Weights
    class_counts = np.bincount(train_labels_idx)
    # Avoid div by zero if a class is missing (unlikely but safe)
    class_weights = 1.0 / np.maximum(class_counts, 1.0)
    
    # 3. Assign weight to each sample
    sample_weights = [class_weights[l] for l in train_labels_idx]
    
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    LOGGER.info(f"WeightedRandomSampler enabled. Class counts: {class_counts}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, 
        sampler=sampler, # Replaces shuffle=True
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True
        )
        
    if test_dataset and len(test_dataset) > 0:
         test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True
        )

    return train_loader, val_loader, test_loader

def create_garbage_test_loader(root_dirs: List[str], batch_size: int, num_workers: int = 2, json_path: str = None):
    transform = get_garbage_transforms(is_training=False)
    if json_path:
        data_root = root_dirs[0] if root_dirs else "./data"
        # For test, we might need a separate test json or just use same dataset class?
        # Assuming test json or splitting manually. For now, just load it.
        dataset = JsonDataset(json_path, data_root, transform=transform)
    else:
        dataset = CombinedFilesDataset(root_dirs, split='test', transform=transform)
        
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
