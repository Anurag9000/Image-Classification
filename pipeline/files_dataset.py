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
from .augmentations import get_advanced_transforms
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
            LOGGER.error(f"Image not found or corrupt: {img_path}. Returning dummy image.")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
             # Standard normalization must be applied even in fallback
             image = A.Compose([
                 A.Resize(224, 224), 
                 A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                 ToTensorV2()
             ])(image=image)['image']

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
            LOGGER.error(f"Image not found or corrupt: {img_path}. Returning dummy image.")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Standard normalization must be applied even in fallback
            image = A.Compose([
                A.Resize(224, 224), 
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])(image=image)['image']

        return image, label



def create_data_loader(
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
    train_transform = get_advanced_transforms(is_training=augment_online)
    val_transform = get_advanced_transforms(is_training=False)
    
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
        
        LOGGER.info(f"Splitting dataset: Train={train_len} (est), Val={val_len} (est), Test={test_len} (est) [Stratified by Class]")
        
        # Stratified Split Logic
        from collections import defaultdict
        class_groups = defaultdict(list)
        for item in full_metadata:
            class_groups[item['label']].append(item)

        train_meta, val_meta, test_meta = [], [], []
        
        # Generator for reproducibility
        g = torch.Generator().manual_seed(42)

        for label, items in class_groups.items():
            n = len(items)
            n_val = int(n * val_split)
            n_test = int(n * test_split)
            n_train = n - n_val - n_test
            
            # Ensure at least 1 training sample if possible
            if n_train == 0 and n > 0:
                n_train = 1
                if n_val > 0: n_val -= 1
                elif n_test > 0: n_test -= 1
            
            # Shuffle indices for this class
            indices = torch.randperm(n, generator=g).tolist()
            
            # Slice
            idx_train = indices[:n_train]
            idx_val = indices[n_train : n_train + n_val]
            idx_test = indices[n_train + n_val :]
            
            for i in idx_train: train_meta.append(items[i])
            for i in idx_val: val_meta.append(items[i])
            for i in idx_test: test_meta.append(items[i])

        # Shuffle again to mix classes in the final lists (optional but good for batches)
        # Actually DataLoader shuffles, so not strictly needed, but good for sanity
        
        # Optimization: Explicitly clear the large raw metadata list
        del full_metadata
        import gc
        gc.collect()

        train_dataset = JsonDataset(train_meta, data_root, transform=train_transform)
        val_dataset = JsonDataset(val_meta, data_root, transform=val_transform)
        test_dataset = JsonDataset(test_meta, data_root, transform=val_transform)
        
        LOGGER.info(f"Created RAM-Optimized Datasets: Train({len(train_dataset)}), Val({len(val_dataset)}), Test({len(test_dataset)})")
        
    else:
        # CSV/Folder Mode
        # Check if 'valid' exists in the first root as a heuristic
        has_valid_folder = False
        if root_dirs and len(root_dirs) > 0 and os.path.exists(os.path.join(root_dirs[0], 'valid')):
             has_valid_folder = True
        
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
            # WARN: This path assumes CombinedFilesDataset handles the split? No, it loads 'train' folder.
            # If no 'valid' folder exists, we might need to split 'train' manually?
            # But CombinedFilesDataset implementation above loads explicit 'split'.
            LOGGER.info("No 'valid' folder found, using 'train' folder for everything (this might be wrong if you wanted validation).")
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
        num_workers=num_workers, pin_memory=True,
        drop_last=True # Better for stable batch sizes during training
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

def create_test_loader(root_dirs: List[str], batch_size: int, num_workers: int = 2, json_path: str = None):
    transform = get_advanced_transforms(is_training=False)
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

