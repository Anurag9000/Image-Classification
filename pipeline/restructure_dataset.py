import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Configuration
SOURCE_ROOT = r"d:\Done,Toreview\Image Classification\data\Dataset_Final"
DEST_ROOT = r"d:\Done,Toreview\Image Classification\data\Dataset_Restructured"

# Source folders to scan
SOURCE_FOLDERS = ['organic', 'other', 'paper', 'plastic']

# Target Classes
CLASSES = [
    'battery',
    'biological',
    'cardboard',
    'clothes',
    'e-waste',
    'glass',
    'metal',
    'paper',
    'plastic',
    'trash'
]

# Keyword Rules (ORDER MATTERS: Top priority first)
# Format: (keyword, target_class)
KEYWORD_RULES = [
    ('battery', 'battery'),
    ('batteries', 'battery'),
    ('accumulators', 'battery'),
    ('organic', 'biological'),
    ('biological', 'biological'),
    ('vegetable', 'biological'),
    ('fruit', 'biological'),
    ('food', 'biological'),
    ('cardboard', 'cardboard'),
    ('clothes', 'clothes'),
    ('cloth', 'clothes'),
    ('shoe', 'clothes'),
    ('jeans', 'clothes'),
    ('shirt', 'clothes'),
    ('dress', 'clothes'),
    ('jacket', 'clothes'),
    ('coat', 'clothes'),
    ('pant', 'clothes'),
    ('textile', 'clothes'),
    ('fabric', 'clothes'),
    ('e-waste', 'e-waste'),
    ('electronic', 'e-waste'),
    ('mobile', 'e-waste'),
    ('cell', 'e-waste'),
    ('phone', 'e-waste'),
    ('computer', 'e-waste'),
    ('laptop', 'e-waste'),
    ('pc', 'e-waste'),
    ('monitor', 'e-waste'),
    ('keyboard', 'e-waste'),
    ('mouse', 'e-waste'),
    ('motherboard', 'e-waste'),
    ('circuit', 'e-waste'),
    ('chip', 'e-waste'),
    ('cable', 'e-waste'),
    ('wire', 'e-waste'),
    ('hard disk', 'e-waste'),
    ('hard drive', 'e-waste'),
    ('printer', 'e-waste'),
    ('tablet', 'e-waste'),
    ('glass', 'glass'),
    ('metal', 'metal'),
    ('alum', 'metal'),
    ('steel', 'metal'),
    ('copper', 'metal'),
    ('iron', 'metal'),
    ('tin', 'metal'),
    ('can', 'metal'),
    ('paper', 'paper'),
    ('plastic', 'plastic'),
    ('pet', 'plastic'),
    ('trash', 'trash'),
    ('garbage', 'trash'),
    ('rubbish', 'trash'),
]

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_target_class(filename, source_folder):
    filename_lower = filename.lower()
    
    # 1. Check Keywords
    for keyword, target in KEYWORD_RULES:
        if keyword in filename_lower:
            return target
            
    # 2. Contextual Fallback (if no trademark keyword found)
    if source_folder == 'paper':
        return 'paper'
    elif source_folder == 'plastic':
        return 'plastic'
    elif source_folder == 'organic':
        return 'biological'
    elif source_folder == 'other':
        return 'trash' # Safe default for ambiguous mixed items
        
    return 'trash' # Updates fallthrough

def main():
    print(f"Starting dataset restructuring...")
    print(f"Source: {SOURCE_ROOT}")
    print(f"Destination: {DEST_ROOT}")

    if os.path.exists(DEST_ROOT):
        print(f"Cleaning existing destination: {DEST_ROOT}")
        shutil.rmtree(DEST_ROOT)
    
    ensure_dir(DEST_ROOT)
    for cls in CLASSES:
        ensure_dir(os.path.join(DEST_ROOT, cls))
        
    stats = {cls: 0 for cls in CLASSES}
    total_moved = 0
    errors = 0
    
    for source_folder in SOURCE_FOLDERS:
        src_path = os.path.join(SOURCE_ROOT, source_folder)
        if not os.path.exists(src_path):
            print(f"Warning: Source folder not found: {src_path}")
            continue
            
        print(f"Processing folder: {source_folder}...")
        files = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]
        
        for filename in tqdm(files, desc=source_folder):
            try:
                target_class = get_target_class(filename, source_folder)
                
                src_file = os.path.join(src_path, filename)
                dest_file = os.path.join(DEST_ROOT, target_class, filename)
                
                # Handle duplicate filenames if simple copy
                if os.path.exists(dest_file):
                    base, ext = os.path.splitext(filename)
                    dest_file = os.path.join(DEST_ROOT, target_class, f"{base}_{source_folder}{ext}")
                
                shutil.copy2(src_file, dest_file)
                
                stats[target_class] += 1
                total_moved += 1
                
            except Exception as e:
                print(f"Error moving {filename}: {e}")
                errors += 1

    print("\n" + "="*30)
    print("Restructuring Complete!")
    print("="*30)
    print(f"Total files moved: {total_moved}")
    print(f"Errors: {errors}")
    print("\nClass Distribution:")
    for cls, count in stats.items():
        print(f"  {cls:<12}: {count}")

if __name__ == "__main__":
    main()
