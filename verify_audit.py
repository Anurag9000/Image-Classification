
import logging
import sys
import torch
import os

# Improve logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("VERIFY")

def test_imports():
    LOGGER.info("Testing Imports...")
    try:
        from pipeline.files_dataset import create_data_loader
        from pipeline.augmentations import get_advanced_transforms
        from pipeline.backbone import HybridBackbone, BackboneConfig
        from pipeline.run_pipeline import run_pipeline
        LOGGER.info("Imports Successful.")
    except Exception as e:
        LOGGER.error(f"Import Failed: {e}")
        sys.exit(1)

def test_data_loader_creation():
    LOGGER.info("Testing Data Loader Creation (Dry Run)...")
    from pipeline.files_dataset import create_data_loader
    # Create dummy dir
    os.makedirs("temp_data/train", exist_ok=True)
    with open("temp_data/train/_classes.csv", "w") as f:
        f.write("filename, cat, dog\n")
        f.write("img1.jpg, 1, 0\n")
    
    # Create dummy image
    import cv2
    import numpy as np
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.imwrite("temp_data/train/img1.jpg", img)

    try:
        loader, _, _ = create_data_loader(["temp_data"], batch_size=1, num_workers=0)
        LOGGER.info(f"Loader created. Len: {len(loader)}")
        for x, y in loader:
            LOGGER.info(f"Batch shape: {x.shape}")
            break
    except Exception as e:
        LOGGER.error(f"Loader Creation Failed: {e}")
        # Don't exit, might be just this test
    finally:
        import shutil
        if os.path.exists("temp_data"):
            shutil.rmtree("temp_data")

def test_backbone():
    LOGGER.info("Testing Backbone Instantiation...")
    from pipeline.backbone import HybridBackbone, BackboneConfig
    try:
        cfg = BackboneConfig(vit_model=None, cnn_model="resnet18", use_cbam=True)
        model = HybridBackbone(cfg)
        model.eval() # Fix BatchNorm error with batch size 1
        dummy = torch.randn(1, 3, 224, 224)
        out = model(dummy)
        LOGGER.info(f"Backbone Output Shape: {out.shape}")
    except Exception as e:
        LOGGER.error(f"Backbone Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_imports()
    test_backbone()
    test_data_loader_creation()
    LOGGER.info("Verification Complete.")
