
import torch
import cv2
import numpy as np
import os
from pipeline.train_arcface import ArcFaceConfig, ArcFaceTrainer
from pipeline.files_dataset import create_data_loader
import torch.nn.functional as F

def predict_single_image(image_path):
    print(f"--- Single Image Inference ---")
    print(f"Image: {image_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    snapshot_dir = "./snapshots_tiny"
    
    # Init Model (Same Config)
    cfg = ArcFaceConfig(
        num_classes=4,
        backbone={"cnn_model": "mobilenetv3_large_100", "pretrained": True, "vit_model": None, "use_cbam": False},
        image_size=224
    )
    trainer = ArcFaceTrainer(None, None, cfg)
    
    # Load Weights (Correct Final Files)
    backbone_path = os.path.join(snapshot_dir, "backbone_final.pth")
    head_path = os.path.join(snapshot_dir, "head", "head_final.pth")
    
    if os.path.exists(backbone_path) and os.path.exists(head_path):
        trainer.backbone.load_state_dict(torch.load(backbone_path, map_location=device))
        trainer.head.load_state_dict(torch.load(head_path, map_location=device))
        print("Model loaded successfully.")
    else:
        print("Error: Final model weights not found.")
        return

    trainer.backbone.eval()
    trainer.head.eval()
    
    # Load and Preprocess Image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    
    # Normalize (Standard ImageNet mean/std)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    
    # To Tensor (C, H, W)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        features = trainer.backbone(img_tensor)
        
        # 1. Classification Logits (Head)
        logits = trainer.head(features, labels=None)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        
        # 2. Confidence (Feature Norm)
        norm = torch.norm(features, dim=1).item()
        
    classes = ['organic', 'other', 'paper', 'plastic'] # Standard mapping (check json to be sure if numeric id needed)
    # JSON said: {'organic': 0, 'other': 1, 'paper': 2, 'plastic': 3} (from previous logs)
    
    print(f"\nPrediction for {os.path.basename(image_path)}:")
    print(f"Feature Norm (Quality/Confidence): {norm:.4f}")
    print("-" * 30)
    for i, cls in enumerate(classes):
        print(f"{cls}: {probs[i]*100:.2f}%")
    print("-" * 30)
    print(f"Result: {classes[np.argmax(probs)]}")

if __name__ == "__main__":
    # Test Image
    img_path = "data/Dataset_Final/organic/biological_631.jpg"
    predict_single_image(img_path)
