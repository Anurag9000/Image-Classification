from __future__ import annotations

import argparse
import csv
import os
from typing import Optional, Tuple

import torch
import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image

import yaml

from .backbone import BackboneConfig, HybridBackbone
from .gradcam import GradCAM
from .losses import AdaFace


def load_model(
    backbone_path: str,
    head_path: Optional[str] = None,
    num_classes: int = 100,
    device: Optional[torch.device] = None,
    backbone_cfg: Optional[dict] = None,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = BackboneConfig(**(backbone_cfg or {}))
    model = HybridBackbone(config).to(device).eval()
    head = AdaFace(config.fusion_dim, num_classes).to(device).eval()

    backbone_state = torch.load(backbone_path, map_location=device)
    if isinstance(backbone_state, dict):
        if "model_state_dict" in backbone_state:
            backbone_state = backbone_state["model_state_dict"]
        elif "state_dict" in backbone_state:
            backbone_state = backbone_state["state_dict"]
    if isinstance(backbone_state, dict) and "backbone" in backbone_state:
        model.load_state_dict(backbone_state["backbone"], strict=False)
        if head_path is None and "head" in backbone_state:
            head.load_state_dict(backbone_state["head"], strict=False)
    else:
        model.load_state_dict(backbone_state, strict=False)

    if head_path:
        head_state = torch.load(head_path, map_location=device)
        if isinstance(head_state, dict) and "state_dict" in head_state:
            head_state = head_state["state_dict"]
        head.load_state_dict(head_state, strict=False)

    return model, head


def get_image_tensor(image_path: str, device: Optional[torch.device] = None, img_size: int = 224) -> torch.Tensor:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use EXACTLY the same transform pipeline as validation
    from .files_dataset import get_garbage_transforms
    transform = get_garbage_transforms(is_training=False, img_size=img_size)
    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Albumentations expects 'image' kwarg and returns dict
    augmented = transform(image=image)
    tensor = augmented['image']
    
    return tensor.unsqueeze(0).to(device)


def predict(image_tensor: torch.Tensor, model, head) -> Tuple[int, float]:
    with torch.no_grad():
        features = model(image_tensor)
        logits = head(features, labels=None)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = torch.max(probs).item()
    return pred, conf


def run_inference(
    image_dir: str,
    backbone_path: str,
    head_path: Optional[str] = None,
    output_csv: str = "./inference_results.csv",
    gradcam_dir: Optional[str] = None,
    backbone_cfg: Optional[dict] = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, head = load_model(backbone_path, head_path=head_path, device=device, backbone_cfg=backbone_cfg)
    image_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if gradcam_dir:
        os.makedirs(gradcam_dir, exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "prediction", "confidence"])

        for image_path in image_paths:
            tensor = get_image_tensor(image_path, device=device)
            pred, conf = predict(tensor, model, head)

            writer.writerow([os.path.basename(image_path), pred, f"{conf:.4f}"])
            print(f"{os.path.basename(image_path)} => class {pred}, conf {conf:.4f}")

            if gradcam_dir:
                target_layer = None
                cnn = model.cnn_backbone
                if hasattr(cnn, "stages"): # ConvNeXt
                    target_layer = cnn.stages[-1][-1]
                elif hasattr(cnn, "layer4"): # ResNet
                    target_layer = cnn.layer4[-1]
                elif hasattr(cnn, "blocks"): # Some ViTs/other CNNs
                    target_layer = cnn.blocks[-1]
                
                if target_layer:
                    # GradCAM requires logits, but 'model' only outputs features.
                    # We must wrap them.
                    class ModelWrapper(torch.nn.Module):
                        def __init__(self, bb, h):
                            super().__init__()
                            self.bb = bb
                            self.h = h
                        def forward(self, x):
                            return self.h(self.bb(x))
                    
                    wrapper = ModelWrapper(model, head)
                    # We need to point GradCAM to the backbone INSIDE the wrapper
                    # GradCAM registers hooks on 'target_layer'. 
                    # target_layer is already an object reference to a layer in 'model'.
                    # Even if 'model' is inside wrapper, target_layer object is same.
                    # So hooks will work.
                    
                    cam = GradCAM(wrapper, target_layer, mode="gradcam++")
                    heatmap = cam.generate(tensor)
                    cam.overlay_heatmap(heatmap, tensor[0], os.path.join(gradcam_dir, os.path.basename(image_path)))
                    cam.remove_hooks()
                else:
                    print(f"Warning: Could not identify target layer for GradCAM on {type(cnn)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a directory of images.")
    parser.add_argument("--image_dir", required=True, help="Directory containing images.")
    parser.add_argument("--backbone_path", required=True, help="Path to backbone weights (.pth).")
    parser.add_argument("--head_path", help="Optional path to head weights (.pth).")
    parser.add_argument("--output_csv", default="./logs/inference_results.csv", help="Path to output CSV file.")
    parser.add_argument("--gradcam_dir", help="Optional directory to dump GradCAM overlays.")
    parser.add_argument("--backbone_cfg", help="Optional YAML/JSON string or file path defining BackboneConfig.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    backbone_cfg: Optional[dict] = None
    if args.backbone_cfg:
        if os.path.exists(args.backbone_cfg):
            with open(args.backbone_cfg, "r", encoding="utf-8") as f:
                backbone_cfg = yaml.safe_load(f)
        else:
            backbone_cfg = yaml.safe_load(args.backbone_cfg)
    run_inference(
        image_dir=args.image_dir,
        backbone_path=args.backbone_path,
        head_path=args.head_path,
        output_csv=args.output_csv,
        gradcam_dir=args.gradcam_dir,
        backbone_cfg=backbone_cfg,
    )

