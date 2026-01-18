import torch
import cv2
import numpy as np
from pipeline.inference import predict, load_model
from pipeline.backbone import BackboneConfig

def main():
    # Configuration (Ensure this matches your trained model)
    bb_cfg = BackboneConfig(
        cnn_model="resnet50",
        fusion_dim=512,
        use_cbam=True
    )
    
    weights_path = "./snapshots/backbone_best.pth"
    head_path = "./snapshots/head/head_best.pth"
    
    # Load model
    try:
        model, head = load_model(weights_path, head_path=head_path, bb_cfg=bb_cfg)
        model.eval()
        head.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load image
    img_path = "path/to/your/test_image.jpg"
    if not torch.cuda.is_available():
        print("Running on CPU.")

    # Dummy inference for demonstration
    dummy_img = torch.randn(1, 3, 224, 224)
    pred_idx, confidence = predict(dummy_img, model, head)
    print(f"Prediction: {pred_idx}, Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()
