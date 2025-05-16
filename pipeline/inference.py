import os
import torch
import torchvision.transforms as T
from PIL import Image
import csv
from backbone import HybridBackbone
from losses import AdaFace
from gradcam import GradCAM


def load_model(weight_path, num_classes=100):
    model = HybridBackbone(use_cbam=True).eval().cuda()
    head = AdaFace(1024, num_classes).eval().cuda()

    state_dict = torch.load(weight_path, map_location="cuda")
    model.load_state_dict(state_dict, strict=False)

    return model, head


def get_image_tensor(image_path):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict(image_tensor, model, head):
    image_tensor = image_tensor.cuda()
    with torch.no_grad():
        features = model(image_tensor)
        logits = head(features, torch.zeros(1, dtype=torch.long).cuda())
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = torch.max(probs).item()
    return pred, conf


def run_inference(image_dir, weight_path, output_csv="./inference_results.csv", gradcam_dir=None):
    model, head = load_model(weight_path)
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg'))]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if gradcam_dir:
        os.makedirs(gradcam_dir, exist_ok=True)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "prediction", "confidence"])

        for image_path in image_paths:
            tensor = get_image_tensor(image_path)
            pred, conf = predict(tensor, model, head)

            writer.writerow([os.path.basename(image_path), pred, f"{conf:.4f}"])
            print(f"{os.path.basename(image_path)} => class {pred}, conf {conf:.4f}")

            if gradcam_dir:
                cam = GradCAM(model, model.cnn_backbone.stages[-1][-1], mode="gradcam++")
                heatmap = cam.generate(tensor.cuda())
                cam.overlay_heatmap(heatmap, tensor[0], os.path.join(gradcam_dir, f"{os.path.basename(image_path)}"))
                cam.remove_hooks()


if __name__ == "__main__":
    run_inference(
        image_dir="./inference_images/",
        weight_path="best_model.pth",
        output_csv="./logs/inference_results.csv",
        gradcam_dir="./gradcam_outputs/"
    )
