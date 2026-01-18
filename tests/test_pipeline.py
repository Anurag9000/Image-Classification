import os
import unittest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pipeline.backbone import BackboneConfig, HybridBackbone
from pipeline.evaluate import EvaluationConfig, Evaluator
from pipeline.inference import predict, load_model
from pipeline.gradcam import GradCAM
from pipeline.export import export_backbone_onnx, quantize_backbone_dynamic
from pipeline.losses import AdaFace

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone_cfg = BackboneConfig(
            cnn_model="resnet18",
            vit_model=None,
            fusion_dim=128
        )
        self.model = HybridBackbone(self.backbone_cfg).to(self.device).eval()
        self.head = AdaFace(embedding_size=128, num_classes=10).to(self.device).eval()
        
        # Create dummy data
        x = torch.randn(10, 3, 192, 192)
        y = torch.randint(0, 10, (10,))
        dataset = TensorDataset(x, y)
        self.dataloader = DataLoader(dataset, batch_size=2)

    def test_evaluator(self):
        eval_cfg = EvaluationConfig(tta_runs=1)
        evaluator = Evaluator(self.dataloader, num_classes=10, config=eval_cfg)
        metrics = evaluator.evaluate(self.model, self.head)
        self.assertIn("accuracy", metrics)
        self.assertIn("top_top1", metrics)

    def test_inference_predict(self):
        x = torch.randn(1, 3, 192, 192).to(self.device)
        pred, conf = predict(x, self.model, self.head)
        self.assertIsInstance(pred, int)
        self.assertIsInstance(conf, float)
        self.assertTrue(0 <= conf <= 1)

    def test_gradcam(self):
        target_layer = None
        if hasattr(self.model, "cnn_backbone"):
            cnn = self.model.cnn_backbone
            if hasattr(cnn, "stages"):
                target_layer = cnn.stages[-1][-1]
            elif hasattr(cnn, "layer4"):
                target_layer = cnn.layer4[-1]
            elif hasattr(cnn, "blocks"):
                target_layer = cnn.blocks[-1]
        
        if target_layer:
            # Wrap for GradCAM (Backbone + Head)
            class ModelWrapper(torch.nn.Module):
                def __init__(self, bb, h):
                    super().__init__()
                    self.bb = bb
                    self.h = h
                def forward(self, x):
                    return self.h(self.bb(x))
            
            wrapper = ModelWrapper(self.model, self.head)
            cam = GradCAM(wrapper, target_layer)
            x = torch.randn(1, 3, 192, 192).to(self.device)
            heatmap = cam.generate(x)
            self.assertEqual(heatmap.shape, (192, 192))
            cam.remove_hooks()

    def test_export_onnx(self):
        # Save dummy weights
        weights_path = "test_backbone.pth"
        torch.save(self.model.state_dict(), weights_path)
        
        output_onnx = "test_backbone.onnx"
        export_backbone_onnx(weights_path, output_onnx, backbone_cfg={"fusion_dim": 128, "cnn_model": "resnet18", "vit_model": None}, image_size=192)
        self.assertTrue(os.path.exists(output_onnx))
        
        # Cleanup
        if os.path.exists(weights_path): os.remove(weights_path)
        if os.path.exists(output_onnx): os.remove(output_onnx)

    def test_quantization(self):
        # Save dummy weights
        weights_path = "test_backbone_q.pth"
        torch.save(self.model.state_dict(), weights_path)
        
        output_q = "test_backbone_quantized.pth"
        quantize_backbone_dynamic(weights_path, output_q, backbone_cfg={"fusion_dim": 128, "cnn_model": "resnet18", "vit_model": None})
        self.assertTrue(os.path.exists(output_q))
        
        # Cleanup
        if os.path.exists(weights_path): os.remove(weights_path)
        if os.path.exists(output_q): os.remove(output_q)

if __name__ == "__main__":
    unittest.main()
