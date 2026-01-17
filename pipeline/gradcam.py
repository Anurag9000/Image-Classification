import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


class GradCAM:
    def __init__(self, model, target_layer, mode="gradcam"):
        """
        mode: "gradcam" or "gradcam++"
        """
        self.model = model
        self.target_layer = target_layer
        self.mode = mode
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)

        loss = output[:, class_idx]
        self.model.zero_grad()
        loss.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        if self.mode == "gradcam":
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)
        elif self.mode == "gradcam++":
            grads = gradients
            grads_power_2 = grads ** 2
            grads_power_3 = grads ** 3
            sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
            eps = 1e-8
            alpha = grads_power_2 / (2 * grads_power_2 + sum_activations * grads_power_3 + eps)
            weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)

        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def overlay_heatmap(self, cam, image, output_path):
        """
        image: original image tensor in shape (3, H, W)
        """
        img = image.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        overlay = heatmap + np.float32(img) / 255
        overlay = overlay / overlay.max()

        plt.figure(figsize=(6, 6))
        plt.axis('off')
        plt.imshow(overlay)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
