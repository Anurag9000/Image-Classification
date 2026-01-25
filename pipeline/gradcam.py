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
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        # Ensure gradients are enabled for input if needed (though usually we need gradients w.r.t weights/activations)
        # But GradCAM often runs on 'requires_grad=True' input for some variants, or just relies on hooks.
        # Hooks work even if input doesn't require grad, provided model has params requiring grad?
        # No, for inference, params don't require grad.
        # But we need backward().
        # So we must set requires_grad=True on activation or input.
        # The hook captures 'grad_output', which requires backward() to propagate there.
        # If no leaf variable requires grad, backward() fails?
        # Yes. So we set input_tensor.requires_grad = True.
        input_tensor.requires_grad_(True)
        
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)

        loss = output[:, class_idx]
        self.model.zero_grad()
        loss.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        # Handle ViT 3D output (B, N, C) -> (B, C, H, W)
        if activations.dim() == 3:
             b, n, c = activations.shape
             h = w = int(np.sqrt(n))
             # If n is not a perfect square (contains CLS token), skip CLS
             if h * w != n:
                 activations = activations[:, 1:]
                 gradients = gradients[:, 1:]
                 n = n - 1
                 h = w = int(np.sqrt(n))
             
             activations = activations.transpose(1, 2).reshape(b, c, h, w)
             gradients = gradients.transpose(1, 2).reshape(b, c, h, w)

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
        # Batch-aware squeezing
        cam = cam.squeeze(1).cpu().numpy() # Resulting (B, H, W)
        
        # Min-max norm per image in batch
        for i in range(cam.shape[0]):
            c_min, c_max = cam[i].min(), cam[i].max()
            cam[i] = (cam[i] - c_min) / (c_max - c_min + 1e-8)
        
        return cam[0] if cam.shape[0] == 1 else cam 

    def overlay_heatmap(self, cam, image, output_path):
        """
        image: original image tensor in shape (3, H, W). Should be un-normalized if possible.
        """
        img = image.permute(1, 2, 0).cpu().numpy()
        # Denormalize simple min-max to 0-1 range for visualization
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
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
