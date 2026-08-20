"""Minimal, self-contained Grad-CAM for CNN backbones.

Grad-CAM (Selvaraju et al., 2017) highlights the image regions most responsible
for a class prediction by weighting the last convolutional feature maps with the
gradient of the target logit. We implement it directly (forward/backward hooks)
rather than relying on a library, both for transparency and because it is a core
interpretability deliverable of the project.

Works out-of-the-box for convolutional backbones (EfficientNet, ResNet). Pure
ViTs need attention rollout instead - out of scope for this module.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None
                 ) -> tuple[np.ndarray, int]:
        """Return (heatmap in [0,1] at feature resolution, predicted/target class)."""
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        self.model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()

        # weights = global-average-pooled gradients; cam = ReLU(sum_k w_k * A_k)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)      # (1,C,1,1)
        cam = (weights * self.activations).sum(dim=1).squeeze(0)     # (H,W)
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), class_idx


def overlay_cam(rgb_image: np.ndarray, cam: np.ndarray, alpha: float = 0.4
                ) -> np.ndarray:
    """Blend a heatmap over an RGB uint8 image, resizing the CAM to match."""
    h, w = rgb_image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.uint8(alpha * heatmap + (1 - alpha) * rgb_image)
