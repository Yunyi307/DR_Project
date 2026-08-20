"""Backbone factory.

A single entry point builds any supported architecture from ``timm`` with an
ImageNet-pretrained backbone and a fresh 5-class head. Keeping CNNs and
Transformers behind one interface is what makes the CNN-vs-ViT comparison a fair,
controlled experiment: only the ``backbone`` string changes between runs.
"""
from __future__ import annotations

import timm
import torch
from torch import nn

from .config import NUM_CLASSES

# Friendly aliases -> concrete timm model names.
BACKBONE_ALIASES = {
    "efficientnet_b4": "tf_efficientnet_b4",
    "effnet_b4": "tf_efficientnet_b4",
    "vit_b16": "vit_base_patch16_224",
    "vit_base": "vit_base_patch16_224",
    "swin_b": "swin_base_patch4_window7_224",
    "swin_base": "swin_base_patch4_window7_224",
}


def build_model(backbone: str, num_classes: int = NUM_CLASSES,
                pretrained: bool = True, drop_rate: float = 0.3) -> nn.Module:
    """Create a classifier. ``backbone`` may be an alias or a raw timm name."""
    name = BACKBONE_ALIASES.get(backbone, backbone)
    model = timm.create_model(
        name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate,
    )
    return model


def find_target_layer(model: nn.Module) -> nn.Module:
    """Return a sensible Grad-CAM target layer for common backbones.

    For CNNs this is the last conv block; for ViT/Swin we return the final
    normalisation layer (attention-rollout is a better fit for pure ViTs, but the
    last norm still yields usable CAMs via reshape transforms).
    """
    # EfficientNet / ResNet style: use the last convolutional module found.
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is not None and not hasattr(model, "blocks"):
        return last_conv
    # Transformers expose a final norm layer.
    if hasattr(model, "norm"):
        return model.norm
    if last_conv is not None:
        return last_conv
    raise ValueError("Could not infer a Grad-CAM target layer for this model.")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
