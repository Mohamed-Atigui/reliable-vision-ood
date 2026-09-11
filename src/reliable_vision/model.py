from __future__ import annotations


def build_resnet18(num_classes: int = 10):
    """ResNet-18 adapted to 32x32 images (3x3 stem, no max-pooling)."""
    import torch.nn as nn
    from torchvision.models import resnet18

    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
