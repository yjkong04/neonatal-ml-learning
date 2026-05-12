"""Model architecture for the Kramer classifier."""

import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, backbone: str = 'resnet50', freeze_backbone: bool = True) -> nn.Module:
    """
    Fine-tune a pretrained backbone for Kramer zone classification.

    freeze_backbone=True: only the final classifier layer trains (faster, good starting point).
    freeze_backbone=False: all layers train (slower, use after initial convergence).
    """
    if backbone == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif backbone == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif backbone == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown backbone: {backbone}. Choose resnet18, resnet50, or efficientnet_b0.")

    return model
