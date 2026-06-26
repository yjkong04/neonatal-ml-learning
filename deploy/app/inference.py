"""Model loading and calibrated inference for the Kramer deploy demo.

Copied from projects/kramer-classifier/src/model.py and src/data.py rather than
imported — this directory has no path dependency back into the training project.
Only the resnet50 branch is needed here since that's the architecture behind
results/v2/best_model.pt (the checkpoint this demo serves).
"""

from io import BytesIO

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from . import config

_DEVICE = torch.device("cpu")  # CPU inference only — no GPU on the deploy target.


def _build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)  # weights come from the checkpoint, not ImageNet
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model() -> nn.Module:
    model = _build_model(num_classes=len(config.CLASS_NAMES))
    state_dict = torch.load(config.MODEL_PATH, map_location=_DEVICE)
    model.load_state_dict(state_dict)
    model.to(_DEVICE)
    model.eval()
    return model


_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
])


def predict(model: nn.Module, image_bytes: bytes) -> dict:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = _TRANSFORM(image).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        # Temperature scaling: divide logits by T before softmax. Changes confidence,
        # not the argmax prediction — see notebooks/05-temperature-scaling.ipynb.
        probs = torch.softmax(logits / config.TEMPERATURE, dim=1).squeeze(0)

    probs = probs.tolist()
    ranked = sorted(zip(config.CLASS_NAMES, probs), key=lambda x: x[1], reverse=True)
    top_class, top_prob = ranked[0]

    return {
        "predicted_class": top_class,
        "predicted_class_display": config.CLASS_DISPLAY_NAMES[top_class],
        "calibrated_probability": round(top_prob, 4),
        "all_probabilities": {
            name: round(p, 4) for name, p in ranked
        },
    }
