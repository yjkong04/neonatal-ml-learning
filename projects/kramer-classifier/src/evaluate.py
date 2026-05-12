"""Evaluation metrics, confusion matrix, and calibration for the Kramer classifier."""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch.utils.data import DataLoader


def evaluate(model: nn.Module, test_loader: DataLoader, class_names: list) -> dict:
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = (all_preds == all_labels).mean()
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix — Kramer Classifier")
    plt.tight_layout()
    plt.savefig('../results/confusion_matrix.png', dpi=150)
    plt.show()

    return {'accuracy': acc, 'preds': all_preds, 'labels': all_labels, 'probs': all_probs}
