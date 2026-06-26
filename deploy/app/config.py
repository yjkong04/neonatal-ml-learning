"""Constants for the Kramer deploy demo.

Values here are copied from projects/kramer-classifier (model.py, data.py,
notebooks/05-temperature-scaling.ipynb) rather than imported, so this
directory has no path dependency back into the training project.
"""

import os
from pathlib import Path

# Alphabetical order — this is what torchvision.datasets.ImageFolder assigns
# during training, and the checkpoint's final layer was trained against it.
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Full HAM10000 lesion names, for display only — does not affect inference.
CLASS_DISPLAY_NAMES = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}

# Learned on the val set in notebooks/05-temperature-scaling.ipynb.
# T < 1 because the class-weighted model (Run 2) is underconfident.
TEMPERATURE = 0.8363

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", str(Path(__file__).resolve().parent.parent / "model" / "best_model.pt"))
)

DISCLAIMER = (
    "Demo only — not a clinical Kramer-zone classifier. Trained on public HAM10000 "
    "dermoscopy images as a stand-in for neonatal jaundice imagery, which is not "
    "publicly available. Predicted classes are HAM10000 lesion types, not Kramer "
    "zones. See the project README for full limitations."
)
