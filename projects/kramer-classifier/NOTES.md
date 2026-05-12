# Kramer Classifier — Notes and Data Setup

## Dataset

**ISIC skin lesion dataset (Option B)** — transfer learning on dermoscopy imagery.
Same technique as the production `production-jaundice-classifier` (pretrained backbone, fine-tuned classifier).
The training pipeline is identical; the dataset is adult skin lesions, not neonatal jaundice.

## Data download instructions

Place the ISIC dataset inside `data/` using the ImageFolder layout:

```
data/
├── train/
│   ├── class_a/
│   └── class_b/
├── val/
│   ├── class_a/
│   └── class_b/
└── test/
    ├── class_a/
    └── class_b/
```

The `data/` folder is gitignored — never commit images.
