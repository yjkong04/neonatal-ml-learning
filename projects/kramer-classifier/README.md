# Kramer Classifier — Phase 2 Capstone

A neonatal jaundice severity classifier based on the Kramer dermal staging system.

**Status:** Not started. Planned for Week 2.

---

## Background

The Kramer scale (Kramer 1969) describes the cephalocaudal progression of jaundice in newborns through five visual zones:

- **Zone 1:** Face and neck
- **Zone 2:** Upper trunk
- **Zone 3:** Lower trunk and thighs
- **Zone 4:** Arms and lower legs
- **Zone 5:** Palms and soles

Higher zones correspond to higher serum bilirubin levels and greater clinical urgency. Visual Kramer assessment is widely used as a screening tool in resource-limited settings where transcutaneous bilirubinometers are unavailable.

## Why this project

It's a contained, well-defined image classification problem with clear clinical relevance. The technique (transfer learning from a pretrained backbone) generalizes immediately to a half-dozen other image-based clinical ML problems.

## Approach

1. Curate a public dataset of neonatal photos with Kramer zone labels (or simulate via color transformation on baseline neonatal photos when public data is sparse)
2. Fine-tune a pretrained ResNet50 / EfficientNet-B0 backbone
3. Evaluate with stratified test split, confusion matrix across zones, and calibration analysis
4. Document failure modes — particularly skin tone bias, which is the known fairness concern in skin-based ML

## Planned structure

```
kramer-classifier/
├── README.md            (this file)
├── data/                (git-ignored; download instructions in NOTES.md)
├── notebooks/
│   ├── 01-eda.ipynb
│   ├── 02-baseline-resnet.ipynb
│   └── 03-skin-tone-fairness.ipynb
├── src/
│   ├── data.py          (data loading, augmentation)
│   ├── model.py         (architecture)
│   ├── train.py         (training loop)
│   └── evaluate.py      (metrics, calibration, fairness)
├── requirements.txt
└── results/
```

## Success criteria

- Reproducible from a clean clone
- Baseline accuracy > 70% on held-out test set (zone classification)
- Clear documentation of dataset limitations
- Fairness analysis across Fitzpatrick skin tone (acknowledging that this is the entire point — a Kramer classifier that fails on darker skin tones is worse than no classifier at all)
