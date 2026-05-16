# Kramer Classifier — Phase 2 Capstone

A neonatal jaundice severity classifier based on the Kramer dermal staging system.

**Status:** Learning prototype — in progress. Demo target 2026-05-22.

---

## Honest framing (read this first)

This is a learning project, not a deployable model. Public neonatal jaundice imagery is sparse, so I'm training on the HAM10000 skin lesion dataset as a stand-in. The model architecture (pretrained ResNet50 with a fine-tuned classifier head), the training loop, and the evaluation pipeline are the same techniques the production `production-jaundice-classifier` will use. The main difference is the training data — adult dermoscopy images instead of neonatal jaundice photos.

**What this project demonstrates:** end-to-end image classification, transfer learning, handling severe class imbalance, evaluation beyond raw accuracy.

**What this project does NOT demonstrate:** validation on real neonatal jaundice, clinical readiness, or skin-tone fairness on the actual target population.

## Background — Kramer scale

The Kramer scale (Kramer 1969) describes the cephalocaudal progression of jaundice in newborns through five visual zones:

- **Zone 1:** Face and neck
- **Zone 2:** Upper trunk
- **Zone 3:** Lower trunk and thighs
- **Zone 4:** Arms and lower legs
- **Zone 5:** Palms and soles

Higher zones correspond to higher serum bilirubin levels and greater clinical urgency. Visual Kramer assessment is widely used as a screening tool in resource-limited settings where transcutaneous bilirubinometers are unavailable.

## Dataset — HAM10000

Harvard Dataverse, ~10k dermoscopy images across 7 lesion classes. After a stratified 70/15/15 split: 5229 train / 1120 val / 1121 test. Severely imbalanced — `nv` (melanocytic nevi) is ~72% of training data, `df` (dermatofibroma) is <2%. That imbalance is useful here: it mirrors the dynamic `production-jaundice-classifier` will face, where most neonatal images will be healthy/low-zone and high-severity zones are rare.

Data setup instructions in [NOTES.md](NOTES.md). `data/` is gitignored — images are never committed.

## Approach

1. Stratified 70/15/15 split using [src/prepare_data.py](src/prepare_data.py).
2. EDA — class distribution, image dimensions, sample images per class. See [notebooks/01-eda.ipynb](notebooks/01-eda.ipynb).
3. Train ResNet50 with the backbone frozen, two runs back-to-back. See [notebooks/02-baseline-resnet.ipynb](notebooks/02-baseline-resnet.ipynb):
   - **Run 1** — no imbalance correction. Watch it fail on minority classes.
   - **Run 2** — class-weighted cross-entropy. Same architecture, same data, same LR. Compare.
4. Held-out test-set evaluation on the better run. *(in progress)*
5. Calibration analysis — reliability diagram, ECE. *(in progress)*
6. Skin-tone fairness audit using a brightness proxy. See [notebooks/03-skin-tone-fairness.ipynb](notebooks/03-skin-tone-fairness.ipynb). *(in progress)*

## Results so far (validation set)

| Metric | Run 1 — no weighting | Run 2 — class-weighted |
|---|---|---|
| Val accuracy | 0.7964 | 0.7027 |
| Macro-avg F1 | 0.44 | 0.47 |
| `df` recall | 0.18 | 0.55 |
| `vasc` recall | 0.07 | 0.67 |
| `mel` recall | 0.24 | 0.40 |
| `nv` recall | 0.96 | 0.78 |

Run 1 looks better on raw accuracy, but it's mostly predicting `nv` for everything — the dominant class makes that strategy "work." Run 2's accuracy drops because the model stops cheating, and minority-class recall jumps 2-9× in exchange. That tradeoff is the whole point of the comparison.

![Run 1 vs Run 2 confusion matrices](results/comparison.png)

## Limitations

- HAM10000 is **adult dermoscopy imagery**, not **neonatal jaundice**. Visual cues overlap (skin tone, pigmentation, color gradients), but the clinical task is different. A model trained here cannot be used clinically.
- All numbers above are on the validation set. Held-out test evaluation is pending.
- HAM10000 ships no Fitzpatrick skin tone labels, so the fairness notebook uses a brightness proxy — a real fairness audit would require labeled tones and a representative dataset. Documenting this limit is part of the point.
- No calibration analysis yet; predicted probabilities may not reflect true confidence.

## Structure

```
kramer-classifier/
├── README.md                       (this file)
├── NOTES.md                        (data download + setup)
├── requirements.txt
├── data/                           (gitignored)
│   ├── train/  val/  test/
├── notebooks/
│   ├── 01-eda.ipynb                done
│   ├── 02-baseline-resnet.ipynb    done (Run 1 + Run 2)
│   └── 03-skin-tone-fairness.ipynb in progress
├── src/
│   ├── prepare_data.py             stratified split
│   ├── data.py                     dataloaders + augmentation
│   ├── model.py                    ResNet50 with frozen backbone
│   ├── train.py                    training loop
│   └── evaluate.py                 metrics, confusion matrix
└── results/
    ├── v1_training_curves.png
    ├── v1_confusion_matrix.png
    ├── v2_confusion_matrix.png
    └── comparison.png
```

## What's next

- [ ] Held-out test-set evaluation on Run 2 (the winning model)
- [ ] Reliability diagram + ECE for calibration
- [ ] Skin-tone fairness notebook — brightness-binned per-class recall
- [ ] Final README pass with test-set numbers

## Reproducing

```bash
pip3 install -r requirements.txt
# follow NOTES.md to download HAM10000 and run src/prepare_data.py
jupyter lab notebooks/
```

Run notebooks in order: 01 → 02 → 03.

## Success criteria

- Reproducible from a clean clone via NOTES.md
- Test-set accuracy > 70% on the class-weighted model
- Limitations clearly stated (dataset, fairness, calibration)
- Fairness analysis present, with honest framing about what HAM10000 can and can't tell us about skin-tone bias
