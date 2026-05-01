# Multi-Modal Sepsis Early-Warning System — Anchor Project

The capstone project for this entire learning plan. A multi-modal early-warning system that fuses physiological time-series with computer vision to predict sepsis onset, with calibrated probability outputs and demographic subgroup fairness analysis.

**Status:** Not started. Planned for Weeks 14–16, building on components from earlier phases.

---

## Background

Sepsis early prediction is one of the most studied multi-modal clinical ML problems. The PhysioNet/Computing in Cardiology 2019 Challenge established a public benchmark for predicting sepsis onset from ICU clinical time-series, with thousands of participating teams and well-documented baselines.

The clinical motivation: sepsis mortality is heavily time-dependent — every hour of delayed treatment increases mortality. Early-warning systems that detect sepsis hours before clinical recognition have the potential for substantial mortality reduction (Reyna et al. 2020).

## Why this is the anchor project

It exercises every major skill in this learning plan in a single integrated system:

- **1D signal CNNs** — ECG-derived heart rate variability features
- **Classical feature engineering** — Temperature variance, vital sign trends
- **Image/video CNNs** — Movement quality (analogous to General Movement Assessment)
- **Late fusion** — Combining heterogeneous specialist outputs into a single prediction
- **Probability calibration** — Brier score, calibration curves, isotonic regression
- **SHAP explainability** — Per-prediction feature attribution
- **Subgroup fairness** — Disaggregated performance across demographic strata
- **Time-series cross-validation** — No random splits, no patient leakage

It's also a strong portfolio project that produces a single sentence that hits hard outside the medical context: *"Built a multi-modal early-warning system fusing physiological time-series and computer vision, with calibrated probability outputs and demographic fairness analysis, validated against PhysioNet open data."*

## Approach

1. **Data foundation:** PhysioNet 2019 Sepsis Challenge data as the primary source. Public ICU video datasets for the movement quality component (or synthetic stand-in if access is limited).
2. **Specialist 1 — HRV trend:** 1D CNN on ECG-derived R-R interval sequences. Outputs a sepsis-related risk score over time.
3. **Specialist 2 — Temperature variance:** Tabular MLP on rolling temperature statistics (variance over 1h, 4h, 24h windows).
4. **Specialist 3 — Movement quality:** 2D/3D CNN on video data, classifying movement patterns.
5. **Fusion engine:** Late fusion via weighted Bayesian combination, with weights learned on a validation set.
6. **Calibration:** Isotonic regression on the fusion output, evaluated via Brier score and calibration plots.
7. **Explainability:** SHAP values for the fusion-layer features (which specialist drove which prediction).
8. **Fairness:** Performance disaggregated by sex, age strata, and any other demographic variables in the dataset.

## Planned structure

```
sepsis-early-warning/
├── README.md            (this file)
├── data/                (git-ignored)
├── notebooks/
│   ├── 01-eda.ipynb
│   ├── 02-hrv-specialist.ipynb
│   ├── 03-temp-variance-specialist.ipynb
│   ├── 04-movement-specialist.ipynb
│   ├── 05-fusion-engine.ipynb
│   ├── 06-calibration.ipynb
│   ├── 07-shap-explanations.ipynb
│   └── 08-fairness-analysis.ipynb
├── src/
│   ├── specialists/
│   ├── fusion.py
│   ├── calibration.py
│   ├── train.py
│   └── evaluate.py
├── requirements.txt
└── results/
```

## Success criteria

- All three specialists are independently trained and evaluated
- Fusion model meaningfully outperforms the best individual specialist (expected — that's the whole point of fusion)
- Calibration plot is reasonably linear; Brier score reported and compared to baselines
- SHAP explanations are coherent (the specialists most relevant to a positive prediction are the ones flagged by SHAP)
- Subgroup analysis surfaces any disparities clearly, even if those disparities reflect dataset limitations
- Clear writeup that frames methodology, results, and limitations honestly

## Note

This is a *learning project*, not a clinical product. Nothing here is intended to inform clinical decisions. All work is on public data. The point is to demonstrate competence with the full multi-modal clinical ML pipeline, end to end.
