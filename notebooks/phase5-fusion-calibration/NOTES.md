# Phase 5 — Fusion & Calibration

**Week 5** — Multi-modal fusion, probability calibration, fairness, explainability.

## Goal

This is the underrated phase. Most online courses skip it entirely. This is where I move from "person who trains CNNs" to "person who builds clinical decision systems."

## Topics
- Probability calibration (Platt scaling, isotonic regression, Brier score)
- Multi-modal fusion (weighted Bayesian, stacking, late fusion, attention)
- Class imbalance for rare conditions (focal loss, SMOTE, threshold tuning)
- SHAP for explainability
- Subgroup fairness analysis (Fitzpatrick skin type, gestational age, sex, ethnicity)
- Time-series cross-validation (no random splits, no patient leakage)

## Key references
- Guo et al. 2017 (calibration of modern neural networks)
- Lundberg & Lee 2017 (SHAP)
- Rajkomar et al. 2018 (clinical ML at scale, npj Digital Medicine)
- Pranav Rajpurkar's *AI for Medical Diagnosis* on Coursera (companion course)

## Planned notebooks
- `week05-probability-calibration.ipynb` — Calibration walkthrough on imbalanced data
- `week05-shap-explanations.ipynb` — SHAP-based explanations for a clinical model
- `week05-fairness-analysis.ipynb` — Disaggregated metrics across demographic subgroups

## Running notes
*(populate as the phase progresses)*
