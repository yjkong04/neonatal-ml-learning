# Stage 1 — ML Foundations

**Goal:** Understand the core ML concepts that underpin every NOA-AI module — probability, classification, evaluation metrics, feature engineering, and generalization. These are the building blocks before touching any model architecture.

## Goal

Fluent enough in classification theory and probability to reason about what a NOA model is actually doing when it outputs `P(sepsis) = 0.73` — and to evaluate whether that number means anything.

## Key references
- `scipy.stats` and `numpy` for probability exercises
- scikit-learn docs: `metrics` module (classification_report, roc_auc_score, calibration_curve)
- STUDY_PLAN.md Stage 1 (lines 267–291) — NOA-AI connection for each subtopic
- internal system specification — 11-stage inference pipeline, disease head probability outputs
- DOC 6 — a neonatal movement quality classifier (98% sensitivity / 91% specificity targets)

## Notebooks
- `01-probability-bayes.ipynb` — probability fundamentals, Bayes' theorem, NOA-AI application 🚧 active
- `02-classification-outcomes.ipynb` — binary classification, decision boundaries, sigmoid
- `03-sensitivity-specificity-auc.ipynb` — ROC curves, threshold tuning, PPV/NPV
- `04-feature-engineering.ipynb` — feature construction, the a multi-scale physiological feature vector input vector
- `05-overfitting-regularization.ipynb` — bias-variance, cross-validation, L1/L2

## Running notes
*(populate as the stage progresses — questions, gotchas, things that clicked)*
