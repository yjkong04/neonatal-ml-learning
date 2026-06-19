# Neonatal ML Learning

A public learning log as I build ML and data science skills on the tech team at Corvita Biomedical — the team building the ARK Incubator, a portable edge-native AI neonatal life support platform.

This repo tracks three parallel learning tracks:

1. **ML concepts** — the techniques that underpin NOA-AI (CNNs, signal processing, fusion, calibration), learned on public datasets
2. **AI engineering foundations** — reading and notes from *AI Engineering* (Chip Huyen), *TinyML* (Warden & Situnayake), and *Interpretable Machine Learning* (Molnar), connected to what Corvita is actually building
3. **General data science** — Python, NumPy, pandas, and the broader toolkit

A parallel **math foundations** track (probability, linear algebra, calculus — conceptual fluency, not derivation) runs alongside these, closing the theory gaps that the above tracks assume. Detail in [resources/math_study_plan.md](resources/math_study_plan.md).

All work uses public datasets (PhysioNet, ISIC, Kaggle releases) or self-generated synthetic data. No proprietary Corvita data, code, or documents are committed here.

---

# Neonatal ML Learning

Applied ML engineer in clinical decision support. This repo contains supervised portfolio 
projects and textbook notes built toward production multi-modal medical AI work.

---

## Projects

### Kramer Classifier — Neonatal Jaundice Severity via Transfer Learning

Transfer learning prototype for Kramer zone classification on the HAM10000 dermoscopy dataset.
Demonstrates the full clinical-CNN pipeline: handling severe class imbalance (72% dominant class),
calibration analysis (ECE 0.083), brightness-proxy fairness audit, lighting-robustness training,
and temperature-scaling recalibration.

**Results:** 72.4% test accuracy (macro-F1 0.474) on 7-class HAM10000 with class weighting.
Minority-class recall improved 2–9× over unweighted baseline. ECE 0.083 (production gate: ≤0.04;
temperature scaling applied in notebook 05).

Code: [`projects/kramer-classifier/`](projects/kramer-classifier/)

---

## Skills — what this work demonstrates

- Handling severe class imbalance in clinical datasets (weighted cross-entropy, recall-focused eval)
- Model calibration for decision-support contexts (ECE, reliability diagrams, temperature scaling)
- Skin-tone fairness auditing with honest framing of proxy limitations
- Leakage-safe evaluation (stratified hold-out, patient-level reasoning)
- Transfer learning from general-domain data to a new clinical domain

---

## Repo structure

```
.
├── README.md                    This file
├── STUDY_PLAN.md                Long-term learning roadmap
├── SETUP.md                     Environment setup
├── LICENSE                      MIT
├── .gitignore
├── requirements.txt
│
├── notebooks/                   Learning notebooks and reading notes
│   ├── stage1-ml-foundations/       Probability, Bayesian inference, classification
│   ├── phase1-foundations/          Python, NumPy, pandas
│   ├── phase2-image-cnns/           Image classification, transfer learning
│   ├── phase3-signal-cnns/          1D CNNs, ECG, biomedical signals
│   ├── phase4-audio-ml/             Mel-spectrograms, audio classification
│   ├── phase5-fusion-calibration/   Multi-modal fusion, calibration, fairness
│   └── reading-notes/
│       └── ai-engineering/          Chapter notes — Chip Huyen AI Engineering
│
├── projects/                    Portfolio projects
│   ├── kramer-classifier/           Image CNN for jaundice severity — done
│   ├── emg-fatigue/                 1D CNN for respiratory muscle fatigue — done
│   ├── ecg-arrhythmia/              1D CNN on PhysioNet 2017 — scaffolded
│   └── sepsis-early-warning/        Multi-modal fusion — future
│
└── resources/                   Reading list, dataset links, tooling notes, math study plan
```

---

## Learning roadmap

Five phases across the core ML skill domains relevant to clinical decision support:

| Phase | Focus | Capstone |
|---|---|---|
| 1 — Foundations | Python, NumPy, pandas, time-series indexing | Rolling HRV from synthetic vitals |
| 2 — Image CNNs | Transfer learning, fine-tuning, augmentation | Kramer classifier ✅ |
| 3 — Signal CNNs | 1D CNNs, spectrograms, signal processing | ECG arrhythmia classifier (PhysioNet 2017) |
| 4 — Audio ML | Mel-spectrograms, audio CNNs | Cry / respiratory sound classifier |
| 5 — Fusion & Calibration | Late fusion, calibration, SHAP, fairness | Multi-modal sepsis early-warning system |

Full plan with weekly checkpoints and textbook reading strategy: [STUDY_PLAN.md](STUDY_PLAN.md)

---

## Textbooks

| Book | How to use it | Notes |
|---|---|---|
| *AI Engineering* — Chip Huyen | Non-linear; high-priority chapters first | [`notebooks/reading-notes/ai-engineering/`](notebooks/reading-notes/ai-engineering/) |
| *TinyML* — Warden & Situnayake | Front-load concept chapters; skim hardware-specific examples | — |
| *Interpretable ML* — Molnar (free) | Reference by chapter; pull SHAP and feature importance when you reach Phase 5 | — |
| *Hands-On ML* — Géron | Reference only; pull Ch. 14 for CNNs, Ch. 15 for sequences | — |
| *Python for Data Analysis* — McKinney | Pull Ch. 8, 10, 11; already done Ch. 4–5 | — |

---

## Progress

| Milestone | Status |
|---|---|
| McKinney Ch. 4 (NumPy) | ✅ done |
| McKinney Ch. 5 (pandas intro) | ✅ done |
| Kramer classifier | ✅ done — 72.4% test acc, ECE 0.083, fairness + calibration complete |
| EMG fatigue detector | ✅ synthetic pipeline + CNN + concept bottleneck done |
| AI Engineering Ch. 2 — Foundation Models | ✅ done |
| AI Engineering Ch. 3 — Evaluation Methodology | ✅ done |
| AI Engineering Ch. 4 — Evaluate AI Systems | ✅ done |
| AI Engineering Ch. 8 — Dataset Engineering | 🚧 in progress |
| AI Engineering Ch. 9 — Inference Optimization | 🚧 in progress |
| AI Engineering Ch. 10 — AI Engineering Architecture | 🚧 in progress |
| Stage 1.1 — Probability & Bayesian Inference (math sprint) | 🚧 in progress |
| ECG arrhythmia classifier | ⏸️ scaffolded, not started |
| Rolling HRV notebook | ⏸️ queued |
| Audio classifier | — |
| Multi-modal sepsis early-warning | — |

---

## Tooling

- Python 3.11 on macOS, `venv` per project
- VS Code with Jupyter extension
- PyTorch, torchvision, fastai, scikit-learn, pandas, NumPy
- Signals: SciPy, `wfdb`, `librosa`
- Explainability and fairness: SHAP, Captum, Fairlearn
- Experiment tracking: Weights & Biases (from Phase 3)
- Version control: git CLI + GitHub

---

## License

MIT. See [LICENSE](LICENSE).
