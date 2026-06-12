# Neonatal ML Learning

A public learning log as I build ML and data science skills on the tech team at Corvita Biomedical — the team building the ARK Incubator, a portable edge-native AI neonatal life support platform.

This repo tracks three parallel learning tracks:

1. **ML concepts** — the techniques that underpin NOA-AI (CNNs, signal processing, fusion, calibration), learned on public datasets
2. **AI engineering foundations** — reading and notes from *AI Engineering* (Chip Huyen), *TinyML* (Warden & Situnayake), and *Interpretable Machine Learning* (Molnar), connected to what Corvita is actually building
3. **General data science** — Python, NumPy, pandas, and the broader toolkit

All work uses public datasets (PhysioNet, ISIC, Kaggle releases) or self-generated synthetic data. No proprietary Corvita data, code, or documents are committed here.

---

## Completed projects

### Kramer classifier

An 8-day compressed sprint that shipped a working transfer-learning prototype for neonatal jaundice severity classification. Trained on HAM10000 (adult dermoscopy) as a public stand-in for neonatal jaundice imagery. Includes class-weighted training, calibration analysis (ECE), skin-tone fairness audit, lighting-robustness training run, temperature-scaling recalibration, and a 10-minute demo notebook.

**Honest scope:** prototype of the technique, not a deployable model. The architecture and training pipeline mirror what the production neonatal jaundice classifier will use; the training data does not.


Code and results: [`projects/kramer-classifier/`](projects/kramer-classifier/)

---

### EMG fatigue detector

A 1D CNN for detecting respiratory muscle fatigue from diaphragmatic surface EMG, built as a prototype of the specialist-CNN architecture pattern used in multi-modal clinical inference systems. Trains on synthetic data from a physiologically calibrated simulator, uses a 1D ResNet with a deterministic feature branch, and produces five concept-bottleneck outputs (fatigue probability, respiratory effort, MDF slope, spectral compression, sample entropy).


Code: [`projects/emg-fatigue/`](projects/emg-fatigue/)

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
└── resources/                   Reading list, dataset links, tooling notes
```

---

## Learning roadmap

Five phases across the core ML skill domains relevant to clinical decision support:

| Phase | Focus | Capstone |
|---|---|---|
| Stage 1 — ML Foundations | Probability, classification, evaluation metrics | — |
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
| AI Engineering Ch. 8 — Dataset Engineering | ✅ done |
| AI Engineering Ch. 9 — Inference Optimization | ✅ done |
| AI Engineering Ch. 10 — AI Engineering Architecture | ✅ done |
| Stage 1.1 — Probability & Bayesian Inference | 🚧 in progress |
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
