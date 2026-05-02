# Study Plan — 5 Weeks

A compressed, intensive curriculum to build the data science and ML skills needed to work on multi-modal neonatal monitoring systems. Anchored on a multi-modal sepsis early-warning capstone that fuses physiological signals, tabular features, and video.

> **Note on pace.** This is one phase per week. It is aggressive on purpose — the goal is rapid working competence, not exhaustive coverage. Capstones in Phases 2, 3, and 5 may extend beyond their nominal week if the foundational work needs more time; that is expected and not a failure mode.

## Phases at a glance

| Phase | Week | Focus | Capstone |
|---|---|---|---|
| 1 | 1 | Python, NumPy, pandas, time-series indexing | (no capstone — foundations) |
| 2 | 2 | Image CNNs, transfer learning | Neonatal jaundice (Kramer zone) classifier |
| 3 | 3 | 1D signal CNNs on biomedical waveforms | ECG arrhythmia classifier |
| 4 | 4 | Audio ML with mel-spectrograms | Infant cry classification |
| 5 | 5 | Multi-modal fusion, calibration, fairness | Anchor project |

---

## Phase 1 — Foundations (Week 1)

Comfortable enough with the data stack to do real work in Phases 2+. The non-negotiable skill is **time-series indexing** — every modality downstream is timestamped.

**Resource:** McKinney, *Python for Data Analysis* (3rd ed.). Chapters 1–11. Chapter 11 (time series) is the highest-value chapter.

**Topics this week:**
- NumPy: arrays, broadcasting, vectorized ops, random number generation
- pandas: DataFrames, indexing, groupby, joins, missing data
- Time-series indexing: `DatetimeIndex`, resampling, rolling windows, lag features
- Apply: compute basic HRV metrics (SDNN, RMSSD, pNN50) from R-R intervals

**Notebooks:**
- `week01-numpy-basics.ipynb`
- `week01-pandas-fundamentals.ipynb`
- `week01-time-series-indexing.ipynb`
- `week01-hrv-from-rr-intervals.ipynb`

---

## Phase 2 — Image CNNs (Week 2)

Practical transfer learning and fine-tuning on RGB image data. Build muscle memory for the train → evaluate → tune loop.

**Resource:** fast.ai *Practical Deep Learning for Coders*, Lessons 1–4.

**Topics this week:**
- Lesson 1: standard fast.ai training loop on a baseline classifier (pets)
- Lessons 2–3: data augmentation, learning rate finder, fine-tuning
- Lesson 4: production considerations
- Begin Kramer dataset exploration

**Notebooks:**
- `week02-pets-classifier.ipynb`
- `week02-data-augmentation.ipynb`
- `week02-pytorch-transfer-learning.ipynb`
- `week02-kramer-prototype.ipynb`

**Capstone — Kramer Zone Jaundice Classifier**
- Public neonatal jaundice imagery
- ResNet50 baseline → fine-tune with augmentation
- Evaluate per-zone accuracy and confusion matrix
- See [projects/kramer-classifier/](projects/kramer-classifier/)

---

## Phase 3 — Signal CNNs (Week 3)

1D CNNs on biomedical time-series. The most clinically relevant phase — most NICU monitoring modalities are 1D physiological waveforms.

**Datasets:**
- PhysioNet Computing in Cardiology Challenge 2017 (atrial fibrillation, single-lead ECG)
- PTB-XL (12-lead ECG, large dataset)
- MIT-BIH Arrhythmia Database (classic benchmark)

**Topics this week:**
- Signal preprocessing: filtering, segmentation, windowing, R-peak detection
- 1D CNN architectures (ResNet-style for waveforms)
- Spectrogram → 2D CNN as a comparison approach
- Class imbalance handling, threshold tuning, AUROC vs PR-AUC

**Notebooks:**
- `week03-signal-processing-fundamentals.ipynb`
- `week03-1d-cnn-ecg.ipynb`
- `week03-spectrogram-2d-cnn.ipynb`
- `week03-hrv-features-and-fusion.ipynb`

**Capstone — ECG Arrhythmia Classifier**
- See [projects/ecg-arrhythmia/](projects/ecg-arrhythmia/)

---

## Phase 4 — Audio ML (Week 4)

Mel-spectrograms + 2D CNN. Short phase because the technique transfers directly from Phase 2 (treat the spectrogram as an image).

**Datasets:**
- ESC-50 (environmental sounds, for warm-up)
- Donate-a-Cry corpus (infant cry classification)

**Topics this week:**
- Audio signal processing: STFT, mel scale, mel-spectrograms, MFCCs
- ESC-50 baseline
- Donate-a-Cry classifier (limited data — practice augmentation and regularization)

**Notebooks:**
- `week04-audio-fundamentals.ipynb`
- `week04-esc50-classifier.ipynb`
- `week04-cry-prototype.ipynb`

---

## Phase 5 — Fusion & Calibration (Week 5)

Where the curriculum points: combining specialist models, producing calibrated probability outputs, and validating fairness across demographic subgroups.

**Topics this week:**
- Multi-modal fusion strategies: late fusion, feature concatenation, attention-based
- Probability calibration: Platt scaling, isotonic regression, temperature scaling
- Calibration metrics: Brier score, reliability diagrams, ECE
- SHAP and Captum for explainability
- Fairness analysis across demographic strata (Fitzpatrick skin tone, gestational age bracket, sex, ethnicity per IEEE 2801)
- Time-series cross-validation (no leakage)

**Notebooks:**
- `week05-probability-calibration.ipynb`
- `week05-shap-explanations.ipynb`
- `week05-fairness-analysis.ipynb`

---

## Anchor Project — Multi-Modal Neonatal Sepsis Early-Warning

The capstone that ties everything together. Validated against PhysioNet 2019 Sepsis Challenge data.

**Modalities:**
- HRV trend → 1D ECG CNN (Phase 3)
- Temperature variance → tabular features (Phase 1 skills)
- Movement quality → video CNN (extension of Phase 2)

**Outputs:**
- Calibrated probability of sepsis onset within a defined horizon
- SHAP-based explanation per prediction
- Demographic subgroup fairness report

See [projects/sepsis-early-warning/](projects/sepsis-early-warning/) for details.

---

## Tracking progress

- One commit per study session (~30 min minimum)
- Commit message format: `phase1-week1: numpy broadcasting practice` or `kramer: baseline ResNet50 architecture`
- Each phase folder has a `NOTES.md` with running observations, gotchas, and questions
- Capstone projects have their own `README.md` with results and lessons
