# Data Science Study Plan — NOA-AI Aligned

**Author:** Yejin Kong
**Started:** April 2026
**Last revised:** May 2026 — added compressed sprint to ship Kramer prototype
**Goal:** Build production-relevant data science skills aligned with Corvita's NOA-AI roadmap, transferable to big tech / general ML opportunities.

---

## Status

Phase 1 in progress.
- ✅ McKinney Chapter 4 (NumPy)
- ✅ McKinney Chapter 5 (pandas intro)
- 🚧 Compressed sprint — Kramer classifier prototype (target: end of next week)
- ⏸️ McKinney Chapters 6-11 (resumes after Kramer demo)

---

## Compressed sprint — Kramer classifier prototype

**Why this exists:** My boss wants to see concrete proof by end of next week that my learning translates to real technical work. Rather than dropping the foundations work, I'm running an 8-day compressed sprint that front-loads enough basics to ship a working trained model. The original Phase 1 schedule resumes after the demo.

**The deliverable:** A working image classifier prototype for neonatal jaundice severity, demonstrated end-to-end, using the same transfer-learning techniques `cnn_kramer` will eventually use in production.

**Honest scope:** This is a learning prototype, not a deployable model. Trained on public skin imagery as a stand-in for real neonatal jaundice data. Architecture and training pipeline mirror what the production version would use.

### Day-by-day

**Days 1-2:** McKinney Chapter 6 (data loading, fast skim) + Chapter 7 (cleaning, fast skim). Start fast.ai Lesson 1 in parallel — train the pet classifier as a warm-up.

**Day 3:** Fast.ai Lesson 2 (data augmentation, training loop intuition). Set up project scaffold at `projects/kramer-classifier/`. Find and download dataset.

**Day 4:** Load and explore dataset. Build baseline model using fast.ai high-level API. Get *some* end-to-end training run to complete, even if accuracy is poor.

**Day 5:** Improve model. Try ResNet18, ResNet50, EfficientNet backbones. Tune learning rate. Add augmentation. Track what helps.

**Day 6:** Evaluation. Confusion matrix, per-class accuracy, calibration plot if time. Generate visualizations.

**Day 7:** Demo notebook + README + final polish. Write honest framing of what this is and isn't.

**Day 8 (buffer):** Fix anything broken.

### Dataset options (in priority order)

**A) Existing public neonatal jaundice data** — small Kaggle/academic releases. On-topic; smaller/messier data.
**B) ISIC skin lesion dataset as proxy** — same technique, much larger data. Off-topic content; well-curated.
**C) General skin tone / color classifier** — fallback only.

### Success criteria for the demo

- Reproducible from a clean clone
- Some reasonable accuracy on held-out test data (target: above random chance, doesn't need to be impressive)
- Clear documentation of dataset choice, model, and results
- Honest discussion of limitations
- A 10-minute walkthrough notebook for the boss demo
- README that frames this correctly: prototype of the technique, not a deployable model

---

## Why the long-term plan looks the way it does

NOA-AI is a multi-modal sensor fusion system, not a single-modality CNN classifier. The clinical alerts (HIE, RDS, PDA, Hyperbili, Sepsis) are all multi-parameter — they fuse outputs from specialist models (CNNs on images, signals, audio) with rule-based scorers (Sarnat, Bhutani, nSOFA, HeRO). The platform is mature; the model fleet is the bottleneck.

This plan targets four skill domains in order of leverage to the codebase:

1. **Foundations** — Python, pandas, NumPy, time-series indexing
2. **Image CNNs** — anchors `cnn_skin`, `cnn_kramer`, `cnn_neurovision`, `cnn_retraction`, `cnn_abdominal`
3. **1D signal CNNs** — anchors `cnn_cardio`, `cnn_resp`, `cnn_emg_fatigue`, supports the existing `cnn_eeg`
4. **Fusion & calibration** — the layer where the actual NOA alerts live (ARK-107 through ARK-114)

Audio ML (`cnn_cry`, `cnn_pphn`) is folded in as a shorter side-phase since it's technique-adjacent to image CNNs (mel-spectrograms → 2D CNN).

---

## Anchor project (long-term)

**Multi-modal neonatal sepsis early-warning system.** Fuses HRV trend (1D ECG CNN) + temperature variance (tabular features) + movement quality (video CNN), with calibrated probability outputs and demographic subgroup fairness analysis. Validated on PhysioNet 2019 Sepsis Challenge data.

Direct analog to ARK-114 in Corvita's alarm registry. Hits 1D CNNs, classical feature engineering, video CNNs, fusion, calibration, and time-series validation — every major skill in this plan — in one project.

---

## Phase 1 — Foundations (Weeks 1–3)

**Goal:** Comfortable with Python, NumPy, pandas. Specifically time-series indexing because every NOA modality is timestamped.

### Resources
- *Python for Data Analysis* (3rd ed.), Wes McKinney
- Corey Schafer's Python YouTube series (only if Python syntax feels rusty)

### Chapters in priority order
- **Chapter 4 (NumPy)** ✅ done
- **Chapter 5 (pandas intro)** ✅ done
- **Chapter 6 (data loading)** — covered fast during compressed sprint
- **Chapter 7 (cleaning)** — covered fast during compressed sprint
- **Chapter 8 (wrangling: join, combine, reshape)** — after Kramer demo
- **Chapter 10 (groupby aggregation)** — after Kramer demo
- **Chapter 11 (time series)** — after Kramer demo, the most important chapter

### Deliverables (some already done)
- Notebook: NumPy basics ✅
- Notebook: pandas fundamentals ✅
- Notebook: rolling-window HRV from R-R intervals (post-Kramer)
- Notebook: time-series indexing on synthetic vitals (post-Kramer)

---

## Phase 2 — Image CNNs (Weeks 4–7)

**Goal:** Train and fine-tune image classification models. Build the Kramer classifier (compressed sprint version) and follow up with the deeper PyTorch version.

### Why this collapses five target CNNs
`cnn_kramer`, `cnn_skin`, `cnn_neurovision`, `cnn_retraction`, `cnn_thermal_mottling`, `cnn_abdominal`, `cnn_pulse_pressure`, `cnn_precordial` are all 2D image classification (or thermal multi-channel image). The technique — transfer learning from a pretrained ResNet/EfficientNet — is identical across all of them.

### Resources
- fast.ai *Practical Deep Learning for Coders* — Lessons 1–4
- Official PyTorch tutorials (Image Classification, Transfer Learning) as a parallel to fast.ai

### Updated checkpoints (reflecting compressed sprint)
- **Sprint Days 2-3:** fast.ai Lesson 1 (pets classifier) + Lesson 2 (augmentation)
- **Sprint Days 4-7:** Kramer prototype — fast.ai high-level → improve → evaluate → demo
- **After demo:** fast.ai Lessons 3-4 (deeper understanding)
- **Later in Phase 2:** Redo Kramer in raw PyTorch (no fast.ai wrapper) to lock in the fundamentals

### Public datasets
- ISIC Archive (Skin Lesions) — primary proxy
- Public neonatal jaundice photo collections (Kaggle, academic)
- ImageNet pretrained weights via `torchvision.models`

### Deliverables
- Notebook: pets classifier (fast.ai Lesson 1) — during sprint
- **Portfolio project: Kramer classifier v0** — sprint deliverable
- Notebook: skin lesion classifier (deeper PyTorch version) — post-sprint
- Notebook: data augmentation experiments — post-sprint

---

## Phase 3 — 1D signal CNNs (Weeks 8–11)

**Goal:** Train models on biomedical time-series. Build foundations for `cnn_cardio` and `cnn_resp`.

### Resources
- *Forecasting: Principles and Practice* (Hyndman & Athanasopoulos) — free online
- PhysioNet `wfdb` Python library documentation
- Research papers from PhysioNet/Computing in Cardiology challenges

### Weekly checkpoints
- **Week 8:** Signal processing fundamentals — FFT, STFT, mel-spectrograms, filtering, R-peak detection
- **Week 9:** 1D convolutions in PyTorch; ECG arrhythmia classifier on PhysioNet 2017
- **Week 10:** Spectrogram → 2D CNN approach; compare to pure 1D CNN
- **Week 11:** HRV feature engineering (SDNN, RMSSD, pNN50, frequency-domain) combined with deep features

### Public datasets
- **PhysioNet/CinC 2017 Challenge** — AF detection, anchor for `cnn_cardio`
- **PTB-XL** — large public ECG benchmark
- **MIT-BIH Arrhythmia Database** — classic ECG benchmark
- **MIMIC-IV-WDB** — high-resolution ICU waveforms (credentialed access)

### Deliverables
- Notebook: ECG signal processing from scratch
- Notebook: 1D CNN for AF detection
- Notebook: spectrogram + 2D CNN comparison
- **Portfolio project: ECG arrhythmia classifier**

---

## Phase 4 — Audio ML (Weeks 12–13)

**Goal:** Audio classification via spectrograms. Foundations for `cnn_cry` and `cnn_pphn`.

### Resources
- `librosa` library tutorials
- Valerio Velardo's "Deep Learning for Audio" YouTube series

### Weekly checkpoints
- **Week 12:** Audio fundamentals — sampling, mel-spectrograms, MFCCs, augmentation
- **Week 13:** CNN classifier on ESC-50 or UrbanSound8K; transfer to cry/respiratory

### Public datasets
- ESC-50 (technique practice)
- Donate-a-Cry corpus (infant cry)
- ICBHI 2017 (respiratory sounds)

---

## Phase 5 — Fusion & calibration (Weeks 14–16)

**Goal:** Combine specialist outputs into calibrated multi-modal predictions. The underrated phase that separates "person who trains CNNs" from "person who builds clinical decision systems."

### Topics
- Probability calibration (Platt scaling, isotonic regression, Brier score)
- Fusion strategies (weighted Bayesian, stacking, late fusion, attention)
- Class imbalance (focal loss, SMOTE, threshold tuning)
- Explainability (SHAP, attention maps, saliency)
- Fairness (subgroup analysis: Fitzpatrick, GA, sex, ethnicity per IEEE 2801)
- Temporal validation (time-series CV, patient-level holdout)

### Resources
- scikit-learn calibration documentation
- SHAP library docs and tutorials
- Pranav Rajpurkar's *AI for Medical Diagnosis* (Coursera)
- Selected papers (see reading-list.md)

### Deliverables
- Notebook: probability calibration walkthrough
- Notebook: SHAP-based explanations
- Notebook: subgroup fairness analysis
- **Anchor project completion:** multi-modal sepsis early-warning system

---

## Required reading

See [resources/reading-list.md](resources/reading-list.md) for full list. Priority reads in Phase 1-2 timeframe:

- **Reyna et al. 2020** — PhysioNet/CinC 2019 Sepsis Challenge (for anchor project context)
- **Moorman et al. 2011** — Original HeRO score paper
- **Esteva et al. 2017** — *Nature* dermatology CNN paper (directly relevant to Kramer)
- **Kramer 1969** — Original Kramer staging reference

---

## NOA-AI target CNN map

| Target CNN | Modality | Public proxy dataset | Phase |
|---|---|---|---|
| `cnn_skin` | RGB image | ISIC + neonatal jaundice photos | 2 |
| `cnn_kramer` | RGB image | Same as above | 2 (sprint anchor) |
| `cnn_neurovision` | RGB video | UCF-101, GMA if accessible | 2+ |
| `cnn_retraction` | RGB video | Custom synthetic | 2 |
| `cnn_thermal_mottling` | Thermal image | Public thermal datasets | 2 |
| `cnn_abdominal` | RGB image | Technique transfer | 2 |
| `cnn_pulse_pressure` | RGB video | Custom | 2 |
| `cnn_precordial` | RGB video | Custom | 2 |
| `cnn_cardio` | 1D ECG | PhysioNet 2017, PTB-XL, MIT-BIH | 3 |
| `cnn_resp` | 1D respiratory | ICBHI 2017 | 3 |
| `cnn_emg_fatigue` | 1D EMG | Ninapro | 3 |
| `cnn_eeg` (runs) | 1D EEG / aEEG | TUH EEG, CHB-MIT | 3 reference |
| `cnn_cry` | Audio | Donate-a-Cry | 4 |
| `cnn_pphn` | Audio + multi-modal | ICBHI + fusion | 4-5 |

---

## Tooling

- **Repo:** public GitHub, frequent commits
- **Environment:** local Python 3.11.9 on macOS; `pip3 install --user` for now (will move to venvs in Phase 2 when adding heavy DL libs)
- **Editor:** VS Code with Jupyter extension
- **Tracking:** W&B from Phase 3 onward
- **Version control:** never push proprietary data; only public/synthetic

### Core libraries
- Numerical: `numpy`, `pandas`, `matplotlib`, `seaborn`
- ML: `scikit-learn`, `xgboost`, `pytorch`, `torchvision`, `fastai`
- Signals: `scipy`, `wfdb`, `librosa`
- Explainability: `shap`, `captum`
- Fairness: `fairlearn`

---

## What to deprioritize

- Kaggle competitions on housing/Titanic — low learning-per-hour
- NLP/RL/recommenders — not on critical path
- Goodfellow textbook cover-to-cover — reference only
- Hyperparameter tuning before model architecture is right
- Building from scratch when transfer learning works

---

## Progress markers

- **End of compressed sprint (next week):** Kramer prototype demoed to boss
- **End of Week 3 (post-sprint, regular schedule):** Can load a CSV of vitals, resample to 1Hz, compute rolling HRV features, plot
- **End of Week 7:** Polished Kramer classifier (deeper PyTorch version)
- **End of Week 11:** ECG arrhythmia classifier
- **End of Week 13:** Audio classifier
- **End of Week 16:** Multi-modal sepsis early-warning system

---

## Transferable signal for big tech

- *Built a prototype neonatal jaundice classifier using transfer learning, demonstrating the architecture and training pipeline for production CNN deployment in medical edge AI.*
- *Built a multi-modal early sepsis warning system fusing physiological time-series and computer vision, with calibrated probability outputs and demographic fairness analysis, validated against PhysioNet open data.*
- *Designed an evaluation framework with subgroup fairness metrics, calibration analysis, and SHAP-based explainability for clinical ML models.*

Underlying tooling (PyTorch, MLflow, SHAP, scikit-learn, pandas) is identical to general ML roles. Medical AI is a premium specialization, not a narrowing.

---

*This plan is a living document. The compressed sprint section is the active focus; the rest is the long-term roadmap.*
