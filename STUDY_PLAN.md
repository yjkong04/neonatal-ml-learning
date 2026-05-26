# Data Science Study Plan — Neonatal ML

**Author:** Yejin Kong
**Started:** April 2026
**Last revised:** May 2026 — updated for tech team role, added textbook reading strategy
**Goal:** Build production-relevant ML and data science skills as a member of the Corvita tech team. Transferable to broader ML engineering roles long-term.

---

## Status

- ✅ McKinney Chapter 4 (NumPy)
- ✅ McKinney Chapter 5 (pandas intro)
- ✅ Kramer classifier — done, demoed, landed the tech team role
- ✅ EMG fatigue detector — synthetic pipeline + 1D CNN + concept bottleneck done
- 🚧 ECG arrhythmia classifier — data loading started (PhysioNet 2017)
- ⏸️ McKinney Chapters 8, 10, 11 (wrangling, groupby, time series) — queued
- 📖 AI Engineering (Chip Huyen) — reading in parallel

---

## Why this plan looks the way it does

The production system this work supports is a multi-modal sensor fusion stack, not a single-modality CNN. Its clinical alerts fuse outputs from specialist models (CNNs on images, signals, audio) with rule-based scorers (Sarnat, Bhutani, nSOFA, HeRO). The platform is mature; the specialist-model fleet is the bottleneck.

This plan targets four skill domains in order of leverage:

1. **Foundations** — Python, pandas, NumPy, time-series indexing
2. **Image CNNs** — anchors skin / jaundice / neurovision / retraction classifiers
3. **1D signal CNNs** — anchors cardiac and respiratory waveform models
4. **Fusion & calibration** — where multi-modal clinical alerts actually live

Audio ML is folded in as a shorter side-phase (mel-spectrograms → 2D CNN, technique-adjacent to image CNNs).

---

## Textbooks and reading strategy

### *Python for Data Analysis* — McKinney
**How to read it:** Not cover-to-cover. Use it as a reference and pull chapters when you need them.
- Ch. 4, 5 — done
- Ch. 6 (data loading), 7 (cleaning) — skim, pull what's relevant to active project
- Ch. 8 (wrangling), 10 (groupby), 11 (time series) — read these carefully, they matter for signal work
- Skip Ch. 12-14 (modeling chapters) — that's what PyTorch and fast.ai are for

### *AI Engineering* — Chip Huyen
**How to read it:** Not linear. Hit the high-priority chapters first, then fill in context chapters, skim the rest. The book is written through an LLM/foundation model lens — you'll need to mentally translate concepts to the CNN + edge AI context of NOA-AI.

**Notes format:** Jupyter notebooks in `textbooks/ai-engineering/` — markdown cells for concepts and NOA-AI connections, code cells only where there's something worth running (Ch. 9 has hands-on content; most other chapters are conceptual). Think reading journal, not coding workbook.

**Read fully (high priority):**
- **Ch. 3** — Evaluation Methodology (how to know if a model is any good — most transferable skill in the book)
- **Ch. 4** — Evaluate AI Systems (building an eval pipeline, model selection)
- **Ch. 8** — Dataset Engineering (data curation, augmentation, synthesis — directly relevant to building CNN training sets)
- **Ch. 9** — Inference Optimization (quantization, model compression, latency — edge AI, exactly what ARK needs)
- **Ch. 10** — AI Engineering Architecture (production systems, monitoring, pipelines — maps to NOA-AI's orchestration layer)

**Read for context:**
- **Ch. 2** — Understanding Foundation Models (architecture, post-training, quantization basics — builds vocabulary)
- **Ch. 7** — Finetuning (memory math, LoRA, quantization deep-dive — relevant when CNNs move to production)

**Skim:**
- **Ch. 5** — Prompt Engineering (not on critical path)
- **Ch. 6** — RAG and Agents (not on critical path)

**Suggested reading order:** Ch. 3 → 4 → 8 → 9 → 10, then 2 and 7 for context.

**Key question for each chapter:** How does this apply to NOA-AI or the ARK platform?

### *TinyML* — Pete Warden & Daniel Situnayake
**How to read it:** Front-load the concept chapters, skim the Arduino-specific examples.
- **Read fully:** Ch. 1-2 (the edge AI opportunity — directly relevant to ARK), the quantization and model optimization chapters, the deployment pipeline
- **Skim:** "hello world" intro examples if you already have ML foundations
- **Skip:** Chapters tied to specific microcontroller boards you don't use
- **Key idea:** how do models get small enough to run on the ARK hardware? What tradeoffs does quantization make?

### *Interpretable Machine Learning* — Christoph Molnar (free: christophm.github.io)
**How to read it:** Skip to the methods you're actively using, don't read linearly.
- **Read when you reach Phase 5:** SHAP (shapley values chapter is the most important), LIME, partial dependence plots, feature importance
- **Read now if relevant:** the introduction chapters on why interpretability matters in high-stakes domains — directly applicable to clinical AI
- **Skip:** counterfactual explanation chapters until you need them
- **Key idea:** clinical models need to be explainable. Every output the fusion engine generates needs a "why."

### *Hands-On Machine Learning* — Aurélien Géron (reference, not primary)
**How to read it:** Use as a reference when you need a clear technical explanation of something. Don't read linearly.
- **Pull when needed in Phase 2:** Ch. 14 (CNNs) — clearest technical walkthrough of convolution, pooling, architecture
- **Pull when needed in Phase 3:** Ch. 15 (RNNs/sequences) — relevant for time-series signal work
- **Pull when needed in Phase 5:** Calibration and ensemble chapters

---

## Phase 1 — Foundations (Weeks 1–3)

**Goal:** Comfortable with Python, NumPy, pandas. Specifically time-series indexing — every NOA modality is timestamped.

### Resources
- *Python for Data Analysis* (McKinney) — Ch. 8, 10, 11 are the priority
- fast.ai Lesson 1 as a parallel warm-up (trains a pet classifier in 20 min)

### Checkpoints
- Ch. 4 (NumPy) ✅
- Ch. 5 (pandas intro) ✅
- Ch. 8 (wrangling: join, combine, reshape)
- Ch. 10 (groupby aggregation)
- Ch. 11 (time series) — most important chapter in McKinney for this work

### Deliverables
- Notebook: NumPy basics ✅
- Notebook: pandas fundamentals ✅
- Notebook: rolling-window HRV from R-R intervals
- Notebook: time-series indexing on synthetic vitals

---

## Phase 2 — Image CNNs (Weeks 4–7)

**Goal:** Train and fine-tune image classification models. Understand the full training loop in raw PyTorch (not just the fast.ai wrapper).

### Why this phase anchors many production models
Most image-based specialist CNNs — jaundice zones, skin perfusion, neurovision, respiratory retraction, abdominal distension, thermal mottling — are all 2D image classification. Transfer learning from a pretrained backbone (ResNet, EfficientNet) is the same technique across all of them.

### Resources
- fast.ai *Practical Deep Learning for Coders* — Lessons 1–4
- PyTorch official tutorials (Image Classification, Transfer Learning)
- *Hands-On ML* Ch. 14 (CNNs) — pull when you need a clear architecture explanation

### Checkpoints
- fast.ai Lesson 1 (pets classifier)
- fast.ai Lesson 2 (augmentation, training loop)
- fast.ai Lessons 3-4 (deeper understanding)
- Redo Kramer in raw PyTorch — no fast.ai wrapper

### Public datasets
- ISIC Archive (skin lesions) — used for Kramer
- ImageNet pretrained weights via `torchvision.models`

### Deliverables
- **Portfolio project: Kramer classifier** ✅ — 72.4% test acc, ECE 0.083, fairness + calibration
- Notebook: raw PyTorch image classifier (post-sprint deepening)
- Notebook: data augmentation experiments

---

## Phase 3 — 1D signal CNNs (Weeks 8–11)

**Goal:** Train models on biomedical time-series. Foundation for cardiac and respiratory waveform work.

### Resources
- PhysioNet `wfdb` library documentation
- *Hands-On ML* Ch. 15 (sequences) — pull for signal architecture reference
- Research papers from PhysioNet/Computing in Cardiology challenges

### Checkpoints
- **Week 8:** Signal processing — FFT, STFT, filtering, R-peak detection
- **Week 9:** 1D convolutions in PyTorch; ECG arrhythmia on PhysioNet 2017
- **Week 10:** Spectrogram → 2D CNN; compare to pure 1D CNN
- **Week 11:** HRV feature engineering (SDNN, RMSSD, pNN50, frequency-domain)

### Public datasets
- PhysioNet/CinC 2017 — AF detection (anchor for this phase)
- PTB-XL — large public ECG benchmark
- MIT-BIH Arrhythmia Database

### Deliverables
- Notebook: ECG signal processing from scratch
- Notebook: 1D CNN for AF detection
- Notebook: spectrogram + 2D CNN comparison
- **Portfolio project: ECG arrhythmia classifier**

---

## Phase 4 — Audio ML (Weeks 12–13)

**Goal:** Audio classification via spectrograms. Foundations for cry and respiratory-sound classifiers.

### Resources
- `librosa` library tutorials
- Valerio Velardo's "Deep Learning for Audio" YouTube series

### Checkpoints
- **Week 12:** Audio fundamentals — sampling, mel-spectrograms, MFCCs, augmentation
- **Week 13:** CNN classifier on ESC-50 or UrbanSound8K

### Public datasets
- ESC-50 (technique practice)
- Donate-a-Cry corpus (infant cry)
- ICBHI 2017 (respiratory sounds)

---

## Phase 5 — Fusion & calibration (Weeks 14–16)

**Goal:** Combine specialist outputs into calibrated multi-modal predictions. The layer that separates "person who trains CNNs" from "person who builds clinical decision systems."

### Resources
- *Interpretable Machine Learning* (Molnar) — SHAP and feature importance chapters
- scikit-learn calibration documentation
- Pranav Rajpurkar's *AI for Medical Diagnosis* (Coursera)
- Selected papers (see reading-list.md)

### Topics
- Probability calibration (Platt scaling, isotonic regression, Brier score, ECE)
- Fusion strategies (weighted Bayesian, stacking, late fusion, attention)
- Class imbalance (focal loss, SMOTE, threshold tuning)
- Explainability (SHAP, attention maps, saliency)
- Fairness (subgroup analysis: Fitzpatrick, GA, sex, ethnicity per IEEE 2801)
- Temporal validation (time-series CV, patient-level holdout)

### Deliverables
- Notebook: probability calibration walkthrough
- Notebook: SHAP-based explanations
- Notebook: subgroup fairness analysis
- **Anchor project: multi-modal sepsis early-warning system**

---

## Anchor project

**Multi-modal neonatal sepsis early-warning system.** Fuses HRV trend (1D ECG CNN) + temperature variance (tabular) + movement quality (video CNN), with calibrated probability outputs and demographic subgroup fairness analysis. Validated on PhysioNet 2019 Sepsis Challenge data.

Hits every major skill in this plan in one project: 1D CNNs, classical feature engineering, video CNNs, fusion, calibration, SHAP, time-series validation, and fairness analysis.

---

## Required reading (papers)

See [resources/reading-list.md](resources/reading-list.md) for full list. Priority reads:

- **Reyna et al. 2020** — PhysioNet/CinC 2019 Sepsis Challenge
- **Moorman et al. 2011** — Original HeRO score paper
- **Esteva et al. 2017** — *Nature* dermatology CNN paper
- **Kramer 1969** — Original Kramer staging reference

---

## What to deprioritize

- Kaggle competitions on housing/Titanic — low learning-per-hour
- NLP, RL, recommenders — not on critical path
- Goodfellow textbook cover-to-cover — reference only
- Hyperparameter tuning before model architecture is right
- Building from scratch when transfer learning works

---

## Progress markers

- **End of Week 3:** Can load a CSV of vitals, resample to 1Hz, compute rolling HRV features, plot
- **End of Week 7:** Raw PyTorch image classifier (Kramer redone without fast.ai)
- **End of Week 11:** ECG arrhythmia classifier on PhysioNet 2017
- **End of Week 13:** Audio classifier
- **End of Week 16:** Multi-modal sepsis early-warning system

---

## Transferable signal for big tech

- *Built a prototype neonatal jaundice classifier using transfer learning, demonstrating the architecture and training pipeline for production CNN deployment in medical edge AI.*
- *Built a multi-modal early sepsis warning system fusing physiological time-series and computer vision, with calibrated probability outputs and demographic fairness analysis, validated against PhysioNet open data.*
- *Designed an evaluation framework with subgroup fairness metrics, calibration analysis, and SHAP-based explainability for clinical ML models.*

Underlying tooling (PyTorch, scikit-learn, SHAP, pandas) is identical to general ML roles. Medical AI is a premium specialization, not a narrowing.

---

*This is a living document. Update it when scope changes, not after every session — git log is the activity record.*
