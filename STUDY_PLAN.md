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
- 🚧 Stage 1.1 — Probability & Bayesian Inference (active)
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

**Notes format:** Markdown files in `notebooks/reading-notes/ai-engineering/` — concepts and NOA-AI connections in plain text, code snippets in fenced blocks where relevant (Ch. 9 has hands-on content; most other chapters are conceptual). Think reading journal, not coding workbook.

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

## NOA-AI Specific ML Learning 
### Stage 1 — ML Foundations 

Probability, classification, evaluation metrics — mapped to NOA-AI's clinical probability engine and three-tier alert system.

**1.1 Probability & Bayesian Inference**
*Why for NOA-AI:* NOA outputs calibrated disease probabilities (0.0–1.0), not binary answers. Bayesian thinking is required to understand what `P(sepsis) = 0.73` actually means and how to set clinically meaningful alert thresholds.
*Corvita connection:* NOA v2.1 confidence scores; six disease head probability outputs; `P(sepsis | HRV_drop, temp_instability)`.

**1.2 Classification & Binary Outcomes**
*Why for NOA-AI:* Every NOA module is a classifier at heart. The sepsis predictor, jaundice screener, and apnea detector all classify sensor inputs into disease/no-disease.
*Corvita connection:* Six NOA disease heads (HIE, RDS, sepsis, NEC, PDA, hyperbilirubinemia) structured as probabilistic classifiers.

**1.3 Sensitivity, Specificity & AUC**
*Why for NOA-AI:* NOA targets 98% sensitivity for the fidgety movement CP classifier. The three-tier alert system is calibrated around these metrics. You cannot evaluate any NOA module without fluency in these.
*Corvita connection:* a neonatal movement quality classifier (98% sensitivity / 91% specificity); three-tier alert system tuned to minimize false negatives in NICU settings.

**1.4 Feature Engineering**
*Why for NOA-AI:* Raw surface-EMG-sensor sensor readings must be transformed into the 67-feature input vector that feeds NOA v2.1's inference network. Understanding feature construction lets you extend or debug models.
*Corvita connection:* NOA v2.1 input vector `F ∈ ℝ⁶⁷ × T₄` — 67 features across 4 temporal resolutions (1s, 1 min, 1 hr, 24 hr).

**1.5 Overfitting, Regularization & Validation**
*Why for NOA-AI:* Neonatal datasets are small and rare. Every NOA module risks memorizing training cases rather than generalizing to new patients. This is the most common practical failure mode.
*Corvita connection:* Limited neonatal training data; NOA must generalize across gestational ages, NICU contexts (Level III/IV), and global geographies (LMIC vs developed-world).

---

### Stage 2 — Time Series & Biosignals *(4–6 weeks)*

Most of NOA-AI runs on continuous physiological signals. HRV, ECG, EEG, SpO₂ — the core competency for clinical early warning.

**2.1 Signal Processing (FFT, Filtering, Spectral Analysis)**
*Why for NOA-AI:* Raw ECG and EEG from surface-EMG-sensor sensors are noisy. Signal processing cleans them before any ML model sees the data. HRV frequency-domain analysis — the basis of sepsis detection — uses FFT on the ECG waveform.
*Corvita connection:* ECG preprocessing; EEG band extraction (delta/theta/alpha/beta); SpO₂ artifact rejection; HRV LF/HF ratio from ECG.

**2.2 LSTM & Recurrent Networks**
*Why for NOA-AI:* Neonatal physiology changes over time, not in snapshots. LSTMs explicitly model sequential dependencies and are the workhorse model for several NOA modules.
*Corvita connection:* NOA M05 — LSTM on 5-minute pose trajectories for fidgety movement classification; HRV trend modeling for sepsis prediction 12–24h before clinical onset.

**2.3 Heart Rate Variability (HRV) Analysis**
*Why for NOA-AI:* HRV is NOA's single most important biosignal for sepsis prediction. The HeRO paradigm — the clinical foundation of M06 — detects HRV characteristic changes 12–24h before sepsis manifests clinically.
*Corvita connection:* M06 Sepsis Predictor — HeRO (Heart Rate Observation) paradigm; SDNN, RMSSD, LF/HF metrics; HRV changes precede clinical sepsis by 12–24h. See NOA-SEPSIS-001.

**2.4 Anomaly Detection**
*Why for NOA-AI:* Most of NOA's clinical value comes from detecting when something is wrong *before* clinicians can see it. Anomaly detection models learn "normal" per patient and flag deviations.
*Corvita connection:* M11 Apnea Detection; thermal anomaly hotspots for NEC; three-tier alert system (Advisory / Warning / Critical) across all NOA modules.

**2.5 Multi-Scale Temporal Features**
*Why for NOA-AI:* A single SpO₂ reading tells you little; a 4-hour downward trend is clinically critical. NOA encodes time at four scales — you need to understand why and how.
*Corvita connection:* NOA v2.1 `T₄` temporal resolutions — instantaneous (1s), short-term (1 min), medium (1 hr), long-term (24 hr), fed as separate feature channels.

---

### Stage 3 — Computer Vision *(4–6 weeks)*

NOA's 10-camera array (5 RGB + 5 thermal) powers jaundice screening, movement analysis, seizure detection, and pain assessment.

**3.1 CNN Architecture & Image Classification**
*Why for NOA-AI:* CNNs power NOA's vision modules. Understanding convolutions, pooling, and feature maps lets you reason about how the jaundice screener extracts bilirubin signals from skin color.
*Corvita connection:* CNN-Kramer model for RGB skin color → bilirubin estimation; Concept Complement Bottleneck Models for FDA-aligned interpretable diagnosis.

**3.2 YOLO Object Detection**
*Why for NOA-AI:* YOLOv8-Pose is a real dependency in the NOA codebase. Understanding it means you can debug and improve the movement analyzer directly.
*Corvita connection:* NOA M05 — YOLOv8-Pose for 17-point infant skeleton; bilateral symmetry analysis; position-invariant with gravity compensation; <50ms per frame requirement.

**3.3 Pose Estimation & Movement Tracking**
*Why for NOA-AI:* Pose → fidgety movement → CP risk is one of NOA's most clinically valuable pipelines. Vision work here directly affects patient outcomes.
*Corvita connection:* a neonatal movement quality classifier — OpenPose-style tracking; fidgety movement classification (normal/abnormal/absent at 9–20 weeks post-term); 98% sensitivity for CP prediction.

**3.4 Transfer Learning on Limited Data**
*Why for NOA-AI:* You will not have 100,000 neonatal training images. Transfer learning from general-purpose vision models is the practical solution for every NOA vision module.
*Corvita connection:* All NOA vision models fine-tuned from pre-trained backbones; catastrophic forgetting prevention; domain adaptation from adult to neonatal anatomy.

**3.5 Thermal Imaging Analysis**
*Why for NOA-AI:* FLIR thermal cameras give NOA a different visual modality. Understanding thermal characteristics lets you contribute to NEC and sepsis detection pipelines.
*Corvita connection:* FLIR Lepton 3.5 × 5 cameras — thermal mottling detection (sepsis), abdominal hotspot detection (NEC), peripheral perfusion gradient analysis.

---

### Stage 4 — Sensor Fusion & Multi-Modal ML *(3–4 weeks)*

No single sensor drives NOA clinical decisions. Fusing 14 modalities is NOA's defining architectural challenge — and its biggest competitive moat.

**4.1 Three-Level Fusion Architecture (Signal → Feature → Decision)**
*Why for NOA-AI:* This is NOA's defining design pattern. Understanding when to fuse at the signal, feature, or decision level is a core architectural skill for extending the system.
*Corvita connection:* NOA Sensor Fusion Engine — L1 Signal Fusion → L2 Feature Fusion → L3 Decision Fusion; combined inference vector `R_d ∈ ℝ⁹³⁺`; 16 specialist + 10 scorer models.

**4.2 Attention Mechanisms in Multi-Modal ML**
*Why for NOA-AI:* Attention is how NOA learns which sensors matter most per clinical condition. Sepsis weights HRV heavily; HIE weights EEG and thermal.
*Corvita connection:* NOA v2.1 references ARMOUR (attention-based multimodal fusion with contrast); Hadamard product in 5 disease head outputs; cross-attention across vital sign and vision streams.

**4.3 Missing Modality & Graceful Degradation**
*Why for NOA-AI:* In real NICUs, sensors disconnect and BLE drops mid-session. NOA must degrade gracefully, not silently fail.
*Corvita connection:* NOA 5-level degradation cascade; per-module circuit breakers; surface-EMG-sensor BLE dropout → fallback to camera + environmental only; no single-point-of-failure.

**4.4 Concept Bottleneck Models**
*Why for NOA-AI:* Regulatory compliance requires NOA to explain its reasoning. Concept bottleneck models make intermediate clinical concepts explicit and auditable — a key FDA differentiator.
*Corvita connection:* NOA v2.1 intermediate concepts (`HRV suppressed`, `thermal mottling present`); FDA prefers auditable intermediate states over black-box neural outputs.

---

### Stage 5 — Edge AI & Deployment *(2–3 weeks)*

NOA runs entirely on-device. No cloud for real-time inference. 16+ models on a Jetson Thor SoC with sub-second latency. This is hard engineering.

**5.1 Model Quantization (FP32 → INT8)**
*Why for NOA-AI:* FP32 models are 4× larger and slower than INT8. On the Jetson Thor, INT8 quantization is required to run 16+ models simultaneously in real time. Hard constraint.
*Corvita connection:* Jetson Thor 275 TOPS INT8; all NOA specialist models quantized without clinical accuracy loss; QAT (quantization-aware training) vs PTQ (post-training quantization) tradeoffs.

**5.2 TensorRT & ONNX Pipeline**
*Why for NOA-AI:* TensorRT is the production inference runtime for all NOA models. Knowing how to export, convert, and profile is required for shipping.
*Corvita connection:* NOA pipeline — PyTorch training → ONNX export → TensorRT compilation → Jetson deployment; latency profiling and kernel optimization.

**5.3 Real-Time Latency & Throughput**
*Why for NOA-AI:* Real-time neonatal monitoring has hard latency requirements. Multi-model scheduling on a shared GPU is non-trivial.
*Corvita connection:* NOA requirements — <50ms per frame (pose), sub-second all clinical alerts, 14+ models running simultaneously on one SoC.

**5.4 Model Compression (Pruning & Distillation)**
*Why for NOA-AI:* Fitting 26+ models on a single SoC requires aggressive compression without sacrificing clinical accuracy.
*Corvita connection:* Knowledge distillation from large teacher models to compact clinical modules; structured pruning for Jetson DLA compatibility.

---

### Stage 6 — Clinical AI & Regulatory *(2–3 weeks)*

Medical AI has requirements general ML does not. FDA SaMD guidance, calibration, and explainability are regulatory clearance gates — not nice-to-haves. **Tackle this early** even though it's numbered later.

**6.1 Calibration (Confidence = Accuracy)**
*Why for NOA-AI:* An uncalibrated model that outputs `P = 0.9` for everything is dangerous and legally indefensible in a clinical setting.
*Corvita connection:* NOA v2.1 explicit Calibration stage in the 11-step inference pipeline; FDA requires `P(sepsis) = 0.8` to genuinely correspond to 80% case rate; temperature scaling and Platt scaling techniques.

**6.2 Explainability (XAI, SHAP, Attention Maps)**
*Why for NOA-AI:* A NICU clinician will not act on "the model says so." NOA must show evidence. XAI is how NOA earns clinical trust and satisfies FDA Class II SaMD expectations.
*Corvita connection:* Every NOA alert includes contributing factors, confidence score, and evidence basis per NOA Insight Format spec; SHAP values for feature attribution.

**6.3 Data Drift Detection**
*Why for NOA-AI:* NOA deploys globally across very different NICU contexts. A model trained on Canadian data may silently fail in an LMIC facility without drift detection.
*Corvita connection:* NOA Drift Monitor — distribution tracking across global ARK fleet; covariate shift vs concept drift; triggers retraining pipeline and PCCP review.

**6.4 FDA SaMD & Predetermined Change Control Plans**
*Why for NOA-AI:* Every model update to a shipped NOA device must comply with FDA PCCP rules. These rules determine what engineering is legally possible post-clearance — understand them before you build.
*Corvita connection:* NOA-AI is SaMD Class II/III; Predetermined Change Control Plan defines locked vs adaptive algorithms; IEC 62304 Class B/C software lifecycle.

---

### Stage 7 — NOA v2.1 Advanced *(Ongoing)*

NOA v2.1's cutting-edge architecture: neuro-symbolic AI, offline RL, federated learning, causal graphs. This is what differentiates Corvita from every competitor.

**7.1 Neuro-Symbolic AI Architecture**
*Why for NOA-AI:* NOA v2.1 is explicitly a neuro-symbolic system. The symbolic layer encodes non-negotiable clinical rules that override neural outputs. Essential for extending NOA safely.
*Corvita connection:* NOA v2.1 "Neuro-Symbolic Clinical Inference Network"; symbolic veto rules (e.g., `SpO₂ < 80% → CRITICAL`, overrides all model outputs); ICD-10 encoded deterministic constraints.

**7.2 Offline Reinforcement Learning**
*Why for NOA-AI:* You cannot experiment with drug dosing on real neonates. Offline RL trains optimization policies from historical data — the only safe approach for medical treatment optimization.
*Corvita connection:* NOA v2.1 — CQL (conservative Q-learning) for offline safe RL; treatment optimization for dexmedetomidine dosing; NeurIPS 2025 spotlight paper (Yan et al.) in references.

**7.3 Federated Learning**
*Why for NOA-AI:* Patient data cannot leave the hospital. Federated learning lets NOA improve from cases across many hospitals without centralizing raw data — required for PIPEDA/HIPAA compliance.
*Corvita connection:* NOA continuous-learning-ready; federated averaging across global ARK fleet; PIPEDA/HIPAA-compliant model updates without raw data sharing.

**7.4 Causal Graphs & Causal Inference**
*Why for NOA-AI:* Correlation is insufficient for clinical AI. Causal graphs let NOA reason about *why* something is happening — critical for FDA justification and for avoiding dangerous spurious correlations in small datasets.
*Corvita connection:* NOA v2.1 causal graph boosting in 5 disease head outputs; encodes clinical causal pathways (e.g., infection → inflammatory response → HRV suppression).

## 5. Project Document Map

When teaching, reference these by name:

| Document                                                    | What's In It                                                                                                                                  |
| ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **internal system specification**                        | Mathematical model, neuro-symbolic architecture, 11-stage inference pipeline, 16 specialist + 10 scorer models, all Stage 7 advanced concepts |
| **DOC 6 — NOA-AI Intelligence System Requirement Document** | 14 clinical AI modules (M01–M13), inputs/outputs, model architectures, performance requirements                                               |
| **NOA-SEPSIS-001 Clinical Algorithm**                       | Sepsis-specific: HRV analysis, comprehensive sensor mapping, three-tier alert logic, treatment protocols                                      |
| **DOC 0 — ARK System Executive Summary**                    | High-level architecture, subsystem map, system context for new team members                                                                   |
| **DOC 1 — ARK Incubator Main Unit**                         | Physical hardware, environmental control, integration backbone                                                                                |
| **DOC 2 — surface-EMG-sensor (Skin-contact Advanced Life Monitor)**       | Three sensor modules (foot/thorax/head), vital sign acquisition                                                                               |
| **DOC 3 — D.O.V.E. (Dual-Output Ventilation Engine)**       | Respiratory support, ventilation modes                                                                                                        |
| **DOC 4 — ARK Device Manager**                              | Embedded software stack, NOA UI integration, model toggles                                                                                    |
| **DOC 5 — Communication & Multi-ARK Manager**               | HL7 FHIR, cybersecurity, fleet management                                                                                                     |
| **DOC 7 — CORVITA Health Data Platform**                    | Cloud pipeline, data flow, anonymization                                                                                                      |
| **NOA Master Metrics Matrix**                               | Performance targets per module                                                                                                                |
| **Product Development Roadmap**                             | Timeline, phasing, ship sequence                                                                                                              |
| **Business Plan / Company Profile**                         | Market context, business model, target users                                                                                                  |

---

## 6. Default Behaviors

- When I ask "explain X," assume I want it tied to NOA-AI unless I say "in general."
- When I share code, review it from a production-ML-in-medical-context lens (latency, calibration, robustness to missing modalities, INT8 compatibility).
- When I share a paper, summarize what it means for NOA specifically — don't just restate the abstract.
- When I get stuck, suggest concrete next steps, not generic encouragement.
- Use the ML learning roadmap widget pattern when I want a structured view; use prose when I want depth on one topic.
- It's fine to push back if I'm asking the wrong question or pursuing something that won't help us ship.

## 7. What NOT to Do

- Don't restate the entire NOA architecture every response — assume project context is loaded.
- Don't add boilerplate "consult a medical professional" disclaimers — I work *in* this domain.
- Don't recommend learning resources unrelated to our actual work (no generic Coursera reading lists).
- Don't generate fictional patient data or clinical scenarios — use the real specs in our project docs.
- Don't over-format. Headers and bullet lists are tools, not defaults.



*This is a living document. Update it when scope changes, not after every session — git log is the activity record.*
