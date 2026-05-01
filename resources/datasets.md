# Datasets

Public datasets used across this repo. None of these contain proprietary or patient-identifiable data without proper credentialing.

---

## Phase 1 — Foundations

For pandas/time-series practice, generate synthetic vital signs traces or use any clean public time-series. No specialized dataset required for foundations.

---

## Phase 2 — Image CNNs

| Dataset | Use | Access |
|---|---|---|
| Oxford-IIIT Pet Dataset | fast.ai Lesson 1 reproduction | Open, ~800MB |
| ISIC Archive (Skin Lesions) | Transfer learning technique on medical imaging | Free with registration |
| Public neonatal jaundice photo collections | Kramer classifier prototype | Search Kaggle and academic repositories; small sets exist |
| ImageNet (pretrained weights) | Backbone for transfer learning | No download; via `torchvision.models` |

---

## Phase 3 — Signal CNNs

| Dataset | Use | Access |
|---|---|---|
| PhysioNet/CinC 2017 Challenge | AF detection from short ECG segments | Open, ~700MB. [physionet.org/content/challenge-2017](https://physionet.org/content/challenge-2017/) |
| PTB-XL | Large public ECG benchmark with diagnostic labels | Open, ~3GB. [physionet.org/content/ptb-xl](https://physionet.org/content/ptb-xl/) |
| MIT-BIH Arrhythmia Database | Classic ECG benchmark | Open, ~75MB |
| ICBHI 2017 Respiratory Sound Database | For `cnn_resp` proxy work | Open with registration |
| MIMIC-IV-WDB (Waveform Database) | High-resolution ICU waveforms | **Credentialed access required** — complete CITI training course first |

---

## Phase 4 — Audio ML

| Dataset | Use | Access |
|---|---|---|
| ESC-50 | Environmental sound classification (technique practice) | Open, ~600MB |
| UrbanSound8K | Audio classification benchmark | Open with registration |
| Donate-a-Cry Corpus | Infant cry classification | Open, small (~500 clips) |

---

## Phase 5 — Fusion & Calibration

| Dataset | Use | Access |
|---|---|---|
| PhysioNet 2019 Sepsis Challenge | Multi-modal sepsis prediction (anchor project) | Open, ~3GB |
| MIMIC-IV (clinical) | Multi-modal clinical data for fusion experiments | **Credentialed access required** |

---

## On credentialed access

Several PhysioNet datasets (MIMIC-IV in particular) require completing a CITI Program human subjects research training course. This takes 4–6 hours and produces a recognized credential. Doing this is itself part of the learning — it's the same regulatory framework that governs real clinical data work.

[CITI Program: Data or Specimens Only Research](https://about.citiprogram.org/series/human-subjects-research-hsr/)

---

## Synthetic data for prototyping

When public data is sparse (e.g., neonatal jaundice imagery), synthetic data is acceptable for prototyping the training pipeline — the goal is to validate the architecture and harness, not to produce a deployable model. Document clearly when synthetic data is used.
