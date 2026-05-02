# ECG Arrhythmia Classifier — Phase 3 Capstone

A 1D CNN for atrial fibrillation detection from short single-lead ECG segments.

**Status:** Not started. Planned for Week 3.

---

## Background

Arrhythmia detection from short single-lead ECG is a well-studied problem with a strong public benchmark in the PhysioNet/Computing in Cardiology 2017 Challenge. The challenge dataset contains 8,528 ECG recordings (9–60 seconds) labeled as Normal, AF, Other rhythm, or Noisy.

The technique used here — 1D CNN on raw waveforms, optionally combined with spectrogram-based 2D CNN — is the foundation for a wide range of biomedical signal classification problems beyond cardiology (EEG, EMG, respiratory waveforms).

## Why this project

It's the canonical entry point for biomedical signal CNNs. The dataset is well-curated, the task is well-defined, the published baselines are abundant, and the technique transfers directly to any 1D physiological signal.

## Approach

1. Load PhysioNet 2017 data via the `wfdb` library
2. Signal preprocessing: bandpass filter, normalization, segmentation
3. Baseline model: 1D ResNet on raw waveforms
4. Comparison model: 2D CNN on STFT/mel-spectrograms
5. Ensemble both approaches; evaluate on held-out test set
6. Proper temporal validation — no random splits across recordings from the same patient

## Planned structure

```
ecg-arrhythmia/
├── README.md            (this file)
├── data/                (git-ignored; download script in src/)
├── notebooks/
│   ├── 01-eda-and-signal-processing.ipynb
│   ├── 02-1d-cnn-baseline.ipynb
│   ├── 03-spectrogram-2d-cnn.ipynb
│   └── 04-ensemble-and-evaluation.ipynb
├── src/
│   ├── download_data.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── train.py
│   └── evaluate.py
├── requirements.txt
└── results/
```

## Success criteria

- F1 score on AF class within published baseline range (~0.80 was the 2017 winning score)
- Clear comparison of 1D vs spectrogram approaches
- Proper patient-level holdout (no leakage)
- Well-documented preprocessing pipeline that could be reused for other signals
