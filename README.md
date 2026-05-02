# neonatal-ml-learning

A compressed 5-week curriculum for learning data science and machine learning, anchored on multi-modal neonatal monitoring problems. Public datasets only.

## Plan

Five phases, one week each:

1. **Foundations** (week 1) — Python, NumPy, pandas, time-series indexing
2. **Image CNNs** (week 2) — Transfer learning → neonatal jaundice classifier
3. **Signal CNNs** (week 3) — 1D CNNs on biomedical waveforms → ECG arrhythmia classifier
4. **Audio ML** (week 4) — Mel-spectrograms + CNN → infant cry classification
5. **Fusion & Calibration** (week 5) — Multi-modal fusion, probability calibration, SHAP, fairness analysis

**Anchor project:** Multi-modal neonatal sepsis early-warning system fusing HRV, temperature variance, and movement quality. Validated against PhysioNet 2019 Sepsis Challenge data.

Full breakdown in [STUDY_PLAN.md](STUDY_PLAN.md).

## Repo layout

```
.
├── notebooks/         Per-phase notebooks (one per checkpoint)
│   ├── phase1-foundations/
│   ├── phase2-image-cnns/
│   ├── phase3-signal-cnns/
│   ├── phase4-audio-ml/
│   └── phase5-fusion-calibration/
├── projects/          Capstone projects with their own READMEs
│   ├── kramer-classifier/
│   ├── ecg-arrhythmia/
│   └── sepsis-early-warning/
└── resources/         Reading list, datasets, tooling notes
```

Each phase folder has a `NOTES.md` with running observations.

## Setup

See [SETUP.md](SETUP.md) for how to install dependencies and launch JupyterLab.

## License

MIT — see [LICENSE](LICENSE).
