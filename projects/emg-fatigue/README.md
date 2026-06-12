# CNN-EMG-Fatigue — NOA Specialist Module

Detects respiratory muscle exhaustion from diaphragmatic surface EMG.


**Status:** In progress. Training on synthetic data; real-sensor integration pending.

---

## Signal
- No Layer 2 polynomial threshold — CNN-only path

## Approach

Fatigue manifests breath-by-breath in the EMG spectrum: as the diaphragm tires, the median frequency (MDF) of each inspiratory burst decreases and the signal loses high-frequency content. Rather than integrating over a long window, we track the MDF trajectory across successive bursts within a 30-second segment (~20–30 breaths for a neonate).

1. Segment each inspiratory burst from the raw signal using the respiratory envelope
2. Compute per-burst spectral features (MDF, MPF, high/low power ratio)
3. Stack burst spectrograms into a 2D trajectory image fed to the CNN
4. Deterministic branch: RMS trend, MDF slope, spectral compression ratio, sample entropy
5. Fuse both branches → 5 concept bottleneck outputs for SA scorer

## Concept bottleneck outputs

| Index | Name | Description |
|-------|------|-------------|
| 0 | `fatigue_prob` | Probability of respiratory muscle fatigue |
| 1 | `resp_effort` | Normalized neural drive / respiratory effort |
| 2 | `mdf_slope` | Rate of median frequency decrease per breath |
| 3 | `spectral_compression` | High/low frequency power ratio (decreases with fatigue) |
| 4 | `entropy` | Sample entropy of burst envelope (decreases with fatigue) |

## References

### Neonatal diaphragmatic EMG — foundational

- [Muller et al. (1979) — The consequences of diaphragmatic muscle fatigue in the newborn infant](https://pubmed.ncbi.nlm.nih.gov/385813/) — landmark paper establishing that a >20% fall in the high/low frequency power ratio indicates diaphragmatic fatigue in neonates, and that fatigue preceded apnea when intercostal compensation failed. Primary justification for `spectral_compression` as a concept bottleneck output.
- [Muller et al. (1981) — Synergistic behavior of inspiratory muscles after diaphragmatic fatigue in the newborn](https://pubmed.ncbi.nlm.nih.gov/7327953/) — shows intercostal recruitment as a compensatory response to diaphragmatic fatigue; relevant to how CNN-EMG-Fatigue output interacts with CNN-Retraction in the SA scorer.
- [Gaultier et al. (1977) — Respiratory muscle EMG in newborns: a non-intrusive method](https://pubmed.ncbi.nlm.nih.gov/617314/) — earliest validation that surface (non-intrusive) diaphragmatic EMG is clinically usable in neonates. Supports the use of surface electrode placement for diaphragmatic EMG in clinical neonatal settings.
- [Pons-Odena et al. (2017) — Low-power system for diaphragmatic EMG acquisition in neonates](https://pubmed.ncbi.nlm.nih.gov/28260954/) — hardware-level validation for surface electrode placement and signal quality in a NICU setting.
- [Ratnovsky et al. (2021) — Respiratory muscle function in the newborn: a narrative review](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8053897/) — covers all assessment methods (EMG, maximal pressures, thoraco-abdominal asynchrony, tension-time index); context for what the SA scorer synthesises.

### Median frequency & spectral methods

- [Merletti et al. (2002) — EMG median frequency, spectral compression and muscle fibre conduction velocity](https://pubmed.ncbi.nlm.nih.gov/11955983/) — mechanistic basis for MDF shift: fiber conduction velocity slowing → spectral compression toward lower frequencies. Core justification for `mdf_slope` and `spectral_compression` features.
- [Alkan & Günay (2022) — Simulation study: factors influencing mean and median frequency of sEMG during fatigue](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9459987/) — quantifies realistic MDF shift ranges; used to calibrate simulator parameters (MDF_FRESH=100 Hz → MDF_EXHAUSTED=45 Hz).
- [Oskoei & Hu (2017) — Real-time muscle fatigue monitoring based on median frequency](https://ieeexplore.ieee.org/document/8068428/) — demonstrates per-window MDF tracking is feasible at clinical speeds; supports the burst-by-burst approach in `features.py`.

### Sample entropy & nonlinear complexity

- [Xie et al. (2010) — Fuzzy approximate entropy: detecting muscle fatigue using EMG](https://pubmed.ncbi.nlm.nih.gov/20099031/) — shows entropy decreases with fatigue and tracks alongside mean frequency; primary justification for `entropy` as a concept bottleneck output.
- [Wang et al. (2021) — Effects of muscle fatigue and recovery on complexity of surface EMG](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8391607/) — SampEn decreases during fatigue and recovers on rest; validates entropy as a reversible fatigue marker.
- [Phinyomark et al. (2023) — What are the best indicators of myoelectric manifestation of fatigue?](https://www.medrxiv.org/content/10.1101/2023.03.02.23286583.full.pdf) — comparative study finding fuzzy and multiscale entropy outperform standard SampEn; consider upgrading `_sample_entropy()` to multiscale SampEn if basic version proves noisy on real data.
- [Li et al. (2021) — Analysis of fatigue using rapid refined composite multiscale sample entropy](https://www.sciencedirect.com/science/article/abs/pii/S1746809421001075) — multiscale SampEn is more sensitive to early-stage fatigue than single-scale; practical implementation reference.

### CNN / deep learning for EMG fatigue

- [PMC (2025) — Muscle fatigue assessment using sEMG: transfer learning approach](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12846094/) — CWT scalogram → pretrained 2D CNN achieves 98.6% binary / 95.6% multiclass accuracy; supports using spectrogram-based input as an alternative branch.
- [PMC (2025) — Multimodal fatigue detection: hybrid CNN-LSTM-Attention](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12157230/) — adding an LSTM over the burst-by-burst feature sequence (temporal trend) improves detection; relevant if the 1D ResNet branch underfits the temporal trajectory.
- [Atzori et al. (2021) — MFFNet: multi-dimensional feature fusion for sEMG fatigue detection](https://www.sciencedirect.com/science/article/abs/pii/S0957417421010332) — validates parallel-branch (deterministic + CNN) fusion design used in `CNNEMGFatigue`.

## Project structure

```
emg-fatigue/
├── data/
│   └── synthetic/        (git-ignored; generated by src/simulate.py)
├── notebooks/
│   ├── 01-simulation-and-eda.ipynb
│   ├── 02-feature-extraction.ipynb
│   └── 03-cnn-training.ipynb
├── src/
│   ├── simulate.py       (synthetic EMG generator)
│   ├── features.py       (deterministic branch — per-burst spectral features)
│   └── models.py         (CNN-EMG-Fatigue architecture)
├── results/
└── requirements.txt
```
