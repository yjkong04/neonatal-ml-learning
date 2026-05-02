# neonatal-ml-learning

Public learning-log repo for a compressed 5-week DS/ML curriculum (one phase per week). Capstones map to specialist CNNs that NOA-AI (Corvita Biomedical's neonatal inference engine) needs but doesn't yet have real models for. This is professional development with real deliverables — not a side project.

## Hard rules

- **No Corvita-internal data, code, or documents in this repo. Ever.** Public datasets only (PhysioNet, ISIC, Donate-a-Cry, ESC-50, PTB-XL, MIT-BIH). MIMIC-IV requires CITI training first.
- **Never commit data files** (`.csv`, `.parquet`, `.pt`, `.pth`, etc.). `.gitignore` should handle this — flag if a workflow would create one in a tracked path.
- **Phases run in order.** Don't skip foundations to jump to deep learning.
- **No generic Kaggle** (Titanic, housing prices) — low learning-per-hour for these goals.

## Plan (5 weeks, 5 phases — one phase per week)

1. **Week 1 — Foundations:** Python, NumPy, pandas, time-series indexing. McKinney *Python for Data Analysis* (3rd ed.), esp. ch. 11.
2. **Week 2 — Image CNNs:** fast.ai lessons 1–4. Capstone: `cnn_kramer` jaundice classifier on public neonatal imagery.
3. **Week 3 — Signal CNNs:** 1D CNNs on PhysioNet 2017 AF, PTB-XL, MIT-BIH. Capstone: ECG arrhythmia classifier.
4. **Week 4 — Audio ML:** Mel-spectrograms + CNN. ESC-50, Donate-a-Cry.
5. **Week 5 — Fusion & Calibration:** Multi-modal fusion, Platt/isotonic calibration, Brier score, SHAP, fairness (Fitzpatrick, GA, sex, ethnicity per IEEE 2801), time-series CV. Anchor capstone.

**Anchor project:** Multi-modal neonatal sepsis early-warning (≈ Corvita's ARK-114 alert). Fuses HRV (1D ECG CNN) + temperature variance (tabular) + movement quality (video CNN). Validated against PhysioNet 2019 Sepsis Challenge.

## Environment

- macOS, zsh. Python 3.11.9 at `/usr/local/bin/python3`.
- Only `python3` and `pip3` work — `python` and `pip` are NOT aliased. No Homebrew yet.
- Always write `python3` / `pip3` in commands.

## Repo conventions

- Notebooks: `notebooks/phaseN-topic/weekN-topic.ipynb` (e.g., `week04-pets-classifier.ipynb`).
- Each phase folder has a `NOTES.md` with running observations.
- Each project folder has its own `README.md`, `requirements.txt`, structured code (not just one notebook).
- Commit messages: `phase1-week1: numpy broadcasting practice` or `kramer: baseline ResNet50 architecture`.
- One commit per study session (~30 min minimum).
- Editor: VS Code (with Jupyter and Python extensions). Notebooks open natively in VS Code.
- Workflow: VS Code's built-in Git integration for commits/pushes. Walk through CLI git explicitly when it's needed.

## How to help

- Explain at "coded a little before but not a software engineer" level. Don't assume git, Python tooling, or ML jargon.
- No giant code blocks without explanation — Yejin needs to actually learn this, not have it written for her.
- When she writes broken code, help her understand *why* rather than just fixing it.
- Flag rabbit holes that don't serve the plan.
- Frame as structured professional development, not "fun" or "side projects."
