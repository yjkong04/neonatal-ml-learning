# Context for Claude Code — Data Science Learning Journey

## Who I am and what I do

I'm Yejin Kong, working at Corvita Biomedical (Toronto). My role is non-technical, but I'm transitioning toward technical contributions on the team that builds the ARK Incubator — a portable, edge-native AI neonatal life support platform that consolidates 13+ critical NICU devices into a single bedside unit. The AI engine inside it is called NOA-AI.

I'm picking up data science skills to (1) demonstrate to my boss that I can do technical work, (2) become more useful to the team long-term, and (3) open up future opportunities (big tech, broader ML roles).

## What Corvita is building (relevant context)

**ARK Incubator** — hardware platform, all-in-one NICU bedside device.

**NOA-AI** — the inference engine. Multi-modal sensor fusion system, not a single CNN. Architecture:
- Specialist models (CNNs) process individual modalities — image, video, 1D signals, audio
- Rule-based scorers compute clinical scores (Sarnat, Bhutani, nSOFA, HeRO, APGAR)
- Fusion engine combines specialist outputs + scorer outputs into final clinical alerts
- Narrative layer generates explanations and alarm commands

**Current state of NOA-AI:**
- Platform/orchestration: ~68% complete (gateway, quality gate, feature assembly, fusion engine, narrative layer all real)
- Specialist CNN portfolio: ~14% complete (most CNNs are wrappers without real models)
- The training framework, taxonomy, importers, and per-CNN folders are ready
- Bedside hot path still depends on a demo simulator instead of the real specialist→scorer→fusion graph
- Frontend UI exists for live ARK monitoring; currently uses real camera + public datasets (ECG/EEG) since not all sensors are integrated yet

**Target CNNs the company is building** (in priority order):
1. `cnn_skin` — RGB image, skin color/perfusion classification
2. `cnn_kramer` — RGB image, Kramer zone jaundice classification ← my project anchor
3. `cnn_resp` — 1D respiratory waveform or video classification
4. `cnn_cardio` — 1D ECG/PPG waveform classification

Other CNNs in the pipeline: `cnn_neurovision`, `cnn_retraction`, `cnn_thermal_mottling`, `cnn_abdominal`, `cnn_pulse_pressure`, `cnn_precordial`, `cnn_emg_fatigue`, `cnn_cry`, `cnn_pphn`. The only specialist CNN currently runtime-capable is `cnn_eeg`.

**CORVITA data is proprietary.** I will never push internal Corvita data, code, or documents to any public repo. Everything I do publicly is on public datasets (PhysioNet, ISIC, Kaggle releases) or self-generated synthetic data.

## Active goal: ship the Kramer classifier project by end of next week

My boss wants concrete proof by end of next week that what I'm learning translates to real technical capability. Plan revised accordingly.

**The project:** Kramer Zone Jaundice Classifier — a prototype image classifier for neonatal jaundice severity, trained on public data, using the same transfer-learning techniques that `cnn_kramer` will eventually use in production.

**What it is NOT:**
- A production-ready model
- Validated on real neonatal jaundice data
- Ready for clinical deployment
- A fairness-audited system

**What it IS:**
- A working trained image classifier
- A learning project that demonstrates I understand the full ML pipeline
- A prototype of the technique that `cnn_kramer` will use
- A clean, well-documented, demoable project

**Honest framing for the demo:** "I built a prototype image classifier as a learning project, using public skin imagery as a stand-in for what `cnn_kramer` would eventually classify. The model architecture and training pipeline are the same techniques we'd use for the real Corvita version — the main difference is the training data."

## Compressed study plan (revised)

The original 16-week plan is still the long-term roadmap (see STUDY_PLAN.md). For the next 7-8 days, I'm running a compressed version that front-loads enough foundations to ship Kramer, then resumes the regular schedule afterward.

**Days 1-2:** McKinney chapters 6 (loading) and 7 (cleaning) — fast skim, the high-value parts. Start fast.ai Lesson 1 in parallel (trains a pet classifier in 20 min; warm-up).

**Day 3:** Fast.ai Lesson 2 (data augmentation, training loop intuition). Set up project scaffold for `kramer-classifier`. Find and download dataset.

**Day 4:** Load and explore dataset. Build baseline model using fast.ai high-level API. Get *some* end-to-end training run to complete, even if results are bad.

**Day 5:** Improve model. Try different backbones (ResNet18, ResNet50, EfficientNet). Tune learning rate. Add augmentation. Track what helps.

**Day 6:** Evaluation. Confusion matrix, per-class accuracy, calibration plot if time. Generate visualizations.

**Day 7:** Demo notebook + README + final polish. Write the honest framing for what this is and isn't.

**Day 8 (buffer):** Fix anything broken.

**After the demo:** Resume the original Phase 1 schedule. Pick up McKinney chapters 8, 10, 11 (wrangling, groupby, time series) and continue with fast.ai Lessons 3-4. The compressed sprint doesn't replace the foundations work — it front-loads a deliverable so I can prove the trajectory while still doing the real learning underneath.

## Dataset for Kramer

Public neonatal jaundice imagery is sparse. Three approaches in order of preference:

**A) Use existing public neonatal jaundice data** — small Kaggle/academic releases. Pros: on-topic. Cons: small datasets, more curation.

**B) ISIC skin lesion dataset as proxy** — same technique (transfer learning on dermatology imagery), much larger data. Pros: well-curated, plentiful. Cons: adult skin lesions, not neonatal jaundice. Framing: "I learned the technique that applies to `cnn_kramer`."

**C) General skin tone / color classifier** — easiest technically, furthest from `cnn_kramer`.

Lean A if usable data exists; fall back to B.

## My GitHub repo

Public repo: `neonatal-ml-learning`. Working from that folder.

**Repo structure:**
```
.
├── README.md
├── STUDY_PLAN.md
├── CLAUDE.md                    ← this file
├── SETUP.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── notebooks/
│   ├── phase1-foundations/      with NOTES.md, ch04 + ch05 notebooks done
│   ├── phase2-image-cnns/       with NOTES.md
│   ├── phase3-signal-cnns/      with NOTES.md
│   ├── phase4-audio-ml/         with NOTES.md
│   └── phase5-fusion-calibration/  with NOTES.md
├── projects/
│   ├── kramer-classifier/       ← active project, Phase 2 capstone (compressed)
│   ├── ecg-arrhythmia/          future, Phase 3 capstone
│   └── sepsis-early-warning/    future, anchor project
└── resources/
    ├── reading-list.md
    ├── datasets.md
    └── tooling.md
```

**Workflow:**
- Using GitHub Desktop or CLI git (still learning CLI)
- One commit per study session minimum (~30 min)
- Commit message format: `phase1-week1: chapter 4 numpy practice` or `kramer: baseline ResNet50`
- Never commit data files — `.gitignore` handles this
- Public datasets only

**What's been completed so far:**
- Repo scaffold up and on GitHub
- McKinney Chapter 4 (NumPy) — notebook with notes, committed
- McKinney Chapter 5 (pandas intro) — notebook with notes, committed
- Python 3.11.9 working, pip is `pip3` on this system, jupyterlab installed

## My environment

**Machine:** MacBook Pro, macOS, zsh shell.

**Python:** 3.11.9, at `/usr/local/bin/python3`. Only `python3` and `pip3` work — `python` and `pip` are not aliased.

**No Homebrew installed** — fine for now.

**Editor:** VS Code with Jupyter extension.

**Already installed:**
- numpy, pandas, matplotlib, jupyterlab

**To install for Kramer project:**
- torch, torchvision (deep learning)
- fastai (high-level wrapper for fast initial training)
- Pillow (image handling)
- scikit-learn (evaluation metrics)

**Stack for full plan (in requirements.txt):**
- Numerical: `numpy`, `scipy`, `pandas`
- Visualization: `matplotlib`, `seaborn`, `plotly`
- Classical ML: `scikit-learn`, `xgboost`
- Deep learning: `torch`, `torchvision`, `torchaudio`, `fastai`
- Signals (later): `wfdb`, `librosa`, `pyhrv`
- Explainability (later): `shap`, `captum`, `fairlearn`
- Tracking (later): `wandb`
- Notebook: `jupyterlab`, `ipywidgets`

## What I need from Claude Code

1. **Pacing me through the compressed plan** — flagging when I'm behind, when I'm fine, when I'm rabbit-holing
2. **Teaching CNN concepts as they come up** — explaining what's happening when I train models, not just writing code for me
3. **Scaffolding the Kramer project** — proper structure, README, design doc, module layout
4. **Debugging help** — when something doesn't work, walking me through *why* before fixing it
5. **Honest feedback on my model results** — telling me if accuracy is suspicious, if I'm overfitting, if my evaluation is wrong
6. **Help with the demo framing** — when it comes time to write the README and present to my boss, helping me frame results honestly without overclaiming
7. **Git workflow help** — I'm learning CLI git, still rely on GitHub Desktop sometimes
8. **Foundations review** — quick conceptual explanations of NumPy/pandas concepts when they come up in real code, since I'm still consolidating chapters 4-5

## What NOT to do

- Don't push generic Kaggle competitions (Titanic, housing). The plan deliberately skips those.
- Don't let me skip understanding to "ship faster" — I want to be able to explain every line of my code to my boss. If I can't explain it, I haven't learned it.
- Don't assume I know Python tooling, ML jargon, or git deeply. Explain things at the level of "coded a little before, not a software engineer."
- Don't write giant code blocks without explaining what's happening section by section.
- Don't suggest I include Corvita-internal data, code, or documents in the public repo. Ever.
- Don't reframe my learning as a side project or hobby. This is structured professional development with a deadline and a real audience.
- Don't help me overclaim the Kramer project. It's a learning prototype, not a deployable model. The honest framing is the whole point.

## Today

I just finished McKinney chapters 4 and 5 in one day (ahead of original Phase 1 pace). All notebooks pushed. Ready to start the Kramer project scaffold and begin Day 1-2 of the compressed plan.

Next concrete steps:
1. Scaffold `projects/kramer-classifier/` folder structure
2. Start McKinney Chapter 6 (data loading) and Chapter 7 (cleaning) in parallel
3. Start fast.ai Lesson 1 — train the pet classifier as a warm-up
4. Find and evaluate dataset options (Kaggle search + academic links)

After the Kramer demo lands, return to the full Phase 1 schedule (chapters 8, 10, 11 of McKinney) and continue fast.ai Lessons 3-4.
