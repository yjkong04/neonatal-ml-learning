# Tooling & Setup

How to get this repo running locally. Stack chosen for production relevance, not novelty.

---

## Environment

Python 3.11+. I recommend `uv` for environment management — it's fast, reproducible, and what most modern Python work uses now.

```bash
# Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up
git clone <this-repo-url>
cd <repo>
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

If you prefer `conda` or plain `venv + pip`, that works too — `requirements.txt` is standard.

---

## Core libraries

| Category | Libraries |
|---|---|
| Numerical | `numpy`, `scipy`, `pandas` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Classical ML | `scikit-learn`, `xgboost` |
| Deep learning | `torch`, `torchvision`, `fastai` |
| Signals | `wfdb`, `librosa`, `pyhrv` |
| Explainability | `shap`, `captum` |
| Fairness | `fairlearn` |
| Tracking | `wandb` (Phase 3+) |

---

## Hardware

Most of Phase 1 and 2 runs fine on a laptop CPU. From Phase 3 onward, training time benefits from a GPU — options:

- Local GPU if available
- Google Colab (free tier — sufficient for most Phase 3/4 work)
- Kaggle Notebooks (free GPU hours, especially good for PhysioNet datasets that are already mirrored there)
- Paperspace, Lambda Labs, or Vast.ai for longer training runs

---

## Experiment tracking

Starting Phase 3, experiments are tracked in Weights & Biases (free tier). This becomes essential once there's more than one model run to compare. The free tier covers all educational work.

---

## Repo conventions

- One notebook per checkpoint, named `weekN-topic.ipynb` (e.g., `week04-pets-classifier.ipynb`)
- Each phase folder has a `NOTES.md` with running observations and questions
- Each project folder has its own `README.md`, `requirements.txt`, and structured code (not just a single notebook)
- Commits are small and frequent — at minimum one per study session
- Commit messages follow the pattern: `phase2-week5: implement data augmentation pipeline`

---

## What I am NOT using

- TensorFlow / Keras — PyTorch is the production standard for medical ML and what fast.ai builds on
- Anaconda full distribution — too heavy; `uv` + minimal env is faster
- Docker for daily work — would add for production deployment, not for learning
