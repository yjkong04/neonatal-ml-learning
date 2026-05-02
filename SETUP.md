# Setup

How to get this repo running on a fresh machine.

## Prerequisites

- Python 3.11+ (this project is developed against 3.11.9)
- `git` (used through VS Code's built-in Source Control panel)
- VS Code with the *Python* and *Jupyter* extensions installed

On macOS without Homebrew, the system Python at `/usr/local/bin/python3` works fine. The commands `python` and `pip` may not be aliased — use `python3` and `pip3` everywhere instead.

## Clone and create a virtual environment

```bash
git clone https://github.com/<your-handle>/neonatal-ml-learning.git
cd neonatal-ml-learning

python3 -m venv .venv
source .venv/bin/activate
```

Once the venv is activated, plain `pip` and `python` will work because the venv puts them on PATH. Outside the venv, keep using `pip3` / `python3`.

## Install dependencies

Phase 1 only needs a small subset:

```bash
pip3 install numpy pandas matplotlib jupyterlab
```

Or install the full stack at once (downloads PyTorch — ~700 MB — which you do not need until Phase 2):

```bash
pip3 install -r requirements.txt
```

Phase-by-phase, you can install incrementally:

| Phase | What to add |
|---|---|
| 1 | `numpy pandas matplotlib jupyterlab` |
| 2 | `torch torchvision fastai scikit-learn` |
| 3 | `wfdb pyhrv wandb` |
| 4 | `librosa soundfile` |
| 5 | `shap captum fairlearn` |

## Running notebooks

**VS Code (primary):** Install the *Python* and *Jupyter* extensions, then open any `.ipynb` file in [notebooks/](notebooks/) and select the `.venv` Python interpreter when prompted. Notebooks run inline with full debugger and Git integration.

**JupyterLab (alternative):** if you'd rather work in a browser tab:

```bash
jupyter lab
```

Notebooks live under [notebooks/](notebooks/), organized by phase.

## Hardware notes

- **Phases 1–2:** Laptop CPU is fine.
- **Phases 3–5:** Training time benefits from a GPU. Options:
  - Local GPU if available
  - [Google Colab](https://colab.research.google.com) (free tier — sufficient for most Phase 3/4 work)
  - [Kaggle Notebooks](https://www.kaggle.com/code) (free GPU hours; PhysioNet datasets are often already mirrored there)

## Datasets

Public datasets only. Data files are gitignored — download fresh into a local `data/` folder, not into the repo.

See [resources/datasets.md](resources/datasets.md) for sources per phase.

## What this repo deliberately doesn't use

- TensorFlow / Keras — PyTorch is the production standard for medical ML and what fast.ai builds on
- Anaconda full distribution — too heavy
- Docker for daily work — would add for production deployment, not for learning
