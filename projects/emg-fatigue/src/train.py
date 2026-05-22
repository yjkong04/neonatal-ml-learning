"""
Train CNN-EMG-Fatigue and save a checkpoint to results/.

Usage:
    python src/train.py                          # defaults
    python src/train.py --epochs 80 --lr 1e-3
    python src/train.py --data-dir data/real     # swap in real data
    python src/train.py --no-generate            # skip dataset generation
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import extract_features, normalize_features
from models import CNNEMGFatigue, EMGDataset, evaluate, train_one_epoch
from simulate import FS, generate_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",    default="data/synthetic",  help="directory with X.npy / y.npy")
    p.add_argument("--results-dir", default="results",         help="where to write checkpoint and stats")
    p.add_argument("--n-samples",   type=int,   default=1000)
    p.add_argument("--epochs",      type=int,   default=60)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--batch",       type=int,   default=32)
    p.add_argument("--patience",    type=int,   default=10)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--no-generate", action="store_true", help="skip dataset generation even if X.npy is missing")
    return p.parse_args()


def load_or_generate(data_dir: Path, n_samples: int, no_generate: bool):
    x_path = data_dir / "X.npy"
    y_path = data_dir / "y.npy"

    if not x_path.exists():
        if no_generate:
            raise FileNotFoundError(f"{x_path} not found and --no-generate was set")
        print(f"Generating {n_samples} synthetic samples → {data_dir}")
        return generate_dataset(n_samples, out_dir=data_dir, seed=42)

    X = np.load(x_path)
    y = np.load(y_path)
    print(f"Loaded {x_path.name}: {X.shape}  {y_path.name}: {y.shape}")
    return X, y


def extract_or_load_det(data_dir: Path, X: np.ndarray):
    det_path = data_dir / "det_features.npy"
    if det_path.exists():
        det = np.load(det_path)
        print(f"Loaded {det_path.name}: {det.shape}")
        return det

    print("Extracting deterministic features (first run — will be cached) ...")
    det = np.zeros((len(X), 5), dtype=np.float32)
    for i, sig in enumerate(X):
        det[i] = extract_features(sig, FS)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(X)}")
    np.save(det_path, det)
    print(f"Saved {det_path}")
    return det


def split_and_normalize(X, det, y, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_train = int(0.8 * len(X))
    tr, va = idx[:n_train], idx[n_train:]

    X_tr, X_va     = X[tr], X[va]
    det_tr, det_va = det[tr], det[va]
    y_tr, y_va     = y[tr], y[va]

    det_tr_n, stats = normalize_features(det_tr)
    det_va_n, _     = normalize_features(det_va, stats)

    return (X_tr, det_tr_n, y_tr), (X_va, det_va_n, y_va), stats


def main():
    args = parse_args()

    data_dir    = ROOT / args.data_dir
    results_dir = ROOT / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    print(f"Device: {device}")

    X, y   = load_or_generate(data_dir, args.n_samples, args.no_generate)
    det    = extract_or_load_det(data_dir, X)
    train_data, val_data, stats = split_and_normalize(X, det, y, args.seed)

    (results_dir / "feat_stats.json").write_text(
        json.dumps({k: v.tolist() for k, v in stats.items()}, indent=2)
    )

    train_loader = DataLoader(EMGDataset(*train_data), batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(EMGDataset(*val_data),   batch_size=args.batch, shuffle=False, num_workers=0)
    print(f"Train: {len(train_data[0])}  Val: {len(val_data[0])}")

    model = CNNEMGFatigue(n_det_features=5, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_path = results_dir / "production-emg-module_best.pt"
    best_val, patience_count = float("inf"), 0

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        vl_loss = evaluate(model, val_loader, device)
        scheduler.step()

        if vl_loss < best_val:
            best_val = vl_loss
            patience_count = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": best_val}, ckpt_path)
        else:
            patience_count += 1

        if epoch % 5 == 0 or patience_count == 0:
            print(f"Epoch {epoch:3d}  train={tr_loss:.5f}  val={vl_loss:.5f}  best={best_val:.5f}")

        if patience_count >= args.patience:
            print(f"Early stop at epoch {epoch}")
            break

    print(f"\nBest val MSE: {best_val:.5f}")
    print(f"Checkpoint  → {ckpt_path}")
    print(f"Feat stats  → {results_dir / 'feat_stats.json'}")

    # Per-output report
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, d, yb in val_loader:
            preds.append(model(x.to(device), d.to(device)).cpu().numpy())
            trues.append(yb.numpy())
    preds = np.vstack(preds)
    trues = np.vstack(trues)

    names = ["fatigue_prob", "resp_effort", "mdf_slope_norm", "spectral_comp", "entropy"]
    print(f"\n{'Output':<22}  {'MSE':>8}  {'MAE':>8}")
    print("-" * 42)
    for name, p, t in zip(names, preds.T, trues.T):
        print(f"{name:<22}  {np.mean((p-t)**2):8.5f}  {np.mean(np.abs(p-t)):8.5f}")


if __name__ == "__main__":
    main()
