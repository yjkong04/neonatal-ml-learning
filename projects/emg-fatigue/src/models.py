"""
CNN-EMG-Fatigue — f_CNN architecture.

Parallel branches:
  - Statistical: 1D residual CNN on raw EMG → learned spectral-temporal patterns
  - Deterministic: per-burst features from features.py → interpretable biomarkers

Both fused → 5 concept bottleneck outputs for the clinical assessment scorer.
"""

import torch
import torch.nn as nn


N_DET_FEATURES = 5
N_OUTPUTS = 5  # concept bottleneck outputs


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(x) + self.skip(x))


class CNNEMGFatigue(nn.Module):
    """
    CNN-EMG-Fatigue specialist module.

    Args:
        n_det_features: size of deterministic feature vector (default 5)
        dropout: dropout rate in fusion head

    Inputs:
        x            : [B, 1, T]  raw EMG window (float32, 1 kHz)
        det_features : [B, 5]     normalised deterministic features from features.py

    Output:
        [B, 5] — concept bottleneck values, each in (0, 1) via sigmoid:
            [0] fatigue_prob
            [1] resp_effort
            [2] mdf_slope_norm
            [3] spectral_compression
            [4] entropy
    """

    def __init__(self, n_det_features: int = N_DET_FEATURES, dropout: float = 0.3):
        super().__init__()

        # Statistical branch — 1D ResNet
        # Input: [B, 1, 30000] @ 1 kHz, 30 s
        # After stem (stride 10): [B, 32, 3000]
        # After 3 blocks (stride 4 each): [B, 128, ~47] → pooled to [B, 128]
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=10, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            _ResBlock(32, 64, stride=4),
            _ResBlock(64, 128, stride=4),
            _ResBlock(128, 128, stride=4),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Fusion head
        self.head = nn.Sequential(
            nn.Linear(128 + n_det_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, N_OUTPUTS),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, det_features: torch.Tensor) -> torch.Tensor:
        cnn_out = self.pool(self.res_blocks(self.stem(x))).squeeze(-1)  # [B, 128]
        return self.head(torch.cat([cnn_out, det_features], dim=1))


# ── Training utilities ────────────────────────────────────────────────────────

class EMGDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        X: "np.ndarray",
        det: "np.ndarray",
        y: "np.ndarray",
    ):
        import numpy as np
        self.X = torch.from_numpy(X).unsqueeze(1)   # [N, 1, T]
        self.det = torch.from_numpy(det.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.det[idx], self.y[idx]


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.MSELoss()
    total = 0.0
    for x, det, y in loader:
        x, det, y = x.to(device), det.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x, det)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.MSELoss()
    total = 0.0
    for x, det, y in loader:
        x, det, y = x.to(device), det.to(device), y.to(device)
        pred = model(x, det)
        total += loss_fn(pred, y).item()
    return total / len(loader)
