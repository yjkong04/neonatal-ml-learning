"""
Deterministic feature extraction for CNN-EMG-Fatigue.

All functions operate on a single EMG window (1D numpy array).
extract_features() returns the 5-element vector fed to the CNN fusion layer.
"""

import numpy as np
import scipy.signal
from typing import Dict, List, Optional, Tuple


# ── Burst segmentation ────────────────────────────────────────────────────────

def find_bursts(
    signal: np.ndarray,
    fs: int,
    resp_rate_hz: float = 50 / 60,
    min_burst_s: float = 0.2,
) -> List[Tuple[int, int]]:
    """
    Detect inspiratory burst boundaries using the rectified EMG envelope.
    Returns list of (start_sample, end_sample) pairs.
    """
    # Rectify and smooth to get the amplitude envelope
    env = np.abs(signal)
    kernel = int(0.05 * fs)  # 50ms smoothing
    env = np.convolve(env, np.ones(kernel) / kernel, mode="same")

    threshold = np.percentile(env, 40)
    above = env > threshold

    # Find contiguous regions
    bursts = []
    in_burst = False
    start = 0
    min_len = int(min_burst_s * fs)

    for i, val in enumerate(above):
        if val and not in_burst:
            start = i
            in_burst = True
        elif not val and in_burst:
            if i - start >= min_len:
                bursts.append((start, i))
            in_burst = False

    if in_burst and len(signal) - start >= min_len:
        bursts.append((start, len(signal)))

    return bursts


# ── Per-burst spectral features ───────────────────────────────────────────────

def burst_mdf(burst: np.ndarray, fs: int) -> float:
    """Median frequency of power spectrum for one burst."""
    freqs, psd = scipy.signal.periodogram(burst, fs=fs)
    mask = (freqs >= 20) & (freqs <= 500)
    freqs, psd = freqs[mask], psd[mask]
    cumulative = np.cumsum(psd)
    if cumulative[-1] == 0:
        return 0.0
    return float(freqs[np.searchsorted(cumulative, cumulative[-1] / 2)])


def burst_rms(burst: np.ndarray) -> float:
    return float(np.sqrt(np.mean(burst ** 2)))


def burst_spectral_compression(burst: np.ndarray, fs: int, split_hz: float = 100.0) -> float:
    """High/low power ratio. Decreases as fatigue shifts energy to lower frequencies."""
    freqs, psd = scipy.signal.periodogram(burst, fs=fs)
    mask = (freqs >= 20) & (freqs <= 500)
    freqs, psd = freqs[mask], psd[mask]
    low = psd[freqs < split_hz].sum()
    high = psd[freqs >= split_hz].sum()
    return float(high / (low + 1e-10))


# ── Window-level deterministic features ──────────────────────────────────────

def _sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Approximate sample entropy (lower = more regular/fatigued)."""
    r = r_factor * np.std(x)
    if r == 0:
        return 0.0
    n = len(x)
    # Use a downsampled version for speed
    x = x[::4]
    n = len(x)
    if n < m + 2:
        return 0.0

    def _count_matches(template_len):
        count = 0
        for i in range(n - template_len):
            for j in range(i + 1, n - template_len):
                if np.max(np.abs(x[i:i+template_len] - x[j:j+template_len])) < r:
                    count += 1
        return count

    B = _count_matches(m)
    A = _count_matches(m + 1)
    if B == 0:
        return 0.0
    return float(-np.log(A / B + 1e-10))


def extract_features(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Extract the 5 deterministic features for the CNN fusion layer.

    Returns float32 array [5]:
        [0] mean_rms           — mean burst RMS (respiration effort amplitude)
        [1] mdf_slope          — linear slope of per-burst MDF over time (negative = fatigue)
        [2] spectral_comp      — mean high/low power ratio across bursts
        [3] rms_trend          — ratio of late-window RMS to early-window RMS
        [4] entropy            — sample entropy of burst envelope
    """
    bursts = find_bursts(signal, fs)

    if len(bursts) < 3:
        # Not enough bursts found — return zeros (Module A quality gating handles this)
        return np.zeros(5, dtype=np.float32)

    mdf_vals = np.array([burst_mdf(signal[s:e], fs) for s, e in bursts])
    rms_vals = np.array([burst_rms(signal[s:e]) for s, e in bursts])
    comp_vals = np.array([burst_spectral_compression(signal[s:e], fs) for s, e in bursts])

    mean_rms = float(np.mean(rms_vals))

    # MDF slope (Hz per burst — negative means fatigue)
    x = np.arange(len(mdf_vals), dtype=float)
    mdf_slope = float(np.polyfit(x, mdf_vals, 1)[0])

    spectral_comp = float(np.mean(comp_vals))

    # RMS trend: late 25% vs early 25%
    quarter = max(1, len(rms_vals) // 4)
    rms_trend = float(np.mean(rms_vals[-quarter:]) / (np.mean(rms_vals[:quarter]) + 1e-9))

    # Sample entropy on the burst envelope
    envelope = np.array([burst_rms(signal[s:e]) for s, e in bursts])
    entropy = _sample_entropy(envelope)

    return np.array([mean_rms, mdf_slope, spectral_comp, rms_trend, entropy], dtype=np.float32)


def normalize_features(X_feats: np.ndarray, stats: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
    """
    Z-score normalize a [N, 5] feature matrix.
    Pass stats from the training set to normalize val/test consistently.
    """
    if stats is None:
        stats = {"mean": X_feats.mean(0), "std": X_feats.std(0) + 1e-8}
    return (X_feats - stats["mean"]) / stats["std"], stats
