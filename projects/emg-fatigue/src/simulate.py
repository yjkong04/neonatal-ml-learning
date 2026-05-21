"""
Synthetic neonatal diaphragmatic surface EMG generator.

Fatigue is modelled breath-by-breath: as fatigue level rises, each inspiratory
burst shifts toward lower frequencies and the amplitude envelope changes to
reflect increased neural drive followed by eventual exhaustion.
"""

import numpy as np
import scipy.signal
from pathlib import Path

FS = 1000          # Hz — matches emg-module
WINDOW_S = 30      # seconds per training sample (~20-30 neonatal breaths)
RESP_RATE = 50     # breaths/min (neonatal; 40-60 range)

# Diaphragmatic surface EMG spectral parameters (literature values)
MDF_FRESH = 100    # Hz — median frequency at rest
MDF_EXHAUSTED = 45 # Hz — median frequency at exhaustion
BW = 80            # Hz — bandwidth around MDF


def simulate_window(
    fatigue_start: float,
    fatigue_end: float,
    fs: int = FS,
    window_s: float = WINDOW_S,
    resp_rate: float = RESP_RATE,
    snr_db: float = 22.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate one 30-second synthetic EMG window with a fatigue trajectory.

    fatigue_start / fatigue_end: 0.0 = fresh, 1.0 = exhausted.
    Returns float32 array of shape (window_s * fs,).
    """
    rng = np.random.default_rng(seed)
    n = int(window_s * fs)
    t = np.arange(n) / fs

    fatigue = np.linspace(fatigue_start, fatigue_end, n)

    # MDF shifts linearly from fresh to exhausted value
    mdf = MDF_FRESH - fatigue * (MDF_FRESH - MDF_EXHAUSTED)

    # Amplitude: neural drive increases early to compensate, peaks ~60% fatigue, then falls
    amp = 1.0 + 0.6 * np.sin(np.pi * np.clip(fatigue / 0.65, 0, 1))

    # Build signal in 50ms chunks so we can track time-varying MDF
    chunk = int(0.05 * fs)
    signal = np.empty(n, dtype=np.float64)
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        f0 = mdf[i]
        low, high = max(20.0, f0 - BW / 2), min(499.0, f0 + BW / 2)
        noise = rng.standard_normal(end - i)
        sos = scipy.signal.butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
        seg = scipy.signal.sosfilt(sos, noise)
        seg /= (np.std(seg) + 1e-9)
        signal[i:end] = seg * amp[i]

    # Respiratory envelope: bursts during inspiration (~60% of breath cycle)
    period = 60.0 / resp_rate
    phase = (t % period) / period
    insp = phase < 0.6
    envelope = np.where(insp, np.sin(np.pi * phase / 0.6) ** 0.5, 0.05)
    signal *= envelope

    # Add broadband noise
    sig_pwr = np.mean(signal ** 2) + 1e-12
    noise_pwr = sig_pwr / (10 ** (snr_db / 10))
    signal += rng.standard_normal(n) * np.sqrt(noise_pwr)

    return signal.astype(np.float32)


def make_labels(
    fatigue_start: float,
    fatigue_end: float,
    signal: np.ndarray,
    fs: int = FS,
) -> np.ndarray:
    """
    Ground-truth labels for the 5 concept bottleneck outputs.
    All values normalised to [0, 1].
    """
    mean_fatigue = (fatigue_start + fatigue_end) / 2

    # MDF slope: negative means fatigue, normalise so 0 = fresh, 1 = max decline
    mdf_decline = fatigue_end - fatigue_start   # 0..1
    mdf_slope_norm = mdf_decline                # already 0..1

    # Spectral compression proxy: high/low power ratio decreases with fatigue
    # Use ground-truth MDF end to model it; CNN will learn this from signal
    spectral_comp = 1.0 - fatigue_end * 0.8    # 1 = fresh, 0.2 = exhausted

    # Sample entropy proxy: decreases with fatigue (more regular, periodic)
    entropy_proxy = 1.0 - fatigue_end * 0.7

    return np.array(
        [fatigue_end, mean_fatigue, mdf_slope_norm, spectral_comp, entropy_proxy],
        dtype=np.float32,
    )


# Fatigue trajectory profiles (start, end)
PROFILES = [
    (0.00, 0.05),  # fresh — stable, no fatigue
    (0.00, 0.40),  # mild progression
    (0.10, 0.60),  # moderate
    (0.30, 0.80),  # severe progression
    (0.70, 1.00),  # near-exhaustion
]


def generate_dataset(
    n_samples: int = 1000,
    window_s: float = WINDOW_S,
    fs: int = FS,
    out_dir: Path | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset of synthetic EMG windows.

    Returns:
        X: float32 [n_samples, window_s * fs]
        y: float32 [n_samples, 5]  — concept bottleneck labels
    """
    rng = np.random.default_rng(seed)
    n = int(window_s * fs)
    X = np.empty((n_samples, n), dtype=np.float32)
    y = np.empty((n_samples, 5), dtype=np.float32)

    for i in range(n_samples):
        f_start, f_end = PROFILES[i % len(PROFILES)]
        # Add small random jitter to avoid exact duplicates
        f_start = float(np.clip(f_start + rng.uniform(-0.05, 0.05), 0, 0.95))
        f_end = float(np.clip(f_end + rng.uniform(-0.05, 0.05), f_start, 1.0))
        resp_rate = float(rng.uniform(40, 60))

        sig = simulate_window(
            f_start, f_end, fs, window_s, resp_rate,
            seed=int(rng.integers(0, 2 ** 31)),
        )
        X[i] = sig
        y[i] = make_labels(f_start, f_end, sig, fs)

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n_samples} samples generated")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "X.npy", X)
        np.save(out_dir / "y.npy", y)
        print(f"Saved to {out_dir}")

    return X, y


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/synthetic")
    print(f"Generating 1000 samples → {out}")
    generate_dataset(1000, out_dir=out)
