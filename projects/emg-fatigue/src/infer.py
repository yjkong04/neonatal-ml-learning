"""
Inference wrapper for CNN-EMG-Fatigue.

Public API
----------
EMGInferencer.from_checkpoint(ckpt_path, stats_path) -> EMGInferencer
EMGInferencer.predict(signal)  -> ConceptScores (dataclass)
ConceptBottleneck.score(concepts) -> float  (0-10 clinical distress aggregate)

Example usage:
    inferencer = EMGInferencer.from_checkpoint(...)
    scores = inferencer.predict(raw_emg_window)
    distress_score = ConceptBottleneck().score(scores)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from features import extract_features, normalize_features
from models import CNNEMGFatigue
from simulate import FS

CONCEPT_NAMES = ["fatigue_prob", "resp_effort", "mdf_slope_norm", "spectral_comp", "entropy"]


@dataclass
class ConceptScores:
    fatigue_prob: float     # overall muscle fatigue probability
    resp_effort: float      # respiratory effort amplitude
    mdf_slope_norm: float   # normalised MDF decline rate (0 = no decline, 1 = maximal)
    spectral_comp: float    # spectral compression (low = more fatigued)
    entropy: float          # signal complexity (low = more regular/fatigued)

    def to_array(self) -> np.ndarray:
        return np.array([self.fatigue_prob, self.resp_effort, self.mdf_slope_norm,
                         self.spectral_comp, self.entropy], dtype=np.float32)

    def to_dict(self) -> dict:
        return asdict(self)


class EMGInferencer:
    """Loads a trained checkpoint and runs inference on a single EMG window."""

    def __init__(self, model: CNNEMGFatigue, feat_stats: dict, device: str):
        self.model = model.eval()
        self.feat_stats = feat_stats
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        stats_path: str | Path,
        device: str | None = None,
    ) -> "EMGInferencer":
        if device is None:
            device = (
                "mps"  if torch.backends.mps.is_available() else
                "cuda" if torch.cuda.is_available() else
                "cpu"
            )

        ckpt = torch.load(ckpt_path, map_location=device)
        model = CNNEMGFatigue().to(device)
        model.load_state_dict(ckpt["model_state"])

        raw_stats = json.loads(Path(stats_path).read_text())
        feat_stats = {k: np.array(v, dtype=np.float32) for k, v in raw_stats.items()}

        return cls(model, feat_stats, device)

    @torch.no_grad()
    def predict(self, signal: np.ndarray) -> ConceptScores:
        """
        Args:
            signal: 1D float32 array, 30 s at 1 kHz (30 000 samples)

        Returns:
            ConceptScores with one value per concept, each in [0, 1]
        """
        det = extract_features(signal, FS).reshape(1, -1)
        det_norm, _ = normalize_features(det, self.feat_stats)

        x_t = torch.from_numpy(signal.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        d_t = torch.from_numpy(det_norm).to(self.device)

        out = self.model(x_t, d_t).cpu().numpy().squeeze()
        return ConceptScores(*out.tolist())


class ConceptBottleneck:
    """
    Maps the 5 EMG concept scores to a clinical distress aggregate (0–10).

    Weights derived from clinical reasoning (not learned):
      - fatigue_prob and resp_effort are the strongest distress indicators
      - mdf_slope_norm and spectral_comp are spectral fatigue markers (inverted)
      - entropy: low complexity signals → higher distress (inverted)

    Each concept is mapped to a 0–2 sub-score each weighted 0–2 by severity,
    then summed for a 0–10 total.
    """

    # (weight, invert) — invert=True means low value → high distress
    _CONCEPT_WEIGHTS: list[tuple[float, bool]] = [
        (2.0, False),  # fatigue_prob      → high = bad
        (2.0, False),  # resp_effort       → high = bad
        (2.0, False),  # mdf_slope_norm    → high = bad (normalised decline)
        (2.0, True),   # spectral_comp     → low  = bad (energy shifted down)
        (2.0, True),   # entropy           → low  = bad (regular/fatigued)
    ]

    def score(self, concepts: ConceptScores) -> float:
        """Returns aggregate distress score in [0, 10]. Higher = worse."""
        vals = concepts.to_array()
        total = 0.0
        for v, (w, invert) in zip(vals, self._CONCEPT_WEIGHTS):
            sub = (1.0 - v) if invert else v
            total += w * sub
        return round(float(total), 3)

    def interpret(self, score: float) -> str:
        if score < 3:
            return "minimal distress"
        if score < 5:
            return "mild distress"
        if score < 7:
            return "moderate distress"
        return "severe distress"
