"""SAS feature pipeline shared by the real-time engine and offline evaluation.

The pipeline keeps a sliding CSI-amplitude buffer, computes the motion channel
on a short (1 s) window and the breathing channel on a long (8 s) window every
100 ms (10 Hz decisions), then feeds both into the CPD state machine.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .config import CSI_FS, DECISION_FS, DetectorConfig
from .detector import CPDDetector
from .dsp import compute_features


class SASPipeline:
    def __init__(self, cfg: DetectorConfig | None = None):
        self.cfg = cfg or DetectorConfig()
        self.detector = CPDDetector(self.cfg)
        self.hist: deque[np.ndarray] = deque(
            maxlen=int(round(self.cfg.breath_window_s * CSI_FS))
        )
        self.frame_count = 0
        self.seq = 0
        self.decision_every = max(1, int(round(CSI_FS / DECISION_FS)))

    def step(self, t: float, amp: np.ndarray, truth: dict | None = None) -> dict | None:
        """Feed one 50 Hz CSI frame; return a decision frame every 100 ms."""
        self.hist.append(np.asarray(amp, dtype=np.float64))
        self.frame_count += 1
        if self.frame_count % self.decision_every:
            return None

        n = len(self.hist)
        motion_n = int(round(self.cfg.motion_window_s * CSI_FS))
        if n < motion_n:
            return None

        A = np.asarray(self.hist)
        features = compute_features(A[-motion_n:], A, self.cfg, fs=CSI_FS)
        det = self.detector
        snap = det.update(t, features, features["peak"])

        self.seq += 1
        peak = features["peak"]
        frame = {
            "seq": self.seq,
            "t": round(float(t), 2),
            "state": snap,
            "features": {
                # decision statistic: residual energy / calibrated background
                "motion": snap["motion_ratio"],
                "motion_energy": round(float(features["energy"]), 4),
                "motion_coherence": round(float(features["coherence"]), 3),
                "interference": bool(features["interference"]),
                "gains": [round(float(g), 4) for g in features["gains"]],
                "bpm_raw": None if peak.get("bpm") is None else round(float(peak["bpm"]), 1),
                "confidence": round(float(peak.get("confidence", 0.0)), 3),
                "height": round(float(peak.get("height", 0.0)), 4),
                "prominence": round(float(peak.get("prominence", 0.0)), 4),
                "lag_s": None if peak.get("lag_s") is None else round(float(peak["lag_s"]), 3),
                "breath_ready": bool(features["breath_ready"]),
            },
            "acf": [round(float(v), 4) for v in features["acf_display"]],
            "acf_step_s": features["acf_step_s"],
            "csi": [round(float(v), 4) for v in amp],
            "truth": _clean_truth(truth),
        }
        return frame


def _clean_truth(truth: dict | None) -> dict | None:
    if truth is None:
        return None
    out = {}
    for k, v in truth.items():
        if isinstance(v, (float, np.floating)):
            out[k] = None if not np.isfinite(v) else round(float(v), 4)
        elif isinstance(v, (bool, np.bool_)):
            out[k] = bool(v)
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        else:
            out[k] = v
    return out
