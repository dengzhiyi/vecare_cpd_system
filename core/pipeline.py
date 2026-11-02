"""SAS feature pipeline shared by the real-time engine and offline evaluation.

The pipeline keeps a sliding CSI-amplitude buffer, computes the motion channel
on a short (1 s) window and the breathing channel on a long (8 s) window every
100 ms (10 Hz decisions), then feeds both into the CPD state machine.

``fs`` is the incoming CSI frame rate. It defaults to the 50 Hz the simulator
produces; a real capture sets it to the rate implied by its probe period
(48 000 / 960 samples = 50 Hz for the default chirp).

Carrier pruning
---------------
Planning the band from the hardware ceiling (``plan_band``) is the design-time
fix, but a driver's -3 dB point is rarely where its datasheet claims, and the
rolloff is not the only thing that can kill a carrier -- a microphone grille,
phone placement, or a cabin absorber can gut part of the band on one vehicle
and not the next. So the pipeline also checks at runtime: during the quiet
calibration window it measures each carrier's relative jitter and drops the
ones that are an order of magnitude above their peers (``carrier_health``).

Pruning only removes columns from the DSP input. The ``csi`` field written to
each frame keeps every carrier, because a dead carrier is real information --
it is what tells an operator the hardware, not the algorithm, is out of band.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .audio import carrier_health
from .config import CSI_FS, DECISION_FS, DetectorConfig
from .detector import CPDDetector
from .dsp import compute_features


class SASPipeline:
    def __init__(
        self,
        cfg: DetectorConfig | None = None,
        fs: float = CSI_FS,
        auto_prune: bool = True,
    ):
        self.cfg = cfg or DetectorConfig()
        self.fs = float(fs)
        self.detector = CPDDetector(self.cfg)
        self.hist: deque[np.ndarray] = deque(
            maxlen=int(round(self.cfg.breath_window_s * self.fs))
        )
        self.frame_count = 0
        self.seq = 0
        self.decision_every = max(1, int(round(self.fs / DECISION_FS)))

        # --- runtime carrier pruning ---------------------------------------
        self.auto_prune = auto_prune
        self.n_carriers = None            # fixed by the first frame
        self.carrier_mask: np.ndarray | None = None
        self.carrier_jitter: np.ndarray | None = None
        self.prune_threshold = None
        self._calib: list[np.ndarray] = []
        self._calib_frames = int(round(max(2.0, self.cfg.calibration_s) * self.fs))

    def _maybe_prune(self, amp: np.ndarray) -> None:
        """Decide once, from the calibration window, then never again."""
        if not self.auto_prune or self.carrier_mask is not None:
            return
        self._calib.append(amp)
        if len(self._calib) < self._calib_frames:
            return
        alive, jitter, threshold = carrier_health(np.asarray(self._calib))
        self.carrier_jitter = jitter
        self.prune_threshold = threshold
        self.carrier_mask = alive
        self._calib = []          # release the buffer either way

    def step(self, t: float, amp: np.ndarray, truth: dict | None = None) -> dict | None:
        """Feed one CSI frame; return a decision frame every 100 ms."""
        amp = np.asarray(amp, dtype=np.float64)
        if self.n_carriers is None:
            self.n_carriers = int(amp.size)
        self.hist.append(amp)
        self.frame_count += 1
        self._maybe_prune(amp)
        if self.frame_count % self.decision_every:
            return None

        n = len(self.hist)
        motion_n = int(round(self.cfg.motion_window_s * self.fs))
        if n < motion_n:
            return None

        A = np.asarray(self.hist)
        if self.carrier_mask is not None and not self.carrier_mask.all():
            A = A[:, self.carrier_mask]
        features = compute_features(A[-motion_n:], A, self.cfg, fs=self.fs)
        det = self.detector
        snap = det.update(t, features, features["peak"])

        gains = np.asarray(features["gains"], dtype=float)
        if self.carrier_mask is not None and gains.size != amp.size:
            full = np.zeros(amp.size)
            full[self.carrier_mask] = gains
            gains = full

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
                "gains": [round(float(g), 4) for g in gains],
                "bpm_raw": None if peak.get("bpm") is None else round(float(peak["bpm"]), 1),
                "confidence": round(float(peak.get("confidence", 0.0)), 3),
                "height": round(float(peak.get("height", 0.0)), 4),
                "prominence": round(float(peak.get("prominence", 0.0)), 4),
                "lag_s": None if peak.get("lag_s") is None else round(float(peak["lag_s"]), 3),
                "breath_ready": bool(features["breath_ready"]),
                "carriers_used": int(gains.size if self.carrier_mask is None
                                     else self.carrier_mask.sum()),
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
