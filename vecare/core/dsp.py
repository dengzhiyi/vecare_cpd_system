"""Statistical Acoustic Sensing (SAS) primitives.

This module implements the signal-processing core of VeCare:

* per-subcarrier ACF of CSI amplitude after mean/linear-trend removal,
* the motion statistic g(f) = rho(f, small lag) and its equal-weight average,
* Maximal-Ratio Combining (MRC) of subcarrier ACFs weighted by g(f),
* prominence/height/width based breathing peak selection in 10-60 BPM.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import next_fast_len, rfft, irfft
from scipy.signal import butter, sosfiltfilt, find_peaks

from .config import DetectorConfig, MOTION_HIGHPASS_HZ, MRC_WEIGHT_FLOOR


def _detrend(X: np.ndarray) -> np.ndarray:
    """Remove mean and linear trend along axis 0 (time)."""
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    n = X.shape[0]
    if n > 2:
        tt = np.arange(n, dtype=np.float64)
        tt -= tt.mean()
        denom = float(np.dot(tt, tt))
        if denom > 0:
            slope = np.einsum("ij,i->j", X, tt) / denom
            X = X - slope[None, :] * tt[:, None]
    return X


def autocorr(X: np.ndarray, max_lag: int) -> np.ndarray:
    """Unbiased sample ACF for each column of X.

    Returns shape (max_lag + 1, n_columns); lag 0 equals 1 for active columns.
    """
    X = _detrend(X)
    n = X.shape[0]
    if n < 3:
        out = np.zeros((max_lag + 1, X.shape[1]))
        out[0] = 1.0
        return out
    nfft = next_fast_len(2 * n)
    F = rfft(X, nfft, axis=0)
    power = irfft(F * np.conj(F), nfft, axis=0)[:max_lag + 1]
    denom = np.array([(n - lag) for lag in range(max_lag + 1)], dtype=np.float64)
    power = power / denom[:, None]
    norm = power[0]
    norm = np.where(norm > 1e-12, norm, 1.0)
    return power / norm[None, :]


def motion_statistic(acf_motion: np.ndarray, lag_samples: int) -> np.ndarray:
    """g(f): first usable ACF tap for every subcarrier (Eq. 4)."""
    lag = int(np.clip(lag_samples, 1, acf_motion.shape[0] - 1))
    return acf_motion[lag, :]


def motion_highpass(X: np.ndarray, fs: int, cutoff_hz: float = MOTION_HIGHPASS_HZ) -> np.ndarray:
    """Suppress breathing (<1 Hz) so body motion gets a separate statistic."""
    if X.shape[0] < 12:
        return X
    sos = butter(2, cutoff_hz / (fs / 2), btype="high", output="sos")
    return sosfiltfilt(sos, X, axis=0)


def combine_acf(acf: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """MRC combination of subcarrier ACFs, Eq. (6)."""
    w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0) + MRC_WEIGHT_FLOOR
    total = w.sum()
    if total < 1e-9:
        w = np.ones_like(w) / len(w)
    else:
        w = w / total
    return acf @ w


def _parabolic_peak(x: np.ndarray, idx: int) -> float:
    """Sub-sample peak location using a 3-point parabola."""
    if idx <= 0 or idx >= len(x) - 1:
        return float(idx)
    y0, y1, y2 = x[idx - 1], x[idx], x[idx + 1]
    denom = y0 - 2 * y1 + y2
    if abs(denom) < 1e-12:
        return float(idx)
    return float(idx + 0.5 * (y0 - y2) / denom)


def find_breathing_peak(
    combined: np.ndarray,
    fs: int,
    cfg: DetectorConfig,
    window_samples: int | None = None,
) -> dict:
    """Select a plausible breathing peak from the combined ACF.

    Returns lag, bpm, confidence, peak height, prominence and validity.
    """
    lag_min = int(round(cfg.breath_lag_min_s * fs))
    # Unbiased ACF becomes unreliable when less than ~40 % of the window
    # overlaps; cap the search lag accordingly.
    nwin = window_samples if window_samples is not None else len(combined)
    lag_max = min(
        int(round(cfg.breath_lag_max_s * fs)),
        int(0.62 * nwin),
        len(combined) - 2,
    )
    if lag_max <= lag_min + 2:
        return {"valid": False, "bpm": None, "confidence": 0.0,
                "lag_s": None, "height": 0.0, "prominence": 0.0}

    seg = combined[lag_min:lag_max + 1]
    peaks, props = find_peaks(
        seg,
        height=cfg.breath_min_height,
        prominence=cfg.breath_min_prominence,
        distance=max(2, int(0.4 * fs)),
    )
    if len(peaks) == 0:
        return {"valid": False, "bpm": None, "confidence": 0.0,
                "lag_s": None, "height": float(np.max(seg)) if len(seg) else 0.0,
                "prominence": 0.0}

    heights = props["peak_heights"]
    proms = props["prominences"]
    best = int(np.argmax(heights * (0.5 + proms)))
    # Prefer the earliest plausible (fundamental) peak whose strength is close
    # to the strongest one; harmonics of breathing appear at 2x/3x cycle.
    keep = [i for i in range(len(peaks))
            if heights[i] >= heights[best] * 0.82 and proms[i] >= proms[best] * 0.65]
    chosen = keep[0] if keep else best

    peak_idx = lag_min + peaks[chosen]
    fine = lag_min + _parabolic_peak(seg, int(peaks[chosen]))
    lag_s = fine / fs
    bpm = 60.0 / lag_s
    height = float(heights[chosen])
    prominence = float(proms[chosen])

    conf = 0.55 * min(1.0, height / 0.35) + 0.45 * min(1.0, prominence / 0.25)
    # Add a small bonus when the peak is broad (a real breathing cycle is
    # distributed over adjacent lags rather than a single noisy spike).
    width = props.get("widths", np.zeros(len(peaks)))[chosen]
    conf += 0.06 * min(1.0, float(width) / (0.5 * fs))
    conf = float(np.clip(conf, 0.0, 1.0))

    return {
        "valid": bool(bpm >= 10.0 and bpm <= 60.0),
        "bpm": float(bpm) if bpm >= 10.0 and bpm <= 60.0 else None,
        "confidence": conf,
        "lag_s": float(lag_s),
        "height": height,
        "prominence": prominence,
        "peak_idx": int(peak_idx),
    }


def compute_features(
    amp_motion: np.ndarray,
    amp_breath: np.ndarray,
    cfg: DetectorConfig,
    fs: int = 50,
) -> dict:
    """Compute SAS features from sliding CSI amplitude windows.

    amp_motion: (N_m, S) amplitudes for the motion window (e.g. 2 s).
    amp_breath: (N_b, S) amplitudes for the breathing window (e.g. 8 s).
    """
    lag_m = int(round(cfg.motion_lag_s * fs))
    max_lag_m = max(lag_m + 1, int(round(cfg.motion_window_s * fs * 0.5)))
    amp_motion_hp = motion_highpass(amp_motion, fs)
    acf_m = autocorr(amp_motion_hp, max_lag_m)
    gains = motion_statistic(acf_m, lag_m)
    gbar = float(np.mean(gains))

    if amp_breath.shape[0] < int(0.94 * cfg.breath_window_s * fs):
        return {
            "gains": gains.astype(float).tolist(),
            "motion": gbar,
            "peak": {"valid": False, "bpm": None, "confidence": 0.0,
                     "lag_s": None, "height": 0.0, "prominence": 0.0},
            "acf_display": [],
            "acf_step_s": 0.04,
            "breath_ready": False,
        }

    max_lag_b = min(int(round(cfg.breath_lag_max_s * fs)), amp_breath.shape[0] - 2)
    acf_b = autocorr(amp_breath, max_lag_b)
    weights = np.maximum(gains, 0.0)
    combined = combine_acf(acf_b, weights)

    peak = find_breathing_peak(combined, fs, cfg, amp_breath.shape[0])

    step = max(1, int(round(0.04 * fs)))
    acf_display = combined[::step].astype(float).tolist()

    return {
        "gains": gains.astype(float).tolist(),
        "motion": gbar,
        "peak": peak,
        "acf_display": acf_display,
        "acf_step_s": step / fs,
        "breath_ready": True,
    }
