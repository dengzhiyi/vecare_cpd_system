"""Statistical Acoustic Sensing (SAS) primitives.

This module implements the signal-processing core of VeCare:

* per-subcarrier ACF of CSI amplitude after mean/linear-trend removal,
* the motion channel: detrended residual energy + inter-subcarrier coherence,
  plus the paper's g(f) = rho(f, 1/Fs) gain vector (Eq. 4),
* Maximal-Ratio Combining (MRC) of subcarrier ACFs weighted by g(f) (Eq. 6),
* prominence/height/width based breathing peak selection in 10-66 BPM,
* the impulse-noise ("linear decrease then increase") ACF pattern detector
  from §5 of the paper.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import next_fast_len, rfft, irfft
from scipy.signal import find_peaks

from .config import DetectorConfig, INTERFERENCE_DEPTH, MRC_WEIGHT_FLOOR


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


# ---------------------------------------------------------------------------
# motion channel
# ---------------------------------------------------------------------------
def residual_energy(X: np.ndarray) -> float:
    """Detrended dynamic energy of a short CSI window.

    Over 1 s a breathing excursion is almost a straight line, so detrending
    removes most of it while a limb movement (a bump inside the window) leaves
    a large residual.
    """
    R = _detrend(X)
    if R.size == 0:
        return 0.0
    return float(np.mean(R.std(axis=0)))


def subcarrier_coherence(X: np.ndarray) -> float:
    """Share of residual variance explained by the first singular vector.

    A body displaces every multipath by the same delay, so the per-subcarrier
    residuals stay correlated (kappa well above the noise floor). Ambient
    noise bursts hit subcarriers independently and stay at the floor.
    """
    R = _detrend(X)
    if R.shape[0] < 4 or R.shape[1] < 2:
        return 0.0
    R = R - R.mean(axis=0, keepdims=True)
    try:
        s = np.linalg.svd(R, compute_uv=False)
    except np.linalg.LinAlgError:
        return 0.0
    total = float((s ** 2).sum())
    if total < 1e-12:
        return 0.0
    return float(min(1.0, s[0] ** 2 / total))


def interference_pattern(acf_mean: np.ndarray, depth: float = INTERFERENCE_DEPTH) -> bool:
    """Impulse-noise signature from §5: the ACF decreases then increases.

    A horn beep or door slam injects a short burst; its ACF is a triangle, i.e.
    monotone down to a trough and monotone back up. Breathing and smooth body
    motion do not produce that V shape within the first ~0.5 s of lag.
    """
    n = min(len(acf_mean), 26)
    if n < 8:
        return False
    seg = np.asarray(acf_mean[:n], dtype=np.float64)
    k = int(np.argmin(seg[1:-1])) + 1
    if k < 2 or k > n - 3:
        return False
    left, right = seg[1:k + 1], seg[k:n]
    if left.size < 2 or right.size < 2:
        return False
    if not np.all(np.diff(left) <= 0.02) or not np.all(np.diff(right) >= -0.02):
        return False
    return float(min(seg[0], seg[-1]) - seg[k]) > depth


def motion_features(
    X: np.ndarray,
    fs: int,
    lag_s: float,
    context: np.ndarray | None = None,
) -> dict:
    """Motion channel features.

    X carries the short (responsive) window used for energy and for the
    impulse-pattern check. ``context`` optionally supplies a longer window used
    only for the coherence estimate, which is a second-order statistic and is
    much more stable over 2 s than over 1 s.
    """
    lag = max(1, int(round(lag_s * fs)))
    max_lag = max(lag + 2, int(round(0.5 * X.shape[0])))
    acf = autocorr(X, max_lag)
    gains = motion_statistic(acf, lag)
    acf_mean = acf.mean(axis=1)
    coh_win = X if context is None else context
    return {
        "energy": residual_energy(X),
        "coherence": subcarrier_coherence(coh_win),
        "gains": gains,
        "g_mean": float(np.mean(gains)),
        "interference": interference_pattern(acf_mean),
    }


# ---------------------------------------------------------------------------
# breathing channel
# ---------------------------------------------------------------------------
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


def _descend_to_fundamental(idx: np.ndarray, heights: np.ndarray, depth: int = 4) -> int:
    """Pick the fundamental peak, not its harmonic.

    A periodic signal peaks at tau_b, 2*tau_b, 3*tau_b with nearly equal
    height, so "the strongest peak" can be any of them and "the earliest peak"
    can be a slow sway unrelated to breathing. Walk down instead: if a
    comparable peak exists near lag/2, the current one is a harmonic.
    """
    best = int(np.argmax(heights))
    for _ in range(depth):
        half = float(idx[best]) / 2.0
        if half < idx[0]:
            break
        near = [i for i in range(len(idx)) if abs(idx[i] - half) <= 0.18 * half]
        if not near:
            break
        cand = max(near, key=lambda i: heights[i])
        if heights[cand] < 0.55 * heights[best]:
            break
        best = cand
    return best


_INVALID_PEAK = {"valid": False, "bpm": None, "confidence": 0.0,
                 "lag_s": None, "height": 0.0, "prominence": 0.0}


def find_breathing_peak(
    combined: np.ndarray,
    fs: int,
    cfg: DetectorConfig,
    window_samples: int | None = None,
) -> dict:
    """Select a plausible breathing peak from the combined ACF.

    Peaks are *searched* only inside the 10-66 BPM lag range, but prominence and
    height are measured on the full ACF starting at lag 1. Slicing the array at
    lag_min first would truncate the first peak's left trough and make its
    prominence half of a harmonic's, which silently doubles the reported rate.
    """
    lag_min = int(round(cfg.breath_lag_min_s * fs))
    nwin = window_samples if window_samples is not None else len(combined)
    lag_max = min(
        int(round(cfg.breath_lag_max_s * fs)),
        int(cfg.breath_max_overlap * nwin),
        len(combined) - 2,
    )
    if lag_max <= lag_min + 2:
        return dict(_INVALID_PEAK)

    search = combined[1:lag_max + 2]
    peaks, props = find_peaks(
        search,
        height=cfg.breath_min_height,
        prominence=cfg.breath_min_prominence,
        distance=max(2, int(0.4 * fs)),
    )
    n_found = len(peaks)
    idx = peaks + 1
    keep_range = (idx >= lag_min) & (idx <= lag_max)
    idx = idx[keep_range]
    heights = props["peak_heights"][keep_range]
    proms = props["prominences"][keep_range]
    widths_all = props.get("widths")
    if widths_all is None or len(widths_all) != n_found:
        widths_all = np.zeros(n_found)
    widths = widths_all[keep_range]

    if len(idx) == 0:
        seg = combined[lag_min:lag_max + 1]
        return {"valid": False, "bpm": None, "confidence": 0.0, "lag_s": None,
                "height": float(np.max(seg)) if len(seg) else 0.0,
                "prominence": 0.0}

    best = int(np.argmax(heights * (0.5 + proms)))
    # Prefer the earliest plausible (fundamental) peak whose strength is close
    # to the strongest one; harmonics of breathing appear at 2x/3x cycle.
    # A periodic signal peaks at tau_b, 2*tau_b, 3*tau_b with nearly equal
    # height; the fundamental is the earliest one. Prominence must NOT be used
    # as a relative gate here: an interior harmonic always spans a full trough
    # while the first peak only spans half of one.
    keep = [i for i in range(len(idx)) if heights[i] >= heights[best] * 0.85]
    chosen = keep[0] if keep else best

    fine = _parabolic_peak(combined, int(idx[chosen]))
    lag_s = fine / fs
    bpm = 60.0 / lag_s if lag_s > 0 else 0.0
    height = float(heights[chosen])
    prominence = float(proms[chosen])

    conf = 0.55 * min(1.0, height / 0.35) + 0.45 * min(1.0, prominence / 0.25)
    # A real breathing cycle is distributed over adjacent lags rather than a
    # single noisy spike, so give broad peaks a small bonus.
    conf += 0.06 * min(1.0, float(widths[chosen]) / (0.5 * fs))
    conf = float(np.clip(conf, 0.0, 1.0))

    valid = bool(10.0 <= bpm <= 75.0)
    return {
        "valid": valid,
        "bpm": float(bpm) if valid else None,
        "confidence": conf,
        "lag_s": float(lag_s),
        "height": height,
        "prominence": prominence,
        "peak_idx": int(idx[chosen]),
    }


def compute_features(
    amp_motion: np.ndarray,
    amp_breath: np.ndarray,
    cfg: DetectorConfig,
    fs: int = 50,
) -> dict:
    """Compute SAS features from sliding CSI amplitude windows.

    amp_motion: (N_m, S) amplitudes for the motion window (e.g. 1 s).
    amp_breath: (N_b, S) amplitudes for the breathing window (e.g. 8 s); the
    trailing part of it doubles as the coherence context window.
    """
    coh_n = int(round(cfg.motion_coherence_window_s * fs))
    context = amp_breath[-coh_n:] if amp_breath.shape[0] >= coh_n else amp_breath
    motion = motion_features(amp_motion, fs, cfg.motion_lag_s, context)

    ready_n = int(0.94 * cfg.breath_window_s * fs)
    if amp_breath.shape[0] < ready_n:
        return {
            "gains": motion["gains"].astype(float).tolist(),
            "energy": motion["energy"],
            "coherence": motion["coherence"],
            "interference": motion["interference"],
            "peak": dict(_INVALID_PEAK),
            "acf_display": [],
            "acf_step_s": 0.04,
            "breath_ready": False,
        }

    max_lag_b = min(
        int(round(cfg.breath_lag_max_s * fs)),
        int(round(cfg.breath_window_s * fs)) - 2,
    )
    acf_b = autocorr(amp_breath, max_lag_b)
    # MRC weights are the subcarriers' own gains over the breathing window
    # (Eq. 4 on the raw amplitude), i.e. how much target energy each one sees.
    weights = np.maximum(motion_statistic(acf_b, max(1, int(round(cfg.motion_lag_s * fs)))), 0.0)
    combined = combine_acf(acf_b, weights)

    peak = find_breathing_peak(combined, fs, cfg, amp_breath.shape[0])

    step = max(1, int(round(0.04 * fs)))
    acf_display = combined[::step].astype(float).tolist()

    return {
        "gains": weights.astype(float).tolist(),
        "energy": motion["energy"],
        "coherence": motion["coherence"],
        "interference": motion["interference"],
        "peak": peak,
        "acf_display": acf_display,
        "acf_step_s": step / fs,
        "breath_ready": True,
    }
