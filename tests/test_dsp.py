"""Unit tests for the SAS signal-processing core."""
import numpy as np

from core.config import DetectorConfig
from core.dsp import (
    autocorr,
    combine_acf,
    compute_features,
    find_breathing_peak,
    motion_statistic,
)


def test_autocorr_periodic_peak():
    fs = 50
    period_s = 3.3
    t = np.arange(int(fs * 10)) / fs
    x = np.sin(2 * np.pi * t / period_s)
    acf = autocorr(x[:, None], max_lag=int(period_s * fs * 2))
    lag = int(round(period_s * fs))
    assert acf[0, 0] == 1.0
    assert acf[lag, 0] > 0.90
    assert acf[lag, 0] > acf[lag - 5, 0]


def test_motion_statistic_reads_requested_tap():
    acf = np.arange(20, dtype=float)[:, None]
    gains = motion_statistic(acf, 5)
    assert gains[0] == acf[5, 0]


def test_combine_acf_equal_weights_is_mean():
    acf = np.array([[1.0, 2.0], [3.0, 4.0]])
    combined = combine_acf(acf, np.array([1.0, 1.0]))
    assert combined.shape == (2,)
    assert np.allclose(combined, acf.mean(axis=1))


def test_find_breathing_peak_recovers_period():
    fs = 50
    cfg = DetectorConfig()
    t = np.arange(fs * 10) / fs
    acf = np.exp(-((t - 3.33) ** 2) / 0.08) + 0.5 * np.exp(-((t - 6.66) ** 2) / 0.08)
    peak = find_breathing_peak(acf, fs, cfg, window_samples=8 * fs)
    assert peak["valid"]
    assert abs(peak["bpm"] - 18.0) < 2.0


def test_compute_features_waits_for_full_breath_window():
    cfg = DetectorConfig()
    amp = 1.0 + 0.1 * np.random.default_rng(0).standard_normal((100, 24))
    feat = compute_features(amp, amp, cfg, fs=50)
    assert feat["breath_ready"] is False
    assert len(feat["gains"]) == 24


def test_compute_features_detects_synthetic_breathing():
    fs = 50
    cfg = DetectorConfig()
    rng = np.random.default_rng(1)
    t = np.arange(int(8 * fs)) / fs
    base = 1.0 + 0.1 * np.sin(2 * np.pi * 0.3 * t)[:, None]
    amp_breath = base + 0.01 * rng.standard_normal((len(t), 24))
    amp_motion = 1.0 + 0.005 * rng.standard_normal((int(2 * fs), 24))
    feat = compute_features(amp_motion, amp_breath, cfg, fs=fs)
    assert feat["breath_ready"] is True
    assert abs(feat["peak"]["bpm"] - 18.0) < 2.0
    assert feat["peak"]["confidence"] > 0.5
