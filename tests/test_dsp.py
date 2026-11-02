"""Unit tests for the SAS signal-processing core."""
import numpy as np

from core.config import DetectorConfig
from core.dsp import (
    autocorr,
    combine_acf,
    compute_features,
    find_breathing_peak,
    interference_pattern,
    motion_statistic,
    subcarrier_coherence,
)

FS = 50


def _breathing_acf(bpm, window_s=8.0, fs=FS):
    """Combined ACF of a clean sinusoidal breathing signal."""
    T = 60.0 / bpm
    t = np.arange(int(window_s * fs)) / fs
    x = (1.0 + 0.1 * np.sin(2 * np.pi * t / T))[:, None] * np.ones((1, 24))
    return combine_acf(autocorr(x, int(window_s * fs) - 2), np.ones(24))


def test_autocorr_periodic_peak():
    period_s = 3.3
    t = np.arange(int(FS * 10)) / FS
    x = np.sin(2 * np.pi * t / period_s)
    acf = autocorr(x[:, None], max_lag=int(period_s * FS * 2))
    lag = int(round(period_s * FS))
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


def test_mrc_weights_actually_concentrate():
    """A single informative subcarrier must dominate the MRC weights."""
    acf = np.random.default_rng(0).standard_normal((200, 8))
    weights = np.zeros(8)
    weights[3] = 4.0
    combined = combine_acf(acf, weights)
    w = (np.maximum(weights, 0) + 0.02)
    w /= w.sum()
    expected = acf @ w
    assert np.allclose(combined, expected)
    assert w.max() > 0.6  # concentrated, not flattened to 1/8


def test_subcarrier_coherence_separates_common_mode_from_noise():
    rng = np.random.default_rng(1)
    n = 100
    common = rng.standard_normal(n)[:, None]
    coherent = common * np.linspace(0.5, 1.5, 24)[None, :] + 0.01 * rng.standard_normal((n, 24))
    incoherent = rng.standard_normal((n, 24))
    assert subcarrier_coherence(coherent) > 0.9
    assert subcarrier_coherence(incoherent) < 0.20


def test_interference_pattern_flags_v_shaped_acf():
    """§5 signature: the ACF falls to a trough and climbs back (V shape)."""
    lag = np.arange(26)
    v_shape = np.abs(lag - 8) / 8.0 * 0.8  # 1.0 -> 0 -> 1.0 over 0..16
    assert interference_pattern(v_shape)

    # a monotone decay, as produced by slow body motion, must not match
    assert not interference_pattern(np.exp(-lag / 6.0))
    # a breathing ACF over the first 0.5 s only dips slightly
    t = np.arange(8 * FS) / FS
    breath = (1 + 0.1 * np.sin(2 * np.pi * t / 3.3))[:, None] * np.ones((1, 4))
    assert not interference_pattern(autocorr(breath, 25).mean(axis=1))


def test_find_breathing_peak_recovers_period():
    cfg = DetectorConfig()
    t = np.arange(FS * 10) / FS
    acf = np.exp(-((t - 3.33) ** 2) / 0.08) + 0.5 * np.exp(-((t - 6.66) ** 2) / 0.08)
    peak = find_breathing_peak(acf, FS, cfg, window_samples=8 * FS)
    assert peak["valid"]
    assert abs(peak["bpm"] - 18.0) < 2.0


def test_breathing_rate_not_halved_above_45_bpm():
    """Regression: boundary-truncated prominence used to pick the harmonic."""
    cfg = DetectorConfig()
    for bpm in (15.0, 20.0, 30.0, 40.0, 50.0, 55.0, 60.0, 70.0):
        peak = find_breathing_peak(_breathing_acf(bpm), FS, cfg, int(8 * FS))
        assert peak["valid"], f"{bpm} BPM not detected"
        assert abs(peak["bpm"] - bpm) < 2.0, f"{bpm} BPM -> {peak['bpm']}"


def test_breathing_rate_extends_down_to_12_bpm():
    cfg = DetectorConfig()
    for bpm in (12.0, 13.0, 14.0):
        peak = find_breathing_peak(_breathing_acf(bpm), FS, cfg, int(8 * FS))
        assert peak["valid"], f"{bpm} BPM not detected"
        assert abs(peak["bpm"] - bpm) < 2.0, f"{bpm} BPM -> {peak['bpm']}"


def test_compute_features_waits_for_full_breath_window():
    cfg = DetectorConfig()
    amp = 1.0 + 0.1 * np.random.default_rng(0).standard_normal((100, 24))
    feat = compute_features(amp, amp, cfg, fs=FS)
    assert feat["breath_ready"] is False
    assert len(feat["gains"]) == 24
    assert feat["energy"] > 0.0
    assert feat["coherence"] >= 0.0 and feat["interference"] in (True, False)


def test_compute_features_detects_synthetic_breathing():
    cfg = DetectorConfig()
    rng = np.random.default_rng(1)
    t = np.arange(int(8 * FS)) / FS
    base = 1.0 + 0.1 * np.sin(2 * np.pi * 0.3 * t)[:, None]
    amp_breath = base + 0.01 * rng.standard_normal((len(t), 24))
    amp_motion = 1.0 + 0.005 * rng.standard_normal((int(FS), 24))
    feat = compute_features(amp_motion, amp_breath, cfg, fs=FS)
    assert feat["breath_ready"] is True
    assert abs(feat["peak"]["bpm"] - 18.0) < 2.0
    assert feat["peak"]["confidence"] > 0.5
