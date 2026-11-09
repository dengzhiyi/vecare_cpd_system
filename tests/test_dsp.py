"""SAS 信号处理核心的单元测试。"""
import numpy as np
import pytest

from core.config import DetectorConfig
from core.dsp import (
    autocorr,
    coherence_noise_floor,
    combine_acf,
    compute_features,
    find_breathing_peak,
    interference_pattern,
    motion_statistic,
    residual_energy,
    subcarrier_coherence,
)

FS = 50


def test_residual_energy_is_scale_invariant():
    """载波归一化后，整体增益不能改变运动统计量。"""
    rng = np.random.default_rng(11)
    X = 1.0 + 0.02 * rng.standard_normal((100, 24))
    base = residual_energy(X)
    for k in (0.25, 0.5, 2.0, 10.0):
        assert residual_energy(X * k) == pytest.approx(base, rel=1e-12)


def test_coherence_floor_follows_the_actual_carrier_count():
    """相干噪声底必须随 S 变化；24 路约 0.0925，12 路约 0.151。"""
    assert coherence_noise_floor(100, 24) == pytest.approx(0.0925, abs=2e-4)
    assert coherence_noise_floor(100, 17) > 0.11
    assert coherence_noise_floor(100, 12) == pytest.approx(0.151, abs=2e-3)
    assert coherence_noise_floor(100, 12) > coherence_noise_floor(100, 24)


def _breathing_acf(bpm, window_s=8.0, fs=FS):
    """干净正弦呼吸信号的合并 ACF。"""
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
    """单个有信息的子载波必须主导 MRC 权重。"""
    acf = np.random.default_rng(0).standard_normal((200, 8))
    weights = np.zeros(8)
    weights[3] = 4.0
    combined = combine_acf(acf, weights)
    w = (np.maximum(weights, 0) + 0.02)
    w /= w.sum()
    expected = acf @ w
    assert np.allclose(combined, expected)
    assert w.max() > 0.6  # 权重集中，而不是被摊平成 1/8


def test_subcarrier_coherence_separates_common_mode_from_noise():
    rng = np.random.default_rng(1)
    n = 100
    common = rng.standard_normal(n)[:, None]
    coherent = common * np.linspace(0.5, 1.5, 24)[None, :] + 0.01 * rng.standard_normal((n, 24))
    incoherent = rng.standard_normal((n, 24))
    assert subcarrier_coherence(coherent) > 0.9
    assert subcarrier_coherence(incoherent) < 0.20


def test_interference_pattern_flags_v_shaped_acf():
    """§5 特征：ACF 下降到一个谷底再回升（V 形）。"""
    lag = np.arange(26)
    v_shape = np.abs(lag - 8) / 8.0 * 0.8  # 在 0..16 上从 1.0 -> 0 -> 1.0
    assert interference_pattern(v_shape)

    # 缓慢身体运动产生的单调衰减，必须不匹配
    assert not interference_pattern(np.exp(-lag / 6.0))
    # 呼吸 ACF 在前 0.5 s 内只会轻微下凹
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


def test_isolated_side_lobe_is_not_accepted_as_breathing():
    """安静环境里的 ACF 旁瓣：高度和突出度都够，但没有周期结构。

    旧实现在这类曲线上给出 40 BPM、置信度 0.86 —— 一个从未有人呼吸过的
    呼吸率。真实呼吸必须在谷底之后于 2*tau 处再现第二个峰；孤立旁瓣没有。
    """
    cfg = DetectorConfig()
    t = np.arange(8 * FS) / FS
    acf = 0.9 * np.exp(-t / 3.0) + 0.35 * np.exp(-((t - 1.5) ** 2) / (2 * 0.15 ** 2))
    peak = find_breathing_peak(acf, FS, cfg, 8 * FS)
    assert not peak["valid"]
    assert peak["bpm"] is None


def test_decaying_baseline_without_a_peak_is_rejected():
    cfg = DetectorConfig()
    t = np.arange(8 * FS) / FS
    peak = find_breathing_peak(np.exp(-t / 2.0), FS, cfg, 8 * FS)
    assert not peak["valid"]
    assert peak["bpm"] is None


def test_breathing_rate_not_halved_above_45_bpm():
    """回归测试：过去被边界截断的突出度会导致选中谐波。"""
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
