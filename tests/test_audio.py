"""真实音频前端：探测信号设计、WAV 读写、CSI 恢复、WAV 回放。

正是这些测试让一次真实采集变得可信。没有它们，音频链路就只是一堆看起来
合理的 DSP：每一种失效模式（渐变错误、帧同步错误、插值错误）都仍会产出一个
数值。
"""
import numpy as np
import pytest

from core.audio import (
    carrier_bins,
    csi_amplitude,
    estimate_csi,
    frame_offsets,
    make_probe,
    probe_condition_db,
    read_wav,
    write_wav,
)
from core.config import CSI_FS, F_HIGH, F_LOW, N_SUBCARRIERS
from core.replay import export_probe, load_csi, render_wav, run_wav


def _render_frame(probe, delays, gains):
    """麦克风会录到的一个探测周期，其中携带着已知的 CIR。"""
    L, fs = probe.length, probe.fs
    spectrum = np.fft.fft(probe.samples)
    freqs = np.fft.fftfreq(L, 1.0 / fs)
    H = (gains[:, None] * np.exp(-1j * 2 * np.pi * freqs[None, :] * delays[:, None])).sum(0)
    full = np.zeros(L, dtype=complex)
    full[: L // 2 + 1] = (spectrum * H)[: L // 2 + 1]
    full[L // 2 + 1:] = np.conj(full[L // 2 - 1: 0: -1])
    return np.fft.ifft(full).real


# ---------------------------------------------------------------------------
# 探测信号
# ---------------------------------------------------------------------------

def test_probe_period_is_one_csi_frame():
    probe = make_probe()
    assert probe.csi_fs == pytest.approx(CSI_FS)
    assert probe.length == 960


def test_probe_illuminates_every_carrier():
    """若扫频从频带边缘开始，外侧载波会损失 23 dB。"""
    probe = make_probe()
    db = probe_condition_db(probe, carrier_bins(probe))
    assert db.min() > -6.0, f"weakest carrier at {db.min():.1f} dB"


def test_probe_refuses_a_sample_rate_below_the_sweep():
    with pytest.raises(ValueError):
        make_probe(fs=44100)


def test_carrier_bins_sit_on_the_dft_grid():
    """不在网格上的载波无法被精确估计，只能插值。"""
    probe = make_probe()
    carriers = carrier_bins(probe)
    assert carriers.size == N_SUBCARRIERS
    pos = carriers * probe.length / probe.fs
    assert np.allclose(pos, np.round(pos), atol=1e-9)
    assert carriers.min() >= F_LOW and carriers.max() <= F_HIGH


# ---------------------------------------------------------------------------
# CSI 恢复
# ---------------------------------------------------------------------------

def test_csi_estimate_recovers_known_channel_exactly():
    probe = make_probe()
    delays = np.array([0.00058, 0.0017, 0.0039, 0.0072])
    gains = np.array([1.0, 0.5 * np.exp(1j * 0.7), 0.35 * np.exp(-1j * 2.1), 0.2j])
    x = _render_frame(probe, delays, gains)

    carriers = carrier_bins(probe)
    expected = np.abs(
        (gains[None, :] * np.exp(-1j * 2 * np.pi * carriers[:, None] * delays[None, :])).sum(1)
    )
    cfr, _ = estimate_csi(x, probe, carriers, starts=np.array([0]))
    error = np.abs(np.abs(cfr[0]) - expected) / expected
    # 无噪声：谱除法是精确的，因此任何超过舍入误差的结果都是 bug
    assert error.max() < 1e-4, f"max relative error {error.max():.2e}"


def test_csi_estimate_matches_the_analytic_cfr_at_modest_snr():
    probe = make_probe()
    rng = np.random.default_rng(4)
    delays = np.array([0.00058, 0.0028, 0.0051, 0.0092])
    gains = np.abs(rng.normal(1.0, 0.2, 4)) * np.exp(1j * rng.uniform(-np.pi, np.pi, 4))
    x = _render_frame(probe, delays, gains)
    x = x + rng.standard_normal(x.size) * np.sqrt(np.mean(x ** 2)) * 10 ** (-40 / 20)

    carriers = carrier_bins(probe)
    expected = np.abs(
        (gains[None, :] * np.exp(-1j * 2 * np.pi * carriers[:, None] * delays[None, :])).sum(1)
    )
    cfr, _ = estimate_csi(x, probe, carriers, starts=np.array([0]))
    error = np.abs(np.abs(cfr[0]) - expected) / expected
    assert np.median(error) < 0.01


def test_frame_sync_fills_a_dropped_period():
    """丢失的匹配滤波峰绝不能拉伸时间基准。"""
    probe = make_probe()
    reps = 20
    stream = np.tile(probe.samples, reps)
    stream[9 * probe.length:10 * probe.length] = 0.0   # 模拟一次丢帧

    starts, stats = frame_offsets(stream, probe)
    assert stats["filled"] == 1
    assert stats["holes"] == 0
    assert stats["period_samples"] == probe.length
    # 第一个和最后一个周期永远不会成为匹配滤波峰（没有相邻周期），
    # 因此 20 次重复只产生 18 个帧起始位置 —— 但网格必须保持均匀
    assert starts.size == reps - 2
    assert np.all(np.diff(starts) == probe.length)


def test_frame_sync_reports_a_long_hole_instead_of_stretching():
    probe = make_probe()
    reps = 20
    stream = np.tile(probe.samples, reps)
    stream[8 * probe.length:14 * probe.length] = 0.0   # 缺失 6 帧

    _, stats = frame_offsets(stream, probe, max_fill=4)
    assert stats["holes"] == 1
    assert stats["longest_gap_s"] > 0.1


# ---------------------------------------------------------------------------
# WAV 读写
# ---------------------------------------------------------------------------

def test_wav_roundtrip_keeps_the_waveform(tmp_path):
    probe = make_probe()
    path = write_wav(tmp_path / "p.wav", probe.samples, fs=probe.fs)
    back, fs = read_wav(path)
    assert fs == probe.fs
    assert back.size == probe.samples.size
    assert np.max(np.abs(back - probe.samples)) < 2.0 / 32768


def test_probe_export_is_playable(tmp_path):
    info = export_probe(tmp_path / "probe.wav", seconds=2.0)
    data, fs = read_wav(tmp_path / "probe.wav")
    assert info["fs"] == fs == 48000
    assert info["period_ms"] == pytest.approx(20.0)
    assert data.size == pytest.approx(2.0 * fs, rel=1e-3)


def test_low_sample_rate_recording_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        make_probe(fs=44100)
    path = write_wav(tmp_path / "low.wav", np.zeros(22050), fs=22050)
    with pytest.raises(RuntimeError, match="Nyquist"):
        load_csi(path)


def test_recording_without_the_probe_raises_a_clear_error(tmp_path):
    rng = np.random.default_rng(0)
    path = write_wav(tmp_path / "noise.wav", rng.standard_normal(48000 * 3) * 0.1)
    with pytest.raises(RuntimeError, match="probe repetition"):
        load_csi(path)


# ---------------------------------------------------------------------------
# 端到端：场景 -> WAV -> 判决
# ---------------------------------------------------------------------------

def test_rendered_wav_reproduces_the_simulator_verdict(tmp_path):
    """音频链路的全部意义所在：同一场景，同一判决。"""
    path = tmp_path / "sleeping.wav"
    render_wav(path, "sleeping", duration_s=32.0, snr_db=40.0)

    amp, times, meta = load_csi(path)
    assert meta["csi_fs"] == pytest.approx(CSI_FS, abs=0.05)
    assert meta["sync"]["holes"] == 0
    assert amp.shape[1] == N_SUBCARRIERS

    result = run_wav(path)
    metrics = result["metrics"]
    assert metrics["detected"], "breathing infant was not detected from its WAV"
    assert metrics["alarm_at"] is not None
    assert metrics["response_time_s"] is not None
    assert metrics["response_time_s"] <= 10.2
    assert 15.0 <= metrics["bpm_median"] <= 21.0     # 真值为 18 BPM


def test_quiet_rendered_wav_does_not_alarm(tmp_path):
    path = tmp_path / "empty.wav"
    render_wav(path, "empty", duration_s=26.0, snr_db=40.0)
    metrics = run_wav(path)["metrics"]
    assert not metrics["detected"]
    assert metrics["alarm_at"] is None


def test_breathing_rate_survives_a_quiet_recording(tmp_path):
    """低噪声车厢才是困难情形，而不是嘈杂的车厢。

    噪声底更低时，胸腔起伏自身的去趋势残差已是噪声底的好几倍，于是运动门控会
    被置位；如果只要运动被标记就把呼吸静音，那么*更干净*的录音反而会报告*更少*
    的呼吸。这里在 60 dB SNR 下验证，因为仿真器自身的 35 dB 会掩盖这一失效。
    """
    path = tmp_path / "quiet.wav"
    render_wav(path, "sleeping", duration_s=30.0, snr_db=60.0)
    metrics = run_wav(path)["metrics"]
    assert metrics["detected"]
    assert metrics["bpm_median"] is not None, "quiet recording lost the breathing rate"
    assert 15.0 <= metrics["bpm_median"] <= 21.0


def test_csi_amplitude_wrapper_agrees_with_estimate_csi():
    probe = make_probe()
    x = _render_frame(probe, np.array([0.001, 0.004]), np.array([1.0, 0.3j]))
    amp, times, cfr = csi_amplitude(np.tile(x, 4), probe)
    assert amp.shape == (3, N_SUBCARRIERS)
    # 匹配滤波器锁定最先到达的路径，因此帧起始位置在每个周期内都带有一个
    # 固定偏移 —— 这没有影响，|H(f)| 对其不变
    assert np.allclose(np.diff(times), probe.period_s)
    assert 0.0 <= times[0] < probe.period_s
    assert np.allclose(amp, np.abs(cfr))
