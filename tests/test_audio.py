"""真实音频前端：探测信号设计、WAV 读写、CSI 恢复、WAV 回放。

正是这些测试让一次真实采集变得可信。没有它们，音频链路就只是一堆看起来
合理的 DSP：每一种失效模式（渐变错误、帧同步错误、插值错误）都仍会产出一个
数值。
"""
import numpy as np
import pytest

from core.audio import (
    _MLS_TAPS,
    _kasami_codes,
    _mls_sequence,
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
from core.replay import check_audio, export_probe, load_csi, render_wav, run_wav


def _corr(x, y, shift):
    """两条 0/1 序列在循环移位下的 ±1 相关值。"""
    n = x.size
    diff = int(np.sum((x ^ np.roll(y, shift)).astype(np.int16)))
    return n - 2 * diff


# ---------------------------------------------------------------------------
# 探测码：LFSR 与 Kasami
# ---------------------------------------------------------------------------

def test_every_lfsr_polynomial_produces_a_true_m_sequence():
    """反馈端与输出端必须分开，否则 12 条多项式都不是 m 序列。

    真正的 m 序列有两个可判定的性质：一个周期内 0/1 各半，且周期自相关在
    所有非零 lag 上恰为 -1。旧实现用 ``reg[-t]``，把反馈端算进了输出端，
    序列仍然“看起来随机”，但周期自相关旁瓣高达几十，Kasami 的
    ``2^(n/2)+1`` 互相关上界随之失效。
    """
    for order, taps in _MLS_TAPS.items():
        seq = _mls_sequence(order, taps)
        assert seq.size == 2 ** order - 1
        assert int(seq.sum()) == 2 ** (order - 1), f"order {order} is not balanced"
        sidelobes = [_corr(seq, seq, k) for k in range(1, seq.size)]
        assert max(abs(v) for v in sidelobes) == 1, f"order {order} is not max-length"


def test_kasami_small_set_has_the_expected_size_and_cross_correlation():
    """小集合含基序列 a，共 2^(n/2) 条，互相关不超过 2^(n/2)+1。"""
    order = 8
    codes = _kasami_codes(order)
    bound = 2 ** (order // 2) + 1
    assert len(codes) == 2 ** (order // 2) == 16
    assert len({tuple(c) for c in codes}) == len(codes), "duplicate codes"
    worst = max(
        abs(_corr(codes[i], codes[j], k))
        for i in range(len(codes))
        for j in range(i + 1, len(codes))
        for k in range(codes[0].size)
    )
    assert worst <= bound, f"cross-correlation {worst} exceeds {bound}"


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
    """若探测带在载波频带边缘截止，外侧载波会损失大量探测功率。"""
    probe = make_probe()
    db = probe_condition_db(probe, carrier_bins(probe))
    assert db.min() > -6.0, f"weakest carrier at {db.min():.1f} dB"


def test_probe_refuses_a_sample_rate_below_the_probe_band():
    # 默认 20 kHz 上限的探测带只到 18.7 kHz，44.1 kHz 录音已够用；
    # 只有显式要求更宽的上限时才会撞上 Nyquist。
    make_probe(fs=44100)
    with pytest.raises(ValueError):
        make_probe(fs=44100, hw_f_max=24_000)


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
    # Kasami 的匹配滤波在录音首尾也可能给出有效峰；关键是检测到的起始点
    # 必须落在均匀网格上，且丢帧被补回而不是把时间基准拉伸。
    assert starts.size >= reps - 2
    # 首尾两个峰受周期环绕影响会偏移几个采样，内部网格必须严格等间距。
    assert np.all(np.diff(starts)[1:-1] == probe.length)


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
        make_probe(fs=44100, hw_f_max=24_000)
    path = write_wav(tmp_path / "low.wav", np.zeros(22050), fs=22050)
    with pytest.raises(RuntimeError, match="Nyquist"):
        load_csi(path)


def test_recording_without_the_probe_raises_a_clear_error(tmp_path):
    rng = np.random.default_rng(0)
    path = write_wav(tmp_path / "noise.wav", rng.standard_normal(48000 * 3) * 0.1)
    with pytest.raises(RuntimeError, match="probe repetition"):
        load_csi(path)


# ---------------------------------------------------------------------------
# 录音质检：判定前的门
# ---------------------------------------------------------------------------

def test_quality_check_passes_a_usable_capture(tmp_path):
    path = tmp_path / "usable.wav"
    render_wav(path, "sleeping", duration_s=20.0, snr_db=40.0)
    qc = check_audio(path)
    assert qc["passed"], qc
    assert qc["status"] in {"PASS", "WARN"}
    assert qc["sync"]["match_ratio"] >= 9.0
    assert qc["sync"]["holes"] == 0


def test_quality_check_accepts_the_exported_probe(tmp_path):
    """探测音文件本身是满刻度，但每周期只有孤立峰值，不应被当成削波。"""
    path = tmp_path / "probe18k.wav"
    export_probe(path, seconds=20.0, hw_f_max=18000)
    qc = check_audio(path, hw_f_max=18000)
    assert qc["passed"], qc
    level = next(item for item in qc["checks"] if item["name"] == "电平")
    assert level["status"] == "PASS", level


def test_quality_check_flags_real_clipping(tmp_path):
    """真正的削波会在波峰处留下连续满刻度样本，仍必须判 FAIL。"""
    x = np.zeros(48000 * 12)
    x[48000:48200] = 1.0
    x[49000:49200] = -1.0
    path = write_wav(tmp_path / "clipped.wav", x)
    qc = check_audio(path)
    level = next(item for item in qc["checks"] if item["name"] == "电平")
    assert level["status"] == "FAIL", level
    assert not qc["passed"]


def test_quality_check_rejects_a_capture_without_probe(tmp_path):
    """没有探测音的录音必须被这道门拦住，而不是进入判定给出一个数值。"""
    path = write_wav(tmp_path / "silence.wav", np.zeros(48000 * 10))
    qc = check_audio(path)
    assert not qc["passed"]
    assert qc["status"] == "FAIL"
    assert any(c["name"] == "帧同步" and c["status"] == "FAIL" for c in qc["checks"])


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
