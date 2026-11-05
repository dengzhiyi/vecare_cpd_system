"""带宽受限硬件：载波规划、健康门控、安全降级。

论文实测的 -3 dB 上限约 20 kHz，默认频带为 10.5-18.5 kHz。真实驱动器常常是
18 kHz，有时只有 16 kHz，
而且失效并不温和：滚降之上的载波几乎没有探测功率，于是谱除法返回的是噪声除以
噪声。那些载波不仅带来误差 —— 它们注入一段宽带随机幅度序列，残差能量把它
当作运动，MRC 权重把它当作相关性。在空车厢上实测，使用论文频带配 18 kHz
驱动器，曾在 17.6 s 产生一次已确认的存在判定。

这些测试把答案的两半都固定下来：依据硬件规划频带，使死载波根本不存在；
同时保留一道运行时检查，以应对数据手册不准的情况。
"""
import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from core.audio import (
    carrier_bins,
    carrier_health,
    make_probe,
    probe_condition_db,
    read_wav,
    write_wav,
)
from core.config import (
    CARRIER_MIN_SPACING_HZ,
    F_HIGH,
    F_LOW,
    HW_GUARD_HZ,
    plan_band,
)
from core.offline import run_offline
from core.replay import export_probe, render_wav, run_wav


# ---------------------------------------------------------------------------
# 规划
# ---------------------------------------------------------------------------

def test_default_ceiling_is_20k_when_none_is_given():
    """论文实测的 -3 dB 上限是 20 kHz，因此 None 必须落到 18.5 kHz 带顶。"""
    band = plan_band(None)
    assert band.hw_f_max == 20_000
    assert band.f_hi == 18_500
    assert band.sweep_hi == 18_700
    assert band.n_carriers == 24
    assert band.f_lo == F_LOW


def test_low_edge_never_moves_with_the_ceiling():
    """这里的每个驱动器都是 10 kHz 高通：只规划频带的上边界。

    如果频带从两端同时收缩，两次运行之间它所平均的物理路径就会改变，而呼吸 ACF
    的标定假定噪声底固定。因此低端边界被钉住，且必须与 10 kHz 高通拐点保持
    距离。
    """
    for hw in (None, 20_000, 18_000, 16_000, 14_000):
        band = plan_band(hw)
        assert band.f_lo == F_LOW, f"{hw}: low edge moved to {band.f_lo}"
        assert band.f_lo > 10_000, "must sit above the 10 kHz high-pass corner"
        assert band.sweep_lo < band.f_lo, "sweep must overhang the low edge too"


def test_declared_ceiling_is_never_sounded():
    """载波必须比上限低 HW_GUARD_HZ，而不是落在上限上。"""
    for hw in (24_000, 20_000, 18_000, 16_000):
        band = plan_band(hw)
        assert band.f_hi <= hw - HW_GUARD_HZ + 1e-9
        assert band.sweep_hi > band.f_hi, "sweep must overhang the carrier band"


def test_narrow_band_keeps_diversity_and_drops_carrier_count():
    """用数量换间距，而不是用间距换数量。"""
    wide, mid, narrow = plan_band(None), plan_band(18_000), plan_band(16_000)
    assert wide.n_carriers == 24 and mid.n_carriers == 24
    assert narrow.n_carriers < mid.n_carriers
    for band in (wide, mid, narrow):
        assert band.spacing_hz >= CARRIER_MIN_SPACING_HZ - 1e-9
        # 干涉条纹必须在相邻载波之间旋转，否则它们就不独立
        assert band.diversity >= 1.0, band.describe()


def test_impossible_band_is_rejected_loudly():
    with pytest.raises(ValueError):
        plan_band(12_000)


def test_carriers_snap_onto_the_probe_grid_for_every_ceiling():
    """不在网格上的载波要插值一个不断振荡的量：10 % 误差。"""
    for hw in (None, 24_000, 18_000, 16_000):
        probe = make_probe(hw_f_max=hw)
        carriers = carrier_bins(probe)
        df = probe.fs / probe.length
        residual = np.abs(carriers / df - np.round(carriers / df))
        assert residual.max() < 1e-9
        # 并且探测信号在那里必须确实有功率
        assert probe_condition_db(probe, carriers).min() > -12.0


def test_probe_stops_at_the_hardware_ceiling():
    """在驱动器上限之上发声，会把周期能量浪费在没人听的频率上。"""
    wide, narrow = make_probe(hw_f_max=24_000), make_probe(hw_f_max=16_000)
    spec_w = np.abs(np.fft.fft(wide.samples))
    spec_n = np.abs(np.fft.fft(narrow.samples))
    df = wide.fs / wide.length
    k18 = int(round(18_000 / df))
    # 宽频带探测信号在 18 kHz 处仍有激励；Kasami 的频谱不是平顶，18 kHz
    # 只是众多载波之一，因此比较的是“同一量级”而非最大值的 50 %。
    assert spec_w[k18] > 0.2 * spec_w.max()
    assert spec_n[k18] < 1e-2 * spec_n.max()
    # 并且窄频带探测信号在所有实际放置载波处都必须足够强
    carriers = carrier_bins(narrow)
    assert probe_condition_db(narrow, carriers).min() > -3.0


# ---------------------------------------------------------------------------
# 运行时健康门控
# ---------------------------------------------------------------------------

def _bandlimited(path_in, path_out, fc, order=6):
    x, fs = read_wav(path_in)
    sos = butter(order, fc, btype="low", fs=fs, output="sos")
    write_wav(path_out, sosfiltfilt(sos, x), fs=fs)
    return path_out


def test_health_flags_dead_carriers_and_keeps_live_ones(tmp_path):
    probe = make_probe()
    src = tmp_path / "empty.wav"
    render_wav(src, "empty", duration_s=16.0, snr_db=35.0)
    limited = _bandlimited(src, tmp_path / "lp18.wav", 18_000)

    from core.replay import load_csi

    amp, _, meta = load_csi(limited)
    alive, rel, threshold = carrier_health(amp[:200])
    freqs = carrier_bins(probe)

    # 远低于截止频率处，每个载波都能存活；远高于截止频率处则一个都不剩。
    #（在两者之间，载波也可能因多径零点而失效 —— 那是真实状况，而非带宽
    # 伪影，因此该区间不做断言。）
    below = freqs < 12_000
    above = freqs > 20_500   # 6 阶 18 kHz 滚降在 18k 处才 -3 dB
    assert alive[below].all(), f"false kill below cutoff: {rel[below]}"
    assert not alive[above].any(), f"dead carriers kept: {rel[above]}"
    assert 0.0 < threshold < 1.0


def test_pruning_survives_half_the_band_being_dead():
    """基线绝不能是均值 —— 一旦频带一半已死，均值就已经移入死载波之中，
    什么都裁剪不掉。"""
    rng = np.random.default_rng(0)
    live = np.abs(1.0 + 0.02 * rng.standard_normal((400, 12)))
    dead = np.abs(0.01 * rng.standard_normal((400, 12)))   # 噪声/噪声
    alive, rel, threshold = carrier_health(np.hstack([live, dead]))
    assert alive[:12].sum() >= 10
    assert alive[12:].sum() == 0, f"threshold {threshold:.3f}, dead rel {rel[12:].round(3)}"


def test_all_dead_band_is_not_silently_pruned_to_nothing():
    rng = np.random.default_rng(1)
    dead = np.abs(0.01 * rng.standard_normal((400, 20)))
    alive, _, _ = carrier_health(dead)
    assert alive.all(), "pruning everything would hide the hardware fault"


# ---------------------------------------------------------------------------
# 端到端降级
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hw", [18_000, 16_000])
def test_narrow_hardware_keeps_a_quiet_cabin_quiet(tmp_path, hw):
    """核心要点：一次关门不能被判定成儿童。"""
    path = tmp_path / "stress.wav"
    render_wav(path, "stress", snr_db=35.0, hw_f_max=hw)
    metrics = run_wav(path, hw_f_max=hw)["metrics"]
    assert not metrics["detected"], f"false presence at {metrics['presence_at']} s"
    assert metrics["alarm_at"] is None


@pytest.mark.parametrize("hw", [18_000, 16_000])
def test_narrow_hardware_still_sees_a_breathing_child(tmp_path, hw):
    path = tmp_path / "sleeping.wav"
    render_wav(path, "sleeping", snr_db=35.0, hw_f_max=hw)
    metrics = run_wav(path, hw_f_max=hw)["metrics"]
    assert metrics["detected"]
    assert metrics["bpm_median"] == pytest.approx(18.0, abs=1.5)
    # 响应预算从首个证据起算，而不是从确认时刻起算
    assert 9.5 <= metrics["response_time_s"] <= 11.0


def test_unplanned_band_is_caught_by_runtime_pruning(tmp_path):
    """18 kHz 驱动器背后放全频带探测信号：设计是错的，但前端本身绝不能
    把一个安静的车厢变成告警。"""
    src = tmp_path / "empty.wav"
    render_wav(src, "empty", duration_s=26.0, snr_db=35.0, hw_f_max=24_000)
    limited = _bandlimited(src, tmp_path / "lp18.wav", 18_000)
    result = run_wav(limited, hw_f_max=24_000)
    assert result["source"]["carriers_used"] < result["source"]["carriers"]
    assert result["source"]["carriers_pruned"], "pruned carriers should be named"
    assert result["metrics"]["alarm_at"] is None


def test_four_second_noise_burst_is_not_a_breathing_child(tmp_path):
    """4 s 噪声突发会绕过只看前 0.5 s 的脉冲检测，并伪装成一次呼吸。

    它既不是 V 形（谷底在 2–5 s），也不是周期性的：长时判据必须在 8 s ACF
    上看到绝对为负、且没有回升到呼吸峰的宽平台。
    """
    path = tmp_path / "burst.wav"
    render_wav(path, "empty", duration_s=26.0, snr_db=40.0)
    x, fs = read_wav(path)
    rng = np.random.default_rng(5)
    start, stop = int(8.0 * fs), int(12.0 * fs)
    x[start:stop] += rng.standard_normal(stop - start) * float(np.max(np.abs(x))) * 0.9
    write_wav(path, x, fs=fs)

    result = run_wav(path)
    metrics = result["metrics"]
    assert not metrics["detected"], f"burst became presence at {metrics['presence_at']} s"
    assert metrics["alarm_at"] is None
    long_frames = [f for f in result["frames"] if f["features"].get("interference_long")]
    assert long_frames, "the 4 s burst never entered the long-interference path"


def test_offline_runner_reports_the_band_it_used():
    wide = run_offline("empty", hw_f_max=None)["metrics"]
    narrow = run_offline("empty", hw_f_max=16_000)["metrics"]
    assert wide["carriers_planned"] == 24
    assert narrow["carriers_planned"] < wide["carriers_planned"]
    assert "carriers" in narrow["band"]


def test_exported_probe_matches_the_declared_ceiling(tmp_path):
    info = export_probe(tmp_path / "p.wav", seconds=0.2, hw_f_max=16_000)
    assert info["sweep_hz"][1] < 16_000
    assert info["carriers"] == plan_band(16_000).n_carriers


def test_ui_offers_exactly_the_bands_the_planner_accepts():
    """这些选项不能请求任何 plan_band 会拒绝的频带。

    UI 从该返回数据中读取载波数量，而不是在浏览器里重新推导，因此这里一旦出现
    偏差，屏幕上的数字就会与 DSP 实际使用的数字脱节。
    """
    from server import BAND_CHOICES, band_options

    opts = band_options()
    assert [o["hw_f_max"] for o in opts] == list(BAND_CHOICES)
    for opt, hw in zip(opts, BAND_CHOICES):
        band = plan_band(hw)
        assert opt["n_carriers"] == band.n_carriers
        assert opt["f_lo_hz"] == band.f_lo
        assert opt["f_hi_hz"] == band.f_hi
        assert opt["label"], "every chip needs a short label"
        assert opt["describe"] == band.describe()
