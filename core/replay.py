"""把录制的 WAV 经 SAS 流水线回放。

这是真实采集与检测器之间的桥梁。仿真器每 20 ms 交给 ``SASPipeline``
24 个复信道系数；而录音只能提供麦克风波形。处理链路为：

    probe -> (speaker, cabin, mic) -> WAV -> frame sync -> |H(f)| -> pipeline

``render_wav`` 则向相反方向闭环：它把一个仿真场景转成 WAV，因此整条音频链路
无需任何硬件即可端到端检验。这项检验是唯一能让人信任真实采集数值的诚实方式
—— 如果渲染出的 WAV 无法重现仿真器自己的判定，那问题就在前端，而不是座舱。

已知的建模假设：探测信号每 20 ms 重复一次，因此该估计假设座舱冲激响应短于
20 ms（即 6.9 m 的路径）。大型体育馆式的混响会发生环绕并产生混叠；而车内座舱
不会。

“座舱录音”从哪里来：
    * 直接仿真（UI / run_offline）使用 core.channel，完全不用 WAV；
    * ``--selfcheck`` 每次运行都把全部七个场景渲染成 ``data/selfcheck/*.wav``
      （已被 git 忽略），再通过本模块回放；
    * 真实验证需要一段由用户录制的 ``capture.wav``，录制时需在座舱内循环播放
      导出的 ``probe.wav``。
    ``data/sessions`` 下的会话日志保存的是 CSI 幅度和判决 —— 它们不是音频，
    无法通过音频前端回放。
"""
from __future__ import annotations

import wave
import zlib
from pathlib import Path

import numpy as np

from .audio import (
    Probe,
    carrier_bins,
    estimate_csi,
    frame_offsets,
    make_probe,
    probe_condition_db,
    read_wav,
    write_wav,
)
from .channel import AcousticChannel
from .config import (
    AUDIO_FS,
    BAND_CHOICES,
    CSI_FS,
    F_HIGH,
    MRC_WEIGHT_FLOOR,
    SWEEP_HI,
    SWEEP_LO,
    DetectorConfig,
    plan_band,
)
from .pipeline import SASPipeline
from .scenarios import get_scenario


# ---------------------------------------------------------------------------
# 探测信号导出
# ---------------------------------------------------------------------------

def export_probe(
    path: str | Path,
    seconds: float = 120.0,
    fs: int = AUDIO_FS,
    hw_f_max: float | None = None,
) -> dict:
    """把探测信号写成 WAV，供采集时播放。

    录音时必须同时播放探测信号；没有它就没有 CSI 可估计，只有语音和路噪。

    ``hw_f_max`` 是扬声器 + 麦克风实测的 -3 dB 上限。在该上限之上发声并非没有
    代价：探测音会把周期能量花在驱动器无法重现的频率上，于是每个*实际使用*的
    载波都会变得更嘈杂。传入真实上限，Kasami 探测带就会收缩以适配。
    """
    probe = make_probe(fs=fs, hw_f_max=hw_f_max)
    reps = max(1, int(round(seconds * probe.csi_fs)))
    write_wav(path, np.tile(probe.samples, reps), fs=fs)
    band = probe.band
    return {
        "path": str(path),
        "seconds": round(reps * probe.period_s, 2),
        "fs": fs,
        "period_ms": round(probe.period_s * 1000, 3),
        "sweep_hz": [round(band.sweep_lo, 1), round(band.sweep_hi, 1)],
        "carrier_hz": [round(band.f_lo, 1), round(band.f_hi, 1)],
        "carriers": band.n_carriers,
        "band": band.describe(),
        "samples_per_period": probe.length,
    }


# ---------------------------------------------------------------------------
# WAV -> CSI
# ---------------------------------------------------------------------------

def _clipped_samples(x: np.ndarray, level: float = 0.999, min_run: int = 3) -> int:
    """统计真正削波的样本数：只有连续 ``min_run`` 个满刻度样本才算。

    归一化到满量程的播放文件，其峰值也会落在 16-bit 的 32767（读回约 0.99997），
    但通常每个周期只有孤立的一个样本。若按“逐样本 >= 0.999”计数，会把正常的
    满刻度波形误判成削波；真实削波则会在波峰处留下连续的平坦段。
    """
    hot = np.abs(np.asarray(x)) >= level
    if not hot.any():
        return 0
    edges = np.diff(np.concatenate(([0], hot.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    runs = ends - starts
    return int(runs[runs >= min_run].sum())


def _band_mismatch_hint(x: np.ndarray, fs: int, hw_f_max: float | None) -> str:
    """同步失败时，用其它频响档位试一次，给出可操作的提示。

    18 kHz 档的录音在默认 20 kHz 档下检查，探测音的高频部分可能刚好落在
    滤波器边缘，匹配滤波峰不足以过门；此时原始报错“没有探测音”会把人引向
    录音流程，而真正的原因是档位不一致。只在失败路径上多试几次，代价可接受。
    """
    head = x[: min(x.size, 10 * fs)]
    best = None
    for ceiling in BAND_CHOICES:
        if hw_f_max is not None and abs(ceiling - float(hw_f_max)) < 1e-6:
            continue
        try:
            probe = make_probe(fs=fs, hw_f_max=ceiling)
            _, sync = frame_offsets(head, probe)
        except (ValueError, RuntimeError):
            continue
        ratio = float(sync["match_ratio"])
        if best is None or ratio > best[1]:
            best = (ceiling, ratio)
    if best is None or best[1] < 9.0:
        return ""
    return (f" This recording matches the {best[0]/1000:.0f} kHz band "
            f"(quality {best[1]:.1f}); add `--hw-f-max {best[0]:.0f}`.")


def load_csi(
    path: str | Path,
    fs: int | None = None,
    probe: Probe | None = None,
    carriers: np.ndarray | None = None,
    min_peak_ratio: float = 0.2,
    hw_f_max: float | None = None,
    max_seconds: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """读取一个 WAV，并返回 ``(amplitude, times, meta)``。

    ``amplitude`` 的形状为 ``(n_frames, n_carriers)``；``times`` 是文件中的真实
    位置，因此丢失的帧会表现为时间上的真实空隙，而不是无声无息地把序列拉伸。

    ``hw_f_max`` 依据硬件可用上限来规划载波频带。不传它时使用默认的
    20 kHz 上限（10.5–18.5 kHz 载波带）；如果实际驱动器更窄，必须显式传入，
    否则 16 kHz 的驱动器会被要求提供它无法发出的载波 —— 见
    ``config.plan_band``。

    ``max_seconds`` 只分析文件开头的一段，供超长录音的可视化使用。
    """
    x, native_fs = read_wav(path, target_fs=fs, max_seconds=max_seconds)
    band = plan_band(hw_f_max)
    if 2 * band.sweep_hi >= native_fs:
        raise RuntimeError(
            f"{Path(path).name}: sample rate is {native_fs} Hz, but the "
            f"{band.sweep_hi/1000:.1f} kHz sweep needs at least "
            f"{2*band.sweep_hi/1000:.1f} kHz of Nyquist bandwidth. Record at 48 kHz."
        )
    probe = probe or make_probe(fs=native_fs, hw_f_max=hw_f_max)
    starts, sync = frame_offsets(x, probe, min_ratio=min_peak_ratio)
    if starts.size < 2 or sync["match_ratio"] < 9.0:
        hint = _band_mismatch_hint(x, native_fs, hw_f_max)
        raise RuntimeError(
            f"{Path(path).name}: no probe repetition found "
            f"(correlation peak/floor {sync['match_ratio']:.1f}, need >= 9; a "
            f"recording that contains the Kasami probe scores around 12–18)."
            + hint + " The probe must be playing during the recording, at its "
            f"original {probe.period_s*1000:.1f} ms period -- no resampling, no "
            f"tempo change, no noise suppression. Run "
            f"`python replay_audio.py --emit-probe probe.wav` for the file, and "
            f"see README「实采录音接入」for the recording setup."
        )

    carriers = carrier_bins(probe) if carriers is None else np.asarray(carriers, dtype=float)
    cfr, used = estimate_csi(x, probe, carriers, starts=starts)
    if used.size == 0:
        raise RuntimeError(f"{Path(path).name}: frames detected but none complete")

    amp = np.abs(cfr)
    times = used / float(probe.fs)

    meta = {
        "path": str(path),
        "fs": native_fs,
        "frames": int(used.size),
        "carriers": int(carriers.size),
        "carrier_span_hz": [round(float(carriers.min()), 1), round(float(carriers.max()), 1)],
        "carrier_spacing_hz": (round(float(np.mean(np.diff(carriers))), 1)
                               if carriers.size > 1 else None),
        "band": band.describe(),
        "hw_f_max_hz": round(float(band.hw_f_max), 1),
        "sweep_hz": [round(band.sweep_lo, 1), round(band.sweep_hi, 1)],
        "csi_fs": round(float(probe.csi_fs), 3),
        "period_samples": int(sync["period_samples"]),
        "duration_s": round(float(times[-1]), 2),
        "sync": sync,
        "probe_db_min": round(float(probe_condition_db(probe, carriers).min()), 2),
        # 粗略的电平检查：削波或近乎无声的采集，是录音“毫无反应”最常见的原因
        "peak": round(float(np.max(np.abs(x))), 4),
        "rms": round(float(np.sqrt(np.mean(x**2))), 5),
        "clipped_samples": _clipped_samples(x),
        # 交给 run_wav，以便按频率指认被裁剪的载波；
        # 在 meta 到达调用方之前会被弹出。
        "_carriers": [round(float(f), 1) for f in carriers],
    }
    return amp, times, meta


# ---------------------------------------------------------------------------
# 录音质检
# ---------------------------------------------------------------------------

_STATUS_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


def _check(name: str, status: str, detail: str) -> dict:
    return {"name": name, "status": status, "detail": detail}


def check_audio(
    path: str | Path,
    fs: int | None = None,
    hw_f_max: float | None = None,
    max_seconds: float | None = None,
) -> dict:
    """判定前的录音质量门：只检查音频前端，不运行检测器。

    为什么需要这道门：``run_wav`` 的判决依赖帧同步、载波频带与电平，任何一项
    不合格时检测器仍会给出一个数值，却无法区分“场景里没人”和“录音本身废了”。
    这里把能提前查出的失效逐项标成 PASS / WARN / FAIL，FAIL 时不应继续判定。

    ``fs`` 仅用于重采样试听；真实采集应保持原生采样率，所以原生采样率低于
    48 kHz 一律判 FAIL，即使重采样后能跑通。

    ``max_seconds`` 只检查文件开头的一段；时长一项仍按文件头报告的完整时长
    判定，因此截取分析不会把一段长录音误判成“太短”。
    """
    path = Path(path)
    checks: list[dict] = []

    try:
        with wave.open(str(path), "rb") as w:
            native_fs = w.getframerate()
            channels = w.getnchannels()
            width = w.getsampwidth()
            n_frames = w.getnframes()
        x, used_fs = read_wav(path, target_fs=fs, max_seconds=max_seconds)
    except (wave.Error, RuntimeError) as exc:
        return {
            "path": str(path),
            "status": "FAIL",
            "passed": False,
            "checks": [_check("格式", "FAIL", str(exc))],
            "meta": None,
            "sync": None,
            "error": str(exc),
        }

    duration = float(n_frames / native_fs) if native_fs else 0.0
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
    clipped = _clipped_samples(x)

    depth = {1: "8-bit", 2: "16-bit", 3: "24-bit", 4: "32-bit"}.get(width)
    fmt_detail = (f"WAV PCM {depth or '未知位深'} · {native_fs} Hz · "
                  f"{'单声道' if channels == 1 else f'{channels} 声道（自动下混）'}")
    checks.append(_check("格式", "PASS" if depth else "FAIL", fmt_detail))

    fs_detail = f"{native_fs} Hz"
    if fs is not None and used_fs != native_fs:
        fs_detail += f"，已重采样到 {used_fs} Hz"
    if native_fs < 48000:
        fs_detail += "，低于 48000 Hz"
    checks.append(_check("采样率", "PASS" if native_fs >= 48000 else "FAIL",
                         fs_detail))

    if duration >= 20.0:
        dur_status = "PASS"
    elif duration >= 12.0:
        dur_status = "WARN"
    else:
        dur_status = "FAIL"
    checks.append(_check("时长", dur_status, f"{duration:.2f} s（建议 ≥20 s）"))

    if clipped > 0:
        lvl_status = "FAIL"
        lvl_detail = f"峰值 {peak:.3f} · 连续削波 {clipped} 样本"
    elif peak < 0.02:
        lvl_status = "WARN"
        lvl_detail = f"峰值 {peak:.3f} · RMS {rms:.3f}（电平过低）"
    elif peak > 0.98:
        # 满刻度不等于削波：归一化波形每个周期的峰值本就会落在量化上限。
        lvl_status = "PASS"
        lvl_detail = f"峰值 {peak:.3f} · RMS {rms:.3f} · 满刻度，无连续削波"
    else:
        lvl_status = "PASS"
        lvl_detail = f"峰值 {peak:.3f} · RMS {rms:.3f} · 无连续削波"
    checks.append(_check("电平", lvl_status, lvl_detail))

    meta = None
    try:
        amp, _, meta = load_csi(
            path, fs=fs, hw_f_max=hw_f_max, max_seconds=max_seconds
        )
    except RuntimeError as exc:
        checks.append(_check("帧同步", "FAIL", str(exc)))
    else:
        sync = meta["sync"]
        ratio = float(sync["match_ratio"])
        if ratio >= 12.0:
            sync_status = "PASS"
        elif ratio >= 9.0:
            sync_status = "WARN"
        else:
            sync_status = "FAIL"
        checks.append(_check(
            "帧同步", sync_status,
            f"周期 {sync['period_samples']} 采样 · 匹配质量 {ratio:.1f}（阈值 9）",
        ))

        nominal = int(round(used_fs / CSI_FS))
        drift = abs(int(sync["period_samples"]) - nominal)
        std = float(sync.get("period_std_samples", 0.0))
        strict = float(sync.get("strict_ratio", 0.0))
        # 中位数取整永远等于标称周期，必须看间隔分布本身。真机的播放与
        # 采集时钟不同源，实测严格 960 的占比只有 47–86%；同一台设备的
        # 回环或仿真则 ≥99%。
        if drift > 2:
            period_status = "FAIL"
        elif strict < 0.95:
            period_status = "FAIL"
        elif strict < 0.99:
            period_status = "WARN"
        else:
            period_status = "PASS"
        checks.append(_check(
            "周期", period_status,
            f"{sync['period_samples']} 采样/周期 · 标称 {nominal} "
            f"· 间隔 σ {std:.2f} 采样 · 严格占比 {strict*100:.1f}%",
        ))

        frames = int(meta["frames"])
        filled, holes = int(sync["filled"]), int(sync["holes"])
        dropped = int(sync.get("dropped", 0))
        coverage = float(sync["coverage"])
        if (holes > 0 or coverage < 0.95
                or dropped > max(10, 0.02 * frames)
                or filled > max(10, 0.02 * frames)):
            cont = "FAIL"
        elif (dropped > max(2, 0.005 * frames) or coverage < 0.98
              or filled > max(2, 0.005 * frames)):
            cont = "WARN"
        else:
            cont = "PASS"
        checks.append(_check(
            "连续性", cont,
            f"{frames} 帧 · 补齐 {filled} · 丢弃伪峰 {dropped} · 空洞 {holes} "
            f"· 覆盖 {coverage*100:.1f}% · 栅格偏差 {sync.get('max_deviation_ms', 0):.1f} ms",
        ))

        # 探头裕量来自理想探测信号的频谱，只能说明频带规划合理；真正决定
        # 能不能用的是录音里每个载波 bin 的实际功率。手机扬声器/麦克风在
        # 18 kHz 附近滚降时，边缘载波会低 20 dB 以上，相当于只用了少数几路。
        level = np.asarray(amp).mean(axis=0)
        spread = 20.0 * np.log10(
            max(float(level.min()), 1e-12) / max(float(level.max()), 1e-12)
        )
        meta["carrier_spread_db"] = round(float(spread), 2)
        margin = float(meta["probe_db_min"])
        # 只作警告，不作硬门：载波功率差同时包含硬件滚降与多径零点，二者
        # 无法从单次录音里区分。实测仿真车厢的合法基线也会出现 -38 dB 的
        # 多径零点；若按 -10 dB 硬拦，会把正常录音全部拒掉。真正的硬门是
        # 帧间隔分布——多径不会改变探测音的帧栅格。
        band_status = "PASS" if spread >= -10.0 else "WARN"
        checks.append(_check(
            "频带", band_status,
            f"{meta['carriers']} 载波 @ {meta['carrier_span_hz'][0]:.0f}–"
            f"{meta['carrier_span_hz'][1]:.0f} Hz · 实测跨度 {spread:.1f} dB "
            f"· 探头裕量 {margin:.1f} dB",
        ))

    if meta is not None:
        meta.pop("_carriers", None)

    worst = max((c["status"] for c in checks), key=_STATUS_ORDER.__getitem__)
    return {
        "path": str(path),
        "status": worst,
        "passed": worst != "FAIL",
        "checks": checks,
        "meta": meta,
        "sync": None if meta is None else meta["sync"],
        "error": None,
    }


# ---------------------------------------------------------------------------
# WAV -> 判决
# ---------------------------------------------------------------------------

def run_wav(
    path: str | Path,
    sensitivity: float = 1.0,
    fs: int | None = None,
    probe: Probe | None = None,
    carriers: np.ndarray | None = None,
    cfg: DetectorConfig | None = None,
    hw_f_max: float | None = None,
    background: dict | None = None,
    require_qc: bool = True,
) -> dict:
    """让一段录制的 WAV 走完整个检测链路。

    ``background`` 是空车标定基线；实采若不做标定就只能冷启动自适应，
    两者得到的 δ₀ 与相干门限不同，必须显式选择而不是隐式回退。

    ``require_qc`` 为真时先过录音质检。质检 FAIL 意味着帧时间轴、载波
    功率或格式已经不可信，此时再输出“存在/呼吸”结论只会给操作员一个
    看似精确的错误答案，因此直接拒绝，而不是降级。
    """
    if require_qc:
        qc = check_audio(path, fs=fs, hw_f_max=hw_f_max)
        if not qc["passed"]:
            failed = [c for c in qc["checks"] if c["status"] == "FAIL"]
            detail = "；".join(f"{c['name']}（{c['detail']}）" for c in failed)
            raise RuntimeError(f"{Path(path).name}: 输入不可信，请重录。未通过：{detail}")
    amp, times, meta = load_csi(
        path, fs=fs, probe=probe, carriers=carriers, hw_f_max=hw_f_max
    )
    cfg = cfg or DetectorConfig(sensitivity=sensitivity)
    pipe = SASPipeline(cfg, fs=float(meta["csi_fs"]), background=background)

    frames: list[dict] = []
    for t, a in zip(times, amp):
        frame = pipe.step(float(t), a)
        if frame is not None:
            frames.append(frame)

    if pipe.carrier_mask is not None:
        meta["carriers_planned"] = int(pipe.carrier_mask.size)
        meta["carriers_used"] = int(pipe.carrier_mask.sum())
        meta["carriers_pruned"] = [
            round(float(f), 1)
            for f, ok in zip(meta.get("_carriers", []), pipe.carrier_mask)
            if not ok
        ]
        meta["prune_threshold"] = round(float(pipe.prune_threshold), 4)
    meta.pop("_carriers", None)

    return {
        "source": meta,
        "frames": frames,
        "events": pipe.detector.events,
        "metrics": summarise(frames, meta),
        "detector_config": cfg.to_dict(),
    }


def summarise(frames: list[dict], meta: dict | None = None) -> dict:
    """无需真值的指标。

    真实座舱没有带标注的 BPM，因此与 ``evaluate_frames`` 不同，这里报告的是
    检测器判定了什么、在何时判定 —— 而不是准确率。
    """
    if not frames:
        return {"frames": 0, "detected": False}

    presence_frame = next((f for f in frames if f["state"]["presence"]), None)
    alarm_frame = next((f for f in frames if f["state"]["alarm"]), None)
    apnea_frame = next((f for f in frames if f["state"]["alarm_reason"] == "APNEA"), None)
    bpm_frame = next((f for f in frames if f["state"]["bpm"] is not None), None)

    bpms = [f["state"]["bpm"] for f in frames if f["state"]["bpm"] is not None]
    first_evidence = None if presence_frame is None else presence_frame["state"].get("first_evidence_t")
    alarm_at = None if alarm_frame is None else alarm_frame["t"]

    neff = []
    for f in frames:
        g = np.asarray(f["features"].get("gains") or [], dtype=float)
        if g.size == 0:
            continue
        w = np.maximum(g, 0.0) + MRC_WEIGHT_FLOOR
        total = w.sum()
        if total > 0:
            w = w / total
            neff.append(float(1.0 / np.sum(w ** 2)))

    out = {
        "frames": len(frames),
        "duration_s": frames[-1]["t"] if frames else 0.0,
        "detected": presence_frame is not None,
        "presence_at": None if presence_frame is None else presence_frame["t"],
        "confirmed_by": None if presence_frame is None else presence_frame["state"].get("breath_source"),
        "alarm_at": alarm_at,
        "alarm_reason": None if alarm_frame is None else alarm_frame["state"]["alarm_reason"],
        # 预算是端到端的：从开始监测（t=0）起算，因此响应时间就是告警时刻。
        "response_time_s": None if alarm_at is None else round(alarm_at, 2),
        "apnea_escalated_at": None if apnea_frame is None else apnea_frame["t"],
        "breathing_at": None if bpm_frame is None else bpm_frame["t"],
        "bpm_median": None if not bpms else round(float(np.median(bpms)), 1),
        "bpm_range": None if not bpms else [round(float(min(bpms)), 1), round(float(max(bpms)), 1)],
        "bpm_samples": len(bpms),
        "motion_evidence_frames": sum(1 for f in frames if f["state"].get("motion_detected")),
        "interference_frames": sum(1 for f in frames if f["features"].get("interference")),
        "max_motion_ratio": round(
            float(max(f["state"].get("motion_ratio") or 0.0 for f in frames)), 2
        ),
        "mrc_neff": None if not neff else round(float(np.mean(neff)), 2),
    }
    if meta is not None:
        out["source"] = {
            "fs": meta.get("fs"),
            "csi_fs": meta.get("csi_fs"),
            "carriers": meta.get("carriers"),
            "duration_s": meta.get("duration_s"),
            "sync_filled": (meta.get("sync") or {}).get("filled"),
            "sync_holes": (meta.get("sync") or {}).get("holes"),
        }
    return out


# ---------------------------------------------------------------------------
# 自检：场景 -> WAV
# ---------------------------------------------------------------------------

def _default_seed(key: str) -> int:
    return 20231102 + zlib.crc32(key.encode("utf-8")) % 997


def render_wav(
    path: str | Path,
    key: str = "sleeping",
    duration_s: float | None = None,
    fs: int = AUDIO_FS,
    snr_db: float = 40.0,
    seed: int | None = None,
    hw_f_max: float | None = None,
) -> dict:
    """把仿真场景渲染成真实前端可读取的 WAV。

    探测信号与该场景连续频率的 CIR 相卷积，只保留正频率频带，这恰好就是仿真器
    为其 24 个载波所使用的解析信道模型。因此把结果再经由 ``run_wav`` 回放，
    检验的是前端，而不是场景。

    ``snr_db`` 是相对于渲染信号 RMS 的带内 SNR。仿真器自身的 CSI 噪声约为
    35 dB，因此这里约 40 dB 与之相当。

    ``hw_f_max`` 会让渲染出的采集如同带限驱动器所产生的那样：探测信号本身被收
    窄，因此上限之上不会浪费能量。那是*配置正确*的情形。*未配置*的情形 ——
    全频带探测信号经过一个低通驱动器 —— 是另一种失效模式，值得单独测试，
    因为那里的死载波注入的是噪声，而不只是消失；``SASPipeline`` 会在运行时
    通过裁剪捕获它。
    """
    scenario = get_scenario(key)
    band = plan_band(hw_f_max)
    ch = AcousticChannel(
        scenario,
        seed=seed if seed is not None else _default_seed(key),
        band=band,
    )
    probe = make_probe(fs=fs, hw_f_max=hw_f_max)
    L = probe.length
    spectrum = np.fft.fft(probe.samples)
    freqs = np.fft.fftfreq(L, 1.0 / fs)

    n = int(round((duration_s or scenario.duration_s) * CSI_FS))
    out = np.empty(n * L)
    truth: list[dict] = []
    for i in range(n):
        delays, gains, tr = ch.sounding_step(1.0 / CSI_FS)
        H = (
            gains[:, None] * np.exp(-1j * 2 * np.pi * freqs[None, :] * delays[:, None])
        ).sum(axis=0)
        # 只保留正频带并镜像，使波形为实数，且其在 f > 0 处的 CFR 精确等于 H
        #（复路径增益不是实信号，重新对称化才能让它可播放）
        full = np.zeros(L, dtype=complex)
        full[: L // 2 + 1] = (spectrum * H)[: L // 2 + 1]
        full[L // 2 + 1:] = np.conj(full[L // 2 - 1: 0: -1])
        out[i * L: (i + 1) * L] = np.fft.ifft(full).real
        truth.append(tr)

    rng = np.random.default_rng(seed if seed is not None else _default_seed(key))
    if snr_db is not None and np.isfinite(snr_db):
        rms = float(np.sqrt(np.mean(out ** 2)))
        gain = np.repeat([t["noise_gain"] for t in truth], L)[: out.size]
        out = out + rng.standard_normal(out.size) * rms * 10 ** (-snr_db / 20.0) * gain

    peak = float(np.max(np.abs(out)))
    if peak > 0.98:
        out *= 0.98 / peak
    write_wav(path, out, fs=fs)

    return {
        "path": str(path),
        "scenario": key,
        "seconds": round(n / CSI_FS, 2),
        "fs": fs,
        "snr_db": snr_db,
        "peak": round(float(np.max(np.abs(out))), 3),
        "band": band.describe(),
        "carriers": band.n_carriers,
        "sweep_hz": [round(band.sweep_lo, 1), round(band.sweep_hi, 1)],
    }
