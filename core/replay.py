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

def load_csi(
    path: str | Path,
    fs: int | None = None,
    probe: Probe | None = None,
    carriers: np.ndarray | None = None,
    min_peak_ratio: float = 0.2,
    hw_f_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """读取一个 WAV，并返回 ``(amplitude, times, meta)``。

    ``amplitude`` 的形状为 ``(n_frames, n_carriers)``；``times`` 是文件中的真实
    位置，因此丢失的帧会表现为时间上的真实空隙，而不是无声无息地把序列拉伸。

    ``hw_f_max`` 依据硬件可用上限来规划载波频带，而不是采用论文的 23.5 kHz。
    不传它时，16 kHz 的驱动器会被要求提供它无法发出的载波 —— 见
    ``config.plan_band``。
    """
    x, native_fs = read_wav(path, target_fs=fs)
    band = plan_band(hw_f_max)
    if 2 * band.sweep_hi >= native_fs:
        raise RuntimeError(
            f"{Path(path).name}: sample rate is {native_fs} Hz, but the "
            f"{band.sweep_hi/1000:.1f} kHz sweep needs at least "
            f"{2*band.sweep_hi/1000:.1f} kHz of Nyquist bandwidth. Record at 48 kHz."
        )
    probe = probe or make_probe(fs=native_fs, hw_f_max=hw_f_max)

    probe = probe or make_probe(fs=native_fs)
    starts, sync = frame_offsets(x, probe, min_ratio=min_peak_ratio)
    if starts.size < 2 or sync["match_ratio"] < 9.0:
        raise RuntimeError(
            f"{Path(path).name}: no probe repetition found "
            f"(correlation peak/floor {sync['match_ratio']:.1f}, need >= 9; a "
            f"recording that contains the Kasami probe scores around 12–18). The probe "
            f"must be playing during the recording, at its original "
            f"{probe.period_s*1000:.1f} ms period -- no resampling, no tempo "
            f"change, no noise suppression. Run "
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
        "clipped_samples": int(np.sum(np.abs(x) >= 0.999)),
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
) -> dict:
    """判定前的录音质量门：只检查音频前端，不运行检测器。

    为什么需要这道门：``run_wav`` 的判决依赖帧同步、载波频带与电平，任何一项
    不合格时检测器仍会给出一个数值，却无法区分“场景里没人”和“录音本身废了”。
    这里把能提前查出的失效逐项标成 PASS / WARN / FAIL，FAIL 时不应继续判定。

    ``fs`` 仅用于重采样试听；真实采集应保持原生采样率，所以原生采样率低于
    48 kHz 一律判 FAIL，即使重采样后能跑通。
    """
    path = Path(path)
    checks: list[dict] = []

    try:
        with wave.open(str(path), "rb") as w:
            native_fs = w.getframerate()
            channels = w.getnchannels()
            width = w.getsampwidth()
            n_frames = w.getnframes()
        x, used_fs = read_wav(path, target_fs=fs)
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
    clipped = int(np.sum(np.abs(x) >= 0.999))

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
        lvl_detail = f"峰值 {peak:.3f} · 削波 {clipped} 样本"
    elif peak < 0.02:
        lvl_status = "WARN"
        lvl_detail = f"峰值 {peak:.3f} · RMS {rms:.3f}（电平过低）"
    elif peak > 0.98:
        lvl_status = "WARN"
        lvl_detail = f"峰值 {peak:.3f} · RMS {rms:.3f}（接近满刻度）"
    else:
        lvl_status = "PASS"
        lvl_detail = f"峰值 {peak:.3f} · RMS {rms:.3f} · 削波 0"
    checks.append(_check("电平", lvl_status, lvl_detail))

    meta = None
    try:
        _, _, meta = load_csi(path, fs=fs, hw_f_max=hw_f_max)
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
        checks.append(_check(
            "周期", "PASS" if drift <= 1 else ("WARN" if drift <= 2 else "FAIL"),
            f"{sync['period_samples']} 采样/周期 · 标称 {nominal}",
        ))

        frames = int(meta["frames"])
        filled, holes = int(sync["filled"]), int(sync["holes"])
        coverage = float(sync["coverage"])
        if holes > 0 or coverage < 0.95 or filled > max(10, 0.02 * frames):
            cont = "FAIL"
        elif filled > max(2, 0.005 * frames) or coverage < 0.98:
            cont = "WARN"
        else:
            cont = "PASS"
        checks.append(_check(
            "连续性", cont,
            f"{frames} 帧 · 补齐 {filled} · 空洞 {holes} · 覆盖 {coverage*100:.1f}%",
        ))

        margin = float(meta["probe_db_min"])
        if margin >= -10.0:
            band_status = "PASS"
        elif margin >= -20.0:
            band_status = "WARN"
        else:
            band_status = "FAIL"
        checks.append(_check(
            "频带", band_status,
            f"{meta['carriers']} 载波 @ {meta['carrier_span_hz'][0]:.0f}–"
            f"{meta['carrier_span_hz'][1]:.0f} Hz · 探头裕量 {margin:.1f} dB",
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
) -> dict:
    """让一段录制的 WAV 走完整个检测链路。"""
    amp, times, meta = load_csi(
        path, fs=fs, probe=probe, carriers=carriers, hw_f_max=hw_f_max
    )
    cfg = cfg or DetectorConfig(sensitivity=sensitivity)
    pipe = SASPipeline(cfg, fs=float(meta["csi_fs"]))

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
        "response_time_s": None
        if alarm_at is None or first_evidence is None
        else round(alarm_at - first_evidence, 2),
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
