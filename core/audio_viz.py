"""录音音频可视化：把一段 WAV 拆成波形、频谱、CSI 与判定趋势。

两条展示路径共用本模块的分析结果：

* Web UI 的「音频可视化」页上传 WAV，服务端调用 :func:`analyze_audio` 返回 JSON；
* ``python replay_audio.py capture.wav --viz`` 用 matplotlib 打开本地窗口。

分析的顺序与检测链路一致：先按 :func:`core.replay.check_audio` 做录音质检，再用
Kasami 探测音做帧同步、逐载波反演 CSI，最后把同一串 CSI 喂给 ``SASPipeline``。
因此图上看到的是检测器真正使用的量，而不是另一套演示数据。
"""
from __future__ import annotations

import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import welch

from .audio import (
    carrier_bins,
    estimate_csi,
    make_probe,
    probe_condition_db,
    read_wav,
)
from .config import DetectorConfig
from .pipeline import SASPipeline
from .replay import check_audio, load_csi, summarise

# 数组下采样上限：前端 Canvas 与 JSON 传输都不需要原始 48 kHz 采样。
WAVE_POINTS = 1600
SPECTRUM_POINTS = 1400
FRAME_POINTS = 700


def _round_list(values, digits: int = 3) -> list[float]:
    return [round(float(v), digits) for v in values]


def _envelope(x: np.ndarray, fs: float, target: int = WAVE_POINTS) -> dict:
    """按块取最大值/最小值，保留瞬态的同时把点数压到 Canvas 可承受的量级。"""
    n = int(x.size)
    if n == 0:
        return {"t": [], "max": [], "min": [], "rms": []}
    step = max(1, int(np.ceil(n / target)))
    blocks = n // step
    if blocks == 0:
        return {"t": [0.0], "max": [float(abs(x[0]))], "min": [float(x[0])], "rms": []}
    view = x[: blocks * step].reshape(blocks, step)
    mid = (np.arange(blocks, dtype=float) * step + 0.5 * step) / fs
    rms = np.sqrt(np.mean(view ** 2, axis=1))
    return {
        "t": _round_list(mid, 4),
        "max": _round_list(view.max(axis=1)),
        "min": _round_list(view.min(axis=1)),
        "rms": _round_list(rms),
    }


def _spectrum(x: np.ndarray, fs: float, target: int = SPECTRUM_POINTS) -> dict:
    """Welch 功率谱，按频点块平均后返回 dB。"""
    nper = min(8192, int(x.size))
    if nper < 256:
        return {"f_hz": [], "db": []}
    freqs, power = welch(
        x, fs=fs, nperseg=nper, noverlap=nper // 2, detrend=False, scaling="density"
    )
    db = 10.0 * np.log10(np.maximum(power, 1e-20))
    if freqs.size > target:
        step = int(np.ceil(freqs.size / target))
        m = freqs.size // step
        freqs = freqs[: m * step].reshape(m, step).mean(axis=1)
        db = db[: m * step].reshape(m, step).mean(axis=1)
    return {"f_hz": _round_list(freqs, 1), "db": _round_list(db, 2)}


def _downsample_frames(amp: np.ndarray, times: np.ndarray, target: int = FRAME_POINTS):
    if amp.shape[0] == 0:
        return amp, times, np.zeros(0, dtype=int)
    stride = max(1, int(np.ceil(amp.shape[0] / target)))
    idx = np.arange(0, amp.shape[0], stride)
    return amp[idx], times[idx], idx


def _prepare_upload(data: bytes) -> Path:
    """把上传的 WAV 落到临时文件，供 ``wave``/``scipy`` 读取。"""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        tmp.write(data)
    finally:
        tmp.close()
    return Path(tmp.name)


def _wav_duration_s(path: str) -> float | None:
    """从 WAV 头读取完整时长，用于在截取分析时仍能显示原始长度。"""
    try:
        with wave.open(path, "rb") as w:
            fs = w.getframerate()
            return round(float(w.getnframes()) / float(fs), 2) if fs else None
    except (wave.Error, OSError):
        return None


def analyze_bytes(
    data: bytes,
    hw_f_max: float | None = None,
    max_seconds: float | None = None,
) -> dict:
    """分析上传的 WAV 字节；调用方负责删除临时文件。"""
    path = _prepare_upload(data)
    try:
        return analyze_audio(path, hw_f_max=hw_f_max, max_seconds=max_seconds)
    finally:
        try:
            path.unlink()
        except OSError:
            pass


def analyze_audio(
    path: str | Path,
    hw_f_max: float | None = None,
    sensitivity: float = 1.0,
    fs: int | None = None,
    max_seconds: float | None = None,
) -> dict:
    """分析一段录音并返回可直接 JSON 序列化的可视化数据。

    ``max_seconds`` 只分析文件开头的一段，避免几分钟的录音在页面上等待过久；
    默认 ``None`` 分析完整文件。
    """
    path = str(path)
    qc = check_audio(path, fs=fs, hw_f_max=hw_f_max, max_seconds=max_seconds)
    x, native_fs = read_wav(path, target_fs=fs, max_seconds=max_seconds)
    waveform = _envelope(x, float(native_fs))
    spectrum = _spectrum(x, float(native_fs))
    duration_s = round(float(x.size) / float(native_fs), 2)
    file_duration_s = _wav_duration_s(path)
    window = None
    if file_duration_s is not None and duration_s + 0.05 < file_duration_s:
        window = {"analyzed_s": duration_s, "file_s": file_duration_s}

    probe = make_probe(fs=native_fs, hw_f_max=hw_f_max)
    band = probe.band
    carrier_hz = carrier_bins(probe)
    probe_gain = probe_condition_db(probe, carrier_hz)

    result: dict[str, Any] = {
        "path": path,
        "fs": int(native_fs),
        "duration_s": duration_s,
        "file_duration_s": file_duration_s,
        "qc": qc,
        "waveform": waveform,
        "spectrum": spectrum,
        "band": {
            "describe": band.describe(),
            "f_lo": round(float(band.f_lo), 1),
            "f_hi": round(float(band.f_hi), 1),
            "sweep_lo": round(float(band.sweep_lo), 1),
            "sweep_hi": round(float(band.sweep_hi), 1),
            "n_carriers": int(band.n_carriers),
            "hw_f_max": round(float(band.hw_f_max), 1),
        },
        "probe": {
            "period_samples": int(probe.length),
            "period_ms": round(float(probe.period_s * 1000), 3),
            "csi_fs": round(float(probe.csi_fs), 3),
            "kind": probe.kind,
            "carriers": _round_list(carrier_hz, 1),
            "gain_db": _round_list(probe_gain, 2),
        },
    }

    if not qc["passed"] and qc.get("error"):
        result["error"] = qc["error"]
        result["csi"] = {"t": [], "amp": []}
        result["carriers"] = {"hz": _round_list(carrier_hz, 1), "power_db": [], "used": []}
        result["trends"] = {}
        result["events"] = []
        result["summary"] = {"detected": False}
        result["params"] = _params(qc, {}, None, window)
        return result

    try:
        amp, times, meta = load_csi(
            path, fs=fs, hw_f_max=hw_f_max, max_seconds=max_seconds
        )
    except RuntimeError as exc:
        result["error"] = str(exc)
        result["csi"] = {"t": [], "amp": []}
        result["carriers"] = {"hz": _round_list(carrier_hz, 1), "power_db": [], "used": []}
        result["trends"] = {}
        result["events"] = []
        result["summary"] = {"detected": False}
        result["params"] = _params(qc, {}, None, window)
        return result

    cfg = DetectorConfig(sensitivity=sensitivity)
    pipe = SASPipeline(cfg, fs=float(meta["csi_fs"]))
    frames: list[dict] = []
    for t, a in zip(times, amp):
        frame = pipe.step(float(t), a)
        if frame is not None:
            frames.append(frame)

    used = pipe.carrier_mask
    if used is None:
        used = np.ones(amp.shape[1], dtype=bool)
    power = np.mean(amp ** 2, axis=0)
    power_db = 10.0 * np.log10(np.maximum(power, 1e-20) / max(float(power.max()), 1e-20))

    amp_s, times_s, _ = _downsample_frames(amp, times)
    trends = _trends(frames)
    summary = summarise(frames, meta)
    coherence = _coherence_median(frames)

    result.update(
        {
            "csi": {
                "t": _round_list(times_s, 3),
                "amp": [[round(float(v), 3) for v in row] for row in amp_s],
                "carrier_hz": _round_list(carrier_hz, 1),
            },
            "carriers": {
                "hz": _round_list(carrier_hz, 1),
                "power_db": _round_list(power_db, 2),
                "used": [bool(u) for u in used],
            },
            "trends": trends,
            "events": _events(pipe.detector.events),
            "summary": summary,
            "processing": {
                "frames": int(len(frames)),
                "filled": int(meta["sync"].get("filled", 0)),
                "holes": int(meta["sync"].get("holes", 0)),
                "mrc_neff": summary.get("mrc_neff"),
            },
            "params": _params(qc, summary, result["probe"], window, coherence),
        }
    )
    return result


def _events(events: list[dict]) -> list[dict]:
    out = []
    for ev in events:
        out.append(
            {
                "t": round(float(ev.get("t", 0.0)), 2),
                "kind": str(ev.get("kind", "")),
                "message": str(ev.get("message", "")),
            }
        )
    return out


def _coherence_median(frames: list[dict]) -> float | None:
    """整段录音的运动相干性中位数，作为参数表一行保留。"""
    vals = [
        float(f["features"]["motion_coherence"])
        for f in frames
        if f["features"].get("motion_coherence") is not None
    ]
    return round(float(np.median(vals)), 3) if vals else None


def _trends(frames: list[dict]) -> dict:
    if not frames:
        return {}
    stride = max(1, int(np.ceil(len(frames) / FRAME_POINTS)))
    picked = frames[::stride]
    return {
        "t": [round(float(f["t"]), 3) for f in picked],
        "motion_ratio": [
            None if f["state"].get("motion_ratio") is None
            else round(float(f["state"]["motion_ratio"]), 3)
            for f in picked
        ],
        "coherence": [
            None if f["features"].get("motion_coherence") is None
            else round(float(f["features"]["motion_coherence"]), 4)
            for f in picked
        ],
        "bpm": [
            None if f["state"].get("bpm") is None
            else round(float(f["state"]["bpm"]), 1)
            for f in picked
        ],
        "motion": [bool(f["state"].get("motion_detected")) for f in picked],
        "interference": [bool(f["features"].get("interference")) for f in picked],
        "threshold": (
            None if picked[-1]["state"].get("motion_threshold") is None
            else round(float(picked[-1]["state"]["motion_threshold"]), 3)
        ),
    }


def _params(
    qc: dict,
    summary: dict,
    probe: dict | None,
    window: dict | None = None,
    coherence: float | None = None,
) -> list[dict]:
    """把质检项与判决摘要整理成一张可读的参数表。"""
    params = [
        {"name": item["name"], "value": item["detail"], "status": item["status"]}
        for item in qc.get("checks", [])
    ]
    if window:
        params.append(
            {
                "name": "分析长度",
                "value": (
                    f"前 {window['analyzed_s']:.1f} s / 共 {window['file_s']:.1f} s"
                ),
                "status": "INFO",
            }
        )
    if probe:
        params.append(
            {
                "name": "CSI 帧率",
                "value": f"{probe['csi_fs']:.1f} Hz · 周期 {probe['period_ms']:.0f} ms",
                "status": "INFO",
            }
        )
    if summary:
        detected = summary.get("detected")
        presence = summary.get("presence_at")
        params.extend(
            [
                {
                    "name": "存在判定",
                    "value": (f"检测到儿童 @ {presence:.1f} s" if detected else "未检测到"),
                    "status": "WARN" if detected else "PASS",
                },
                {
                    "name": "告警",
                    "value": (
                        f"{summary.get('alarm_reason') or 'PRESENCE'} @ "
                        f"{summary.get('alarm_at'):.1f} s"
                        if summary.get("alarm_at") is not None
                        else "—"
                    ),
                    "status": "WARN" if summary.get("alarm_at") is not None else "PASS",
                },
                {
                    "name": "呼吸率",
                    "value": (
                        f"{summary.get('bpm_median'):.1f} BPM"
                        if summary.get("bpm_median") is not None
                        else "—"
                    ),
                    "status": "INFO",
                },
                {
                    "name": "运动证据",
                    "value": (
                        f"{summary.get('motion_evidence_frames', 0)} 帧 / "
                        f"{summary.get('frames', 0)}"
                    ),
                    "status": "INFO",
                },
                {
                    "name": "MRC 有效载波",
                    "value": (
                        f"{summary.get('mrc_neff'):.2f} 个" if summary.get("mrc_neff") else "—"
                    ),
                    "status": "INFO",
                },
            ]
        )
        if coherence is not None:
            params.append(
                {
                    "name": "运动相干性 κ",
                    "value": f"中位数 {coherence:.3f}",
                    "status": "INFO",
                }
            )
    return params


# ---------------------------------------------------------------------------
# 本地 matplotlib 窗口
# ---------------------------------------------------------------------------

_ZH_FONTS = (
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Microsoft YaHei",
    "PingFang SC",
    "WenQuanYi Zen Hei",
    "SimHei",
)


def _font_family() -> str:
    from matplotlib.font_manager import fontManager

    available = {f.name for f in fontManager.ttflist}
    for name in _ZH_FONTS:
        if name in available:
            return name
    return "DejaVu Sans"


def show_gui(path: str | Path, hw_f_max: float | None = None,
             fs: int | None = None) -> None:
    """用 matplotlib 打开一个本地分析窗口（需要图形环境）。"""
    import matplotlib.pyplot as plt

    font = _font_family()
    zh = font != "DejaVu Sans"
    plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    data = analyze_audio(path, hw_f_max=hw_f_max, fs=fs)
    band = data["band"]
    carrier_hz = np.asarray(data["probe"]["carriers"])
    fig = plt.figure(figsize=(15.5, 10.5), facecolor="#0d141c")
    gs = fig.add_gridspec(3, 2, left=0.07, right=0.97, top=0.91, bottom=0.11,
                          hspace=0.42, wspace=0.24)
    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]
    for ax in axes:
        ax.set_facecolor("#111a24")
        ax.tick_params(colors="#8b9bb0", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#223141")
        ax.grid(color="#1d2b3a", linewidth=0.6, alpha=0.8)

    labels = (
        dict(wave="波形包络", spec="频谱与探测带", csi="CSI 幅度热力图",
             carrier="每载波功率", motion="运动能量比 / 相干性", bpm="呼吸率趋势",
             x_time="时间 (s)", x_freq="频率 (kHz)", y_amp="幅度",
             y_db="dB", y_motion="能量比", y_bpm="BPM", params="关键参数")
        if zh else
        dict(wave="Waveform envelope", spec="Spectrum & probe band", csi="CSI amplitude",
             carrier="Per-carrier power", motion="Motion ratio / coherence", bpm="Breathing rate",
             x_time="time (s)", x_freq="frequency (kHz)", y_amp="amplitude",
             y_db="dB", y_motion="ratio", y_bpm="BPM", params="key parameters")
    )

    wave = data["waveform"]
    if wave["t"]:
        axes[0].fill_between(wave["t"], wave["min"], wave["max"], color="#4ea1ff", alpha=0.55, linewidth=0)
        axes[0].plot(wave["t"], wave["rms"], color="#f2b35c", linewidth=1.1)
        axes[0].plot(wave["t"], [-v for v in wave["rms"]], color="#f2b35c", linewidth=1.1, alpha=0.6)
    axes[0].set_title(labels["wave"], color="#dfe9f5", fontsize=10)
    axes[0].set_xlabel(labels["x_time"], color="#8b9bb0", fontsize=8)
    axes[0].set_ylabel(labels["y_amp"], color="#8b9bb0", fontsize=8)

    spec = data["spectrum"]
    if spec["f_hz"]:
        fk = np.asarray(spec["f_hz"]) / 1000.0
        axes[1].plot(fk, spec["db"], color="#68d6a8", linewidth=1.0)
        axes[1].axvspan(band["sweep_lo"] / 1000, band["sweep_hi"] / 1000,
                        color="#4ea1ff", alpha=0.12)
        for f in carrier_hz:
            axes[1].axvline(f / 1000, color="#4ea1ff", linewidth=0.5, alpha=0.35)
    axes[1].set_title(labels["spec"], color="#dfe9f5", fontsize=10)
    axes[1].set_xlabel(labels["x_freq"], color="#8b9bb0", fontsize=8)
    axes[1].set_ylabel(labels["y_db"], color="#8b9bb0", fontsize=8)

    csi = data["csi"]
    if csi["amp"]:
        amp = np.asarray(csi["amp"])
        axes[2].imshow(amp, aspect="auto", origin="lower", cmap="viridis",
                       extent=[0, data["duration_s"], 0, amp.shape[1]])
        axes[2].set_title(labels["csi"], color="#dfe9f5", fontsize=10)
        axes[2].set_xlabel(labels["x_time"], color="#8b9bb0", fontsize=8)
        axes[2].set_ylabel("carrier #", color="#8b9bb0", fontsize=8)

    car = data["carriers"]
    if car["power_db"]:
        used = np.asarray(car["used"], dtype=bool)
        colors = ["#4ea1ff" if u else "#ff6070" for u in used]
        axes[3].bar(np.arange(len(car["power_db"])), car["power_db"], color=colors, width=0.75)
        axes[3].set_title(labels["carrier"], color="#dfe9f5", fontsize=10)
        axes[3].set_ylabel(labels["y_db"], color="#8b9bb0", fontsize=8)
        axes[3].set_xlabel("carrier #", color="#8b9bb0", fontsize=8)

    tr = data["trends"]
    if tr:
        ratio = [np.nan if v is None else v for v in tr["motion_ratio"]]
        coh = [np.nan if v is None else v for v in tr["coherence"]]
        axes[4].plot(tr["t"], ratio, color="#f2b35c", linewidth=1.1, label="ratio")
        axes[4].set_ylabel(labels["y_motion"], color="#f2b35c", fontsize=8)
        twin = axes[4].twinx()
        twin.plot(tr["t"], coh, color="#4ea1ff", linewidth=1.0, label="coherence")
        twin.set_ylim(0, 1)
        twin.tick_params(colors="#8b9bb0", labelsize=8)
        twin.spines["right"].set_color("#223141")
        axes[4].set_title(labels["motion"], color="#dfe9f5", fontsize=10)
        axes[4].set_xlabel(labels["x_time"], color="#8b9bb0", fontsize=8)

        bpm = [np.nan if v is None else v for v in tr["bpm"]]
        axes[5].plot(tr["t"], bpm, color="#68d6a8", linewidth=1.2)
        axes[5].set_ylim(0, 80)
        axes[5].set_title(labels["bpm"], color="#dfe9f5", fontsize=10)
        axes[5].set_xlabel(labels["x_time"], color="#8b9bb0", fontsize=8)
        axes[5].set_ylabel(labels["y_bpm"], color="#8b9bb0", fontsize=8)

    status = data["qc"]["status"]
    color = {"PASS": "#68d6a8", "WARN": "#f2b35c", "FAIL": "#ff6070"}.get(status, "#8b9bb0")
    fig.suptitle(
        f"{Path(path).name} · {'录音质检' if zh else 'QC'} {status} · "
        f"{data['fs']} Hz · {data['duration_s']:.1f} s · "
        f"{band['n_carriers']} carriers @ {band['f_lo']/1000:.1f}-{band['f_hi']/1000:.1f} kHz",
        color=color, fontsize=11.5, x=0.07, ha="left",
    )
    key = [f"{p['name']}: {p['value']}" for p in data["params"][:8]]
    fig.text(0.07, 0.045, "  |  ".join(key), color="#95a7bd", fontsize=8.2, ha="left")
    if data.get("error"):
        fig.text(0.07, 0.02, data["error"], color="#ff6070", fontsize=8.5, ha="left")
    fig.subplots_adjust(bottom=0.13)
    plt.show()
