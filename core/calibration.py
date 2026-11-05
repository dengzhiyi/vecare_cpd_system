"""空车标定：在确认无人的座舱里采集背景残差并持久化。

检测器在冷启动时使用自适应背景估计，因此**不要求**开机后先安静 5 s。若用户
明确执行空车标定，本模块会把空车观测段里的运动残差统计成一份基线文件，之后
每次启动直接加载，避免开机初期把低幅噪声当成运动，或在有人时才开始估计背景。

两种数据源：
    * 仿真：直接驱动 ``AcousticChannel`` 的 ``empty`` 场景（默认，用于样机自检）；
    * 实采：读取一段空车录音（必须含周期性 Kasami 探测音），经 ``core.replay.load_csi``
      解调成每载波复幅度后计算，适配层已内置。

标定只影响运动支路的判据：
    * 残差能量基线（相对能量的分母）；
    * 运动相干性阈值（由空车相干性中位数 + 5.5×MAD 得到）。
呼吸支路依赖 8 s ACF，不使用这份基线。

产物路径：``data/calibration/empty_cabin.json``（不入库）。基线按载波数区分，
频带变化时不会误用旧基线。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .channel import AcousticChannel
from .config import CSI_FS, DetectorConfig, plan_band
from .pipeline import SASPipeline
from .scenarios import get_scenario

CALIBRATION_DIR = Path(__file__).resolve().parent.parent / "data" / "calibration"
PROFILE_PATH = CALIBRATION_DIR / "empty_cabin.json"

# 与 Web/API 保持一致的硬件频响上限档位（Hz）。
BAND_CHOICES = (20000.0, 18000.0, 16000.0, 14000.0)


def _robust_threshold(values: list[float], factor: float, floor: float) -> tuple[float, float]:
    """返回 (中位数, max(下限, 中位数 + factor×MAD))。"""
    vals = np.asarray(values, dtype=float)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    return med, max(float(floor), med + factor * mad)


def _build_profile(
    energies: list[float],
    coherences: list[float],
    *,
    cfg: DetectorConfig,
    duration_s: float,
    hw_f_max: float | None,
    extra: dict | None = None,
) -> dict:
    """把空车观测段的原始统计量写成基线文件。"""
    if len(energies) < 20:
        raise ValueError("空车标定样本不足，请把采集时长延长到至少 3 s")

    band = plan_band(hw_f_max)
    med_e, _ = _robust_threshold(energies, 0.0, 1e-9)
    # 能量基线取中位数附近的低分位，避免把偶发噪声尖峰写进分母。
    baseline_energy = max(float(np.percentile(energies, 35)), 1e-9)
    med_c, coherence_threshold = _robust_threshold(
        coherences, 5.5, cfg.motion_coherence_min
    )
    coh_spread = float(np.median(np.abs(np.asarray(coherences) - med_c))) * 1.4826

    profile = {
        "profile": "empty_cabin",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "duration_s": round(float(duration_s), 2),
        "samples": len(energies),
        "hw_f_max_hz": round(float(band.hw_f_max), 1),
        "n_carriers": int(band.n_carriers),
        "band": band.describe(),
        "f_lo_hz": band.f_lo,
        "f_hi_hz": band.f_hi,
        "baseline_energy": baseline_energy,
        "energy_median": med_e,
        "coherence_median": med_c,
        "coherence_spread": coh_spread,
        "coherence_threshold": min(0.22, coherence_threshold),
        "motion_threshold": cfg.motion_ratio_floor(),
    }
    if extra:
        profile.update(extra)

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile


def calibrate_empty_cabin(
    duration_s: float = 5.0,
    cfg: DetectorConfig | None = None,
    hw_f_max: float | None = None,
    seed: int = 20231102,
) -> dict:
    """运行一段空车仿真并生成基线文件。

    默认 5 s 是产品建议的空车采集时长；它不参与在线检测，检测可在任意时刻
    冷启动。返回的字典同时写入 ``data/calibration/empty_cabin.json``。
    """
    cfg = cfg or DetectorConfig()
    duration_s = float(np.clip(duration_s, 3.0, 30.0))
    band = plan_band(hw_f_max)
    scenario = get_scenario("empty")
    channel = AcousticChannel(scenario, seed=seed, band=band)
    pipeline = SASPipeline(cfg, auto_prune=False)

    energies: list[float] = []
    coherences: list[float] = []
    # 前 1 s 用来填满运动支路的短窗；此后的决策帧才是有效的背景样本。
    warmup_s = 1.0
    n = int(round(duration_s * CSI_FS))
    for i in range(n):
        t = i / CSI_FS
        _, amp, truth = channel.step(1.0 / CSI_FS)
        frame = pipeline.step(t, amp, truth)
        if frame is None or t < warmup_s:
            continue
        # 判决帧把两个原始统计量分别放在 motion_energy / motion_coherence；
        # features["motion"] 是相对基线的倍率，不能用来重新估计基线。
        energies.append(float(frame["features"]["motion_energy"]))
        coherences.append(float(frame["features"]["motion_coherence"]))

    return _build_profile(
        energies,
        coherences,
        cfg=cfg,
        duration_s=duration_s,
        hw_f_max=hw_f_max,
        extra={
            "source": "simulation",
            "source_file": None,
            "csi_fs": float(CSI_FS),
            "source_frames": n,
        },
    )


def calibrate_empty_cabin_from_wav(
    wav_path: str | Path,
    duration_s: float = 5.0,
    start_s: float = 0.0,
    cfg: DetectorConfig | None = None,
    hw_f_max: float | None = None,
    fs: int | None = None,
) -> dict:
    """用一段真实空车录音生成基线文件（实采适配层）。

    ``wav_path`` 必须是“播放 Kasami 探测音时录回”的座舱录音，且录音里能找到
    周期性探测音；否则 ``load_csi`` 会直接报错。``start_s`` 用于跳过开机/关门等
    瞬态，``duration_s`` 是参与统计的窗口长度。录音比窗口短时会用实际可用
    部分，但至少要有 20 个判决帧（约 3 s 录音）。
    """
    # 延迟导入：只看/清除基线时不必加载 scipy 音频前端。
    from .replay import load_csi

    cfg = cfg or DetectorConfig()
    duration_s = float(np.clip(duration_s, 3.0, 30.0))
    start_s = max(0.0, float(start_s))

    amp, times, meta = load_csi(wav_path, fs=fs, hw_f_max=hw_f_max)
    pipeline = SASPipeline(cfg, fs=float(meta["csi_fs"]))

    window_end = start_s + duration_s
    energies: list[float] = []
    coherences: list[float] = []
    for t, a in zip(times, amp):
        # 窗口之前的帧也要喂给流水线，让 1 s 运动窗和载波健康窗先填满。
        frame = pipeline.step(float(t), a)
        if frame is None:
            continue
        if not (start_s <= float(t) <= window_end):
            continue
        energies.append(float(frame["features"]["motion_energy"]))
        coherences.append(float(frame["features"]["motion_coherence"]))

    if len(energies) < 20:
        raise ValueError(
            f"录音中可用于标定的判决帧只有 {len(energies)} 个，至少需要 20 个；"
            f"请录制更长的空车段（建议不少于 6 s），或调整 --start/--seconds"
        )

    effective_end = min(window_end, float(times[-1]))
    effective_start = max(start_s, float(times[0]))
    return _build_profile(
        energies,
        coherences,
        cfg=cfg,
        duration_s=max(0.0, effective_end - effective_start),
        hw_f_max=hw_f_max,
        extra={
            "source": "wav",
            "source_file": Path(wav_path).name,
            "source_fs": int(meta["fs"]),
            "csi_fs": float(meta["csi_fs"]),
            "source_frames": int(meta["frames"]),
            "sync_ratio": round(float(meta["sync"]["match_ratio"]), 2),
            "clipped_samples": int(meta["clipped_samples"]),
        },
    )


def load_empty_calibration(
    n_carriers: int | None = None,
    source: str | None = None,
) -> dict | None:
    """读取最近一次空车标定；载波数或数据源不匹配时返回 None。

    ``source="simulation"`` 只接受仿真基线，``"wav"`` 只接受真实录音基线。
    两种来源的绝对幅度尺度不同（录音电平取决于播放/麦克风增益），不能混用。
    旧文件没有 ``source`` 字段时按仿真基线处理。
    """
    if not PROFILE_PATH.exists():
        return None
    try:
        profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if n_carriers is not None and int(profile.get("n_carriers", 0)) != int(n_carriers):
        return None
    profile_source = profile.get("source", "simulation")
    if source is not None and profile_source != source:
        return None
    return profile


def clear_empty_calibration() -> None:
    """删除已保存的空车基线。"""
    PROFILE_PATH.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """``python3 -m core.calibration`` 的命令行入口。"""
    parser = argparse.ArgumentParser(
        prog="python3 -m core.calibration",
        description="空车标定：生成、查看或清除 data/calibration/empty_cabin.json",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--simulate", action="store_true",
                     help="用仿真 empty 场景生成基线（无实车硬件时自检用）")
    src.add_argument("--wav", metavar="WAV",
                     help="用真实空车录音生成基线（录音必须包含周期性 Kasami 探测音）")
    src.add_argument("--show", action="store_true", help="打印当前基线")
    src.add_argument("--clear", action="store_true", help="删除当前基线")

    parser.add_argument("--seconds", type=float, default=5.0,
                        help="参与统计的窗口长度，默认 5 s（有效范围 3–30 s）")
    parser.add_argument("--start", type=float, default=0.0,
                        help="从录音第几秒开始统计，默认 0 s；可用来跳过开关门瞬态")
    parser.add_argument("--hw-f-max", type=float, choices=BAND_CHOICES, default=None,
                        help="硬件频响 -3 dB 上限（Hz）；省略表示默认 20 kHz（24 载波）")
    args = parser.parse_args(argv)

    try:
        if args.show:
            profile = load_empty_calibration()
            if profile is None:
                print("尚未标定：将使用开机冷启动自适应背景")
            else:
                print(json.dumps(profile, ensure_ascii=False, indent=2))
            return 0
        if args.clear:
            existed = PROFILE_PATH.exists()
            clear_empty_calibration()
            print("已删除空车基线" if existed else "当前没有空车基线")
            return 0
        if args.simulate:
            profile = calibrate_empty_cabin(
                duration_s=args.seconds, hw_f_max=args.hw_f_max
            )
        else:
            profile = calibrate_empty_cabin_from_wav(
                args.wav,
                duration_s=args.seconds,
                start_s=args.start,
                hw_f_max=args.hw_f_max,
            )
    except (RuntimeError, ValueError, OSError) as exc:
        parser.exit(2, f"错误：{exc}\n")

    print(json.dumps(profile, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
