"""空车标定：在确认无人的座舱里采集背景残差并持久化。

检测器在冷启动时使用自适应背景估计，因此**不要求**开机后先安静 5 s。若用户
明确执行空车标定，本模块会把空车场景下的运动残差统计成一份基线文件，之后
每次启动直接加载，避免开机初期把低幅噪声当成运动，或在有人时才开始估计背景。

标定只影响运动支路的判据：
    * 残差能量基线（相对能量的分母）；
    * 运动相干性阈值（由空车相干性中位数 + 5.5×MAD 得到）。
呼吸支路依赖 8 s ACF，不使用这份基线。

产物路径：``data/calibration/empty_cabin.json``（不入库）。基线按载波数区分，
频带变化时不会误用旧基线。
"""
from __future__ import annotations

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


def _robust_threshold(values: list[float], factor: float, floor: float) -> tuple[float, float]:
    """返回 (中位数, max(下限, 中位数 + factor×MAD))。"""
    vals = np.asarray(values, dtype=float)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    return med, max(float(floor), med + factor * mad)


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

    if len(energies) < 20:
        raise ValueError("空车标定样本不足，请把采集时长延长到至少 3 s")

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
        "scenario": "empty",
        "duration_s": round(duration_s, 2),
        "samples": len(energies),
        "hw_f_max_hz": None if hw_f_max is None else float(hw_f_max),
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

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile


def load_empty_calibration(n_carriers: int | None = None) -> dict | None:
    """读取最近一次空车标定；载波数不匹配时返回 None。"""
    if not PROFILE_PATH.exists():
        return None
    try:
        profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if n_carriers is not None and int(profile.get("n_carriers", 0)) != int(n_carriers):
        return None
    return profile


def clear_empty_calibration() -> None:
    """删除已保存的空车基线。"""
    PROFILE_PATH.unlink(missing_ok=True)
