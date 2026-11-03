"""离线场景运行器与样机指标。

供测试套件和 ``replay_audio.py --selfcheck`` 使用：不按真实时钟节奏推进，
因此整个场景可以在很短的时间内完成评分。
"""
from __future__ import annotations

import zlib

import numpy as np

from .channel import AcousticChannel
from .config import CSI_FS, MRC_WEIGHT_FLOOR, DetectorConfig, plan_band
from .pipeline import SASPipeline
from .scenarios import get_scenario


def _default_seed(key: str) -> int:
    return 20231102 + zlib.crc32(key.encode("utf-8")) % 997


def run_offline(
    key: str,
    sensitivity: float = 1.0,
    duration_s: float | None = None,
    seed: int | None = None,
    hw_f_max: float | None = None,
) -> dict:
    scenario = get_scenario(key)
    cfg = DetectorConfig(sensitivity=sensitivity)
    band = plan_band(hw_f_max)
    channel = AcousticChannel(
        scenario,
        seed=seed if seed is not None else _default_seed(key),
        band=band,
    )
    pipe = SASPipeline(cfg)
    frames: list[dict] = []
    total = int(round((duration_s or scenario.duration_s) * CSI_FS))
    dt = 1.0 / CSI_FS
    for i in range(total):
        t = i / CSI_FS
        _, amp, truth = channel.step(dt)
        frame = pipe.step(t, amp, truth)
        if frame is not None:
            frames.append(frame)

    metrics = evaluate_frames(frames, scenario)
    metrics["band"] = band.describe()
    metrics["carriers_planned"] = int(band.n_carriers)
    metrics["carriers_used"] = int(
        band.n_carriers if pipe.carrier_mask is None else pipe.carrier_mask.sum()
    )
    return {
        "scenario": scenario.public(),
        "frames": frames,
        "events": pipe.detector.events,
        "metrics": metrics,
        "detector_config": cfg.to_dict(),
    }


def evaluate_frames(frames: list[dict], scenario) -> dict:
    target = scenario.target
    enter_t = target.enter_t if target else None
    exit_t = target.exit_t if target else None

    presence_frame = next((f for f in frames if f["state"]["presence"]), None)
    alarm_frame = next((f for f in frames if f["state"]["alarm"]), None)
    apnea_frame = next((f for f in frames if f["state"]["alarm_reason"] == "APNEA"), None)
    bpm_frame = next((f for f in frames if f["state"]["bpm"] is not None), None)

    presence_at = presence_frame["t"] if presence_frame else None
    first_evidence = None if presence_frame is None else presence_frame["state"].get("first_evidence_t")

    false_frames = sum(
        1
        for f in frames
        if not (f.get("truth") or {}).get("present", False) and f["state"]["presence"]
    )

    bpm_pairs = [
        (f["state"]["bpm"], (f.get("truth") or {}).get("bpm"))
        for f in frames
        if f["state"]["bpm"] is not None and (f.get("truth") or {}).get("breathing")
    ]
    bpm_mae = (
        float(np.mean([abs(b - g) for b, g in bpm_pairs])) if bpm_pairs else None
    )

    # MRC 可观测性：实际有多少个子载波承载了合并后的 ACF。
    neff = []
    for f in frames:
        g = np.asarray(f["features"].get("gains") or [], dtype=float)
        if g.size == 0:
            continue
        w = np.maximum(g, 0.0) + MRC_WEIGHT_FLOOR
        total = w.sum()
        if total <= 0:
            continue
        w = w / total
        neff.append(float(1.0 / np.sum(w ** 2)))
    mrc_neff = float(np.mean(neff)) if neff else None

    detected = presence_at is not None and (
        enter_t is None or (exit_t is None or presence_at <= exit_t + 0.5)
    )

    alarm_at = None if alarm_frame is None else alarm_frame["t"]
    metrics = {
        "detected": bool(detected) if enter_t is not None else None,
        "presence_at": presence_at,
        "confirmed_by": None if presence_frame is None
        else presence_frame["state"].get("breath_source"),
        "detection_latency_s": None
        if presence_at is None or enter_t is None
        else round(presence_at - enter_t, 2),
        "breathing_at": None if bpm_frame is None else bpm_frame["t"],
        "alarm_at": alarm_at,
        "alarm_latency_s": None
        if alarm_at is None or enter_t is None
        else round(alarm_at - enter_t, 2),
        # 该数值由 10 s 的 CPD 预算决定，而不是 alarm_latency_s。
        "response_time_s": None
        if alarm_at is None or first_evidence is None
        else round(alarm_at - first_evidence, 2),
        "alarm_reason": None if alarm_frame is None else alarm_frame["state"]["alarm_reason"],
        "apnea_escalated_at": None if apnea_frame is None else apnea_frame["t"],
        "bpm_mae": None if bpm_mae is None else round(bpm_mae, 3),
        "bpm_samples": len(bpm_pairs),
        "false_presence_frames": false_frames,
        "motion_evidence_frames": sum(1 for f in frames if f["state"].get("motion_detected")),
        "mrc_neff": None if mrc_neff is None else round(mrc_neff, 2),
        "interference_frames": sum(1 for f in frames if f["features"].get("interference")),
        "frames": len(frames),
        "duration_s": frames[-1]["t"] if frames else 0.0,
    }
    return metrics
