"""Offline scenario runner and prototype metrics.

Used by the test-suite and by the `tools.evaluate` entry point: no wall-clock
pacing, so an entire scenario can be scored in a fraction of the time.
"""
from __future__ import annotations

import zlib

import numpy as np

from .channel import AcousticChannel
from .config import CSI_FS, DetectorConfig
from .pipeline import SASPipeline
from .scenarios import get_scenario


def _default_seed(key: str) -> int:
    return 20231102 + zlib.crc32(key.encode("utf-8")) % 997


def run_offline(
    key: str,
    sensitivity: float = 1.0,
    duration_s: float | None = None,
    seed: int | None = None,
) -> dict:
    scenario = get_scenario(key)
    cfg = DetectorConfig(sensitivity=sensitivity)
    channel = AcousticChannel(scenario, seed=seed if seed is not None else _default_seed(key))
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

    return {
        "scenario": scenario.public(),
        "frames": frames,
        "events": pipe.detector.events,
        "metrics": evaluate_frames(frames, scenario),
        "detector_config": cfg.to_dict(),
    }


def evaluate_frames(frames: list[dict], scenario) -> dict:
    target = scenario.target
    enter_t = target.enter_t if target else None
    exit_t = target.exit_t if target else None

    presence_at = next((f["t"] for f in frames if f["state"]["presence"]), None)
    alarm_frame = next((f for f in frames if f["state"]["alarm"]), None)
    apnea_frame = next((f for f in frames if f["state"]["alarm_reason"] == "APNEA"), None)
    bpm_at = next((f["t"] for f in frames if f["state"]["bpm"] is not None), None)

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

    detected = presence_at is not None and (
        enter_t is None or (exit_t is None or presence_at <= exit_t + 0.5)
    )
    metrics = {
        "detected": bool(detected) if enter_t is not None else None,
        "presence_at": presence_at,
        "detection_latency_s": None
        if presence_at is None or enter_t is None
        else round(presence_at - enter_t, 2),
        "breathing_at": bpm_at,
        "alarm_at": None if alarm_frame is None else alarm_frame["t"],
        "alarm_latency_s": None
        if alarm_frame is None or enter_t is None
        else round(alarm_frame["t"] - enter_t, 2),
        "alarm_reason": None if alarm_frame is None else alarm_frame["state"]["alarm_reason"],
        "apnea_escalated_at": None if apnea_frame is None else apnea_frame["t"],
        "bpm_mae": None if bpm_mae is None else round(bpm_mae, 3),
        "bpm_samples": len(bpm_pairs),
        "false_presence_frames": false_frames,
        "frames": len(frames),
        "duration_s": frames[-1]["t"] if frames else 0.0,
    }
    return metrics
