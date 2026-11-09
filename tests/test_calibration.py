"""空车标定与冷启动背景估计的回归测试。"""
from __future__ import annotations

import json

import pytest

from core import calibration
from core.config import DetectorConfig, plan_band
from core.detector import CPDDetector
from core.replay import render_wav
from core.scenarios import SCENARIOS


def _motion(energy: float, coherence: float) -> dict:
    return {"energy": energy, "coherence": coherence, "interference": False}


def test_single_early_frame_cannot_confirm_presence():
    """开机头几帧只有一帧触发运动判据时，不能靠 “1/2 = 50 %” 锁存存在。"""
    det = CPDDetector(
        DetectorConfig(),
        background={"baseline_energy": 0.014, "coherence_threshold": 0.115},
    )
    breath = {"valid": False, "confidence": 0.0, "bpm": None}
    det.update(0.0, _motion(0.032, 0.20), breath)
    assert det.presence is False
    for i in range(1, 30):
        snap = det.update(i / 10.0, _motion(0.014, 0.10), breath)
    assert snap["presence"] is False


def test_evidence_gap_does_not_extend_budget_after_presence():
    """确认存在后预算锚点冻结，短暂断档不能让 10 s 响应预算被推迟。"""
    cfg = DetectorConfig(
        response_budget_s=4.0, evidence_gap_reset_s=0.5, clear_timeout_s=20.0
    )
    det = CPDDetector(cfg)
    breath = {"valid": False, "confidence": 0.0, "bpm": None}
    t = 0.0
    while not det.calibrated:
        det.update(t, _motion(0.014, 0.10), breath)
        t += 0.1

    for _ in range(20):
        snap = det.update(t, _motion(0.032, 0.20), breath)
        t += 0.1
    assert snap["presence"] is True
    anchor = det.first_evidence_t
    assert anchor is not None

    for _ in range(10):
        det.update(t, _motion(0.014, 0.10), breath)
        t += 0.1
    for _ in range(20):
        snap = det.update(t, _motion(0.032, 0.20), breath)
        t += 0.1
        if snap["alarm"]:
            break
    assert snap["alarm"] is True
    assert det.alarm_since - anchor == pytest.approx(cfg.response_budget_s, abs=1e-6)


def test_empty_calibration_roundtrip(tmp_path, monkeypatch):
    profile_path = tmp_path / "empty_cabin.json"
    monkeypatch.setattr(calibration, "CALIBRATION_DIR", tmp_path)
    monkeypatch.setattr(calibration, "PROFILE_PATH", profile_path)

    profile = calibration.calibrate_empty_cabin(duration_s=3, hw_f_max=18000)
    assert profile["profile"] == "empty_cabin"
    assert profile["samples"] >= 20
    assert profile["n_carriers"] > 0
    assert profile["baseline_energy"] > 0
    assert profile["energy_unit"] == "relative"
    assert profile["coherence_threshold"] >= profile["coherence_floor"] > 0
    assert profile_path.exists()

    loaded = calibration.load_empty_calibration(profile["n_carriers"])
    assert loaded is not None and loaded["baseline_energy"] == profile["baseline_energy"]
    # 载波数不一致时不应误用旧基线。
    assert calibration.load_empty_calibration(profile["n_carriers"] + 3) is None

    calibration.clear_empty_calibration()
    assert calibration.load_empty_calibration() is None


def test_detector_starts_monitoring_without_quiet_window():
    """儿童在场冷启动：首帧即监测，不再进入 CALIBRATING。"""
    det = CPDDetector(DetectorConfig())
    assert det.state == "MONITORING"
    breath = {"valid": False, "confidence": 0.0, "bpm": None}
    for i in range(40):
        det.update(i / 10.0, _motion(0.02, 0.35), breath)
    assert det.calibrated is True
    assert det.background_source == "冷启动自适应"
    assert det.baseline_energy is not None and det.baseline_energy > 0
    assert det.state == "MONITORING"


def test_detector_loads_empty_calibration():
    profile = {
        "profile": "empty_cabin",
        "baseline_energy": 0.0142,
        "coherence_threshold": 0.142,
        "coherence_spread": 0.0049,
    }
    det = CPDDetector(DetectorConfig(), background=profile)
    assert det.calibrated is True
    assert det.background_source == "空车标定"
    assert abs(det.baseline_energy - 0.0142) < 1e-9
    assert abs(det.coherence_threshold - 0.142) < 1e-9


def test_legacy_absolute_energy_baseline_is_rejected(tmp_path, monkeypatch):
    """旧基线存的是绝对幅度；改成相对能量后必须强制重新标定。"""
    path = tmp_path / "empty_cabin.json"
    path.write_text(json.dumps({
        "profile": "empty_cabin",
        "n_carriers": plan_band().n_carriers,
        "baseline_energy": 0.014,
        "coherence_threshold": 0.22,
    }), encoding="utf-8")
    monkeypatch.setattr(calibration, "PROFILE_PATH", path)
    assert calibration.load_empty_calibration(plan_band().n_carriers) is None


def test_empty_calibration_from_wav(tmp_path, monkeypatch):
    """实采适配层：空车 WAV -> CSI -> 同一套基线统计。"""
    wav = tmp_path / "empty.wav"
    render_wav(wav, key="empty", duration_s=6.0, snr_db=45.0)
    profile_path = tmp_path / "empty_cabin.json"
    monkeypatch.setattr(calibration, "CALIBRATION_DIR", tmp_path)
    monkeypatch.setattr(calibration, "PROFILE_PATH", profile_path)

    profile = calibration.calibrate_empty_cabin_from_wav(
        wav, duration_s=4.0, start_s=1.0
    )
    assert profile["source"] == "wav"
    assert profile["source_file"] == "empty.wav"
    assert profile["samples"] >= 20
    assert profile["n_carriers"] == plan_band().n_carriers
    assert profile["baseline_energy"] > 0
    assert profile["energy_unit"] == "relative"
    assert profile["coherence_threshold"] >= profile["coherence_floor"] > 0
    assert profile_path.exists()

    loaded = calibration.load_empty_calibration(profile["n_carriers"])
    assert loaded is not None and loaded["source"] == "wav"
    # 不同来源的绝对幅度尺度不可比，仿真基线加载时必须拒绝实录音基线。
    assert calibration.load_empty_calibration(profile["n_carriers"], source="simulation") is None
    assert calibration.load_empty_calibration(profile["n_carriers"], source="wav") is not None


def test_cli_show_and_clear(tmp_path, monkeypatch, capsys):
    """命令行入口：查看与清除不需要加载音频前端。"""
    monkeypatch.setattr(calibration, "CALIBRATION_DIR", tmp_path)
    monkeypatch.setattr(calibration, "PROFILE_PATH", tmp_path / "empty_cabin.json")

    assert calibration.main(["--show"]) == 0
    assert "尚未标定" in capsys.readouterr().out
    assert calibration.main(["--clear"]) == 0
    assert "当前没有空车基线" in capsys.readouterr().out


def test_child_scenarios_do_not_require_empty_entry():
    """七场景中有人场景必须从 0 s 起就在车内，不能依赖空车入场。"""
    for key, scenario in SCENARIOS.items():
        if scenario.target is None:
            continue
        assert scenario.target.enter_t <= 0.0, key
        for start, _ in scenario.target.breathing_windows:
            assert start <= 0.0, key
