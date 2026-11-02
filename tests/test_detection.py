"""Detection state-machine, alarm timing and offline scenario tests."""
import pytest

from core.config import DetectorConfig
from core.detector import CPDDetector
from core.offline import run_offline

INVALID_PEAK = {"valid": False, "confidence": 0.0, "bpm": None}
VALID_PEAK = {"valid": True, "confidence": 0.9, "bpm": 20.0}


def _calibrate(cfg, det, duration_s=None, step=0.1):
    """Feed quiet frames until the detector leaves CALIBRATING."""
    t = 0.0
    limit = int((duration_s or cfg.calibration_s) * 1.5 / step) + 20
    for _ in range(limit):
        det.update(t, 0.0, INVALID_PEAK, True)
        t += step
        if det.calibrated:
            break
    assert det.calibrated, "detector failed to calibrate on a quiet cabin"
    return t


def test_calibration_enters_monitoring():
    cfg = DetectorConfig(calibration_s=1.0)
    det = CPDDetector(cfg)
    t = _calibrate(cfg, det)
    assert det.state == "MONITORING"
    assert det.motion_threshold >= cfg.motion_threshold_floor() - 1e-9


def test_motion_presence_then_alarm_after_delay():
    cfg = DetectorConfig(calibration_s=1.0, alarm_delay_s=2.0)
    det = CPDDetector(cfg)
    t = _calibrate(cfg, det)

    for _ in range(15):
        snap = det.update(t, 0.5, INVALID_PEAK, True)
        t += 0.1
    assert snap["presence"] is True
    assert snap["alarm"] is False
    presence_at = det.presence_since

    for _ in range(25):
        snap = det.update(t, 0.5, INVALID_PEAK, True)
        t += 0.1
        if snap["alarm"]:
            break
    assert snap["alarm"] is True
    assert snap["alarm_reason"] == "PRESENCE"
    assert snap["state"] == "ALARM"
    assert det.alarm_since - presence_at == pytest.approx(cfg.alarm_delay_s, abs=0.15)


def test_quiet_cabin_never_alarms():
    cfg = DetectorConfig(calibration_s=1.0)
    det = CPDDetector(cfg)
    t = _calibrate(cfg, det)
    for _ in range(200):
        snap = det.update(t, 0.0, INVALID_PEAK, True)
        t += 0.1
    assert snap["presence"] is False
    assert snap["alarm"] is False


def test_acknowledge_mutes_but_keeps_presence():
    cfg = DetectorConfig(calibration_s=1.0, alarm_delay_s=1.0)
    det = CPDDetector(cfg)
    t = _calibrate(cfg, det)
    for _ in range(30):
        snap = det.update(t, 0.5, INVALID_PEAK, True)
        t += 0.1
    assert snap["alarm"] is True
    snap = det.acknowledge(t)
    assert snap["acknowledged"] is True
    assert snap["presence"] is True
    assert snap["state"] == "ACKED"


def test_breathing_stop_escalates_to_apnea():
    cfg = DetectorConfig(calibration_s=1.0, alarm_delay_s=60.0, apnea_timeout_s=3.0)
    det = CPDDetector(cfg)
    t = _calibrate(cfg, det)

    for _ in range(60):
        det.update(t, 0.0, VALID_PEAK, True)
        t += 0.1
    assert det.presence is True
    assert det.alarm is False
    assert det.bpm is not None

    for _ in range(40):
        snap = det.update(t, 0.0, INVALID_PEAK, True)
        t += 0.1
        if snap["alarm"]:
            break
    assert snap["alarm"] is True
    assert snap["alarm_reason"] == "APNEA"
    assert snap["state"] == "ALARM"


def test_offline_sleeping_scenario_meets_response_budget():
    result = run_offline("sleeping")
    m = result["metrics"]
    assert m["detected"] is True
    assert m["false_presence_frames"] == 0
    assert m["detection_latency_s"] <= 10.0
    assert m["alarm_at"] is not None
    # The paper's 10 s response budget counts from presence confirmation,
    # not from the moment the child enters the cabin.
    assert m["alarm_at"] - m["presence_at"] <= 10.0 + 0.3
    assert m["alarm_reason"] == "PRESENCE"
    assert m["bpm_mae"] is not None and m["bpm_mae"] < 2.0


def test_offline_empty_cabin_stays_safe():
    result = run_offline("empty")
    m = result["metrics"]
    assert m["presence_at"] is None
    assert m["alarm_at"] is None
    assert m["false_presence_frames"] == 0
