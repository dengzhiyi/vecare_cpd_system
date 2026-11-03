"""检测状态机、告警时序与离线场景测试。"""
import pytest

from core.config import DetectorConfig
from core.detector import CPDDetector
from core.offline import run_offline

INVALID_PEAK = {"valid": False, "confidence": 0.0, "bpm": None}
VALID_PEAK = {"valid": True, "confidence": 0.9, "bpm": 20.0}

# 合成运动特征（能量为绝对值，检测器会用校准后的背景进行归一化）。
QUIET = {"energy": 0.014, "coherence": 0.10, "interference": False}
MOVING = {"energy": 0.032, "coherence": 0.20, "interference": False}
NOISY = {"energy": 0.030, "coherence": 0.10, "interference": False}


def _calibrate(det, step=0.1):
    """持续送入安静帧，直到冷启动背景估计冻结（约 2 s、20 个判决帧）。"""
    t = 0.0
    for _ in range(40):
        det.update(t, QUIET, INVALID_PEAK)
        t += step
        if det.calibrated:
            break
    assert det.calibrated, "detector failed to freeze a quiet-cabin background"
    return t


def test_calibration_enters_monitoring():
    cfg = DetectorConfig()
    det = CPDDetector(cfg)
    _calibrate(det)
    assert det.state == "MONITORING"
    assert det.motion_threshold == pytest.approx(cfg.motion_ratio_floor())
    assert det.baseline_energy is not None


def test_motion_presence_then_alarm_within_budget():
    cfg = DetectorConfig(response_budget_s=3.0)
    det = CPDDetector(cfg)
    t = _calibrate(det)

    for _ in range(25):
        snap = det.update(t, MOVING, INVALID_PEAK)
        t += 0.1
        if snap["presence"]:
            break
    assert snap["presence"] is True
    assert snap["alarm"] is False
    first_evidence = det.first_evidence_t

    for _ in range(60):
        snap = det.update(t, MOVING, INVALID_PEAK)
        t += 0.1
        if snap["alarm"]:
            break
    assert snap["alarm"] is True
    assert snap["alarm_reason"] == "PRESENCE"
    assert snap["state"] == "ALARM"
    # 预算从首个证据样本起算，因此确认延迟是计入预算的，
    # 而不是额外叠加在上面
    assert det.alarm_since - first_evidence == pytest.approx(
        cfg.response_budget_s, abs=1e-6
    )


def test_incoherent_noise_never_confirms_presence():
    """残差能量高但子载波相干性低，属于环境噪声。"""
    cfg = DetectorConfig()
    det = CPDDetector(cfg)
    t = _calibrate(det)
    for _ in range(300):
        snap = det.update(t, NOISY, INVALID_PEAK)
        t += 0.1
    assert snap["presence"] is False
    assert snap["alarm"] is False


def test_quiet_cabin_never_alarms():
    cfg = DetectorConfig()
    det = CPDDetector(cfg)
    t = _calibrate(det)
    for _ in range(200):
        snap = det.update(t, QUIET, INVALID_PEAK)
        t += 0.1
    assert snap["presence"] is False
    assert snap["alarm"] is False


def test_interference_frame_suppresses_evidence():
    cfg = DetectorConfig()
    det = CPDDetector(cfg)
    t = _calibrate(det)
    blocked = dict(MOVING, interference=True)
    for _ in range(200):
        snap = det.update(t, blocked, INVALID_PEAK)
        t += 0.1
    assert snap["presence"] is False


def test_acknowledge_mutes_but_keeps_presence():
    cfg = DetectorConfig(response_budget_s=2.0)
    det = CPDDetector(cfg)
    t = _calibrate(det)
    for _ in range(60):
        snap = det.update(t, MOVING, INVALID_PEAK)
        t += 0.1
        if snap["alarm"]:
            break
    assert snap["alarm"] is True
    snap = det.acknowledge(t)
    assert snap["acknowledged"] is True
    assert snap["presence"] is True
    assert snap["state"] == "ACKED"


def test_acknowledged_alarm_clears_after_evidence_disappears():
    """静音不是锁存：车厢安静下来后系统必须复位。"""
    cfg = DetectorConfig(
        response_budget_s=2.0, clear_timeout_s=2.0
    )
    det = CPDDetector(cfg)
    t = _calibrate(det)
    for _ in range(60):
        snap = det.update(t, MOVING, INVALID_PEAK)
        t += 0.1
        if snap["alarm"]:
            break
    assert snap["alarm"] is True
    det.acknowledge(t)
    assert det.state == "ACKED"

    for _ in range(30):
        snap = det.update(t, QUIET, INVALID_PEAK)
        t += 0.1
        if not snap["presence"]:
            break
    assert snap["presence"] is False
    assert snap["alarm"] is False
    assert snap["acknowledged"] is False
    assert snap["state"] == "MONITORING"


def test_breathing_stop_escalates_to_apnea():
    cfg = DetectorConfig(response_budget_s=120.0, apnea_timeout_s=3.0)
    det = CPDDetector(cfg)
    t = _calibrate(det)

    for _ in range(60):
        det.update(t, QUIET, VALID_PEAK)
        t += 0.1
    assert det.presence is True
    assert det.alarm is False
    assert det.bpm is not None

    for _ in range(80):
        snap = det.update(t, QUIET, INVALID_PEAK)
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
    assert m["alarm_at"] is not None
    # 论文的 10 s 要求约束的是从首个证据样本起算的预算，
    # 而不是从确认时刻起算的预算
    assert m["response_time_s"] is not None and m["response_time_s"] <= 10.05
    assert m["alarm_reason"] == "PRESENCE"
    assert m["bpm_mae"] is not None and m["bpm_mae"] < 2.0


def test_offline_active_child_drives_the_motion_channel():
    """肢体运动必须产生运动证据，而不仅仅是呼吸。"""
    result = run_offline("active")
    m = result["metrics"]
    assert m["detected"] is True
    assert m["false_presence_frames"] == 0
    assert m["motion_evidence_frames"] >= 10
    assert m["alarm_at"] is not None


def test_offline_quiet_scenarios_have_no_motion_evidence():
    for key in ("empty", "stress"):
        m = run_offline(key)["metrics"]
        assert m["false_presence_frames"] == 0
    assert run_offline("empty")["metrics"]["motion_evidence_frames"] == 0


def test_offline_empty_cabin_stays_safe():
    result = run_offline("empty")
    m = result["metrics"]
    assert m["presence_at"] is None
    assert m["alarm_at"] is None
    assert m["false_presence_frames"] == 0


def test_offline_stress_stays_safe():
    result = run_offline("stress")
    m = result["metrics"]
    assert m["presence_at"] is None
    assert m["alarm_at"] is None
    assert m["false_presence_frames"] == 0
