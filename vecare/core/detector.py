"""Child-presence detection state machine and alarm logic.

Motion and breathing evidence are aggregated over a 5 s decision window; a
child is claimed when evidence occupies more than 30 % of that window, as in
the paper.  A confirmed presence starts a 10 s response countdown.  Loss of
breathing after a confirmed breathing-only presence raises an apnea alarm.
"""
from __future__ import annotations

from collections import deque
import numpy as np

from .config import DetectorConfig


class CPDDetector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.state = "CALIBRATING"
        self.calibrated = False
        self.calib_values: list[float] = []
        self.motion_threshold = cfg.motion_threshold_floor()

        self.evidence = deque()          # (t, bool)
        self.presence = False
        self.presence_since: float | None = None
        self.presence_score = 0.0
        self.last_evidence_t: float | None = None
        self.last_motion_t: float | None = None
        self.last_breath_t: float | None = None
        self.breathing_ever = False

        self.alarm = False
        self.alarm_reason: str | None = None
        self.alarm_since: float | None = None
        self.acknowledged = False
        self.countdown = cfg.alarm_delay_s

        self.bpm_history: deque = deque(maxlen=13)
        self.bpm = None
        self.bpm_stale_s = None
        self.breath_confidence = 0.0
        self.events: list[dict] = []

    # ------------------------------------------------------------------
    def _event(self, t: float, kind: str, message: str) -> dict:
        ev = {"t": round(t, 3), "kind": kind, "message": message}
        self.events.append(ev)
        return ev

    def _set_state(self, t: float, new_state: str, message: str | None = None) -> None:
        if new_state == self.state:
            return
        old = self.state
        self.state = new_state
        labels = {
            "CALIBRATING": "背景校准",
            "MONITORING": "监测中 · 无儿童",
            "PRESENCE": "检测到儿童",
            "ALARM": "告警",
            "ACKED": "告警已静音",
        }
        self._event(
            t,
            "state",
            message or f"{labels.get(old, old)} → {labels.get(new_state, new_state)}",
        )

    def calibrate(self, t: float, motion: float) -> None:
        self.calib_values.append(motion)
        if t >= self.cfg.calibration_s and len(self.calib_values) >= 20:
            vals = np.asarray(self.calib_values, dtype=float)
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med))) * 1.4826
            adaptive = med + 3.5 * mad
            self.motion_threshold = max(self.cfg.motion_threshold_floor(), adaptive)
            self.calibrated = True
            self._set_state(t, "MONITORING", "背景校准完成")

    # ------------------------------------------------------------------
    def update(self, t: float, motion: float, breath: dict, allow_breath: bool) -> dict:
        if not self.calibrated:
            self.calibrate(t, motion)
            return self.snapshot(t)

        motion_det = bool(motion > self.motion_threshold)

        peak_valid = bool(
            allow_breath
            and breath.get("valid")
            and breath.get("confidence", 0) >= self.cfg.breath_conf_threshold
        )
        bpm = breath.get("bpm") if peak_valid else None
        conf = float(breath.get("confidence", 0.0)) if peak_valid else 0.0

        if bpm is not None:
            self.bpm_history.append((t, float(bpm)))
            self.last_breath_t = t
            self.breathing_ever = True
            recent = [v for _, v in self.bpm_history]
            self.bpm = float(np.median(recent)) if recent else bpm
            self.bpm_stale_s = 0.0
        elif self.bpm is not None and self.last_breath_t is not None:
            self.bpm_stale_s = t - self.last_breath_t
            if self.bpm_stale_s > 4.0:
                self.bpm = None

        breath_det = peak_valid and bpm is not None
        self.breath_confidence = conf

        evidence = motion_det or breath_det
        self.evidence.append((t, evidence))
        while self.evidence and self.evidence[0][0] < t - self.cfg.evidence_window_s:
            self.evidence.popleft()
        self.presence_score = (
            float(np.mean([e for _, e in self.evidence])) if self.evidence else 0.0
        )

        if motion_det:
            self.last_motion_t = t
        if evidence:
            self.last_evidence_t = t

        if not self.presence and self.presence_score >= self.cfg.presence_ratio:
            self.presence = True
            self.presence_since = t
            self._event(
                t, "presence",
                "检测到车内儿童（运动/呼吸证据持续）"
            )

        # Presence latch release: only after a long evidence-free period, and
        # only if no alarm is active (alarms remain until acknowledged).
        if (
            self.presence
            and not self.alarm
            and self.last_evidence_t is not None
            and t - self.last_evidence_t > self.cfg.clear_timeout_s
        ):
            self.presence = False
            self.presence_since = None
            self.presence_score = 0.0
            self._event(t, "cleared", "连续无存在证据，恢复安全状态")

        # Apnea alarm: confirmed breathing presence, then breathing stops while
        # the infant is still and no motion evidence is available.
        apnea = (
            self.presence
            and self.breathing_ever
            and self.last_breath_t is not None
            and (t - self.last_breath_t) >= self.cfg.apnea_timeout_s
            and self.last_motion_t is not None
            and (t - self.last_motion_t) >= min(6.0, self.cfg.apnea_timeout_s)
        )
        # Also support the (less common) case where no motion was ever seen.
        if self.presence and self.breathing_ever and self.last_breath_t is not None:
            apnea = apnea or (
                (t - self.last_breath_t) >= self.cfg.apnea_timeout_s
                and (self.last_motion_t is None or (t - self.last_motion_t) >= 6.0)
            )

        if not self.alarm and self.presence:
            self.countdown = max(0.0, self.cfg.alarm_delay_s - (t - (self.presence_since or t)))
            if apnea:
                self._raise_alarm(t, "APNEA", "检测到儿童呼吸停止，触发呼吸异常告警")
            elif self.countdown <= 0:
                self._raise_alarm(t, "PRESENCE", "儿童被留在车内，触发儿童存在告警")
        elif self.alarm and apnea and self.alarm_reason != "APNEA":
            # Already alarming for presence: escalate the reason if breathing
            # subsequently stops (critical condition).
            self.alarm_reason = "APNEA"
            self.alarm_since = t
            self.acknowledged = False
            self._event(t, "alarm_escalate", "告警升级：儿童呼吸停止")

        self._refresh_state(t)
        return self.snapshot(t)

    def _raise_alarm(self, t: float, reason: str, message: str) -> None:
        self.alarm = True
        self.alarm_reason = reason
        self.alarm_since = t
        self.acknowledged = False
        self._event(t, "alarm", message)

    def _refresh_state(self, t: float) -> None:
        if not self.calibrated:
            self._set_state(t, "CALIBRATING")
        elif self.alarm and not self.acknowledged:
            self._set_state(t, "ALARM")
        elif self.alarm and self.acknowledged:
            self._set_state(t, "ACKED")
        elif self.presence:
            self._set_state(t, "PRESENCE")
        else:
            self._set_state(t, "MONITORING")

    def acknowledge(self, t: float) -> dict:
        if self.alarm:
            self.acknowledged = True
            self._event(t, "ack", "用户已确认并静音告警（儿童仍在车内）")
            self._refresh_state(t)
        return self.snapshot(t)

    # ------------------------------------------------------------------
    def snapshot(self, t: float) -> dict:
        countdown = None
        if self.presence and not self.alarm:
            countdown = max(0.0, self.cfg.alarm_delay_s - (t - (self.presence_since or t)))
        return {
            "state": self.state,
            "calibrated": self.calibrated,
            "calibration_progress": float(
                min(1.0, t / self.cfg.calibration_s) if not self.calibrated else 1.0
            ),
            "motion_threshold": round(self.motion_threshold, 4),
            "presence": self.presence,
            "presence_score": round(self.presence_score, 3),
            "presence_since": self.presence_since,
            "countdown": None if countdown is None else round(countdown, 2),
            "alarm": self.alarm,
            "alarm_reason": self.alarm_reason,
            "alarm_since": self.alarm_since,
            "acknowledged": self.acknowledged,
            "bpm": None if self.bpm is None else round(float(self.bpm), 1),
            "bpm_stale_s": None if self.bpm_stale_s is None else round(float(self.bpm_stale_s), 1),
            "breath_confidence": round(float(self.breath_confidence), 3),
        }
