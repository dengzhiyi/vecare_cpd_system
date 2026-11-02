"""Child-presence detection state machine and alarm logic.

Motion and breathing evidence are aggregated over a 5 s decision window; a
child is claimed when evidence occupies more than 30 % of that window, as in
the paper. The response budget (10 s by default) is counted from the *first
evidence sample*, not from the moment presence is confirmed: detection latency
and the confirmation delay share the same budget, so the end-to-end response
cannot silently drift past the regulatory figure.

Motion is decided by two numbers from the 1 s window:

* ``ratio``     -- detrended residual energy divided by the calibrated
                   background level. Detrending over 1 s removes most of a
                   breathing excursion while a limb movement leaves a large
                   residual.
* ``coherence`` -- share of residual variance explained by the first singular
                   vector across subcarriers. A body pushes every multipath by
                   the same delay, so the residuals stay correlated; ambient
                   noise bursts hit subcarrier independently and stay at the
                   background level.
"""
from __future__ import annotations

from collections import deque
import numpy as np

from .config import DetectorConfig

STATE_LABELS = {
    "CALIBRATING": "背景校准",
    "MONITORING": "监测中 · 无儿童",
    "PRESENCE": "检测到儿童",
    "ALARM": "告警",
    "ACKED": "告警已静音",
}


class CPDDetector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.state = "CALIBRATING"
        self.calibrated = False
        self.calib_values: list[float] = []
        self.calib_coherence: list[float] = []
        self.baseline_energy: float | None = None
        self.motion_threshold = cfg.motion_ratio_floor()
        self.coherence_threshold = cfg.motion_coherence_min
        self.coherence_spread = 0.0

        self.motion_ratio = 1.0
        self.motion_coherence = 0.0
        self.motion_detected = False
        self.interference = False

        self.evidence = deque()          # (t, bool)
        self.presence = False
        self.presence_since: float | None = None
        self.presence_score = 0.0
        self.first_evidence_t: float | None = None
        self.last_evidence_t: float | None = None
        self.last_motion_t: float | None = None
        self.last_breath_t: float | None = None
        self.breathing_ever = False
        self.breath_source = None        # "motion" | "breathing"

        self.alarm = False
        self.alarm_reason: str | None = None
        self.alarm_since: float | None = None
        self.acknowledged = False
        self.countdown = cfg.response_budget_s

        self.bpm_history: deque = deque(maxlen=13)
        self.bpm = None
        self.bpm_stale_s = None
        self.breath_confidence = 0.0
        self._lag_history: deque = deque(maxlen=7)
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
        self._event(
            t,
            "state",
            message or f"{STATE_LABELS.get(old, old)} → {STATE_LABELS.get(new_state, new_state)}",
        )

    def calibrate(self, t: float, energy: float, coherence: float) -> None:
        self.calib_values.append(float(energy))
        self.calib_coherence.append(float(coherence))
        if t >= self.cfg.calibration_s and len(self.calib_values) >= 20:
            vals = np.asarray(self.calib_values, dtype=float)
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med))) * 1.4826
            # Background level plus a small guard band so a calm cabin sits
            # comfortably below ratio = 1.
            self.baseline_energy = max(med + 0.5 * mad, 1e-6)
            # The coherence gate has to scale with the carrier count, not sit on
            # a fixed offset from the quiet-window median. A rank-1 energy share
            # estimated from N carriers has a null expectation near 1/N and a
            # spread that grows as N shrinks, so a fixed +0.015 keeps almost no
            # margin once the band narrows: measured at 17 carriers the
            # quiet-window median was 0.113 (versus 0.087 at 24), putting a
            # +0.015 threshold at 0.128 against a sustained-noise peak of 0.129
            # -- a door slam confirmed as a child. Median + 5.5 robust sigma
            # tracks the spread instead: it still lands on 0.115 at 24 carriers
            # (absolute floor unchanged) and rises to 0.142 at 17, which clears
            # the noise peak (0.124) while the weakest real-motion frames
            # (0.185) stay well above it.
            coh_vals = np.asarray(self.calib_coherence, dtype=float)
            coh = float(np.median(coh_vals))
            coh_mad = float(np.median(np.abs(coh_vals - coh))) * 1.4826
            self.coherence_threshold = min(
                0.22, max(self.cfg.motion_coherence_min, coh + 5.5 * coh_mad)
            )
            self.coherence_spread = coh_mad
            self.motion_threshold = self.cfg.motion_ratio_floor()
            self.calibrated = True
            self._set_state(t, "MONITORING", "背景校准完成")

    def _breath_lag_is_stable(self, breath: dict) -> bool:
        """Does this breathing peak agree with the recent ones?

        Used only when the motion channel is flagged, to tell "the body really is
        moving, but it is also still breathing" (a stable periodic peak) apart
        from "the ACF is being bent by motion" (a wandering peak).
        """
        lag = breath.get("lag_s")
        if not lag or not breath.get("valid"):
            return False
        if float(breath.get("confidence", 0.0)) < self.cfg.breath_conf_threshold:
            return False
        history = list(self._lag_history)
        if len(history) < 3:
            return False
        median_lag = float(np.median(history))
        return abs(float(lag) - median_lag) <= 0.10 * median_lag

    # ------------------------------------------------------------------
    def update(self, t: float, motion: dict, breath: dict) -> dict:
        energy = float(motion.get("energy", 0.0))
        coherence = float(motion.get("coherence", 0.0))
        self.motion_coherence = coherence
        self.interference = bool(motion.get("interference", False))

        if not self.calibrated:
            self.calibrate(t, energy, coherence)
            return self.snapshot(t)

        base = self.baseline_energy or 1e-6
        ratio = energy / base
        self.motion_ratio = ratio
        motion_det = bool(
            not self.interference
            and ratio > self.motion_threshold
            and coherence >= self.coherence_threshold
        )

        # Only look for breathing when the body is still (paper §4.3) -- with one
        # exception. "Still" is inferred from the motion channel, and the motion
        # channel measures coherent dynamics *relative to the noise floor*. In a
        # quiet cabin the chest's own residual (a 1 s linear detrend does not
        # remove the curvature of an 18 BPM excursion) is by itself several times
        # the floor, so the motion flag latches on and the breathing channel is
        # muted forever -- the quieter the recording, the less breathing you get.
        # A limb bump, unlike breathing, does not leave a stable periodic ACF
        # peak, so while motion is flagged the breathing peak is accepted only
        # if its lag agrees with the last few estimates.
        allow_breath = not motion_det or self._breath_lag_is_stable(breath)
        peak_valid = bool(
            allow_breath
            and breath.get("valid")
            and breath.get("confidence", 0) >= self.cfg.breath_conf_threshold
        )
        bpm = breath.get("bpm") if peak_valid else None
        conf = float(breath.get("confidence", 0.0)) if peak_valid else 0.0

        # Track plausible lags even while motion mutes the output: the history is
        # what makes the stability test above possible at all.
        if (
            breath.get("valid")
            and breath.get("confidence", 0) >= self.cfg.breath_conf_threshold
            and breath.get("lag_s")
        ):
            self._lag_history.append(float(breath["lag_s"]))

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
        # Impulse interference voids the frame entirely (paper §5).
        if self.interference:
            evidence = False

        self.motion_detected = motion_det
        self.evidence.append((t, evidence))
        while self.evidence and self.evidence[0][0] < t - self.cfg.evidence_window_s:
            self.evidence.popleft()
        self.presence_score = (
            float(np.mean([e for _, e in self.evidence])) if self.evidence else 0.0
        )

        if motion_det:
            self.last_motion_t = t
        if evidence:
            # A fresh episode restarts the response budget.
            if (
                self.last_evidence_t is None
                or t - self.last_evidence_t > self.cfg.evidence_gap_reset_s
            ):
                self.first_evidence_t = t
            self.last_evidence_t = t

        if not self.presence and self.presence_score >= self.cfg.presence_ratio:
            self.presence = True
            self.presence_since = t
            self.breath_source = "motion" if motion_det else "breathing"
            self._event(
                t, "presence",
                "检测到车内儿童（%s证据持续）" % ("运动" if motion_det else "呼吸")
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
            self.first_evidence_t = None
            self._event(t, "cleared", "连续无存在证据，恢复安全状态")

        # Apnea alarm: confirmed breathing presence, then breathing stops while
        # the infant is still and no motion evidence is available.
        apnea = (
            self.presence
            and self.breathing_ever
            and self.last_breath_t is not None
            and (t - self.last_breath_t) >= self.cfg.apnea_timeout_s
            and (self.last_motion_t is None or (t - self.last_motion_t) >= min(6.0, self.cfg.apnea_timeout_s))
        )

        if not self.alarm and self.presence:
            anchor = self.first_evidence_t
            if anchor is None:
                anchor = self.presence_since if self.presence_since is not None else t
            self.countdown = max(0.0, self.cfg.response_budget_s - (t - anchor))
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
    def countdown_at(self, t: float) -> float | None:
        if not self.presence or self.alarm:
            return None
        anchor = self.first_evidence_t
        if anchor is None:
            anchor = self.presence_since if self.presence_since is not None else t
        return max(0.0, self.cfg.response_budget_s - (t - anchor))

    def snapshot(self, t: float) -> dict:
        countdown = self.countdown_at(t)
        return {
            "state": self.state,
            "calibrated": self.calibrated,
            "calibration_progress": float(
                min(1.0, t / self.cfg.calibration_s) if not self.calibrated else 1.0
            ),
            "motion_threshold": round(self.motion_threshold, 4),
            "coherence_threshold": round(self.coherence_threshold, 3),
            "motion_ratio": round(float(self.motion_ratio), 3),
            "motion_detected": bool(self.motion_detected),
            "motion_coherence": round(float(self.motion_coherence), 3),
            "interference": bool(self.interference),
            "presence": self.presence,
            "presence_score": round(self.presence_score, 3),
            "presence_since": self.presence_since,
            "first_evidence_t": self.first_evidence_t,
            "breath_source": self.breath_source,
            "countdown": None if countdown is None else round(countdown, 2),
            "response_budget_s": self.cfg.response_budget_s,
            "alarm": self.alarm,
            "alarm_reason": self.alarm_reason,
            "alarm_since": self.alarm_since,
            "acknowledged": self.acknowledged,
            "bpm": None if self.bpm is None else round(float(self.bpm), 1),
            "bpm_stale_s": None if self.bpm_stale_s is None else round(float(self.bpm_stale_s), 1),
            "breath_confidence": round(float(self.breath_confidence), 3),
        }
