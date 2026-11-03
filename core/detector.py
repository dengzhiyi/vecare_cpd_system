"""儿童存在检测状态机与告警逻辑。

运动证据和呼吸证据在 5 s 的判决窗内聚合；当证据占据该窗 30 % 以上时，即判定
车内有儿童，与论文一致。响应预算（默认 10 s）从*首个证据样本*起算，
而不是从确认存在的那一刻起算：检测延迟和确认延迟共用同一预算，因此端到端
响应时间不会在无人察觉的情况下超出法规规定的数值。

运动由 1 s 窗内的两个数值来判定：

* ``ratio``     —— 去趋势后的残差能量除以校准得到的背景电平。在 1 s 上
                   去趋势会去掉呼吸起伏的大部分，而肢体运动则会留下很大的残差。
* ``coherence`` —— 跨子载波由第一奇异向量解释的残差方差占比。人体会让
                   每条多径产生相同的时延位移，因此残差保持相关；环境噪声突发
                   独立地作用于各子载波，维持在背景电平附近。
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

        self.evidence = deque()          # (时间, 布尔值)
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
            # 背景电平加上一个小的保护带，使安静的车厢舒适地落在 ratio = 1 以下。
            self.baseline_energy = max(med + 0.5 * mad, 1e-6)
            # 相干性门限必须随载波数量缩放，而不能固定在安静窗中位数之上的
            # 某个常量偏移。由 N 个载波估计出的秩 1 能量占比，其零假设期望约为
            # 1/N，且离散程度随 N 减小而增大，因此一旦频带收窄，固定的 +0.015
            # 几乎留不下余量：实测在 17 个载波时，安静窗中位数为 0.113
            #（24 个载波时为 0.087），+0.015 的门限为 0.128，而持续噪声的峰值
            # 为 0.129 —— 一次关门被确认为儿童。中位数 + 5.5 倍稳健 sigma 则能
            # 跟随离散程度：在 24 个载波时它仍落在 0.115（绝对下限不变），
            # 在 17 个载波时升到 0.142，既高于噪声峰值（0.124），而最弱的真实
            # 运动帧（0.185）仍远高于它。
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
        """这个呼吸峰与最近的几个峰一致吗？

        仅在 motion 通道被标记时使用，用于区分“身体确实在动，但仍在呼吸”
        （稳定的周期峰）和“ACF 正被运动扭曲”（漂移的峰）。
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

        # 只有在身体静止时才去寻找呼吸（论文 §4.3）—— 但有一个例外。
        # “静止”是由 motion 通道推断的，而 motion 通道度量的是*相对于噪声底*
        # 的相干动态。在安静车厢里，胸腔自身的残差（1 s 线性去趋势无法去除
        # 18 BPM 起伏的曲率）本身就已是噪声底的好几倍，于是运动标志一直置位，
        # 呼吸通道被永久静音 —— 录音越安静，反而越测不到呼吸。
        # 肢体碰撞与呼吸不同，不会留下稳定的周期性 ACF 峰，因此在运动被标记期间，
        # 只有当呼吸峰的 lag 与最近几次估计一致时才接受它。
        allow_breath = not motion_det or self._breath_lag_is_stable(breath)
        peak_valid = bool(
            allow_breath
            and breath.get("valid")
            and breath.get("confidence", 0) >= self.cfg.breath_conf_threshold
        )
        bpm = breath.get("bpm") if peak_valid else None
        conf = float(breath.get("confidence", 0.0)) if peak_valid else 0.0

        # 即使运动导致输出被静音，也要记录可信的 lag：正是这段历史让上面的
        # 稳定性检验成为可能。
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
        # 脉冲干扰会整帧作废（论文 §5）。
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
            # 新的一段证据会重新开始计算响应预算。
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

        # 存在锁存的解除：在长时间没有证据之后。未被确认的告警会一直保持锁存，
        # 直到操作员做出响应；一旦被确认，静默即意味着儿童已经离开，此时告警
        # 和存在锁存都会被清除，并恢复监测。
        if (
            self.presence
            and (not self.alarm or self.acknowledged)
            and self.last_evidence_t is not None
            and t - self.last_evidence_t > self.cfg.clear_timeout_s
        ):
            self.presence = False
            self.presence_since = None
            self.presence_score = 0.0
            self.first_evidence_t = None
            self.last_evidence_t = None
            self.last_motion_t = None
            self.last_breath_t = None
            self.breathing_ever = False
            self.bpm = None
            self.bpm_history.clear()
            self.alarm = False
            self.alarm_reason = None
            self.alarm_since = None
            self.acknowledged = False
            self.countdown = self.cfg.response_budget_s
            self._event(t, "cleared", "连续无存在证据，恢复安全状态")

        # 呼吸暂停告警：已确认存在呼吸，随后呼吸停止，而婴儿保持静止且
        # 没有可用的运动证据。
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
            elif self.countdown <= 1e-6:
                # 50 Hz 逐帧累积浮点误差可能让最后一帧距离预算还差几个 ulp；
                # 没有这个保护，10.0 s 的预算会晚一个判决帧触发（10.1 s）。
                self._raise_alarm(t, "PRESENCE", "儿童被留在车内，触发儿童存在告警")
        elif self.alarm and apnea and self.alarm_reason != "APNEA":
            # 已经在为存在告警：如果随后呼吸停止（危急情况），则升级告警原因。
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
