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

from .config import DECISION_FS, DetectorConfig

STATE_LABELS = {
    "MONITORING": "监测中 · 无儿童",
    "PRESENCE": "检测到儿童",
    "ALARM": "告警",
    "ACKED": "告警已静音",
}


class CPDDetector:
    def __init__(self, cfg: DetectorConfig, background: dict | None = None):
        self.cfg = cfg
        # 冷启动即进入监测，不再要求开机后先安静 5 s。
        self.state = "MONITORING"
        self.calibrated = False
        self.background_source = "冷启动自适应"
        self._bg_energy: deque = deque(maxlen=20)
        self._bg_coherence: deque = deque(maxlen=20)
        self.baseline_energy: float | None = None
        self.motion_threshold = cfg.motion_ratio_floor()
        self.coherence_threshold = 0.0
        self.coherence_floor = 0.0
        self.coherence_spread = 0.0
        if background:
            self.install_background(0.0, background)

        self.motion_ratio = 1.0
        self.motion_coherence = 0.0
        self.motion_detected = False
        self.interference = False
        self.implausible_energy = False

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

    def install_background(self, t: float, profile: dict) -> None:
        """加载空车标定基线，冻结自适应估计。"""
        self.baseline_energy = max(float(profile["baseline_energy"]), 1e-9)
        self.coherence_threshold = float(profile.get("coherence_threshold", 0.0))
        self.coherence_floor = float(profile.get("coherence_floor", 0.0))
        self.coherence_spread = float(profile.get("coherence_spread", 0.0))
        self.background_source = "空车标定"
        self.calibrated = True
        if t > 0:
            self._event(t, "calibration", "已加载空车标定基线")

    def _update_background(self, t: float, energy: float, coherence: float,
                           coherence_floor: float) -> None:
        """冷启动背景估计：无空车基线时用开机短窗的低分位残差。

        只使用约 2 s 的样本：它刚好看得见一个运动窗内的安静间隙，又不足以把
        儿童持续的呼吸起伏吸收进背景。估计完成后即冻结；若座舱环境发生长期
        变化，应重新执行空车标定，而不是在线慢慢“学”成背景。
        """
        if self.interference or self.presence or self.calibrated:
            return
        self._bg_energy.append(float(energy))
        self._bg_coherence.append(float(coherence))
        vals = np.asarray(self._bg_energy, dtype=float)
        # 低分位而非中位数：开机时儿童可能正在动，噪声底由短窗内最安静的
        # 10% 帧代表；若取 25% 分位，一次 1 s 的落座动作就会把分母抬高约 30%，
        # 使后续真实肢体运动擦着 1.6 倍门限漏掉。
        self.baseline_energy = max(float(np.percentile(vals, 10)), 1e-9)
        coh_vals = np.asarray(self._bg_coherence, dtype=float)
        # 相干性同样取低分位，避免把运动期的高相干值写进背景门限。
        coh = float(np.percentile(coh_vals, 25))
        coh_mad = float(np.median(np.abs(coh_vals - coh))) * 1.4826
        # 噪声底由 iid 噪声的 Marchenko–Pastur 上界给出，并随实际载波数变化；
        # 不再夹在 0.22 以内 —— 真实空车的相干残差可达 0.64，硬封顶会让门控
        # 完全失去区分力。
        self.coherence_floor = float(coherence_floor)
        self.coherence_threshold = max(
            self.coherence_floor, coh + 5.5 * coh_mad
        )
        self.coherence_spread = coh_mad
        if len(self._bg_energy) >= 20:
            self.calibrated = True
            self._event(t, "calibration", "冷启动背景已建立（自适应，无需空车静置）")

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
        coherence_floor = float(motion.get("coherence_floor", 0.0))
        self.motion_coherence = coherence
        self.coherence_floor = max(self.coherence_floor, coherence_floor)
        self.interference = bool(motion.get("interference", False))

        if self.background_source == "冷启动自适应":
            self._update_background(t, energy, coherence, coherence_floor)

        base = self.baseline_energy or max(energy, 1e-9)
        ratio = energy / base
        self.motion_ratio = ratio
        # 强噪声突发会让相对残差远超真实体动的物理上限。这不是长时 ACF 判据
        # （那条在真实语料上从未越过门控，且会把新生儿呼吸误标），而是一道
        # 独立的安全兜底：这类帧既不产生运动证据，也不进入呼吸判决。
        self.implausible_energy = bool(ratio > self.cfg.motion_ratio_max)
        interference_now = bool(self.interference or self.implausible_energy)
        # 门限本身来自空车分布的低分位。若只要求“达到门限”，噪声尾巴和
        # 真实运动会在边界上互相重叠（4 s 突发恰好能贴着门限停留 1.5 s）。
        # 再抬高一个实测散布（MAD），使证据必须明显高于背景分布；散布是
        # 标定/冷启动观测到的量，不是新写死的常数。
        coherence_gate = max(self.coherence_floor, self.coherence_threshold) \
            + self.coherence_spread
        motion_det = bool(
            not interference_now
            and ratio > self.motion_threshold
            and coherence > coherence_gate
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
        # 脉冲干扰会整帧作废（论文 §5）；长时高能突发同样作废。
        if interference_now:
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
            # 响应预算锚定在“当前这段证据”的起点。未确认存在前，长间隔会
            # 开启新的一段证据并重置预算；一旦确认存在，锚点就冻结 —— 否则
            # 后续每次短暂运动都会把预算往后推，端到端响应时间就会突破 10 s。
            if self.first_evidence_t is None:
                self.first_evidence_t = t
            elif (
                not self.presence
                and self.last_evidence_t is not None
                and t - self.last_evidence_t > self.cfg.evidence_gap_reset_s
            ):
                self.first_evidence_t = t
            self.last_evidence_t = t

        # 确认存在需要足够的观测样本：5 s 窗的 30 % 至少是 1.5 s（15 个判决帧）。
        # 少了这个下限，开机头几帧里只要有 1 帧噪声触发运动判据，就会让
        # “1/2 = 50 %” 的占比成立，从而过早锁存存在。
        min_evidence_samples = max(
            1, int(round(self.cfg.evidence_window_s * DECISION_FS * self.cfg.presence_ratio))
        )
        if (
            not self.presence
            and len(self.evidence) >= min_evidence_samples
            and self.presence_score >= self.cfg.presence_ratio
        ):
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
        if self.alarm and not self.acknowledged:
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
            "background_source": self.background_source,
            "baseline_energy": None if self.baseline_energy is None
            else round(float(self.baseline_energy), 5),
            "calibration_progress": 1.0 if self.calibrated
            else min(1.0, len(self._bg_energy) / 20.0),
            "motion_threshold": round(self.motion_threshold, 4),
            "coherence_threshold": round(self.coherence_threshold, 3),
            "coherence_floor": round(self.coherence_floor, 4),
            "motion_ratio": round(float(self.motion_ratio), 3),
            "motion_detected": bool(self.motion_detected),
            "motion_coherence": round(float(self.motion_coherence), 3),
            "interference": bool(self.interference),
            "implausible_energy": bool(self.implausible_energy),
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
