"""座舱声学多径模拟器。

实现 VeCare 论文中 Eq. (1) 的信号模型：

    H(f, t) = sum_r a_r(t) exp(-j 2 pi f tau_r(t)) + N(f, t)

一个静态的座舱多径剖面（直达路径、座椅、车窗、车顶）与来自呼吸/运动婴儿的
动态反射相混合。微弱的人体反射与静态路径相互干涉，产生 SAS 模型所利用的
频率选择性灵敏度。噪声突发和座舱振动让我们可以检验误报抑制能力。

介质来源说明（解释验证过程时很重要）：
    本路径每 20 ms 依据物理参数合成 ``|H(f)|``。它从不读取 WAV 文件，因此直接
    仿真（Web UI 与 ``run_offline``）完全不需要座舱录音。WAV 仅用于检验音频
    前端的自检；``core.replay.render_wav`` 会按需在 ``data/selfcheck/``
    （已被 git 忽略）下生成它们，这些文件从不随仓库发布。
    真车验证则需要一段在播放 ``probe.wav`` 时录制的 ``capture.wav``；
    :mod:`core.replay` 会把它转换成同样的 CSI。
"""
from __future__ import annotations

import numpy as np

from .config import CSI_FS, F_HIGH, F_LOW, N_SUBCARRIERS, SOUND_SPEED, AcousticBand, plan_band
from .scenarios import Scenario, TargetSpec


class AcousticChannel:
    def __init__(
        self,
        scenario: Scenario,
        seed: int = 20231102,
        band: AcousticBand | None = None,
    ):
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        # 载波集合跟随硬件上限，与音频前端完全一致 —— 否则仿真的真值就会在
        # 一个真实驱动器无法激励的频带上被测量。
        self.band = band or plan_band(None)
        self.carriers = np.linspace(
            self.band.f_lo, self.band.f_hi, self.band.n_carriers
        )
        self.t = 0.0
        self._phase = self.rng.uniform(0, 2 * np.pi)
        self._last_breath_t = 0.0
        self._present_since = None

        # 静态座舱多径。各时延为往返路径时延，单位为秒。
        self.static_delays = np.array(
            [0.00058, 0.0017, 0.0028, 0.0039, 0.0051, 0.0064, 0.0079, 0.0092]
        )
        base_gain = np.array([1.0, 0.58, 0.44, 0.35, 0.30, 0.24, 0.19, 0.15])
        self.static_gains = base_gain * np.exp(
            1j * self.rng.uniform(-np.pi, np.pi, len(base_gain))
        )
        self._target = scenario.target
        self._vib_phase = self.rng.uniform(0, 2 * np.pi, len(base_gain))

    def _target_visible(self, t: float) -> bool:
        tg = self._target
        if tg is None:
            return False
        return t >= tg.enter_t and (tg.exit_t is None or t < tg.exit_t)

    def _breathing_active(self, t: float) -> bool:
        tg = self._target
        if tg is None:
            return False
        if not tg.breathing_windows:
            return True
        return any(a <= t < b for a, b in tg.breathing_windows)

    def _motion_offset(self, t: float) -> np.ndarray:
        tg = self._target
        if tg is None or not self._target_visible(t):
            return 0.0

        offset = 0.0
        for ev in tg.motion_events:
            start, dur = ev["t"], ev["duration"]
            if start <= t < start + dur:
                u = (t - start) / dur
                offset += ev["amplitude_m"] * 0.5 * (1 - np.cos(2 * np.pi * u))

        if tg.awake_jitter_mm > 0:
            # 清醒人体的缓慢姿态漂移（<0.15 Hz）。它必须远低于 10-66 BPM 的
            # 呼吸频带：一个相干的 0.2-0.4 Hz 摆动会被当成第二个呼吸节律，
            # 使呼吸率真正变得模糊 —— 那是建模产生的伪影，而不是传感难题。
            j = tg.awake_jitter_mm / 1000.0
            offset += j * (0.6 * np.sin(2 * np.pi * 0.06 * t + 0.4)
                           + 0.4 * np.sin(2 * np.pi * 0.11 * t + 1.7))
        return offset

    def _noise_sigma(self, t: float) -> float:
        sigma = self.scenario.noise_sigma
        for b in self.scenario.noise_bursts:
            if b["t"] <= t < b["t"] + b["duration"]:
                u = (t - b["t"]) / b["duration"]
                shape = np.sin(np.pi * u) ** 2
                sigma += self.scenario.noise_sigma * (b["multiplier"] - 1) * shape
        return sigma

    def _paths(self, t: float, dt: float, advance: bool = True) -> dict:
        """解析出一个 20 ms 帧的时延/增益剖面。

        由 ``step``（会推进呼吸相位）和 ``impulse_response``（绝不能扰动时间线）
        共用。
        """
        tg = self._target
        present = self._target_visible(t)

        breath_amp_m = 0.0
        breathing = False
        if present and tg is not None:
            if self._breathing_active(t):
                if advance:
                    drifting_bpm = tg.bpm * (1 + 0.02 * np.sin(2 * np.pi * 0.013 * t))
                    self._phase += 2 * np.pi * drifting_bpm / 60.0 * dt
                    self._last_breath_t = t
                # 胸腔位移（峰值）；若呼吸停止则平滑衰减
                breath_amp_m = tg.chest_mm / 1000.0 * np.sin(self._phase)
                breathing = True

        motion = self._motion_offset(t)

        # 目标反射：幅度弱，且随距离衰减。进入后在短暂时间内逐渐接入。
        a_target = 0.0
        if present and tg is not None:
            age = t - tg.enter_t
            ramp = min(1.0, age / 0.25)
            a_target = tg.reflectivity * ramp * (1.4 / max(1.0, tg.distance_m))

        # 往返路径长度。呼吸和身体运动都会调制该时延。
        d = 0.0
        if tg is not None:
            d = tg.distance_m + motion + breath_amp_m
        tau_target = 2 * d / SOUND_SPEED if present else 0.0

        # 关门 / 车辆驶过事件期间的座舱振动：差分路径长度抖动会使静态信道去相关。
        delays = self.static_delays.copy()
        vib_amp = 0.0
        for b in self.scenario.vibration_bursts:
            if b["t"] <= t < b["t"] + b["duration"]:
                u = (t - b["t"]) / b["duration"]
                vib_amp = b["amplitude_mm"] / 1000.0 * np.sin(np.pi * u) ** 2
        if vib_amp > 0:
            # vib_amp 是*位移*，单位为米；而时延单位为秒，所以必须除以声速。
            # 把米直接加到秒上，会把 0.05 mm 的面板振动放大成 50 us 的时延抖动
            # （相当于 17 mm 的表观位移），使每次关门看起来都像一个运动的人体。
            delays = delays + (vib_amp / SOUND_SPEED) * np.sin(
                2 * np.pi * 120.0 * t + self._vib_phase
            )

        return {
            "present": present,
            "breathing": breathing,
            "breath_amp_m": breath_amp_m,
            "motion": motion,
            "a_target": a_target,
            "tau_target": tau_target,
            "delays": delays,
            "vib_amp": vib_amp,
        }

    def sounding_step(self, dt: float) -> tuple[np.ndarray, np.ndarray, dict]:
        """推进场景并返回该帧的 (时延, 增益, 真值)。

        与 ``step`` 不同，它既不计算 24 载波 CSI，也不绘制噪声：音频渲染器需要
        连续频率的 CIR，以便探测信号能在每个音频频点与它卷积。
        """
        t = self.t
        p = self._paths(t, dt=dt, advance=True)
        delays = list(p["delays"])
        gains = list(self.static_gains)
        if p["present"] and p["a_target"] > 0:
            delays.append(p["tau_target"])
            gains.append(complex(p["a_target"]))

        tg = self._target
        truth = {
            "present": bool(p["present"]),
            "breathing": bool(p["breathing"]),
            "bpm": float(tg.bpm) if tg else None,
            "distance_m": float(tg.distance_m + p["motion"] + p["breath_amp_m"])
            if tg
            else None,
            "motion_offset_m": float(p["motion"]),
            "target_amplitude": float(p["a_target"]),
            "vibration": float(p["vib_amp"]),
            "noise_sigma": float(self.scenario.noise_sigma),
            "noise_gain": float(self._noise_sigma(t) / max(1e-9, self.scenario.noise_sigma)),
        }
        self.t += dt
        return np.asarray(delays, dtype=float), np.asarray(gains, dtype=complex), truth

    def step(self, dt: float) -> tuple[np.ndarray, np.ndarray, dict]:
        """把场景推进 dt 秒，并返回 (CSI, 幅度, 真值)。"""
        t = self.t
        tg = self._target
        p = self._paths(t, dt=dt, advance=True)
        present = p["present"]
        breathing = p["breathing"]
        breath_amp_m = p["breath_amp_m"]
        motion = p["motion"]
        a_target = p["a_target"]
        tau_target = p["tau_target"]
        delays = p["delays"]
        vib_amp = p["vib_amp"]

        # 公共相位偏移用于建模扬声器/麦克风的异步；当只使用 CSI 幅度时它会抵消。
        common_phase = self.rng.uniform(-np.pi, np.pi)
        static = self.static_gains[:, None] * np.exp(
            -1j * 2 * np.pi * self.carriers[None, :] * delays[:, None]
        )
        h = static.sum(axis=0) * np.exp(1j * common_phase)

        if present and a_target > 0:
            h = h + a_target * np.exp(
                -1j * 2 * np.pi * self.carriers * tau_target
            ) * np.exp(1j * common_phase)

        sigma = self._noise_sigma(t)
        n_carriers = self.carriers.size
        noise = sigma / np.sqrt(2) * (
            self.rng.standard_normal(n_carriers)
            + 1j * self.rng.standard_normal(n_carriers)
        )
        h = h + noise

        true_distance = float(tg.distance_m + motion + breath_amp_m) if tg else None
        truth = {
            "present": bool(present),
            "breathing": bool(breathing),
            "bpm": float(tg.bpm) if tg else None,
            "distance_m": true_distance,
            "motion_offset_m": float(motion),
            "noise_sigma": float(sigma),
            "target_amplitude": float(a_target),
            "vibration": float(vib_amp),
        }
        self.t += dt
        return h, np.abs(h), truth
