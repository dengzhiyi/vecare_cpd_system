"""功能样机使用的预置座舱场景。

每个场景都描述一条真值时间线（仅用于仿真以及 UI 中的事件标记）以及声学场景参数。
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict


@dataclass
class TargetSpec:
    enter_t: float = 0.0
    exit_t: float | None = None
    distance_m: float = 1.4
    bpm: float = 20.0
    chest_mm: float = 4.5
    reflectivity: float = 0.045
    breathing_windows: list[list[float]] = field(default_factory=list)
    motion_events: list[dict] = field(default_factory=list)
    awake_jitter_mm: float = 0.0
    label: str = "婴儿模拟器"


@dataclass
class Scenario:
    key: str
    name: str
    subtitle: str
    description: str
    duration_s: float
    category: str = "presence"
    target: TargetSpec | None = None
    noise_sigma: float = 0.020
    noise_bursts: list[dict] = field(default_factory=list)
    vibration_bursts: list[dict] = field(default_factory=list)
    ambient: str = "停车场 · 闭窗 · 空调关闭"
    expected: str = ""

    def public(self) -> dict:
        d = asdict(self)
        d["target"] = asdict(self.target) if self.target else None
        return d


def _target(**kw) -> TargetSpec:
    return TargetSpec(**kw)


SCENARIOS: dict[str, Scenario] = {}


def _add(s: Scenario) -> None:
    SCENARIOS[s.key] = s


_add(Scenario(
    key="empty",
    name="空车厢",
    subtitle="No occupant",
    description="闭窗停车、空调关闭的安静车厢。用于建立背景噪声基线并验证零误报。",
    duration_s=55,
    category="baseline",
    noise_sigma=0.020,
    expected="保持 SAFE / MONITORING，全程不触发告警",
))

_add(Scenario(
    key="sleeping",
    name="入睡婴儿",
    subtitle="Breathing only",
    description="婴儿被留在后排儿童座椅上入睡，仅胸腔呼吸起伏（约 18 BPM），无肢体动作。",
    duration_s=65,
    category="presence",
    target=_target(
        enter_t=0.0, distance_m=1.55, bpm=18.0, chest_mm=3.0,
        reflectivity=0.030,
        breathing_windows=[[0.0, 200.0]],
        motion_events=[{"t": 0.0, "duration": 1.0, "amplitude_m": 0.16, "kind": "settle"}],
        label="入睡婴儿 · 18 BPM",
    ),
    noise_sigma=0.020,
    expected="开机时儿童已在车内且无空车基线；确认后 10 s 内告警",
))

_add(Scenario(
    key="active",
    name="清醒婴儿",
    subtitle="Motion + breathing",
    description="婴儿清醒，间歇挥动四肢（每 8–11 s 一次小幅肢体动作），同时保持呼吸。",
    duration_s=70,
    category="presence",
    target=_target(
        enter_t=0.0, distance_m=1.35, bpm=24.0, chest_mm=3.0,
        reflectivity=0.035, awake_jitter_mm=1.2,
        breathing_windows=[[0.0, 200.0]],
        motion_events=[
            {"t": 0.0, "duration": 1.2, "amplitude_m": 0.18, "kind": "settle"},
            {"t": 17.0, "duration": 1.5, "amplitude_m": 0.10, "kind": "limb"},
            {"t": 27.0, "duration": 1.6, "amplitude_m": 0.12, "kind": "limb"},
            {"t": 38.0, "duration": 1.5, "amplitude_m": 0.09, "kind": "limb"},
            {"t": 50.0, "duration": 1.4, "amplitude_m": 0.11, "kind": "limb"},
        ],
        label="清醒婴儿 · 肢体动作",
    ),
    noise_sigma=0.022,
    expected="开机即已存在；运动证据快速确认，呼吸率随肢体间隙更新",
))

_add(Scenario(
    key="apnea",
    name="呼吸暂停",
    subtitle="Apnea event",
    description="婴儿入睡后被确认存在，但在 42 s 后呼吸停止，且无肢体动作，触发呼吸异常告警。",
    duration_s=80,
    category="critical",
    target=_target(
        enter_t=0.0, distance_m=1.45, bpm=20.0, chest_mm=3.0,
        reflectivity=0.032,
        breathing_windows=[[0.0, 42.0]],
        motion_events=[{"t": 0.0, "duration": 1.0, "amplitude_m": 0.16, "kind": "settle"}],
        label="呼吸暂停事件",
    ),
    noise_sigma=0.020,
    expected="42 s 呼吸停止；约 15–18 s 后升级为 APNEA 告警",))

_add(Scenario(
    key="newborn",
    name="弱呼吸新生儿",
    subtitle="Low SNR",
    description="1.6 m 距离、低反射强度的新生儿模拟器，呼吸约 30 BPM，背景噪声偏高。",
    duration_s=72,
    category="critical",
    target=_target(
        enter_t=0.0, distance_m=1.60, bpm=30.0, chest_mm=2.6,
        reflectivity=0.024,
        breathing_windows=[[0.0, 200.0]],
        motion_events=[{"t": 0.0, "duration": 0.8, "amplitude_m": 0.14, "kind": "settle"}],
        label="新生儿 · 弱呼吸",
    ),
    noise_sigma=0.034,
    expected="挑战场景：仍应检测到呼吸（置信度较低），可调高灵敏度",
))

_add(Scenario(
    key="stress",
    name="噪声压力测试",
    subtitle="No child",
    description="空车厢中叠加关门、车外谈话、车辆经过，以及一段持续 3.4 s 的强噪声，验证告警抑制。",
    duration_s=65,
    category="baseline",
    noise_sigma=0.024,
    noise_bursts=[
        {"t": 16.0, "duration": 0.8, "multiplier": 4.5},
        {"t": 31.0, "duration": 2.6, "multiplier": 2.2},
        {"t": 46.0, "duration": 1.0, "multiplier": 3.8},
        # 持续干扰：同时考验相干性门控和脉冲模式检测器
        {"t": 55.0, "duration": 3.4, "multiplier": 2.6},
    ],
    vibration_bursts=[
        {"t": 16.0, "duration": 0.3, "amplitude_mm": 0.06},
        {"t": 46.0, "duration": 0.3, "amplitude_mm": 0.05},
    ],
    expected="瞬态与持续噪声都不应形成存在证据，保持 SAFE",
))

_add(Scenario(
    key="adult",
    name="成人乘客",
    subtitle="Strong breathing",
    description="2.6 m 处的成人静坐呼吸（约 14 BPM），用于展示强反射条件下的稳定跟踪。",
    duration_s=60,
    category="presence",
    target=_target(
        enter_t=0.0, distance_m=2.6, bpm=14.0, chest_mm=4.0,
        reflectivity=0.045,
        breathing_windows=[[0.0, 200.0]],
        motion_events=[{"t": 0.0, "duration": 1.0, "amplitude_m": 0.17, "kind": "settle"}],
        label="成人 · 14 BPM",
    ),
    noise_sigma=0.018,
    expected="强信号：快速确认并稳定输出 14 BPM",
))


def get_scenario(key: str) -> Scenario:
    return SCENARIOS.get(key, SCENARIOS["sleeping"])


def list_scenarios() -> list[dict]:
    return [s.public() for s in SCENARIOS.values()]
