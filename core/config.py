"""VeCare 原型配置。

各项取值遵循 NSDI'23 论文中描述的 SAS / VeCare 处理链路。
该原型使用模拟声学前端（一个座舱扬声器 + 一个麦克风），声采样率为 48 kHz，
并压缩为 50 Hz 的 CSI 帧。

两个通道共享同一 CSI，但测量的是不同的物理量：

* motion —— 1 s 窗（论文 §4.3 “e.g., 1s”），使用原始幅度，不做高通滤波。
  判决统计量为去趋势后的残差能量，用校准得到的背景电平做归一化，
  并由子载波间相干性进行门控，使不相关的环境噪声突发无法伪装成一个运动的人体。
* breathing —— 8 s 窗，逐子载波 ACF 按权重 g(f) = rho(f, 1/Fs) 经 MRC 合并，
  与 Eq. (4) 和 Eq. (6) 一致。
"""
from dataclasses import dataclass, asdict

SOUND_SPEED = 343.0          # m/s，舱内空气声速
AUDIO_FS = 48000             # Hz，通用音频采样率
CSI_FS = 50                  # Hz，信道探测速率（Kasami 周期 = 20 ms）
DECISION_FS = 10             # Hz，CPD 判决速率
N_SUBCARRIERS = 24           # 上限，并非承诺值：窄带会用更少的子载波
# 低端边界是固定的，不参与规划：我们关心的每个驱动器都标定为 10 kHz 高通，
# 因此 10.5 kHz —— 论文自己的下边界 —— 距离高通拐点已有 500 Hz 的保护带，
# 并且对任何硬件上限都保持不变。只有频带的上边界跟随硬件变化。如果将来某个设备
# 也提高了它的高通频率，那么需要修改的就只有这一个常量。
F_LOW = 10_500               # Hz，10 kHz 高通 + 500 Hz 保护带，从不参与规划
F_HIGH = 23_500              # Hz，硬件允许时的设计上限

# 探测信号刻意扫得比载波频带更宽，好让升余弦边缘（它必然损失能量）落在频带之外。
# 如果扫频的起点和终点正好落在载波频带上，最外侧的两个载波只能获得 -23 dB 的
# 探测功率，也就是比其余载波的噪声大 200 倍。
SWEEP_LO = 9_800             # Hz
SWEEP_HI = 23_900            # Hz，对 fs = 48 kHz 必须保持在 Nyquist 频率以下
PROBE_TAPER_S = 0.0006       # 0.6 ms；2 ms 的边缘会在频带两端损失 23 dB

# --- 硬件带宽 ----------------------------------------------------------------
# 扬声器和麦克风的频响从来不会平坦到 23.5 kHz。驱动器标称的 “18 kHz” 意味着
# 在该处已是 -3 dB，再往上则是陡峭的滚降，而落在滚降上的载波几乎没有探测功率：
# 谱除法估计会退化成噪声除以噪声。那些载波不只是损失精度，它们还会注入一段
# 宽带随机幅度序列，残差能量和 MRC 权重都无法把它与运动的人体区分开 ——
# 在空车厢上实测，它们曾在 17.6 s 触发一次虚假的存在判定。
#
# 因此频带必须依据硬件来规划，而不能依据论文：
HW_F_MAX = None              # 实测的 -3 dB 上限；None = 假定为 F_HIGH
HW_GUARD_HZ = 1_500          # 让载波比硬件上限低这么多
CARRIER_MIN_SPACING_HZ = 250  # 频率分集下限，见 plan_band 的 docstring
SWEEP_MARGIN_HZ = 700        # 扫频在载波频带两侧各外延的量


@dataclass(frozen=True)
class AcousticBand:
    """某一硬件上限下的一组载波，以及激励它们的扫频信号。"""
    f_lo: float
    f_hi: float
    n_carriers: int
    sweep_lo: float
    sweep_hi: float
    hw_f_max: float

    @property
    def span_hz(self) -> float:
        return self.f_hi - self.f_lo

    @property
    def spacing_hz(self) -> float:
        return self.span_hz / max(1, self.n_carriers - 1)

    @property
    def diversity(self) -> float:
        """``spacing x longest path delay``：相邻载波之间旋转的干涉条纹数。

        当该值低于约 1 时，相邻载波看到的相位几乎相同，不再算作相互独立的观测，
        这会损失 MRC 分集增益，并使子载波间相干性检验变得过于乐观。
        """
        return self.spacing_hz * 9.2e-3

    def describe(self) -> str:
        return (
            f"{self.n_carriers} carriers @ {self.f_lo/1e3:.1f}-{self.f_hi/1e3:.1f} kHz "
            f"({self.spacing_hz:.0f} Hz spacing, diversity {self.diversity:.2f}), "
            f"sweep {self.sweep_lo/1e3:.1f}-{self.sweep_hi/1e3:.1f} kHz"
        )


def plan_band(hw_f_max: float | None = None) -> AcousticBand:
    """根据硬件可用带宽推导载波集合。

    当频带上边界消失时，有三个相互制约的约束：

    1. **保持在滚降之下。** 距离硬件上限 ``HW_GUARD_HZ`` 以内的载波会损失探测
       功率，变成“噪声计”。
    2. **保持足够的间距。** 频率分集需要
       ``spacing x delay_spread >~ 1``；舱内时延扩展为 9.2 ms 时，该值约为
       110 Hz，而 250 Hz 留出了舒适的余量。
    3. **保持足够的载波数。** MRC 要将它们合并，而相干性检验也需要一定的样本
       数量来估计秩 1 分量所占的比例。

    在 18 kHz 上限下仍可容纳 24 个载波（6 kHz 跨度、261 Hz 间距）。在
    16 kHz 上限下跨度只有 4 kHz，因此按 250 Hz 间距载波数降到 17 个 ——
    用分集阶数换间距是这笔交易中正确的一边，因为 17 个载波仍能对噪声取平均，
    而 24 个载波按 174 Hz 间距会开始共享干涉条纹。

    ``hw_f_max=None`` 表示“没有实测限制”：直接使用设计频带，不设保护带，
    因为没有滚降需要避让。

    只有频带的**上边界**参与规划；``f_lo`` 始终等于 ``F_LOW``。10 kHz 高通
    拐点与 ``F_LOW`` 之间的 500 Hz 已经覆盖了它，因此降低上限只会从上方收缩
    频带，绝不会移动低端边界。保持低端固定不只是为了整洁：呼吸通道的逐载波
    ACF 增益是针对固定噪声底标定的，若滑动较低的载波，两次运行之间频带所平均
    的物理路径就会改变。

    实测：通过缩小保护带来反向拓宽频带并没有收益。在 16 kHz 上限下，
    1500 / 800 / 400 Hz 的保护带对应的入睡场景 BPM MAE 分别为
    0.249 / 0.268 / 0.290，运动证据帧数分别为 0 / 1 / 2 ——
    载波更多，BPM 反而略差，且第一种仍会在静止车厢里误触发运动。真正的限制
    是呼吸信号的相干时间，而不是载波数量，所以保守的保护带保留不变。
    """
    if hw_f_max:
        f_hi = min(float(F_HIGH), float(hw_f_max) - HW_GUARD_HZ)
        ceiling = float(hw_f_max)
    else:
        f_hi = float(F_HIGH)
        ceiling = float(F_HIGH)
    if f_hi <= F_LOW + CARRIER_MIN_SPACING_HZ * 3:
        raise ValueError(
            f"hardware ceiling {ceiling/1e3:.1f} kHz leaves only "
            f"{f_hi/1e3:.1f} kHz of usable band above {F_LOW/1e3:.1f} kHz; "
            f"need at least 4 carriers at {CARRIER_MIN_SPACING_HZ:.0f} Hz spacing. "
            f"Lower F_LOW or use a wider driver."
        )
    span = f_hi - F_LOW
    n = min(N_SUBCARRIERS, int(span // CARRIER_MIN_SPACING_HZ) + 1)
    return AcousticBand(
        f_lo=float(F_LOW),
        f_hi=float(f_hi),
        n_carriers=int(n),
        sweep_lo=max(1_000.0, F_LOW - SWEEP_MARGIN_HZ),
        sweep_hi=min(AUDIO_FS / 2 - 100.0, f_hi + SWEEP_MARGIN_HZ),
        hw_f_max=ceiling,
    )

# --- motion 通道 ------------------------------------------------------------
MOTION_WINDOW_S = 1.0        # 论文 §4.3：短窗以保证响应速度
MOTION_LAG_S = 1.0 / CSI_FS  # Eq. (4)：g(f) = rho(f, tau = 1/Fs) = 20 ms
MOTION_COHERENCE_WINDOW_S = 2.0  # 更长的窗可稳定相干性的估计
MOTION_ENERGY_RATIO = 1.60   # 残差能量 / 校准得到的背景电平
MOTION_COHERENCE_MIN = 0.115  # 跨子载波的第一奇异值占比
MOTION_RATIO_FLOOR = 1.25    # 灵敏度 = 1.0 时施加的下限

# --- breathing 通道 ---------------------------------------------------------
BREATH_WINDOW_S = 8.0
BREATH_LAG_MIN_S = 0.80      # 75 BPM，比新生儿呼吸率区间再留出余量
BREATH_LAG_MAX_S = 6.00      # 10 BPM
BREATH_MAX_OVERLAP = 0.75    # 重叠低于约 25 % 时无偏 ACF 会失效
BREATH_MIN_HEIGHT = 0.10
BREATH_MIN_PROMINENCE = 0.05
BREATH_CONF_THRESHOLD = 0.50

# --- 判决 ------------------------------------------------------------------
EVIDENCE_WINDOW_S = 5.0
PRESENCE_RATIO = 0.30
RESPONSE_BUDGET_S = 10.0     # CPD 总响应预算，从首个证据样本起算
                             # （而不是从确认时刻起算）
APNEA_TIMEOUT_S = 15.0
CLEAR_TIMEOUT_S = 25.0
CALIBRATION_S = 5.0
EVIDENCE_GAP_RESET_S = 5.0   # 新的一段证据会重新开始计算预算
MRC_WEIGHT_FLOOR = 0.02      # 防止零/负增益让某个子载波失效
INTERFERENCE_DEPTH = 0.35    # 用于抑制脉冲噪声的 V 形 ACF 深度


@dataclass
class DetectorConfig:
    motion_window_s: float = MOTION_WINDOW_S
    motion_lag_s: float = MOTION_LAG_S
    motion_coherence_window_s: float = MOTION_COHERENCE_WINDOW_S
    motion_energy_ratio: float = MOTION_ENERGY_RATIO
    motion_coherence_min: float = MOTION_COHERENCE_MIN
    breath_window_s: float = BREATH_WINDOW_S
    breath_lag_min_s: float = BREATH_LAG_MIN_S
    breath_lag_max_s: float = BREATH_LAG_MAX_S
    breath_max_overlap: float = BREATH_MAX_OVERLAP
    breath_min_height: float = BREATH_MIN_HEIGHT
    breath_min_prominence: float = BREATH_MIN_PROMINENCE
    breath_conf_threshold: float = BREATH_CONF_THRESHOLD
    evidence_window_s: float = EVIDENCE_WINDOW_S
    presence_ratio: float = PRESENCE_RATIO
    response_budget_s: float = RESPONSE_BUDGET_S
    apnea_timeout_s: float = APNEA_TIMEOUT_S
    clear_timeout_s: float = CLEAR_TIMEOUT_S
    calibration_s: float = CALIBRATION_S
    evidence_gap_reset_s: float = EVIDENCE_GAP_RESET_S
    sensitivity: float = 1.0

    def motion_ratio_floor(self) -> float:
        # sensitivity > 1 会提高灵敏度（即降低比值阈值）
        return max(MOTION_RATIO_FLOOR - 0.1, self.motion_energy_ratio / max(0.3, self.sensitivity))

    def to_dict(self) -> dict:
        return asdict(self)
