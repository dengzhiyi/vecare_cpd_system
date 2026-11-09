"""真实音频前端：探测信号设计、WAV 读写与 CSI 估计。

仿真器交给流水线的是在 24 个载波上采样的复 CFR。真实采集只能得到麦克风波形，
因此本模块弥合这一差距：

    probe -> (speaker, cabin, mic) -> waveform -> frame sync -> CIR
          -> DFT at the 24 carrier frequencies -> |H(f)| -> pipeline.step()

为什么数值能对得上：探测信号每 20 ms 重复一次，而 48 000 Hz / 50 Hz = 960 个
采样，因此一个 960 采样的探测周期恰好每秒产生 50 个 CSI 帧 —— 这正是 SAS 模型
所定义的帧率。默认探测信号是 Kasami 序列：把 255 位 Kasami 码扩频到 20 ms，
再搬移到载波频带中心。20 ms 内不是 2^n - 1 个采样（960 ≠ 1023），所以这里用
**码片扩频**而不是逐采样序列：Kasami 码的码率约 12.75 kHz，经过带限与载波
均衡后仍保持 50 Hz CSI 帧率。线性 chirp 与 MLS 模式保留用于对比和回归。

关于帧对齐的说明：由于探测信号是周期的，在任意偏移处截取一个 L 采样的窗，
得到的都是同一周期循环移位后的结果。移位 delta 个采样会使 H(f) 乘以
exp(-j 2 pi f delta / fs)，即只引入一个相位斜坡，而不影响 |H(f)| ——
后者是 SAS 模型唯一使用的量。因此帧同步只需好到能把窗保持在一个周期之内，
不要求采样级精确。
"""
from __future__ import annotations

import wave
from dataclasses import dataclass
from functools import lru_cache
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, find_peaks, resample_poly

from .config import (
    AUDIO_FS,
    CSI_FS,
    F_HIGH,
    F_LOW,
    N_SUBCARRIERS,
    PROBE_TAPER_S,
    SWEEP_HI,
    SWEEP_LO,
    AcousticBand,
    plan_band,
)

DEFAULT_PERIOD_S = 1.0 / CSI_FS


# ---------------------------------------------------------------------------
# 探测信号
# ---------------------------------------------------------------------------

_MLS_TAPS = {5: [5, 3], 6: [6, 5], 7: [7, 6], 8: [8, 6, 5, 4],
             9: [9, 5], 10: [10, 7], 11: [11, 9], 12: [12, 11, 10, 4],
             13: [13, 12, 11, 8], 14: [14, 13, 12, 2], 15: [15, 14],
             16: [16, 15, 13, 4]}

# 探测波形的载波均衡下限：最弱载波不能比最强载波低这么多。低于该值时，
# 谱除法基本是噪声除以噪声，该载波在运行时会被裁剪，白白少一个分集支路。
_PROBE_MIN_CONDITION_DB = -3.0


@dataclass(frozen=True)
class Probe:
    """（曾经或将要）播放的周期性探测信号。"""

    samples: np.ndarray
    fs: int
    kind: str
    band: AcousticBand | None = None

    @property
    def length(self) -> int:
        return int(self.samples.size)

    @property
    def period_s(self) -> float:
        return self.length / self.fs

    @property
    def csi_fs(self) -> float:
        """由该探测周期所隐含的 CSI 帧率。"""
        return self.fs / self.length

    @property
    def energy(self) -> float:
        return float(np.dot(self.samples, self.samples))


def _mls_sequence(order: int, taps: list[int]) -> np.ndarray:
    """生成长度 2^n-1 的 m 序列（0/1），Kasami 码的原料。"""
    length = 2 ** order - 1
    reg = np.ones(order, dtype=np.int8)
    out = np.empty(length, dtype=np.int8)
    for i in range(length):
        out[i] = reg[-1]
        fb = 0
        # 抽头以多项式的幂次给出（x^n + x^k + 1），而寄存器下标从反馈端
        # （reg[0]）开始计数，因此第 k 个抽头是 reg[k-1]，不是 reg[-k]。
        # 用 reg[-k] 会把反馈端和输出端混在一起，得到的最长序列周期远小于
        # 2^n-1，Kasami 的互相关上界也就不再成立。
        for t in taps:
            fb ^= reg[t - 1]
        reg[1:] = reg[:-1]
        reg[0] = fb
    return out


def _kasami_codes(order: int = 8) -> list[np.ndarray]:
    """构造 Kasami 小集合：a 本身，以及 a 与 b 的循环移位异或。

    n 为偶数时，用抽因子 ``q = 2^(n/2)+1`` 从 m 序列得到短周期 b，再与 a
    异或，得到 ``2^(n/2)`` 条互相关有界的序列。order=8 时长度为 255。
    基序列 a 必须包含在集合内，否则只有 2^(n/2)-1 条，工程上会白白少掉一条
    （同时也是 PAPR 最低的那条候选）。
    """
    if order % 2 or order not in _MLS_TAPS:
        raise ValueError(f"Kasami needs an even order in {sorted(_MLS_TAPS)}")
    a = _mls_sequence(order, _MLS_TAPS[order])
    n = a.size
    dec = a[(np.arange(n) * (2 ** (order // 2) + 1)) % n]
    period = next(
        p for p in range(1, n + 1)
        if n % p == 0 and np.all(dec == np.tile(dec[:p], n // p))
    )
    b = np.tile(dec[:period], n // period)
    return [a] + [a ^ np.roll(b, shift) for shift in range(period)]


def _bandlimit(samples: np.ndarray, fs: float, f_lo: float, f_hi: float) -> np.ndarray:
    """在频域做余弦边缘的带通，保留周期性、避免时域滤波的边沿瞬态。"""
    spectrum = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(samples.size, 1.0 / fs)
    transition = 200.0
    lo, hi = f_lo - transition, f_hi + transition
    mask = ((freqs >= lo) & (freqs <= hi)).astype(float)
    rise = (freqs > lo - transition) & (freqs < lo)
    fall = (freqs > hi) & (freqs < hi + transition)
    mask[rise] = 0.5 - 0.5 * np.cos(np.pi * (freqs[rise] - (lo - transition)) / transition)
    mask[fall] = 0.5 + 0.5 * np.cos(np.pi * (freqs[fall] - hi) / transition)
    return np.fft.irfft(spectrum * mask, samples.size)


@lru_cache(maxsize=16)
def _kasami_waveform(fs: int, length: int, f_lo: float, f_hi: float, n_carriers: int) -> np.ndarray:
    """把 Kasami 码扩频、搬移到载波频带，并均衡各载波的探测功率。"""
    codes = _kasami_codes(8)
    oversample = 4
    high_fs = fs * oversample
    high_len = length * oversample
    n = np.arange(high_len)
    fc = 0.5 * (f_lo + f_hi)
    candidates: list[tuple[float, float, np.ndarray]] = []

    for code in codes:
        chips = code[(n * code.size) // high_len].astype(np.float64) * 2.0 - 1.0
        s = chips * np.cos(2 * np.pi * fc * n / high_fs)
        s = _bandlimit(s, high_fs, f_lo, f_hi)
        # 带限后每 4 个采样取平均，等价于一个短低通 + 抽取，输出仍为 20 ms。
        s = s.reshape(length, oversample).mean(axis=1)

        # 在探测信号自身的 DFT 网格上把各载波拉平。Kasami 码的频谱起伏可达
        # 10 dB 以上；不均衡时最弱载波会退化成噪声计。
        spectrum = np.fft.rfft(s)
        spacing = fs / length
        bins = np.round(
            (f_lo + np.arange(n_carriers) * (f_hi - f_lo) / max(1, n_carriers - 1)) / spacing
        ).astype(int)
        bins = np.unique(bins[(bins > 0) & (bins < spectrum.size)])
        mag = np.abs(spectrum[bins])
        target = float(np.median(mag))
        spectrum[bins] *= np.clip(target / (mag + 1e-12), 0.25, 12.0)
        s = np.fft.irfft(spectrum, length)

        peak = float(np.max(np.abs(s)))
        if peak <= 0:
            continue
        s /= peak
        rms = float(np.sqrt(np.mean(s ** 2)))
        # 一条码的单载波深零点可以被均衡器的钳位放大到最弱载波，这是设计期
        # 缺陷（最弱载波退化成噪声计），与 PAPR 无关，必须先排除。
        levels = np.abs(np.fft.rfft(s)[bins]) ** 2
        worst_db = 10.0 * np.log10(float(np.min(levels)) / float(np.max(levels)))
        candidates.append((worst_db, rms, s))

    if not candidates:
        raise ValueError("could not build a Kasami probe for this band")
    # 先保住每个载波都有可用功率，再在合格者中选最“温和”（低 PAPR）的一条。
    good = [c for c in candidates if c[0] >= _PROBE_MIN_CONDITION_DB]
    pick = max(good, key=lambda c: c[1]) if good else max(candidates, key=lambda c: c[0])
    return pick[2]


def make_probe(
    fs: int = AUDIO_FS,
    period_s: float = DEFAULT_PERIOD_S,
    kind: str = "kasami",
    taper_s: float = PROBE_TAPER_S,
    hw_f_max: float | None = None,
) -> Probe:
    """构造周期性探测信号。

    ``kasami``（默认）把 255 位 Kasami 码扩频到 20 ms，再搬移到载波频带中心，
    并在 DFT 网格上均衡各载波功率。码片扩频让 960 采样的周期也能使用
    2^n-1 长度的 Kasami 码，同时保持 50 Hz CSI 帧率。``chirp`` 是带限线性
    扫频，``mls`` 是逐采样最大长度序列 —— 后两者保留用于对比与回归。

    探测频带必须覆盖载波并留有余量：若激励起止于载波频带，两端的载波会损失
    探测功率。Kasami 模式先带限到 ``f_lo-f_hi``，再在载波网格上拉平；因此
    带宽收缩时只有实际使用的载波被均衡，不会把能量浪费在驱动器上限之外。

    ``hw_f_max`` 是扬声器 + 麦克风实测的 -3 dB 上限。传入它，频带就会收缩以适配；
    不传则采用默认的 20 kHz 硬件上限。
    """
    band = plan_band(hw_f_max)
    length = int(round(fs * period_s))
    if length < 32:
        raise ValueError("probe period too short for this sample rate")

    if kind == "kasami":
        if band.f_hi + 400.0 >= fs / 2:
            raise ValueError(
                f"probe band reaches {band.f_hi:.0f} Hz but the Nyquist "
                f"limit at fs={fs} is {fs/2:.0f} Hz; record at 48 kHz or higher"
            )
        s = _kasami_waveform(
            int(fs), int(length), float(band.f_lo), float(band.f_hi), int(band.n_carriers)
        ).copy()
        return Probe(samples=s, fs=int(fs), kind=kind, band=band)

    if kind == "chirp":
        if band.sweep_hi >= fs / 2:
            raise ValueError(
                f"probe sweep reaches {band.sweep_hi:.0f} Hz but the Nyquist "
                f"limit at fs={fs} is {fs/2:.0f} Hz; record at 48 kHz or higher"
            )
        n = np.arange(length)
        u = n / max(1, length - 1)
        inst_f = band.sweep_lo + (band.sweep_hi - band.sweep_lo) * u
        s = np.sin(2 * np.pi * np.cumsum(inst_f) / fs)
    elif kind == "mls":
        order = int(round(np.log2(length + 1)))
        if 2 ** order - 1 != length or order not in _MLS_TAPS:
            raise ValueError(
                f"MLS probe needs a period of 2^n-1 samples; {length} is not. "
                f"Use a Kasami probe, or pick period_s = {(2 ** 10 - 1) / fs:.6f} "
                f"(MLS order 10) and account for the resulting "
                f"{fs / (2 ** 10 - 1):.3f} Hz CSI rate."
            )
        s = _mls_sequence(order, _MLS_TAPS[order]).astype(np.float64) * 2.0 - 1.0
    else:
        raise ValueError(f"unknown probe kind: {kind!r}")

    n_taper = max(1, int(round(taper_s * fs)))
    if n_taper * 2 < length:
        w = np.ones(length)
        edge = 0.5 - 0.5 * np.cos(np.pi * np.arange(n_taper) / n_taper)
        w[:n_taper] = edge
        w[-n_taper:] = edge[::-1]
        s = s * w

    peak = float(np.max(np.abs(s)))
    if peak > 0:
        s = s / peak
    return Probe(samples=s.astype(np.float64), fs=int(fs), kind=kind, band=band)


# ---------------------------------------------------------------------------
# WAV 读写（仅使用标准库）
# ---------------------------------------------------------------------------

def read_wav(
    path: str | Path,
    target_fs: int | None = None,
    max_seconds: float | None = None,
) -> tuple[np.ndarray, int]:
    """把 WAV 文件读成 [-1, 1] 范围内的单声道 float64。

    支持 8/16/24/32 位 PCM。浮点 WAV 会被拒绝，并给出转换提示，因为标准库
    ``wave`` 模块无法解析它们。

    ``max_seconds`` 只读取文件开头的一段，用于分析超长录音时限制耗时；
    默认 ``None`` 读取完整文件。
    """
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as w:
            fs = w.getframerate()
            nch = w.getnchannels()
            sw = w.getsampwidth()
            n_frames = w.getnframes()
            if max_seconds is not None and max_seconds > 0:
                n_frames = min(n_frames, int(round(float(max_seconds) * fs)))
            raw = w.readframes(n_frames)
    except wave.Error as exc:
        raise RuntimeError(
            f"{path.name}: unsupported WAV encoding ({exc}). "
            "Convert to 16-bit PCM first, e.g. "
            f"ffmpeg -i \"{path.name}\" -ac 1 -ar {AUDIO_FS} -c:a pcm_s16le out.wav"
        ) from exc

    if sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sw == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        ints = np.where(ints >= 1 << 23, ints - (1 << 24), ints)
        data = ints.astype(np.float64) / float(1 << 23)
    elif sw == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / float(1 << 31)
    else:
        raise RuntimeError(f"{path.name}: unsupported sample width {sw} bytes")

    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)

    if target_fs is not None and target_fs != fs:
        g = gcd(int(target_fs), int(fs))
        data = resample_poly(data, target_fs // g, fs // g)
        fs = int(target_fs)
    return np.ascontiguousarray(data), int(fs)


def write_wav(path: str | Path, samples: np.ndarray, fs: int = AUDIO_FS) -> Path:
    """写入单声道 16 位 PCM，并裁剪到 [-1, 1]。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(fs))
        w.writeframes(pcm.tobytes())
    return path


# ---------------------------------------------------------------------------
# 帧同步
# ---------------------------------------------------------------------------

def matched_filter(x: np.ndarray, probe: Probe) -> np.ndarray:
    """探测信号相关性的包络，每个可能的帧起始位置对应一个值。"""
    return np.abs(fftconvolve(x, probe.samples[::-1], mode="valid"))


def match_quality(mf: np.ndarray) -> float:
    """探测检测统计量：匹配滤波峰值与稳健底噪之比。

    Kasami 是宽带编码信号，其匹配滤波底噪不是高斯的；用全局 RMS 会被周期性
    峰本身抬高，噪声录音也能拿到 4–5。这里改用中位数绝对偏差（MAD）作为
    稳健底噪：含探测信号的录音约为 12–18，纯噪声约为 8，门限可稳定放在 9。
    """
    if mf.size == 0:
        return 0.0
    med = float(np.median(mf))
    sigma = float(np.median(np.abs(mf - med))) * 1.4826
    return float(np.max(mf)) / sigma if sigma > 0 else 0.0


def frame_starts(x: np.ndarray, probe: Probe, min_ratio: float = 0.2) -> np.ndarray:
    """通过匹配滤波定位探测信号的重复周期。"""
    mf = matched_filter(x, probe)
    if mf.size == 0:
        return np.zeros(0, dtype=int)
    height = min_ratio * float(np.max(mf))
    peaks, _ = find_peaks(mf, height=height, distance=max(1, int(0.8 * probe.length)))
    return peaks.astype(int)


def frame_offsets(
    x: np.ndarray,
    probe: Probe,
    min_ratio: float = 0.2,
    max_fill: int = 4,
) -> tuple[np.ndarray, dict]:
    """帧起始采样位置，并插值补齐单个缺失的周期。

    周期性的探测信号意味着每一帧都可以独立截取，因此错过一个匹配滤波峰只是一处
    空隙。小的空隙按实测周期补齐（这同时吸收了播放/麦克风时钟漂移，因为锚点
    始终落在检测到的峰上）；大的空隙则原样保留并上报，因为对幅度序列做时间拉伸
    会破坏呼吸 lag 的估计。

    检测到的峰还必须落在帧栅格上。真实录音会在两个真实帧之间出现伪峰
    （实测间隔 609/768 采样，约为 0.6–0.8 个周期），这种帧并不对应完整探测
    周期，参与解调只会让 CSI 时间轴错位；因此偏离栅格超过 ``tol`` 且不足
    1.5 个周期的峰直接丢弃，并计入 ``dropped``，而不是悄悄当成真实帧。
    """
    mf = matched_filter(x, probe)
    found = frame_starts(x, probe, min_ratio)
    stats = {
        "match_ratio": round(match_quality(mf), 2),
        "detected_periods": int(found.size),
        "nominal_period": probe.length,
        "nominal_csi_fs": probe.csi_fs,
        "filled": 0,
        "holes": 0,
        "dropped": 0,
        "irregular": 0,
        "period_samples": probe.length,
        "csi_fs": probe.csi_fs,
        "coverage": 0.0,
        "jitter_ms": 0.0,
        "max_deviation_ms": 0.0,
        "longest_gap_s": 0.0,
        "period_std_samples": 0.0,
        "strict_ratio": 0.0,
    }
    if found.size < 2:
        return found, stats

    d = np.diff(found)
    # 周期必须围绕探测信号的标称周期估计。若用检测间隔的中位数，伪峰会把
    # 它拉偏（实测 609/768 混入后半段中位数变成 613），随后真实帧反而被当成
    # 伪峰。12% 的窗口只允许时钟漂移，不允许差一个周期的误检。
    tol = 0.12
    nominal = float(probe.length)
    near = d[np.abs(d - nominal) <= tol * nominal]
    period = int(round(float(np.median(near)) if near.size else nominal))
    period = max(1, period)
    stats["period_samples"] = period
    stats["csi_fs"] = probe.fs / period
    # 只用中位数/取整会丢掉“帧间隔到底有多稳”这一信息：真实手机录音的
    # 播放与采集时钟不同源，间隔会在 959/960/961 之间跳动，而中位数仍
    # 恰好等于 960。质量门必须看到分布本身。
    if near.size:
        stats["period_std_samples"] = round(float(np.std(near)), 3)
        stats["strict_ratio"] = round(float(np.mean(near == nominal)), 4)

    starts = [int(found[0])]
    last = int(found[0])
    filled = 0
    holes = 0
    dropped = 0
    irregular = 0
    for b in found[1:]:
        b = int(b)
        gap = b - last
        n = int(round(gap / period))
        err = abs(gap - n * period) / period if n else 1.0
        if n <= 0:
            continue
        if err > tol:
            irregular += 1
            stats["max_deviation_ms"] = max(
                stats["max_deviation_ms"],
                round(abs(gap - n * period) / probe.fs * 1000.0, 2),
            )
            if gap < 1.5 * period:
                # 栅格之间的伪峰：不对应完整探测周期，丢弃。
                dropped += 1
                continue
            # 长间隙且偏离栅格，无法安全插值；保留端点并按空洞上报。
            holes += 1
            stats["longest_gap_s"] = max(stats["longest_gap_s"], gap / probe.fs)
        elif 2 <= n <= max_fill:
            for k in range(1, n):
                starts.append(int(round(last + k * period)))
            filled += n - 1
        elif n > max_fill:
            holes += 1
            stats["longest_gap_s"] = max(stats["longest_gap_s"], gap / probe.fs)
        starts.append(b)
        last = b

    starts = np.unique(np.asarray(starts, dtype=int))
    stats["filled"] = filled
    stats["holes"] = holes
    stats["dropped"] = dropped
    stats["irregular"] = irregular
    stats["jitter_ms"] = round(float(np.std(near - np.median(near)) / probe.fs * 1000.0), 3) \
        if near.size else 0.0
    span = max(1.0, (float(found[-1]) - float(found[0])) / period + 1.0)
    stats["coverage"] = float(min(1.0, starts.size / span))
    return starts, stats


# ---------------------------------------------------------------------------
# CSI 估计
# ---------------------------------------------------------------------------

def carrier_bins(
    probe: Probe,
    f_lo: float | None = None,
    f_hi: float | None = None,
    n: int | None = None,
) -> np.ndarray:
    """吸附到探测信号自身 DFT 网格上的载波频率。

    真实信道探测器只能在其子载波网格上测量 H。当网格间距为 50 Hz、而多径干涉
    条纹间隔约 150 Hz 时，直接要求 10 500 Hz 意味着要对一个在采样点之间振荡的
    量做插值 —— 即使零噪声也会带来 10 % 的幅度误差。把载波吸附到网格频点上，
    估计才是精确的。

    默认值来自 ``probe.band``，即来自构造该探测信号时所用的硬件上限，因此
    16 kHz 的驱动器得到 4 kHz 的跨度，而不会去索要它无法重现的载波。
    """
    band = probe.band or plan_band(None)
    f_lo = band.f_lo if f_lo is None else f_lo
    f_hi = band.f_hi if f_hi is None else f_hi
    n = band.n_carriers if n is None else n

    df = probe.fs / probe.length
    k0 = int(round(f_lo / df))
    k1 = int(round(f_hi / df))
    ks = np.unique(np.round(np.linspace(k0, k1, n)).astype(int))
    ks = ks[(ks > 0) & (ks < probe.length // 2)]
    if ks.size < 4:
        raise ValueError(
            f"carrier band {f_lo}-{f_hi} Hz holds only {ks.size} grid points at "
            f"{df:.1f} Hz spacing; lengthen the probe period"
        )
    return ks * df


def carrier_health(
    amplitudes: np.ndarray,
    rel_floor: float = 3.0,
    abs_floor: float = 0.15,
    baseline_percentile: float = 25.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """依据安静窗判断哪些载波实际承载了信号。

    扬声器到达不了、或麦克风听不到的载波，其频点上的探测功率几乎为零。此时谱
    除法是用噪声除以这个近似零值，恢复出的 ``|H|`` 就变成一段宽带随机序列 ——
    实测相对抖动为 0.20-0.48，而健康载波只有 0.013-0.029。这段序列不是叠加在
    信号上的小幅误差，它*就是*该频率上的信号，于是残差能量会看到运动，MRC 权重
    会看到相关性。

    基线采用*低*分位数而非中位数，这一点很关键：一旦频带有一半已经死掉，中位数
    就已经移动到死载波之中，``3 x median`` 会位于它们之上，什么都裁剪不掉。
    在 18 kHz 上限下实测，中位数落在 0.11，而第 25 百分位数仍为 0.028；加上绝对
    下限后门限为 ``max(0.15, 3 x 0.028) = 0.15``，仍能裁剪掉三个位于滚降区的载波。

    该检验刻意采用相对判据：绝对抖动会在车厢嘈杂时一次性把所有载波都标记出来，
    而一个死载波则始终比健康载波高出一个数量级。``abs_floor`` 用于防止所有载波
    都死掉的情况；如果存活者不足一半，函数会保留全部载波，让调用方明确地失败，
    而不是仅凭三个载波继续运行。
    """
    A = np.asarray(amplitudes, dtype=float)
    if A.ndim != 2 or A.shape[0] < 8:
        n = A.shape[-1] if A.ndim == 2 else 0
        return np.ones(n, dtype=bool), np.zeros(n), 0.0
    mu = np.abs(A).mean(axis=0)
    sd = A.std(axis=0)
    rel = sd / np.maximum(mu, 1e-12)
    baseline = float(np.percentile(rel, baseline_percentile))
    threshold = max(abs_floor, rel_floor * baseline)
    alive = rel <= threshold
    if alive.sum() < max(4, A.shape[1] // 2):
        alive[:] = True
    return alive, rel, threshold


def probe_condition_db(probe: Probe, carriers: np.ndarray) -> np.ndarray:
    """每个载波上的探测功率，以 dB 表示，并相对于最强载波。

    可以把它理解为前端付出的噪声代价：位于 -20 dB 的载波，其估计噪声是 0 dB
    载波的 10 倍。这是一个*设计期*检查，而不是运行时检查 —— 该数值由探测波形
    固定决定。
    """
    spectrum = np.fft.fft(probe.samples)
    power = np.abs(spectrum) ** 2
    pos = (np.asarray(carriers, dtype=float) * probe.length / probe.fs) % probe.length
    bins = np.round(pos).astype(int) % probe.length
    p = power[bins]
    peak = float(p.max()) or 1.0
    return 10.0 * np.log10(np.maximum(p, peak * 1e-12) / peak)


def estimate_csi(
    x: np.ndarray,
    probe: Probe,
    carriers: np.ndarray | None = None,
    starts: np.ndarray | None = None,
    rel_floor: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """估计每一帧在 ``carriers`` 上的复 CFR。

    探测信号以周期 L 重复，且比任何座舱路径都长，因此录音实际上是探测信号与
    座舱 CIR 的*循环*卷积：在 L 点 DFT 网格上，``R[k] = S[k] * H[k]``。
    因此，正则化谱除法可以在探测信号激励的每个频点上精确恢复 H。

    时域相关估计 —— ``ifft(fft(block) * conj(fft(probe)))`` —— 并*不*等价，
    那正是早先的 bug：它的 CIR 会被探测信号自相关的主瓣抹开（13 kHz 带宽下约
    4 个采样，即 83 us 的矩形窗），而 sinc(83 us x 23.5 kHz) 已越过第一个零点，
    因此顶部载波的结果会偏低几十个百分点。

    如果 ``carriers`` 不在网格上，则改为从估计出的 CIR 求 H（带限插值），
    对稀疏且短的座舱 CIR 而言这是精确的，但明显不如直接挑选网格频率。

    返回 ``(cfr, starts)``，其中 ``cfr`` 的形状为 ``(n_frames, n_carriers)``。
    """
    L, fs = probe.length, probe.fs
    carriers = carrier_bins(probe) if carriers is None else np.asarray(carriers, dtype=float)
    if starts is None:
        starts, _ = frame_offsets(x, probe)

    spectrum = np.fft.fft(probe.samples)
    power = np.abs(spectrum) ** 2
    inverse = np.conj(spectrum) / (power + rel_floor * float(power.max()))

    pos = carriers * L / fs
    on_grid = bool(np.all(np.abs(pos - np.round(pos)) < 1e-6))
    if on_grid:
        pick = np.round(pos).astype(int) % L
        kernel = None
    else:
        pick = None
        kernel = np.exp(-1j * 2 * np.pi * np.outer(carriers, np.arange(L)) / fs)

    cfr = np.empty((len(starts), len(carriers)), dtype=np.complex128)
    used = 0
    for start in starts:
        block = x[start:start + L]
        if block.size < L:
            break
        hk = np.fft.fft(block) * inverse
        cfr[used] = hk[pick] if kernel is None else kernel @ np.fft.ifft(hk)
        used += 1
    return cfr[:used], np.asarray(starts)[:used]


def csi_amplitude(x: np.ndarray, probe: Probe, carriers: np.ndarray | None = None):
    """便捷封装，返回 ``(|CFR|, frame_times, cfr)``。"""
    cfr, starts = estimate_csi(x, probe, carriers)
    times = starts / float(probe.fs)
    return np.abs(cfr), times, cfr
