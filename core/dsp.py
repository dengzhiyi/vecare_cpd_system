"""统计声学感知（SAS）基础原语。

本模块实现 VeCare 的信号处理核心：

* CSI 幅度去除均值/线性趋势后的逐子载波 ACF，
* motion 通道：去趋势残差能量 + 子载波间相干性，
  以及论文中的 g(f) = rho(f, 1/Fs) 增益向量（Eq. 4），
* 以 g(f) 为权重对各子载波 ACF 做最大比合并（MRC）（Eq. 6），
* 在 10-66 BPM 范围内基于突出度/高度/宽度的呼吸峰选择，
* 论文 §5 的脉冲噪声（“先线性下降再上升”）ACF 模式检测器。
"""
from __future__ import annotations

import numpy as np
from scipy.fft import next_fast_len, rfft, irfft
from scipy.signal import find_peaks

from .config import (
    DetectorConfig,
    INTERFERENCE_DEPTH,
    LONG_INTERFERENCE_DEPTH,
    LONG_INTERFERENCE_LAG_S,
    LONG_INTERFERENCE_MIN_LAG_S,
    LONG_INTERFERENCE_RECOVERY,
    MRC_WEIGHT_FLOOR,
)


def _detrend(X: np.ndarray) -> np.ndarray:
    """沿轴 0（时间方向）去除均值和线性趋势。"""
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    n = X.shape[0]
    if n > 2:
        tt = np.arange(n, dtype=np.float64)
        tt -= tt.mean()
        denom = float(np.dot(tt, tt))
        if denom > 0:
            slope = np.einsum("ij,i->j", X, tt) / denom
            X = X - slope[None, :] * tt[:, None]
    return X


def autocorr(X: np.ndarray, max_lag: int) -> np.ndarray:
    """计算 X 每一列的无偏样本 ACF。

    返回形状为 (max_lag + 1, n_columns)；对活跃列而言 lag 0 处的值等于 1。
    """
    X = _detrend(X)
    n = X.shape[0]
    if n < 3:
        out = np.zeros((max_lag + 1, X.shape[1]))
        out[0] = 1.0
        return out
    nfft = next_fast_len(2 * n)
    F = rfft(X, nfft, axis=0)
    power = irfft(F * np.conj(F), nfft, axis=0)[:max_lag + 1]
    denom = np.array([(n - lag) for lag in range(max_lag + 1)], dtype=np.float64)
    power = power / denom[:, None]
    norm = power[0]
    norm = np.where(norm > 1e-12, norm, 1.0)
    return power / norm[None, :]


def motion_statistic(acf_motion: np.ndarray, lag_samples: int) -> np.ndarray:
    """g(f)：每个子载波第一个可用的 ACF 抽头（Eq. 4）。"""
    lag = int(np.clip(lag_samples, 1, acf_motion.shape[0] - 1))
    return acf_motion[lag, :]


# ---------------------------------------------------------------------------
# motion 通道
# ---------------------------------------------------------------------------
def residual_energy(X: np.ndarray) -> float:
    """短 CSI 窗去趋势后的动态能量。

    在 1 s 内，呼吸引起的起伏几乎是一条直线，因此去趋势会去掉其中大部分；
    而肢体运动（在窗内形成一个鼓包）会留下较大的残差。
    """
    R = _detrend(X)
    if R.size == 0:
        return 0.0
    return float(np.mean(R.std(axis=0)))


def subcarrier_coherence(X: np.ndarray) -> float:
    """由第一奇异向量解释的残差方差占比。

    人体会让每条多径产生相同的时延位移，因此各子载波的残差保持相关
    （kappa 明显高于噪声底）。环境噪声突发则会独立地作用于各子载波，保持在噪声底附近。
    """
    R = _detrend(X)
    if R.shape[0] < 4 or R.shape[1] < 2:
        return 0.0
    R = R - R.mean(axis=0, keepdims=True)
    try:
        s = np.linalg.svd(R, compute_uv=False)
    except np.linalg.LinAlgError:
        return 0.0
    total = float((s ** 2).sum())
    if total < 1e-12:
        return 0.0
    return float(min(1.0, s[0] ** 2 / total))


def interference_pattern(acf_mean: np.ndarray, depth: float = INTERFERENCE_DEPTH) -> bool:
    """§5 中的脉冲噪声特征：ACF 先下降再上升。

    喇叭鸣笛或关门会注入一段短促的突发；其 ACF 呈三角形，即单调下降到谷底，
    再单调回升。呼吸和平滑的身体运动不会在前约 0.5 s 的 lag 范围内产生这种 V 形。
    """
    n = min(len(acf_mean), 26)
    if n < 8:
        return False
    seg = np.asarray(acf_mean[:n], dtype=np.float64)
    k = int(np.argmin(seg[1:-1])) + 1
    if k < 2 or k > n - 3:
        return False
    left, right = seg[1:k + 1], seg[k:n]
    if left.size < 2 or right.size < 2:
        return False
    if not np.all(np.diff(left) <= 0.02) or not np.all(np.diff(right) >= -0.02):
        return False
    return float(min(seg[0], seg[-1]) - seg[k]) > depth


def interference_pattern_long(acf: np.ndarray, fs: int) -> bool:
    """§5 的长时版本：0.5–6 s 内出现绝对为负的宽平台，且区间内没有呼吸正峰。

    短 V 检测只看前 0.5 s，4 s 的噪声突发在 1 s 运动窗里根本看不到谷底。
    这里改用 8 s 呼吸 ACF：持续噪声抬高了整窗均值，使窗外样本整体变为负值，
    于是 ACF 在数秒的 lag 上留下一个 *绝对* 为负的平台；真实呼吸一定会在 tau
    处回升到接近 1 的正峰。

    判据用绝对高度而不是“从 1 掉了多少”：安静车厢的 ACF 与 δ 函数无异，
    任何噪声小凹都满足“下降 0.9”，过去因此把空车误标成长时干扰。现在要求
    谷底自身低于 ``-LONG_INTERFERENCE_DEPTH``，并且 0.5–6 s 内再没有呼吸量级
    的正峰（``LONG_INTERFERENCE_RECOVERY``；18 BPM 的首峰约为 0.5–0.8，
    4 s 噪声的残余起伏不超过 0.25）。

    该标志仍需由检测器用能量比值门控，以免把缓慢的真实体动误判成干扰。
    """
    n = min(len(acf), int(round(LONG_INTERFERENCE_LAG_S * fs)) + 1)
    start = int(round(LONG_INTERFERENCE_MIN_LAG_S * fs))
    if n <= start + 2:
        return False
    seg = np.asarray(acf[:n], dtype=np.float64)
    k = int(np.argmin(seg[start:n])) + start
    if float(seg[k]) > -LONG_INTERFERENCE_DEPTH:
        return False
    # 呼吸的正峰可能出现在谷底之前（例如 18 BPM 的首峰在 3.3 s，而更深的下凹
    # 在 5 s），因此要看整个 0.5–6 s，而不只是谷底之后的尾巴。
    return float(np.max(seg[start:])) < LONG_INTERFERENCE_RECOVERY


def motion_features(
    X: np.ndarray,
    fs: int,
    lag_s: float,
    context: np.ndarray | None = None,
) -> dict:
    """motion 通道特征。

    X 携带用于能量计算和脉冲模式检验的短（响应快）窗。``context`` 可选地提供
    一个更长的窗，仅用于相干性估计，因为相干性是二阶统计量，在 2 s 上比在 1 s 上
    稳定得多。
    """
    lag = max(1, int(round(lag_s * fs)))
    max_lag = max(lag + 2, int(round(0.5 * X.shape[0])))
    acf = autocorr(X, max_lag)
    gains = motion_statistic(acf, lag)
    acf_mean = acf.mean(axis=1)
    coh_win = X if context is None else context
    return {
        "energy": residual_energy(X),
        "coherence": subcarrier_coherence(coh_win),
        "gains": gains,
        "g_mean": float(np.mean(gains)),
        "interference": interference_pattern(acf_mean),
    }


# ---------------------------------------------------------------------------
# breathing 通道
# ---------------------------------------------------------------------------
def combine_acf(acf: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """对子载波 ACF 做 MRC 合并，Eq. (6)。"""
    w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0) + MRC_WEIGHT_FLOOR
    total = w.sum()
    if total < 1e-9:
        w = np.ones_like(w) / len(w)
    else:
        w = w / total
    return acf @ w


def _parabolic_peak(x: np.ndarray, idx: int) -> float:
    """用三点抛物线拟合求亚采样精度的峰值位置。"""
    if idx <= 0 or idx >= len(x) - 1:
        return float(idx)
    y0, y1, y2 = x[idx - 1], x[idx], x[idx + 1]
    denom = y0 - 2 * y1 + y2
    if abs(denom) < 1e-12:
        return float(idx)
    return float(idx + 0.5 * (y0 - y2) / denom)


def _descend_to_fundamental(start: int, idx: np.ndarray, heights: np.ndarray, depth: int = 4) -> int:
    """从给定峰出发选取基频峰，而不是它的谐波。

    周期信号会在 tau_b, 2*tau_b, 3*tau_b 处出现高度几乎相等的峰，因此
    “最强的峰”可能是其中任意一个，而“最早的峰”可能是与呼吸无关的缓慢摆动。
    换一种做法：从已经选中的峰向下查找，如果在 lag/2 附近存在一个强度相当的峰，
    就说明当前峰是谐波。只能向下走 —— 从最强峰出发会反向把 40 BPM 的首峰
    误判成它的三次谐波（13 BPM）。
    """
    best = int(start)
    for _ in range(depth):
        half = float(idx[best]) / 2.0
        if half < idx[0]:
            break
        near = [i for i in range(len(idx)) if abs(idx[i] - half) <= 0.18 * half]
        if not near:
            break
        cand = max(near, key=lambda i: heights[i])
        if heights[cand] < 0.55 * heights[best]:
            break
        best = cand
    return best


_INVALID_PEAK = {"valid": False, "bpm": None, "confidence": 0.0,
                 "lag_s": None, "height": 0.0, "prominence": 0.0}


def find_breathing_peak(
    combined: np.ndarray,
    fs: int,
    cfg: DetectorConfig,
    window_samples: int | None = None,
) -> dict:
    """从合并后的 ACF 中选出一个可信的呼吸峰。

    峰只在 10-66 BPM 的 lag 区间内*搜索*，但突出度和高度是在从 lag 1 开始的
    完整 ACF 上度量的。如果一开始就在 lag_min 处对数组切片，会截断第一个峰的
    左侧谷底，使它的突出度只有谐波的一半，从而在无声无息中把报告的呼吸率翻倍。
    """
    lag_min = int(round(cfg.breath_lag_min_s * fs))
    nwin = window_samples if window_samples is not None else len(combined)
    lag_max = min(
        int(round(cfg.breath_lag_max_s * fs)),
        int(cfg.breath_max_overlap * nwin),
        len(combined) - 2,
    )
    if lag_max <= lag_min + 2:
        return dict(_INVALID_PEAK)

    search = combined[1:lag_max + 2]
    peaks, props = find_peaks(
        search,
        height=cfg.breath_min_height,
        prominence=cfg.breath_min_prominence,
        distance=max(2, int(0.4 * fs)),
    )
    n_found = len(peaks)
    idx = peaks + 1
    keep_range = (idx >= lag_min) & (idx <= lag_max)
    idx = idx[keep_range]
    heights = props["peak_heights"][keep_range]
    proms = props["prominences"][keep_range]
    widths_all = props.get("widths")
    if widths_all is None or len(widths_all) != n_found:
        widths_all = np.zeros(n_found)
    widths = widths_all[keep_range]

    if len(idx) == 0:
        seg = combined[lag_min:lag_max + 1]
        return {"valid": False, "bpm": None, "confidence": 0.0, "lag_s": None,
                "height": float(np.max(seg)) if len(seg) else 0.0,
                "prominence": 0.0}

    best = int(np.argmax(heights * (0.5 + proms)))
    # 优先选择最早且强度接近最强峰的可信（基频）峰；呼吸的谐波出现在
    # 2 倍/3 倍周期处。
    # 周期信号会在 tau_b, 2*tau_b, 3*tau_b 处出现高度几乎相等的峰，基频就是
    # 最早的那个。这里绝不能把突出度当作相对门限：内部谐波总是跨越完整的谷底，
    # 而第一个峰只跨越半个谷底。
    keep = [i for i in range(len(idx)) if heights[i] >= heights[best] * 0.85]
    chosen = keep[0] if keep else best

    # 若选中的其实是一个谐波，向下找到它的基频。过去这个函数存在但从未接入，
    # 导致“最早且强度接近”的启发式是唯一的基频保护。
    fund = _descend_to_fundamental(chosen, idx, heights)
    if fund != chosen and heights[fund] >= 0.55 * heights[chosen]:
        chosen = fund

    fine = _parabolic_peak(combined, int(idx[chosen]))
    lag_s = fine / fs
    bpm = 60.0 / lag_s if lag_s > 0 else 0.0
    height = float(heights[chosen])
    prominence = float(proms[chosen])

    conf = 0.55 * min(1.0, height / 0.35) + 0.45 * min(1.0, prominence / 0.25)
    # 真实的呼吸周期分布在相邻的几个 lag 上，而不是单个带噪的尖峰，
    # 因此给较宽的峰一点小幅加成。
    conf += 0.06 * min(1.0, float(widths[chosen]) / (0.5 * fs))
    conf = float(np.clip(conf, 0.0, 1.0))

    valid = bool(10.0 <= bpm <= 75.0)
    if valid and not _peak_is_periodic(combined, idx, heights, chosen, lag_max, cfg):
        # 单个孤立旁瓣也能满足高度/突出度，但它不是呼吸：要么前面没有足够深
        # 的谷底，要么 2*tau 处没有第二个峰。
        valid = False
    return {
        "valid": valid,
        "bpm": float(bpm) if valid else None,
        "confidence": conf,
        "lag_s": float(lag_s),
        "height": height,
        "prominence": prominence,
        "peak_idx": int(idx[chosen]),
    }


def _peak_is_periodic(
    combined: np.ndarray,
    peak_lags: np.ndarray,
    peak_heights: np.ndarray,
    chosen: int,
    lag_max: int,
    cfg: DetectorConfig,
) -> bool:
    """一个可信的呼吸峰必须同时满足谷底深度和（可观测时的）二次谐波。"""
    lag = int(peak_lags[chosen])
    height = float(peak_heights[chosen])
    trough = float(np.min(combined[:lag + 1]))
    if height - trough < cfg.breath_min_contrast:
        return False
    two_lag = 2 * lag
    tolerance = cfg.breath_harmonic_tolerance * two_lag
    if two_lag + tolerance > lag_max:
        # 8 s 窗看不到第二个周期（例如 10–20 BPM，或 2*tau 恰好落在边界上），
        # 此时只能依赖谷底深度。
        return True
    near = np.abs(peak_lags - two_lag) <= tolerance
    if not np.any(near):
        return False
    return bool(np.max(peak_heights[near]) >= cfg.breath_harmonic_min_ratio * height)


def compute_features(
    amp_motion: np.ndarray,
    amp_breath: np.ndarray,
    cfg: DetectorConfig,
    fs: int = 50,
) -> dict:
    """由滑动 CSI 幅度窗计算 SAS 特征。

    amp_motion：motion 窗（例如 1 s）的 (N_m, S) 幅度。
    amp_breath：breathing 窗（例如 8 s）的 (N_b, S) 幅度；其尾部同时充当
    相干性上下文窗。
    """
    coh_n = int(round(cfg.motion_coherence_window_s * fs))
    context = amp_breath[-coh_n:] if amp_breath.shape[0] >= coh_n else amp_breath
    motion = motion_features(amp_motion, fs, cfg.motion_lag_s, context)

    ready_n = int(0.94 * cfg.breath_window_s * fs)
    if amp_breath.shape[0] < ready_n:
        return {
            "gains": motion["gains"].astype(float).tolist(),
            "energy": motion["energy"],
            "coherence": motion["coherence"],
            "interference": motion["interference"],
            "interference_long": False,
            "peak": dict(_INVALID_PEAK),
            "acf_display": [],
            "acf_step_s": 0.04,
            "breath_ready": False,
        }

    max_lag_b = min(
        int(round(cfg.breath_lag_max_s * fs)),
        int(round(cfg.breath_window_s * fs)) - 2,
    )
    acf_b = autocorr(amp_breath, max_lag_b)
    # MRC 权重是在呼吸窗上由各子载波自身增益得到的（Eq. 4 作用于原始幅度），
    # 即每个子载波能看到多少目标能量。
    weights = np.maximum(motion_statistic(acf_b, max(1, int(round(cfg.motion_lag_s * fs)))), 0.0)
    combined = combine_acf(acf_b, weights)

    peak = find_breathing_peak(combined, fs, cfg, amp_breath.shape[0])
    interference_long = interference_pattern_long(combined, fs)

    step = max(1, int(round(0.04 * fs)))
    acf_display = combined[::step].astype(float).tolist()

    return {
        "gains": weights.astype(float).tolist(),
        "energy": motion["energy"],
        "coherence": motion["coherence"],
        "interference": motion["interference"],
        "interference_long": interference_long,
        "peak": peak,
        "acf_display": acf_display,
        "acf_step_s": step / fs,
        "breath_ready": True,
    }
