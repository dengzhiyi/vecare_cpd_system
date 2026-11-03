"""实时引擎与离线评估共用的 SAS 特征流水线。

流水线维护一个滑动的 CSI 幅度缓冲区，每 100 ms（即 10 Hz 判决）在短窗（1 s）上
计算 motion 通道、在长窗（8 s）上计算 breathing 通道，然后两者都送入 CPD 状态机。

``fs`` 是输入的 CSI 帧率。默认为仿真器产生的 50 Hz；真实采集时会将其设为
探测周期所隐含的帧率（默认 chirp 下为 48 000 / 960 采样 = 50 Hz）。

载波裁剪
--------
依据硬件上限规划频带（``plan_band``）是设计期的修正手段，但驱动器的 -3 dB
点很少与数据手册声称的一致，而且滚降并不是唯一会“杀死”载波的因素 ——
麦克风网罩、手机摆放位置或座舱吸声材料都可能削弱某个频带的一部分，
在一辆车上出现而在另一辆车上不出现。因此流水线还在运行时做检查：在安静的
校准窗内测量每个载波的相对抖动，并剔除那些比同伴高出一个数量级的载波
（``carrier_health``）。

裁剪只会从 DSP 输入中移除列。写入每一帧的 ``csi`` 字段仍保留全部载波，
因为死掉的载波本身就是真实信息 —— 它告诉操作员的是硬件而非算法出了问题。
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .audio import carrier_health
from .config import CSI_FS, DECISION_FS, DetectorConfig
from .detector import CPDDetector
from .dsp import compute_features


class SASPipeline:
    def __init__(
        self,
        cfg: DetectorConfig | None = None,
        fs: float = CSI_FS,
        auto_prune: bool = True,
    ):
        self.cfg = cfg or DetectorConfig()
        self.fs = float(fs)
        self.detector = CPDDetector(self.cfg)
        self.hist: deque[np.ndarray] = deque(
            maxlen=int(round(self.cfg.breath_window_s * self.fs))
        )
        self.frame_count = 0
        self.seq = 0
        self.decision_every = max(1, int(round(self.fs / DECISION_FS)))

        # --- 运行时载波裁剪 ---------------------------------------------
        self.auto_prune = auto_prune
        self.n_carriers = None            # 由第一帧固定下来
        self.carrier_mask: np.ndarray | None = None
        self.carrier_jitter: np.ndarray | None = None
        self.prune_threshold = None
        self._calib: list[np.ndarray] = []
        self._calib_frames = int(round(max(2.0, self.cfg.calibration_s) * self.fs))

    def _maybe_prune(self, amp: np.ndarray) -> None:
        """只依据校准窗决定一次，之后不再改变。"""
        if not self.auto_prune or self.carrier_mask is not None:
            return
        self._calib.append(amp)
        if len(self._calib) < self._calib_frames:
            return
        alive, jitter, threshold = carrier_health(np.asarray(self._calib))
        self.carrier_jitter = jitter
        self.prune_threshold = threshold
        self.carrier_mask = alive
        self._calib = []          # 无论结果如何都释放该缓冲区

    def step(self, t: float, amp: np.ndarray, truth: dict | None = None) -> dict | None:
        """送入一帧 CSI；每 100 ms 返回一个判决帧。"""
        amp = np.asarray(amp, dtype=np.float64)
        if self.n_carriers is None:
            self.n_carriers = int(amp.size)
        self.hist.append(amp)
        self.frame_count += 1
        self._maybe_prune(amp)
        if self.frame_count % self.decision_every:
            return None

        n = len(self.hist)
        motion_n = int(round(self.cfg.motion_window_s * self.fs))
        if n < motion_n:
            return None

        A = np.asarray(self.hist)
        if self.carrier_mask is not None and not self.carrier_mask.all():
            A = A[:, self.carrier_mask]
        features = compute_features(A[-motion_n:], A, self.cfg, fs=self.fs)
        det = self.detector
        snap = det.update(t, features, features["peak"])

        gains = np.asarray(features["gains"], dtype=float)
        if self.carrier_mask is not None and gains.size != amp.size:
            full = np.zeros(amp.size)
            full[self.carrier_mask] = gains
            gains = full

        self.seq += 1
        peak = features["peak"]
        frame = {
            "seq": self.seq,
            "t": round(float(t), 2),
            "state": snap,
            "features": {
                # 判决统计量：残差能量 / 校准得到的背景电平
                "motion": snap["motion_ratio"],
                "motion_energy": round(float(features["energy"]), 4),
                "motion_coherence": round(float(features["coherence"]), 3),
                "interference": bool(features["interference"]),
                "gains": [round(float(g), 4) for g in gains],
                "bpm_raw": None if peak.get("bpm") is None else round(float(peak["bpm"]), 1),
                "confidence": round(float(peak.get("confidence", 0.0)), 3),
                "height": round(float(peak.get("height", 0.0)), 4),
                "prominence": round(float(peak.get("prominence", 0.0)), 4),
                "lag_s": None if peak.get("lag_s") is None else round(float(peak["lag_s"]), 3),
                "breath_ready": bool(features["breath_ready"]),
                "carriers_used": int(gains.size if self.carrier_mask is None
                                     else self.carrier_mask.sum()),
            },
            "acf": [round(float(v), 4) for v in features["acf_display"]],
            "acf_step_s": features["acf_step_s"],
            "csi": [round(float(v), 4) for v in amp],
            "truth": _clean_truth(truth),
        }
        return frame


def _clean_truth(truth: dict | None) -> dict | None:
    if truth is None:
        return None
    out = {}
    for k, v in truth.items():
        if isinstance(v, (float, np.floating)):
            out[k] = None if not np.isfinite(v) else round(float(v), 4)
        elif isinstance(v, (bool, np.bool_)):
            out[k] = bool(v)
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        else:
            out[k] = v
    return out
