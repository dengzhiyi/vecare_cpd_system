"""平台可测性预检：在一组空车/有人录音上先判断“这套平台能不能测”。

这不是产品指标，而是入场券。若帧栅格不稳、载波跨度不足或帧级特征分不开，
先修硬件与摆位；此时算出来的 TPR/误报率没有意义，不能写进验收报告。

用法::

    python3 tools/check_platform.py \\
        --empty empty1.wav empty2.wav \\
        --people people1.wav people2.wav \\
        --hw-f-max 18000

退出码 0 表示硬门全部通过，1 表示至少一项硬门 FAIL。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.replay import check_audio, load_csi, run_wav  # noqa: E402


def _auc(empty: np.ndarray, people: np.ndarray) -> float | None:
    """方向性 AUC：P(空车特征 < 有人特征)，平局记 0.5。"""
    if empty.size == 0 or people.size == 0:
        return None
    gt = (empty[:, None] < people[None, :]).mean()
    eq = (empty[:, None] == people[None, :]).mean()
    return float(gt + 0.5 * eq)


def _best_operating_point(
    empty: np.ndarray, people: np.ndarray, max_fpr: float = 0.05
) -> tuple[float | None, float | None]:
    """在空车误报率 ≤ max_fpr 的约束下，取有人检出率最高的工作点。"""
    if empty.size == 0 or people.size == 0:
        return None, None
    thresholds = np.unique(np.concatenate([empty, people]))
    best = (None, None)
    for thr in thresholds:
        fpr = float((empty > thr).mean())
        if fpr > max_fpr:
            continue
        tpr = float((people > thr).mean())
        if best[1] is None or tpr > best[1]:
            best = (float(thr), tpr)
    return best


def _common_mode_fraction(path: Path, hw_f_max: float | None) -> float:
    """CSI 第一主成分中 <0.65 Hz 的能量占比（共模/慢变主导程度）。"""
    amp, _, meta = load_csi(path, hw_f_max=hw_f_max)
    a = amp - amp.mean(axis=0)
    a = a - a.mean(axis=0)
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    pc = u[:, 0] * s[0]
    pc = pc - pc.mean()
    spec = np.abs(np.fft.rfft(pc * np.hanning(pc.size))) ** 2
    freqs = np.fft.rfftfreq(pc.size, 1.0 / float(meta["csi_fs"]))
    total = float(spec.sum())
    return float(spec[freqs < 0.65].sum() / total) if total > 0 else 0.0


def _collect(
    paths: list[Path], hw_f_max: float | None
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    rows, ratios, coherences, bpm_flags = [], [], [], []
    for path in paths:
        qc = check_audio(path, hw_f_max=hw_f_max)
        # 诊断阶段仍要拿到特征；是否可信由 QC 门单独判定。
        result = run_wav(path, hw_f_max=hw_f_max, require_qc=False)
        frames = result["frames"]
        ratios.extend(float(f["features"]["motion"]) for f in frames)
        coherences.extend(float(f["features"]["motion_coherence"]) for f in frames)
        bpm_flags.extend(f["features"]["bpm_raw"] is not None for f in frames)
        sync = qc.get("sync") or {}
        rows.append({
            "name": path.name,
            "qc": qc["status"],
            "passed": qc["passed"],
            "strict_ratio": float(sync.get("strict_ratio", 0.0)),
            "period_std": float(sync.get("period_std_samples", 0.0)),
            "spread_db": None if not qc.get("meta")
            else float(qc["meta"].get("carrier_spread_db", 0.0)),
            "alarm_at": result["metrics"]["alarm_at"],
            "bpm_median": result["metrics"]["bpm_median"],
            "common_mode": _common_mode_fraction(path, hw_f_max),
        })
    return (
        rows,
        np.asarray(ratios, dtype=float),
        np.asarray(coherences, dtype=float),
        np.asarray(bpm_flags, dtype=bool),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="空车/有人两组录音的平台可测性预检")
    ap.add_argument("--empty", nargs="+", required=True, help="空车录音")
    ap.add_argument("--people", nargs="+", required=True, help="有人录音")
    ap.add_argument("--hw-f-max", type=float, default=None, metavar="HZ")
    ap.add_argument("--max-fpr", type=float, default=0.05, help="允许的空车误报率上限")
    args = ap.parse_args(argv)

    empty = [Path(p) for p in args.empty]
    people = [Path(p) for p in args.people]
    for path in empty + people:
        if not path.exists():
            print(f"找不到文件：{path}", file=sys.stderr)
            return 2

    e_rows, e_ratio, e_coh, e_bpm = _collect(empty, args.hw_f_max)
    p_rows, p_ratio, p_coh, p_bpm = _collect(people, args.hw_f_max)

    print("== 单段录音 ==")
    for row in e_rows + p_rows:
        spread = "  n/a  " if row["spread_db"] is None else f"{row['spread_db']:6.1f}"
        print(
            f"  {row['name'][:44]:44s} QC={row['qc']:4s} "
            f"严格栅格={row['strict_ratio']*100:5.1f}% "
            f"σ={row['period_std']:.2f} 跨度={spread}dB "
            f"共模={row['common_mode']:.2f} 告警={row['alarm_at']}"
        )

    auc_ratio = _auc(e_ratio, p_ratio)
    auc_coh = _auc(e_coh, p_coh)
    thr, tpr = _best_operating_point(e_ratio, p_ratio, args.max_fpr)
    fpr = None if thr is None else float((e_ratio > thr).mean())
    false_breath = float(e_bpm.mean()) if e_bpm.size else 0.0
    true_breath = float(p_bpm.mean()) if p_bpm.size else 0.0
    alarms = [r["alarm_at"] for r in p_rows]

    print("\n== 平台指标 ==")
    print(f"  帧级可分性 AUC：motion={auc_ratio}, coherence={auc_coh}")
    print(f"  最佳工作点：阈值={thr} TPR={tpr} FPR={fpr}")
    print(f"  呼吸存在率：空车 {false_breath*100:.1f}% · 有人 {true_breath*100:.1f}%")
    print(f"  有人段告警时刻：{alarms}")

    gates: list[tuple[str, bool, str]] = []
    strict_ok = all(r["strict_ratio"] >= 0.99 for r in e_rows + p_rows)
    gates.append(("帧栅格（严格 960 占比 ≥99%）", strict_ok,
                  "手机播放/采集时钟不同源时该门必失败，先换同一时钟或回环"))
    spread_ok = all(
        r["spread_db"] is not None and r["spread_db"] >= -10.0
        for r in e_rows + p_rows
    )
    gates.append(("载波跨度（≥-10 dB，仅提示）", spread_ok,
                  "多径零点也会造成差距，需结合频响扫频判断"))
    best_auc = max(v for v in (auc_ratio, auc_coh) if v is not None) if (auc_ratio is not None or auc_coh is not None) else None
    gates.append(("帧级可分性（AUC ≥0.9）", bool(best_auc is not None and best_auc >= 0.9),
                  "低于 0.9 时 TPR/误报率无统计意义"))
    gates.append((f"空车误报率（FPR ≤{args.max_fpr:.0%}）", bool(fpr is not None and fpr <= args.max_fpr), ""))
    gates.append(("空车误呼吸率（≤5%）", false_breath <= 0.05, ""))
    gates.append(("有人呼吸存在率（≥50%）", true_breath >= 0.50, ""))
    alarms_ok = all(a is not None and a <= 10.2 for a in alarms)
    gates.append(("有人端到端告警（≤10.2 s）", alarms_ok, "确认晚于预算时应当立即告警"))

    print("\n== 门禁 ==")
    for name, ok, note in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ← {note}" if note and not ok else ""))

    # 只有“平台本身是否可用”的三项作为硬门；FPR/TPR 在不可测时只做展示。
    hard = [gates[0][1], gates[2][1]]
    if any(r["qc"] == "FAIL" for r in e_rows + p_rows):
        print("\n  质检存在 FAIL：这些录音不能用于任何指标计算。")
        hard.append(False)
    return 0 if all(hard) else 1


if __name__ == "__main__":
    raise SystemExit(main())
