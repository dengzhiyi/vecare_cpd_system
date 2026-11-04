#!/usr/bin/env python3
"""真实音频路径的命令行入口。

    # 1. 生成 Kasami 探测音（录音时必须循环播放它）
    python replay_audio.py --emit-probe probe.wav --seconds 300

    # 2. 判定前先做录音质量检查；FAIL 时不要继续
    python replay_audio.py capture.wav --qc

    # 3. 把已录好的音频跑一遍检测
    python replay_audio.py capture.wav

    # 4. 没有硬件也能验证这条链路
    python replay_audio.py --selfcheck

    # 5. 存成可回放的会话，之后在 Web UI「日志回放」里看
    python replay_audio.py capture.wav --save data/sessions

介质来源：
    ``--selfcheck`` 把七个场景渲染成 ``data/selfcheck/*.wav``（已被 git 忽略）
    并回放它们；这是本项目唯一自行生成的 WAV，且每次运行都会重新生成。
    真实验证需要在 ``probe.wav`` 循环播放时，以 >= 48 kHz 录制一段
    ``capture.wav``；仓库不附带任何真实座舱录音。没有 Kasami 探测音时，座舱
    录音只包含语音/路噪，无法产生 CSI。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import core.scenarios as scenarios
from core.config import AUDIO_FS
from core.offline import run_offline
from core.config import plan_band
from core.replay import check_audio, export_probe, render_wav, run_wav


def _fmt(value, digits=1, unit=""):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}{unit}"
    return f"{value}{unit}"


def print_report(result: dict) -> None:
    src, m = result["source"], result["metrics"]
    sync = src.get("sync") or {}
    print("=" * 66)
    print(f"文件        {src['path']}")
    print(f"采样率      {src['fs']} Hz      帧率  {src['csi_fs']} Hz"
          f"（周期 {src['period_samples']} 采样）")
    print(f"时长/帧数   {src['duration_s']} s / {src['frames']} 帧"
          f"      补齐 {sync.get('filled', 0)}  空洞 {sync.get('holes', 0)}")
    used = src.get("carriers_used", src["carriers"])
    pruned = src.get("carriers_pruned") or []
    print(f"带宽        {src.get('band', '—')}")
    print(f"载波        {src['carriers']} 个 @ {src['carrier_span_hz'][0]:.0f}"
          f"–{src['carrier_span_hz'][1]:.0f} Hz      探头裕量 {src['probe_db_min']} dB")
    if len(pruned) != 0 or used != src["carriers"]:
        note = ""
        if len(pruned) > 8:
            note = "  ← 硬件带宽不足，建议用 --hw-f-max 重新规划频带"
        print(f"           实际使用 {used} 个，裁剪 {len(pruned)} 个"
              + (f"（{', '.join(f'{f/1e3:.1f}k' for f in pruned[:8])}"
                 f"{'...' if len(pruned) > 8 else ''}）" if pruned else "") + note)
    clip = src.get("clipped_samples", 0)
    print(f"电平        峰值 {src['peak']}  RMS {src['rms']}"
          f"      削波样本 {clip}" + ("  ← 削波会破坏 CSI" if clip else ""))

    print("-" * 66)
    source_label = {"motion": "运动", "breathing": "呼吸"}.get(m.get("confirmed_by"), "—")
    if not m.get("detected"):
        print("判定        未检测到儿童")
    else:
        print(f"判定        检测到儿童 @ {_fmt(m['presence_at'])} s（{source_label}证据）")
    if m.get("alarm_at") is None:
        print("告警        未触发")
    else:
        print(f"告警        @ {_fmt(m['alarm_at'])} s   原因 {m.get('alarm_reason')}")
        print(f"响应时间    {_fmt(m.get('response_time_s'))} s（自首条证据起算）")
    if m.get("bpm_median") is not None:
        lo, hi = m["bpm_range"]
        print(f"呼吸率      {_fmt(m['bpm_median'], 1)} BPM"
              f"（{lo:.1f}–{hi:.1f}，n={m['bpm_samples']}）")
    else:
        print("呼吸率      未输出")
    print(f"运动证据    {m['motion_evidence_frames']} 帧 / {m['frames']}"
          f"      干扰抑制 {m['interference_frames']} 帧"
          f"      最大能量比 {m['max_motion_ratio']}")
    n_carriers = src.get("carriers_used", src.get("carriers"))
    print(f"MRC 有效子载波 {_fmt(m.get('mrc_neff'), 2)} / {n_carriers}")
    print("=" * 66)


def print_qc(qc: dict) -> None:
    """打印录音质检结果；FAIL 时调用方应停止，不再跑判定。"""
    print("=" * 66)
    print(f"录音质检    {qc['path']}")
    for item in qc["checks"]:
        print(f"{item['name']}  {item['status']}  {item['detail']}")
    print("-" * 66)
    advice = {
        "PASS": "可以进入判定",
        "WARN": "可以判定，但结果可能不稳，建议先改善录音",
        "FAIL": "不要进入判定，先修录音",
    }[qc["status"]]
    print(f"结论  {qc['status']}  {advice}")
    print("=" * 66)


def selfcheck(snr_db: float, out_dir: Path, hw_f_max: float | None = None) -> int:
    """把每个场景渲染成 WAV、回放它，并与仿真结果对比。

    这是唯一能把前端故障与场景难度区分开的检查：两边看到的是同一个场景，
    因此结果不一致就意味着音频链路有问题。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    band = plan_band(hw_f_max)
    print(f"自检：仿真 -> WAV -> 回放   信噪比 {snr_db:.0f} dB  输出 {out_dir}")
    print(f"频带：{band.describe()}\n")
    head = (f"{'场景':<10}{'存在(直连/回放)':>20}{'告警(直连/回放)':>20}"
            f"{'BPM(真值/回放)':>18}{'运动帧':>12}")
    print(head)
    print("-" * 82)
    failures = 0
    for key in scenarios.SCENARIOS:
        scenario = scenarios.SCENARIOS[key]
        path = out_dir / f"{key}.wav"
        render_wav(path, key, snr_db=snr_db, hw_f_max=hw_f_max)
        direct = run_offline(key, hw_f_max=hw_f_max)["metrics"]
        replayed = run_wav(path, hw_f_max=hw_f_max)["metrics"]

        true_bpm = scenario.target.bpm if scenario.target else None
        cells = [
            f"{_fmt(direct['presence_at'])}/{_fmt(replayed['presence_at'])}",
            f"{_fmt(direct['alarm_at'])}/{_fmt(replayed['alarm_at'])}",
            f"{_fmt(true_bpm, 0)}/{_fmt(replayed['bpm_median'], 1)}",
            f"{direct['motion_evidence_frames']}/{replayed['motion_evidence_frames']}",
        ]
        print(f"{key:<10}{cells[0]:>20}{cells[1]:>20}{cells[2]:>18}{cells[3]:>12}")

        # 契约：告警判决相同，BPM 区间相同
        alarm_mismatch = (direct["alarm_at"] is None) != (replayed["alarm_at"] is None)
        bpm = replayed["bpm_median"]
        bpm_ok = true_bpm is None or (bpm is not None and abs(bpm - true_bpm) <= 0.15 * true_bpm)
        if alarm_mismatch or not bpm_ok:
            failures += 1
    print("-" * 82)
    if failures:
        print(f"自检未通过：{failures} 个场景与仿真不一致")
        return 1
    print("自检通过：告警判定与呼吸率在全部场景与仿真一致")
    return 0


def save_session(result: dict, sessions_root: Path) -> Path:
    """把回放结果写成 Web UI 可加载的会话。"""
    now = datetime.now()
    stem = Path(result["source"]["path"]).stem[:40] or "wav"
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in stem).strip("-") or "wav"
    session_id = f"{now:%Y%m%d-%H%M%S}-wav-{safe}"
    target = sessions_root / session_id
    n = 2
    while target.exists():
        target = sessions_root / f"{session_id}-{n}"
        n += 1
    target.mkdir(parents=True)
    session_id = target.name

    m = result["metrics"]
    meta = {
        "session_id": session_id,
        "app": "VeCare CPD 功能样机",
        "started_at": now.isoformat(timespec="seconds"),
        "ended_at": now.isoformat(timespec="seconds"),
        "scenario_key": "wav-capture",
        "scenario_name": f"实采回放 · {Path(result['source']['path']).name}",
        "scenario": {"key": "wav-capture", "name": "实采回放", "target": None},
        "duration_s": m.get("duration_s"),
        "sim_time_s": m.get("duration_s"),
        "sensitivity": result["detector_config"].get("sensitivity"),
        "detector_config": result["detector_config"],
        "speed": 1.0,
        "seed": None,
        "status": "completed",
        "frames": len(result["frames"]),
        "events": len(result["events"]),
        "summary": m,
        "source": {k: v for k, v in result["source"].items() if k != "sync"},
    }
    (target / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with open(target / "frames.jsonl", "w", encoding="utf-8") as fh:
        for frame in result["frames"]:
            fh.write(json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n")
    with open(target / "events.jsonl", "w", encoding="utf-8") as fh:
        for ev in result["events"]:
            fh.write(json.dumps(ev, ensure_ascii=False, separators=(",", ":")) + "\n")
    return target


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="把录音文件接入 VeCare CPD 检测链路",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("audio", nargs="?", help="已录好的 WAV 文件")
    ap.add_argument("--emit-probe", metavar="WAV",
                    help="写出 Kasami 探测音，录音时循环播放")
    ap.add_argument("--seconds", type=float, default=300.0,
                    help="Kasami 探测音时长（默认 300 s）")
    ap.add_argument("--qc", action="store_true",
                    help="只做录音质量检查，不跑检测；FAIL 时退出码 1")
    ap.add_argument("--selfcheck", action="store_true", help="渲染全部场景并回放自检")
    ap.add_argument("--snr", type=float, default=35.0, help="自检渲染信噪比（默认 35 dB）")
    ap.add_argument("--out", default="data/selfcheck", help="自检输出目录")
    ap.add_argument("--fs", type=int, default=None, help="重采样目标采样率（默认保持原样）")
    ap.add_argument("--sensitivity", type=float, default=1.0, help="灵敏度 0.3–2.5")
    ap.add_argument("--json", action="store_true", help="只输出 JSON")
    ap.add_argument("--save", metavar="DIR", help="存成可回放会话（如 data/sessions）")
    ap.add_argument("--hw-f-max", type=float, default=None, metavar="HZ",
                    help="扬声器+麦克风的 -3 dB 频响上限（如 18000）；不给则按论文 23.5 kHz")
    args = ap.parse_args(argv)

    if args.emit_probe:
        info = export_probe(args.emit_probe, seconds=args.seconds, hw_f_max=args.hw_f_max)
        print(json.dumps(info, ensure_ascii=False, indent=2))
        print(
            "\n用法：把它循环播放给座舱，同时录麦克风。\n"
            "  · 采样率必须与录音一致（默认 48 kHz）\n"
            "  · 录音不能开降噪 / 回声消除 / 自动增益\n"
            "  · 播放设备要能覆盖 10.3–23.7 kHz（窄带器件请传 --hw-f-max）"
        )
        return 0

    if args.selfcheck:
        return selfcheck(args.snr, Path(args.out), hw_f_max=args.hw_f_max)

    if not args.audio:
        ap.print_help()
        return 2

    if args.qc:
        qc = check_audio(args.audio, fs=args.fs, hw_f_max=args.hw_f_max)
        if args.json:
            print(json.dumps(qc, ensure_ascii=False, indent=2, default=str))
        else:
            print_qc(qc)
        return 0 if qc["passed"] else 1

    result = run_wav(args.audio, sensitivity=args.sensitivity, fs=args.fs,
                     hw_f_max=args.hw_f_max)
    if args.json:
        print(json.dumps(
            {k: v for k, v in result.items() if k != "frames"},
            ensure_ascii=False, indent=2, default=str,
        ))
    else:
        print_report(result)
    if args.save:
        path = save_session(result, Path(args.save))
        print(f"会话已保存：{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
