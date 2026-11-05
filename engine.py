"""带会话记录的实时仿真引擎。

按照真实时钟节奏（可乘以速度倍数）运行声学信道 + SAS 流水线，驱动 CPD 状态机，
并写出可回放的 JSONL 日志。

Web UI 走的正是这条路径，因此实时监测就是直接仿真：CSI 每 20 ms 合成一次，
不涉及任何 WAV/音频。JSONL 会话保存 CSI 幅度和判决以供回放；它们不是录音。
WAV 输入属于 ``replay_audio.py`` / ``core.replay``，而自检则属于被 git 忽略的
``data/selfcheck/`` 渲染产物。
"""
from __future__ import annotations

import json
import re
import threading
import time
import zlib
from datetime import datetime
from pathlib import Path

from core import calibration as calib
from core.channel import AcousticChannel
from core.config import CSI_FS, DetectorConfig, plan_band
from core.offline import evaluate_frames
from core.pipeline import SASPipeline
from core.scenarios import get_scenario

SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


class SimulationEngine:
    def __init__(self, sessions_dir: str | Path):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

        self.running = False
        self.paused = False
        self.done = True
        self.finished_reason: str | None = None
        self.speed = 1.0

        self.scenario = None
        self.channel: AcousticChannel | None = None
        self.pipeline: SASPipeline | None = None
        self.session_id: str | None = None
        self.session_dir: Path | None = None
        self.meta: dict = {}

        self.sim_t = 0.0
        self.frames: list[dict] = []
        self.events: list[dict] = []
        self._frames_file = None
        self._events_file = None
        self._events_written = 0
        self._wall_anchor = 0.0
        self._finalized = True
        self.summary: dict | None = None
        self.calibration: dict | None = calib.load_empty_calibration(source="simulation")

    # ------------------------------------------------------------------
    # 控制
    def start(
        self,
        scenario_key: str,
        sensitivity: float = 1.0,
        speed: float = 1.0,
        seed: int | None = None,
        hw_f_max: float | None = None,
    ) -> dict:
        self.stop(wait=True)
        scenario = get_scenario(scenario_key)
        band = plan_band(hw_f_max)
        speed = max(0.25, min(16.0, float(speed)))
        sensitivity = max(0.3, min(2.5, float(sensitivity)))

        with self.lock:
            self.speed = speed
            self.paused = False
            self.done = False
            self.finished_reason = None
            self.scenario = scenario
            self.summary = None
            self.sim_t = 0.0
            self.frames = []
            self.events = []
            self._events_written = 0
            self.stop_event.clear()
            self._finalized = False

            cfg = DetectorConfig(sensitivity=sensitivity)
            # 有空车标定且载波数与本次频带一致时直接加载；否则检测器从开机
            # 第一帧起做 2 s 自适应背景估计，不再等待 5 s 安静期。
            # 仿真器只加载仿真来源的基线；真实录音基线的绝对幅度尺度不同，
            # 混用会把能量比整体放大或缩小。载波数不匹配时同样回退冷启动。
            profile = calib.load_empty_calibration(band.n_carriers, source="simulation")
            self.pipeline = SASPipeline(cfg, background=profile)
            session_id = self._new_session_id(scenario.key)
            if seed is None:
                seed = zlib.crc32(f"{session_id}-{scenario.key}".encode("utf-8")) % (2 ** 31)
            self.channel = AcousticChannel(scenario, seed=int(seed), band=band)
            self.session_id = session_id
            self.session_dir = self.sessions_dir / session_id
            self.session_dir.mkdir(parents=True, exist_ok=True)

            self._frames_file = open(self.session_dir / "frames.jsonl", "w", encoding="utf-8")
            self._events_file = open(self.session_dir / "events.jsonl", "w", encoding="utf-8")

            self.meta = {
                "session_id": session_id,
                "app": "VeCare CPD 功能样机",
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "scenario": scenario.public(),
                "scenario_key": scenario.key,
                "scenario_name": scenario.name,
                "duration_s": scenario.duration_s,
                "sensitivity": cfg.sensitivity,
                "band": band.describe(),
                "f_lo_hz": band.f_lo,
                "f_hi_hz": band.f_hi,
                "hw_f_max_hz": round(float(band.hw_f_max), 1),
                "carriers_planned": int(band.n_carriers),
                "carrier_spacing_hz": round(band.spacing_hz, 1),
                "detector_config": cfg.to_dict(),
                "background": {
                    "source": "空车标定" if profile else "冷启动自适应",
                    "profile_created_at": None if profile is None else profile.get("created_at"),
                },
                "speed": speed,
                "seed": int(seed),
                "status": "running",
            }
            self._write_meta()

            self.running = True
            self._wall_anchor = time.monotonic()
            self.thread = threading.Thread(target=self._loop, name="vecare-sim", daemon=True)
            self.thread.start()

        return self.status()

    def stop(self, wait: bool = True) -> dict:
        self.stop_event.set()
        thread = self.thread
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        self._finish("stopped" if not self.done else self.finished_reason)
        return self.status()

    def set_paused(self, paused: bool) -> dict:
        with self.lock:
            self.paused = bool(paused)
            if self.running:
                self._wall_anchor = time.monotonic() - self.sim_t / self.speed
        return self.status()

    def set_speed(self, speed: float) -> dict:
        with self.lock:
            self.speed = max(0.25, min(16.0, float(speed)))
            self._wall_anchor = time.monotonic() - self.sim_t / self.speed
        return self.status()

    def acknowledge(self) -> dict:
        with self.lock:
            if self.pipeline is not None:
                self.pipeline.detector.acknowledge(self.sim_t)
                self._collect_events()
        return self.status()

    # ------------------------------------------------------------------
    # 仿真循环
    def _loop(self) -> None:
        dt = 1.0 / CSI_FS
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    paused = self.paused
                    speed = self.speed
                if paused:
                    time.sleep(0.04)
                    with self.lock:
                        self._wall_anchor = time.monotonic() - self.sim_t / speed
                    continue

                target = self._wall_anchor + self.sim_t / speed
                now = time.monotonic()
                if target > now:
                    time.sleep(min(0.04, target - now))
                    continue

                _, amp, truth = self.channel.step(dt)
                frame = self.pipeline.step(self.sim_t, amp, truth)
                with self.lock:
                    self.sim_t += dt
                    if frame is not None:
                        self.frames.append(frame)
                        if self._frames_file is not None:
                            self._frames_file.write(
                                json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n"
                            )
                        self._collect_events()
                    if self.sim_t >= self.scenario.duration_s:
                        break
        finally:
            self._finish("stopped" if self.stop_event.is_set() else "completed")

    def _finish(self, reason: str | None) -> None:
        with self.lock:
            if self._finalized:
                return
            self._finalized = True
            self.running = False
            self.paused = False
            self.done = True
            self.finished_reason = reason

            try:
                if self._frames_file is not None:
                    self._frames_file.flush()
                    self._frames_file.close()
            finally:
                self._frames_file = None
            try:
                if self._events_file is not None:
                    self._events_file.flush()
                    self._events_file.close()
            finally:
                self._events_file = None

            if self.scenario is not None:
                self.summary = evaluate_frames(self.frames, self.scenario)
            if self.meta:
                self.meta.update(
                    {
                        "ended_at": datetime.now().isoformat(timespec="seconds"),
                        "status": "stopped" if reason == "stopped" else "completed",
                        "sim_time_s": round(self.sim_t, 2),
                        "frames": len(self.frames),
                        "events": len(self.events),
                        "summary": self.summary,
                    }
                )
                self._write_meta()

    def _new_session_id(self, key: str) -> str:
        base = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{key}"
        candidate = base
        n = 2
        while (self.sessions_dir / candidate).exists():
            candidate = f"{base}-{n}"
            n += 1
        return candidate

    def _write_meta(self) -> None:
        if self.session_dir is None:
            return
        with open(self.session_dir / "meta.json", "w", encoding="utf-8") as fh:
            json.dump(self.meta, fh, ensure_ascii=False, indent=2)

    def _collect_events(self) -> None:
        if self.pipeline is None:
            return
        all_events = self.pipeline.detector.events
        while self._events_written < len(all_events):
            ev = dict(all_events[self._events_written])
            self._events_written += 1
            self.events.append(ev)
            if self._events_file is not None:
                self._events_file.write(
                    json.dumps(ev, ensure_ascii=False, separators=(",", ":")) + "\n"
                )

    # ------------------------------------------------------------------
    # 查询
    def status(self) -> dict:
        with self.lock:
            det = None
            if self.pipeline is not None:
                det = self.pipeline.detector.snapshot(self.sim_t)
            # 频带在启动时即固定；一旦运行时健康门控裁剪掉死载波，实际使用的
            # 载波数量会下降，因此从最新一帧读取它，而不是回显规划值。
            carriers_used = None
            if self.frames:
                carriers_used = self.frames[-1]["features"].get("carriers_used")
            elif self.pipeline is not None:
                carriers_used = self.pipeline.n_carriers
            return {
                "running": self.running,
                "paused": self.paused,
                "done": self.done,
                "finished_reason": self.finished_reason,
                "speed": self.speed,
                "t": round(self.sim_t, 2),
                "duration_s": None if self.scenario is None else self.scenario.duration_s,
                "scenario": None if self.scenario is None else self.scenario.key,
                "scenario_name": None if self.scenario is None else self.scenario.name,
                "session_id": self.session_id,
                "band": self.meta.get("band"),
                "band_range_hz": [self.meta.get("f_lo_hz"), self.meta.get("f_hi_hz")],
                "hw_f_max_hz": self.meta.get("hw_f_max_hz"),
                "carriers_planned": self.meta.get("carriers_planned"),
                "carriers_used": carriers_used,
                "detector": det,
                "calibration": {
                    "available": self.calibration is not None,
                    "profile": self.calibration,
                },
                "summary": self.summary,
                "frame_count": self.frames[-1]["seq"] if self.frames else 0,
                "events_count": len(self.events),
                "target": None if self.scenario is None or self.scenario.target is None
                else self.scenario.public()["target"],
            }

    def calibrate_empty(self, duration_s: float = 5.0, hw_f_max: float | None = None) -> dict:
        """执行一次空车标定并更新内存中的基线。"""
        with self.lock:
            if self.running:
                raise RuntimeError("请先停止当前监测，再执行空车标定")
            self.calibration = calib.calibrate_empty_cabin(
                duration_s=duration_s, hw_f_max=hw_f_max
            )
            return {"available": True, "profile": self.calibration}

    def calibration_status(self) -> dict:
        """返回磁盘上的基线与仿真可用性。

        仿真监测只接受 ``source="simulation"`` 的基线；实录音基线（绝对幅度
        尺度不同）会被列出但不加载。
        """
        with self.lock:
            stored = calib.load_empty_calibration()
            usable = calib.load_empty_calibration(source="simulation")
            return {
                "available": usable is not None,
                "profile": usable,
                "stored": stored,
                "stored_source": None if stored is None else stored.get("source", "simulation"),
            }

    def clear_calibration(self) -> dict:
        with self.lock:
            calib.clear_empty_calibration()
            self.calibration = None
            return {"available": False, "profile": None}

    def live_payload(self, after_seq: int = 0, session: str | None = None) -> dict:
        with self.lock:
            if session == self.session_id:
                frames = [f for f in self.frames if f["seq"] > after_seq]
            else:
                # 客户端切换到了一个新的（或重连的）会话：发送全部帧。
                frames = list(self.frames)
            scenario = None if self.scenario is None else self.scenario.public()
            return {
                "status": self.status(),
                "scenario": scenario,
                "frames": frames,
                "events": list(self.events),
            }

    def list_sessions(self) -> list[dict]:
        out = []
        for path in sorted(self.sessions_dir.iterdir(), reverse=True):
            if not path.is_dir():
                continue
            meta_path = path / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            summary = meta.get("summary") or {}
            out.append(
                {
                    "id": path.name,
                    "scenario_key": meta.get("scenario_key"),
                    "scenario_name": meta.get("scenario_name"),
                    "started_at": meta.get("started_at"),
                    "ended_at": meta.get("ended_at"),
                    "duration_s": meta.get("sim_time_s", meta.get("duration_s")),
                    "status": meta.get("status"),
                    "frames": meta.get("frames", 0),
                    "summary": summary,
                    "has_alarm": bool(summary.get("alarm_at")),
                }
            )
        return out

    def load_session(self, session_id: str) -> dict | None:
        if not SESSION_RE.match(session_id):
            return None
        path = self.sessions_dir / session_id
        if not path.is_dir() or not (path / "meta.json").exists():
            return None
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        frames = _read_jsonl(path / "frames.jsonl")
        events = _read_jsonl(path / "events.jsonl")
        return {"meta": meta, "frames": frames, "events": events}

    def session_file(self, session_id: str, name: str) -> Path | None:
        if not SESSION_RE.match(session_id):
            return None
        if name not in {"meta.json", "frames.jsonl", "events.jsonl"}:
            return None
        path = self.sessions_dir / session_id / name
        return path if path.exists() else None


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows
