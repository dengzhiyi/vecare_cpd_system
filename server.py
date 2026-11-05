#!/usr/bin/env python3
"""VeCare CPD 原型 HTTP 服务器（仅使用 Python 标准库）。

提供仪表盘以及一个小型 JSON API：

  GET  /api/scenarios                 预置座舱场景 + 模型参数
  GET  /api/status                    当前引擎 / 检测器状态
  GET  /api/live?since=N              增量实时帧 + 事件
  POST /api/live/start                {scenario, sensitivity, speed, seed?, hw_f_max?}
  POST /api/live/pause                {paused}
  POST /api/live/speed                {speed}
  POST /api/live/ack                  确认 / 静音告警
  POST /api/live/stop                 停止并收尾会话
  GET  /api/calibration               空车标定基线状态
  POST /api/calibration               {duration_s?, hw_f_max?} 执行空车标定
  POST /api/calibration/clear         删除已保存的空车基线
  GET  /api/sessions                  已记录的会话
  GET  /api/sessions/<id>             回放数据（meta + frames + events）
  GET  /api/sessions/<id>/download/<f> 下载 JSONL / meta
  POST /api/audio/analyze?hw_f_max=18000&name=capture.wav&seconds=60
                                      上传 WAV（请求体为原始字节），返回波形、
                                      频谱、CSI 与关键参数，供「音频可视化」页
                                      使用；seconds 只分析开头一段，缺省为
                                      完整文件
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.config import (  # noqa: E402
    RESPONSE_BUDGET_S,
    APNEA_TIMEOUT_S,
    BREATH_WINDOW_S,
    CSI_FS,
    DECISION_FS,
    EVIDENCE_WINDOW_S,
    MOTION_WINDOW_S,
    N_SUBCARRIERS,
    PRESENCE_RATIO,
    plan_band,
)
from core.audio_viz import analyze_bytes  # noqa: E402
from core.scenarios import SCENARIOS, list_scenarios  # noqa: E402
from engine import SimulationEngine  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
ENGINE = SimulationEngine(BASE_DIR / "data" / "sessions")

STATIC_FILES = {"index.html", "styles.css", "app.js"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_ANALYZE_SECONDS = 3600.0

# UI 可以提供的硬件上限。首项 24 kHz 是最高可用频响上限的实际值；20 kHz
# 是论文实测上限，也是默认档。plan_band 会拒绝载波少于四个的上限（低于约
# 12.75 kHz），因此 API 刻意只暴露这组经过测试的取值。
BAND_CHOICES = (24_000, 20_000, 18_000, 16_000, 14_000)


def band_options() -> list[dict]:
    """UI 渲染一个带宽选项所需的一切，均已预先算好。"""
    out = []
    for hw in BAND_CHOICES:
        band = plan_band(hw)
        out.append(
            {
                "hw_f_max": hw,
                "label": f"{hw / 1000:.0f}k",
                "f_lo_hz": band.f_lo,
                "f_hi_hz": band.f_hi,
                "n_carriers": band.n_carriers,
                "spacing_hz": round(band.spacing_hz, 1),
                "sweep_hz": [band.sweep_lo, band.sweep_hi],
                "describe": band.describe(),
            }
        )
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "VeCareCPD/0.1"
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------------
    def log_message(self, fmt, *args):  # 演示时保持控制台安静
        if getattr(self.server, "verbose", False):
            sys.stderr.write("[http] " + fmt % args + "\n")

    def _send_json(self, obj, code: int = 200) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, data: bytes, content_type: str, code: int = 200,
                    download_name: str | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            # 二进制音频误投到 JSON 接口时给出明确提示，而不是泄漏 codec 报错。
            raise ValueError(
                "请求体不是 JSON；音频请上传到 /api/audio/analyze"
            ) from exc

    def _analyze_upload(self, parsed) -> None:
        """接收原始 WAV 字节并返回音频可视化数据。

        这里不走 multipart：前端直接把 File 作为请求体发送，服务端只需读一次
        Content-Length，避免为标准库手写一个 multipart 解析器。
        """
        query = parse_qs(parsed.query)
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._send_json({"error": "空请求体：请上传 WAV 文件"}, 400)
            return
        if length > MAX_UPLOAD_BYTES:
            self._send_json({"error": "文件过大：上限 200 MB"}, 413)
            return
        raw = self.rfile.read(length)
        if not raw.startswith(b"RIFF"):
            self._send_json({"error": "只支持 WAV（RIFF）文件"}, 400)
            return

        hw_text = _first(query, "hw_f_max", "")
        hw_f_max: float | None = None
        if hw_text not in ("", "null", "None"):
            try:
                hw_f_max = float(hw_text)
            except ValueError:
                self._send_json({"error": f"invalid hw_f_max: {hw_text}"}, 400)
                return
            if hw_f_max not in BAND_CHOICES:
                valid = [int(v) for v in BAND_CHOICES if v is not None]
                self._send_json(
                    {"error": f"hw_f_max must be one of {valid}; omit it for the default 20k"},
                    400,
                )
                return

        seconds_text = _first(query, "seconds", "")
        max_seconds: float | None = None
        if seconds_text not in ("", "null", "None"):
            try:
                max_seconds = float(seconds_text)
            except ValueError:
                self._send_json({"error": f"invalid seconds: {seconds_text}"}, 400)
                return
            if not 0 < max_seconds <= MAX_ANALYZE_SECONDS:
                self._send_json(
                    {"error": f"seconds must be in (0, {MAX_ANALYZE_SECONDS:.0f}]"},
                    400,
                )
                return

        result = analyze_bytes(raw, hw_f_max=hw_f_max, max_seconds=max_seconds)
        name = unquote(_first(query, "name", "upload.wav"))
        result["path"] = Path(name).name
        self._send_json(result)

    # ------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            query = parse_qs(parsed.query)

            if path == "/api/health":
                self._send_json({"ok": True})
                return
            if path == "/api/scenarios":
                self._send_json(
                    {
                        "scenarios": list_scenarios(),
                        "params": {
                            "csi_fs": CSI_FS,
                            "decision_fs": DECISION_FS,
                            # 24 是上限，而非实际使用的数量：载波集合会依据
                            # 下方的硬件上限逐项推导。
                            "subcarriers_ceiling": N_SUBCARRIERS,
                            "bands": band_options(),
                            "motion_window_s": MOTION_WINDOW_S,
                            "breath_window_s": BREATH_WINDOW_S,
                            "evidence_window_s": EVIDENCE_WINDOW_S,
                            "presence_ratio": PRESENCE_RATIO,
                            "response_budget_s": RESPONSE_BUDGET_S,
                            "apnea_timeout_s": APNEA_TIMEOUT_S,
                        },
                    }
                )
                return
            if path == "/api/status":
                self._send_json(ENGINE.status())
                return
            if path == "/api/calibration":
                self._send_json(ENGINE.calibration_status())
                return
            if path == "/api/live":
                since = int(_first(query, "since", "0"))
                self._send_json(ENGINE.live_payload(since, _first(query, "session", None)))
                return
            if path == "/api/sessions":
                self._send_json({"sessions": ENGINE.list_sessions()})
                return
            if path.startswith("/api/sessions/"):
                parts = [p for p in path.split("/") if p]
                if len(parts) >= 3:
                    session_id = parts[2]
                    if len(parts) >= 5 and parts[3] == "download":
                        self._download(session_id, parts[4])
                        return
                    if len(parts) == 3:
                        data = ENGINE.load_session(session_id)
                        if data is None:
                            self._send_json({"error": "session not found"}, 404)
                        else:
                            self._send_json(data)
                        return
                self._send_json({"error": "bad session path"}, 400)
                return

            if path in ("/", "/index.html"):
                self._static("index.html")
                return
            if path.lstrip("/") in STATIC_FILES:
                self._static(path.lstrip("/"))
                return
            self._send_json({"error": "not found", "path": path}, 404)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send_json({"error": str(exc)}, 500)

    def do_POST(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            if path == "/api/audio/analyze":
                self._analyze_upload(parsed)
                return
            body = self._read_json()

            if path == "/api/live/start":
                key = str(body.get("scenario", "sleeping"))
                if key not in SCENARIOS:
                    self._send_json({"error": f"unknown scenario: {key}"}, 400)
                    return
                hw = body.get("hw_f_max")
                if hw in (None, "", 0):
                    hw_f_max = None
                else:
                    try:
                        hw_f_max = float(hw)
                    except (TypeError, ValueError):
                        self._send_json({"error": f"invalid hw_f_max: {hw}"}, 400)
                        return
                    if hw_f_max not in BAND_CHOICES:
                        valid = [int(v) for v in BAND_CHOICES if v is not None]
                        self._send_json(
                            {"error": f"hw_f_max must be one of {valid}; omit it for the default 20k"},
                            400,
                        )
                        return
                status = ENGINE.start(
                    scenario_key=key,
                    sensitivity=float(body.get("sensitivity", 1.0)),
                    speed=float(body.get("speed", 1.0)),
                    seed=body.get("seed"),
                    hw_f_max=hw_f_max,
                )
                self._send_json(status, 201)
                return
            if path == "/api/live/pause":
                self._send_json(ENGINE.set_paused(bool(body.get("paused", True))))
                return
            if path == "/api/live/speed":
                self._send_json(ENGINE.set_speed(float(body.get("speed", 1.0))))
                return
            if path == "/api/live/ack":
                self._send_json(ENGINE.acknowledge())
                return
            if path == "/api/live/stop":
                self._send_json(ENGINE.stop(wait=True))
                return
            if path == "/api/calibration":
                if ENGINE.status().get("running"):
                    self._send_json({"error": "请先停止当前监测，再执行空车标定"}, 409)
                    return
                hw = body.get("hw_f_max")
                if hw in (None, "", 0):
                    hw_f_max = None
                else:
                    try:
                        hw_f_max = float(hw)
                    except (TypeError, ValueError):
                        self._send_json({"error": f"invalid hw_f_max: {hw}"}, 400)
                        return
                    if hw_f_max not in BAND_CHOICES:
                        valid = [int(v) for v in BAND_CHOICES if v is not None]
                        self._send_json(
                            {"error": f"hw_f_max must be one of {valid}; omit it for the default 20k"},
                            400,
                        )
                        return
                result = ENGINE.calibrate_empty(
                    duration_s=float(body.get("duration_s", 5.0)),
                    hw_f_max=hw_f_max,
                )
                self._send_json(result, 201)
                return
            if path == "/api/calibration/clear":
                self._send_json(ENGINE.clear_calibration())
                return
            self._send_json({"error": "not found", "path": path}, 404)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON body"}, 400)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send_json({"error": str(exc)}, 500)

    # ------------------------------------------------------------------
    def _static(self, name: str) -> None:
        path = WEB_DIR / name
        if not path.exists():
            self._send_json({"error": "static file missing"}, 404)
            return
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        self._send_bytes(path.read_bytes(), ctype)

    def _download(self, session_id: str, filename: str) -> None:
        if filename not in {"meta.json", "frames.jsonl", "events.jsonl"}:
            self._send_json({"error": "file not allowed"}, 400)
            return
        path = ENGINE.session_file(session_id, filename)
        if path is None:
            self._send_json({"error": "file not found"}, 404)
            return
        ctype = "application/json" if filename.endswith(".json") else "application/x-ndjson"
        self._send_bytes(
            path.read_bytes(), ctype,
            download_name=f"{session_id}-{filename}",
        )


def _first(query: dict, key: str, default: str) -> str:
    vals = query.get(key)
    return vals[0] if vals else default


def main() -> None:
    parser = argparse.ArgumentParser(description="VeCare CPD functional prototype")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.verbose = args.verbose
    print(f"VeCare CPD prototype running at http://{args.host}:{args.port}")
    print(f"  sessions: {ENGINE.sessions_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ENGINE.stop(wait=True)
        server.server_close()


if __name__ == "__main__":
    main()
