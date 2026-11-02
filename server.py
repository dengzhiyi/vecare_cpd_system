#!/usr/bin/env python3
"""VeCare CPD prototype HTTP server (Python standard library only).

Serves the dashboard plus a small JSON API:

  GET  /api/scenarios                 preset cabin scenarios + model parameters
  GET  /api/status                    current engine / detector status
  GET  /api/live?since=N               incremental live frames + events
  POST /api/live/start                {scenario, sensitivity, speed, seed?, hw_f_max?}
  POST /api/live/pause                {paused}
  POST /api/live/speed                {speed}
  POST /api/live/ack                  acknowledge / mute the alarm
  POST /api/live/stop                 stop and finalize the session
  GET  /api/sessions                  recorded sessions
  GET  /api/sessions/<id>             replay data (meta + frames + events)
  GET  /api/sessions/<id>/download/<f> download JSONL / meta
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
from core.scenarios import SCENARIOS, list_scenarios  # noqa: E402
from engine import SimulationEngine  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
ENGINE = SimulationEngine(BASE_DIR / "data" / "sessions")

STATIC_FILES = {"index.html", "styles.css", "app.js"}

# Hardware ceilings the UI may offer. None is the paper's own band; the rest are
# the driver specs we actually meet (see README, "硬件带宽受限"). plan_band
# rejects ceilings that leave fewer than four carriers (below ~12.75 kHz), and
# the API deliberately exposes only this tested set.
BAND_CHOICES = (None, 20_000, 18_000, 16_000, 14_000)


def band_options() -> list[dict]:
    """Everything the UI needs to render one bandwidth chip, precomputed."""
    out = []
    for hw in BAND_CHOICES:
        band = plan_band(hw)
        out.append(
            {
                "hw_f_max": hw,
                "label": "论文" if hw is None else f"{hw / 1000:.0f}k",
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
    def log_message(self, fmt, *args):  # keep the console quiet during demos
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
        return json.loads(raw.decode("utf-8"))

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
                            # 24 is a ceiling, not the count actually used: the
                            # carrier set is derived per hardware ceiling below.
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
            path = urlparse(self.path).path
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
                            {"error": f"hw_f_max must be null or one of {valid}"},
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
            self._send_json({"error": "not found", "path": path}, 404)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON body"}, 400)
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
