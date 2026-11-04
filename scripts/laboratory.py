"""Serve the BlindSpot laboratory over a loopback-only HTTP connection."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from uuid import uuid4

from scripts.laboratory_run import Laboratory, LaboratoryError

WEB_ROOT = Path(__file__).resolve().parents[1] / "web" / "laboratory"
ASSETS = {
    "/": ("index.html", "text/html"),
    "/logo.png": ("logo.png", "image/png"),
    "/laboratory.css": ("laboratory.css", "text/css"),
    "/laboratory.js": ("laboratory.js", "text/javascript"),
    "/laboratory-viewer.js": ("laboratory-viewer.js", "text/javascript"),
}
RUN_HISTORY = 8


class RunQueue:
    """Keep a bounded session history and execute at most one analysis at a time."""

    def __init__(self, laboratory: Laboratory, executor: ThreadPoolExecutor):
        self.laboratory = laboratory
        self.executor = executor
        self.runs: OrderedDict[str, dict] = OrderedDict()
        self.lock = Lock()

    def submit(self, payload: dict) -> str:
        cid, model, mode = (payload.get(key) for key in ("case_id", "model", "mode"))
        if not all(isinstance(value, str) for value in (cid, model, mode)):
            raise ValueError("Select a case, a model and an execution mode.")
        if cid not in self.laboratory.cases or model not in self.laboratory.profiles:
            raise ValueError("Select a case and model from the protocol.")
        if mode == "replay" and f"{model}/{cid}" not in self.laboratory.recorded["attempts"]:
            raise ValueError("No archived response is available for this model and case.")
        if mode not in ("replay", "live") or (mode == "live" and payload.get("confirm_live") is not True):
            raise ValueError("Select replay or explicitly authorize the live request.")
        with self.lock:
            if self.runs and next(reversed(self.runs.values()))["state"] == "running":
                raise LaboratoryError("An analysis is already running.")
            if mode == "live":
                self.laboratory.authorize_live(cid, model)
            run_id = uuid4().hex
            self.runs[run_id] = {"state": "running", "stage": "queued", "stages": []}
            if len(self.runs) > RUN_HISTORY:
                self.runs.popitem(last=False)
        self.executor.submit(self._execute, run_id, cid, model, mode)
        return run_id

    def _execute(self, run_id: str, cid: str, model: str, mode: str) -> None:
        def progress(stage: str, **fields) -> None:
            with self.lock:
                run = self.runs[run_id]
                run.update(stage=stage, stages=[*run["stages"], stage], **fields)

        try:
            result = self.laboratory.run(cid, model, mode, progress)
        except LaboratoryError as exc:
            progress("failed", state="failed", error=str(exc))
        except Exception as exc:
            logging.error("Laboratory analysis stopped (%s).", type(exc).__name__)
            progress(
                "failed",
                state="failed",
                error="Analysis stopped because local evidence could not be reproduced.",
            )
        else:
            progress("complete", state="complete", result=result)

    def get(self, run_id: str) -> dict | None:
        with self.lock:
            run = self.runs.get(run_id)
            return dict(run) if run else None


class LaboratoryServer(ThreadingHTTPServer):
    """Local HTTP server for static assets and single-case analysis."""

    def __init__(self, port: int, queue: RunQueue):
        self.queue = queue
        super().__init__(("127.0.0.1", port), LaboratoryRequest)
        self.origin = f"http://127.0.0.1:{self.server_port}"


class LaboratoryRequest(BaseHTTPRequestHandler):
    """Restrict requests to the exact loopback host and the laboratory API."""

    server: LaboratoryServer

    def _send(self, status: HTTPStatus, media_type: str, body: bytes) -> None:
        self.send_response(status)
        content_type = (
            f"{media_type}; charset=utf-8"
            if media_type.startswith("text/") or media_type == "application/json"
            else media_type
        )
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy", "default-src 'self'; object-src 'none'; frame-ancestors 'none'"
        )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: HTTPStatus, content: dict) -> None:
        self._send(status, "application/json", json.dumps(content, allow_nan=False).encode("utf-8"))

    def _is_local(self) -> bool:
        host = f"127.0.0.1:{self.server.server_port}"
        allowed = self.headers.get("Host") == host and self.headers.get("Origin") in (
            None,
            self.server.origin,
        )
        if not allowed:
            self._json(
                HTTPStatus.FORBIDDEN, {"error": "Requests must originate from the loopback laboratory."}
            )
        return allowed

    def setup(self) -> None:
        self.request.settimeout(5)
        super().setup()

    def do_GET(self) -> None:
        if not self._is_local():
            return
        if self.path in ASSETS:
            filename, media_type = ASSETS[self.path]
            self._send(HTTPStatus.OK, media_type, (WEB_ROOT / filename).read_bytes())
            return
        if self.path == "/api/catalog":
            self._json(HTTPStatus.OK, self.server.queue.laboratory.catalog())
            return
        if self.path.startswith("/api/runs/"):
            run = self.server.queue.get(self.path.removeprefix("/api/runs/"))
            if run:
                self._json(HTTPStatus.OK, run)
                return
        self._json(HTTPStatus.NOT_FOUND, {"error": "The requested resource was not found."})

    def do_POST(self) -> None:
        if not self._is_local():
            return
        if self.path != "/api/runs":
            self._json(HTTPStatus.NOT_FOUND, {"error": "The requested resource was not found."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if (
                self.headers.get_content_type() != "application/json"
                or self.headers.get("Transfer-Encoding") is not None
                or not 0 < length <= 2048
            ):
                raise ValueError("Expected a JSON request of at most 2048 bytes.")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object.")
            run_id = self.server.queue.submit(payload)
        except (json.JSONDecodeError, UnicodeError):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "The request contains invalid JSON."})
        except ValueError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except LaboratoryError as exc:
            self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
        else:
            self._json(HTTPStatus.ACCEPTED, {"id": run_id})

    def log_message(self, format: str, *args: object) -> None:
        # Run identifiers and request paths are unnecessary in the terminal.
        return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, default=Path("data/ztf_images.zip"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--live-budget-usd", type=float, default=0, help="Enable paid requests within this session budget."
    )
    args = parser.parse_args()
    try:
        if not args.fits.exists():
            raise ValueError("FITS input is missing. Supply the verified ZIP or FITS directory with --fits.")
        laboratory = Laboratory(args.fits, args.live_budget_usd)
        with ThreadPoolExecutor(max_workers=1) as executor:
            with LaboratoryServer(args.port, RunQueue(laboratory, executor)) as server:
                print(f"BlindSpot Laboratory is available at {server.origin}", flush=True)
                server.serve_forever()
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError) as exc:
        print(f"Laboratory startup failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
