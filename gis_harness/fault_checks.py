"""Fault injection on a live local agent server: a stalled model call, a slow 500, and Stop.

Stages 85, 88 and 94 each claim to bound a failure that a normal harness run never shows: in
the 683 archived harness turns there are no model-call timeouts and no Stops. So this module
makes those failures happen, on purpose, in a real server turn, and measures what the server
does with them.

How. A proxy sits between the agent server and Lumen (``LUMEN_URL`` points at it). Every
request is forwarded to the real Lumen unchanged, except the CODE PEER's model calls, which are
recognised by a tool only that peer binds (``stage_url``; the harness server sets
``AGENT_EXTRACTION=1``, which binds it). Those are, by scenario:

* ``stall``: accepted and never answered. The turn should end at the request timeout
  (``AGENT_LLM_TIMEOUT_SECONDS``) times the attempts (``AGENT_LLM_MAX_RETRIES`` + 1).
* ``error500``: answered with HTTP 500 after a delay, the shape of Lumen's "Upstream error"
  (2026-10-09). The call should be tried twice, not three times.
* ``stop``: stalled, and the person presses Stop (``/agent/chat/cancel``) three seconds into the
  stalled call. Measured: when the proxy sees the connection close, and whether any model
  request arrives AFTER the Stop (a call that is still made, and charged, after the person
  stopped the turn).

Each scenario runs on two configurations of the same commit: ``current`` (the default
``ChatOpenAI`` client and the ``create_agent`` loop) and ``redesign`` (``AGENT_MODEL_CLIENT=own``,
``AGENT_CODE_LOOP=own``). Everything else, including the query, is the same.

    python -m gis_harness.fault_checks --label fault-checks-<sha>

The run costs a few Lumen calls per scenario (the supervisor, the synthesis): the code peer's
own calls never reach Lumen. Results go to ``gis_harness/runs/<label>/fault_checks.json`` and a
table to stdout. Requests are forwarded with the server's own Authorization header; the proxy
never logs a header or a body.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

from gis_harness.client import _reduce, _sse_events
from gis_harness.run import HERE, REPO, start_server

QUERY = ("Write and run Python code that computes the sum of the first 30 Fibonacci numbers "
         "and prints it.")
CODE_PEER_MARKER = "stage_url"
CONFIGS = {
    "current": {"AGENT_MODEL_CLIENT": "langchain", "AGENT_CODE_LOOP": "langchain"},
    "redesign": {"AGENT_MODEL_CLIENT": "own", "AGENT_CODE_LOOP": "own"},
}
SCENARIOS = {
    # name: (proxy mode, AGENT_LLM_TIMEOUT_SECONDS, error delay s, press Stop)
    "stall": ("stall", 20, 0, False),
    "error500": ("error500", 20, 8, False),
    "stop": ("stall", 45, 0, True),
}
_HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "content-encoding"}


def _lumen_upstream() -> str:
    """The real Lumen base URL: the LUMEN_URL a server here would read (the nearest .env), or
    the built-in default. Only the URL is read from the file."""
    from agent_runtime.executor_factory import LUMEN_DEFAULT_URL, normalize_openai_base_url

    value = os.environ.get("LUMEN_URL")
    if not value:
        for parent in [REPO, *REPO.parents]:
            env = parent / ".env"
            if env.is_file():
                from dotenv import dotenv_values

                value = dotenv_values(env).get("LUMEN_URL")
                break
    return normalize_openai_base_url(value or LUMEN_DEFAULT_URL).rstrip("/")


class Proxy:
    def __init__(self, upstream: str, mode: str, delay: float) -> None:
        self.upstream = upstream
        self.mode = mode
        self.delay = delay
        self.t0 = time.monotonic()
        self.requests: List[Dict[str, Any]] = []
        self.code_inflight = threading.Event()
        self._lock = threading.Lock()
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802
                proxy._forward(self, None)

            def do_POST(self) -> None:  # noqa: N802
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                proxy._post(self, raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def now(self) -> float:
        return round(time.monotonic() - self.t0, 2)

    def close(self) -> None:
        self.server.shutdown()

    def _record(self, **entry: Any) -> Dict[str, Any]:
        with self._lock:
            entry["n"] = len(self.requests) + 1
            self.requests.append(entry)
        return entry

    def _post(self, h: BaseHTTPRequestHandler, raw: bytes) -> None:
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            body = {}
        tools = [((t or {}).get("function") or {}).get("name") for t in body.get("tools") or []]
        code = CODE_PEER_MARKER in tools
        if not code or self.mode == "pass":
            self._forward(h, raw, peer="code" if code else "other")
            return
        entry = self._record(peer="code", start=self.now(), action=self.mode, gone=None)
        self.code_inflight.set()
        if self.mode == "stall":
            entry["gone"] = self._wait_for_hangup(h, 900)
            return
        gone = self._wait_for_hangup(h, self.delay)
        if gone is not None:
            entry["gone"] = gone
            return
        payload = json.dumps({"error": {"message": "Upstream error (injected by fault_checks)",
                                        "type": "server_error"}}).encode()
        try:
            h.send_response(500)
            h.send_header("Content-Type", "application/json")
            h.send_header("Content-Length", str(len(payload)))
            h.end_headers()
            h.wfile.write(payload)
        except OSError:
            pass
        entry["answered"] = self.now()

    def _wait_for_hangup(self, h: BaseHTTPRequestHandler, seconds: float) -> Optional[float]:
        """Hold the request; return when the client closed the connection, or None at *seconds*."""
        sock = h.connection
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            readable, _, _ = select.select([sock], [], [], 0.2)
            if readable:
                try:
                    if sock.recv(1, socket.MSG_PEEK) == b"":
                        return self.now()
                except OSError:
                    return self.now()
        return None

    def _forward(self, h: BaseHTTPRequestHandler, raw: Optional[bytes], peer: str = "other") -> None:
        entry = self._record(peer=peer, start=self.now(), action="forward", path=urlparse(h.path).path)
        suffix = h.path[len("/v1"):] if h.path.startswith("/v1") else h.path
        headers = {k: v for k, v in h.headers.items()
                   if k.lower() in ("authorization", "content-type", "accept")}
        try:
            r = requests.request("POST" if raw is not None else "GET", self.upstream + suffix,
                                 data=raw, headers=headers, timeout=300)
            content = r.content
            h.send_response(r.status_code)
            for k, v in r.headers.items():
                if k.lower() not in _HOP:
                    h.send_header(k, v)
            h.send_header("Content-Length", str(len(content)))
            h.end_headers()
            h.wfile.write(content)
            entry.update(status=r.status_code, answered=self.now())
        except Exception as exc:  # noqa: BLE001 - reported, never raised into the server
            entry.update(status=None, error=type(exc).__name__)
            try:
                h.send_response(502)
                h.end_headers()
            except OSError:
                pass


def run_scenario(config: str, scenario: str, port: int, out: Path, upstream: str) -> Dict[str, Any]:
    mode, timeout_s, delay, press_stop = SCENARIOS[scenario]
    proxy = Proxy(upstream, mode, delay)
    env = {**CONFIGS[config], "LUMEN_URL": proxy.url, "AGENT_LLM_PROVIDER": "lumen",
           "AGENT_LLM_TIMEOUT_SECONDS": str(timeout_s), "AGENT_LLM_MAX_RETRIES": "1"}
    log_path = out / f"{config}-{scenario}.server.log"
    proc = start_server(port, log_path, env)
    base = f"http://localhost:{port}"
    turn: Dict[str, Any] = {"answer": None, "tool_calls": [], "tool_results": [], "tool_errors": [],
                            "usage": [], "map_layers": 0, "events": 0, "route": None,
                            "error": None, "audit_severity": None}
    result: Dict[str, Any] = {"config": config, "scenario": scenario, "timeout_s": timeout_s,
                              "retries": 1, "error_delay_s": delay or None}
    turn_id: Dict[str, Optional[str]] = {"id": None}
    stop_at: Dict[str, Optional[float]] = {"t": None, "status": None}

    def stopper() -> None:
        if not proxy.code_inflight.wait(600):
            return
        time.sleep(3.0)
        for _ in range(50):
            if turn_id["id"]:
                break
            time.sleep(0.1)
        stop_at["t"] = proxy.now()
        try:
            r = requests.post(f"{base}/agent/chat/cancel", json={"turnId": turn_id["id"]}, timeout=10)
            stop_at["status"] = r.status_code
        except Exception as exc:  # noqa: BLE001
            stop_at["status"] = type(exc).__name__

    if press_stop:
        threading.Thread(target=stopper, daemon=True).start()
    raw = open(out / f"{config}-{scenario}.events.jsonl", "w")
    try:
        body = {"userQuery": QUERY, "threadId": f"fault-{config}-{scenario}-{int(time.time())}",
                "agentDev": True, "provider": "lumen", "model": "deepseek-v4-flash"}
        with requests.post(f"{base}/agent/chat/stream", json=body, stream=True,
                           timeout=(15, 900)) as resp:
            for name, data in _sse_events(resp):
                if name == "_keepalive":
                    continue
                raw.write(json.dumps({"event": name, "data": data}, default=str) + "\n")
                if name == "turn" and not turn_id["id"]:
                    turn_id["id"] = data.get("turn_id") or data.get("turnId")
                turn["events"] += 1
                _reduce(turn, name, data)
        result["stream_ended"] = proxy.now()
        if press_stop:
            # A call made after the Stop arrives at the proxy within the old timeout and its
            # retry; watch long enough to see it.
            time.sleep(timeout_s * 2 + 10)
    except Exception as exc:  # noqa: BLE001
        result["client_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        raw.close()
        proc.terminate()
        try:
            proc.wait(20)
        except Exception:  # noqa: BLE001
            proc.kill()
        proxy.close()
    code = [r for r in proxy.requests if r["peer"] == "code"]
    result.update(
        code_peer_reached=bool(code),
        code_requests=len(code),
        code_held_s=[round((r.get("gone") or r.get("answered") or proxy.now()) - r["start"], 1)
                     for r in code],
        model_requests=len([r for r in proxy.requests if r.get("path", "/chat").endswith("/chat/completions")
                            or r["peer"] == "code"]),
        answer=(turn.get("answer") or "")[:400],
        turn_error=turn.get("error"),
    )
    if press_stop:
        t = stop_at["t"]
        result["stop_pressed_at"] = t
        result["stop_status"] = stop_at["status"]
        after = [r for r in proxy.requests if t is not None and r["start"] > t
                 and (r["peer"] == "code" or str(r.get("path", "")).endswith("/chat/completions"))]
        result["model_requests_after_stop"] = len(after)
        first = next((r for r in code if r["start"] <= (t or 0)), None)
        result["stalled_call_closed_after_stop_s"] = (
            round(first["gone"] - t, 2) if first and first.get("gone") and t is not None else None)
        result["stream_ended_after_stop_s"] = (
            round(result["stream_ended"] - t, 2) if t is not None and "stream_ended" in result else None)
    result["seconds"] = result.get("stream_ended")
    return result


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default=time.strftime("fault-checks-%Y%m%d-%H%M%S"))
    ap.add_argument("--port", type=int, default=5431, help="first port; one per scenario")
    ap.add_argument("--configs", default="current,redesign")
    ap.add_argument("--scenarios", default="stall,error500,stop")
    a = ap.parse_args(argv)
    out = HERE / "runs" / a.label
    out.mkdir(parents=True, exist_ok=True)
    upstream = _lumen_upstream()
    print(f"upstream host: {urlparse(upstream).hostname}", flush=True)
    results = []
    port = a.port
    for scenario in a.scenarios.split(","):
        for config in a.configs.split(","):
            r = run_scenario(config, scenario, port, out, upstream)
            port += 1
            results.append(r)
            print(json.dumps(r, default=str), flush=True)
            (out / "fault_checks.json").write_text(json.dumps(results, indent=2, default=str))
    print("\nconfig    scenario  code reqs  held (s)            seconds  after Stop")
    for r in results:
        stop = ""
        if r["scenario"] == "stop":
            stop = (f"call closed +{r.get('stalled_call_closed_after_stop_s')} s, "
                    f"{r.get('model_requests_after_stop')} model request(s) after Stop")
        print(f"{r['config']:9s} {r['scenario']:9s} {r['code_requests']:>9}  "
              f"{str(r['code_held_s']):18s}  {str(r.get('seconds')):>7}  {stop}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
