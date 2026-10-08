"""Talk to a running agent server the way the map UI does: upload, then one streamed turn.

The harness goes through the HTTP API on purpose. The 2026-10-08 failures lived in the
supervisor, the audit and the banners, the layers a user sees through `/agent/chat/stream`,
and an in-process call that skipped `api/server.py` would skip some of them. The request
carries the server's defaults (memory, MCP tools, code execution) and sets only the model,
a thread id, and `agentDev` so the stream includes tool calls and token usage.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests


def upload(base_url: str, thread_id: str, files: Dict[str, Path]) -> List[str]:
    ids = []
    for name, path in files.items():
        with open(path, "rb") as fh:
            r = requests.post(f"{base_url}/agent/files/upload",
                              files={"file": (name, fh)}, data={"threadId": thread_id},
                              timeout=120)
        r.raise_for_status()
        ids.extend(f["file_id"] for f in r.json()["files"])
    return ids


def _sse_events(resp: requests.Response) -> Iterator[Tuple[str, Dict[str, Any]]]:
    event, data = None, []
    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.rstrip("\r")
        if not line:
            if event and data:
                try:
                    yield event, json.loads("\n".join(data))
                except json.JSONDecodeError:
                    yield event, {"_raw": "\n".join(data)}
            event, data = None, []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())


def run_turn(base_url: str, query: str, *, provider: str, model: str,
             file_ids: Optional[List[str]] = None, thread_id: Optional[str] = None,
             timeout_s: float = 1200.0, raw_log: Optional[Path] = None) -> Dict[str, Any]:
    """Stream one turn and reduce it to what the scorer reads. Raw events go to `raw_log`."""
    thread_id = thread_id or f"harness-{uuid.uuid4().hex[:12]}"
    body = {"userQuery": query, "threadId": thread_id, "agentDev": True,
            "provider": provider, "model": model}
    if file_ids:
        body["fileIds"] = file_ids
    turn: Dict[str, Any] = {"thread_id": thread_id, "provider": provider, "model": model,
                            "answer": None, "tool_calls": [], "tool_results": [],
                            "tool_errors": [], "usage": [], "map_layers": 0, "events": 0,
                            "route": None, "error": None, "audit_severity": None}
    t0 = time.monotonic()
    log = open(raw_log, "w") if raw_log else None
    try:
        with requests.post(f"{base_url}/agent/chat/stream", json=body, stream=True,
                           timeout=(15, 600)) as resp:
            resp.raise_for_status()
            for name, data in _sse_events(resp):
                turn["events"] += 1
                if log:
                    log.write(json.dumps({"event": name, "data": data}, default=str) + "\n")
                _reduce(turn, name, data)
                if time.monotonic() - t0 > timeout_s:
                    turn["error"] = f"harness timeout after {timeout_s:.0f}s"
                    break
    except Exception as exc:  # noqa: BLE001
        turn["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if log:
            log.close()
    turn["seconds"] = round(time.monotonic() - t0, 1)
    return turn


def _reduce(turn: Dict[str, Any], name: str, data: Dict[str, Any]) -> None:
    kind = data.get("type")
    detail = data.get("detail") if isinstance(data.get("detail"), dict) else {}
    if name in ("search", "analysis", "code") and kind in ("tool_call", "tool_result", "tool_error"):
        if kind == "tool_call":
            turn["tool_calls"].append({"name": detail.get("name"), "args": detail.get("args"),
                                       "agent": data.get("agent")})
        elif kind == "tool_result":
            turn["tool_results"].append({"name": detail.get("tool_name") or detail.get("name"),
                                         "content": detail.get("content")})
            if (detail.get("tool_name") or detail.get("name")) == "execute_code":
                gate = _gate_verdict(detail.get("content"))
                turn.setdefault("gate", []).append(gate)
        else:
            turn["tool_errors"].append(detail)
    elif name == "agent_trace" and kind == "llm_usage":
        turn["usage"].append({k: detail.get(k) for k in
                              ("model", "input_tokens", "output_tokens", "cached_input_tokens",
                               "reasoning_tokens", "usage")})
    elif name == "map_layer":
        turn["map_layers"] += 1
    elif name == "result":
        turn["answer"] = data.get("answer")
        rt = data.get("routeTrace") or {}
        turn["route"] = rt.get("supervisor_actions")
        turn["audit_severity"] = rt.get("audit_severity")
    elif name == "answer" and kind == "result" and not turn["answer"]:
        turn["answer"] = data.get("answer")
    elif name == "error":
        turn["error"] = str(data.get("error"))[:2000]


def _gate_verdict(content: Any) -> Dict[str, Any]:
    """The invariant gate's verdict on one execute_code run, and the checks behind it."""
    import re

    text = str(content or "").replace('\\"', '"')
    m = re.search(r'"verification": \{"verdict": "(\w+)"', text)
    if not m:
        return {"verdict": "none" if '"verification": {}' in text else "absent", "checks": []}
    checks = re.findall(r'"check": "(\w+)", "status": "(?:cannot_determine|fail)"', text)
    return {"verdict": m.group(1), "checks": sorted(set(checks))}
