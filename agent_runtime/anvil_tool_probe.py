"""Which AnvilGPT models can make a structured tool call. Measured, not declared.

This agent binds function tools on every step, so a model that cannot return `tool_calls` cannot
run a single turn. Purdue configures tool calling per model, and its catalogue says nothing about
it. Swept 2026-10-04 with one tool and `tool_choice` left to the server, as the agent sends it:
qwen3:4b and qwen2.5:7b answer HTTP 400 in 0.2 s, "\"auto\" tool choice requires
--enable-auto-tool-choice and --tool-call-parser to be set". No request parameter avoids it:
`"required"` and a named function are refused the same way, for the same missing flag. On
2026-10-02, qwen3:32b and qwen3-vl:32b took the tools and answered without a structured call.

So the picker asks each model once and caches the answer. Only a DEFINITIVE answer is cached:
- a structured tool call: the model can drive the agent;
- a refusal that names tool calling (vLLM's missing parser, Ollama's "does not support tools");
- two answers in a row without a structured call.

Everything else is a failure to ask, not an answer, and hides nothing: a timeout, a 5xx, a 429,
an auth error, and a 400 that does not name tool calling. That last one matters. In the same
sweep, Purdue's Ollama backend was down. LiteLLM reported each of its seven models as HTTP 400,
"litellm.APIConnectionError: OllamaException - Cannot connect to host ...", after 14-73 s, and
three more timed out at 75 s. Reading every 400 as "cannot call tools" would have hidden ten
working models during an outage. The Anthropic list makes the same distinction for the same
reason: availability flaps by the minute, and a cached liveness verdict is wrong both ways.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

# One request per model, bounded. Without a ceiling a dead backend holds a probe thread for as
# long as the server keeps the socket open. There is no max_tokens: a reasoning model spends its
# first tokens thinking, and a ceiling would cut it off before the call and read as "no call".
PROBE_TIMEOUT_S = 45.0
# Purdue changes a model's serving flags on the scale of days, so a definitive answer is kept for
# hours. A model that gains tool calling reappears within this long, and one that loses it goes.
VERDICT_TTL_S = 6 * 3600.0
# After a failed probe, wait before asking again rather than piling probes onto a backend that is
# down. Any earlier definitive answer stays in force meanwhile.
RETRY_AFTER_S = 10 * 60.0
# Enough for every model at once. With 8, measured against the live roster of 16 during an
# Ollama outage: the slots went to probes that hung for up to the timeout, and the refusals that
# take 0.2 s queued behind them, so the first catalogue hid nothing. This is still a bound, for
# a roster that grows.
_WORKERS = 32

# The refusals that are about tool calling itself. vLLM: '"auto" tool choice requires
# --enable-auto-tool-choice and --tool-call-parser to be set', 'tool_choice="required" requires
# --tool-call-parser to be set'. Ollama: '... does not support tools'.
_REFUSAL = re.compile(r"tool[ _-]?choice|tool-call-parser|does not support tools", re.I)

_TOOL = {"type": "function", "function": {
    "name": "lookup_code",
    "description": "Return the access code stored under a key. The code cannot be known any "
                   "other way.",
    "parameters": {"type": "object", "properties": {"key": {"type": "string"}},
                   "required": ["key"]}}}
_ASK = [{"role": "user", "content": "What is the access code stored under the key 'alpha'? "
                                    "Use the lookup_code tool."}]


@dataclass(frozen=True)
class Verdict:
    """``tools`` is True or False once a model has answered definitively, None until then."""
    tools: Optional[bool]
    reason: str


@dataclass
class _Entry:
    tools: Optional[bool] = None
    reason: str = ""
    next_probe_at: float = 0.0


_LOCK = threading.Lock()
_ENTRIES: Dict[Tuple[str, str], _Entry] = {}
_INFLIGHT: Dict[Tuple[str, str], threading.Event] = {}
# Daemon threads under a semaphore rather than a ThreadPoolExecutor: an executor's threads are
# joined at interpreter exit, so a probe in flight against a dead backend would hold a container
# stop for up to PROBE_TIMEOUT_S. A probe is worth nothing once the process is going away.
_SLOTS = threading.BoundedSemaphore(_WORKERS)


def classify(status: int, text: str) -> Tuple[str, str]:
    """One probe response as ``(kind, reason)``. ``kind`` is "call", "refused", "no_call" or
    "failed". Only "failed" is not an answer about tool calling."""
    text = text or ""
    if status == 200:
        try:
            message = json.loads(text)["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError):
            return "failed", "unreadable 200 response"
        if message.get("tool_calls"):
            return "call", "it made a structured tool call"
        return "no_call", "it answered a tool-call request in plain text"
    if status in (400, 422) and _REFUSAL.search(text):
        # Read by people, in the picker and in a notice, so in words. The raw body is logged.
        if "tool-call-parser" in text:
            return "refused", ("AnvilGPT refuses tool calls for it (its server runs without a "
                               "tool-call parser)")
        if "does not support tools" in text:
            return "refused", "AnvilGPT says it does not support tools"
        return "refused", "AnvilGPT refuses tool calls for it"
    return "failed", f"HTTP {status}"


def _post(base_url: str, api_key: str, model: str) -> Tuple[int, str]:
    import requests

    resp = requests.post(f"{base_url}/chat/completions",
                         headers={"Authorization": f"Bearer {api_key}"},
                         json={"model": model, "messages": _ASK, "tools": [_TOOL]},
                         timeout=PROBE_TIMEOUT_S)
    return resp.status_code, resp.text


def probe(base_url: str, api_key: str, model: str,
          post: Optional[Callable[[str, str, str], Tuple[int, str]]] = None
          ) -> Tuple[Optional[bool], str]:
    """Ask ``model`` for one tool call. ``(True|False, reason)`` is an answer; ``(None, reason)``
    means the question could not be asked. An answer without a call is asked once more before it
    counts, since a model able to call may still, once, decide to reply in text."""
    post = post or _post
    for attempt in (1, 2):
        try:
            status, text = post(base_url, api_key, model)
        except Exception as exc:  # noqa: BLE001 - a timeout or a reset is not an answer
            return None, f"probe failed: {type(exc).__name__}"
        kind, reason = classify(status, text)
        if kind == "refused":
            logger.info("AnvilGPT refused tool calls for %s: HTTP %s %s", model, status,
                        re.sub(r"\s+", " ", text or "")[:300])
        if kind == "no_call" and attempt == 1:
            continue
        if kind == "no_call":
            return False, f"{reason}, twice"
        return {"call": True, "refused": False}.get(kind), reason
    return None, "unreachable"  # pragma: no cover - the loop always returns


def _record(key: Tuple[str, str], tools: Optional[bool], reason: str) -> None:
    now = time.time()
    with _LOCK:
        entry = _ENTRIES.setdefault(key, _Entry())
        _INFLIGHT.pop(key, None)
        if tools is None:
            # Not an answer: keep whatever the last answer was, and ask again later.
            entry.next_probe_at = now + RETRY_AFTER_S
        else:
            entry.tools, entry.reason = tools, reason
            entry.next_probe_at = now + VERDICT_TTL_S
    level = logging.INFO if tools is not None else logging.DEBUG
    logger.log(level, "AnvilGPT tool probe: %s -> %s (%s)", key[1],
               {True: "offered", False: "hidden", None: "no verdict"}[tools], reason)


def _run(key: Tuple[str, str], api_key: str, post: Callable, done: threading.Event) -> None:
    try:
        with _SLOTS:
            tools, reason = probe(key[0], api_key, key[1], post)
    except Exception as exc:  # noqa: BLE001 - a background probe must never die silently
        tools, reason = None, f"probe crashed: {type(exc).__name__}"
    _record(key, tools, reason)
    done.set()


def tool_support(models: Iterable[str], *, base_url: str, api_key: str, wait: float = 5.0,
                 schedule: bool = True,
                 post: Optional[Callable[[str, str, str], Tuple[int, str]]] = None
                 ) -> Dict[str, Verdict]:
    """What is known about each model, after starting probes for the ones due one.

    Waits up to ``wait`` seconds for the probes of models it has never asked, so the fast
    answers land in this response, while a dead backend takes up to PROBE_TIMEOUT_S. A refusal
    takes 0.2 s on its own, but 2.7 s when the whole roster of 16 is asked at once (measured on
    a fresh process), which a 2.5 s wait missed. So only the first catalogue after a restart, or
    after a new model appears, waits. A re-probe, after a verdict expires or a probe failed,
    runs in the background while the last answer stands; otherwise an outage would make one
    page load in every RETRY_AFTER_S wait out the timeout. Probes still running are answered as
    unknown, and a later call sees their verdict. ``schedule=False`` reports only what is known.
    """
    started = []
    now = time.time()
    keys = [(base_url, m) for m in dict.fromkeys(models) if m]
    with _LOCK:
        if schedule:
            for key in keys:
                entry = _ENTRIES.get(key)
                if key in _INFLIGHT or (entry is not None and entry.next_probe_at > now):
                    continue
                done = _INFLIGHT[key] = threading.Event()
                # Started here, in the serving process, so nothing exists before a fork.
                threading.Thread(target=_run, args=(key, api_key, post, done), daemon=True,
                                 name=f"anvil-tool-probe:{key[1]}").start()
                if entry is None:
                    started.append(done)
    deadline = time.time() + max(0.0, wait)
    for done in started:
        done.wait(max(0.0, deadline - time.time()))
    out: Dict[str, Verdict] = {}
    with _LOCK:
        for key in keys:
            entry = _ENTRIES.get(key)
            out[key[1]] = (Verdict(entry.tools, entry.reason or "no verdict yet") if entry
                           else Verdict(None, "not probed yet"))
    return out


def _reset_for_tests() -> None:
    with _LOCK:
        _ENTRIES.clear()
        _INFLIGHT.clear()
