"""Record and replay the own client's model calls (architecture stage 96).

Comparing two versions of the agent on the harness measures the change AND the model's run-to-run
variation, and on deepseek the second is larger than the first: stage 94's three trials of T06
cost 0.17 to 0.55 coins on one loop. Paying for more trials averages the noise out slowly. Replay
removes it: record one run's model responses, then run the other version against the recording.
Wherever the two send the same request they get the same answer, at no cost, and the first request
that differs is exactly where the versions' behaviour parts.

The own client (``model_client.ChatClient``) owns the wire, so this lives there and nothing above
it changes. ``AGENT_MODEL_CASSETTE=<dir>`` turns it on and ``AGENT_MODEL_CASSETTE_MODE`` picks:

``record``
    Every call goes to the provider; every successful answer is stored.
``replay``
    A recorded request is answered from the cassette; anything else raises ``ModelCallError``
    ("cassette miss"). No network, no cost. The turn ends at the first divergence.
``replay_or_live``
    A recorded request is answered from the cassette; anything else goes to the provider and is
    NOT written into the cassette, so the recording stays the reference. For comparisons.

What makes a request "the same". The request is compared after removing what differs between two
runs of the same conversation and means nothing to the model's choice: ids minted per run (file,
layer, run, thread and session ids, uuids, long hex digests), temporary paths, timestamps, and
measured durations. Ids are not blanked, they are NUMBERED in order of first appearance, and an
answer is stored with the same numbering: an answer that names "the file from the first tool
result" names the live run's file when it is replayed, not the recorded run's. Everything else
must match byte for byte (sorted keys).

What a replay log says. Every call appends a line to ``<dir>/replay_log.jsonl``: hit or miss, the
turn (``turn_control``), the caller (the start of its system message: the code peer, the
supervisor, the code reviewer), and the message count. ``python -m agent_runtime.model_cassette
report <dir>`` reduces it to, per turn, how many calls were served and where the first miss was.

Limits, stated rather than hidden:

* a tool whose output changes between runs (a live web search, Overpass, a stack sample in the
  stage 78 reviewer's prompt) makes the next request differ: that is a miss, and the log says
  which caller it was;
* a replayed answer carries its recorded usage, so a replay run's cost figures are what the path
  WOULD cost; the real spend is the misses (each replayed call is marked ``replayed``);
* a cassette holds prompts, tool results and user data. It is refused outside ``AGENT_MODE``
  ``local`` / ``dev``, never stores a header, and belongs under ``gis_harness/runs/`` (git-ignored).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

DIR_ENV = "AGENT_MODEL_CASSETTE"
MODE_ENV = "AGENT_MODEL_CASSETTE_MODE"
MODES = ("record", "replay", "replay_or_live")
LOG_NAME = "replay_log.jsonl"

# Tokens numbered by first appearance: ids minted per run.
_TOKEN_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"            # uuid
    r"|\b(?:file|layer|run|toolu|chatcmpl|harness|sess|fault|auto-thread|claudesess)"
    r"[_-][0-9A-Za-z_-]{4,}"                                                    # minted ids
    r"|\b[0-9a-f]{16,}\b"                                                       # digests
    r"|(?:/private)?/(?:var/folders|tmp)/[^\s\"'\\,)\]]+"                      # temp paths
)
# Blanked to a constant: values that differ every run and are never referred back to.
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?")
# Any field whose name says it is a time in seconds or milliseconds: a word for time anywhere in
# the key, or the `_s` / `_ms` suffix. The first live replay missed at every turn's first code
# result on execute_code's `ran_seconds` (1.4 s recorded, 1.3 s live), which a list of exact
# names had left out. A setting like `time_limit_s` is blanked too, harmlessly: it is constant.
_DURATION_RE = re.compile(
    r'(\\?"(?:[a-z_]*(?:seconds|secs|duration|elapsed|took|runtime|wall_time|latency)[a-z_]*'
    r'|[a-z_]+_s|[a-z_]+_ms)\\?"\s*:\s*)-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')
_PLACEHOLDER_RE = re.compile(r"\u27e6id:(\d+)\u27e7")


def _placeholder(n: int) -> str:
    return f"\u27e6id:{n}\u27e7"


def normalize(body: Dict[str, Any]) -> Tuple[str, Dict[str, int]]:
    """The request as compared, and its token numbering (token -> number)."""
    text = json.dumps(body, sort_keys=True, ensure_ascii=False, default=str)
    text = _TIMESTAMP_RE.sub("\u27e6ts\u27e7", text)
    text = _DURATION_RE.sub(lambda m: m.group(1) + "0", text)
    numbering: Dict[str, int] = {}

    def number(m: "re.Match[str]") -> str:
        token = m.group(0)
        if token not in numbering:
            numbering[token] = len(numbering) + 1
        return _placeholder(numbering[token])

    return _TOKEN_RE.sub(number, text), numbering


def request_key(normalized: str) -> str:
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def encode_answer(answer: Any, numbering: Dict[str, int]) -> str:
    """An answer as stored: tokens the REQUEST carried become their numbers; new ones stay."""
    text = json.dumps(answer, ensure_ascii=False, default=str)
    if not numbering:
        return text
    return _TOKEN_RE.sub(lambda m: _placeholder(numbering[m.group(0)])
                         if m.group(0) in numbering else m.group(0), text)


def decode_answer(stored: str, numbering: Dict[str, int]) -> Any:
    """A stored answer for THIS request: numbers become this run's tokens."""
    live = {n: token for token, n in numbering.items()}
    text = _PLACEHOLDER_RE.sub(lambda m: json.dumps(live[int(m.group(1))])[1:-1]
                               if int(m.group(1)) in live else m.group(0), stored)
    return json.loads(text)


def _caller(body: Dict[str, Any]) -> str:
    for m in body.get("messages") or []:
        if isinstance(m, dict) and m.get("role") == "system":
            content = m.get("content")
            text = content if isinstance(content, str) else json.dumps(content)[:200]
            return " ".join(text.split())[:80]
    return ""


def _turn_id() -> Optional[str]:
    try:
        from agent_runtime import turn_control

        handle = turn_control.current()
        return handle.id if handle is not None else None
    except Exception:  # noqa: BLE001
        return None


class CassetteMiss(Exception):
    """Raised inside the client and restated as ModelCallError (strict replay only)."""


class Cassette:
    """One cassette directory: an entry file per request key, answers in recorded order."""

    def __init__(self, directory: Path, mode: str) -> None:
        self.dir = Path(directory)
        self.mode = mode
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._next: Dict[str, int] = defaultdict(int)
        self.hits = 0
        self.misses = 0

    # -- storage ----------------------------------------------------------------------------
    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def _load(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(self._path(key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _log(self, **entry: Any) -> None:
        entry.setdefault("ts", round(time.time(), 3))
        try:
            with open(self.dir / LOG_NAME, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            logger.debug("replay log write failed", exc_info=True)

    def _describe(self, body: Dict[str, Any], key: str) -> Dict[str, Any]:
        messages = body.get("messages") or []
        last = messages[-1] if messages and isinstance(messages[-1], dict) else {}
        return {"key": key, "model": body.get("model"), "turn": _turn_id(),
                "caller": _caller(body), "messages": len(messages),
                "last_role": last.get("role"), "stream": bool(body.get("stream"))}

    def _keep_miss(self, key: str, normalized: str, info: Dict[str, Any]) -> None:
        try:
            misses = self.dir / "misses"
            misses.mkdir(exist_ok=True)
            (misses / f"{key}.json").write_text(
                json.dumps({**info, "request": normalized}, ensure_ascii=False), encoding="utf-8")
        except OSError:
            logger.debug("miss write failed", exc_info=True)

    # -- the two directions -----------------------------------------------------------------
    def replay(self, body: Dict[str, Any], kind: str) -> Optional[Any]:
        """The recorded answer for *body*, or None. Strict replay raises CassetteMiss."""
        if self.mode == "record":
            return None
        normalized, numbering = normalize(body)
        key = request_key(normalized)
        with self._lock:
            entry = self._load(key)
            answers = [a for a in (entry or {}).get("answers", []) if a.get("kind") == kind]
            if answers:
                i = min(self._next[key], len(answers) - 1)
                self._next[key] += 1
                self.hits += 1
                stored = answers[i]["answer"]
            else:
                stored = None
                self.misses += 1
        info = self._describe(body, key)
        if stored is None:
            self._log(hit=False, mode=self.mode, served="live" if self.mode == "replay_or_live"
                      else "refused", **info)
            self._keep_miss(key, normalized, info)
            if self.mode == "replay":
                raise CassetteMiss(f"cassette miss: no recorded answer for request {key} "
                                   f"({info['caller'][:40]!r}, {info['messages']} messages)")
            return None
        self._log(hit=True, mode=self.mode, served="cassette", answer_index=i, **info)
        if kind == "stream":
            return [decode_answer(chunk, numbering) for chunk in stored]
        return decode_answer(stored, numbering)

    def record(self, body: Dict[str, Any], answer: Any, kind: str) -> None:
        if self.mode != "record":
            return
        normalized, numbering = normalize(body)
        key = request_key(normalized)
        stored: Any = ([encode_answer(c, numbering) for c in answer] if kind == "stream"
                       else encode_answer(answer, numbering))
        info = self._describe(body, key)
        with self._lock:
            # The compared text is kept so a later miss can be diffed against it (`report --diff`).
            entry = self._load(key) or {"key": key, "model": body.get("model"),
                                        "caller": info["caller"], "messages": info["messages"],
                                        "request": normalized, "answers": []}
            entry["answers"].append({"kind": kind, "answer": stored,
                                     "recorded_at": round(time.time(), 3)})
            tmp = self._path(key).with_suffix(".tmp")
            tmp.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self._path(key))
        self._log(hit=None, mode="record", served="live", **info)


_ACTIVE: Dict[Tuple[str, str], Cassette] = {}
_ACTIVE_LOCK = threading.Lock()
_REFUSED: List[bool] = []


def active() -> Optional[Cassette]:
    """The cassette the environment selects, or None. Refused outside local and dev mode."""
    directory = (os.getenv(DIR_ENV) or "").strip()
    if not directory:
        return None
    mode = (os.getenv(MODE_ENV) or "replay_or_live").strip().lower()
    if mode not in MODES:
        logger.warning("%s=%r is not one of %s; model calls are not recorded or replayed",
                       MODE_ENV, mode, ", ".join(MODES))
        return None
    try:
        from agent_runtime.deployment_mode import current_mode

        deployment = current_mode()
    except Exception:  # noqa: BLE001 - an unreadable mode is not a reason to record users
        deployment = "unknown"
    if deployment not in ("local", "dev"):
        if not _REFUSED:
            _REFUSED.append(True)
            logger.warning("%s is set but AGENT_MODE=%s: a cassette stores prompts and user data, "
                           "so it is refused outside local and dev mode", DIR_ENV, deployment)
        return None
    with _ACTIVE_LOCK:
        key = (str(Path(directory).resolve()), mode)
        if key not in _ACTIVE:
            _ACTIVE[key] = Cassette(Path(directory), mode)
            logger.info("model cassette: %s, mode %s", directory, mode)
        return _ACTIVE[key]


# --- the report ---------------------------------------------------------------------------------
def summarize(directory: Path) -> Dict[str, Any]:
    """Per turn: calls, served from the cassette, first miss (call index and caller)."""
    turns: Dict[str, Dict[str, Any]] = {}
    path = Path(directory) / LOG_NAME
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("hit") is None:
            continue
        t = turns.setdefault(str(e.get("turn") or "unknown"),
                             {"calls": 0, "hits": 0, "first_miss": None})
        t["calls"] += 1
        if e.get("hit"):
            t["hits"] += 1
        elif t["first_miss"] is None:
            t["first_miss"] = {"call": t["calls"], "caller": e.get("caller"),
                               "messages": e.get("messages")}
    calls = sum(t["calls"] for t in turns.values())
    hits = sum(t["hits"] for t in turns.values())
    return {"turns": turns, "calls": calls, "hits": hits,
            "hit_rate": round(hits / calls, 3) if calls else None}


def renormalize(normalized: str) -> str:
    """A stored compared request under the CURRENT blanking rules. Token numbering is already
    in place, so only the constant blanks are reapplied (idempotent)."""
    text = _TIMESTAMP_RE.sub("\u27e6ts\u27e7", normalized)
    return _DURATION_RE.sub(lambda m: m.group(1) + "0", text)


def rekey(directory: Path) -> Dict[str, int]:
    """Bring a cassette's keys up to the current rules without re-recording: every entry that
    kept its compared request is re-keyed, and entries that now share a key are merged."""
    directory = Path(directory)
    moved = merged = kept = 0
    for path in sorted(directory.glob("*.json")):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if not isinstance(entry.get("request"), str):
            kept += 1
            continue
        text = renormalize(entry["request"])
        key = request_key(text)
        if key == entry.get("key"):
            kept += 1
            continue
        target = directory / f"{key}.json"
        if target.exists():
            other = json.loads(target.read_text(encoding="utf-8"))
            other["answers"] = other.get("answers", []) + entry.get("answers", [])
            other["request"] = text
            target.write_text(json.dumps(other, ensure_ascii=False), encoding="utf-8")
            merged += 1
        else:
            entry.update(key=key, request=text)
            target.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
            moved += 1
        path.unlink()
    return {"rekeyed": moved, "merged": merged, "unchanged": kept}


def first_difference(a: str, b: str, context: int = 160) -> Dict[str, Any]:
    """Where two compared requests part: the offset and the text around it in each."""
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    start = max(0, n - context // 2)
    return {"offset": n, "recorded": a[start:n + context], "live": b[start:n + context]}


def diff_misses(directory: Path) -> List[Dict[str, Any]]:
    """Each miss against the recorded request it shares the longest prefix with."""
    directory = Path(directory)
    recorded = []
    for path in directory.glob("*.json"):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if isinstance(entry.get("request"), str):
            recorded.append(entry)
    out = []
    for path in sorted((directory / "misses").glob("*.json")):
        miss = json.loads(path.read_text(encoding="utf-8"))
        live = miss.get("request") or ""
        best = None
        for entry in recorded:
            d = first_difference(entry["request"], live)
            if best is None or d["offset"] > best[1]["offset"]:
                best = (entry, d)
        if best:
            out.append({"miss": miss.get("key"), "caller": miss.get("caller"),
                        "messages": miss.get("messages"), "nearest": best[0].get("key"),
                        **best[1]})
    return out


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m agent_runtime.model_cassette")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rep = sub.add_parser("report", help="hits and first misses per turn")
    rep.add_argument("directory", type=Path)
    rep.add_argument("--diff", action="store_true",
                     help="for each miss, where it first differs from the nearest recorded request")
    rk = sub.add_parser("rekey", help="re-key a cassette under the current matching rules")
    rk.add_argument("directory", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "rekey":
        print(rekey(a.directory))
        return 0
    if a.diff:
        for d in diff_misses(a.directory):
            print(f"miss {d['miss'][:12]} ({d['messages']} messages, {(d['caller'] or '')[:40]!r}) "
                  f"vs {d['nearest'][:12]}, differs at char {d['offset']}:")
            print(f"  recorded: {d['recorded']!r}")
            print(f"  live:     {d['live']!r}")
        return 0
    s = summarize(a.directory)
    print(f"{s['hits']}/{s['calls']} calls served from the cassette (hit rate {s['hit_rate']})")
    for turn, t in s["turns"].items():
        miss = t["first_miss"]
        where = (f"first miss at call {miss['call']} ({miss['messages']} messages, "
                 f"{(miss['caller'] or '')[:50]!r})" if miss else "no miss")
        print(f"  {turn[:12]}  {t['hits']}/{t['calls']}  {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["Cassette", "CassetteMiss", "DIR_ENV", "LOG_NAME", "MODE_ENV", "MODES", "active",
           "decode_answer", "diff_misses", "encode_answer", "first_difference", "normalize",
           "rekey", "renormalize", "request_key", "summarize"]
