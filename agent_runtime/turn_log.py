"""The turn's event log: one append-only record of what this turn did.

Before this, "what happened this turn" lived in seven places: the peers' result slots, the
search node's `action_rows`, the session ledger (written only at the end of synthesis), each
peer's checkpointed thread, the decision list, the failure lists and the trace. Each consumer
read a different one, and each disagreement between two of them was an incident: turn 2
reporting `has_evidence=False`, a good file_id written onto the failed row, the auditor
flagging GEOID 17019 that `admin_boundary` had returned (docs/design-review-2026-10.md, F3).

The log is written where the work happens: every tool call that goes through a peer's agent
loop is recorded by the middleware every peer already has (`executor_factory`), and anything
that reaches the supervisor by another route (a test double, a CLI peer) is ingested by the
node, deduplicated by tool_call_id. Every other view of the turn is derived from it.

It also answers the two questions the step bounds never could:

* **Has this exact call already been answered?** An identical `(tool, args)` call is answered
  from the log, with no tool execution, unless the world changed in between (a call produced a
  new file) or the earlier answer was a first failure (an outage can recover once).
* **Did this step add anything?** A step is productive when one of its results is new: a
  success whose content has not been seen this turn, or a failure whose error class has not.
  Two unproductive steps in a row end a peer run (`PROGRESS_LIMIT`). A count of steps measures
  how long a loop has run; this measures whether it is going anywhere.

The log lives for one turn. It is found through the state (`turn_log_id`) by nodes, and through
a context variable by the middleware, which the node sets before it invokes a peer.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from collections import OrderedDict
from contextvars import ContextVar
from typing import Any, Dict, Iterable, List, Optional

# Two consecutive steps that add nothing end a peer run. One unproductive step is always
# allowed: a model that reads a result and then decides is not looping.
PROGRESS_LIMIT = 2

# A failed call may be retried with identical arguments once (an Overpass mirror that 504s can
# answer the second time); after that, the failure itself is the answer.
FAILED_RETRIES = 1

_ID_RE = re.compile(r"\b(?:file|layer|run|call|toolu|chatcmpl)[_-][0-9a-zA-Z_-]{4,}\b")
_QUOTED_RE = re.compile(r"(['\"]).*?\1")
_PATH_RE = re.compile(r"(?:/[\w.@-]+){2,}")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def call_key(name: Any, args: Any) -> str:
    """(tool, args) identity, independent of key order."""
    try:
        canon = json.dumps(args, sort_keys=True, ensure_ascii=True, default=str)
    except Exception:  # noqa: BLE001 - an unserialisable arg simply never matches
        canon = repr(args)
    return f"{name}\x00{canon}"


def error_class(text: str) -> str:
    """An error with its particulars removed: two failures to open different paths for the same
    reason are one class, so guessing filenames does not look like progress."""
    t = _PATH_RE.sub("<path>", str(text or ""))
    t = _QUOTED_RE.sub("<q>", t)
    t = _ID_RE.sub("<id>", t)
    t = _NUM_RE.sub("<n>", t)
    return " ".join(t.lower().split())[:240]


def parse_result(content: Any, status: Optional[str] = None) -> Dict[str, Any]:
    """`{ok, error, artifacts}` from a tool result, whatever shape the tool chose.

    Tools have no shared result schema (review §2.1): `ok:false` + `error` by convention,
    `{"error": ..., "count": 0}` from Overpass, `found:false` from geocoding, a ToolMessage with
    status "error" when the tool raised. This reads all of them. `artifacts` are the file and
    layer ids a result introduced, which is what makes a later identical call worth repeating.
    """
    text = content if isinstance(content, str) else json.dumps(content, default=str)
    body: Any = None
    if isinstance(content, dict):
        body = content
    elif isinstance(text, str) and text.lstrip().startswith("{"):
        try:
            body = json.loads(text)
        except ValueError:
            body = None
    ok = True
    error = None
    if status == "error":
        ok, error = False, text
    elif isinstance(body, dict):
        if body.get("ok") is False or (body.get("error") and body.get("ok") is not True):
            ok = False
            error = str(body.get("error") or body.get("stderr") or body.get("message") or text)
        elif body.get("timed_out") or (isinstance(body.get("exit_code"), int)
                                       and body.get("exit_code") != 0):
            ok = False
            error = str(body.get("stderr") or body.get("error") or text)
    elif isinstance(text, str) and re.match(r"\s*(?:Error|Traceback)\b", text):
        ok, error = False, text
    artifacts = sorted(set(re.findall(r"\b(?:file|layer)_[0-9a-f]{8,}\b", text or "")))
    return {"ok": ok, "error": error, "artifacts": artifacts}


class TurnLog:
    """Append-only. Events are dicts with a monotonically increasing `seq`."""

    def __init__(self, log_id: Optional[str] = None) -> None:
        self.id = log_id or uuid.uuid4().hex[:16]
        self.events: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._seen_content: set = set()
        self._seen_errors: set = set()
        self._call_ids: set = set()
        self._artifacts: set = set()
        # Advances when a step adds something new. A memoised answer is valid only while it
        # has not moved since that answer's step: anything new may have changed what the same
        # call would return (a listing after a write, a read after an edit).
        self.world = 0
        self._runs: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ writing

    def append(self, kind: str, **fields: Any) -> Dict[str, Any]:
        with self._lock:
            event = {"seq": len(self.events) + 1, "kind": kind, **fields}
            self.events.append(event)
            return event

    def begin_run(self, peer: str) -> str:
        """A peer invocation. Steps and progress are counted per run."""
        run_id = f"{peer}#{sum(1 for e in self.events if e['kind'] == 'run_started') + 1}"
        with self._lock:
            self._runs[run_id] = {"peer": peer, "steps": 0, "unproductive": 0, "productive": 0,
                                  "pending": []}
        self.append("run_started", peer=peer, run=run_id)
        return run_id

    def end_run(self, run_id: str, **fields: Any) -> None:
        r = self._runs.get(run_id) or {}
        self.append("run_ended", run=run_id, peer=r.get("peer"),
                    productive_steps=r.get("productive", 0), steps=r.get("steps", 0), **fields)

    def record_call(self, *, peer: str, run: Optional[str], name: str, args: Any,
                    call_id: Optional[str]) -> Dict[str, Any]:
        with self._lock:
            if call_id:
                self._call_ids.add(call_id)
        return self.append("tool_call", peer=peer, run=run, tool=name, args=args,
                           key=call_key(name, args), call_id=call_id)

    def record_result(self, *, peer: str, run: Optional[str], name: str, args: Any,
                      call_id: Optional[str], content: Any, status: Optional[str] = None,
                      memo_of: Optional[int] = None) -> Dict[str, Any]:
        """Record a result and say whether it added anything to the turn."""
        parsed = parse_result(content, status)
        text = content if isinstance(content, str) else json.dumps(content, default=str)
        with self._lock:
            if memo_of is not None:
                new = False
            elif parsed["ok"]:
                digest = hashlib.sha1(f"{name}\x00{text}".encode("utf-8", "replace")).hexdigest()
                new = digest not in self._seen_content
                self._seen_content.add(digest)
            else:
                cls = f"{name}\x00{error_class(parsed['error'] or text)}"
                new = cls not in self._seen_errors
                self._seen_errors.add(cls)
            self._artifacts.update(parsed["artifacts"])
            in_step = run in self._runs
            if not in_step and new and parsed["ok"]:
                self.world += 1         # outside a step there is nothing to batch with
            event = self.append("tool_result", peer=peer, run=run, tool=name,
                                key=call_key(name, args), call_id=call_id, ok=parsed["ok"],
                                error=parsed["error"], artifacts=parsed["artifacts"], new=new,
                                world=self.world, memo_of=memo_of, content=text)
            if in_step:
                self._runs[run]["pending"].append(event)
        return event

    def ingest(self, peer: str, result: Any, run: Optional[str] = None) -> int:
        """Record calls a peer reported that the middleware did not see (a test double, a CLI
        peer, a peer built without the default middleware). Matched by tool_call_id."""
        if not isinstance(result, dict):
            return 0
        calls = {c.get("id"): c for c in (result.get("tool_calls") or [])
                 if isinstance(c, dict)}
        added = 0
        for res in result.get("tool_results") or []:
            if not isinstance(res, dict):
                continue
            cid = res.get("tool_call_id")
            if cid and cid in self._call_ids:
                continue
            call = calls.get(cid) or {}
            name = res.get("name") or call.get("name") or "unknown_tool"
            self.record_call(peer=peer, run=run, name=name, args=call.get("args"), call_id=cid)
            self.record_result(peer=peer, run=run, name=name, args=call.get("args"),
                               call_id=cid, content=res.get("content"))
            added += 1
        return added

    def record_needs(self, peer: str, run: Optional[str], needs: Any) -> Optional[Dict]:
        """A request for a capability the peer is missing. Asking for what is missing is
        progress: the peer has said what blocks it, and the supervisor can supply it."""
        items = [f"{n.get('capability')}:{n.get('reason') or ''}" if isinstance(n, dict) else str(n)
                 for n in (needs or [])]
        if not items:
            return None
        digest = hashlib.sha1(("needs\x00" + "|".join(sorted(items))).encode()).hexdigest()
        with self._lock:
            new = digest not in self._seen_content
            self._seen_content.add(digest)
        return self.append("needs", peer=peer, run=run, new=new, ok=True, needs=items)

    def record_answer(self, peer: str, run: Optional[str], text: Any) -> Optional[Dict]:
        """A peer's own answer for a run. New text is an addition to the record: a peer that
        answers from what it already holds has made progress even with no tool call."""
        body = str(text or "").strip()
        if not body:
            return None
        digest = hashlib.sha1(f"answer\x00{body}".encode("utf-8", "replace")).hexdigest()
        with self._lock:
            new = digest not in self._seen_content
            self._seen_content.add(digest)
        return self.append("peer_answer", peer=peer, run=run, new=new, ok=True,
                           content=body[:4000])

    # ------------------------------------------------------------------ progress

    def close_step(self, run: Optional[str]) -> Optional[bool]:
        """Called before each model call of a run: was the step that just finished productive?

        None when the previous model call made no tool call (there was no step to judge).
        """
        r = self._runs.get(run or "")
        if not r or not r["pending"]:
            return None
        pending, r["pending"] = r["pending"], []
        productive = any(e.get("new") for e in pending)
        with self._lock:
            if any(e.get("new") and e.get("ok") for e in pending):
                self.world += 1
            # Siblings in one step share the world they produced, so a call re-issued in the
            # next step finds its earlier answer still valid.
            for e in pending:
                e["world"] = self.world
        r["steps"] += 1
        if productive:
            r["productive"] += 1
            r["unproductive"] = 0
        else:
            r["unproductive"] += 1
        self.append("step", run=run, peer=r["peer"], productive=productive,
                    consecutive_unproductive=r["unproductive"])
        return productive

    def exhausted(self, run: Optional[str]) -> bool:
        r = self._runs.get(run or "")
        return bool(r) and r["unproductive"] >= PROGRESS_LIMIT

    def run_was_productive(self, run_id: str) -> bool:
        """A run is productive when it added a new successful result, a new answer, or a new
        request for what it is missing."""
        return any(e["kind"] in ("tool_result", "peer_answer", "needs") and e.get("run") == run_id
                   and e.get("new") and e.get("ok") for e in self.events)

    def last_run(self, peer: str) -> Optional[str]:
        for e in reversed(self.events):
            if e["kind"] == "run_started" and e.get("peer") == peer:
                return e["run"]
        return None

    # ------------------------------------------------------------------ memo

    def earlier_answer(self, key: str) -> Optional[Dict[str, Any]]:
        """The result an identical call should get instead of running, or None to run it.

        Valid only while the world is unchanged since that result: a listing after a write, or
        a read after an edit, must run again. A failure is retried `FAILED_RETRIES` times.
        """
        with self._lock:
            real = [e for e in self.events
                    if e["kind"] == "tool_result" and e.get("key") == key
                    and e.get("memo_of") is None]
        if not real:
            return None
        last = real[-1]
        if last.get("world") != self.world:
            return None
        if not last.get("ok"):
            failures = [e for e in real if not e.get("ok")]
            if len(failures) <= FAILED_RETRIES:
                return None
        return last

    # ------------------------------------------------------------------ views

    def results(self, peer: Optional[str] = None) -> List[Dict[str, Any]]:
        return [e for e in self.events if e["kind"] == "tool_result"
                and (peer is None or e.get("peer") == peer)]

    def calls(self, peer: Optional[str] = None) -> List[Dict[str, Any]]:
        return [e for e in self.events if e["kind"] == "tool_call"
                and (peer is None or e.get("peer") == peer)]

    def artifacts_view(self, peers: Optional[Iterable[str]] = None) -> Dict[str, List[Dict]]:
        """The `{tool_calls, tool_results}` shape every existing consumer reads, from the log.

        Memoised answers are left out: they repeat a result already in the view, and counting
        them twice is how a failure becomes "repeated".
        """
        wanted = set(peers) if peers is not None else None
        calls, results = [], []
        memo_ids = {e.get("call_id") for e in self.events
                    if e["kind"] == "tool_result" and e.get("memo_of") is not None}
        for e in self.events:
            if wanted is not None and e.get("peer") not in wanted:
                continue
            if e.get("call_id") in memo_ids:
                continue
            if e["kind"] == "tool_call":
                calls.append({"name": e["tool"], "args": e.get("args"), "id": e.get("call_id")})
            elif e["kind"] == "tool_result":
                results.append({"name": e["tool"], "tool_call_id": e.get("call_id"),
                                "content": e.get("content")})
        return {"tool_calls": calls, "tool_results": results}

    def summary(self) -> Dict[str, Any]:
        res = self.results()
        return {"events": len(self.events), "tool_results": len(res),
                "memo_hits": sum(1 for e in res if e.get("memo_of") is not None),
                "new_results": sum(1 for e in res if e.get("new")),
                "failed": sum(1 for e in res if not e.get("ok")),
                "runs": [{"run": e["run"], "productive_steps": e.get("productive_steps"),
                          "steps": e.get("steps"), "stopped": e.get("stopped")}
                         for e in self.events if e["kind"] == "run_ended"]}


# --------------------------------------------------------------------------- registry

_LOGS: "OrderedDict[str, TurnLog]" = OrderedDict()
_LOGS_LOCK = threading.Lock()
_MAX_LOGS = 200

_ACTIVE: ContextVar[Optional[TurnLog]] = ContextVar("turn_log", default=None)
_ACTIVE_RUN: ContextVar[Optional[str]] = ContextVar("turn_log_run", default=None)
_ACTIVE_PEER: ContextVar[Optional[str]] = ContextVar("turn_log_peer", default=None)
_ACTIVE_BRIEF: ContextVar[Optional[str]] = ContextVar("turn_log_brief", default=None)


def new_log() -> TurnLog:
    log = TurnLog()
    with _LOGS_LOCK:
        _LOGS[log.id] = log
        while len(_LOGS) > _MAX_LOGS:
            _LOGS.popitem(last=False)
    return log


def get_log(log_id: Optional[str]) -> Optional[TurnLog]:
    if not log_id:
        return None
    with _LOGS_LOCK:
        return _LOGS.get(log_id)


def log_for_state(state: Any) -> TurnLog:
    """The state's log, creating one when the graph was invoked without `run_supervisor`."""
    log = get_log((state or {}).get("turn_log_id") if isinstance(state, dict) else None)
    return log if log is not None else new_log()


def active() -> Optional[TurnLog]:
    return _ACTIVE.get()


def active_run() -> Optional[str]:
    return _ACTIVE_RUN.get()


def active_peer() -> Optional[str]:
    return _ACTIVE_PEER.get()


def active_brief() -> Optional[str]:
    """The task and plan rendered for the peer's system message (see executor_factory)."""
    return _ACTIVE_BRIEF.get()


class bind:
    """`with bind(log, peer, brief):` - the middleware inside records into *log* under a new
    run, and renders *brief* (the task and the plan) into every model call's system message."""

    def __init__(self, log: TurnLog, peer: str, brief: Optional[str] = None) -> None:
        self.log, self.peer, self.brief = log, peer, brief
        self.run: Optional[str] = None

    def __enter__(self) -> "bind":
        self.run = self.log.begin_run(self.peer)
        self._tokens = (_ACTIVE.set(self.log), _ACTIVE_RUN.set(self.run),
                        _ACTIVE_PEER.set(self.peer), _ACTIVE_BRIEF.set(self.brief))
        return self

    def __exit__(self, *exc: Any) -> None:
        self.log.close_step(self.run)
        self.log.end_run(self.run, stopped=self.log.exhausted(self.run))
        for var, tok in zip((_ACTIVE, _ACTIVE_RUN, _ACTIVE_PEER, _ACTIVE_BRIEF), self._tokens):
            var.reset(tok)
