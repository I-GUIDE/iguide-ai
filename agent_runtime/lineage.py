"""Where every file, tool result and number came from, recorded as the work happens (stage 56).

The Sources line and the grounding audit used to learn provenance from one tool result at a time.
They broke where one result was built on another. Measured on the validator benchmark's 462
replayable turns (stage 52), 14 sources dropped off a Sources line because a code run reread
files that an earlier call had fetched. The figure resolved to the rereading run, and the fetch's
source was never reached: in `threeway-bridge-fix4` T08L the Planetary Computer vanished and the
line named only TIGERweb. The audit called the Sentinel-2 rule "(DN-1000)/10000" unsupported,
because the rule came from a note the code run was shown, and no record held it.

Provenance in prose can be faked. One planted line defeated a naive provenance check on 94-99%
of runs (AgentDojo/STAC, arXiv 2610.09793), and binding provenance to the lookup that produced
it closed the hole. So every edge here is written by the code that performed the action. The
model's prose never writes one.

The graph, per conversation:

* nodes: ``source`` (a catalogued service, an unknown host, an upload), ``artifact`` (a file_id,
  with its content digest), ``call`` (a tool call or code run), ``fact`` (a number a result
  reported; its ``kind`` slot is step 3's to fill), ``note`` (a tool description, a catalogue
  entry, a service note such as the code bridge's host notes);
* edges: ``produced_by`` (artifact or fact -> call), ``read`` (call -> artifact it was given),
  ``from_source`` (call -> source it contacted), ``cites`` (call -> note it was shown), ``within``
  (a call the code made through the bridge -> the run that made it).

Who writes each edge, live:

* the tool-call middleware binds the call (``calling``), so anything below it knows which call is
  acting; it also cites the tool's own description and any notes registered for the tool;
* ``file_store.create_output_file*`` records ``produced_by`` and persists the file's lineage
  (producer, parents, upstream sources) in its metadata, so a later turn that reads the file
  inherits its sources without this process's memory;
* ``execute_code`` records ``read`` for every staged input, and for every workspace file an
  earlier run wrote that the program names;
* ``public_data_tools.fetch``, ``admin_boundary``, ``overpass_search`` and ``dem_for_region``
  record the service they reached;
* the turn log records each typed output as a fact.

``from_results`` replays a recorded turn through the same API, reading only fields the tools
wrote (call ids, ``input_files``, ``artifacts``, ``bridge_calls``, a fetch's ``source``), never a
summary. The benchmark replays the archive with it, and the supervisor merges it with the live
graph, so a peer that bypassed the middleware (a CLI peer, a test double) is still covered.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

PRODUCED_BY, READ, FROM_SOURCE, CITES, WITHIN = (
    "produced_by", "read", "from_source", "cites", "within")
# Followed upstream from a fact or an artifact to everything it rests on.
_UPSTREAM = frozenset({PRODUCED_BY, READ, FROM_SOURCE, CITES, WITHIN})

_FILE_ID_RE = re.compile(r"\bfile_[0-9a-f]{8,}\b")
_MAX_GRAPHS = 200
_NOTE_CHARS = 4000


@dataclass
class Graph:
    """One conversation's lineage. Add-only and idempotent: recording an edge twice is a no-op,
    so the live record and a replay of the same turn can be merged."""

    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: Dict[Tuple[str, str, str], Dict[str, Any]] = field(default_factory=dict)
    # filename -> file_id of the latest write a run made in the conversation's workspace
    workspace: Dict[str, str] = field(default_factory=dict)
    _lock: Any = field(default_factory=threading.RLock, repr=False, compare=False)

    # ------------------------------------------------------------------ primitives

    def add_node(self, node_id: str, type_: str, **attrs: Any) -> str:
        with self._lock:
            node = self.nodes.setdefault(node_id, {"id": node_id, "type": type_})
            for k, v in attrs.items():
                if v not in (None, "", [], {}) and node.get(k) in (None, "", [], {}):
                    node[k] = v
        return node_id

    def add_edge(self, src: str, kind: str, dst: str, **attrs: Any) -> None:
        if not src or not dst or src == dst:
            return
        with self._lock:
            self.edges.setdefault((src, kind, dst), {k: v for k, v in attrs.items()
                                                     if v not in (None, "")})

    def merge(self, other: "Graph") -> "Graph":
        for nid, node in other.nodes.items():
            self.add_node(nid, node.get("type", ""),
                          **{k: v for k, v in node.items() if k not in ("id", "type")})
        for (s, k, d), attrs in other.edges.items():
            self.add_edge(s, k, d, **attrs)
        for name, fid in other.workspace.items():
            self.workspace.setdefault(name, fid)
        return self

    def copy(self) -> "Graph":
        return Graph().merge(self)

    # ------------------------------------------------------------------ recording

    def call(self, call_id: str, name: Optional[str] = None, args: Any = None,
             parent: Optional[str] = None) -> str:
        nid = self.add_node(f"call:{call_id}", "call", tool=name, call_id=call_id,
                            args=_curated(args))
        if parent:
            self.add_edge(nid, WITHIN, self.call(parent))
        return nid

    def artifact(self, file_id: str, *, call_id: Optional[str] = None,
                 filename: Optional[str] = None, digest: Optional[str] = None,
                 inherited: Optional[Dict[str, Any]] = None) -> str:
        """A file. ``call_id`` produced it; ``inherited`` is the lineage persisted in its file
        record by an earlier turn (see ``file_lineage``), unfolded so its sources are reached."""
        nid = self.add_node(f"artifact:{file_id}", "artifact", file_id=file_id,
                            filename=filename, digest=digest)
        if call_id:
            self.add_edge(nid, PRODUCED_BY, self.call(call_id))
        if inherited and not call_id:
            prod = inherited.get("produced_by") or {}
            if prod.get("call"):
                cnode = self.call(str(prod["call"]), prod.get("tool"))
                self.add_edge(nid, PRODUCED_BY, cnode, via="file_record")
                for parent in inherited.get("parents") or []:
                    self.add_edge(cnode, READ, self.artifact(str(parent)), via="file_record")
                for src in inherited.get("sources") or []:
                    if isinstance(src, dict) and src.get("id"):
                        self.add_node(src["id"], "source", **{k: v for k, v in src.items()
                                                              if k != "id"})
                        self.add_edge(cnode, FROM_SOURCE, src["id"], via="file_record")
        return nid

    def read(self, call_id: str, file_id: str, *, via: str = "staged",
             filename: Optional[str] = None) -> None:
        self.add_edge(self.call(call_id), READ, self.artifact(file_id, filename=filename),
                      via=via)

    def source(self, call_id: str, *, source_id: Optional[str] = None,
               url: Optional[str] = None, host: Optional[str] = None,
               upload: Optional[Tuple[str, str]] = None) -> Optional[str]:
        """The service a call reached: a catalogue id, or a URL/host the catalogue resolves (an
        unknown host is a source of its own), or an upload ``(file_id, filename)``."""
        from agent_runtime import source_catalog as SC

        nid = None
        if upload:
            fid, fname = upload
            nid = self.add_node(f"source:upload:{fid}", "source", name=fname,
                                statement=f"{fname} (your upload)", names=[fname])
        else:
            src = SC.by_id(source_id) if source_id else None
            if host is None and url:
                host = urlparse(url).netloc or None
            if src is None and host:
                src = SC.by_host(host)
            if src is not None:
                nid = self.add_node(f"source:{src.id}", "source", name=src.name,
                                    statement=SC.statement(src),
                                    names=[src.name, *src.aliases, *([host] if host else [])])
                note = SC.note_text(src)
                self.add_edge(self.call(call_id), CITES,
                              self.note(f"catalog:{src.id}", note, about=nid), via="catalogue")
            elif host:
                nid = self.add_node(f"source:host:{host}", "source", name=host, host=host,
                                    statement=host, names=[host], url=url)
        if nid:
            self.add_edge(self.call(call_id), FROM_SOURCE, nid)
        return nid

    def note(self, note_id: str, text: str, about: Optional[str] = None) -> str:
        return self.add_node(f"note:{note_id}", "note", text=str(text or "")[:_NOTE_CHARS],
                             about=about)

    def cite(self, call_id: str, note_id: str, text: str, about: Optional[str] = None,
             via: str = "shown") -> None:
        if str(text or "").strip():
            about = about or self._about(note_id)
            self.add_edge(self.call(call_id), CITES, self.note(note_id, text, about), via=via)

    def _about(self, note_id: str) -> Optional[str]:
        """The source a host note is about (``bridge:<host>``), as a node; no edge, since being
        shown a note about a service is not reaching it."""
        from agent_runtime import source_catalog as SC

        host = str(note_id).rsplit(":", 1)[-1]
        if "." not in host or " " in host:
            return None
        src = SC.by_host(host)
        if src is not None:
            return self.add_node(f"source:{src.id}", "source", name=src.name,
                                 statement=SC.statement(src), names=[src.name, *src.aliases])
        return self.add_node(f"source:host:{host}", "source", name=host, host=host,
                             statement=host, names=[host])

    def fact(self, call_id: str, name: str, value: Any, unit: Optional[str] = None,
             kind: Optional[str] = None) -> str:
        nid = self.add_node(f"fact:{call_id}:{name}", "fact", name=name, value=value, unit=unit,
                            # Step 3 of the validation plan fills this from the computation.
                            kind=kind)
        self.add_edge(nid, PRODUCED_BY, self.call(call_id))
        return nid

    # ------------------------------------------------------------------ queries

    def upstream(self, start: Iterable[str]) -> List[str]:
        """Every node reachable upstream of *start* (node ids), *start* included, in BFS order."""
        out = list(dict.fromkeys(s for s in start if s in self.nodes))
        seen = set(out)
        succ: Dict[str, List[str]] = {}
        for (s, k, d) in self.edges:
            if k in _UPSTREAM:
                succ.setdefault(s, []).append(d)
        i = 0
        while i < len(out):
            for d in succ.get(out[i], ()):
                if d not in seen:
                    seen.add(d)
                    out.append(d)
            i += 1
        return out

    def upstream_of_calls(self, call_ids: Iterable[str]) -> List[str]:
        return self.upstream(f"call:{c}" for c in call_ids if c)

    def calls_upstream(self, call_ids: Iterable[str]) -> List[str]:
        return [self.nodes[n]["call_id"] for n in self.upstream_of_calls(call_ids)
                if self.nodes[n]["type"] == "call"]

    def sources_upstream(self, call_ids: Iterable[str]) -> List[Dict[str, Any]]:
        return [self.nodes[n] for n in self.upstream_of_calls(call_ids)
                if self.nodes[n]["type"] == "source"]

    def notes_upstream(self, call_ids: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        ids = (self.upstream_of_calls(call_ids) if call_ids is not None else list(self.nodes))
        return [self.nodes[n] for n in ids if self.nodes[n]["type"] == "note"]

    def producer(self, file_id: str) -> Optional[str]:
        for (s, k, d) in self.edges:
            if s == f"artifact:{file_id}" and k == PRODUCED_BY:
                return self.nodes.get(d, {}).get("call_id")
        return None

    def file_lineage(self, file_id: str) -> Dict[str, Any]:
        """What a file record persists: its producer, the files that producer read, and every
        source upstream, flattened so a reader needs no graph to inherit them."""
        call = self.producer(file_id)
        if not call:
            return {}
        cnode = self.nodes.get(f"call:{call}", {})
        parents = [self.nodes[d]["file_id"] for (s, k, d) in self.edges
                   if s == f"call:{call}" and k == READ and d in self.nodes]
        sources = [{k: v for k, v in n.items() if k in ("id", "name", "statement", "names", "host")}
                   for n in self.sources_upstream([call])]
        return {"produced_by": {"call": call, "tool": cnode.get("tool")},
                "parents": parents, "sources": sources}

    def to_dict(self) -> Dict[str, Any]:
        return {"nodes": list(self.nodes.values()),
                "edges": [{"from": s, "kind": k, "to": d, **a} for (s, k, d), a in self.edges.items()]}

    def explain(self, call_ids: Iterable[str], *, limit: int = 12) -> List[str]:
        """One line per call the answer used: the files it read, and the sources and notes
        upstream of it. For the auditor, which otherwise sees each result alone."""
        lines = []
        for cid in list(dict.fromkeys(c for c in call_ids if c))[:limit]:
            cnode = self.nodes.get(f"call:{cid}")
            if not cnode:
                continue
            up = self.upstream_of_calls([cid])
            read = [self.nodes[d] for (s, k, d) in self.edges
                    if s == f"call:{cid}" and k == READ and d in self.nodes]
            srcs = [self.nodes[n].get("name") for n in up if self.nodes[n]["type"] == "source"]
            notes = [n[len("note:"):] for n in up if self.nodes[n]["type"] == "note"]
            parts = []
            if read:
                parts.append("read " + ", ".join(
                    f"{f.get('filename') or f['file_id']} ({f['file_id']}"
                    + (f", written by {self.producer(f['file_id'])}"
                       if self.producer(f["file_id"]) else "") + ")" for f in read[:6]))
            if srcs:
                parts.append("data upstream from " + "; ".join(dict.fromkeys(srcs)))
            if notes:
                parts.append("shown " + ", ".join(dict.fromkeys(notes)))
            lines.append(f"{cnode.get('tool') or 'call'} {cid}"
                         + (": " + "; ".join(parts) if parts else ""))
        return lines


# --------------------------------------------------------------------------- registry + context

_GRAPHS: "OrderedDict[str, Graph]" = OrderedDict()
_GRAPHS_LOCK = threading.Lock()
_BOUND: ContextVar[Optional[Graph]] = ContextVar("lineage_graph", default=None)
_CALL: ContextVar[Optional[str]] = ContextVar("lineage_call", default=None)
# Notes shown to every call of a tool (beyond its own description): tool name -> {id: text}.
# The code bridge registers its host notes for execute_code here (see PR #104).
_TOOL_NOTES: Dict[str, Dict[str, str]] = {}


def _conversation() -> Optional[str]:
    try:
        from agent_runtime.file_store import current_session

        return current_session()
    except Exception:  # noqa: BLE001
        return None


def graph_for(conversation: Optional[str]) -> Optional[Graph]:
    if not conversation:
        return None
    with _GRAPHS_LOCK:
        g = _GRAPHS.get(conversation)
        if g is None:
            g = _GRAPHS[conversation] = Graph()
            while len(_GRAPHS) > _MAX_GRAPHS:
                _GRAPHS.popitem(last=False)
        else:
            _GRAPHS.move_to_end(conversation)
        return g


def active() -> Optional[Graph]:
    """The graph recording now: one bound explicitly (a test, a replay), else the
    conversation's. None outside a conversation, and every record_* is then a no-op."""
    return _BOUND.get() or graph_for(_conversation())


def current_call() -> Optional[str]:
    return _CALL.get()


@contextmanager
def bind(graph: Graph) -> Iterator[Graph]:
    tok = _BOUND.set(graph)
    try:
        yield graph
    finally:
        _BOUND.reset(tok)


@contextmanager
def calling(call_id: Optional[str], name: Optional[str] = None, args: Any = None, *,
            description: Optional[str] = None, parent: Optional[str] = None) -> Iterator[None]:
    """Everything recorded inside acts for *call_id*. Records the call, the tool's description
    and every note registered for the tool as notes it was shown."""
    if not call_id:
        yield
        return
    g = active()
    if g is not None:
        try:
            g.call(call_id, name, args, parent=parent)
            if description:
                g.cite(call_id, f"tool:{name}", description, via="tool_description")
            for nid, text in (_TOOL_NOTES.get(str(name)) or {}).items():
                g.cite(call_id, nid, text, via="tool_note")
        except Exception:  # noqa: BLE001 - provenance never costs the tool call
            logger.debug("lineage: recording call failed", exc_info=True)
    tok = _CALL.set(call_id)
    try:
        yield
    finally:
        _CALL.reset(tok)


def register_tool_notes(tool: str, notes: Dict[str, str]) -> None:
    """Notes every call of *tool* is shown (a service note in its description). Replaces what
    was registered for the tool before."""
    _TOOL_NOTES[str(tool)] = {str(k): str(v) for k, v in (notes or {}).items() if v}


def tool_notes(tool: str) -> Dict[str, str]:
    return dict(_TOOL_NOTES.get(str(tool)) or {})


def _guard(fn: Any) -> Any:
    def wrapped(*a: Any, **kw: Any) -> Any:
        g = active()
        call = kw.pop("call_id", None) or current_call()
        if g is None or not call:
            return None
        try:
            return fn(g, call, *a, **kw)
        except Exception:  # noqa: BLE001 - provenance never costs the action it describes
            logger.debug("lineage: %s failed", fn.__name__, exc_info=True)
            return None
    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped


@_guard
def record_call(g: Graph, call: str, name: Optional[str] = None, args: Any = None,
                parent: Optional[str] = None) -> str:
    """A call made outside the tool middleware (the code bridge's calls, with the run as
    *parent*). Pass ``call_id=``."""
    return g.call(call, name, args, parent=parent)


@_guard
def record_artifact(g: Graph, call: str, file_id: str, *, filename: Optional[str] = None,
                    digest: Optional[str] = None) -> Dict[str, Any]:
    """The current call produced *file_id*. Returns the lineage to persist in its record."""
    g.artifact(file_id, call_id=call, filename=filename, digest=digest)
    return g.file_lineage(file_id)


@_guard
def record_read(g: Graph, call: str, file_id: str, *, via: str = "staged",
                filename: Optional[str] = None) -> None:
    """The current call was given *file_id*. A file this graph has not seen (an earlier turn,
    another process) is unfolded from the lineage its file record carries."""
    if f"artifact:{file_id}" not in g.nodes or not g.producer(file_id):
        rec = _file_record(file_id)
        g.artifact(file_id, filename=filename or rec.get("filename"),
                   inherited=rec.get("lineage") if isinstance(rec.get("lineage"), dict) else None)
        if rec and rec.get("kind") == "upload":
            g.source(call, upload=(file_id, str(rec.get("filename") or file_id)))
    g.read(call, file_id, via=via, filename=filename)


@_guard
def record_source(g: Graph, call: str, *, source_id: Optional[str] = None,
                  url: Optional[str] = None, host: Optional[str] = None) -> Optional[str]:
    """The current call reached a service."""
    return g.source(call, source_id=source_id, url=url, host=host)


@_guard
def record_cite(g: Graph, call: str, note_id: str, text: str,
                about: Optional[str] = None) -> None:
    g.cite(call, note_id, text, about)


@_guard
def record_fact(g: Graph, call: str, name: str, value: Any, unit: Optional[str] = None,
                kind: Optional[str] = None) -> str:
    return g.fact(call, name, value, unit, kind)


@_guard
def record_workspace_file(g: Graph, call: str, name: str, artifact_id: str) -> None:
    """The current call put *name* in the conversation's working directory (``stage_url``,
    ``stage_element``), where a run opens it by name. A staged file has no file_id, so it is an
    artifact under ``ws:<name>``."""
    g.artifact(artifact_id, call_id=call, filename=name)
    g.workspace[str(name)] = artifact_id


def _file_record(file_id: str) -> Dict[str, Any]:
    try:
        from agent_runtime.file_store import get_file_record

        return get_file_record(file_id) or {}
    except Exception:  # noqa: BLE001
        return {}


def workspace_reads(code: str, workspace: Dict[str, str]) -> List[Tuple[str, str]]:
    """``(filename, file_id)`` for every file an earlier run left in the workspace that this
    program names, as a string literal or a path ending. A static over-approximation: a name in
    a comment counts too. What the program actually opened is not observable from here (GDAL
    opens files below Python), and naming a source that was only mentioned is the safer error
    for a Sources line than dropping one that was read."""
    text = str(code or "")
    if not text:
        return []
    out = []
    for name, fid in workspace.items():
        if len(name) < 3:
            continue
        if re.search(r"(?<![\w.-])" + re.escape(name) + r"(?![\w.-])", text):
            out.append((name, fid))
    return out


# --------------------------------------------------------------------------- replay

def _payload(content: Any) -> Dict[str, Any]:
    from agent_runtime.source_catalog import _payload as payload

    return payload(content)


def _curated(args: Any) -> Any:
    """Args as a node keeps them: short scalars only (a program is the call's, not the graph's)."""
    if not isinstance(args, dict):
        return None
    out = {}
    for k, v in args.items():
        if k in ("code",):
            continue
        if isinstance(v, (int, float, bool)) or (isinstance(v, str) and len(v) <= 200):
            out[k] = v
        elif isinstance(v, list) and len(v) <= 8 and all(isinstance(x, (int, float, str)) for x in v):
            out[k] = v
    return out or None


def _file_ids_in(node: Any, out: List[str], depth: int = 0) -> None:
    if depth > 4:
        return
    if isinstance(node, str):
        for m in _FILE_ID_RE.findall(node) if len(node) < 64 else []:
            if m not in out:
                out.append(m)
    elif isinstance(node, dict):
        for k, v in node.items():
            if k != "code":
                _file_ids_in(v, out, depth + 1)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _file_ids_in(v, out, depth + 1)


def _produced(payload: Dict[str, Any], given: set) -> List[Tuple[str, Optional[str]]]:
    """Files a result introduced: every object that names a file_id beside a download link or a
    filename (the shape the Downloads panel harvests), minus the files the call was given."""
    out: List[Tuple[str, Optional[str]]] = []

    def walk(node: Any, depth: int) -> None:
        if depth > 4:
            return
        if isinstance(node, dict):
            fid = node.get("file_id")
            if (isinstance(fid, str) and _FILE_ID_RE.fullmatch(fid) and fid not in given
                    and (node.get("download_url") or node.get("filename"))
                    and fid not in {f for f, _ in out}):
                out.append((fid, node.get("filename")))
            for k, v in node.items():
                if k not in ("input_files", "bridge_calls"):
                    walk(v, depth + 1)
        elif isinstance(node, list):
            for v in node:
                walk(v, depth + 1)

    walk(payload, 0)
    return out


def _is_source_artifact(payload: Dict[str, Any], file_id: str) -> bool:
    """The saved copy of a run's own program lives in the file store, not the workspace."""
    return any(isinstance(a, dict) and a.get("file_id") == file_id and a.get("kind") == "source"
               for a in payload.get("artifacts") or [])


def _urls(args: Dict[str, Any], payload: Dict[str, Any]) -> List[str]:
    return [str(v) for k, v in {**args, **payload}.items()
            if k in ("url", "source", "origin") and isinstance(v, str) and v.startswith("http")]


def record_result(g: Graph, call: Dict[str, Any], res: Dict[str, Any], *,
                  parent: Optional[str] = None, description: Optional[str] = None,
                  notes: Optional[Dict[str, Dict[str, str]]] = None) -> None:
    """Replay one recorded (call, result) into *g*, from the fields the tool wrote."""
    from agent_runtime import source_catalog as SC
    from agent_runtime import turn_log

    cid = str(res.get("tool_call_id") or call.get("id") or "")
    if not cid:
        return
    name = str(res.get("name") or call.get("name") or "")
    args = SC._args(call.get("args"))
    g.call(cid, name, args, parent=parent)
    if description:
        g.cite(cid, f"tool:{name}", description, via="tool_description")
    for nid, text in ((notes or {}).get(name) or {}).items():
        g.cite(cid, nid, text, via="tool_note")
    payload = _payload(res.get("content"))
    if not turn_log.parse_result(res.get("content"))["ok"]:
        return
    given: List[str] = []
    _file_ids_in(args, given)
    staged = [f for f in payload.get("input_files") or [] if isinstance(f, dict)]
    for f in staged:
        if f.get("file_id") and str(f["file_id"]) not in given:
            given.append(str(f["file_id"]))
    names = {str(f["file_id"]): f.get("filename") for f in staged if f.get("file_id")}
    for fid in given:
        g.read(cid, fid, via="staged" if fid in names else "argument", filename=names.get(fid))
    if name == "execute_code":
        for fname, fid in workspace_reads(str(args.get("code") or args.get("entrypoint") or ""),
                                          dict(g.workspace)):
            if fid not in given and g.producer(fid):
                g.read(cid, fid, via="workspace", filename=fname)
    src = SC._BY_TOOL.get(name)
    if src is not None:
        g.source(cid, source_id=src.id)
    for url in _urls(args, payload):
        g.source(cid, url=url)
    if name in ("stage_url", "stage_element", "stage_object") and payload.get("filename"):
        g.artifact(f"ws:{payload['filename']}", call_id=cid, filename=payload["filename"])
        g.workspace[str(payload["filename"])] = f"ws:{payload['filename']}"
    for fid, fname in _produced(payload, set(given)):
        g.artifact(fid, call_id=cid, filename=fname)
        if name == "execute_code" and fname and not _is_source_artifact(payload, fid):
            # A run's outputs are what it left in the conversation's working directory, where
            # a later run can open them by name without staging them.
            g.workspace[str(fname)] = fid
    for bc in payload.get("bridge_calls") or []:
        if not isinstance(bc, dict) or not bc.get("call_id"):
            continue
        result = bc.get("result") if isinstance(bc.get("result"), dict) else {}
        record_result(g, {"id": bc["call_id"], "name": bc.get("name"), "args": bc.get("args")},
                      {"tool_call_id": bc["call_id"], "name": bc.get("name"),
                       "content": json.dumps(result if bc.get("ok") else
                                             {"ok": False, **result}, default=str)},
                      parent=cid)
        if bc.get("ok") and result.get("file_id"):
            g.read(cid, str(result["file_id"]), via="bridge", filename=result.get("filename"))
    for typed in turn_log.typed_outputs(res.get("content")):
        g.fact(cid, str(typed.get("name")), typed.get("value"), typed.get("unit"),
               kind=typed.get("figure_kind") or None)


def from_results(pairs: Iterable[Tuple[Dict[str, Any], Dict[str, Any]]], *,
                 tool_docs: Optional[Dict[str, str]] = None,
                 notes: Optional[Dict[str, Dict[str, str]]] = None,
                 graph: Optional[Graph] = None) -> Graph:
    """A graph from a turn's recorded (call, result) pairs, in order. *tool_docs* are the tool
    descriptions the peers were bound with (`bound_tool_docs`); *notes* the notes each tool was
    shown beyond that (`register_tool_notes`, or the archived run's ``tool_notes.json``)."""
    g = graph if graph is not None else Graph()
    docs = tool_docs or {}
    allnotes = {**{t: dict(n) for t, n in _TOOL_NOTES.items()}, **(notes or {})}
    for call, res in pairs:
        if isinstance(res, dict):
            name = str(res.get("name") or (call or {}).get("name") or "")
            record_result(g, call or {}, res, description=docs.get(name), notes=allnotes)
    return g


def for_state(state: Any, pairs: Optional[Sequence[Tuple[Dict, Dict]]] = None) -> Graph:
    """The turn's lineage: the conversation's live graph merged with a replay of the recorded
    results (so a peer that bypassed the middleware is still covered)."""
    st = state if isinstance(state, dict) else {}
    if pairs is None:
        pairs = []
        for slot in ("analysis_results", "code_result"):
            res = st.get(slot)
            if not isinstance(res, dict):
                continue
            calls = {c.get("id"): c for c in (res.get("tool_calls") or []) if isinstance(c, dict)}
            pairs += [(calls.get(r.get("tool_call_id")) or {}, r)
                      for r in res.get("tool_results") or [] if isinstance(r, dict)]
    # The conversation the turn runs in: the file store's session, which the server binds to the
    # request's thread id for the whole turn, else the state's own thread id.
    live = _BOUND.get() or graph_for(_conversation() or (str(st["thread_id"])
                                                         if st.get("thread_id") else None))
    base = live.copy() if live is not None else Graph()
    return from_results(pairs, tool_docs=st.get("bound_tool_docs") or {},
                        notes=st.get("tool_notes") or None, graph=base)


# --------------------------------------------------------------------------- notes as support

_DASHES = str.maketrans({"−": "-", "–": "-", "—": "-", "×": "*",
                         "`": " ", " ": " ", " ": " "})
_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\w|\.\d)")
_EXPR = re.compile(r"\(?[a-z_]\w*\s*[-+*/]\s*\d[\d.]*\)?\s*[*/]\s*\d[\d.]*")
_STOP = frozenset("""this that with from were was have has the and for are which into their there
these those than then them they been being about after before over under each every within only
also more most less such some what when where while would could should computed calculated
converted result results value values answer using used""".split())


def _norm(text: str) -> str:
    t = str(text or "").translate(_DASHES).lower().replace(",", "")
    t = re.sub(r"\s*([-+*/=()])\s*", r"\1", t)
    return re.sub(r"\s+", " ", t)


def note_supports(claim: str, text: str) -> bool:
    """Whether a note states *claim*: every figure in the claim is in the note, and the claim's
    formula is there verbatim (modulo spacing, case and the Unicode minus) or most of its words
    are. Strict on purpose: a note that merely shares a topic does not support a claim."""
    c, t = _norm(claim), _norm(text)
    if not c or not t:
        return False
    nums = [n for n in _NUM.findall(c) if len(n.lstrip("-")) > 1]
    if nums and not all(re.search(r"(?<![\w.])" + re.escape(n.lstrip("-")) + r"(?![\w])", t)
                        for n in nums):
        return False
    exprs = [e.strip("() .") for e in _EXPR.findall(c)]
    if exprs and all(re.search(r"(?<![\w.])" + re.escape(e) + r"(?![\w.])", t) for e in exprs):
        return True
    words = [w for w in re.findall(r"[a-z][a-z'-]{3,}", c) if w not in _STOP]
    if len(words) < 3:
        return False
    return sum(1 for w in dict.fromkeys(words) if w in t) >= max(3, round(0.75 * len(set(words))))


def audit_context(g: Graph, used_calls: Iterable[str]) -> Dict[str, Any]:
    """What the auditor and the deterministic reconciliation are given: one line per used call,
    and every note upstream of what the answer used, with the source it is about."""
    used = [c for c in dict.fromkeys(used_calls) if c and f"call:{c}" in g.nodes]
    calls = used or [n["call_id"] for n in g.nodes.values() if n["type"] == "call"]
    notes = []
    for n in g.notes_upstream(calls):
        about = g.nodes.get(n.get("about") or "", {})
        notes.append({"id": n["id"], "text": n.get("text") or "",
                      "source": about.get("name") or None})
    return {"lines": g.explain(used), "notes": notes,
            "sources": [s.get("name") for s in g.sources_upstream(calls)]}


def supporting_note(claim: str, context: Any) -> Optional[Dict[str, Any]]:
    """The first note in an audit context (``audit_context``) that states *claim*."""
    if not isinstance(context, dict):
        return None
    for n in context.get("notes") or []:
        if isinstance(n, dict) and note_supports(claim, n.get("text") or ""):
            return n
    return None


__all__ = ["Graph", "active", "audit_context", "bind", "calling", "current_call", "for_state",
           "from_results", "graph_for", "note_supports", "record_artifact", "record_call",
           "record_cite", "record_fact", "record_read", "record_source",
           "record_workspace_file", "register_tool_notes", "replay_result", "supporting_note", "tool_notes", "workspace_reads"]
