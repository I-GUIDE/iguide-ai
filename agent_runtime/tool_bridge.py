"""A tool bridge for sandboxed code: agent capabilities as Python functions, no network.

The code sandbox runs ``--network none``. Code that needs public data used to have one way to
get it: ask the peer to fetch a file first, then read it as an input. The bridge lets the code
ask for itself, mid-run, without giving the container a network interface:

    from iguide_bridge import fetch_public_data, overpass_search, admin_boundary, dem_for_region
    tracts = admin_boundary("Champaign County", state="IL", subdivide="tracts")
    gdf = geopandas.read_file(tracts["path"])

How a call travels. Two host directories are mounted into the run container: ``/bridge/in``
read-write and ``/bridge/out`` READ-ONLY. The client (``iguide_bridge.py``, written into
``out``) drops a JSON request into ``in`` and waits. A thread on the host, started for the
length of the run, reads the request, runs the capability AGENT-SIDE with the agent's own
policy, writes any data file into ``out/data`` and the answer into ``out/resp``. No socket, no
DNS, no route: the container's only channel is two directories, and the one it can write holds
nothing but requests.

What the host enforces, per call:

* only the four functions below, with JSON arguments under ``MAX_REQUEST_BYTES``;
* ``fetch_public_data`` goes through ``public_data_tools.fetch``: HTTPS GET to an allowlisted
  host (``bridge_hosts``), every resolved address public, redirects re-checked, a size cap, no
  proxy or ``.netrc``; URLs longer than ``MAX_URL_CHARS`` are refused (a long query string is
  the channel data would leave by);
* caps on calls, fetches and bytes per run;
* every call is recorded as a TOOL CALL of the turn (``turn_log``) and streamed like one, with
  its arguments and result, so the fact set, the verifier and the Sources line see where the
  data came from. The run's own result lists them under ``bridge_calls`` too.

Credentials never enter the container: the capabilities run in the agent process, and the
container sees only their results.

Off unless ``AGENT_CODE_BRIDGE=1``. Design note and threat model: docs/agent-architecture-
changes.md, "Three ways to reach public data".
"""
from __future__ import annotations

import contextvars
import inspect
import json
import logging
import os
import re
import shutil
import stat
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ENABLE_ENV = "AGENT_CODE_BRIDGE"
CLIENT_MODULE = "iguide_bridge"
# Where the two directories appear inside the run container.
CONTAINER_ROOT = "/bridge"

MAX_REQUEST_BYTES = 64 * 1024
MAX_URL_CHARS = 2048
_POLL_S = 0.05
_RID_RE = re.compile(r"^[0-9a-f]{32}$")


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name) or default))
    except ValueError:
        return default


def enabled() -> bool:
    from agent_runtime.capability_registry import flag_on

    return flag_on(ENABLE_ENV)


# --------------------------------------------------------------------------- the client

# Runs INSIDE the sandbox: standard library only. Writes a request atomically (write, then
# rename), and polls for the answer. A refusal or failure raises, so code never carries on with
# data it did not get.
CLIENT_SOURCE = r'''"""I-GUIDE tool bridge: agent capabilities callable from sandboxed code.

Each call is answered by the agent, which fetches on the code's behalf under its own policy and
records the call. The sandbox itself has no network. Every function returns a dict; a refusal
or failure raises BridgeError with the reason.
"""
import json
import os
import time
import uuid

_ROOT = os.environ.get("IGUIDE_BRIDGE", "/bridge")
_TIMEOUT = float(os.environ.get("IGUIDE_BRIDGE_TIMEOUT", "600"))


class BridgeError(RuntimeError):
    """The agent refused or could not complete the call. The message says why."""


def _call(name, **args):
    rid = uuid.uuid4().hex
    inbox = os.path.join(_ROOT, "in")
    tmp = os.path.join(inbox, "." + rid + ".tmp")
    with open(tmp, "w") as fh:
        json.dump({"id": rid, "name": name, "args": args}, fh)
    os.rename(tmp, os.path.join(inbox, rid + ".req"))
    answer = os.path.join(_ROOT, "out", "resp", rid + ".json")
    deadline = time.monotonic() + _TIMEOUT
    delay = 0.02
    while time.monotonic() < deadline:
        try:
            with open(answer) as fh:
                body = json.load(fh)
        except (OSError, ValueError):
            time.sleep(delay)
            delay = min(delay * 1.5, 0.25)
            continue
        if not body.get("ok"):
            raise BridgeError("%s: %s" % (name, body.get("error")))
        return body["result"]
    raise BridgeError("%s: no answer from the agent within %.0f s" % (name, _TIMEOUT))


def fetch_public_data(url, filename=None):
    """HTTPS GET from an approved public data host. Returns {path, source, size_bytes, shape, ...};
    the file is at result["path"]."""
    return _call("fetch_public_data", url=url, filename=filename)


def overpass_search(feature, place=None, bbox=None, max_features=20000):
    """Every OpenStreetMap feature of a kind (e.g. "building", "highway=primary", "amenity=school")
    in a place or bbox [minlon, minlat, maxlon, maxlat], PAGED past Overpass's 500-per-query cap
    by splitting the box. Returns {path (GeoJSON), count, complete, tiles, ...}."""
    return _call("overpass_search", feature=feature, place=place, bbox=bbox,
                 max_features=max_features)


def admin_boundary(area, state=None, level="county", subdivide=None):
    """A US state, county or city boundary from Census TIGERweb; subdivide="tracts" or
    "block_groups" for the zones inside a county. Returns {path (GeoJSON), feature_count, ...}."""
    return _call("admin_boundary", area=area, state=state, level=level, subdivide=subdivide)


def dem_for_region(bbox=None, lon=None, lat=None, buffer_m=2500.0, size=512):
    """USGS 3DEP elevation (metres, EPSG:4326 GeoTIFF) for a bbox or a point and buffer.
    `size` is pixels per side (max 1536). Returns {path (GeoTIFF), region_bbox, min, max, ...}."""
    return _call("dem_for_region", bbox=bbox, lon=lon, lat=lat, buffer_m=buffer_m, size=size)
'''


def model_note() -> str:
    """What execute_code's description and the code peer's prompt say while the bridge is on."""
    from agent_runtime.public_data_tools import BRIDGE_HOST_NOTES, bridge_hosts

    hosts = bridge_hosts()
    listed = "; ".join(f"{h} ({BRIDGE_HOST_NOTES.get(h, '')})".replace(" ()", "")
                       for h in hosts)
    return (
        "TOOL BRIDGE (this deployment): the sandbox still has NO network, but code can call "
        "agent capabilities as Python functions, answered by the agent mid-run: "
        "`from iguide_bridge import fetch_public_data, overpass_search, admin_boundary, "
        "dem_for_region`. Each returns a dict whose `path` is a local file the code reads "
        "(GeoJSON, GeoTIFF, JSON, CSV); a refusal raises iguide_bridge.BridgeError. "
        "fetch_public_data(url) = HTTPS GET from an approved host only: " + listed + ". "
        "overpass_search(feature, place=None, bbox=None) returns EVERY OSM feature, paged past "
        "the 500-per-query cap (check `complete`). admin_boundary(area, state, level, "
        "subdivide) = Census TIGERweb boundaries/tracts. dem_for_region(bbox or lon/lat, "
        "size<=1536) = USGS 3DEP elevation. Each call is logged and named as the data's source, "
        "so get data THROUGH the bridge rather than typing figures into code. Bridge calls "
        "count against the run's timeout: pass timeout_seconds (e.g. 600) to execute_code for "
        f"runs that fetch. Limits per run: {_int_env('AGENT_CODE_BRIDGE_MAX_CALLS', 30)} calls, "
        f"{_int_env('AGENT_CODE_BRIDGE_MAX_FETCHES', 15)} fetches. "
    )




# --------------------------------------------------------------------------- handlers

class Refused(ValueError):
    """The bridge refused the call; the message is returned to the code."""


def _file_ids(node: Any, out: List[str], depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(node, dict):
        fid = node.get("file_id")
        if isinstance(fid, str) and fid and fid not in out:
            out.append(fid)
        for v in node.values():
            _file_ids(v, out, depth + 1)
    elif isinstance(node, list):
        for v in node[:50]:
            _file_ids(v, out, depth + 1)


def _as_dict(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        body = json.loads(raw)
    except (TypeError, ValueError):
        return {"ok": False, "error": str(raw)[:500]}
    return body if isinstance(body, dict) else {"ok": True, "value": body}


def _tool(factory: Callable[[], List[Any]], name: str) -> Callable[..., Any]:
    for t in factory():
        if getattr(t, "name", "") == name:
            return t.func
    raise Refused(f"{name} is not available in this deployment")


def _fetch_handler(url: str, filename: Optional[str] = None) -> Dict[str, Any]:
    from agent_runtime import public_data_tools as pdt

    if len(str(url or "")) > MAX_URL_CHARS:
        raise Refused(f"the URL is over {MAX_URL_CHARS} characters; ask the service for less")
    try:
        return pdt.fetch(url, filename=filename, hosts=pdt.bridge_hosts())
    except pdt.Refused as exc:
        raise Refused(str(exc)) from None


# Overpass answers at most this many features per query (rag_pipeline.search.overpass).
_OVERPASS_PAGE = 500


def _overpass_handler(feature: str, place: Optional[str] = None, bbox: Any = None,
                      max_features: int = 20000) -> Dict[str, Any]:
    """Every feature in the region: a box that comes back full is split in four and re-asked.

    Overpass has no offset, so "page" here means "tile". A way crossing two tiles comes back
    from both and is kept once (by OSM type and id).
    """
    from agent_runtime.file_store import create_output_file_from_path
    from rag_pipeline.search import overpass as ov

    region = ov._resolve_bbox(place, bbox)
    if region is None:
        raise Refused("give a place (e.g. 'Piatt County, Illinois') or a bbox "
                      "[minlon, minlat, maxlon, maxlat]")
    cap = max(1, min(int(max_features or 20000), _int_env("AGENT_CODE_BRIDGE_OSM_MAX", 50000)))
    max_tiles = _int_env("AGENT_CODE_BRIDGE_OSM_TILES", 96)
    pending: List[Tuple[float, float, float, float]] = [tuple(region)]
    seen: Dict[Tuple[str, Any], Dict[str, Any]] = {}
    tiles = splits = 0
    filt = None
    while pending and tiles < max_tiles and len(seen) < cap:
        box = pending.pop(0)
        res = ov.overpass_search(feature, bbox=list(box), limit=_OVERPASS_PAGE)
        tiles += 1
        if res.get("error"):
            raise Refused(f"Overpass failed on tile {tiles} ({res.get('error')}: "
                          f"{str(res.get('message'))[:200]}); no partial result is returned")
        filt = (res.get("query") or {}).get("osm_filter")
        for f in res.get("features") or []:
            seen.setdefault((str(f.get("osm_type")), f.get("osm_id")), f)
        if int(res.get("count") or 0) >= _OVERPASS_PAGE:
            w, s, e, n = box
            mx, my = (w + e) / 2, (s + n) / 2
            pending += [(w, s, mx, my), (mx, s, e, my), (w, my, mx, n), (mx, my, e, n)]
            splits += 1
    complete = not pending and len(seen) < cap
    feats = list(seen.values())[:cap]
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": f.get("geometry"),
         "properties": {"osm_type": f.get("osm_type"), "osm_id": f.get("osm_id"),
                        "name": f.get("name"), "feature_type": f.get("feature_type"),
                        **{k: v for k, v in (f.get("tags") or {}).items()
                           if isinstance(v, (str, int, float))}}}
        for f in feats if f.get("geometry")]}
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(feature)).strip("_")[:40] or "features"
    tmp = Path(tempfile.mkdtemp(prefix="bridge_osm_")) / f"osm_{slug}.geojson"
    try:
        tmp.write_text(json.dumps(fc), encoding="utf-8")
        rec = create_output_file_from_path(str(tmp), filename=tmp.name)
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)
    out = {"ok": True, "source": "OpenStreetMap (via Overpass)", "feature": feature,
           "osm_filter": filt, "bbox": list(region), "count": len(fc["features"]),
           "complete": complete, "tiles": tiles, "splits": splits,
           "file_id": rec.get("file_id"), "filename": rec.get("filename")}
    if not complete:
        out["warning"] = (f"NOT every feature: stopped at {len(feats)} features / {tiles} tiles "
                          "(caps). Narrow the area or say the count is a lower bound.")
    return out


def _admin_handler(area: str, state: Optional[str] = None, level: str = "county",
                   subdivide: Optional[str] = None) -> Dict[str, Any]:
    from agent_runtime.admin_boundary_tools import make_admin_boundary_tools

    fn = _tool(make_admin_boundary_tools, "admin_boundary")
    return _as_dict(fn(area=area, state=state, level=level, subdivide=subdivide))


def _dem_handler(bbox: Any = None, lon: Optional[float] = None, lat: Optional[float] = None,
                 buffer_m: float = 2500.0, size: int = 512) -> Dict[str, Any]:
    from agent_runtime.terrain_tools import make_terrain_tools

    fn = _tool(lambda: make_terrain_tools(default_input_file_ids=None), "dem_for_region")
    return _as_dict(fn(bbox=bbox, lon=lon, lat=lat, buffer_m=buffer_m, size=size))


DEFAULT_HANDLERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "fetch_public_data": _fetch_handler,
    "overpass_search": _overpass_handler,
    "admin_boundary": _admin_handler,
    "dem_for_region": _dem_handler,
}


# --------------------------------------------------------------------------- the server

def _resolve_file(file_id: str) -> Tuple[Optional[Path], Optional[str]]:
    from agent_runtime.file_store import get_file_record, resolve_file_id

    try:
        rec = get_file_record(file_id) or {}
        return resolve_file_id(file_id), rec.get("filename")
    except Exception:  # noqa: BLE001 - a file the store cannot resolve is simply not copied
        return None, None


class BridgeServer:
    """Answers one run's bridge calls. ``start()`` before the container, ``stop()`` after."""

    def __init__(self, root: Path, *, container_root: str = CONTAINER_ROOT,
                 handlers: Optional[Dict[str, Callable[..., Dict[str, Any]]]] = None,
                 resolve_file: Callable[[str], Tuple[Optional[Path], Optional[str]]] = _resolve_file,
                 record: bool = True) -> None:
        self.root = Path(root)
        self.inbox = self.root / "in"
        self.out = self.root / "out"
        self.container_root = container_root.rstrip("/")
        self.handlers = dict(handlers if handlers is not None else DEFAULT_HANDLERS)
        self.resolve_file = resolve_file
        self.record = record
        self.calls: List[Dict[str, Any]] = []
        self.max_calls = _int_env("AGENT_CODE_BRIDGE_MAX_CALLS", 30)
        self.max_fetches = _int_env("AGENT_CODE_BRIDGE_MAX_FETCHES", 15)
        self.max_bytes = _int_env("AGENT_CODE_BRIDGE_MAX_MB", 500) * 1024 * 1024
        self._bytes = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- setup --------------------------------------------------------------------------

    def prepare(self) -> None:
        for d in (self.inbox, self.out / "resp", self.out / "data"):
            d.mkdir(parents=True, exist_ok=True)
        # The container user writes requests; nothing else in the tree is writable to it (the
        # `out` mount is read-only), so a symlink planted to redirect a host write cannot exist.
        os.chmod(self.inbox, 0o777)
        (self.out / f"{CLIENT_MODULE}.py").write_text(CLIENT_SOURCE, encoding="utf-8")

    def start(self) -> "BridgeServer":
        self.prepare()
        ctx = contextvars.copy_context()   # the turn log, trace stream and file-store user
        self._thread = threading.Thread(target=ctx.run, args=(self._loop,),
                                        name="code-bridge", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def container_path(self, host_path: Path) -> str:
        return f"{self.container_root}/{host_path.relative_to(self.root).as_posix()}"

    # -- loop ---------------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                names = sorted(n for n in os.listdir(self.inbox) if n.endswith(".req"))
            except OSError:
                names = []
            for name in names:
                self._serve(name)
            if not names:
                self._stop.wait(_POLL_S)

    def serve_pending(self) -> int:
        """Answer every request waiting now (tests drive the server without its thread)."""
        names = sorted(n for n in os.listdir(self.inbox) if n.endswith(".req"))
        for name in names:
            self._serve(name)
        return len(names)

    def _read_request(self, name: str) -> Dict[str, Any]:
        path = self.inbox / name
        # O_NOFOLLOW + fstat: the container controls this directory, so the "request" may be a
        # symlink to a host file or a FIFO. Neither is read.
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise Refused("a request must be a regular file")
            if st.st_size > MAX_REQUEST_BYTES:
                raise Refused(f"a request is limited to {MAX_REQUEST_BYTES} bytes")
            raw = os.read(fd, MAX_REQUEST_BYTES + 1)
        finally:
            os.close(fd)
        body = json.loads(raw.decode("utf-8"))
        if not isinstance(body, dict) or not isinstance(body.get("args"), dict):
            raise Refused("malformed request")
        return body

    def _serve(self, name: str) -> None:
        rid = name[:-len(".req")]
        try:
            body = self._read_request(name)
        except Exception as exc:  # noqa: BLE001
            body = {"id": rid, "error": str(exc) if isinstance(exc, Refused) else "unreadable request"}
        finally:
            try:
                os.unlink(self.inbox / name)
            except OSError:
                pass
        if not _RID_RE.match(rid) or body.get("id") not in (None, rid):
            logger.warning("code bridge: ignored a request with a malformed id %r", rid[:40])
            return
        fname = str(body.get("name") or "")
        args = {k: v for k, v in (body.get("args") or {}).items() if v is not None}
        started = time.monotonic()
        call_id = f"bridge_{len(self.calls) + 1}_{rid[:8]}"
        self._emit_call(fname, args, call_id)
        try:
            if body.get("error"):
                raise Refused(body["error"])
            result = self._dispatch(fname, args)
            answer = {"ok": True, "result": result}
            record = {"ok": True, **result}
        except Refused as exc:
            answer = record = {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - a failure is an answer, never a crash
            logger.warning("code bridge: %s failed: %s", fname, exc, exc_info=True)
            answer = record = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        entry = {"call_id": call_id, "name": fname, "args": args, "ok": answer["ok"],
                 "seconds": round(time.monotonic() - started, 2),
                 "result": _summary(record)}
        self.calls.append(entry)
        self._record(fname, args, call_id, record)
        self._respond(rid, answer)

    def _dispatch(self, fname: str, args: Dict[str, Any]) -> Dict[str, Any]:
        handler = self.handlers.get(fname)
        if handler is None:
            raise Refused(f"unknown function {fname!r}; the bridge offers "
                          f"{', '.join(sorted(self.handlers))}")
        if len(self.calls) >= self.max_calls:
            raise Refused(f"at most {self.max_calls} bridge calls per run")
        if fname == "fetch_public_data" and sum(
                1 for c in self.calls if c["name"] == "fetch_public_data") >= self.max_fetches:
            raise Refused(f"at most {self.max_fetches} fetches per run; work with the files "
                          "already fetched")
        params = inspect.signature(handler).parameters
        unknown = sorted(k for k in args if k not in params)
        if unknown:
            raise Refused(f"{fname} takes no argument(s) {unknown}")
        result = handler(**args)
        if not isinstance(result, dict):
            result = _as_dict(result)
        if result.get("ok") is False or (result.get("error") and result.get("ok") is not True):
            raise Refused(str(result.get("error") or "failed") + (
                f" (hint: {result['hint']})" if result.get("hint") else ""))
        return self._materialise(result)

    def _materialise(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Copy every file the result names into ``out/data`` and say where the code finds it."""
        ids: List[str] = []
        _file_ids(result, ids)
        local: Dict[str, str] = {}
        for fid in ids:
            src, filename = self.resolve_file(fid)
            if src is None or not Path(src).is_file():
                continue
            size = Path(src).stat().st_size
            if self._bytes + size > self.max_bytes:
                raise Refused(f"the run's bridge data would exceed "
                              f"{self.max_bytes // (1024 * 1024)} MB")
            self._bytes += size
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", filename or Path(src).name)[:120] or fid
            dest = self.out / "data" / f"{len(self.calls) + 1:02d}_{safe}"
            shutil.copyfile(src, dest)
            local[fid] = self.container_path(dest)
        out = dict(result)
        if local:
            out["local_files"] = local
            out["path"] = local[ids[0]] if ids[0] in local else next(iter(local.values()))
        return out

    def _respond(self, rid: str, answer: Dict[str, Any]) -> None:
        resp = self.out / "resp"
        fd, tmp = tempfile.mkstemp(dir=resp, prefix=".", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(answer, fh, default=str)
        os.chmod(tmp, 0o644)
        os.replace(tmp, resp / f"{rid}.json")

    # -- recording ----------------------------------------------------------------------

    def _emit_call(self, fname: str, args: Dict[str, Any], call_id: str) -> None:
        if not self.record:
            return
        try:
            from agent_runtime.streaming_trace import emit_trace_event

            emit_trace_event("tool_call", {
                "kind": "llm_tool_decision", "label": "Tool started (from code)",
                "name": fname, "args": args, "via": "code_bridge", "call_id": call_id,
                "tool_calls": [{"name": fname, "args": args}],
                "message": f"{fname}({json.dumps(args, ensure_ascii=True, default=str)}) [from code]"})
        except Exception:  # noqa: BLE001 - tracing never costs the call
            pass

    def _record(self, fname: str, args: Dict[str, Any], call_id: str,
                record: Dict[str, Any]) -> None:
        if not self.record:
            return
        content = json.dumps(record, ensure_ascii=True, default=str)
        try:
            from agent_runtime import turn_log

            log = turn_log.active()
            if log is not None:
                peer, run = turn_log.active_peer() or "code", turn_log.active_run()
                log.record_call(peer=peer, run=run, name=fname, args=args, call_id=call_id)
                log.record_result(peer=peer, run=run, name=fname, args=args, call_id=call_id,
                                  content=content)
        except Exception:  # noqa: BLE001 - a broken log never costs the call
            logger.debug("code bridge: turn log record failed", exc_info=True)
        try:
            from agent_runtime.streaming_trace import emit_trace_event

            emit_trace_event("tool_result", {
                "kind": "tool_result", "label": f"Tool result {fname} (from code)",
                "tool_name": fname, "name": fname, "via": "code_bridge", "call_id": call_id,
                "content": content[:4000], "message": content[:4000],
                "outcome": "ok" if record.get("ok") else f"failed — {record.get('error')}"})
        except Exception:  # noqa: BLE001
            pass


def _summary(record: Dict[str, Any]) -> Dict[str, Any]:
    """What a run's result carries about one call: enough to name the source and the data."""
    keep = ("ok", "error", "source", "file_id", "filename", "path", "size_bytes", "count",
            "complete", "tiles", "feature_count", "region_bbox", "min", "max", "mean",
            "ground_resolution_m", "warning", "truncated", "shape", "content_type")
    return {k: record[k] for k in keep if k in record}


def bridge_root_for(work: Path) -> Path:
    """The run's bridge directory, beside its work dir (same root, so Docker can mount it)."""
    return work.parent / f"{work.name}_bridge"


__all__ = ["BridgeServer", "CLIENT_SOURCE", "CONTAINER_ROOT", "DEFAULT_HANDLERS", "ENABLE_ENV",
           "Refused", "bridge_root_for", "enabled", "model_note"]
