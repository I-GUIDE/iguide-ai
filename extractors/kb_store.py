"""Agent-KB storage backend selector — LOCAL by default (safe).

The agent KB can live in real OpenSearch OR a local file-backed store. **Default is
local**: nothing is written to (or read from) the production OpenSearch cluster
unless ``AGENT_KB_BACKEND=opensearch`` is set explicitly. This guarantees that
ingestion + agent retrieval are fully runnable on a laptop with no cluster, and that
testing never touches the real database.

Local store layout: ``storage_root()/agent_kb/<index>.json`` (a ``{doc_id: _source}``
map per agent index). ``local_search`` does simple token-overlap scoring over
title/contents/embed_text — enough to exercise the agent's retrieval path offline.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def kb_backend() -> str:
    """'local' (default) or 'opensearch' (explicit opt-in)."""
    return os.getenv("AGENT_KB_BACKEND", "local").strip().lower()


def using_real_db() -> bool:
    return kb_backend() == "opensearch"


def store_dir() -> Path:
    """Where the local store lives — overridable, for the same reason the method library is.

    ``AGENT_KB_STORE_DIR`` exists so a test suite can point the store somewhere empty. Without it
    the suite reads whatever the developer last ingested: three tests that fake every other
    retrieval arm (``test_sweep_adds_implied_methods``, ``test_direct_search_sweep_drops_unlisted``,
    ``test_the_every_turn_sweep_still_never_touches_the_web``) pass on a clean checkout and fail
    the moment the corpus is indexed, because the sweep unions the agent KB deterministically.
    Same code, same commit, different machine state — which is the failure mode
    ``AGENT_METHOD_LIBRARY_DIR`` was already added to prevent for the library half.
    """
    configured = os.getenv("AGENT_KB_STORE_DIR")
    if configured:
        p = Path(configured).expanduser()
    else:
        from agent_runtime.file_store import storage_root
        p = Path(storage_root()) / "agent_kb"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _index_file(index: str) -> Path:
    return store_dir() / f"{index}.json"


def _load(index: str) -> Dict[str, Any]:
    f = _index_file(index)
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(index: str, data: Dict[str, Any]) -> None:
    _index_file(index).write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")


def local_upsert(index: str, doc_id: str, source: Dict[str, Any]) -> None:
    data = _load(index)
    data[doc_id] = source
    _save(index, data)


def local_get(doc_id: str, indices: List[str]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Return (index, source) for a doc_id across indices, or (None, None)."""
    for idx in indices:
        data = _load(idx)
        if doc_id in data:
            return idx, data[doc_id]
    return None, None


def local_blocks_for_parent(element_id: str, indices: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
    """All locally-stored blocks whose parent element is ``element_id`` (sorted by
    block order). Lets a consumer fetch a whole element by its bare element_id."""
    found: List[Tuple[str, Dict[str, Any]]] = []
    for idx in indices:
        for did, src in _load(idx).items():
            if did == element_id:
                continue
            ex = src.get("extracted") or {}
            parent = ex.get("parent_doc_id") or (did.split("::", 1)[0] if "::" in did else did)
            if parent == element_id:
                found.append((did, src))

    def _order(item: Tuple[str, Dict[str, Any]]) -> int:
        try:
            return int(item[0].rsplit("::", 1)[-1])
        except Exception:
            return 9999

    return sorted(found, key=_order)


def local_all(indices: List[str]) -> List[Tuple[str, str, Dict[str, Any]]]:
    out: List[Tuple[str, str, Dict[str, Any]]] = []
    for idx in indices:
        for doc_id, src in _load(idx).items():
            out.append((idx, doc_id, src))
    return out


def _score(query: str, source: Dict[str, Any]) -> float:
    q = [t for t in query.lower().split() if t]
    if not q:
        return 0.0
    text = " ".join(str(source.get(k, "")) for k in ("title", "contents")).lower()
    ex = source.get("extracted") or {}
    if isinstance(ex, dict):
        text += " " + str(ex.get("embed_text", "")).lower()
    return float(sum(1 for t in q if t in text))


def local_search(query: str, indices: List[str], size: int) -> List[Dict[str, Any]]:
    """Return OpenSearch-hit-shaped dicts ranked by token overlap."""
    scored = []
    for idx, doc_id, src in local_all(indices):
        s = _score(query, src)
        if s > 0:
            scored.append((s, idx, doc_id, src))
    scored.sort(key=lambda x: -x[0])
    return [{"_index": idx, "_id": doc_id, "_score": s, "_source": src}
            for s, idx, doc_id, src in scored[:size]]


def clear_local() -> None:
    shutil.rmtree(store_dir(), ignore_errors=True)


__all__ = ["kb_backend", "using_real_db", "store_dir", "local_upsert",
           "local_all", "local_search", "clear_local"]
