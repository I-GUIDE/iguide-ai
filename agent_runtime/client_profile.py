"""Which client a turn is for: the map UI, or a chat-only client such as the I-GUIDE platform.

The agent grew up beside the map UI. It delivers results as interactive map layers
(``add_map_layer``, ``add_raster_layer``, the ``map_layer`` event), retries a peer that answered a
map request without one, and appends "the layer is already on your interactive map" when an
answer says otherwise. The platform calls the same ``/agent/chat/stream`` server-to-server and
shows only the answer text and the sources: no map, and every ``map_layer`` event dropped. A
platform user was therefore told about layers they could not see, and turns spent steps
producing them.

A chat-only turn keeps everything else (search, analysis, code, files), and turns off what only
a map can use:

* the two map tools are not bound to any peer (``MAP_ONLY_TOOLS``);
* no ``map_layer`` event is emitted;
* the "map requested but not delivered" retry and the "already on your map" correction are
  skipped;
* the answer writer is told the reply is shown without a map, and an answer that still claims a
  map gets a marked correction.

A turn is chat-only when any of these holds, checked per request:

* the request sets ``chatOnly: true`` (``chat_only``);
* it presents one of the keys in ``AGENT_CHAT_ONLY_KEYS`` (comma-separated) as its service key;
* it arrives from one of the addresses in ``AGENT_CHAT_ONLY_SOURCES`` (comma-separated). nginx
  sets ``X-Real-IP`` to the caller's address, and the API publishes its port on loopback only,
  so the header is nginx's. This is how the platform's backends become chat-only without a change
  on their side. A spoofed header can only make a turn chat-only, which removes capability and
  grants none.

The value is a ContextVar so every tool and check in the turn reads the same answer without
threading a flag through dozens of signatures. ``graph_runtime`` runs the graph under
``contextvars.copy_context()``, so the worker thread sees it.
"""
from __future__ import annotations

import os
from contextvars import ContextVar, Token
from typing import Iterable, Mapping, Optional

# Tools whose only output is a layer on the map. Everything else a map turn uses (overpass,
# admin_boundary, analysis tools) also writes a downloadable file, so it stays.
MAP_ONLY_TOOLS = frozenset({"add_map_layer", "add_raster_layer"})

# Search sources a chat-only client does not get (user, 2026-10-16): the platform's frontend reads
# its own element format, and hits from the Neo4j graph search and the agent's extracted KB do not
# fit it. The tools are not offered, and the backends refuse during a chat-only turn too, because
# deterministic paths (the search sweep, the KB join, popularity and by-id lookups) call them
# without any tool.
CHAT_ONLY_BLOCKED_TOOLS = frozenset({
    "neo4j_search", "neo4j_get_element_by_id", "neo4j_explore_related_nodes",
    "agent_kb_search", "get_kb_block", "kb_method_search", "get_method_contract",
})

_CHAT_ONLY: ContextVar[bool] = ContextVar("agent_chat_only", default=False)


def chat_only() -> bool:
    """True when the current turn is for a client with no map."""
    return bool(_CHAT_ONLY.get())


def graph_search_allowed() -> bool:
    """False during a chat-only turn: no Neo4j query runs (see CHAT_ONLY_BLOCKED_TOOLS)."""
    return not chat_only()


def agent_kb_allowed() -> bool:
    """False during a chat-only turn: the agent KB is not searched, read or joined."""
    return not chat_only()


def set_chat_only(on: bool) -> Token:
    """Bind this request's profile. Returns a token for ``reset_chat_only``."""
    return _CHAT_ONLY.set(bool(on))


def reset_chat_only(token: Optional[Token]) -> None:
    if token is not None:
        try:
            _CHAT_ONLY.reset(token)
        except ValueError:  # reset from another context: nothing to undo there
            pass


def _listed(env_name: str) -> frozenset:
    return frozenset(v.strip() for v in (os.getenv(env_name) or "").split(",") if v.strip())


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def wants_chat_only(body: Optional[Mapping] = None, *, presented_key: str = "",
                    source_addresses: Iterable[str] = ()) -> bool:
    """Whether a request should run chat-only. See the module docstring for the three rules."""
    body = body or {}
    if _truthy(body.get("chatOnly", body.get("chat_only"))):
        return True
    if presented_key and presented_key in _listed("AGENT_CHAT_ONLY_KEYS"):
        return True
    sources = _listed("AGENT_CHAT_ONLY_SOURCES")
    if sources and any(str(a).strip() in sources for a in source_addresses if a):
        return True
    return False


# What the answer writer is told on a chat-only turn. A capability statement, not a mandate
# (AGENTS.md, "Capability statements, not mandates"); `correct_map_claims` checks the result.
CHAT_ONLY_NOTE = (
    "This reply is shown in a text-only chat: there is no map beside it, and no layer is drawn "
    "anywhere the user can see. Results reach the user as the answer text, the cited sources and "
    "the downloadable files linked in the answer. Describe a spatial result by what it contains "
    "and link its file; do not say it is on a map or tell the user to pan, zoom or click it."
)

# Appended when a chat-only answer still talks about a map.
CHAT_ONLY_CORRECTION = (
    "There is no map in this view, so nothing above is displayed on one. Any spatial result is "
    "available through the download links in this answer."
)


# --------------------------------------------------------------------------- answer validation
#
# After the work is done, a turn checks its answer: the grounding audit (an LLM pass over the
# answer and the turn's record), the number scan, a re-grounding pass that sends the work back to
# a peer when a statement is unsupported, and the cross-run agreement and reference
# recomputations. On Lumen deepseek that took 63% of the Chicago SKATER step (4 min 55 s of
# 7 min 48 s, 2026-10-15), most of it two audits and a second analysis run for one sentence.
#
# The switch turns all of it off, for a client or a demo that wants the answer sooner and accepts
# that it is unchecked. The turn then says so: its verdict is "unverified", with one finding
# naming the reason, never "verified".
#
# Per request, `validation: false` in the body (`validate: false` is accepted too); otherwise
# `AGENT_VALIDATION` (default on; 0/false/off/no turns it off for every turn).

VALIDATION_ENV = "AGENT_VALIDATION"
# A tool a peer may call to audit its own answer; it is not offered when validation is off.
VALIDATION_TOOLS = frozenset({"audit_answer_grounding"})

_VALIDATION: ContextVar[Optional[bool]] = ContextVar("agent_validation", default=None)


def validation_default() -> bool:
    raw = (os.getenv(VALIDATION_ENV) or "").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def validation_enabled() -> bool:
    """Whether this turn checks its answer. The request's choice, else ``AGENT_VALIDATION``."""
    value = _VALIDATION.get()
    return validation_default() if value is None else bool(value)


def set_validation(on: Optional[bool]) -> Token:
    """Bind this request's choice (None = the server default). Returns a token for reset."""
    return _VALIDATION.set(None if on is None else bool(on))


def reset_validation(token: Optional[Token]) -> None:
    if token is not None:
        try:
            _VALIDATION.reset(token)
        except ValueError:
            pass


def requested_validation(body: Optional[Mapping] = None) -> Optional[bool]:
    """The request's `validation` / `validate` field as a bool, or None when it sets neither."""
    body = body or {}
    for key in ("validation", "validate"):
        if key in body and body[key] is not None:
            value = body[key]
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in {"0", "false", "off", "no"}:
                return False
            if text in {"1", "true", "on", "yes"}:
                return True
    return None


VALIDATION_OFF_FINDING = ("not checked: answer validation is turned off for this turn, so no "
                          "statement in it was compared with the turn's record")


__all__ = ["MAP_ONLY_TOOLS", "CHAT_ONLY_BLOCKED_TOOLS", "graph_search_allowed",
           "agent_kb_allowed", "chat_only", "set_chat_only", "reset_chat_only", "wants_chat_only",
           "CHAT_ONLY_NOTE", "CHAT_ONLY_CORRECTION", "VALIDATION_ENV", "VALIDATION_TOOLS",
           "validation_enabled", "validation_default", "set_validation", "reset_validation",
           "requested_validation", "VALIDATION_OFF_FINDING"]
