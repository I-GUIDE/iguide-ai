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

_CHAT_ONLY: ContextVar[bool] = ContextVar("agent_chat_only", default=False)


def chat_only() -> bool:
    """True when the current turn is for a client with no map."""
    return bool(_CHAT_ONLY.get())


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


__all__ = ["MAP_ONLY_TOOLS", "chat_only", "set_chat_only", "reset_chat_only", "wants_chat_only",
           "CHAT_ONLY_NOTE", "CHAT_ONLY_CORRECTION"]
