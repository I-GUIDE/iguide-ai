"""Which deployment this is, and therefore who it is for.

Four modes, exactly one active at a time:

* ``dev``   — the team. Connection settings are shown so a developer can point the page at a
              different endpoint or model. Service access is gated by ``AGENT_CHAT_API_KEY``
              exactly as it always has been.
* ``demo``  — an audience. No credential to paste, settings hidden, and the answering model is
              pinned by ``DEMO_MODEL`` rather than inherited from whatever this deployment
              happens to be set to.
* ``token`` — the integrated platform. The caller is identified by the platform's JWT, and
              conversations and files belong to that user.
* ``local`` — one developer's machine. Everything ``dev`` does for ACCESS (no identity, settings
              shown, model chosen per request), plus one guarantee ``dev`` does not make: this
              process writes NOTHING to a shared store. Persistent memory is off whatever the
              request asks, the memory store refuses to open a client at all, and download links
              stay host-relative so they resolve on the machine serving them.

Why ``local`` exists rather than a recipe of environment variables
-------------------------------------------------------------------
The main checkout's ``.env`` points a laptop at shared infrastructure — the PRODUCTION OpenSearch
cluster among it — so the safe way to run locally used to be a list of overrides. That list
failed twice. On 2026-10-01 a local verification run wrote seven conversations into prod before
anyone noticed. The fix recorded afterwards (``PLATFORM_TIER=dev`` plus a blank
``OPENSEARCH_NODE``) still wrote, to the DEV cluster instead: with the explicit host blank, the
tier supplies one. A guarantee that depends on remembering eight variables correctly is not a
guarantee, so it became one switch.

Knowledge-base search keeps READING. Memory and search used to share ``OPENSEARCH_NODE``, which
made "no memory writes" imply "no search" — and a model tested without search is being tested on
a crippled agent. In local mode the memory store never connects, while the search modules, which
build their own clients, read exactly as before.

The client already uses the word "local" for its MOCK mode (``runLocal`` in the map UI's
``App.tsx``: no server at all). That is a different variable from this one and nothing collides
at runtime, but a reader moving between the two should not assume they mean the same thing.

Why a named mode instead of the booleans it replaces: three independent flags have eight
combinations and five of them are nonsense ("settings hidden AND a key required" is a page that
demands a credential it gives you no way to enter). A mode has a handful of states and each one
sets every axis coherently.

What this module deliberately does NOT decide
---------------------------------------------
**The API key.** ``AGENT_CHAT_API_KEY`` keeps governing service access on its own, in every
mode. Letting the mode imply the key would make ``AGENT_MODE=dev`` mean one thing on a laptop
(harmless, nothing exposed) and something else on the deployed dev tier, which is public and
reachable by anyone who finds the URL. A developer running locally simply does not set the key.

**The model**, except in demo. ``demo`` pins one because a demo is handed to an audience and
"whatever this deployment happens to be set to" is not a demo decision; the other modes take the
model from the request.

An unknown ``AGENT_MODE`` RAISES rather than falling back. This selects security behaviour, and
a typo that silently resolves to a working mode is precisely the failure that would go unnoticed
on a public host — a container that refuses to boot is noticed immediately.
"""

from __future__ import annotations

import os
from typing import Optional

DEV = "dev"
DEMO = "demo"
TOKEN = "token"
LOCAL = "local"
MODES = (DEV, DEMO, TOKEN, LOCAL)

# Shared with the legacy DEMO_MODE flag so both spell truth the same way.
_TRUTHY = {"1", "true", "yes", "on"}


def _legacy_demo_flag() -> bool:
    """The pre-mode ``DEMO_MODE`` boolean, still honoured when ``AGENT_MODE`` is unset."""
    return str(os.getenv("DEMO_MODE") or "").strip().lower() in _TRUTHY


def current_mode() -> str:
    """The active mode.

    ``AGENT_MODE`` wins when set. With it unset, a deployment still carrying the old
    ``DEMO_MODE=true`` keeps behaving exactly as it did — this has to stay true, because the
    running deployment is configured that way and a refactor that quietly changes what a live
    server does is not a refactor.
    """
    raw = str(os.getenv("AGENT_MODE") or "").strip().lower()
    if raw:
        if raw not in MODES:
            raise ValueError(
                f"AGENT_MODE={raw!r} is not a mode. Expected one of: {', '.join(MODES)}.")
        return raw
    return DEMO if _legacy_demo_flag() else DEV


def is_dev() -> bool:
    return current_mode() == DEV


def is_demo() -> bool:
    return current_mode() == DEMO


def is_token() -> bool:
    return current_mode() == TOKEN


def is_local() -> bool:
    return current_mode() == LOCAL


def persistent_memory_allowed() -> bool:
    """Whether this process may read or write the shared conversation store at all.

    The single source of truth for the local-mode guarantee. It is checked in three places on
    purpose — where a request's ``use_persistent_memory`` is normalised, at the conversation
    endpoints, and inside the memory store's own client — because the failure it prevents was
    silent: nothing errors when a local run writes to production, it simply succeeds.
    """
    return not is_local()


# Hosts that are this machine. Anything else is named as REMOTE in the local-mode banner.
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "[::1]", "::1")

# What a local process may still contact, and why — the endpoints a developer would otherwise
# only discover from traffic. Ordered by how much it matters to know.
_ENDPOINTS = (
    ("OPENSEARCH_NODE", "knowledge-base search — READ ONLY; persistent memory is off"),
    ("FLASK_EMBEDDING_URL", "query embeddings for knowledge-base search"),
    ("ANVILGPT_URL", "language model (AnvilGPT-compatible)"),
    ("OPENAI_BASE_URL", "language model (OpenAI-compatible)"),
    ("VLLM_BASE_URL", "language model (vLLM)"),
    ("RS_EMBED_URL", "satellite embedding service"),
    ("MCP_SERVER_URL", "remote MCP tools"),
    ("MINIO_ENDPOINT", "object storage"),
)


def _is_local_url(url: str) -> bool:
    low = url.lower()
    return any(f"://{h}" in low for h in _LOCAL_HOSTS)


def local_mode_report() -> list:
    """Boot lines for ``AGENT_MODE=local``: what this process will contact, local or remote.

    Empty outside local mode. Values are hosts and URLs only — the same rule the rest of the boot
    log follows — and credentials embedded in a URL are masked. ``AGENT_PUBLIC_BASE_URL`` is listed
    as IGNORED rather than omitted, because a variable set in ``.env`` that silently does nothing
    is its own kind of surprise.
    """
    if not is_local():
        return []
    import re

    lines = ["AGENT_MODE=local: persistent memory OFF (no conversation, snapshot or trace is "
             "written anywhere). Endpoints this process may contact:"]
    for name, role in _ENDPOINTS:
        url = str(os.getenv(name) or "").strip().strip('"').strip("'")
        if not url:
            continue
        shown = re.sub(r"(://)[^@/]+@", r"\1***@", url)
        where = "local " if _is_local_url(url) else "REMOTE"
        lines.append(f"  [{where}] {name}={shown}  — {role}")
    public = str(os.getenv("AGENT_PUBLIC_BASE_URL") or "").strip()
    if public:
        lines.append(f"  [ignored] AGENT_PUBLIC_BASE_URL={public}  — download links stay "
                     "host-relative in local mode")
    return lines


def boot_warning() -> Optional[str]:
    """The one line worth logging at startup, or None when the mode needs no warning.

    Emitted once at import rather than per request: a deployment being open should never be
    something an operator discovers from its behaviour.
    """
    mode = current_mode()
    if mode == DEMO:
        return ("AGENT_MODE=demo: the API key is NOT enforced and the UI hides its connection "
                "settings. Every agent endpoint is open to anyone who can reach this server.")
    if mode == TOKEN:
        return None
    return None
