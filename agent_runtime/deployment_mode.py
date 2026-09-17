"""Which deployment this is, and therefore who it is for.

Three modes, exactly one active at a time:

* ``dev``   — the team. Connection settings are shown so a developer can point the page at a
              different endpoint or model. Service access is gated by ``AGENT_CHAT_API_KEY``
              exactly as it always has been.
* ``demo``  — an audience. No credential to paste, settings hidden, and the answering model is
              pinned by ``DEMO_MODEL`` rather than inherited from whatever this deployment
              happens to be set to.
* ``token`` — the integrated platform. The caller is identified by the platform's JWT, and
              conversations and files belong to that user.

Why a named mode instead of the booleans it replaces: three independent flags have eight
combinations and five of them are nonsense ("settings hidden AND a key required" is a page that
demands a credential it gives you no way to enter). A mode has three states and each one sets
every axis coherently.

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
MODES = (DEV, DEMO, TOKEN)

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
