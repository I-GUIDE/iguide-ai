"""Whose agent state a ``threadId`` names: the client's id, scoped to the caller (S9.15).

The client chooses ``threadId``, and the map UI prints it in every answer (it is the conversation
id, ``sess-<uuid>``). Everything the agent keeps between turns is keyed by it: the LangGraph
checkpointer's peer threads, session memory, session files, the action ledger, the sandbox and
Claude-peer workspaces on disk, staged inputs and the lineage graph. The chat edge checks the
conversation document (S9.14), but a turn that sent its OWN ``memoryId`` with someone else's
``threadId`` passed that check, and the agent then ran on the other user's state. Measured
offline: Bob's analysis peer was handed Alice's earlier questions, file id and filename, and his
sandbox runs saw her workspace.

So the key the agent uses is the client's id plus a digest of the signed-in user. Two users who
send the same ``threadId`` get two keys, whatever else they send, with no registry to keep and
nothing to read from the store. The client never sees the scoped key: the chat service hands it
to the agent and translates it back in everything it returns.

With nobody signed in (dev, demo, the service key) the key is the client's id, unchanged, as
before. An unsigned id that already carries a scope is scoped again, so a service-key caller
cannot name a signed-in user's state by spelling out its key.
"""
from __future__ import annotations

import hashlib
from typing import Optional

from agent_runtime import identity

SEP = "@u-"
_ANON = "anon"


def _tag(user_id: str) -> str:
    # A digest, not the id: platform ids are URLs, and this key ends up in directory names,
    # checkpoint ids and log lines.
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]


def scoped(thread_id: Optional[str]) -> Optional[str]:
    """The key the agent's per-conversation state is stored under, for the current caller."""
    if not thread_id:
        return thread_id
    key = str(thread_id)
    user = identity.current_user_id()
    if user:
        return f"{key}{SEP}{_tag(user)}"
    return f"{key}{SEP}{_ANON}" if SEP in key else key


def unscoped(key: Optional[str]) -> Optional[str]:
    """The client's id inside a scoped key (or a child of one, ``<key>::<label>``)."""
    if not key or SEP not in str(key):
        return key
    head, _, rest = str(key).rpartition(SEP)
    _, sep, label = rest.partition("::")
    return f"{head}{sep}{label}"


__all__ = ["SEP", "scoped", "unscoped"]
