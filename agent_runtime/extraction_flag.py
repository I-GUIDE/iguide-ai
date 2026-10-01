"""One switch for the extraction bundle, OFF by default.

The bundle is everything the knowledge-element extraction work adds to the agent: the method-library
tools (`kb_method_search`, `get_method_contract`), the staging tools (`stage_element`, `stage_url`,
`list_staged_inputs`), the read-only library mount in the code sandbox, the in-sandbox invariant
gate, run-artifact emission, the deterministic sweep's method-library arm, and every prompt rule,
tool description and capability clause that tells a model about any of them.

OFF means the agent behaves exactly as prototype did before the 2026-10-01 integration, so the
merge itself changes nothing in production until someone turns this on deliberately. It also
means the model is never TOLD about a capability it does not have: a peer instructed to call
`kb_method_search` without it guessed the package name and wrote `from method_library import ...`,
which fails. So a description is gated with the tool it describes, never separately.

Finer switches still win where they exist: an explicit ``AGENT_INVARIANT_GATE`` or
``AGENT_ARTIFACT_EMIT`` overrides the default this flag supplies. Read at call time, never
captured at import, so a test or a harness can flip it per run.
"""
from __future__ import annotations

import os

FLAG_ENV = "AGENT_EXTRACTION"
_TRUE = {"1", "true", "yes", "on"}


def extraction_enabled() -> bool:
    """Whether the extraction bundle is ON for this process. Default: off."""
    return (os.getenv(FLAG_ENV) or "").strip().lower() in _TRUE


__all__ = ["FLAG_ENV", "extraction_enabled"]
