"""Agent-only OpenSearch index routing.

Extracted contents live in SEPARATE indices from the general platform search index
(``OPENSEARCH_INDEX``), so they are NOT discoverable by general platform search —
only by the agent's search peer (which must be pointed at these indices; see
EXTRACTOR_DESIGN.md "Agent-only indices"). One index per resource-type under a
configurable prefix.
"""

from __future__ import annotations

import os
from typing import Dict


def agent_index_prefix() -> str:
    return os.getenv("AGENT_KB_INDEX_PREFIX", "iguide_agent_")


def _by_resource_type() -> Dict[str, str]:
    p = agent_index_prefix()
    return {
        "NotebookBlock": f"{p}notebook_blocks",
        "CodeAsset": f"{p}code_assets",
        "Dataset": f"{p}datasets",
        "PublicationMethodSpec": f"{p}publication_methodspecs",
        "MethodUnit": f"{p}method_units",
    }


def index_for(resource_type: str) -> str:
    """Agent-only index for a resource-type. Unknown types fall to a catch-all
    agent index (still separate from the general OPENSEARCH_INDEX)."""
    return _by_resource_type().get(resource_type, f"{agent_index_prefix()}misc")


def all_agent_indices() -> list[str]:
    """Every agent-only index (e.g. for the search peer to query)."""
    return sorted(set(_by_resource_type().values()))


def is_agent_index(name: str) -> bool:
    return bool(name) and name.startswith(agent_index_prefix())


__all__ = ["agent_index_prefix", "index_for", "all_agent_indices", "is_agent_index"]
