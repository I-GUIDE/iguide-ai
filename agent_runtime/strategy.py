"""Per-request orchestration configuration, and the entrypoint that runs it.

There was a second path here — ``agent_runtime.legacy``, the agents-as-tools shape the
supervisor replaced in 2026-06 — selectable with ``AGENT_SUPERVISOR=0`` or a per-request
``useSupervisor: false``. It was removed once it stopped being a fallback and became a trap:
it had not been touched since 2026-06-25 and had none of map-layer delivery, the action ledger,
terrain tools, the capability registry, the evidence summary or the grounding gate, and could
not reach roughly half the tool surface even in principle, because the peer builders construct
those toolsets directly and ``collect_tools`` does not.

The registry shape is kept even with one entry: it is what made the paths independent, and it
is the seam a genuine second path would use again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class OrchestrationConfig:
    """Per-request orchestration configuration passed to the selected strategy."""

    llm: Optional[Any] = None
    verbose: bool = False
    return_intermediate_steps: bool = True
    tool_strategy: str = "granular"
    include_mcp_tools: bool = False
    mcp_modules: Optional[List[str]] = None
    enabled_search_methods: Optional[List[str]] = None
    smart_tool_routing: bool = True
    forced_intent: Optional[str] = None
    thread_id: Optional[str] = None
    checkpointer: Optional[Any] = None
    skill_roots: Optional[List[str]] = None
    code_exec: Optional[bool] = None
    input_file_ids: Optional[List[str]] = None
    # Per-request override of AGENT_UNIFIED_PEER: run search+analyze as ONE agent. None means
    # "use the env default", so an unset request keeps whatever the deployment is configured for.
    unified_peer: Optional[bool] = None
    # Which code-peer backend this request wants; None falls back to AGENT_CODE_PEER.
    code_peer: Optional[str] = None
    # Model for that peer when it is a CLI backend; None falls back to its own env default.
    code_peer_model: Optional[str] = None


# (query, chat_history, cfg) -> orchestration result dict (OrchestratorState keys)
OrchestrationStrategy = Callable[[str, Optional[List[Any]], OrchestrationConfig], Dict[str, Any]]


def get_orchestration_strategy() -> OrchestrationStrategy:
    """The orchestration entrypoint. Imported lazily to keep import cost off the module load."""
    from agent_runtime.supervisor.orchestration import run_supervisor_orchestration

    return run_supervisor_orchestration


__all__ = ["OrchestrationConfig", "OrchestrationStrategy", "get_orchestration_strategy"]
