"""Self-description: answer "what tools do you have" from the LIVE tool registries.

In the supervisor architecture no LLM sees the full tool registry (the decider is a bare
router, the search peer's prose is discarded, and the synthesizer is tool-free and
grounded-only), so a meta question like "what tools do you have" retrieved irrelevant KB
documents or hit the no-grounding reply.

This module answers those questions WITHOUT hardcoded capability prose:

1. ``collect_capability_inventory`` reads the actual tool factories — every tool's real name and
   description, plus deployment flags (code execution, skills, MCP) — so a tool added, removed,
   or gated anywhere in the codebase changes this answer with no edit here.
2. ``describe_capabilities`` hands that inventory to the LLM (``CAPABILITY_SUMMARY_PROMPT``),
   which writes it up in plain language. No capability sentence is authored here.

The only non-LLM path is a mechanical fallback that lists the inventory verbatim when no model
is reachable — still derived from the registries, nothing to maintain by hand.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

# Meta/self-descriptive phrasings only. Deliberately requires the question to be about the
# ASSISTANT (you/your, or a bare anchored "list/show tools") so domain questions like
# "what tools are available for flood mapping" still go through normal retrieval.
_CAPABILITY_RE = re.compile(
    r"(?:\b(?:what|which)\s+tools?\s+(?:do\s+you|can\s+you|are\s+you|you)\b"
    r"|\byour\s+(?:tools?|capabilities|skills?)\b"
    r"|^\s*what\s+can\s+you\s+do\b"
    r"|^\s*what\s+are\s+you\s+able\s+to\s+do\b"
    r"|\bwhat\s+(?:kind\s+of\s+)?(?:tools?|skills?|capabilities)\s+do\s+you\s+have\b"
    r"|^\s*(?:list|show)(?:\s+me)?\s+(?:your\s+)?(?:available\s+)?(?:tools?|capabilities|skills?)\s*\??\s*$)",
    re.IGNORECASE,
)


def is_capability_query(query: str) -> bool:
    """True when *query* asks about the assistant's own tools/capabilities/skills."""
    return bool(_CAPABILITY_RE.search(query or ""))


def _tool_entries(factory, **kwargs) -> List[Dict[str, str]]:
    """``[{name, description}]`` from a registry factory; [] when it is unavailable."""
    out: List[Dict[str, str]] = []
    try:
        for tool in (factory(**kwargs) or []):
            name = str(getattr(tool, "name", "") or "").strip()
            if not name:
                continue
            desc = " ".join(str(getattr(tool, "description", "") or "").split())
            out.append({"name": name, "description": desc[:400]})
    except Exception:
        return out
    return out


def collect_capability_inventory(
    *,
    enabled_search_methods: Optional[List[str]] = None,
    include_mcp_tools: Optional[bool] = None,
    mcp_modules: Optional[List[str]] = None,
    code_exec: Optional[bool] = None,
    skill_roots: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """The assistant's live capability inventory, read from the real tool registries.

    Probes the registries UNFILTERED: a per-request search-method allowlist or MCP flag is a
    client choice, not a limit on what the assistant can do. Deployment-level gates are reported
    as facts instead (code execution on/off, which optional backends are present), so the answer
    is truthful for the running deployment. Every registry is probed independently and failures
    degrade to omission — this never raises.
    """
    tools: List[Dict[str, str]] = []
    try:
        from agent_runtime.langchain_granular_tools import (
            make_langchain_geocode_tools,
            make_langchain_granular_tools,
        )

        tools += _tool_entries(make_langchain_granular_tools,
                               enabled_search_methods=None, include_file_tools=True)
        tools += _tool_entries(make_langchain_geocode_tools)
    except Exception:
        pass
    try:
        from agent_runtime.langchain_geo_tools import make_langchain_geo_tools

        tools += _tool_entries(make_langchain_geo_tools, default_input_file_ids=None)
    except Exception:
        pass
    # Overlay / aggregation / temporal analysis registries. Probed unfiltered like the
    # rest: the analyze and code peers only load these once files are attached, but
    # "can you clip a layer?" is true of the deployment regardless of what is attached
    # right now, so the inventory must report them.
    try:
        from agent_runtime.analysis_overlay_tools import make_overlay_tools

        tools += _tool_entries(make_overlay_tools, default_input_file_ids=None)
    except Exception:
        pass
    try:
        from agent_runtime.analysis_aggregate_tools import make_aggregate_tools

        tools += _tool_entries(make_aggregate_tools, default_input_file_ids=None)
    except Exception:
        pass
    try:
        from agent_runtime.analysis_temporal_tools import make_temporal_tools

        tools += _tool_entries(make_temporal_tools, default_input_file_ids=None)
    except Exception:
        pass
    try:
        from agent_runtime.analysis_spatial_stats_tools import make_spatial_stats_tools

        tools += _tool_entries(make_spatial_stats_tools, default_input_file_ids=None)
    except Exception:
        pass
    try:
        from extractors.geo_handles import make_geo_analysis_tools

        tools += _tool_entries(make_geo_analysis_tools)
    except Exception:
        pass

    mcp_on = include_mcp_tools
    if mcp_on is None:
        try:
            from agent_runtime.langchain_mcp_tools import mcp_tools_enabled

            mcp_on = bool(mcp_tools_enabled())
        except Exception:
            mcp_on = False
    if mcp_on:
        try:
            from agent_runtime.langchain_mcp_tools import make_langchain_mcp_tools

            tools += _tool_entries(make_langchain_mcp_tools,
                                   include_modules=mcp_modules or ["spatial_analysis_tools"])
        except Exception:
            pass

    exec_enabled: Optional[bool] = None
    exec_backend: Optional[str] = None
    try:
        from agent_runtime.code_execution import get_code_executor, is_code_exec_enabled

        exec_enabled = bool(code_exec) if code_exec is not None else is_code_exec_enabled()
        if exec_enabled:
            exec_backend = str(getattr(get_code_executor(), "backend", "") or "") or None
    except Exception:
        pass

    skills: List[Dict[str, str]] = []
    try:
        from agent_runtime.skills import SkillRegistry

        for entry in SkillRegistry.discover(skill_roots).catalog():
            name = str(entry.get("name") or "").strip()
            if name:
                skills.append({"name": name,
                               "description": " ".join(str(entry.get("description") or "").split())[:300]})
    except Exception:
        pass

    # De-duplicate by tool name, preserving first-seen order.
    seen: set = set()
    unique: List[Dict[str, str]] = []
    for entry in tools:
        if entry["name"] in seen:
            continue
        seen.add(entry["name"])
        unique.append(entry)

    return {
        "tools": unique,
        "code_execution": {"enabled": exec_enabled, "sandbox_backend": exec_backend},
        "skills": skills,
    }


def _mechanical_summary(inventory: Dict[str, Any]) -> str:
    """Last-resort listing of the inventory when no LLM is reachable.

    Mechanically derived from the registries (no authored capability text), so it also stays
    correct as tools change.
    """
    lines: List[str] = ["Available capabilities:"]
    for entry in inventory.get("tools") or []:
        desc = entry.get("description") or ""
        first = desc.split(". ")[0].rstrip(".")
        lines.append(f"- {entry['name']}" + (f" — {first}." if first else ""))
    code = inventory.get("code_execution") or {}
    if code.get("enabled") is not None:
        lines.append(f"- code execution: {'enabled' if code.get('enabled') else 'disabled'}"
                     + (f" (sandbox: {code.get('sandbox_backend')})" if code.get("sandbox_backend") else ""))
    for skill in inventory.get("skills") or []:
        lines.append(f"- skill: {skill['name']}")
    return "\n".join(lines)


def describe_capabilities(*, llm: Optional[Any] = None, **config: Any) -> str:
    """Answer a capability question in plain language, composed by the LLM from the live
    inventory. Falls back to a mechanical listing of that same inventory if no model answers.

    ``config`` is forwarded to :func:`collect_capability_inventory`.
    """
    inventory = collect_capability_inventory(**config)
    try:
        from agent_runtime.prompts import CAPABILITY_SUMMARY_PROMPT

        active = llm
        if active is None:
            from agent_runtime.executor_factory import build_default_llm

            active = build_default_llm()
        prompt = CAPABILITY_SUMMARY_PROMPT.format(
            inventory=json.dumps(inventory, ensure_ascii=True, indent=1)[:12000]
        )
        if hasattr(active, "invoke"):
            raw = active.invoke(prompt)
            content = getattr(raw, "content", raw)
            if isinstance(content, list):
                text = "".join(
                    str(p.get("text") or p.get("content") or "") if isinstance(p, dict)
                    else str(getattr(p, "text", p)) for p in content
                )
            else:
                text = str(content or "")
        elif callable(active):
            text = str(active(prompt))
        else:
            text = ""
        if text.strip():
            return text.strip()
    except Exception:
        pass
    return _mechanical_summary(inventory)


__all__ = ["is_capability_query", "collect_capability_inventory", "describe_capabilities"]
