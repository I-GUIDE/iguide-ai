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
   which answers the user's ACTUAL question over it.

``describe_capabilities`` is a ReAct loop, not a one-shot composer: the agent introspects its
own surface by topic (``list_my_capabilities``), can look again with a different word, and can
call the real listing tools to name actual models. A fixed prompt could only ever answer the
question it was written for — the previous version received no query at all and returned the
same grouped catalogue however specific the question was.

The only non-LLM path is a mechanical fallback that lists the inventory verbatim when no model
is reachable — still derived from the registries, nothing to maintain by hand.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

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
        # NOT silent. A wrong kwarg here (`modules=` against a keyword-only `include_modules=`)
        # read as "this registry has no tools" and hid 12 MCP tools from the very map built to
        # stop tools being invisible — the same swallowed-failure mechanism, one level up.
        logger.warning("capability registry %s could not be listed",
                       getattr(factory, "__name__", factory), exc_info=True)
        return out
    return out


# Modules that hold tool registries, by convention plus two known outliers. Scoped rather than
# importing all of agent_runtime: most modules here are not registries and some are expensive.
_REGISTRY_MODULE_GLOB = "*_tools.py"
_REGISTRY_EXTRA_MODULES = ("agent_runtime.skills", "extractors.geo_handles")
_REGISTRY_FACTORY_RE = re.compile(r"^make_.*tools$")


def _discover_registry_factories() -> List[tuple]:
    """(name, callable) for every tool-registry factory this deployment has."""
    import importlib
    from pathlib import Path

    found: Dict[str, Any] = {}
    module_names: List[str] = []
    try:
        import agent_runtime

        pkg_dir = Path(list(agent_runtime.__path__)[0])
        module_names += [f"agent_runtime.{p.stem}"
                         for p in sorted(pkg_dir.glob(_REGISTRY_MODULE_GLOB))]
    except Exception:
        pass
    module_names += list(_REGISTRY_EXTRA_MODULES)

    for mod_name in module_names:
        try:
            module = importlib.import_module(mod_name)
        except Exception:      # an optional backend that is not installed simply has no tools
            continue
        for attr in dir(module):
            if not _REGISTRY_FACTORY_RE.match(attr):
                continue
            fn = getattr(module, attr, None)
            if callable(fn):
                found.setdefault(attr, fn)
    return sorted(found.items())


def _factory_kwargs(factory: Any, **available: Any) -> Dict[str, Any]:
    """Only the kwargs *factory* actually accepts — the factories differ in signature."""
    try:
        import inspect

        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return dict(available)
    return {k: v for k, v in available.items() if k in params}


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
    # DISCOVERED, not listed. This function used to carry one hand-written try/except per
    # registry — 20 of them — and a hand-written list is wrong the moment someone adds a module
    # and forgets this file. It was wrong twice over: the satellite-embedding registries were
    # absent for weeks (so "what can you do" never mentioned embeddings while every peer could
    # run them), and an audit right after adding them found make_langchain_file_tools and
    # make_quality_tools still missing.
    #
    # Convention: a `make_*tools` callable in an `agent_runtime/*_tools.py` module (plus the two
    # known outliers) is a registry. Add one and it appears here with no edit.
    for label, factory in _discover_registry_factories():
        if label == "make_langchain_mcp_tools":
            continue                       # gated below on the MCP flag
        tools += _tool_entries(
            factory,
            **_factory_kwargs(factory,
                              default_input_file_ids=None,
                              enabled_search_methods=None,
                              skill_roots=skill_roots,
                              session_id=None),
        )

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

            tools += _tool_entries(make_langchain_mcp_tools, include_modules=mcp_modules)
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


def _area_id(factory_name: str) -> str:
    """A readable area id from a factory name — mechanical, so a new registry names itself.

    make_rs_embed_zonal_tools -> rs_embed_zonal ; make_langchain_granular_tools -> granular
    """
    name = factory_name
    for prefix in ("make_",):
        if name.startswith(prefix):
            name = name[len(prefix):]
    if name.endswith("_tools"):
        name = name[: -len("_tools")]
    if name.startswith("langchain_"):
        name = name[len("langchain_"):]
    return name or factory_name


def _safe_inventory_bits(**config: Any) -> Dict[str, Any]:
    """Deployment facts that belong in the overview (code exec on/off, installed skills)."""
    inv = collect_capability_inventory(**config)
    return {"code_execution": inv.get("code_execution"), "skills": inv.get("skills")}


def _registry_areas(**config: Any) -> List[Dict[str, Any]]:
    """The capability tree: one AREA per discovered registry, its tools beneath it.

    The hierarchy is free — it is the registry structure the code already has, so a new module
    becomes a new area with no branch defined anywhere. Nothing here is authored per tool.
    """
    skill_roots = config.get("skill_roots")
    areas: List[Dict[str, Any]] = []
    for factory_name, factory in _discover_registry_factories():
        if factory_name == "make_langchain_mcp_tools":
            entries = (_tool_entries(factory, include_modules=config.get("mcp_modules"))
                       if config.get("include_mcp_tools") else [])
        else:
            entries = _tool_entries(
                factory,
                **_factory_kwargs(factory, default_input_file_ids=None,
                                  enabled_search_methods=None, skill_roots=skill_roots,
                                  session_id=None),
            )
        if entries:
            areas.append({"area": _area_id(factory_name), "tools": entries})
    return areas


def make_capability_tools(**config: Any) -> List[Any]:
    """Tools that let the agent inspect its OWN tool surface.

    Filterable ON PURPOSE. The previous design dumped the whole inventory into a prompt and cut
    it at 12,000 chars — 56% of a 27,397-char blob for this deployment, always the registries
    added last, and sliced mid-object so the model got malformed JSON. A tool the agent can
    query by topic removes the budget problem instead of tuning it, and lets the agent look
    again with a different word when the first look finds nothing.
    """
    try:
        from langchain_core.tools import StructuredTool
    except Exception:
        return []

    def list_my_capabilities(area: str = "", topic: str = "") -> str:
        """Discover what THIS deployment can do. Start with no arguments, then drill in.

        No arguments returns the capability AREAS with the tool names in each — a small map of
        the whole surface. Pass `area` (an id from that map) for full descriptions of just that
        area. Pass `topic` to search names and descriptions across every area at once; matching
        is LITERAL, so a topic that finds little may just be the wrong word — the area map comes
        back every time so you can navigate instead of guessing.
        """
        areas = _registry_areas(**config)
        index = [{"area": a["area"], "tools": len(a["tools"]),
                  "tool_names": [t["name"] for t in a["tools"]]} for a in areas]
        # DISTINCT, not the sum of the per-area counts: a few tools are registered by more
        # than one registry (select_by_attribute is in both aggregate and overlay), so summing
        # would report 78 where 70 exist. Listing them under each area they live in is right
        # for a map; the total must still be the truth.
        distinct = {t["name"] for a in areas for t in a["tools"]}
        out: Dict[str, Any] = {"areas": index, "total_tools": len(distinct),
                               "listings": sum(a["tools"] for a in index)}

        want = (area or "").strip().lower().replace("-", "_").replace(" ", "_")
        if want:
            picked = [a for a in areas if want in a["area"]] or \
                     [a for a in areas if a["area"] in want]
            out["area_requested"] = area
            out["detail"] = [{"area": a["area"], "tools": a["tools"]} for a in picked]
            if not picked:
                out["hint"] = "no such area; pick one of `areas` above"
            return json.dumps(out, ensure_ascii=True, separators=(",", ":"))

        needle = (topic or "").strip().lower()
        if needle:
            words = [w for w in re.split(r"[^a-z0-9]+", needle) if len(w) > 2]
            matches = []
            for a in areas:
                for t in a["tools"]:
                    blob = f"{t['name']} {t['description']}".lower()
                    if any(w in blob for w in words):
                        matches.append({"area": a["area"], **t})
            out["topic"] = topic
            out["matched"] = len(matches)
            out["detail"] = matches[:40]
            out["hint"] = ("matching is literal, so a small or empty result may be the wrong "
                           "word rather than a missing capability — read `areas` and open the "
                           "one that looks right before concluding anything is unsupported.")
            return json.dumps(out, ensure_ascii=True, separators=(",", ":"))

        out["hint"] = ("this is the map, not the detail: call again with `area` for full "
                       "descriptions of the part you need.")
        out["code_execution"] = _safe_inventory_bits(**config)
        return json.dumps(out, ensure_ascii=True, separators=(",", ":"))

    return [StructuredTool.from_function(list_my_capabilities)]


# The prompt's inventory budget. The old code did json.dumps(..., indent=1)[:12000] on a blob
# that is 27,397 chars for this deployment — so 56% of the tool surface was cut, ALWAYS the
# registries appended last, and the model received JSON truncated mid-object. That is why the
# capability answer never mentioned satellite embeddings even after those registries were added:
# they were at the end of the list and never arrived.
_INVENTORY_PROMPT_CHARS = 24000
_INVENTORY_DESC_CHARS = 220


def _inventory_for_prompt(inventory: Any, *, max_chars: int = _INVENTORY_PROMPT_CHARS) -> str:
    """The inventory as compact JSON that FITS, dropping whole tools rather than mid-object.

    Descriptions are trimmed first (they average ~340 chars and lead with the punchy summary),
    then whole entries are dropped from the end with a count the model can see — so a squeeze is
    visible to the reader and to the log instead of producing malformed JSON.
    """
    payload = dict(inventory) if isinstance(inventory, dict) else {"tools": list(inventory or [])}
    tools = list(payload.get("tools") or [])
    trimmed = []
    for t in tools:
        entry = dict(t) if isinstance(t, dict) else {"name": str(t)}
        desc = str(entry.get("description") or "")
        if len(desc) > _INVENTORY_DESC_CHARS:
            entry["description"] = desc[:_INVENTORY_DESC_CHARS].rstrip() + "…"
        trimmed.append(entry)

    dropped = 0
    while True:
        payload["tools"] = trimmed
        if dropped:
            payload["tools_omitted_for_length"] = dropped
        text = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        if len(text) <= max_chars or len(trimmed) <= 1:
            if dropped:
                logger.warning(
                    "capability inventory too large for the prompt: %d of %d tools omitted",
                    dropped, len(tools))
            return text
        trimmed = trimmed[:-1]
        dropped += 1


def describe_capabilities(*, llm: Optional[Any] = None, query: Optional[str] = None,
                          **config: Any) -> str:
    """Answer a capability question by REASONING over live self-introspection.

    A ReAct loop rather than a one-shot composer, because a fixed prompt can only answer the
    question it was written for. The old version took the inventory and no query at all, was
    told to "cover everything", and returned the same grouped catalogue however specific the
    question — "what can you do with satellite imagery?" got five generic headings and never
    mentioned embeddings, though the inventory carried eight tools for exactly that.

    With a loop the agent can look up its own surface by topic, look again with a different
    word, call the real listing tools to name actual models, and answer a question nobody
    anticipated. Falls back to the mechanical listing when no model is reachable.
    """
    tools = make_capability_tools(**config)
    # Every read-only LISTER the deployment has, found by the same discovery as the inventory —
    # not two hardcoded names. A `list_*` tool enumerates options and takes no destructive
    # action, so it is safe and cheap to hand the agent, and it is what lets "which models can
    # you use?" be answered with the REAL names instead of a guess. Add a new list_* tool
    # anywhere and this picks it up. Guarded per registry: if rs-embed is unreachable its tool
    # returns an error and the agent can say so rather than inventing a list.
    for name, factory in _discover_registry_factories():
        if name == "make_langchain_mcp_tools":
            continue                       # remote, and not a cheap local enumeration
        try:
            built = factory(**_factory_kwargs(factory, default_input_file_ids=None,
                                              skill_roots=None, session_id=None))
        except Exception:
            continue
        tools += [t for t in (built or [])
                  if str(getattr(t, "name", "")).startswith("list_")
                  and str(getattr(t, "name", "")) != "list_my_capabilities"]

    try:
        from agent_runtime.executor_factory import (
            build_agent_executor,
            build_default_llm,
            invoke_agent_with_payload_fallback,
        )
        from agent_runtime.prompts import CAPABILITY_AGENT_PROMPT
        from agent_runtime.runtime_utils import extract_final_answer

        active = llm or build_default_llm()
        executor = build_agent_executor(
            llm=active, preloaded_tools=tools,
            system_prompt_override=CAPABILITY_AGENT_PROMPT,
        )
        resp = invoke_agent_with_payload_fallback(
            executor, query=(query or "What can you do?").strip(), chat_history=None)
        answer = extract_final_answer(resp) or ""
        if answer.strip():
            return answer.strip()
    except Exception:
        logger.warning("capability agent unavailable; falling back to a mechanical listing",
                       exc_info=True)
    return _mechanical_summary(collect_capability_inventory(**config))


__all__ = ["is_capability_query", "collect_capability_inventory",
           "make_capability_tools", "describe_capabilities"]
