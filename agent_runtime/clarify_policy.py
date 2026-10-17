"""When the agent asks the user, and when it states its assumption instead (experiment).

`AGENT_CLARIFY_POLICY` picks one of three behaviours. Unset or `never` is the agent as it was:
it never asks, and the writer is not told to name its assumptions.

* ``assume`` — the final-answer writer gets ASSUME_RULE: when the request leaves out something
  that changes the answer, name the assumption made and give the answer under the main
  alternatives when they differ.
* ``ask`` — ASSUME_RULE as well, plus a clarification GATE before any work: one model call that
  reads the request, the conversation and the attached files, and either lets the turn proceed
  or returns ONE question for the user. The gate asks only when all of these hold: the
  ambiguity is real, the choices give materially different answers, nothing in the request or
  the conversation settles it, and it has not already asked in this conversation.

Why: MultiGlobeQA turns (2026-10-14) failed on things only the user knew. "Which grid cell
contains X both in h3 and s2 formats?" was answered at H3 resolution 9 and S2 level 30, both
right, while the benchmark wanted resolution 5 and level 6; "Nes" resolved to the wrong Nes;
U01 substituted data for a DEM that was never attached. Whether asking helps, and what it
costs in turns and needless questions, is measured by `gis_harness.clarify`.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

POLICY_ENV = "AGENT_CLARIFY_POLICY"
POLICIES = ("never", "assume", "ask")

ASSUME_RULE = (
    "ASSUMPTIONS: if the request leaves out something that changes the answer (a resolution or "
    "level, a date or period, which of several places share the name, which dataset or method), "
    "pick the most reasonable reading, SAY which assumption you made in one sentence, and when "
    "the main alternatives give different answers, state those answers too, briefly. Never call "
    "an arbitrary choice 'standard'. If the request and the conversation already settle it, "
    "say nothing about it."
)

GATE_PROMPT = (
    "You decide whether a geospatial assistant must ask the user ONE question before doing the "
    "work. Ask ONLY if ALL of these hold:\n"
    "1. Something the answer depends on is genuinely open: a parameter with several reasonable "
    "values (a grid resolution or level, a time period, a distance), a place name that matches "
    "several real places, or an attached file that is not what the request needs.\n"
    "2. The reasonable choices would give materially DIFFERENT answers.\n"
    "3. Nothing settles it: not the request's own wording (e.g. two places that must share one "
    "grid cell must be near each other), not the conversation so far, not the attached files.\n"
    "4. The assistant has not already asked about it in this conversation.\n"
    "Do NOT ask about things with a conventional default (great-circle distance for 'how far', "
    "the place's centroid for 'its location') or that the assistant can find out itself (a "
    "file's columns, a place's coordinates). When in doubt, do not ask.\n\n"
    "Conversation so far:\n{history}\n\n"
    "Attached files: {files}\n\n"
    "Request:\n{query}\n\n"
    "Reply with JSON only: {{\"ask\": true|false, \"missing\": \"<what is open, or empty>\", "
    "\"question\": \"<one short question to the user, offering the main options; empty if "
    "ask is false>\"}}"
)


def policy() -> str:
    raw = (os.getenv(POLICY_ENV) or "never").strip().lower()
    return raw if raw in POLICIES else "never"


def writer_rule() -> Optional[str]:
    """What the final-answer writer is told under the active policy (None: nothing)."""
    return ASSUME_RULE if policy() in ("assume", "ask") else None


def _history_text(chat_history: Optional[List[Any]]) -> str:
    lines: List[str] = []
    for item in (chat_history or [])[-6:]:
        if isinstance(item, dict):
            q, a = item.get("userQuery") or item.get("user"), item.get("answer")
            if q:
                lines.append(f"User: {str(q)[:600]}")
            if a:
                lines.append(f"Assistant: {str(a)[:600]}")
        else:
            lines.append(str(item)[:600])
    return "\n".join(lines) or "(none)"


def _file_names(thread_id: Optional[str]) -> List[str]:
    try:
        from agent_runtime.file_store import get_file_record
        from agent_runtime.session_memory import get_session_files

        names = []
        for fid in get_session_files(thread_id)[:20]:
            rec = get_file_record(fid) or {}
            names.append(str(rec.get("filename") or fid))
        return names
    except Exception:  # noqa: BLE001 - the gate works without file names
        return []


def already_asked(thread_id: Optional[str]) -> bool:
    try:
        from agent_runtime.session_memory import get_session_actions

        return any(r.get("tool") == "clarify" for r in get_session_actions(thread_id))
    except Exception:  # noqa: BLE001
        return False


def _record_ask(thread_id: Optional[str], decision: Dict[str, Any], query: str) -> None:
    try:
        from agent_runtime.session_memory import append_session_actions

        append_session_actions(thread_id, [{"tool": "clarify", "missing": decision.get("missing"),
                                            "question": decision.get("question"),
                                            "original_query": str(query)[:4000]}])
    except Exception:  # noqa: BLE001
        logger.debug("clarify: could not record the question", exc_info=True)


def resolved_query(query: str, thread_id: Optional[str]) -> Optional[str]:
    """The original request with the user's answer, on the turn right after the gate asked.

    The user's reply to a clarifying question is a fragment ("Use H3 resolution 3"). Only the
    supervisor sees the conversation; the peers work from the query. Live, 2026-10-15
    (gis_harness.clarify, MG05): given the fragment alone, the peers reported that the place
    names "are not available in my current context" and spent 8 steps before the user repeated
    them. Returns None when the previous turn was not the gate's question.
    """
    try:
        from agent_runtime.session_memory import get_session_actions, get_session_history

        rows = [r for r in get_session_actions(thread_id) if r.get("tool") == "clarify"]
        history = get_session_history(thread_id)
    except Exception:  # noqa: BLE001
        return None
    if not rows or not history:
        return None
    last = rows[-1]
    if str(history[-1].get("answer") or "").strip() != str(last.get("question") or "").strip():
        return None                      # the gate's question was not the previous turn
    original = last.get("original_query") or history[-1].get("userQuery") or ""
    return (f"{original}\n\n(You asked the user: \"{last.get('question')}\" "
            f"They answered: \"{query}\")")


def gate(query: str, chat_history: Optional[List[Any]], thread_id: Optional[str],
         llm: Any) -> Dict[str, Any]:
    """{"ask": bool, "missing": str, "question": str}; never raises (a failure proceeds)."""
    if already_asked(thread_id):
        return {"ask": False, "missing": "", "question": "", "skipped": "asked once already"}
    files = _file_names(thread_id)
    prompt = GATE_PROMPT.format(history=_history_text(chat_history),
                                files=", ".join(files) or "(none)", query=str(query)[:4000])
    try:
        raw = llm.invoke(prompt)
        text = getattr(raw, "content", raw)
        if isinstance(text, list):
            text = " ".join(str(p.get("text", p)) if isinstance(p, dict) else str(p) for p in text)
        from agent_runtime.evidence_quality import _extract_json_object

        data = _extract_json_object(str(text)) or {}
    except Exception as exc:  # noqa: BLE001 - a broken gate must never block the answer
        logger.warning("clarify gate failed (%s); proceeding without asking", exc)
        return {"ask": False, "missing": "", "question": "", "error": str(exc)[:200]}
    question = str(data.get("question") or "").strip()
    decision = {"ask": bool(data.get("ask")) and bool(question),
                "missing": str(data.get("missing") or "").strip(), "question": question}
    if decision["ask"]:
        _record_ask(thread_id, decision, query)
    return decision


__all__ = ["ASSUME_RULE", "POLICIES", "POLICY_ENV", "already_asked", "gate", "policy",
           "resolved_query", "writer_rule"]
