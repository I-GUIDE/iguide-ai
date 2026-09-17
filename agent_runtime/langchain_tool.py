from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional

from rag_pipeline.pipeline import run_pipeline
from agent_runtime.tool_args import accept_null_defaults


def rag_tool(
    query: str,
    memory_id: Optional[str] = None,
    session_context: Optional[Mapping[str, Any]] = None,
    params: Optional[Mapping[str, Any]] = None,
    recent_k: Optional[int] = None,
    extra_state: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    LangChain-friendly wrapper around the existing RAG pipeline orchestration.

    Returns a compact dict aligned with the `/query` API payload shape.
    """
    result = run_pipeline(
        user_input=query,
        memory_id=memory_id,
        session_context=session_context,
        params=params,
        recent_k=recent_k,
        extra_state=extra_state,
    )

    evidence = (result.get("evidence") or {}).get("retrieved_documents") or []
    answer = (result.get("answer") or {}).get("final_composed_answer") or ""
    citations = (result.get("answer") or {}).get("citations") or []
    confidence = (result.get("answer") or {}).get("confidence_score")
    retrieval_steps = (result.get("trace_observability") or {}).get("retrieval_routing_decisions") or []

    return {
        "answer": answer,
        "citations": citations,
        "confidence_score": confidence,
        "retrieval_steps": retrieval_steps,
        "evidence_count": len(evidence),
        "state": result,
    }


def rag_tool_json(
    query: str,
    memory_id: Optional[str] = None,
    session_context: Optional[Mapping[str, Any]] = None,
    params: Optional[Mapping[str, Any]] = None,
    recent_k: Optional[int] = None,
    extra_state: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    String-returning adapter for agents that expect plain text tool outputs.
    """
    payload = rag_tool(
        query=query,
        memory_id=memory_id,
        session_context=session_context,
        params=params,
        recent_k=recent_k,
        extra_state=extra_state,
    )
    return json.dumps(payload, ensure_ascii=True, default=str)


def make_langchain_rag_tool() -> Any:
    """
    Build a LangChain StructuredTool from `rag_tool_json`.

    Raises:
        RuntimeError: if LangChain is not installed.
    """
    try:
        from langchain_core.tools import StructuredTool
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "LangChain is not installed. Add `langchain-core` (or langchain) to dependencies."
        ) from exc

    return StructuredTool.from_function(func=accept_null_defaults(rag_tool_json),
        name="rag_tool",
        description=(
            "Run the full RAG pipeline for a user query. "
            "Inputs: query, optional memory_id, session_context, params, recent_k, extra_state. "
            "Returns JSON string with answer, citations, retrieval steps, and final state."
        ),
        metadata={"category": "retrieval_internal"},
    )


__all__ = ["rag_tool", "rag_tool_json", "make_langchain_rag_tool"]
