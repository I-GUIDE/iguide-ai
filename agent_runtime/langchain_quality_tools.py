"""Granular evidence-quality tools: LLM rerank + grounding/hallucination audit.

These expose ``agent_runtime.evidence_quality`` as LangChain ``StructuredTool``s
so the search/analysis agents can re-rank retrieved documents and audit whether a
composed answer is grounded — the capabilities that previously lived only in the
deprecated ``rag_pipeline``. They use the agent's own LLM (``build_default_llm``).
"""

from __future__ import annotations

import json
from typing import Any, List, Optional
from agent_runtime.tool_args import accept_null_defaults


def make_quality_tools(llm: Optional[Any] = None) -> List[Any]:
    """Build the rerank + grounding-audit StructuredTools."""
    from langchain_core.tools import StructuredTool

    from agent_runtime.evidence_quality import audit_answer_grounding, rerank_documents

    def rerank_evidence(query: str, documents_json: str, top_k: int = 5) -> str:
        try:
            docs = json.loads(documents_json) if documents_json else []
        except Exception:
            return json.dumps({"error": "invalid_documents_json"}, ensure_ascii=True)
        if not isinstance(docs, list):
            return json.dumps({"error": "documents_json must be a JSON array"}, ensure_ascii=True)
        reordered = rerank_documents(query, docs, top_k=top_k, llm=llm)
        return json.dumps(
            {"reranked": reordered, "count": len(reordered)},
            ensure_ascii=True,
            default=str,
        )

    def audit_answer_grounding_tool(question: str, answer: str, evidence_json: str = "") -> str:
        evidence: Any = evidence_json
        if evidence_json:
            try:
                evidence = json.loads(evidence_json)
            except Exception:
                evidence = evidence_json  # fall back to raw text evidence
        verdict = audit_answer_grounding(question, answer, evidence, llm=llm)
        return json.dumps(verdict, ensure_ascii=True, default=str)

    rerank_tool = StructuredTool.from_function(func=accept_null_defaults(rerank_evidence),
        name="rerank_evidence",
        description=(
            "Re-rank retrieved documents by LLM-judged relevance to the query. "
            "Input: query and documents_json (a JSON array of {doc_id, title, contents}); "
            "returns the documents reordered most-relevant-first (top_k)."
        ),
    )
    audit_tool = StructuredTool.from_function(func=accept_null_defaults(audit_answer_grounding_tool),
        name="audit_answer_grounding",
        description=(
            "Audit whether a composed answer is grounded in the evidence (hallucination "
            "check). Input: question, answer, and evidence_json (JSON array of documents or "
            "text). Returns JSON: hallucination_detected, severity, issues, summary."
        ),
    )
    return [rerank_tool, audit_tool]


__all__ = ["make_quality_tools"]
