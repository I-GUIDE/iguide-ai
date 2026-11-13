# Agent Orchestration Optimization Plan

Status: **planning only — no code changes yet.**
Branch: `sigspatial26shortpaper_inspection`.

## Goal & guiding constraints

1. **Granular is the only path.** `full_pipeline` / `rag_tool` is deprecated. No new
   work builds on `rag_tool`; we do not call `rag_pipeline.run_retrieval` from the
   agent. Where that path has useful logic, we reconstruct it natively on the
   granular backend functions.
2. **Agent = superset of the RAG pipeline.** Anything the old pipeline did
   (retrieve → rerank → generate → hallucination-audit, with LLM-gated
   neo4j/spatial) must be reachable through the granular agent before
   `full_pipeline` is deleted.
3. **Cheaper orchestration.** Remove redundant LLM round-trips, parallelize
   independent I/O, and stop re-searching the same query across the agent
   hierarchy — without reducing answer quality.

All behavior-changing work lands behind env flags so we can A/B against current
behavior.

---

## Current-state facts (verified in code)

- Granular search tool output shape (`_build_payload` in
  `langchain_granular_tools.py`):
  `{"source", "count", "documents":[{"doc_id","source","score","title","element_type","contents"}], "citation_ids":[...]}`.
- `SearchAgent` is an LLM `AgentExecutor` loop calling the 5 backend tools one at a
  time; each backend call is its own LLM turn.
- `make_search_agent_evidence_tool` (`graph_nodes.py:147`) calls `_build_route_trace`
  → `build_route_trace` → **`llm_route_decision`** on every invocation, with
  `available_routes = [{"route":"search"}]` — i.e. **an LLM call that chooses among
  exactly one route.** Pure waste; only the `allowed_tools` filter is useful, and
  that comes from keyword-only `classify_intent` + `select_allowed_tools`.
- Each search-tool factory gets its **own empty** `search_invocations` list
  (orchestrator: `graph_nodes.py:544`; analysis: `graph_nodes.py:308`) → no sharing,
  no dedup across the hierarchy.
- `build_agent_executor` hardcodes `max_iterations=15`, `max_execution_time=120` for
  **every** agent role (`executor_factory.py:477`).
- Rerank (`reranker_llm.rerank_evidence_with_llm`), hallucination
  (`hallucination_check.evaluate_hallucination`), and `run_retrieval`'s
  routing/merge live **only** in the deprecated path. They operate on the
  `AgentState` shape: `state["query_information"]["raw_text"]` for the query and
  `state["evidence"]["retrieved_documents"][].document` for docs; both call
  `rag_pipeline.llm_utils.call_llm` (not the LangChain LLM).

---

## Workstream A — Close the superset gap (prerequisite for deletion)

Expose the pipeline-only capabilities as **native granular tools**. New module:
`agent_runtime/langchain_quality_tools.py`.

### A1. `rerank_evidence` tool
- **Signature:** `rerank_evidence(query: str, evidence_json: str, top_k: int = 8) -> str`
- **Adapter:** parse `evidence_json` (the granular `documents` list) → build a minimal
  state: `{"query_information": {"raw_text": query}, "evidence": {"retrieved_documents": [{"document": doc} for doc in documents]}}`
  → `rerank_evidence_with_llm(state, top_k=top_k)` → re-emit the reordered
  `documents` (carry `metadata.llm_rerank_score`/`reason`) in the same payload shape.
- **Why it works:** `_summarize_document` reads `entry["document"]["doc_id"/"title"/"contents"]`,
  which the normalized granular doc already provides.
- **Risk:** low. Pure reordering; isolated module; one `call_llm` round-trip.

### A2. `audit_answer` tool (hallucination check)
- **Signature:** `audit_answer(query: str, answer: str, evidence_json: str) -> str`
- **Adapter:** wrap into state with `answer={"final_composed_answer": answer}` plus the
  same `retrieved_documents` wrapping, call `evaluate_hallucination(state)`, return its
  JSON verdict (`hallucination_detected`, `severity`, `issues`, `summary`).
- **Risk:** low.

### A3. Wire into prompts / tool sets
- Register A1/A2 in `tool_policy.collect_tools` and add their names to a new
  `QUALITY_TOOL_NAMES` set in `graph_state.py`; include them in
  `select_allowed_tools` for all intents.
- **Exposure choice (OPEN DECISION 1):**
  - *Explicit tools* — add a SearchAgent prompt rule "after gathering, call
    `rerank_evidence` before returning" and an AnalysisAgent rule "call
    `audit_answer` on your draft before finalizing." Maximal granularity/control.
  - *Auto post-steps* — call rerank inside `broad_search` (B1) automatically and run
    `audit_answer` automatically at the end of `analysis_agent_answer`
    (`graph_nodes.py`). Fewer turns, closer to old pipeline, less agent control.
  - **Recommendation:** explicit tools, plus auto-rerank inside `broad_search` only.

---

## Workstream B — Search speed (granular-native)

### B1. `broad_search` composite tool
- **Where:** `langchain_granular_tools.py`, new `broad_search_tool(query, limit=8)`.
- **What:** fan out `keyword` + `semantic` + `opengeodata` **concurrently**
  (`concurrent.futures.ThreadPoolExecutor`) over the existing backend functions;
  add `neo4j` when `neo4j_graph_tools.detect_pattern(query)` matches and `spatial`
  when a location is detected (reuse the spatial detector). Merge + dedup by
  `doc_id`, return the standard payload with `source="broad"`. Optionally auto-call
  A1 rerank (per OPEN DECISION 1).
- **Why:** collapses 3–5 serial LLM turns into one and parallelizes independent
  network waits. This is `run_retrieval`'s value, reconstructed natively.
- **Keep individual tools exposed** (OPEN DECISION 2): default = keep them so the
  agent retains autonomy and multi-hop graph exploration; `broad_search` is an
  *additional* breadth option, not a replacement.
- **Risk:** medium — concurrency + new merge logic. Mitigate with unit tests on
  merge/dedup and a flag `AGENT_BROAD_SEARCH_ENABLED`.

### B2. Drop the wasted in-tool routing LLM call
- **Where:** `make_search_agent_evidence_tool` (`graph_nodes.py:193-216`).
- **What:** replace the `_build_route_trace`/`llm_route_decision` call with a
  heuristic-only allowed-tools computation:
  `select_allowed_tools(classify_intent(query)["intent"], available_names)`.
  Keep the emitted `decision` trace event (mark `router_type="heuristic"`).
- **Why:** the route set has exactly one route (`search`) — the LLM call decides
  nothing. Removes one LLM round-trip per search invocation, at every depth.
- **Risk:** low. Gate with `AGENT_INTOOL_LLM_ROUTING=0` (default off after this).

### B3. Cross-hierarchy search memoization
- **What:** introduce a request-scoped cache `Dict[str, str]` keyed by normalized
  query (lowercased/stripped) → evidence JSON. Create it once in
  `run_agent_query` / `stream_agent_query_events` / `run_code_agent_query` and thread
  it into `collect_orchestration_tools` and `make_analysis_agent_answer_tool`, which
  pass the **same** dict into every `make_search_agent_evidence_tool`.
- **Inside `search_agent_evidence`:** on entry, return the cached evidence on a hit
  (emit a `subagent_completed` with `status:"cache_hit"`, skip the nested executor);
  on miss, store after building the payload.
- **Why:** prevents Orchestrator + Analysis (+ Code) re-running an identical search.
- **Risk:** low. Key only on query within a request (strategy/intent are fixed per
  request). Flag `AGENT_SEARCH_CACHE=1`.

---

## Workstream C — Executor hygiene

### C1. Per-role iteration / time budgets
- **Where:** `build_agent_executor` (`executor_factory.py`).
- **What:** add `max_iterations` / `max_execution_time` params (defaults preserve
  current 15/120). Pass lower budgets for leaf roles via the role builders
  (`build_search_agent_executor` ≈ 5 iters; analysis/code keep higher). For the
  modern `create_agent` path (no `max_iterations`), set a lower per-invocation
  `recursion_limit` in `agent_config` for leaf calls.
- **Why:** a confused leaf fails fast instead of consuming the 60-step recursion
  budget; bounds worst-case nested latency.
- **Risk:** low–medium (too-low caps could truncate legitimate multi-hop). Make
  values env-tunable.

### C2. Reuse one LLM client per request
- **Where:** `graph_runtime.*` entry points.
- **What:** build the default LLM once (`build_default_llm()`), pass `llm=` down
  through `collect_orchestration_tools` → all builders (plumbing already accepts
  `llm`). Avoids constructing a `ChatOpenAI` per nested invocation.
- **Risk:** low.

### C3. Gate `answer_from_memory`
- **Where:** orchestrator prompt (`executor_factory.py:83`) + `collect_orchestration_tools`.
- **What:** only attach `answer_from_memory` (or only instruct "try first") when the
  memory-phrase heuristic in `heuristic_route_decision` (`intent_classifier.py:251`)
  matches; otherwise skip the extra LLM call for clearly-external queries.
- **Risk:** low.

---

## Workstream D — Remove `full_pipeline` (only after A is shipped & validated)

- Delete the `full_pipeline` branch in `tool_policy.collect_tools`,
  `langchain_tool.make_langchain_rag_tool` / `rag_tool` / `rag_tool_json`, the
  `"rag_tool"` entry in `graph_state.DISCOVERY_TOOL_NAMES`.
- Remove `--tool-strategy` choices / collapse the param in `graph_runtime.main` and
  `langchain_agent_executor`.
- Update README (Project Structure + the `--tool-strategy full_pipeline` CLI
  examples I added earlier are now misleading) and `docs/router-llm.md`.
- Keep `rag_pipeline/search/*`, `reranker*`, `hallucination_check`, `memory_module`
  — those remain the implementations the granular tools wrap.

---

## Suggested sequencing

1. **C2 + B2 + B3** — pure speed/cost, no output change, lowest risk. Ship behind
   flags, measure LLM-call count + latency on a fixed query set.
2. **A1 + A2 (+ A3 explicit)** — make the agent a real superset.
3. **B1 `broad_search`** — structural speed win; depends on A1 if auto-rerank.
4. **C1, C3** — tuning.
5. **D** — delete `full_pipeline` once A is validated.

## Testing

- Unit: rerank/audit adapters (shape round-trip), `broad_search` merge/dedup,
  cache hit/miss, heuristic allowed-tools selection. Extend
  `rag_pipeline/tests/` + `agent_runtime` tests; reuse `test_mcp_cache.py` pattern.
- Integration: `scripts/run_local_rag_test.py` style E2E comparing
  flags-on vs flags-off — assert (a) identical/again-grounded answers, (b) fewer
  LLM calls, (c) lower wall-clock. Use the existing trace events
  (`decision`, `tool_call`) to count round-trips.

## Open decisions

1. **Rerank/audit exposure** — explicit tools vs auto post-steps (recommend:
   explicit + auto-rerank inside `broad_search`).
2. **`broad_search` vs individual tools** — additive (recommend) vs replace.
3. **Default flag states** for B2/B3/C1 once validated (recommend: B2 heuristic ON,
   B3 cache ON, C1 leaf cap ON with env override).
