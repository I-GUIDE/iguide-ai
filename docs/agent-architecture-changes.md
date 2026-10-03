# Agent architecture: what changed, and why

**Covers** the project's whole commit history — 408 commits on `prototype`, 2025-04-08 through
2026-09-10 — plus two unmerged branches cut from its tip. 222 of those commits touch
`agent_runtime/`.

**This document is maintained incrementally.** See [Adding to this document](#adding-to-this-document)
at the end. It was reconstructed once, from five parallel passes over the history, and that
reconstruction is exactly what the rule exists to prevent happening again: where a reason was
never written down, it is gone, and reading the diff does not bring it back.

## The stages

| | stage | span | the shift |
|---|---|---|---|
| 0 | [Two Flask servers](#stage-0) | 2025-04 → 2025-09 | no LLM anywhere; embeddings and metadata extraction |
| 1 | [A RAG pipeline with a typed state](#stage-1) | 2025-10 → 2026-01 | one `AgentState`, one `EvidenceEntry`, one merge |
| 2 | [The agent arrives](#stage-2) | 2026-03 | an LLM that calls tools, with intent-filtered tool policy |
| 3 | [Agents as tools](#stage-3) | 2026-04 → 2026-06 | the orchestrator LLM calls the other agents as tools |
| 4 | [The monolith becomes a package](#stage-4) | 2026-04 → 2026-05 | 2048 lines split into `agent_runtime/`, twice |
| 5 | [Supervisor over peers](#stage-5) | 2026-06 → 2026-07 | nesting replaced by peers sharing typed state |
| 6 | [Map-native delivery](#stage-6) | 2026-08 | one boundary every layer crosses; prompts stop issuing mandates |
| 7 | [The action ledger](#stage-7) | 2026-09 → `9e35950` | the agent records what tools *did*, not that they ran |
| 8 | [What the decider reads](#stage-8) | `claude/evidence-summary` | evidence described, capabilities generated, the ledger shared |
| 9 | [Who the caller is](#stage-9) | `claude/jwt-identity` | identity, ownership, server-owned history |
| 10 | [Removing the second path](#stage-10) | `claude/evidence-summary` | the agents-as-tools arm and `full_pipeline` deleted |
| 11 | [Where state lives, and who decides](#stage-11) | 2026-09-18 → 2026-09-22 | tiers own the cluster; a silent write failure found |
| 12 | [Staying up, and keeping the evidence](#stage-12) | 2026-09-22 | a watchdog acts on failing health; logs outlive the container |
| 13 | [Shapes nobody owned](#stage-13) | 2026-10-01 → 2026-10-03 | downloads named, qwen3.8, rs-embed off a personal credential, a restart that cut off live turns |
| 14 | [A promise kept by rounding luck](#stage-14) | 2026-10-01 | the default distance band is island-free by construction, not by platform |
| 15 | [What counts as a date is decided here, not by pandas](#stage-15) | 2026-10-01 | the temporal parser states its own rules; pandas 3 had moved them |
| 16 | [Six tests only the Mac passed](#stage-16) | 2026-10-02 | production's spaCy path gets the fallback's filters; a QGIS test stops assuming no QGIS |
| 17 | [The image installs a list, not a laptop](#stage-17) | 2026-10-01 | packages dev had and the image lacked, declared and tested |
| 18 | [Testing what is deployed](#stage-18) | 2026-10-01 | a lock taken from the image; the suite runs on the deployed platform |
| 19 | [What runs in the agent's own process](#stage-19) | 2026-10-01 → 2026-10-02 | the tool that `exec()`'d knowledge-base code in-process is withdrawn; a test guards the class |

Stages 8, 9 and 10 began as independent branches and **merged into `prototype`** at `e0e1f92`
(identity) and `b511460` (the decider and tool-surface work), with `c180490` closing the upload
gap afterwards. Deployment state is at the end; do not infer it from the commits — at the time of
writing the deployment runs `AGENT_MODE=dev`, so stage 9 is shipped but not switched on.

A stage is a coherent shift, not a time slice, so two can overlap: stage 4 (the extraction into
`agent_runtime/`) happened *while* stage 3 was the operating model, and changed nothing about how
the agent decided. Stage 3 is called out separately from the packaging around it because it is
the model the system actually ran on for two months, and the one stage 10 deletes — a reader
asking "what was agents-as-tools?" should land on it, not on a subsection about file layout.

One thread runs through all of them: **the system repeatedly discovers that a component was
deciding from what was cheap to compute rather than what the decision needed**, and the fix is
almost always to give it the record that already existed somewhere else.

---

## Stage 0 — Two Flask servers {#stage-0}

*2025-04-08 (`a5e0a88`) → 2025-09. ~5 commits.*

The repo is two unrelated Flask processes. `embedding-server/dense_embedding_server.py` serves
`POST /get_embedding` from `all-MiniLM-L6-v2`. `metadata-extraction-server/minio_webhook.py`
takes an S3 event and shells out to an extractor that uses `ast` for Python, regex for Java,
`nbformat` for notebooks, and rasterio/fiona for raster and vector bounds, writing results back
as MinIO object tags. Stores: OpenSearch and MinIO.

**There is no LLM in the project until 2025-10-06.** No prompts, no agent, no state object. This
is worth stating because it dates everything else: the first prompt in the history is six months
in, and the supervisor architecture is fourteen months in.

---

## Stage 1 — A RAG pipeline with a typed state {#stage-1}

*2025-10 → 2026-01. ~34 commits.*

### 1.1 The first LLM, and the first prompt (`e885183`)

`generation.py` and `llm_utils.py` arrive together. The state shape —
`query_information` / `session_context` / `evidence` / `planner_reasoning` / `answer` — appears
here only as a hand-written literal in `main()`, not as a type. **Reason not recorded.**

The first system prompt establishes a rule that survives to today: *"Your ONLY source of truth is
the `<doc>` blocks provided"*, with *"If you cannot find an answer, reply exactly: 'I don't have
enough information.'"*

### 1.2 The typed state contract (`a4edd87`) — the foundation everything else sits on

`state.py` introduces `AgentState`, `EvidenceEntry`, `ensure_state_shapes` and `merge_retrieval`.
Before it, five search modules each returned their own dict shape and were called ad hoc. After
it, every retriever is `f(state) -> hits` merging through one deduplicating function.

That contract is why generation, LLM reranking, hallucination auditing and an entire external
catalogue could be added later without touching the retrievers. It is the direct ancestor of
today's `SupervisorState`.

A detail that causes trouble later: `merge_retrieval`'s limit is **cumulative across sources**
(`space = max(limit - len(current_docs), 0)`), not per-source.

### 1.3 The declarative router, deleted five days later (`a4edd87` → `0e0435d`)

`a4edd87` also added a `SearchStrategy(name, predicate, runner, …)` table. `0e0435d` deleted it,
cutting `routing.py` from 206 lines to 20 and replacing it with hardcoded `if` gates. It also
orphaned `search_agents.py` — after this commit nothing imports it, and nothing imports it at the
end of the era either.

**Reason not recorded**, for any part of the reversal.

### 1.4 Two routers written and never wired

`search_agents.py` (orphaned above) and `router_llm.py` (`f997384`, 364 lines) both implement
LLM-driven routing. `router_llm.py`'s own README says the position plainly: *"routing.py #
Original router (unchanged)"*. `docker-compose.yml` exports `USE_LLM_ROUTER`, and **no Python
file in the era reads it**. The flag is dead on arrival.

### 1.5 Prompt revisions in this stage

| commit | change | reason |
|---|---|---|
| `0e0435d` | the generation prompt is cut to *"You are a factual assistant. Use ONLY the provided evidence. Cite by [doc_id]"*; the hyperlink block is deleted; context format goes from `<doc>` blocks to flat `[doc_id] title:` lines | **not recorded** |
| `971fd9d` | the prompt is restored and expanded — `<doc>` blocks return, snippet 800 → **2000** chars, 10-document cap, the `[doc_id]` auto-append hack deleted | *"Replicate the javascript prompts and structure"* — parity with the platform's existing JS implementation |
| `4b9eae8` | first rerank and hallucination-audit prompts, including *"Scores MUST show meaningful variance"* and a JSON verdict schema | reason not recorded; the schema shape becomes a measured defect in stage 6 |

### 1.6 Worth knowing

`api_server.py` was **unimportable for eight days** (2026-01-14 → 01-22). A commit titled
*"Revert to see the references"* left a literal `\n+` on line 20 — a module-level `SyntaxError` —
silently repaired later by an unrelated commit.

`f0cd457` raised `top_k` from 8 to 100 with no recorded reason. Given `merge_retrieval`'s
cumulative cap, a budget of 8 consumed by keyword hits would leave opengeodata zero slots — that
is a reading of the code, not a claim the commit makes.

---

## Stage 2 — The agent arrives {#stage-2}

*2026-03. The repo gets an agent for the first time: an LLM that calls tools, rather than a
pipeline that runs stages.*

### 2.1 One executor, one intent (`2cfcdf8`, `37b7f6a`)

`rag_pipeline/langchain_agent_executor.py` (266 lines) introduces `build_agent_executor` and
`run_agent_query`, with a single inline `system_prompt`. `37b7f6a` grows it to 893 and splits the
one prompt into three roles — `SEARCH_AGENT_PROMPT`, `ANALYSIS_AGENT_PROMPT`, `CODE_AGENT_PROMPT`
— alongside `_classify_intent` (keyword hints) and `_select_allowed_tools`, which narrows the
bound tool list per intent.

That pairing is the stage's shape: **one executor, with the tool list filtered by a
keyword-classified intent.** Tool *policy* is decided before the model runs, not by it.

The origin prompt's core survives everything that follows: *"Use only tool outputs as evidence;
don't hallucinate citations"* and *"Cite only doc_ids that appear in the tool response."*

### 2.2 The first graph, then its removal (`deefd45`, `8c502d6`, `0bf5449`)

`deefd45` adds a LangGraph `StateGraph` — `initialize → route → search → extract → analysis →
finalize` — with `InMemorySaver` keyed on `thread_id`. `8c502d6` flattens it so search and
analysis become *siblings* off a conditional edge rather than a chain, and gives each a child
thread (`{thread}::search`, `{thread}::analysis`) so the two stop sharing checkpoint state.
`0bf5449` adds a `direct_answer` branch and its prompt: *"answer from the supplied conversation
history only… Do not call tools."*

This graph lasts one day. Stage 3 removes it.

---

## Stage 3 — Agents as tools {#stage-3}

*2026-04-08 (`cd031f3`) → 2026-06-09. The operating model for two months, and the one stage 10
deletes.*

### 3.1 The orchestrator becomes an agent (`cd031f3`)

**The `StateGraph` is removed entirely** — only `langgraph.checkpoint.memory` survives the
import. In its place, an orchestrator LLM is handed *the other agents as tools*:
`answer_from_memory`, `search_agent_evidence`, `analysis_agent_answer`, and later
`code_agent_answer`, assembled by `_collect_orchestration_tools`.

There is no route to plan. The route is **reconstructed afterwards** by
`_build_orchestration_trace` from which tools were actually called — which is why the README
wording changed from *"Inspect the agent graph locally"* to *"Inspect the agent flow locally"*.
There was no longer a graph to inspect.

`ORCHESTRATOR_AGENT_PROMPT` states the goal as *"answer the user query with the minimum necessary
work"*, with rule 3: *"prefer calling `search_agent_evidence` before `analysis_agent_answer`."*
Delegation is a suggestion in prose, because nothing structural enforces order any more.

### 3.2 What the shape cost

Nesting is the defining property: Orchestrator → AnalysisAgent → CodeAgent → SearchAgent can run
three deep, and *"SearchAgent is the only leaf that reaches the real RAG search backends."* Each
level is a separate executor with its own context, and evidence is not shared — each search-tool
factory got **its own empty `search_invocations` list**, so there was no dedup across the
hierarchy (fixed in `d2d23df`, which also removed an LLM call that chose among exactly one route:
*"Pure waste"*).

It is also where the monolith peaked. `f3533e1` takes the file to **2048 lines** and is kept on
the mainline as an explicit failure record: *"Try to add an orchestrator agent. The MCP tools are
all under search and try to classify them dynamically. These features are not workin as
expected."*

### 3.3 Prompt revisions in this stage

| commit | change | reason |
|---|---|---|
| `cd031f3` | SearchAgent gains *"Do not infer local file paths or use file tools unless the user explicitly provided attached/uploaded files"* | the agent was inventing file paths |
| `f3533e1` | AnalysisAgent gains *"If the user would benefit from executable code… call `code_agent_answer`"*; CodeAgent gains a `Dependencies:` section | delegation between agents is prose, so it has to be written down |
| `f3533e1` | the orchestrator's file rule is **loosened** to *"you may use file tools directly yourself"* | the strict version from a day earlier blocked legitimate use |
| `8dc7f25` | *"do not fake binary files with `write_output_file`"* | the model had been writing fabricated binaries through the text file tool |
| `7f71a90` | *"Never call `load_skill` twice in the same assistant turn"*, across four prompts, enforced in code by per-run loaded sets | a code comment names it: *"Some models pass the skill directory as `resource_path` after the main skill is already loaded"* |

### 3.4 Why it lasted, and why it ended

It survived stage 4's extraction unchanged — the packaging moved it without altering the model.
Stage 5 replaces it in a day, and `665db95` then freezes it into `agent_runtime/legacy/` behind a
registry, where it remains selectable until **stage 10** removes it. By then it had not been
touched for three months and could not reach half the tool surface.

---

## Stage 4 — The monolith becomes a package {#stage-4}

*2026-04-11 → 2026-05-07. Structure, not behaviour — it overlaps stage 3 in time and changes
nothing about how the agent decides.*

### 4.1 The extraction, done twice, independently

After `9a40136` the history **forks into two branches that both extract `agent_runtime/`**, and
they are not merged for a month.

*Branch A* (`refactor/agent-runtime-extraction`) splits the monolith in place across six commits
on 2026-04-11, each reporting the remaining size: `graph_state.py` (56) → `intent_classifier.py`
(*"2048 → 1717"*) → `tool_policy.py` → `executor_factory.py` (*"1668 → 1271"*) →
`runtime_utils.py` (*"1271 → 875"*, reason: *"pure functions with no side effects — ideal for unit
testing"*) → `graph_nodes.py` + `graph_runtime.py`, which deletes the monolith. Final:
*"7 focused modules totaling 2300 lines, with the largest being 453."*

*Branch B* (`Restructure-agent-repo-architecture`) moves whole files instead, leaving one-line
re-export shims, then decomposes them into **the same seven module names** — arrived at
independently. Its internals differ: it kept a `StateGraph` and defined a
`VERIFICATION_AGENT_PROMPT` that exists nowhere else in the history.

**The merge `c504ec0` lists 21 conflicted paths.** Branch A's seven modules won byte-identically;
branch B's file relocations won. Branch B's graph and its verification agent were discarded.
**No commit records why branch B was started two days after branch A began on the same problem,
or why its decomposition was dropped.**

The residue is still visible: seven `rag_pipeline/*.py` files are five-line `sys.modules` aliases
pointing into `agent_runtime`.

### 4.2 Around the core

`d08d580` makes `rag_pipeline/search/` a subpackage; `eea3bb8` fixes the resulting circular
import with a lazy `__getattr__`. `be91702` lifts the Flask layer into `api/` — and fixes a
Dockerfile that *"was missing both"* new packages. `9af4ea2` replaces a hand-rolled FastAPI MCP
shim with the official SDK (`FastMCP`); `a5d2461` adds dual transport (`/mcp` + `/api`) and fixes
a real bug recorded in a code comment: *"Bypassing this with `spec_from_file_location` created a
second instance of every module and broke shared state between tools."*

`db75167` adds the first loop bounds — `max_iterations=15`, `max_execution_time=120`,
`recursion_limit=25` — *"discovered during live MCP testing where an AnalysisAgent loop had to be
killed manually."*

`53dc3be`: CPU-only torch, *"Full CUDA torch pulls ~4GB of NVIDIA libraries per image, exhausting
disk on the VM."*

### 4.3 Prompt revisions in this stage

Prompts moved file (`fdff3d8` lifts all five into `executor_factory.py`, `e583e60` splits them
per layer) without changing text. The revisions themselves belong to stage 3.

### 4.4 A taxonomy that was never consumed

`722e4ed` adds `@mcp_tool(category=...)` validated at decoration time against six categories, and
says it is *"replacing the hardcoded tool-name sets in graph_state.py (consumption comes in the
next commit)"*. **That consumption never lands.** At the end of the stage `tool_policy` still
switches on the name sets, and no commit in the range touches it with the string `category`. The
taxonomy is metadata-only — and the name sets it was meant to replace are still what stage 10
finds binding half the tool surface out of reach.

### 4.5 A tool unreachable for three weeks

`9c45d82` is a one-line-per-call fix: `neo4j_search_tool` was calling the tier-3 function instead
of the 3-tier dispatcher added in `ad6361b`, so the hierarchy had been unreachable from the agent
since it was written. The same shape as stage 7's `list_conversation_files`: registered, wired,
documented, and not actually reachable.

---

## Stage 5 — Supervisor over peers {#stage-5}

*2026-06 → 2026-07. 42 commits. This is the pivot of the project.*

### Stage S5.1 The model changes (`7adf7d1`, then `fb8bdfa` one day later)

`7adf7d1` introduces `supervisor_graph.py` behind `AGENT_SUPERVISOR`, **off by default**. Its
docstring states the model: search, analysis and code are **peer** capability nodes sharing one
typed `SupervisorState`; an LLM supervisor picks the next action and the graph loops back to it.
*"This is the agentic alternative to nesting search under analysis."* Three rules: peers not
pipeline stages; operators bundled into capabilities (rerank inside search, audit inside
analysis); and **context hygiene** — the heavy evidence lives in shared state, the supervisor sees
only a distilled view.

`fb8bdfa`, one day later, does four things at once and its subject line understates all of them:

1. **The default flips** to on, with a per-request `useSupervisor` override.
2. **`finalize` becomes `synthesize`.** `analyze` stops composing prose: *"It does NOT compose
   prose."* The audit moves with it.
3. **A `needs` FIFO queue** is added to state; the supervisor fulfils the oldest peer request
   before consulting the decider.
4. **`request_capability`** — a tool, *"so this makes the 'needs' signal model-driven — the agent
   decides, mid-reasoning, that it needs another peer."*

Everything after this is consequence work on that one graph.

### Stage S5.2 Bounding the loop (`af1ead0`)

`AGENT_SUPERVISOR_MAX_SEARCHES` (2), `AGENT_SUPERVISOR_MAX_PEER_RUNS` (3), `_search_exhausted`,
`_is_unproductive_repeat` — which guards `analyze`/`code` but **deliberately not `search`**,
because search accumulates into evidence so a follow-up can add documents.

Same commit makes the grounding audit non-cosmetic: *"a flagged verdict changes the text the user
actually sees, rather than being computed and discarded."*

### Stage S5.3 Deterministic short-circuits (`4624187`, `79cb450`, `e31272c`)

Three commits convert LLM tool-choice into code paths, all from one root cause: *"nothing steered
the SearchAgent to the wired `neo4j_explore_related_nodes` tool, so it fell back to
`semantic_search`"*. The symptom was a related-elements query *"returning a generic semantic
search of topically-similar papers presented as if they were curated relationships — which the
grounding audit correctly flagged HIGH."*

`e31272c` adds a subtlety worth keeping: recalling an element from conversation must be
**role-aware**, because *"a prior ANSWER embeds other elements' UUIDs in its citation URLs"*, so a
naive newest-first scan would recall a *cited* element instead of the user's subject —
reintroducing the bug *"in a form the grounding auditor can't catch."*

### Stage S5.4 The packages split (`665db95`)

The two orchestrators move into `agent_runtime/supervisor/` and `agent_runtime/legacy/`, behind a
strategy registry, *"so neither's code/prompts can leak into the other"*. The packages never
import each other. Agents-as-tools survives only as an `AGENT_SUPERVISOR=0` fallback.

### Stage S5.5 Prompt revisions in this stage

| commit | change | reason |
|---|---|---|
| `fb8bdfa` | `analyze` redefined from *"compose an answer"* to *"run a GIS/data analysis workflow"*; `done` becomes *"a grounded final answer is composed automatically"* | stop the decider treating analyze as the answer-writer now that `synthesize` exists |
| `af1ead0` | *"Each peer ITERATES INTERNALLY… do NOT pick it again to 'retry' or 'improve'"* | the observed loop the bounds also address |
| `665db95` | `SYNTHESIS_PROMPT` replaces the reused legacy prompt | the legacy one is *"a tool-calling AnalysisAgent persona whose rule 7 — 'call `code_agent_answer`' — is contradictory here"* |
| `b060d1a` | evidence rendering, not the prompt, is changed to show only `title:`/`url:` | rule 2 alone was insufficient: `_format_documents` *"still led each evidence item with `[<doc_id>] title`, which trained the synthesis LLM (esp. the small default model) to cite `[<uuid>]`" |
| `6e48d65` | the audit prompt: *"the execution record is FIRST-CLASS grounding"* | an answer presenting a real computed result was being flagged for lacking a document |
| `0dc93e8` | the audit gains a precision section: flag only contradictions and checkable specifics | *"A correct answer that adds non-contentious domain framing… was being flagged medium and surfacing a scary '⚠️ Grounding check' caveat"* |

### Stage S5.6 A second code-peer runtime (`4758ea2`)

`AGENT_CODE_PEER=opencode` swaps the whole LangChain code peer for a container-per-run CLI. The
differences are deliberate and recorded: the container **keeps network access** (the CLI must
reach its LLM endpoint), unlike the `execute_code` sandbox which is `--network none`. This is the
boundary that still makes `analyze` and `code` genuinely distinct peers — not their toolsets,
which overlap almost entirely.

---

## Stage 6 — Map-native delivery {#stage-6}

*2026-08. 137 commits — the busiest month. Only 10 of them came through PRs; ~126 landed directly
on `prototype`, so the PR titles are not a useful index.*

There is a hard 12-day gap (Aug 6 → Aug 18) with no recorded reason, and the work either side is
qualitatively different. It is the real seam in the month.

### Stage S6.1 The delivery contract (`30cae40`, `3eaaa2e`, `261772a`, `0456bf2`)

`30cae40` adds `map-ui-prototype/` — React + MapLibre + deck.gl, chat driving the map. But the
structural change is server-side:

- `3eaaa2e` adds `agent_runtime/map_layers.py` and a **`map_layer` SSE event**, registered
  status-tier *"so it streams even without agent_dev"*. Geometry previously *"only traveled as a
  truncated tool_result."*
- `261772a` adds `add_map_layer`, filling the gap that *"There was no tool at all that produced a
  styled MAP LAYER"* — the existing tools only rendered PNGs. The descriptor carries a **URL
  rather than inlining 50k points**.
- `0456bf2` renames five tools so names match products (`plot_vector → render_map_image`,
  `kb_point_heatmap → heatmap_image`, …): *"Four tools had map-sounding names but rendered a
  static PNG."*

**`build_map_layer` becomes the single boundary every layer crosses**, and the rest of the
project either exploits or repairs that invariant.

### Stage S6.2 Prompt philosophy inverted — mandates become capability statements

This is a deliberate, documented reversal, and `28cc904` flags it as *"the shape a future editor
is most likely to 'tidy' back into 'you MUST' without knowing it was tried."*

`818fe4b` is the clearest statement. `CODE_PEER_PROMPT` carried *"you MUST RUN your code… an
answer that only pastes code is a FAILURE"* with nothing checking it:

> *"That is the shape most likely to backfire: a model told non-execution is a failure will claim
> it ran when the sandbox dies — which is exactly what we watched happen when a run exited -11
> with empty stderr and the model reported 'dependency issues'."*

The mandate is replaced by a **structural check**: the peer node looks for an `execute_code`
record in its own tool calls and re-invokes **once** with the observation. The same shape then
recurs four more times (`7e3c356` map delivery, `8489d94` repeated failures, `a19df4b` layer QA).

`839a855` and `f8a2803` apply the principle to the search and decide prompts: *"each duplicated a
deterministic mechanism in supervisor/graph.py… so the prose could not change behaviour when the
detector fired and was unreliable when it did not."* `f8a2803` adds `_available_actions(state)`
so the decider is shown only what it may pick, instead of prose forbidding things.

A related lesson, from `d15bce9`: the steer moved **out of the prompt into the tool result
payload** — *"beside the data rather than only in a system prompt far above it."*

### Stage S6.3 The grounding audit, rewritten from measurement (`11490d6`)

The most carefully measured prompt change in the project. The symptom: the same answer, clean and
with a fabricated journal, date, institution, benchmark score and price appended, got **the same
verdict** — and *"in both cases the flagged claims were the LEGITIMATE ones while not one
fabrication was noticed."*

Two structural causes:

1. **Verdict before proof.** `hallucination_detected` and `severity` were the *first keys* of the
   response schema, so *"the model committed to a verdict autoregressively and then backfilled
   rationalisations."*
2. **No obligation to look.** Every ledger row now demands a VERBATIM span, *"and a row without
   one cannot be 'supported'."*

New wording: *"You must work CLAIM BY CLAIM, and in this order. Do not write a verdict before the
ledger exists."* And: *"Every number and every proper name in the answer gets its OWN row"* —
against a measured failure where *"an invented figure and benchmark embedded in an
otherwise-supported sentence were summarised into one 'supported' row and passed clean 3 times
out of 3."*

The meta-lesson is in the commit: *"Prose rules did NOT work… Four formulations were then measured
against a fixed bar and independently re-verified; two passed the bar but broke under further
probing."* The in-code comment ends *"Keep changes to this prompt measured."*

### Stage S6.4 The web, behind an SSRF guard (`f16ae4b`, `e0741a6`, `aa7105b`)

`web_search` is metadata-only by construction, and deliberately excluded from the every-turn
sweep: *"every other method there is a cheap in-house call, the open web is a third-party network
hop."*

`web_fetch`'s guard is built on **"RESOLVE, THEN CLASSIFY THE ADDRESS — never the hostname
string"**, verified against `http://127.1/` and `http://2130706433/`. A private-range check was
judged insufficient *"because its own services are published on PUBLIC addresses."*

`aa7105b` then closed **12 bypasses from an adversarial review (16 claims, 12 confirmed, 4
refuted)**. The critical one *"needed a single character"*: a trailing DNS root dot
(`storage-dev.i-guide.io.`) missed exact-string deny-list membership and reached the real object
store. DNS rebinding is confirmed exploitable and **only partially closed** — the commit says so
rather than implying otherwise.

### Stage S6.5 The spatial toolkit, and an absence that produced a false answer

`7cb9f47` adds seven PySAL/GeoDa tools with a stated engine policy: *"One engine per job so the
model is never choosing between two ways to compute the same number."*

`2212f8a` adds `select_by_attribute`, and its reason is the sharpest argument in the history for
minding gaps rather than only bugs:

> *"the analyze peer had 21 spatial tools and no way to isolate one feature… So it buffered all
> 708 grid cells — 4,504 overlapping polygons covering the whole city — and reported 'a 2 km
> buffer around the busiest grid cell'. Attribute selection is the plainest GIS operation there
> is, and its absence produced a false answer."*

### Stage S6.6 Remote-sensing embeddings (`28832b5`, `b44a202`, `9b1001b`)

Seven tools **proxy** the rs-embed service rather than importing it, *"which keeps torch /
earthengine-api / geemap out of the agent environment and Earth Engine credentials in one place."*
Georeferencing is computed agent-side because the service's footprint *"is a square in EPSG:3857,
where a metre is 1/cos(latitude) too long."*

Two measurement-driven decisions worth carrying: zones carry **sum and pixel count, not a mean**,
because *"A mean of means is wrong across unequal zones"*; and scoring is **spatial-block CV with
the naive score reported beside it**, because *"a health outcome scored +0.15 under a random split
and -0.91 when whole blocks are held out."*

`9b1001b` establishes a hard rule: nothing on the zonal path may import scikit-learn — *"Its
KMeans does not warn beside torch, it SEGFAULTS: the pytest run died outright, mid-suite."*

### Stage S6.7 Model provider becomes a per-request choice

`81f6e9e` adds `GET /agent/models` and per-request model/provider/effort. `34d1fb5` replaces a
prefix heuristic with a **measured per-model table** after picking `gpt-5.6-luna` made every later
turn fail and, because the choice persisted to localStorage, *"the setting was unrecoverable from
the UI."* Key row: *"gpt-5.6-\* REFUSES tools unless reasoning_effort='none' is sent."*

Two commits **retract their own earlier claims** after re-measuring — `cd1adae` (*"I said twice
that CLAUDE_CODE_OAUTH_TOKEN 'cannot call the Messages API'… It is wrong"*) and `c0d5617`. And
`6198e7b` removes a liveness probe added an hour earlier, with the measurement that killed it:
*"Within the same minute on this credential, claude-haiku-4-5 answered with tool calls while
claude-sonnet-5 and claude-opus-5 both returned 429. The limits are PER MODEL."*

### Stage S6.8 The first move toward collapsing the peers (`299e35d`, `3b7e181`, `daf5862`)

`299e35d` records the design conclusion: *"The peers have never run in parallel — decide() returns
one action per step through conditional edges — so the usual justification does not apply."* What
the split does buy is *"a place to stand outside the loop"*: the in-loop LLM audit passed a wrong
model attribution that only the deterministic outside check caught.

`3b7e181` adds a context budget (*"BoundedInMemorySaver caps THREADS, not messages inside one…
One clay turn hit 199,605"*) and an opt-in unified peer, per-request *"because otherwise the two
architectures could only be compared by restarting the deployment between arms."*

`daf5862` is the immediate fallout, and the finding generalises: removing `search` from the menu
*"also removed the decider's cue that retrieval was the opening move"*, so a retrieval question
went straight to `done`. **Shaping the menu alone did nothing; a veto in `supervisor_node` was
required.**

---

## Stage 7 — The action ledger {#stage-7}

*2026-09-01 → `9e35950` (2026-09-10). 106 commits.*

The stage's centre is one idea: **record what tools DID, not that they ran.** Eight commits build
it, and everything else in the stage either feeds it or reads it.

The justification is stated in `9e1201e` and proved by `7f0888f`, whose self-assessment is the
most useful sentence in the history:

> *"I diagnosed these two lines twice from the trace alone and was wrong both times… Two Earth
> Engine sweeps and two wrong commits, because the trace shows that a tool was called and never
> what it returned."*

### Stage S7.1 Building it

| commit | change | reason |
|---|---|---|
| `3d9587f` | the ledger gets its own delivery channel instead of riding as `chat_history` item 0 | measured present *"at 2-7 history items and ABSENT at 8+"*, while the auditor got it regardless — so the answerer was *"told to answer from a line the auditor could not see, then flagged for hallucinating it."* The browser check that passed when it shipped *"had two history items. It was inside the only window where the wiring worked."* |
| `86f2922` | failure becomes row-level; facts curated | rows were bucketed by tool name and zipped positionally, so a fail-then-succeed pair put the good run's `file_id` **on the FAILED row**, *"where `_map_delivered_earlier` read them as a delivered layer."* `execute_code` had produced zero rows |
| `1deeec1` | retrieval joins the ledger | *"every retrieval method left NO trace and the ledger could not answer 'what did we search for?' — the most common follow-up, and the one whose absence sends the agent searching again for something already in hand"* |
| `9e1201e` | `_visible_state_lines` — what is still on screen | the map is persistent, so *"no map was produced"* was a false statement the answerer had no way to check |
| `02473bf` | the budget is measured in the unit each consumer pays | `_budgeted` sized rows by `json.dumps` while the rendered form expands 3.7x: *"19 rows passed a JSON budget of 6,000 and rendered 14,764 chars"* |
| `4fcd474` | the whole note is bounded | the visible-state section was appended *outside* the budget: *"88,250 characters against a 6,000 ceiling — a context overflow caused by the mechanism whose own comment says it exists BECAUSE a turn overflowed the context window"* |
| `e8f57de` | output `file_id` is rendered, gated on `outputs` **not** on `file_id` | `read_text_file` returns the id of a file the user uploaded and creates nothing; *"the grounding auditor reads these same lines as evidence, so it would have confirmed the fabrication"* |

Three of those eight exist because the mechanism that prevents context overflow was itself
overflowing context.

### Stage S7.2 One authority for "is it on the map?" (`8f9f24a`)

Four signals answered that question, combined with `or`, *"so the weakest won"*: a tool name in
`tool_calls` (never checking success), a bare nested `"on_map": true`, a recursive name match, and
a regex over the JSON blob. A **failed** `admin_boundary` tripped two of them; the supervisor then
*"suppressed its own corrective retry, wrote the conclusion into `result["on_map"]`, and RE-READ
that conclusion as evidence a layer existed."*

`map_layers.delivers_map_layer` becomes the single authority. Found in passing: `vector_spatial_join`
set `on_map` with no descriptor, *"so nothing has ever reached the map from it."*

### Stage S7.3 Per-turn scoping (`2be2b83`)

Peer threads outlived the turn, so four verifiers asking *"what happened THIS turn?"* got the
whole conversation: *"last turn's `execute_code` made this turn's bare code fence report
executed=True"*; *"turn 1's documents came back as turn 2's evidence."* `PeerSession` scopes per
**invocation**, not per turn, because `default_analyze_fn` invokes the same thread three times and
concatenated slices made *"two failures of one tool render as four."*

### Stage S7.4 The audit becomes a gate (`05fa222`, `e9801dd`, `76df5b9`, `29f24c2`)

`05fa222` routes one corrective pass back through the **needs FIFO** rather than a plain edge —
*"the decider is precisely what already said 'done' on this state."* Reproduced with *"Which
counties border Champaign County?"*: the peer downloaded a Census gazetteer, *"computing no
adjacency, and from a gazetteer it could not, since those carry centroids and not geometry"*, then
answered from memory. Corroboration that it was recall: *"Vermilion was placed 'to the east' in
one run and 'to the northeast' in another."*

Three false-positive classes were closed first:

- `e9801dd` — a verified-correct turn shipped a caveat because the auditor flagged *"You can pan,
  zoom, and click the hospital markers"*. Structurally unprovable: an affordance is a property of
  the **client** that no tool result can report. Fixed by injecting a `_MAP_CLIENT_AFFORDANCES`
  line into the auditor's environment **only when a layer really was delivered** — plus
  word-boundary matching, because *"'pan' is inside 'expand'/'Japan'/'company' and 'click' is
  inside the '[popularity: 42 clicks]' real evidence carries."*
- `76df5b9` — the auditor was starved: *"87,648 chars of record, 2,218 reaching the auditor, and
  exactly ONE of the 26 county names surviving"*, because the cut was a blind prefix over a dump
  that is *"94.2% coordinate arrays"*. Elision is **size-gated, not key-based**, because *"a bbox
  is 4 numbers"* and answers quote those.
- `29f24c2` — truncation markers now say a cut was a cut (*"truncated is not absent"*), and the
  turn **under audit** stops being the worst-described section: earlier turns arrived as rendered
  ledger lines while the current turn arrived as raw JSON, so *"a tool's ARGUMENTS were quotable
  for every turn except the one being judged."*

### Stage S7.5 Capability introspection becomes discovery (`6ba1bd3` … `cd083ad`)

`9e33911` found the capability answer received the inventory and **no query**, and truncated it:
*"`json.dumps(inventory, indent=1)[:12000]` against a 27,397-char blob dropped 56% of the surface…
and sliced mid-object so the model received malformed JSON. Measured: 0 of the 6 embedding tools
reached the prompt."*

`fb9baf5` replaces 20 hand-written try/excepts with convention-based discovery: *"Discovery finds
17 factories against 12 hand-listed, and 70 tools against 66"* — and notes the hand list *"was
wrong again within the same session that fixed it."* `cd083ad` then finds 12 more hidden by a
keyword-name mismatch swallowed by a bare `except`.

**A conclusion recorded here and worth keeping** (`cd083ad`): discovery is *not* this system's
failure mode. *"Of 15 documented selection failures 7 were 'the tool was ABSENT', which no lookup
can fix, and of the 8 'present and not chosen' none was ever fixed by a lookup mechanism; every
remedy that worked was a deterministic bypass, an out-of-loop retry, a rewritten description, or
the ledger."* A planned capability map in every peer prompt was therefore deliberately **not**
shipped.

### Stage S7.6 Context budget from measurement, not assumption

`c15c3da` replaces a flat 48,000 ceiling *"wrong in BOTH directions"* with a per-request
derivation, fixing accounting bugs including that `request.messages` excludes the system message
and `tools` is a sibling field — *"the old count omitted the larger half of the analyze peer's
request."*

`4a36a0e`: `gpt-5.6-luna` matched no prefix and inherited the 65,536 floor, so *"every call
capped messages at ceiling=50,825 against a real limit of 922,000 — the budget was discarding
~94% of the usable window."*

`ee77a2c` probes all 26 models with a deliberately oversized request: `gpt-4.1` assumed 128,000,
**actual 1,047,576**. The gpt-5 family splits into 922,000 and 272,000 tiers that *"do NOT follow
version order"*, making prefix **order** load-bearing.

### Stage S7.7 Layer identity becomes a content digest

Seven commits over two days. `349a6f7` → labels carry a region tag; `895fb4a` → `slug[:40]` was
the id, and *"a one-character margin decided which region kept its layer"*; `3c73f3b` → the id
becomes a digest of everything that decides **content** and nothing else, with the caller's `name`
deliberately excluded (*"Renaming a layer now leaves it the same layer"*).

Two adversarial reviews found **twelve** and **fifteen** confirmed defects respectively, including
one the change itself introduced. `895fb4a` also records that the author's own earlier commit
caused the bug it fixes.

### Stage S7.8 Terrain, and two misregistrations found by arithmetic

`c169723` adds `dem_for_region`, deliberately **not** through the rs-embed service: *"That service
holds the Earth Engine credential — a personal Google account that has expired twice — and
elevation does not need it."* Two container-only facts were measured on the deployed image, not a
laptop: `PROJ_LIB` points at a v5 `proj.db` while rasterio needs ≥6, so `CRS.from_epsg(4326)`
*"FAILS in production today — this would have shipped broken while passing locally"*; and
`cm.get_cmap` was removed in matplotlib 3.11.

`3ac9b76` / `0e07fa7` fix the DEM in both directions. 3DEP pads the shorter axis, so *"a
0.0275-degree-tall request came back 0.0360 degrees tall — 466 m added at each edge"*, and draping
over the requested box squeezed it 23.5% — *"8 px at zoom 11, 510 px at zoom 17."* Found by
arithmetic, not by looking, and the tests **could not** have caught it: *"the fixture returned a
raster whose bounds equalled the request, which is the one thing the real server never does."*

### Stage S7.9 File-store session scoping (`9b83546`, `0fa5d25`, `db843f8`)

*"The file record had seven fields and none of them said who wrote it, so the store was one flat
space shared by every session."* A session id is stamped at creation and bound at the request edge
as a ContextVar — *"the same shape the streaming trace state uses… When JWT arrives it changes
where that id comes from, not what is done with it."* That sentence is the plan stage 9 executes.

`0fa5d25` is the follow-up worth remembering: **ContextVars do not cross threads**, and the agent
runs on a worker. *"Every unit test passed and the live path still wrote session=None, because the
tests all ran on one thread."*

`db843f8`: registering `list_conversation_files` was not enough — the file toolset attaches only
on an upload turn, so *"the analyse peer had no file tool, wrote `os.listdir('.')` in
`execute_code`, listed the sandbox working directory, and reported its own scratch script as one
of the artifacts."*

### Stage S7.10 Prompt revisions in this stage

| commit | change | reason |
|---|---|---|
| `3d9587f` | `SYNTHESIS_PROMPT` rule 8: earlier-turn tool records are *"first-class grounding"*; *"A line marked FAILED means the tool did not work"* | worded **conditionally on purpose** — `default_compose_fn` reuses the prompt without that section |
| `9e33911` | the capability prompt gains the question: *"A user has asked what you can do. Answer THEIR question"* | it had been composing an identical grouped catalogue for every capability question |
| `05fa222` | `_REGROUND_DIRECTIVE`: *"downloading or inspecting a file is not the same as computing the answer"* | names the observed gazetteer trap |
| `5f828f4` → `4fcefd7` | two successive rewrites removing instructions the sandbox cannot satisfy — *"REUSE it verbatim — including real data-loading URLs/APIs"* under `--network none`, then its replacement naming `web_fetch`, which *"returns a page's on-topic passages, not the bytes of a dataset"* | a prompt that names an unavailable route teaches a failing habit |
| `ae813f6` | *"UNSURE OF AN API? Look it up before you write against it… a lookup is something you do BEFORE execute_code"* | *"a bound tool the prompt never mentions does not get used"* |
| `d1b171a` | `"Routed to {route}"` becomes a name table; the orchestrator label is renamed on the supervisor arm only | `fast`/`orchestrate` *"are node names in THIS graph"*, not facts about the request; legacy keeps the old label *"because there it is accurate"* |

---

## Stage 8 — What the decider reads {#stage-8}

*Branch `claude/evidence-summary`, 7 commits off `9e35950`. Unmerged.*

Three things the supervisor chose from turned out to be wrong the same way: each was **cheap and
available** rather than **what the decision needed**.

| | before | after |
|---|---|---|
| what the evidence is | counts, titles, `topical_coverage`, `top_score` — all lexical | plus `evidence_summary`, written by the model that read the documents |
| what peers can do | a hand-written paragraph, derived from nothing | generated from `capability_registry`, held against the peer builders by test |
| what this turn did | booleans and counts; the ledger of *previous* turns only | the same `this_turn` lines the answerer and auditor already read |

### Stage S8.1 Evidence gains a description (`1044afb`)

Lexical signals cannot separate PySAL accessibility notebooks from DEM sources when both mention
"elevation". Measured on a self-hosted model: two full search rounds where the second added
nothing, because *"is this enough?"* was being answered from counts.

Two rules keep it from making things worse: it **describes and names gaps, it does not rule on
sufficiency** — that judgement belongs to the decider, and a summary announcing "this is enough"
would collapse two independent checks into one — and it sits **beside** the lexical signals rather
than replacing them, so a wrong summary can be disagreed with.

### Stage S8.2 The capability paragraph becomes generated (`99d71ad`)

The supervisor's description of its peers had drifted behind them **three times**: terrain,
administrative boundaries and geocoding were all bound to a peer while the prompt never mentioned
them. The cost was not cosmetic — asked for a DEM, the supervisor searched the knowledge base,
because as far as it had been told, `analyze` did overlays and embeddings. *That was a correct
decision from a stale description.*

The drift guard was written first, against terrain alone, and **failed on its first run naming two
more** nobody had noticed. It now holds the registry against reality in both directions: a bound
toolset the registry omits, and a toolset the registry claims that no peer binds.

Live result: the same model, told the truth, went `analyze → code → done` with **zero searches**,
against `search → search → analyze → code → done` before.

### Stage S8.3 The decider joins the ledger's readers (`25c3e1e`)

Asked for a DEM, the supervisor ran `analyze` — which fetched and drew it — then routed to `code`,
which fetched the same DEM again. In a sweep that second pass cost **266 s and 16 `execute_code`
iterations** to redo work one tool call had done.

The rows were never missing. `_ledger_lines` had exactly **two** consumers — the answering model
and the grounding auditor — and the decider was not one of them. This is the same fix stage 7's
`29f24c2` made for the auditor, which had the identical blind spot.

**Revised during the work:** this began as a `map_layer_delivered` boolean. The boolean was kept
for the one question the prompt asks directly, but it was treating the symptom; the ledger is the
structural answer, and it is richer, already budgeted, and shared with the other two readers.

### Stage S8.4 The tool surface absorbs how models actually call tools

A sweep of ten spatial prompts through a self-hosted model, collecting every failed tool call,
found four defects of one class — **a parameter whose name or type invites the wrong value**:

| commit | defect | measurement |
|---|---|---|
| `8fe0dcb` | optional parameters rejected an explicit `null` | **141 parameters across 62 of 80 tools**. Not fixable in the function body: pydantic validates against the schema LangChain infers from the signature, so the call dies before any code runs. The wrapper rewrites the signature |
| `ae862b5` | a GeoTIFF was neither vector nor image | four calls to draw one DEM. `add_raster_layer` now reads the extent **from the file**, which wins even when bounds are passed — a caller restating the box can only agree or be wrong |
| `ae862b5` | `session_context_json` refused a dict | the identical search issued twice, the first call wasted |
| `5a10a58`, `aaac2c3` | `area` held the place name but read like the kind of place; `name` meant the output filename | four calls, and the model had the right answer in `name` from the first. `name` now means the place; the filename moved to `output_name` |

`0` and `False` are deliberately **not** treated as null — they are answers, and substituting a
default for them would silently ignore the caller.

After these, a re-run sweep showed zero failed calls on boundary, tracts, geocode, DEM, slope,
buffer, OSM and raster. Two cases got slower, which is single-run variance and not claimed as a
regression either way.

### Stage S8.5 The `code` line is generated too, per backend

*Branch `claude/decider-code-line-from-registry`, off `5ae6d92`.*

S8.2 (`99d71ad`) generated the decider's `analyze` line from `capability_registry`. In the same
commit it wrote the `code` line by hand: *"It binds the same toolkit as analyze, plus packaged
skills and saved workflows."* That one sentence drifted in both directions at once, which is the
lesson worth keeping: **the router's description of a peer has to come from the same source as the
peer's binding, or it drifts, toward promising too much as readily as too little.**

* **Too little.** `describe("code")` had **zero** call sites outside tests. A registry entry that
  only the code peer binds reached the drift test, which composed its own view (the framing,
  `describe("analyze")` and `describe("code")`), and never reached the decider. The test
  certified an artifact production did not use. This is not hypothetical: the extraction branch
  adds `make_langchain_staging_tools` as exactly such an entry.
* **Too much, three ways.** The code peer binds no tool that runs a saved workflow. A notebook
  workflow the extractor packages as a skill runs through an MCP tool (`mcp_run_nbwf_*`), and MCP
  tools are bound in the analyze and search peers behind `include_mcp_tools`, never in the code
  peer. "Packaged skills" was false in production: the image copies neither default skill root,
  and inside the container `SkillRegistry.discover()` finds **0** skills. And "the same toolkit as
  analyze" is false for a CLI code peer (`claude`, `opencode`), which binds **none** of the 15 code
  toolsets. It runs an agentic CLI in its own container, whose only writable mount is `/work`. The
  deployment default is `langchain`, but a browser's dropdown choice persists, so CLI peers do run.

| | before | after |
|---|---|---|
| the code line | hand-written, identical for every backend | `_code_capability_line`: the registry inventory for the LangChain peer, the CLI's own runtime for a CLI peer |
| which backend the decider describes | none; it was never told | `_code_peer_backend(code_peer)`, the one resolution the code node also uses |
| the skills clause | always present | present only when `SkillRegistry.discover` over the request's skill roots finds a skill, the condition `make_skill_tools` binds on |
| "saved workflows" | promised | removed |
| the request example | *"e.g. code needs evidence"* for every backend | *"e.g. analyze needs evidence"* when a CLI runs, since a CLI has no `request_capability` |
| the drift test | read a separately composed description | reads the prompt captured from `default_decide_fn` itself |

**The resolution was the part that would have been missed.** The request's `code_peer` reached
`default_code_fn` and nothing else. `build_supervisor_graph` built the decider without it, so a
per-request choice could only ever have been described as the env default. Now
`run_supervisor_orchestration` builds the decider from the same `cfg.code_peer` and
`cfg.skill_roots` as the code peer, and both resolve through `_code_peer_backend`. A test replaces
that one function and watches both the runner and the description follow it.

**The cost, measured with gpt-4o's tokenizer (`o200k_base`):**

| | before | after |
|---|---|---|
| code line, LangChain peer, no skills (the deployed image) | 30 | 182 |
| code line, LangChain peer, with skills | 30 | 194 |
| code line, a CLI peer | 30 | 189 |
| whole decider prompt at minimal state, deployed image | 954 | 1,106 |

That is +152 tokens, about +16%, on every decision step. Every one of the deployed code line's 14
clauses also appears in the analyze line. Rendering the code line as a difference from analyze
would recover most of that; it is not done here.

**Not fixed here:**

* **The other peers bind skills too.** `build_agent_executor` has added `make_skill_tools` to
  every preloaded-tools peer, analyze included, since `6ba1bd3`, and `collect_tools` adds it for
  search. The registry still lists skills as code-only. The drift test scans only `graph.py` for
  `make_*_tools(` calls, so it cannot see a binding made elsewhere. This errs toward describing
  too little, the cheaper direction.
* **The Dockerfile still ships no skills.** That is a separate fix. The decider is now right
  either way, because it describes skills exactly when the peer will have them.
* **The CLI description is declared, not derived.** `CLI_PEER` in the registry is written from
  `claude_peer.py` and `opencode_peer.py`. One clause is held to the binding by a test: neither
  `build_docker_argv` passes `--network none`. The others are not: staged uploads, inlined evidence
  and analysis, and GeoJSON output becoming a layer.

**For the extraction branch.** Its per-consumer capability table names these same two peers
`code_peer` and `cli_peer`. `CODE_PEER_CONSUMER` maps each backend onto those keys, so the code
line can read that table's rows rather than a parallel structure, whichever branch lands first.

### Stage S8.6 The skill loaders are both peers', and the drift test reads each peer

*Branch `claude/skills-shared-drift-sees-all-bindings`, stacked on S8.5 (`05f562b`).*

The first of S8.5's three open items was that the registry listed the skill loaders as the code
peer's alone, though `build_agent_executor` has given them to analyze since `6ba1bd3`. This closes
it, and closes the reason it could stay open: the drift test could not see where analyze's binding
is made.

**The registry was written from the one file its test read.** `99d71ad` (2026-09-17) created the
registry, with a drift test that regex-scanned `graph.py` for `make_*_tools(` calls and pooled
every peer into one set. Fifteen days earlier, `6ba1bd3` had put the skill loaders into
`build_agent_executor` for every peer that hands it a preloaded tool list, analyze included. In
`graph.py` only the code peer calls `make_skill_tools`, so the registry filed skills as code-only,
and the test agreed: the factory was in the pooled set (the code peer's call) and in the registry
(as code's). **A pooled set cannot fail a toolset described for the wrong peer**, so scanning more
files would not have caught this either. The search peer was invisible for a plainer reason. It
calls no factory in `graph.py` at all, and all four of its toolsets come from
`tool_policy.collect_tools`.

| | before | after |
|---|---|---|
| skills clause on the analyze line | never | when discovery over the request's skill roots finds a skill |
| skills clause on the code line (LangChain peer) | under that same condition | unchanged |
| with a CLI code peer | no line mentioned skills, though analyze could load every one | the analyze line does |
| the analyze inventory | `describe("analyze")`, so the default roots | the request's `skill_roots`, as the code line already read |
| what the drift test reads | one regex over `graph.py`, all peers pooled | per peer: its own function, plus a run of its real assembly |

**Why shared, rather than left code-only or dropped:**

* **The binding is the same.** All three peers are built with the same `cfg.skill_roots`. The code
  peer's own call and `build_agent_executor`'s produce the same `list_available_skills` and
  `load_skill`, and the second is de-duplicated by name.
* **A shared capability described as one peer's is a routing signal.** On the code line alone,
  "loading packaged skills" made matching a skill look like a reason to choose code. The difference
  the two lines exist to draw is existing tools versus new code, and matching a skill does not
  decide that. The checkout's three skills split three ways. `chicago-crime-analysis` names
  `agent_kb_search`, `get_kb_block` and `execute_code`, which the code peer binds by default and
  analyze only partly outside unified mode. `ai-agent-for-chicago-crime-analysis` names
  `mcp_run_nbwf_d01e717421c1b0ff`, which no peer can call, because no registration path produces
  that name. The extractor names tools `mcp_run_<workflow_id>`, generated tools are registered as
  `notebook_<name>`, and the MCP server's one runner, `run_notebook_workflow`, refuses unless
  `AGENT_ALLOW_WORKFLOW_EXEC=1`. `example-skill` names `keyword_search` and `semantic_search`, which
  outside unified mode only search binds. No one peer can follow every skill. So the clause
  promises loading, which both can do, and the rest of each line says which tools a peer has to
  follow one with.
* **With a CLI code peer, the old filing said nothing.** A CLI loads no skills, so the code line
  drops the clause, and with skills filed as code-only, no other line carried it.
* **Dropping it from both lines was the other option.** That needs a bound-but-not-told exemption
  for both toolkit peers, and leaves a request that names a skill with no line mentioning skills.
* **Not measured.** No routing run compares the two descriptions. The effect on decisions is
  argued here, not observed.

**Search binds them too, and is deliberately not told.** `collect_tools` gives search
`make_skill_tools` and `make_quality_tools`, plus `make_langchain_mcp_tools` behind
`include_mcp_tools`, beside the retrieval tools. That flag is on by default: an API request that
omits `includeMcpTools` gets `AGENT_INCLUDE_MCP_TOOLS`, which defaults on. Its line stays
hand-written because it states a job, "retrieve evidence", not an inventory. A skills clause
there would make the retrieval peer a place to send analysis. The test now holds that as a
decision rather than an accident. Each toolset search binds is either stated on its line in the
words `REQUIRED_TERMS` gives it, or listed in `NOT_TOLD` with the reason, and an entry that
outlives its binding fails.

**The drift test reads each peer, wherever the binding is made.** A peer's bindings are the union
of two readings:

* **Static.** The `make_*_tools` calls inside the peer's own function in `graph.py`, each one
  whatever condition guards it. They are parsed with `ast` rather than a regex, so a factory named
  in a comment no longer counts.
* **Probed.** The peer's real assembly code runs once with an upload and once without, with every
  other flag a binding hides behind switched on. Every factory the repo's own discovery
  (`capabilities._discover_registry_factories`) finds is replaced by a recorder, and
  `create_agent` by a stop. Each recorded call keeps the file and function it came from, so a
  failure says where the binding is made. The first run against S8.5's code failed with
  *"the analyze peer binds make_skill_tools (from executor_factory.py:build_agent_executor), but
  capability_registry does not describe them for analyze"*. The same run failed that regression's
  named test, and six tests of the analyze line: whether it carries the clause, and whether it
  reads the request's skill roots.

**Why a probe and not a wider scan.** `build_agent_executor` binds different toolsets depending on
whether it is handed `preloaded_tools`. A scan would have to restate that branch in the test, and
a scan of `collect_tools` would credit its toolsets to analyze and code, which never reach it. The
probe follows the branch the peer really takes, into whichever module it leads, and records every
factory the repo's discovery knows. A synthetic test adds a toolset at the preloaded seam and
checks that the probe sees it for analyze and code while the static reading does not, so the
probe's reach is tested without depending on where today's bindings happen to live. One static
check keeps the old scan's reach: every factory called anywhere in `graph.py` must be bound by
some peer. It works per factory, so a fourth peer defined there fails it by binding a factory no
other peer has, but not by binding only factories the three peers already bind.

**The cost, measured with gpt-4o's tokenizer (`o200k_base`) at minimal state, as in S8.5.** The
prompt changes by one inserted clause, *"; loading packaged skills: step-by-step instructions for
particular analyses"*, which is 12 tokens:

| whole decider prompt | before | after |
|---|---|---|
| LangChain code peer, no skills (the deployed image) | 1,106 | 1,106 |
| LangChain code peer, the checkout's 3 skills (the image once PR #31 lands) | 1,118 | 1,130 |
| CLI code peer (`claude`), no skills | 1,113 | 1,113 |
| CLI code peer (`claude`), the checkout's 3 skills | 1,113 | 1,125 |

That is +12 tokens, about +1.1%, on each decision step while discovery finds a skill, and nothing
in the deployed image until it ships skill roots. The test costs more than the prompt. The probe
imports what the peers import, `torch` included through the retrieval tools' search stack. Run
alone, the file took 0.6–1.1 s before and 6.5–44 s after in pytest's own timing, and 73 s the
first time on a cold cache. That was on a Mac at a load average of 20 to 46, which is why the range
is wide. In the
full suite those modules are imported before this file runs. Run after `test_capabilities.py`,
which imports the same modules, all 58 tests in the file took 0.3 s together.

**Not fixed here:**

* **A factory is one toolset.** The probe replaces each factory whole and does not look inside it.
  `make_langchain_granular_tools` carries `make_langchain_qgis_tools`,
  `make_admin_boundary_tools` and `make_langchain_file_tools` into the search peer, and nothing
  here accounts for them there.
* **The registry describes every binding a peer can have, not the ones a request gets.** Analyze
  is told it retrieves datasets, publications and notebooks, though it binds the retrieval tools
  only in unified mode (`AGENT_UNIFIED_PEER`, off by default). The code peer keeps four of them
  (`agent_kb_search`, `get_kb_block`, `web_search`, `web_fetch`). S8.2's registry has always read
  this way, and it is unchanged here.
* **A binding outside `graph.py` behind a condition the probe does not set** is seen by neither
  reading. The probe sets every flag that gates a binding today.
* **Search binds the widest MCP surface, and the decider is told about none of it.** With MCP on
  (the API default) and no `mcp_modules`, search's MCP tools are every tool the MCP server
  registers. Without a server, the local fallback loads the default module list instead. Analyze's
  are scoped to `spatial_analysis_tools`, yet the analyze line is the only one that mentions MCP.
  Found here, not changed.

---

## Stage 9 — Who the caller is {#stage-9}

*Branch `claude/jwt-identity`, off `9e35950`. Merged to `prototype` at `e0e1f92`; the branch
continues to carry later identity work.*

### Stage S9.1 Named deployment modes (`152a537`, `f9b7081`)

`AGENT_MODE=dev|demo|token` replaces three booleans whose eight combinations included five
nonsensical ones (*"settings hidden AND a key required"* is a page demanding a credential it gives
you no way to enter). `PLATFORM_TIER=dev|prod` does the same for four platform URLs that must
agree. Both raise on an unknown value rather than falling back: this selects security behaviour,
and a typo silently resolving to a working mode is the failure nobody notices.

Deliberately, **the mode does not decide the API key** — that would make `AGENT_MODE=dev` mean one
thing on a laptop and something else on the public dev tier.

### Stage S9.2 Identity (`69863b8`, `95a48de`, `c59a69d`)

The agent is served from `agent.i-guide.io`, same origin as the map UI and same registrable domain
as the platform, and `JWT_TARGET_DOMAIN` is `.i-guide.io` — so the cookie arrives on its own,
including on the `<img>` request for an inline artifact. **No token exchange, and no signed
download URLs.**

Four choices, each a way this fails open if reversed: `algorithms=["HS256"]` pinned (a decoder
trusting the token's own `alg` accepts the `none` forgery); `exp` required; a missing or
non-numeric `role` **refused, never defaulted** (0 would be the most privileged caller); and
expiry raising separately from invalidity, so the endpoint answers **401 = refresh and retry** vs
**403 = stop**.

`95a48de` adds introspection — forwarding the cookie to the platform's own `/api/check-tokens` —
because against **production** the reasoning inverts: the HS256 secret mints a token for any
account, and this host runs LLM-generated code in a Docker-socket sandbox.

`c59a69d` is a defect found live: token mode ran the API-key gate **before** identity, so a
signed-in visitor with no key was refused by the key check — and token mode hides the settings
panel, so they could not supply one. *"Sign in, then: You are not signed in."*

### Stage S9.3 Ownership (`d934cbe`, `e067055`, `1f880d3`)

`GET /agent/files/<id>/download` had never checked anything, and every answer publishes file ids
as links — so each was effectively a permanent public URL. A mismatch answers **404, not 403**,
because a 403 confirms the id exists and makes the endpoint an enumeration oracle.

`get_or_create_memory` fetched by bare UUID with no owner check; worse, the create half would have
**indexed over** a document it had just refused to read. That is why ownership is asserted at the
edge rather than guarded at each call.

`1f880d3` stores the client's own `StoredSession` rather than rebuilding it server-side, because
`sessionStore.ts` was written server-shaped on purpose and rebuilding would duplicate its
layer-descriptor rules in a second place where they would drift.

### Stage S9.4 The browser stops owning the session (`84c60b2`, `4daf2e1`)

Refresh is the **browser's** job: it calls the platform's refresh endpoint, and the agent never
holds a refresh token. Every retry is bounded at one attempt, and only an expired token retries at
all.

**Revised during the work:** signing out left the history list fully populated, because IndexedDB
is per-origin and signing out of the platform does not touch it. The first fix filtered the local
store by owner. That was the wrong fix — conversations belong to the *user* and should follow them
to another browser — so in token mode the server owns history and IndexedDB became a cache. The
owner filter survives for a narrower reason: a cache should not serve another user's data.

A failed fetch returns `null`, not `[]` — *"could not ask"* and *"you have none"* are different,
and rendering an empty history because the server blinked reads as data loss.

### Stage S9.5 Telling someone where they stand before it costs them a question

Everything above answers *"is this caller allowed?"* at the moment a request arrives, and answers
it well — the 401/403 split, the machine-readable `reason`, a refusal that names what is required.
But every one of those answers is a **reply to something the person already did**. In token mode a
signed-out visitor and a signed-in one rendered identically: an empty history, an open composer,
no indication either way. You discovered your access by spending a question on it.

So `/agent/whoami`, which already knew all of this, became the thing the page renders rather than
just a call it makes to learn an owner id. Three changes, each closing a different half of that:

**The scale gets names, server-side.** The platform's roles run backwards and are *sparse* —
1 is the most privileged, and 6, 7 and 9 are not roles at all. The client cannot derive that, so
`identity.ROLE_NAMES` lives next to the scale it names and the names travel with the numbers, on
`whoami` and on the `insufficient_role` refusal. A copy of the scale in TypeScript would be a copy
that stops matching the day the platform adds a tier; `authMessage` had already hardcoded
"contributor" for exactly that reason, and now reads the name off the response. An unnamed number
stays a number — rounding 6 to the nearest tier would print a privilege nobody granted.

**`whoami` stops being able to 500.** Two of the fields it reports come from configuration that
deliberately *refuses to guess*: an unrecognised `PLATFORM_TIER` raises rather than silently
picking a platform, and an unparseable `AGENT_MIN_ROLE` raises rather than widening access. Both
are right, and both turned the one endpoint whose whole job is to explain a refusal into a 500 in
precisely the situation it exists for. Each field is now read independently; a failure nulls that
field and becomes the `reason`.

**The badge.** A header control in token mode only — dev and demo identify nobody, and an account
control for an account that cannot exist is worse than none. Signed out it is a sign-in link, with
the URL carried on `whoami` itself so the state that most needs a link does not have to make a
second call for it. Signed in but under the threshold it says *No access* in warning amber, not
error red: the account is entirely valid, it simply is not permitted here, and the popover says so
along with what would be required and that signing in again will not change it. Signed in and
permitted it is quiet — the id, the role, the tier, and the one fact worth knowing about this
deployment, which is that your conversations and files belong to the account and follow it to
another browser.

Both header variants get it. `TopNav.platform.tsx` is kept as a verbatim copy of the pre-issue-#20
chrome, and its placeholder avatar stands down when there is a real account to show — the same
reasoning that gave that file a `demoMode` condition on the gear: the two headers must not
disagree about identity.

**Found by the badge, within minutes of the deployment reaching token mode:** `History (27)` sat
next to a `Sign in` button. The local owner filter read
`!viewer || !record.ownerId || record.ownerId === viewer`, and that first clause conflates two
meanings of `null` — *"this deployment identifies nobody"* in dev and demo, where nothing is ever
stamped, against *"nobody is signed in yet"* in token mode, where records **are** stamped. So on a
browser someone had used signed-in, opening the page signed out listed their conversation titles
back. The rule is now just the two halves that were always intended: an owned record is its
owner's alone, an unowned one is shared. Extracted as `visibleTo()` so it is a pure function with
a check rather than a clause buried in an IndexedDB callback, and applied to single-record reads
too — otherwise a stale id in React state reopens the last user's transcript.

Worth noting what this does *not* hide: records created before the switch carry no owner, so they
stay visible. They are unattributable by construction and they are this browser's own history;
hiding them would lose it for no gain. Verified live — 28 records in the store, 27 listed, the one
stamped with another owner absent.

**And then, signed in, the conversation list was empty for an account that had three.** Saving
worked, fetching one by id worked, every ownership check worked — only the *list* was blind, and
it failed silently, with a 200 and `[]`. `owner_id` is mapped `text` with a `keyword` subfield
(OpenSearch's default for a string), and `list_memories` ran `term` against the bare field. A term
query on an analysed field compares the whole value to individual **tokens**, and a platform id is
a URL: `http://cilogon.org/serverE/users/137206` is indexed as `http`, `cilogon.org`, `users`,
`137206`, none of which is the id. Everything else worked because everything else addresses
documents *by id* and never searches. The query now names `owner_id.keyword`.

The test that should have caught this is the more interesting half. `test_memory_ownership.py`
had covered the filter for months with a fake OpenSearch whose `search` compared the stored value
to the term — which is what real OpenSearch does for a `keyword` field and not for a `text` one.
The fake asserted the query's *shape*: that we filter by owner. It could not assert that the
filter matches anything. Both were then fixed: the fake models analysis, so a term query on the
bare field matches only a single-token value, and the fixtures stopped being `"alice"`. Every id
in those tests was one token, which is exactly why a bug about tokenisation could not show up in
them. Two new tests use the shape a platform account actually has; with the old query they fail,
which is the only evidence that a regression test is one.

**An hour in, the page presented a signed-in user as signed out.** The access cookie lives one
hour; the refresh cookie was still there and unused. Stage S9.4 gave every *request* path
refresh-and-retry, and it works — but it hangs off a **401**, and `/agent/whoami` answers **200
by design**, because it exists to explain a refusal rather than make one. So the one call that
decides whether you appear signed in at all was the one call that could never trigger a refresh:
the badge read `Sign in`, the history emptied, and nothing was wrong except an aged-out cookie.

The badge is what made this visible. The same expiry before it was silent — the page looked
normal until you sent something, and then `withTokenRetry` quietly fixed it.

Two halves. `whoami` gains `reasonCode`, drawn from the **same vocabulary as the refusals** and
derived from the same exception through a shared `_identity_reason()`, so a 401 on `/agent/chat`
and a `token_expired` from whoami cannot become two names for one fact. The prose `reason` stays
for a human reading a log; nothing matches on it. And `fetchWhoAmI` refreshes once on
`token_expired` and re-asks — one attempt, the same bound as every other retry, and only for
expiry: a forged token or an under-privileged account is not fixed by a new cookie.

### Stage S9.6 The history list, once it was finally readable

Fixing the owner query made the list render for the first time — and immediately produced two
more failures, both of which had been hiding behind an empty list.

**Rows that could not be opened.** Clicking a conversation did nothing: no navigation, no error,
*no network request*. `restoreSession` reads `tokenMode` to decide whether to fall back to the
server, and the callback was memoised as `[resolveUrl, fitView, pushMsg]` — without it.
`tokenMode` starts `false` and flips when `/agent/ui-config` lands, so the closure kept the
initial `false` forever and the server fallback never ran. Every server-listed conversation
failed on `if (!rec) return`. The neighbouring reads use `viewerRef` precisely to dodge this;
`tokenMode` was plain state and went stale. Two deps and a message on the failure path, because
"nothing happens on click" is the least debuggable way to express a real state.

**Rows that could not be opened for a second, unrelated reason.** The list and the detail read
different things: `list_memories` returns every memory this owner has, including ones the
*agent* created mid-turn, while `GET /agent/conversations/<id>` serves `session_snapshot` and
404s without one. Two of four live rows were `conversation-sess-...` entries with 0 messages that
404'd on open. The list now filters on `exists: session_snapshot`, so **listed means restorable**.
Rebuilding the client's view server-side would have been the other way to close it, and S9.3
already rejected that: `sessionStore.ts` is server-shaped on purpose, and a second implementation
of its layer-descriptor rules would drift.

**And the local cache, once more — twice.** In token mode with no viewer yet, `refreshSessions`
fell through to the local store: the same mistake `visibleTo()` closes, reached by a different
route. Fixing the branch was not enough, because the calls also **raced**. On load this runs once
before `/agent/ui-config` answers (`tokenMode` still false, so it reads IndexedDB) and again the
moment it does; the second call takes a synchronous path and finishes first, then the first
resolves its IndexedDB read and overwrites the correct answer with the stale one. The slow call
won. That is why a signed-in page settled on 27 local rows and why signing out left them there —
not a wrong branch, a lost race. Every write is now generation-guarded, which covers all three
callers rather than just the mount effect.

**The map came back empty.** With the list finally clickable, restoring the DEM conversation
brought back its transcript, its downloads and both layers — and drew the county boundary over a
blank basemap, with the raster missing. The stored descriptor was perfect (`kind: raster`, a URL
that serves 200 `image/png`, correct bounds), the file downloaded, the layer was listed in the
panel. It simply was not painted. `map.triggerRepaint()` on its own was enough to make it appear,
which is the whole diagnosis: a raster's image loads **asynchronously**, and in interleaved mode
nothing repaints when it lands. Live delivery hides this because the map is being fitted or
panned while the image loads, so a frame gets drawn anyway; a restore settles first and then
nothing ever asks for another frame. The fix repaints at the one moment that matters — preload
each raster image and `triggerRepaint()` on its `load` — rather than polling for it. One repaint
per image, nothing on a timer.

**The same stale closure a third time — and it was the cause of the orphans.** Reported as
"after I ask a question the wrong history and number show up": finishing a turn flipped the list
from the server's 2 conversations to 30 browser-local rows, including ones from before token mode
existed. `runLive` calls `snapshotSession()` in its `finally` and did not declare it, so it kept
the `snapshotSession` built on the FIRST render — the one that captured `tokenMode: false`. Every
turn therefore saved locally, **skipped the server PUT** because its `tokenMode` said this
deployment has no users, and refreshed the list through the local branch.

That also answers the item left open above. Conversations were not reaching the server on their
own at all — the Champaign snapshot only existed because it had been PUT by hand while
debugging — and the agent's own mid-turn memory was left snapshot-less, which is exactly what an
orphan `conversation-sess-...` document is. One omission, three symptoms that looked unrelated.

Three instances of one class in one file (`refreshSessions`/`viewer`,
`restoreSession`/`tokenMode`, `runLive`/`snapshotSession`), all sharing a shape: the value is
false or null on the first render and becomes real once `/agent/ui-config` and `/agent/whoami`
answer, so a callback memoised before that keeps the pre-identity world forever. None of the
symptoms pointed at a closure — they read as a routing bug, a dead button, a wrong list. So
`npm run check:hooks` now watches those six names and asks one question: if a callback reads one,
is it declared? Not a general exhaustive-deps implementation, and no new dependency — the same
shape as `check:auth` and `check:fold`. Verified the only way a regression check can be: with the
fix reverted it fails, with it applied it passes.

### Stage S9.7 Save-then-list, and a suite that was reading the wrong file

**The count sat one behind.** With conversations finally persisting on their own, the header
still read `History (2)` against a server holding 3 until something re-opened the panel.
OpenSearch is **near-real-time**: an indexed document is not searchable until the next refresh,
about a second, and `snapshotSession` saves and then re-lists within milliseconds. The write
landed; the search issued immediately after could not see it. `save_session_snapshot` now writes
with `refresh="wait_for"`, so the endpoint's contract is true — when the save returns, the
conversation is listable. It costs up to one refresh interval and is paid after the turn has
already been answered, off the streaming path.

The fake in `test_memory_ownership.py` learned to model this, the same way it learned to model
analysis earlier: a write is searchable only if it asked to be, or after an explicit `settle()`.
Remove the `wait_for` and four tests fail, including a save-then-list one that reproduces the
client's actual sequence.

**And the suite had stopped meaning anything.** Thirty tests failed and the run went from two and
a half minutes to twenty-five — with no relevant change in the repository. Four modules call a
bare `load_dotenv()`, which does not read "the repo's `.env`": it walks *upwards* from the
working directory. From a worktree under `.claude/worktrees/<name>/` that walk leaves the tree
and lands on the main checkout's file — the developer's own, tracking whatever was last deployed.
It had grown `AGENT_TOKEN_VERIFY=introspect`, `AGENT_MODE=token` and `PLATFORM_TIER=dev` during
the token-mode rollout, so every identity check became a real HTTPS call to the dev backend,
which answers 403 to an unauthenticated caller. The same checkout passed or failed depending on a
file outside it.

`rag_pipeline/tests/conftest.py` replaces the loader with a no-op before any test module imports.
Pinning the variables was the first attempt and it is not enough: `test_demo_mode` calls
`importlib.reload(api.server)`, and a test that had just `delenv`'d `AGENT_MODE` leaves the name
genuinely absent, so `override=False` stops protecting it and dotenv refills it. Any pinned value
loses to delete-then-reload; the mechanism had to go rather than its output.

Three live-backend tests were running against real OpenSearch and AnvilGPT purely because that
stray file configured them. They self-skip when unconfigured, so they now skip by default and
`RUN_LIVE_BACKEND_TESTS=1` opts back in — the same shape as the existing
`RUN_REAL_OPEN_GEODATA_TEST=1`. Offline and deterministic by default, live when asked for.
**1597 passed, 4 skipped, 86 seconds** — the twenty-five minutes was the network, not the work.

---

*Still open:* one conversation produced **two** memory documents — the agent's own
(`sess-60a7f5ca`, no snapshot) and the client's (`sess-c7d8b460`, snapshotted). The filter hides
the orphan rather than explaining it, and why the two ids diverge is not yet understood.

---

## Stage 10 — Removing the second path {#stage-10}

*Branch `claude/evidence-summary`. Removes what stages 2 and 3 left behind.*

Two orchestration paths existed from `665db95` (stage 5.4) onward: the supervisor, and the
agents-as-tools arm it replaced, kept behind `AGENT_SUPERVISOR=0` and a per-request
`useSupervisor: false`. A third remained inside the tool layer: `tool_strategy="full_pipeline"`,
a single `rag_tool` wrapping the whole stage-1 pipeline.

Both are now gone. The reason is that neither was a fallback any more.

### Stage S10.1 The measurement

`agent_runtime/legacy/` had not been touched since **2026-06-25** — while `agent_runtime/supervisor/`
was being changed the same week this was written. In that gap it acquired none of:

| | legacy arm | supervisor |
|---|---|---|
| `map_layer` delivery (stage 6.1) | ✗ | ✓ |
| the action ledger (stage 7.1) | ✗ | ✓ |
| terrain / DEM tools (stage 7.8) | ✗ | ✓ |
| the capability registry (stage 8.2) | ✗ | ✓ |
| the evidence summary (stage 8.1) | ✗ | ✓ |
| the grounding gate (stage 7.4) | ✗ | ✓ |

And it could not have caught up by accident. `collect_tools` in `tool_policy.py` binds six
factories — granular, file, MCP, quality, rag, skills. The terrain, rs-embed, overlay, aggregate,
temporal, spatial-stats and admin-boundary toolsets are **built directly in the supervisor's peer
builders**, so the legacy arm could not reach roughly half the current tool surface even in
principle. That is the same registered-≠-reachable split that made `list_conversation_files`
invisible in stage 7.9.

So `AGENT_SUPERVISOR=0` did not degrade the agent. It produced a **June 2026 agent**: no layers on
the map, no cross-turn tool memory, no elevation, no grounding gate. For a product whose defining
claim is that analyses land as map layers, that is not a fallback — it is a different and much
worse product, and it would have presented as catastrophic breakage rather than as a mode.

A flag that silently rewinds the system by three months is worse than no flag, because someone
eventually sets it while debugging something else.

### Stage S10.2 What was removed

- `agent_runtime/legacy/` — 792 lines across `orchestration.py`, `graph_nodes.py`, `builders.py`
  and `prompts.py`, including a second full prompt set.
- The `AGENT_SUPERVISOR` env switch and `is_supervisor_enabled()`.
- The `useSupervisor` / `use_supervisor` **request field** — an API change, noted below.
- `tool_strategy="full_pipeline"`, `agent_runtime/langchain_tool.py` and
  `make_langchain_rag_tool`. The granular tools are its superset, and a strategy that bypasses
  them also bypasses everything built on them since.
- `run_code_agent_query` and `agent_runtime/graph_nodes.py`. This one followed rather than being
  chosen: it *is* the agents-as-tools shape — a CodeAgent with SearchAgent bound as a tool — and
  its implementation lived in the deleted package. It was reachable only from a CLI flag.

### Stage S10.3 What was kept, and why

- **The strategy registry**, now with one entry. It is what made the two paths independent, and
  it is the seam a genuine second path would use again. Deleting the shape as well would save a
  file and cost the next fork.
- **`rag_pipeline.pipeline.run_pipeline` and `POST /query`.** These share a name with the removed
  `full_pipeline` strategy and are a different thing: a separate HTTP product surface from stage
  1, still served by `api/server.py`, not an agent arm. Removing an endpoint the platform may
  call is not this change's business.

### Stage S10.4 What happened to the tests

Deleting a path deletes the tests that pin it, which is where the real care was needed:

- **Deleted**, subject gone: the `AGENT_SUPERVISOR` flag-parsing test, the per-request
  `use_supervisor=False` override test, and two tests of `make_search_agent_evidence_tool`.
- **Replaced**: those two covered dedup and failure handling. Dedup is already covered on the
  supervisor arm; failure handling was **not**, so it became a new test — a throwing search peer
  must cost its own result, not the whole turn including evidence already gathered.
- **Repointed**: the response and stream contract tests stubbed the legacy arm and pinned
  `AGENT_SUPERVISOR=0`. They now stub `run_supervisor` — deliberately **below**
  `run_supervisor_orchestration`, because that wrapper emits the orchestrate node lifecycle pair
  one of them asserts on, and stubbing in its place would have quietly deleted the thing under
  test. The contract itself did not change, which is the point: the shape a client sees should
  survive the graph behind it being replaced.
- **Rewritten**: a test asserting the legacy arm keeps the string "Orchestrator agent started"
  *"because there it is accurate"* now asserts no module claims it at all. It scans string
  constants through the AST rather than grepping text — the comment in
  `supervisor/orchestration.py` recording why the string was renamed contains the phrase, and a
  text-shaped test fails on the explanation for its own existence.

### Stage S10.5 The API change

`useSupervisor` was a documented field on `POST /agent/chat` and `/agent/chat/stream`. It is gone
from the request contract and the Swagger docs. A caller still sending it is now ignored rather
than honoured — which is the quiet direction to fail, and worth knowing if anything outside this
repo sends it.

`tool_strategy` still exists and still accepts `"granular"`. `"full_pipeline"` now **raises**
naming what changed, rather than silently resolving to granular: a stale caller should find out.

### Stage S10.6 What is not recorded

`3b7e181` built per-request architecture switching *"because otherwise the two architectures could
only be compared by restarting the deployment between arms"*, and the same mechanism existed here.
**No commit records a measured supervisor-vs-legacy comparison.** If one was run outside the repo,
its result is gone — which is the argument of this document, applied to its own final stage.


## What is deployed

| | state |
|---|---|
| `prototype` through `9e35950` | live; the base both branches were cut from |
| `claude/evidence-summary` through `8fe0dcb` | live on the dev VM |
| `claude/evidence-summary` `25c3e1e`, `aaac2c3` | **not deployed** |
| `claude/jwt-identity` (all 10) | **not deployed**; token mode has never run outside a test |

Neither branch is merged, and the VM can only run one at a time.

---

## Known gaps in this record

- **Reasons that were never written down are gone.** Stages 0–2 have many: the deletion of the
  declarative router, the orphaning of `search_agents.py`, the prompt downgrade at `0e0435d`, the
  `top_k` 8→100 change, why branch B was started or discarded. Reading the diff does not recover
  them. This is the whole argument for the rule below.
- **Part of the system is not in this repo.** The rs-embed service's webapp half was vendored at
  `096efbf` because upstream `.gitignore`s `examples/**` — *"It existed only on the VM."* Other
  commits flag further service-side halves still untracked.
- **Two mislabelled commits.** `91b6f4b` ("Add docker-out-of-docker") contains only file renames;
  the actual work is in `af1ead0`. `8dc7f25` ("Add prototype MCP") contains no MCP change.
- **`analyze` vs `code` is unsettled.** They share 14 of 18 toolsets including the sandbox, so the
  toolset does not distinguish them; what does is that `AGENT_CODE_PEER` can swap the code peer
  for an external CLI runtime. Measured: *"7 of 7 turns reported peer=analysis"*. No commit
  resolves whether the split should remain.
- **A capability taxonomy was added and never consumed** (`722e4ed`, stage 4.4).
- `175fed8` ("Lower-case the rs-embed demo tab") has **no reason recorded at all**.

---

## Adding to this document

**Every architectural change is documented here with its reason, in the same commit that makes
it.** This is a project rule, recorded in `AGENTS.md`. It exists because the alternative was
tried: this document was reconstructed by five parallel agents reading 408 commits, and they could
recover only the reasons somebody had happened to write down.

A **stage** is a coherent shift, not a commit. Thirty commits and one structural change is one
entry. A new change either extends the last stage or opens a new one — it should not require
restructuring what is already here.

An entry needs:

- **The reason, and the measurement with it.** *"266 s and 16 iterations"*, *"141 parameters
  across 62 of 80 tools"*, *"four calls before one worked"*. A reason without its number is an
  opinion. Every strong entry above has a number; every weak one does not.
- **Why the first attempt was wrong**, when a change revises an earlier one. Two examples are
  marked *"Revised during the work"* above, and in both the correction is more instructive than
  the change.
- **"Reason not recorded"** where it genuinely is not. An honest gap beats a plausible
  reconstruction, which is indistinguishable from fact once written down.
- **Prompt revisions and tool signatures count as architectural.** The supervisor's capability
  paragraph drifted behind its peers until a DEM request became a knowledge-base search. A prompt
  is part of the architecture, not commentary on it.

The same drift is what `docs/spatial-toolkit.html` (the capability atlas) exists to prevent, and
it has fallen behind twice.

---

## Stage 11 — Where state lives, and who decides {#stage-11}

Stage 9 gave the agent an identity. This stage is about everything that identity writes to, and
it began because a feature would not start: `chat_traces` (the raw-trace store, S9.6) could not
allocate a shard. Chasing that found the agent talking to a cluster the rest of the platform had
left behind.

### Stage S11.1 The tier owns the cluster, the credential and the index

`PLATFORM_TIER` already existed to stop half-switched states — *"two pointing at dev, one left on
prod, and a verification that fails for a reason nobody can see"*. It named the frontend and
backend. It did not name the **search cluster**, so when dev's OpenSearch moved hosts,
`OPENSEARCH_NODE` stayed pinned to the old one. Nothing failed: the old host kept answering and
kept accepting writes, and the agent went on reading and writing a machine nobody maintained.

The cluster is now the fourth thing a tier names, and three refinements followed from using it:

* **The credential follows the host, not the tier.** Found by testing what a *restart* would do,
  not by a test failing: during the migration `OPENSEARCH_NODE` pinned the old host while the tier
  supplied the new host's credential, leaving the deployment one `docker compose up` from a 401
  that would have stopped conversations saving. A credential must never be paired with a host it
  does not belong to.
* **Any setting can be tiered** (`tiered_env`), because dev and prod disagree about index names
  too. Precedence here is the *reverse* of the URL rule, deliberately: for a URL the tier supplies
  a value and the explicit variable overrides it, but for a credential or an index the tier
  supplies no value at all — secrets and deployment names are not in this repository — so the
  tiered name is simply the more specific one.
* **`SEARCH_TIER` splits from `PLATFORM_TIER`**, because which platform mints your tokens and
  which corpus you search are different questions. Running the dev platform against the prod
  knowledge base is ordinary, and it used to mean editing index names by hand.

Disagreement is now said out loud at boot rather than obeyed in silence — the failure mode was
never the wrong value, it was the wrong value applied quietly.

### Stage S11.2 A write path that failed silently for four days

On 2026-09-22 the old cluster crossed the 95% flood-stage watermark and OpenSearch set
`read_only_allow_delete` on `chat_memory`. **Reads kept working.** Turns streamed complete
answers, the history list rendered the old conversations, and every new one was lost. The only
trace was one `429 cluster_block_exception` per turn, caught on purpose so that a storage failure
never costs someone their answer.

That deliberate catch is right, and it is also what made this invisible. Four days of
conversations went missing before anyone asked a question that happened to surface it.

The fix was the migration the tier work had already made possible: 1,274 documents copied to the
new cluster with ids preserved (`memoryId` *is* the `_id`), verified for the `owner_id.keyword`
subfield the list query depends on, then `OPENSEARCH_NODE` removed so the tier supplies the host.
Copy first, switch second — the destination's `chat_memory` was empty, and flipping first would
have emptied the history UI.

**The lesson is not about disk.** A write path that fails while the read path succeeds is
indistinguishable from a working system from the outside: no error reached a user, a log anyone
watched, or the health check. If a storage failure is survivable by design, it has to be *visible*
by design too — a `warning` event to the client, or a health check that attempts a write, would
have turned four silent days into a first-turn complaint.

### Stage S11.3 Two more places a hidden control left a stale value

* **The model.** Demo mode had always *forced* its model, with the reason written down: the
  settings panel is the only control that can change it, demo hides it, so a value left in a
  returning visitor's `localStorage` pins them to a model they can neither see nor change. Token
  mode hides the same panel and had no such guard — so browsers kept asking for `gpt-oss:120b`
  after the deployment moved to `gpt-5.6-luna`. The condition is now *the panel is hidden*, not
  *this is demo*. The two still differ in what replaces the choice: demo names a model, token mode
  passes `None` so the deployment's own default applies.
* **Sign-in.** Someone sent from the agent to sign in landed on the platform's `/user-profile`,
  having lost what they were doing. The platform's `/auth/login` already accepted
  `redirect-domain-id` and `redirect-path`; the id is a key into the frontend's
  `redirect-whitelist.json`, not a URL, so only hosts that file names can be targets. Off until
  `PLATFORM_REDIRECT_DOMAIN_ID` is set, and an unrecognised id degrades at the far end to exactly
  the old behaviour rather than breaking sign-in.

### Stage S11.4 What this stage did not fix

The agent hung for three days and nothing noticed. It stayed `Up`, so `restart: unless-stopped`
never fired; gunicorn's `--timeout 600` does not kill a threaded worker whose threads are stuck in
I/O; and the health check failed **2,092 consecutive times** while taking no action. One worker
with four threads means four hung calls is the whole service.

The LLM client still sends no request timeout, which is the most likely way those threads were
consumed — though that remains unproven, because the container was force-recreated before its logs
were captured. Both the timeout and something that acts on a failing health check are open.

---

## Stage 12 — Staying up, and keeping the evidence {#stage-12}

Stage S11.4 ended with two things open: an LLM request timeout, and *something that acts on a
failing health check*. This stage closes the second, and closes the reason the first is still
unproven — the evidence was thrown away.

### Stage S12.1 The signal existed; nothing was listening

The three-day hang was not a monitoring gap. Docker knew: the container reported
`Up 3 days (unhealthy)` with a failing streak of 2,092. Every layer that could have acted had a
reason not to.

* `restart: unless-stopped` restarts a container that **exits**. A wedged container never exits.
* gunicorn's `--timeout 600` kills a worker stuck in a **request**, not one whose threads are
  stuck in I/O. With `WEB_CONCURRENCY=1 --threads 4`, four hung calls are the whole service.
* Docker records health status and takes no action on it. That is deliberate; acting is
  somebody else's job, and nobody was doing it.

`deploy/agent-watchdog.sh` is that somebody: a systemd timer on the host, every minute, reading
`.State.Health` for each watched container. Host-side rather than a sidecar because the obvious
alternative — an autoheal container — wants the Docker socket, and adding a second image with
root-equivalent control of the daemon to a public repo buys nothing a 200-line shell script
does not.

Two numbers carry the design:

* **Ten minutes of continuous failure before acting.** A streaming turn can hold a worker thread
  for minutes, so with four threads a genuinely healthy agent under load can look unresponsive.
  Restarting then would kill live turns to fix nothing. Ten minutes is pathological; two is a
  busy afternoon. The threshold is computed from the container's *own* configured interval
  (read as `{{json .Config.Healthcheck.Interval}}` — the plain form renders a Go duration as
  `"30s"`, which is not arithmetic, and the first version of this script died on exactly that),
  so changing the interval in compose does not silently change the threshold.
* **Three restarts per hour, then stop.** If restarting did not fix it, restarting again will
  not either. Past the budget the watchdog refuses and says so loudly, leaving the service down
  for a human. A service that is down and loud beats one that is restarting every ten minutes
  and silent.

It deliberately does **not** restart on a dependency outage. The container health check is a
liveness probe — "can this process still answer?" — and that is the only question whose answer
is "restart me". If OpenSearch or the LLM upstream is down the agent still answers `/health`,
and the watchdog stays out of the way, because a restart loop against somebody else's outage is
worse than the outage.

### Stage S12.2 Capture before restart

The rule that shapes the script, and the one that came from getting it wrong: **restarting a
hung service destroys the only copy of why it hung.**

When the hang was found, the recovery was `docker compose up -d --force-recreate`. That deleted
the container, and its logs went with it. Nobody ever ran `docker logs agent-api > hang.log`. So
the most likely cause — an LLM call with no timeout consuming all four threads — is still a
hypothesis rather than a finding, and will stay one.

The watchdog therefore captures first and restarts second, into
`/var/log/iguide-agent/incidents/<utc>-<container>-<reason>/`: full `inspect`, the health log,
20,000 lines of container output, the host-side process table, `docker stats`, host disk and
memory, the kernel ring buffer, and six hours of this container's journal. Then it restarts —
with `docker restart`, never `up --force-recreate`, because the first keeps the container and
its log history and the second is precisely what destroyed them.

`py-spy` is installed in the image for the one artefact that says *where* it is stuck rather
than *that* it is: a Python stack for every thread, read from outside the process. That matters
specifically here, because a process wedged holding the GIL cannot run its own signal handlers,
so it cannot be asked to dump its own state — it has to be read. `cap_add: [SYS_PTRACE]` makes
that deterministic, which is negligible beside the Docker socket the container already mounts.
When py-spy is absent the bundle records that fact instead of failing, so the watchdog works
against an image that predates it.

### Stage S12.3 Logs that outlive the container

Container logs were on the default `json-file` driver, which stores them inside the container's
own directory. `docker rm` deletes them — and every deploy runs `up -d --build`, which recreates.
So the system's log history was, structurally, never older than the last deploy.

All four services now use the `journald` driver. The host journal is persistent here
(`/var/log/journal` exists), independent of container lifetime, and indexed by container name:

```
journalctl CONTAINER_NAME=agent-api --since "2 days ago"
```

`docker logs` still works. Two settings in `deploy/install-watchdog.sh` make it trustworthy
rather than nominal. `Storage=persistent` is set explicitly, because the default `auto` keeps
logs only if `/var/log/journal` already exists — a reinstalled host would silently fall back to
a memory-only journal that dies on reboot. And `LogRateLimitIntervalSec=0` on `docker.service`
removes journald's per-unit rate limit, which would otherwise apply to every container at once:
a failing service is exactly when logging bursts, and exactly when dropped lines cost most.
Volume is bounded by size instead — `SystemMaxUse=3G`, `SystemKeepFree=5G` — which is the right
axis, and keeps container logging from growing into the kind of disk pressure that took an
OpenSearch cluster read-only at 95%.

### Stage S12.4 The health check now reads the answer

Every health check was `requests.get(url, timeout=5)` with the result discarded. `requests` does
not raise on a 500, so a service answering nothing but errors passed. The probe detected total
unresponsiveness and nothing else. It now calls `.raise_for_status()`. (`mcp-server` used
`urllib.request.urlopen`, which already raises on non-2xx, and needed no change.)

### Stage S12.5 In the same change, unrelated: the badge, and why prod is still not an option

The header's role chip is gone. It was the one account state with nothing to do about it — the
person already knows who they are — while putting their role on screen permanently, in every
screen share and screenshot. `AccountBadge` now renders only when the state needs action: *Sign
in* when signed out, *No access* when signed in and refused, nothing at all when working.
`accountNeedsAttention()` is exported so the platform-variant header can restore its decorative
avatar in that empty space rather than losing it.

Switching the deployment to `PLATFORM_TIER=prod` was investigated and **rejected**, with the
measurements recorded in the tier table itself. Both halves fail, and both fail quietly:

* **Memory.** Prod's cluster (149.165.155.195) is at 95.5% disk, past the flood-stage watermark,
  with 914 indices carrying `read_only_allow_delete`; an index creation times out. Reads still
  succeed, so search would keep working while every conversation silently failed to save. Only
  3.7 GB of that 55.3 GB is OpenSearch — the rest is something else on the box, so it is not a
  problem this repository can fix by deleting indices.
* **Token.** Only partly, and the halves are easy to get backwards. Identity *verification* is
  server-to-server, so CORS never applies and prod would verify fine. The browser's *refresh* is
  blocked: prod answers a preflight from `agent.i-guide.io` with
  `Access-Control-Allow-Origin: https://platform.i-guide.io`. Sign-in would succeed and the
  session would die five minutes later at the first refusal — an expiry, not an error.

Prod's OpenSearch host is deliberately still empty in `_TIERS` rather than filled in with the
now-known address, so `PLATFORM_TIER=prod` fails loudly and demands an explicit
`OPENSEARCH_NODE` instead of quietly writing nowhere.

### Stage S12.6 What this stage did not fix

The LLM client still has no request timeout, so the failure this watchdog now recovers from can
still happen — recovery in ten minutes instead of three days is the improvement, not prevention.
The next hang will produce a thread-stack bundle, which is what the timeout work needs to stop
being guesswork.

`opensearch_credentials()` has a live trap that was found but not fixed, because nothing is
currently on that path: an explicit `OPENSEARCH_NODE` that disagrees with the tier falls back to
the **untiered** credential, and that pair returns 401 against the dev cluster. Anyone pinning a
node across tiers gets a cluster that authenticates for reads at boot and stops saving
conversations. The credential should select by *which tier owns the host being used*, not by
"does this host match my tier, yes or no".

### Stage S12.7 What the tier owns, and what quietly did not move

The switch to prod exposed one fault three times in an afternoon: **a value the platform sets
per tier, stored anywhere other than the tier table, does not move when the tier does.** Each
instance failed silently, and each looked like a different bug.

| value | where it lived | what the switch did |
| --- | --- | --- |
| OpenSearch host | table, but PROD's entry empty | refused to start — the only one that failed LOUDLY, by design |
| access-cookie name | `JWT_ACCESS_TOKEN_NAME` in `.env` | kept reading a cookie prod never sets; every signed-in visitor told to sign in |
| redirect-domain-id | `PLATFORM_REDIRECT_DOMAIN_ID` in `.env` | carried dev's `006` to prod, which numbers this agent `003`; sign-in stopped returning here |

The cookie name compounded it: prod is **not** the suffix-less form. The platform suffixes both
tiers — a browser signed in to prod holds `jwt-access-token-prod` beside dev's
`jwt-access-token-dev` — and `consistency_warning()` missed it because it only asked whether the
name ended in `-dev`. A check that recognises one specific wrong answer certifies every other
wrong answer as correct; it now requires the tier's own suffix, so it is closed rather than open.

The redirect id is now in `_TIERS` alongside the backend, frontend and cluster, with
`PLATFORM_REDIRECT_DOMAIN_ID` still winning when explicitly set. Its failure mode is the reason
it had to move rather than just be corrected: an unknown id is **not** an error at the far end —
the frontend logs it and falls back to the profile page — so sign-in keeps working and merely
stops coming back, and the only signal is a person noticing they landed somewhere else.

The general rule this leaves: when adding anything the platform assigns per tier, put it in
`_TIERS` and give it a resolver with the same precedence as the rest (explicit env wins, table
is the default). An env var holding a per-tier value is a switch someone has to remember, and
the evidence of one afternoon is that they will not.

Still true and not fixed by any of this: prod's OpenSearch host is deliberately absent from the
table, so this deployment names its cluster in `OPENSEARCH_NODE`. Filling it in needs the
credential-selection fix in S12.6 first.

### Stage S12.8 The fifth value kept outside the tier rule: the agent search client's credential

S12.7's rule is that *a value the platform sets per tier, stored anywhere other than the tier
table, does not move when the tier does*. S12.7 lists three instances. The fourth is on the
extraction path, `agent_kb._os_client`, which `claude/extraction-integration` moves. The fifth
was found on 2026-10-01 at `5ae6d92`, in the deployed container's copy as well as the
repository's, and it is in the platform search itself: `rag_pipeline/search/agents.py`'s
`_os_client` resolved `OPENSEARCH_NODE` through the tier rule and `OPENSEARCH_USERNAME` /
`OPENSEARCH_PASSWORD` bare. **A tiered host with the untiered credential.** The three clients
beside it (`keyword.py`, `semantic.py`, `spatial.py`) resolve all three names through
`search/utils.getenv`, `<NAME>_<SEARCH_TIER>` first, and `agents.py`'s own `_getenv` already
delegated to that helper. It was used for one of the three names.

Why nothing failed: the bare pair **is** prod's. Reproduced inside the deployed container,
`SEARCH_TIER=prod` resolves the node to prod's cluster (`149.165.155.195`) and the bare pair
authenticates there. The same pair returned 401 against the dev cluster on 2026-09-22, so
`SEARCH_TIER=dev` would send prod's password to dev's cluster, and the agent search would fail
with a 401 that reads as a network problem. It is the trap S12.6 records for
`opensearch_credentials()`, reached by another path. Latent, not live: the deployment searches
prod.

The fix is the two lines `keyword.py` already had. `claude/extraction-integration` makes the
same change to `agents.py`, byte for byte including its comment, so the two branches merge
cleanly there. No resolver was added. That branch introduces `platform_endpoints.search_cluster()`,
returning `(node, user, pwd)` under a credential-follows-host rule, and a second resolver here
would be a second answer to the same question for it to unpick. When it lands, the four search
clients (`agents.py`, `keyword.py`, `semantic.py`, `spatial.py`) share one shape and can move
onto it together.

Pinned by `rag_pipeline/tests/test_search_tier_credential.py` (seven tests; the suite goes from
1645 to 1652 passed, 4 skipped before and after):

- The client built under `SEARCH_TIER=dev`, with the `_DEV` triple beside a different bare
  triple, carries the DEV pair. It is recorded where `agents.py` binds `OpenSearch`, and checked
  once more through the `Authorization` header the real client would send. Three more
  configurations pin the precedence: bare names alone, `SEARCH_TIER=dev` under
  `PLATFORM_TIER=prod` (the pair follows the SEARCH tier), and prod beside the bare triple.
  Assertions name hosts and pairs by where they came from (dev, prod, bare), never by value.
- The drift guard: no module under `rag_pipeline/search/` or `extractors/emitters/` reads the
  three names bare (`os.getenv`, `os.environ`). It carries two named carve-outs, `agent_kb.py`
  and `opensearch_emitter.py`, which `claude/extraction-integration` moves onto
  `search_cluster()`. Each carve-out expires itself: the test fails the moment its file stops
  reading the bare names, so the list cannot go stale.
- `@lru_cache(maxsize=1)` is on `_os_client` and not on `_os_index`, the settings helper beside
  it. On the sibling branch an anchored edit here moved the decorator onto a settings helper,
  caching a setting; targeted runs passed and the full suite caught it. Pinned so the next edit
  near it fails by name.

Not changed by this: with a tiered node and **no** tiered pair, `tiered_env` still falls back to
the bare pair, per variable. All four search clients share that rule now. `search_cluster()`'s
credential-follows-host rule is its replacement once they call it, and that is the sibling
branch's change, so it is not asserted here either way.

### Stage S12.9 What the image carries, and what quietly did not ship

S12.7's fault has a packaging twin: **a directory the runtime reads, named anywhere other than
the Dockerfile's copy list, does not reach the container.** `agent_runtime/skills.py` discovers
skill bundles under `REPO_ROOT/skills` and `REPO_ROOT/.agents/skills`, and `REPO_ROOT` is `/app`
in the image. `rag_pipeline/Dockerfile` builds `/app` from an explicit list of `COPY`
instructions, one package added each time one was needed, and no version of it on `prototype`
has copied either skill root since skills landed on 2026-05-06 (`d51cd25`): 148 days. Discovery
skips a root that does not exist, by design, so the registry degraded to empty and nothing said
so.

| image | `/app/skills` | `/app/.agents/skills` | `SkillRegistry.discover()` |
| --- | --- | --- | --- |
| deployed `agent-api`, measured 2026-10-01 | absent | absent | 0 skills |
| `prototype` @ `5ae6d92`, built locally | absent | absent | 0 skills, 0 errors |
| the same commit with this change | 2 skills | 1 skill | 3 skills, 0 errors |

The deployed agents did not have skill tools that returned nothing; they had no skill tools.
`make_skill_tools()` returns an empty list for an empty registry, so `list_available_skills` and
`load_skill` were never offered, and `available_skills` was `[]` in every response and in the
stream's `initialized` event. The image now copies both roots, and all three skills load in it as
the image's non-root user.

**The first fix was partial.** `f8f99ef` on `backend_swap` (2026-08-13) copied `.agents/` alone:
one skill of three, with `/app/skills`, the first root, still missing. The test it added passed,
because it asked whether *any* root was copied. That is S12.7's lesson again: a check that
recognises one wrong answer (no root) certifies another (one root of two). The commit is still on
`backend_swap` and in `claude/extraction-integration`'s merge `b911191`; whichever branch meets
this change should keep both `COPY` lines and drop that test.

The guard is `rag_pipeline/tests/test_image_skill_roots.py`. It builds the part of `/app` that
holds the skill roots in a temporary directory by the Dockerfile's own rules (`WORKDIR`, each
`COPY` of the last stage, then `.dockerignore`), plants a skill in each root, and runs the real
discovery with no explicit roots, as every production caller does. Eleven cases check that it
rejects the ways a root can be lost, the partial fix among them. It reads text, so it does not
replace building the image, but it fails on a checkout before anyone builds one.

`.dockerignore` was checked, not assumed. Its `*.md` reads as though it drops every `SKILL.md`,
but Docker anchors patterns at the context root, so it drops only the root's own markdown. The
test's matcher was cross-checked against a real build of this context: of 390 paths on disk it
predicted exactly the 365 that Docker sent.

`AGENT_SKILL_PATHS` appends roots after the defaults; it does not replace them. A deployment that
points it at a writable volume for generated skills (`backend_swap` uses
`/app/agent_chat_files/skills`) keeps the baked-in ones: the built image found four skills with a
third root mounted, and a test pins the order.

Two of the three skills deserve a look before this is deployed, because deploying it is what
puts them in front of the model. `example-skill` is a stub whose description says it exists to
verify skill loading. `ai-agent-for-chicago-crime-analysis` names `mcp_run_nbwf_d01e717421c1b0ff`
as its tool, and no such tool exists on `prototype`. `allowed-tools` is advisory, so loading that
skill tells the agent to call a tool it does not have; nothing blocks it.

The rule this leaves, beside S12.7's: **anything the runtime reads relative to `REPO_ROOT` is a
deployment input, and the copy list is where it has to be named.** The other repository-root
paths in the copied packages are import paths, `.env` files the image leaves out on purpose, a
boundary for user-supplied notebook paths, and the skill emitter's output directory. One is a
real gap and is not fixed here: `/agent/dashboard` serves `examples/agent_chat_stream_demo.html`,
and `examples/` is not in the image either.

### Stage S12.10 The page `/agent/dashboard` serves

The skill-roots fix (S12.8 in I-GUIDE/iguide-ai#31, open at the time of writing) ended by naming
one more gap of its class: `/agent/dashboard` serves `examples/agent_chat_stream_demo.html`,
which `api/server.py` reads relative to the repository root, and no revision of
`rag_pipeline/Dockerfile` on any branch has copied `examples/`. The Dockerfile has built `/app`
from an explicit list since it was created (`1c78e59`, 2025-12-14), and the route landed into it
(`7f71a90`, 2026-05-07). Every image built in the 148 days since answers the route with a 500.

| where | `GET /agent/dashboard`, 2026-10-02 |
| --- | --- |
| `https://agent.i-guide.io` (production) | 500, Flask's default 265-byte page; `/health` 200 at the same time |
| `prototype` @ `5ae6d92`, built locally, under the image's own gunicorn `CMD` | 500; one ERROR traceback per request, `FileNotFoundError: ... '/app/examples/agent_chat_stream_demo.html'` |
| the same commit with this change | 200, `text/html`, byte-identical to the checkout's file |

**Revised during the work: the first decision was the opposite.** The task allowed either
answer, copy the page in or make the route answer 404 cleanly, and `prototype` on its own
pointed to the second. README.md, AGENTS.md and `docs/` never mention the route. Its docstring
calls it "the local streaming agent dashboard". And `CORS(app)` lets the page reach the live
agent from `file://` anyway: a preflight from `Origin: null` was allowed, `x-api-key` included.
A 404 version was written and tested, and the suite passed with it (1647 passed). It was
withdrawn because `backend_swap`, read afterwards, records the opposite intent:

- `8fd97f58` (2026-08-12) closed two holes it described as "reachable from the open internet"
  and kept `/agent/dashboard` open on purpose, as "static HTML with no data".
- `66e71d54` (2026-08-12) fixed hardcoded tool checkboxes in the page that an earlier fix had
  corrected only in the prototype: "fixing one shipped client had left the other broken".
- M0.2b (2026-08-07 in that branch's `docs/DEVLOG.md`) updated it, with the prototype, to
  describe the new auth contract.

That branch also scopes CORS to `AGENT_CORS_ORIGINS`, falling back to `ALLOWED_DOMAIN_LIST`. A
page opened from disk sends `Origin: null`, so unless `null` is on that list, being served by the
agent is the only way the page works against a deployment.
"Nobody noticed the 500 for 148 days" is no evidence either way: the skill roots went unnoticed
for the same 148 days and were plainly meant to ship. `git log -S` on one branch found where the
route came from. The reasons were in another branch's commit messages.

**The change** is one `COPY`, of this file only, placed after `api/`:
`COPY examples/agent_chat_stream_demo.html ./examples/`. Nothing else in the copied packages
reads the root `examples/`, and its other page, `iguide_chat_prototype.html`, is only named in a
docstring. `.dockerignore` does not exclude it: its patterns are anchored at the context root and
none reaches `examples/`, and the image builds with it. The route's path moved into a module
constant, `_DASHBOARD_PAGE`, unchanged in value, so a test reads the path from the code instead
of restating it.

**The guard** is `rag_pipeline/tests/test_image_dashboard_page.py`. It follows the page through
the Dockerfile's own rules (`WORKDIR`, each `COPY` of the last stage, then `.dockerignore`) and
checks that it lands beside wherever `api/` lands, which is where the route reads it. The
readers are #31's, copied byte for byte with one addition. A directory named as a `COPY` source
always has its contents land inside the destination. A single file lands *inside* a destination
that ends in `/`, and *as* any other destination. Fourteen cases check the audit's verdicts, and
all fourteen were also built for real. Every accepted Dockerfile served the page. Five of the six
rejected ones built without an error and left a route that would answer 500. The sixth, an
ignore pattern that drops a file a `COPY` names, failed the build. The subtlest is a missing
trailing slash: `COPY … ./examples` builds cleanly and writes the page as a *file* named
`/app/examples`. Removing the new line fails the guard with "no COPY in the Dockerfile carries
it". Moving it to `./` fails with both paths named.

Not fixed here:

- **Production still answers 500** until the image is rebuilt with this change, and deploying
  is a separate decision. Once it is deployed, the page is public at
  `https://agent.i-guide.io/agent/dashboard`, as `backend_swap` decided it should be.
- On `backend_swap`, `test_api_auth.py`'s docstring says `/agent/dashboard` is asserted to stay
  open, but its `OPEN` list holds only `/health`. That assertion was never written, so nothing
  there notices the route being closed or broken. Add it when that branch meets this one.
- Once #31 merges, the Dockerfile and `.dockerignore` readers exist in two test files. They
  belong in one module, and this file's `_lands_at` is the superset.

S12.8's rule held: **anything the runtime reads relative to `REPO_ROOT` is a deployment input.**
The one path that looked like an exception was not one. What showed that was another branch's
commit messages, not this branch's code or docs.

---

## Stage 13 — Shapes nobody owned {#stage-13}

Four fixes from 2026-10-01 to 10-03 (#28, #29 and two operational changes) share one cause: an
interface with no owner. One was a payload shape that the server emits and the client reads, with
no code standing between them. One was a model id that only a remote catalogue decides. One was
an initialisation path written twice. One was the moment a config change becomes an outage. Each
broke silently, and in each the visible symptom pointed somewhere other than the cause.

> **On numbering.** This is Stage 13 on `prototype` because it merged first. At the time of
> writing, nine open PRs also call themselves Stage 13 (#32, #34, #36, #37, #38, #40, #44, #45,
> #46), several with different content under the same `{#stage-13}` anchor, and #30 and #31 both
> claim S12.8. Whoever lands them must renumber; merged order is the honest sequence.

### Stage S13.1 The download panel is whatever the tools emit

The Downloads panel showed `unnamed file (file_f2929f3dec7d)` three times. The files downloaded,
so it read as cosmetic. It was not.

Nothing server-side assembles that panel. `collectDownloads` (`map-ui-prototype/src/agentClient.ts`)
walks the entire SSE payload and harvests any object holding a `download_url` and a `file_id` or
`filename`, on the assumption that the three describe one file. The shape a tool emits is
therefore the whole contract, and no code sits between the two ends to enforce it.

`align_embedding_colors` broke it in both directions at once. Each region entry carried the
embedding **package**'s `file_id` (an `.npz`) next to the re-coloured **image**'s `download_url`
(a `.png`), and no filename. Each row was labelled from one file and linked to the other. It
appeared to work only because the click still downloaded *something*. `embed_zones`'
`pixel_image` had the milder half: its id and url agreed, but it had no name.

The store had the names all along (`drawn_region_2018_gse_2018-06_2018-09_vectors.npz` and its
siblings). That was established from evidence, not by reading code. `chat_traces` showed that
turn calling `align_embedding_colors` on exactly those three ids, and the store's metadata held
intact names for each. This was the first time the trace store answered a production question.

Each file now travels as its own complete object. The second file **nests** instead of sitting
under sibling `image_*` keys, because the flat shape is what allowed one file's id to be paired
with another's url. `test_download_descriptors.py` mirrors the client's harvesting rule in
Python, since a test checking any other rule would pass while the panel stayed wrong. It also
includes the shape that shipped, to prove the check rejects it.

### Stage S13.2 A model replaced under us, and a test that asserted a fact about the world

Purdue withdrew `qwen3.6:27b` and put `qwen3.8:27b` in its place. Unlike its predecessor, which
was listed and "Recommended" while returning zero bytes in 90 s, the replacement is usable as an
agent model. Verified through `build_llm` rather than raw HTTP:

- it answers in 0.7 s
- it emits correct `tool_calls` with `finish_reason=tool_calls`
- it keeps `reasoning_content` through `ReasoningPreservingChatOpenAI`
- it completes a tool-result round trip in 2.2 s

It reasons harder than 3.6, spending 34 of 38 completion tokens to say "OK", so the deliberate
absence of `max_tokens` matters more with it, not less.

Both hardcoded fallback defaults moved, along with `_ANVIL_FALLBACK_MODELS`. The second default,
in the per-request path of `build_llm`, was nearly missed. Measured claims keep the model they
were measured on: the `max_tokens` numbers still name qwen3.6, with 3.8's own measurement added
beside them.

The instructive part was a test. `test_the_model_defaults_to_the_verified_id` asserted that
"qwen3.6:27b is the id AnvilGPT actually serves". No local test talks to Purdue, so that claim
could not fail when it went stale, and it did go stale. A unit test can pin a **shape** (Open
WebUI `name:tag`, never HuggingFace `Qwen/...`). It cannot pin a fact owned by a remote service.
The test now pins the shape and states that the roster is probed. `/agent/models` fetches it
live, so the picker corrected itself; the fallback tuple is consulted only when that fetch fails.

qwen3.8 was the deployment default from 10-01 until 10-02, when the VM returned to OpenAI
`gpt-5.6-luna`. It remains selectable per request.

### Stage S13.3 rs-embed: three weeks of 500s behind a green health check

rs-embed (a separate repository on the same host) returned 500 on every `/api/embed` from
2026-09-08 until 10-01. Its Earth Engine credential was a person's `earthengine authenticate`
token, and that token stopped being accepted. `/api/models` and `/api/health` kept answering 200
throughout, the pattern already recorded for services that succeed at nothing.

Two defects hid it, and both are worth recognising elsewhere.

- **The error you saw was not the error that happened.** `ensure_ready()` caught
  `ee.Initialize`'s exception into a variable it never used, tried a geemap fallback, and
  propagated only the fallback's failure. The expired credential therefore surfaced as
  `module 'geemap' has no attribute 'ee_initialize'` (geemap removed that function in 0.38),
  which sent debugging after the wrong library. A fallback that can fail must chain the primary
  error, not replace it.
- **Initialisation was written twice, and the copies disagreed.** The library half reads
  `EE_PROJECT`. The web app's `_ensure_ee` read `EARTHENGINE_PROJECT` and called `ee.Initialize()`
  itself with no credentials. A fix applied only to the library changed nothing, because
  `/api/embed` goes through the web app. The web app now defers to the library's helper.

It now authenticates as a service account, and the unit sets `EE_PROJECT` explicitly. The unit
previously set no environment at all, so it had been running with no project and inheriting
whatever project the personal credential named. Verified from inside the agent container:
`ok=True`, `backend=gee`, `nodata_fraction=0.0`, and a grid that is 100% finite. The status code
alone was not enough: with no model named, the service returns `{"results":[]}` and still
answers 200.

Where the fix lives is uneven, and the next maintainer needs to know it. The library half is
committed on the VM on a branch that has never been pushed. The web-app half cannot be committed
at all, because rs-embed gitignores `examples/**`. It exists only on that host and in a backup.

### Stage S13.4 A restart is an outage for whoever is mid-turn

Switching the default model on 2026-10-02 at 14:33 UTC was an env-only change, and it cut off
eight live turns: four 502s and four 200 streams truncated mid-answer, across the prod platform
backend and real users. Nothing reported it. It surfaced only because a later, unrelated error
report led to reading the nginx log.

The Deployment section of AGENTS.md said never to *recover* with `--force-recreate`, to preserve
evidence. It did not cover the other reason to recreate: a config change, which needs one and
has a different cost. Two habits are now written down. Check for streams in flight before
recreating. Use `--no-deps`, because without it the same command also recreated `mcp-server` and
`embedding-server`, and agent-api then waited on their health checks.

### Stage S13.5 What this stage did not fix

- **An error a user saw at about 18:05 UTC on 10-02** ("Error getting response from I-GUIDE AI").
  The agent ruled itself out with evidence: every turn it received from 17:00 to 19:00 finished
  (11 started, 11 traces written, each ending `node_completed`, none with an error event inside
  the stream). No chat request reached it between 17:42 and 18:35. The failure therefore
  happened in the platform frontend or backend, whose logs this repository cannot see.
- **Four emitters whose files never reach the panel**: `image_file_id` at three sites in
  `rs_embed_tools.py` and `predictions_file_id` at one. These files are not unnamed; they are
  absent. Making them appear changes what users see, so that is a decision rather than a fix.
- **rs-embed's web-app half is untracked**, as above. A rebuilt host loses it.
- **This file's numbering**, as noted at the top of the stage.

---

## Stage 14 — A promise kept by rounding luck {#stage-14}

`spatial_weights` with `weights='distance_band'` and no `threshold_km` tells the user it *"used
the smallest distance that leaves no island"*. On the deployed Linux image that was false: the
test lattice came back with one island. The test that checks the promise arrived with the tool
(`7cb9f47`, 2026-08-19) and passes on a Mac, so nobody saw it fail until the suite ran on
Linux: in `backend_swap`'s CI, which fixed it in `4e8d327`; in the deployed-version CI being
added to `prototype` (`460cd25`, branch `claude/ci-deployed-constraints`); and inside the deployed
`agent-api` container on 2026-10-01.

This is stage 14 because that CI change opens stage 13, *Testing what is deployed*. Its first
Linux run found this tie, recorded it, and left the test failing for a separate change. This is
that change.

### Stage S14.1 Where the island came from

The automatic threshold is `libpysal.weights.min_threshold_distance`, which returns the critical
pair's distance as a square root. `DistanceBand` then admits a pair when its **squared** distance
is at most the threshold squared, because scipy's KD-tree compares in squared space. Squaring a
rounded square root can come back one ulp short of where it started, and on the 8×8 test lattice
(projected to UTM 31N) it does. Measured on both machines:

* Both compute the same threshold to the last bit, 110,884.46616304158 m, and the distance the
  KD-tree reports for the critical pair equals it exactly: the "margin of 0.0 m". The Linux band
  still drops the pair, because its squared distance is one ulp above the squared threshold.
  Dropping it leaves one cell with no neighbour at all.
* The two machines project one easting of that pair **one ulp apart** (1.2e-10 m at about
  999 km), with identical PROJ 9.5.1 and GEOS 3.13.1. Their KD-trees also disagree on the same
  coordinates: on the Mac's, the Mac's KD-tree keeps the pair and the Linux one drops it.
  Computed by hand, that sum of squares lands exactly on the squared threshold if `dy*dy` is
  fused into the final addition, and one ulp above it with plain arithmetic.

Swapping the pair's exact coordinates between the machines shows that either difference alone
drops the pair:

| pair's coordinates from | macOS/arm64 KD-tree | Linux/x86-64 KD-tree |
|---|---|---|
| macOS | kept | **dropped** |
| Linux | **dropped** | **dropped** |

The coincidence was the Mac passing, not Linux failing. The pair survived only where the Mac's
coordinates met the Mac's arithmetic. These two machines also run different library versions
(libpysal 4.15.0, scipy 1.15.3, numpy 2.1.3 on the Mac; 4.14.1, 1.17.1, 2.4.6 in the container),
but the session working on the extraction branch pinned CI's exact versions on a Mac and still got
no island, so the split follows the platform rather than a version.

### Stage S14.2 The fix, and what it costs

One line after `min_threshold_distance`: `thresh *= 1.0 + 1e-9`. That is about 0.1 mm at 100 km
and 7.6 million ulps at this threshold, so it absorbs any last-bit disagreement while staying far
below any distance an analysis could care about. It is the line `backend_swap` carries in
`4e8d327`, after which that branch's Linux CI went green. This is the only place the code derives
a band from `min_threshold_distance`.

**Revised during the work:** the line and its comment are now copied from `4e8d327` verbatim.
The first version reworded the comment, and the extraction-integration branch already carries
`4e8d327`, so a test merge of `prototype` into it conflicted in this file. With identical lines
the same test merge has no conflict at all. The note's caveat is a separate comment two lines
further down, where it cannot collide.

What it changes, measured on the lattice: nothing on the Mac (146 links before and after), and on
Linux exactly the dropped pair (144 links and 1 island before, 146 and 0 after). Only two
directed pairs lie within a relative 1e-9 of the threshold, and they are that pair. The note
keeps its wording and prints the padded distance. That differs from the strict minimum far below
the note's metre precision, and the unpadded value is not reliably island-free, so "the smallest
distance that leaves no island" is now true on every platform instead of one.

`test_default_distance_band_survives_a_rounding_disagreement` makes the failure reproducible
anywhere. It shrinks the computed minimum by a relative 1e-12, far more than one ulp and a
thousandth of the pad. Without the pad it fails on the Mac with one island, which the original
test never could.

Verification ran inside the deployed `agent-api` container (Python 3.11.16, x86-64). Before ran
against the deployed module, byte-identical to `prototype`'s; after ran against a copy of
`agent_runtime/` in `/tmp` with only this module replaced. Nothing was deployed.
`test_spatial_stats_tools.py` went from 2 failed and 33 passed to 35 passed. The Mac's full suite
went from 1645 passed and 4 skipped to 1646 and 4, the difference being the new test.

### Stage S14.3 The reason worth keeping

A correctness promise was being met by rounding luck, and only a Linux run could show it. The
developer's machine was the one platform where the luck held, so the test that encoded the
promise passed there from the day it was written while the deployed image broke it. A promise of
the form "the smallest X such that…" over floating point needs a margin by construction, and a
test that perturbs its input past the rounding instead of waiting for an unlucky platform.

Not deployed: the running image still has the unpadded threshold, so a caller that omits
`threshold_km` can still get an island until the next image rebuild.

---

## Stage 15 — What counts as a date is decided here, not by pandas {#stage-15}

*2026-10-01, `claude/temporal-numeric-code-columns`.*

A pandas major upgrade silently widened what counts as a date, and only a run on the deployed
versions could show it. `test_csv_with_coordinates_flows_through` failed inside the deployed
`agent-api` container (Python 3.11.16, pandas 3.0.5) and passed on the development machine
(pandas 2.2.3), with nothing in the repository changed: `requirements.txt` names `pandas`
without a version, so the image takes whatever is current when it is built. `prototype` has no
lock file and no CI; separate tasks cover both.

### Stage S15.1 The mechanism, and why the obvious fix would have missed it

The test's `Beat` column (1234, 1235, 1236) was offered as a time column beside `Date`. The
diagnosis this work started from suspected the numeric path, numbers falling through to the
text ladder where pandas 3 infers formats differently, and proposed skipping that ladder for
numeric columns. It also flagged that 1234 sits inside the year branch's range, so which branch
fired had to be checked first. Checked in a replica of the deployed image, built from the
container's own `pip freeze`, the numeric path was not involved and the version difference was
not format inference:

* **`Beat` was never numeric.** GDAL's CSV reader types every field as text, so the column
  reached `parse_time_series` as `"1234"` (dtype `str` on pandas 3, `object` on pandas 2). The
  year branch never ran. The name gate in `_candidate_columns`, which skips a numeric column
  without a time-ish name, asked the dtype, so for a CSV upload it never fired. Skipping the
  text ladder for numeric series would not have touched this fixture.
* **Format inference did not change; resolution did.** On each version the three text
  strategies agree with one another. pandas 3 parses strings at microsecond resolution, so
  `"1234"` became 1234-01-01. pandas 2 parsed at nanoseconds, whose range starts at 1677-09-21,
  and coerced it to NaT. The test had been passing on an accident of `datetime64[ns]`.

Across 31 column shapes, every difference between the two versions but one is a year outside
1678–2262; the exception is in S15.4. Measured on the two versions:

| input | pandas 2.2.3 | pandas 3.0.5, deployed |
|---|---|---|
| CSV: `incident_id` 1001–1005 beside a clean `Date` | `Date` | **`incident_id`** chosen as the time column |
| CSV: IUCR-style codes (`0486`, `0820`, `1310`, `041A`, `2820`) beside a `Date` with two unusable rows | `Date` | **`IUCR`** chosen, 0.8 against 0.6 |
| Chicago-style beats 111–2535, 40 values | 5 read as years | 20 read as years |
| CSV: `Year` beside a `Beat` holding 1711–2212 | **crash**, `OutOfBoundsDatetime` in `_span` | `Beat` offered beside `Year` |

The last row is pandas 2's own failure: the beats parse as years there too, and their 501-year
span overflows its nanosecond `Timedelta`.

### Stage S15.2 Three rules, each stated where pandas used to decide

1. **A column is numeric by what it holds, not by its dtype.** `_as_numbers` accepts typed
   numbers, and text in which every value is a plain number; blanks and the NA markers
   `read_csv` would read as NaN are allowed. The name gate now treats a CSV's numbers exactly as
   it always treated typed ones.
2. **A number gets only numeric readings:** a four-digit year, YYYYMMDD (19000101–21001231),
   epoch seconds or milliseconds. It no longer falls through to the text ladder, and a number
   that fits none of them is a code and stays unparsed. YYYYMMDD is a new explicit branch; it
   used to work only because the text ladder happened to infer `%Y%m%d`.
3. **Every inferred time must fall in 1678–2262** (`_YEAR_FLOOR`, `_YEAR_CEILING`), on the text
   ladder too. That is the window pandas 2 enforced implicitly, and stating it keeps the verdict
   from moving with the pandas version. It is the rule that stops the IUCR column, which `041A`
   keeps from counting as numeric. A column the source already typed as datetime is not clamped.

After the change the 31 shapes give identical results on both versions, apart from S15.4, and
every CSV in the table detects `Date` (or `Year`) alone.

### Stage S15.3 Behaviour that changed on purpose

* A CSV column of epoch seconds now parses. It parsed as nothing on either version.
* A CSV year column with a gap, which `to_csv` writes as `2019.0` and an empty cell, now reads as
  years: 2 of 3 values, where both versions read 0.
* An 8-digit date held as digits in a CSV column **without** a time-ish name is no longer
  auto-detected. That is the treatment a typed integer column always had; `time_column=` still
  reads it.
* Text dates outside 1678–2262, which pandas 3 had started parsing, are unparsed again, as on
  pandas 2.
* `parse_method` has a new value, `YYYYMMDD number`.

Cost, measured on the development machine with both versions interleaved in one process: a
million Chicago-format dates took 1.83 s against 1.77 s, with identical results. The added work
is 0.04 s for the number check and 0.04 s per strategy for the window, so the slowest case, a
column whose blank cells force the whole 22-strategy ladder, pays under a second on top of about
43 s. A million beats held as text got cheaper, 0.74 s against 1.39 s, because the text ladder no
longer runs on them; before the change pandas 2.2.3 read 125,531 of them as years.

### Stage S15.4 Found here, not fixed

* **Mixed UTC offsets stopped parsing on pandas 3.** An ISO column mixing `Z` and `+01:00`, or a
  US daylight-saving pair such as `-06:00` and `-05:00`, parses 2 of 2 on pandas 2.2.3 and 0 of 2
  on 3.0.5. pandas 3 raises `Mixed timezones detected` unless given `utc=True`, every strategy
  fails, and the column reads as having no time at all. It is the same upgrade moving the line
  the other way, so dates that should parse no longer do. Fixed in S15.5.
* **`_span` overflows on pandas 2** when parsed times span more than about 292 years, and the
  whole `detect_time_column` call then fails. This affects the development machine only.
* **A code column inside 1678–2262 with a time-ish name** still parses as years: a `Report Area`
  holding 1711–2212 would. Only the name tells it from a year column, and here the name says time.

### Stage S15.5 Offsets are converted to UTC by pandas, on every version

*2026-10-02, `claude/temporal-mixed-utc-offsets`, stacked on `claude/temporal-numeric-code-columns`.*

This fixes the first item in S15.4. The module docstring promises that a time carrying a UTC
offset is converted to UTC and made naive. On pandas 2 that promise rested on a deprecated path:
`pd.to_datetime` returned a column whose offsets differ as objects, and `_naive` converted the
objects. Its `FutureWarning` named both the change and the fix: such a column *"will raise an
error unless `utc=True`"*. pandas 3 made it the error, `ValueError: Mixed timezones detected`,
and `errors="coerce"` does not suppress it. The
strategy loop's `except Exception: continue` swallowed it in the inferred and the mixed strategy
alike, and no explicit format has a `%z`, so none can match a time with an offset. On the deployed
image a CSV of a US feed's local times, `-06:00` in winter and `-05:00` in summer, got
`found: false` from `detect_time_column`, and `filter_by_time`, `time_series` and
`temporal_hotspots` each answered "no time/date column could be detected in this dataset".

**The rule:** the inferred and mixed strategies pass `utc=True`, so pandas converts every offset
itself and `_naive` only drops the zone. Checked on both versions, on the raw `pd.to_datetime`
calls and then through `parse_time_series`:

* With `utc=True` the two strategies give the same instants on 2.2.3 and 3.0.5 for all 13 offset
  shapes tried inside pandas 2's nanosecond range: `Z` beside `+01:00`; `-06:00` beside `-05:00`
  with a `T` or a space, without the colon, with fractional seconds, or in RFC 2822; with garbage,
  blanks and nulls among them; the two `01:30` of a fall-back night; single offsets; and text
  with an offset beside text without.
* Text without an offset is read as UTC and made naive again, so its values do not move: ISO,
  Chicago, date-only, day-first, month names, `YYYY-MM`.
* The explicit formats are left as they were. None of the 20 matches any of those offset strings
  on either version, with `utc=True` or without it, because none has a `%z`.

Then `parse_time_series` itself, on 49 column shapes:

| shapes | pandas 2.2.3 (development) | pandas 3.0.5 (deployed) |
|---|---|---|
| 12 whose offsets differ | parsed; unchanged | **0% unfixed**; fixed, the same instants as 2.2.3 |
| 4 mixing text with an offset and text without | 2 of 2; unchanged | **1 of 2 unfixed**, only the rows that fit the first value's layout; fixed, 2 of 2 |
| 3 of offset times outside or at the edges of 1678–2262 | unchanged but one, below | 0 unfixed; fixed, as on 2.2.3 but one, below |
| the other 30 | unchanged | unchanged |

Fixed, the deployed version returns what pandas 2.2.3 always returned on 48 of the 49 shapes. The
49th is an offset time that lands on 2262-04-12 UTC: the year window of S15.2 admits it and pandas
3 keeps it, but it is past pandas 2's nanosecond ceiling of 2262-04-11 23:47. On 2.2.3 the fix
moves one shape, the wrong way; it is the first item under *Found here, not fixed* below.

**Behaviour that changed on purpose:**

* A column whose offsets differ parses again on pandas 3, in UTC, as the docstring says.
* In a column mixing times that carry an offset with times that do not, the ones without are read
  as UTC. pandas 2 did this all along, through `_naive`'s own `utc=True`. Unfixed, pandas 3 kept
  only the rows that fit the first value's layout.

**Found here, not fixed:**

* **pandas 2.2.3 wraps around instead of failing** when `utc=True` pushes an offset time past its
  nanosecond range, in the inferred strategy only. `2262-04-11T23:00:00-05:00` becomes
  1677-09-21, which the year window drops. `1677-09-21T00:30:00+01:00` becomes 2262-04-11 23:04,
  which the window keeps: a wrong time where the unfixed code gave NaT. Only a clock time on
  1677-09-21 with a positive offset can reach it, and only on pandas 2, so only on the
  development machine. pandas 3 does not wrap.
* **`summary_statistics(by=<date column>, period=...)`** in `analysis_aggregate_tools.py` parses
  the column with its own `pd.to_datetime`. A daylight-saving column fails there on both versions,
  with `Can only use .dt accessor with datetimelike values` on 2.2.3 and `Mixed timezones
  detected` on 3.0.5. A single offset is bucketed by local clock time: `2026-01-31T20:00:00-06:00`
  counts in January there and in February in `time_series`, which works in UTC. Fixed in S15.6.
* **Abbreviated zone names** (`CST`, `CDT`) parse on neither version, fixed or not.

**Cost**, as the best of interleaved rounds in one process, with every value distinct so that
`to_datetime`'s cache never helps:

| 100,000 values, deployed container (3.0.5) | unfixed | fixed |
|---|---|---|
| Chicago format | 5.34 s | 5.24 s |
| ISO with no offset; with one offset | 0.04 s; 0.42 s | 0.04 s; 0.42 s |
| ISO with `-06:00` and `-05:00` | 2.78 s, nothing parsed | 0.42 s |

On the deployed version the rule costs nothing measurable, and a daylight-saving column, which
used to run all 22 strategies to fail, now parses at the first. The development machine was not
timed, because other work held its load average at 63, so pandas 2.2.3's cost is not measured.

**Verified where the failure lives:**

| | unfixed (`claude/temporal-numeric-code-columns`) | fixed |
|---|---|---|
| deployed container (pandas 3.0.5), the 38 temporal tests | the 2 new mixed-offset tests fail | all pass |
| development machine (pandas 2.2.3), the 38 temporal tests | all pass | all pass |

The live module in the same container, `prototype`'s, fails 9 of the 38: the 7 of S15.7 and the
same 2. On the development machine the full suite gives 1654 passed and 4 skipped: S15.7's 1651
and the three new tests.

Three tests are new. Two fail unfixed on pandas 3: offsets that differ (the daylight-saving pair,
`Z` beside `+01:00`, and text with and without an offset), and a CSV of daylight-saving local
times through `detect_time_column` and `filter_by_time`, whose window is checked in UTC. The third
is a guard that passes either way: one offset is still converted, and text without one keeps its
clock time. The development machine passes all three unfixed, so it cannot show this failure. As
in S15.7, each version of `agent_runtime/` was imported from `/tmp` ahead of `/app` in the
deployed container; nothing was deployed.

### Stage S15.6 `summary_statistics` reads a date the way the temporal tools do

*2026-10-02, `claude/summary-statistics-utc-periods`, stacked on `claude/temporal-mixed-utc-offsets`.*

This fixes the second item in S15.5's *Found here, not fixed*. `summary_statistics(by=<column>,
period=day|week|month|quarter|year)` read the `by` column with its own
`pd.to_datetime(errors="coerce")`, so none of this stage's rules reached it: an offset was not
converted to UTC, a number got pandas' default reading, and nothing held the years to 1678–2262.
Through the tool, on CSVs of points:

* **A daylight-saving column failed on both versions**: `Can only use .dt accessor with
  datetimelike values` on 2.2.3, `Mixed timezones detected` on 3.0.5. `Z` beside `+01:00` failed
  the same way.
* **One offset was bucketed by local clock time.** `2026-01-31T20:00:00-06:00` is 02:00 UTC on
  February 1, so the same three rows gave January 1 and February 2 here and February 3 in
  `time_series`.
* **On pandas 3 a CSV's beats became years.** `"1234"`, `"1235"` and `"1236"` grouped by year came
  back as the years 1234, 1235 and 1236. pandas 2.2.3 refused the column.

Two more turned up, the first while checking what a numeric `by` does, and a GeoJSON input showed
where the daylight-saving failure lives:

* **A typed number was read as nanoseconds since 1970**, `pd.to_datetime`'s default for an
  integer. A GeoJSON's years (2019, 2020), its beat codes (1234–1236) and its epoch seconds each
  came back as a single `1970` or `1970-01` bucket, with `ok: true`, on both versions.
* **An undated row's group was named by the pandas version**: `NaT` on 2.2.3, `(missing)` on
  3.0.5. pandas 3's `astype(str)` keeps a missing value missing, so the `fillna("(missing)")`
  after it worked there only.
* **GDAL types a GeoJSON's ISO date-times itself**, so the daylight-saving failure was a CSV one.
  One offset arrives as `datetime64[ms, UTC-06:00]` and was bucketed by local clock too; offsets
  that differ arrive already converted to UTC, and were right before.

**The change:** the `period` branch reads the column with `parse_time_series`, so it gets the
reading every temporal tool gets: an offset converted to UTC, a number only numeric readings, a
year in 1678–2262. The import sits inside the branch, not at the top of the module. The supervisor
imports each analysis factory separately so that one module's failure costs only its own tools,
and `analysis_spatial_stats_tools` imports this module when it loads, so a top-level import would
have tied both to the temporal module. An undated row is grouped as `(missing)` on both versions
and counted in a note; `time_series` drops such rows instead and reports them as
`excluded_unparsed_time`. The tool description, which the model reads, and the docstring now say
that the buckets are in UTC when the dates carry an offset.

Through the tool on both versions, 16 column shapes:

| shapes | 2.2.3, before | 3.0.5, before | after, both versions |
|---|---|---|---|
| 2 CSVs whose offsets differ (`-06:00` and `-05:00`; `Z` and `+01:00`) | `.dt` error | `Mixed timezones` error | UTC buckets |
| 2 with one offset, CSV and GeoJSON | local clock | local clock | UTC buckets |
| beats as text, `"1234"`–`"1236"`, by year | refused | the years 1234–1236 | refused |
| epoch seconds as text | refused | refused | their dates |
| 3 typed numbers: years, beat codes, epoch seconds | all `1970` | all `1970` | years; refused; dates |
| one row reading `not a date` | group `NaT` | group `(missing)` | group `(missing)`, and a note |
| 6 others: ISO, Chicago format, years as text, weeks, GeoJSON dates, GeoJSON offsets that differ | | | unchanged |

After the change the two versions agree on all 16; before, they disagreed on 4, two of them only in
the error message. The tool now gives `time_series`'s buckets on 12 of the 16, against 5. The other
4 differ on purpose or from before: both tools refuse the two code columns, in different words; the
undated row is kept here and dropped there; and a week is the same bucket in both, labelled
`2026-01-05/2026-01-11` here and `2026-01-05` in `time_series`.

**Behaviour that changed on purpose:**

* A column whose offsets differ is bucketed, in UTC, on both versions.
* One offset is bucketed in UTC, as in `time_series`, so a row within a few hours of midnight UTC
  can move to the next or the previous bucket.
* Typed years give their years and typed epoch seconds their dates, where both gave `1970`; epoch
  seconds held as text, which were refused, read as dates too.
* A code column outside 1678–2262, such as the beats 1234–1236, is refused with *"could not be
  read as dates"*, typed or as text, where a typed one gave `1970` and, on pandas 3, a text one
  gave years.
* On 2.2.3 an undated row's group is `(missing)`, no longer `NaT`, and on both versions a note
  counts those rows.

**Found here, not fixed:**

* **`time_series`' hour-of-day profile calls UTC hours local.** Its axis reads *"hour of day
  (local clock time)"*, but a time that carried an offset is in UTC by then: `09:00:00-06:00`
  counts at hour 15. For a Chicago feed the peak moves by six hours in winter and five in
  summer, under a label that says it did not.
* **A code column inside 1678–2262 is read as years**, typed or as text, whatever its name: beats
  1711, 1712 and 2212 grouped by year come back as those years. S15.4 records the same ambiguity
  for detection. Here the caller has named the column and asked for periods, and only the values
  could say otherwise.
* **A week has two labels for one bucket**, as above. Each label is part of its tool's output,
  so neither was changed here.

**Cost** of parsing and bucketing the column, without reading the file, interleaved in one
process, the median of three rounds, every date distinct:

| 100,000 values, deployed container (3.0.5) | before | after |
|---|---|---|
| ISO, no offset | 0.03 s | 0.06 s |
| ISO, one offset | 0.40 s | 0.43 s |
| Chicago format | 5.19 s | 5.23 s |
| ISO with 1 value in 100 unreadable | 0.06 s | 1.94 s |

A column that reads in full stops at the first strategy and costs about 0.03 s more. The last row
is the slowest case S15.3 describes for blank cells: a column holding any value no strategy reads
runs all 22 strategies, which `time_series` already does on the same column. Only the deployed
container was timed, at a load average under 0.7 on 8 cores.

**Verified where the failure lives:**

| | before (`claude/temporal-mixed-utc-offsets`) | after |
|---|---|---|
| deployed container (pandas 3.0.5), the 36 aggregate tests | the 4 new tests fail | all pass |
| deployed container, the 38 temporal tests | | all pass |
| development machine (pandas 2.2.3), the 36 aggregate tests | the 4 new tests fail | all pass |
| development machine, full suite | 1654 passed, 4 skipped | 1658 passed, 4 skipped |

Four tests are new, and each fails before on both versions: a daylight-saving CSV grouped by month,
whose buckets are UTC ones; a CSV with one offset, whose months are the ones `time_series` counts;
typed years and epoch seconds read as dates and the beats 1234–1236 refused, typed and as text; and
an undated row grouped as `(missing)` with a note, which fails at the label on 2.2.3 and at the
note on 3.0.5. In the deployed container *before* was `/app`'s `agent_runtime/` with S15.5's
`analysis_temporal_tools.py`, and *after* was that plus this module. Each was imported from `/tmp`
ahead of `/app`, `diff -rq` named only those files, and it was the same container from the first
run to the last. `prototype`'s `summary_statistics`, live in the same container, gives *before*'s
answer on all 16 shapes. Nothing was deployed.

### Stage S15.7 Verified where the failure lives (S15.1–S15.4)

| | unfixed | fixed |
|---|---|---|
| deployed container, `test_csv_with_coordinates_flows_through` | fails | passes |
| deployed container, the 35 temporal tests | 7 fail | all pass |
| local replica of the deployed versions, the 35 temporal tests | the same 7 fail | all pass |
| development machine (pandas 2.2.3), the 35 temporal tests | 4 fail | all pass |
| development machine, full suite | 1645 passed, 4 skipped | 1651 passed, 4 skipped |

In the deployed container each version of `agent_runtime/` was imported from `/tmp` ahead of
`/app`, whose copy of the module is identical to `prototype`'s; nothing was deployed. The six new
tests pin the three rules, the YYYYMMDD and bare-year readings, and the CSV shapes above. The
original test is unchanged.

### Stage S15.8 A year column is found by its name, and the blank row that let Beat win

*2026-10-02, `claude/temporal-yr-and-blank-date`, stacked on `claude/temporal-numeric-code-columns`.*

Rule 1 of S15.2 sends a CSV's numbers through the name gate, which is what keeps the beats out.
The same gate decides whether a real year column is tried at all, and no hint matched `yr`. On
`prototype` a CSV's `yr` column never needed one: GDAL hands it over as text, text is always a
candidate, and the text ladder read `"2019"` as a year. Counted as numbers, the column faced the
gate and failed it, so a dataset whose only time is a `yr` column reported no time column. `Year`
passed only because `year` was already a hint. A typed `yr` column, from GeoJSON or a GeoPackage,
had never been tried on any branch, for the same reason.

**The name is the fix, because nothing else tells the two apart.** The year window cannot separate
a year column from a code column whose values fall inside it: Chicago's beats include 1834 and
2011. `yr` is now a hint. Hints match as substrings, which catches run-together assessor names such
as `YRBUILT`, `SALEYR` and `TAXYR` that a whole-word rule would miss. A substring also matches
names like `gyro` or `copyright`, but such a column still has to hold years, YYYYMMDD or epochs to
parse, and a count like `yr2020_pop` is tried exactly as `year2020_pop` already was.

**The blank date row is the deployed failure in full, and no test had it.** On the deployed
versions `test_csv_with_coordinates_flows_through` failed only at
`assert 'Beat' not in ['Date', 'Beat']`: its `Date` has no gap, so the two tied at 1.0, the name
hint broke the tie, and `Date` still won. The first Linux CI run, on
`claude/ci-deployed-constraints`, measured the sharper form. GDAL hands a blank cell over as `""`,
which counts against `Date`, so with one blank in four rows `Date` parses 0.75 and `Beat`, read as
the years 1234–1237, wins at 1.0: the deployed tool chose a beat number as the time column.
`test_csv_beat_cannot_outrank_a_date_with_a_blank_row` pins that shape and keeps the original's
`Beat` assertion.

Measured through the real upload path, in a linux/amd64 `python:3.11-slim` replica of the deployed
versions (`pip install -r requirements.txt -c constraints.txt`, the lock from
`claude/ci-deployed-constraints`: CPython 3.11.16, pandas 3.0.5, geopandas 1.1.4, pyogrio 0.13.0
with GDAL 3.12.4):

| upload | `prototype` | S15.2 without `yr` | with `yr` |
|---|---|---|---|
| CSV: `Date` blank in one row of four, `Beat` 1234–1237 | **`Beat`** chosen, 1.0 against 0.75 | `Date` | `Date` |
| CSV: `Year` beside `Beat` 1234–1236 | `Year`, with `Beat` offered too | `Year` | `Year` |
| CSV: `yr` beside `Beat` 1834–2012 | `yr`, with `Beat` offered too | **no time column** | `yr` |
| CSV: `YRBUILT` beside `Beat` 1834–2012 | `YRBUILT`, with `Beat` offered too | **no time column** | `YRBUILT` |
| GeoJSON: integer `yr` beside `Beat` 1834–2012 | no time column | no time column | `yr` |

**Behaviour that changed on purpose:** a column whose name contains `yr` is now tried when it holds
numbers, typed or as CSV text, and gets the numeric readings any hinted number gets.

**Found here, not fixed:** a CSV year column named without any hint, such as `FY` or `season`, is
still not auto-detected. `prototype` read both through the text ladder; under S15.2 they report no
time column, with or without `yr`, measured the same way as the table. `time_column=` still reads
them. Each further hint widens what the gate lets through, so which names to add is a separate
decision.

**Verified where the failure lives**, in the same replica, with this section's four new test cases
copied into each version's test file:

| `agent_runtime/` | the 39 temporal tests |
|---|---|
| `prototype` | 11 fail: this stage's original 7, the blank-row test (`Beat` chosen) and the 3 `yr` cases |
| `claude/temporal-numeric-code-columns` | 3 fail: the 3 `yr` cases, each with no time column |
| with `yr` | all pass |

The full suite in the same replica gives 1 failed, 1654 passed and 4 skipped with this change, and 2
failed, 1643 passed and 4 skipped on `prototype`, the counts of the first Linux CI run. The failure
left is `test_distance_band_without_a_threshold_leaves_no_island`, the platform-rounding failure
`prototype` has too, which a separate change fixes; the one this change and its parent remove is
`test_csv_with_coordinates_flows_through`. The development machine (pandas 2.2.3) passes the 39
temporal tests as well. Nothing was deployed.

---

## Stage 16 — Six tests only the Mac passed {#stage-16}

A replica of the deployed `agent-api` Python environment was built on 2026-10-01 and again on
2026-10-02: `python:3.11-slim` for linux/amd64, the image's GDAL apt layer, `requirements.txt`
installed with the running container's `pip freeze` as constraints, and the `en_core_web_sm`
3.8.0 wheel. Its own `pip freeze` matches the deployed container's (image `deeb331964f6`) on 176
of 177 lines; the missing one is `py-spy`, which the real Dockerfile adds in a later layer. Five
tests that pass on the development Mac fail in it, and an image built from
`rag_pipeline/Dockerfile` fails a sixth. Each passed on the Mac because the Mac lacks something
production has, or has something production does not.

| test | why the Mac passed it | which side was wrong | fixed in |
|---|---|---|---|
| `test_csv_with_coordinates_flows_through` | pandas 2.2.3, whose nanosecond range turns `"1234"` into NaT | **the code**. In the replica, a CSV with one blank date in four rows gets `Beat` as its time column, and `filter_by_time(start="2026-07")` answers `ok` with 0 matches | `claude/temporal-numeric-code-columns` (stage 13 there) |
| three in `test_spatial_locations.py` | no spaCy model, so the capitalization fallback ran | **the code**. Production's NER path skipped the vocabulary and the normalization the fallback applies | this stage, S16.1 |
| `test_distance_band_without_a_threshold_leaves_no_island` | the Mac's floating point kept a pair that sits exactly on the threshold | **the code**. The threshold had no margin | `claude/distance-band-no-island` (stage 14) |
| `test_pyqgis_available_probes_worker_python` | no QGIS installed | **the test**. It assumed the machine had no QGIS | this stage, S16.2 |

The island's cause was measured again in the replica rather than taken from the stage 14 branch.
The automatic threshold, 110,884.46616304158 m, is bit-identical on both machines. The Mac
projects the critical pair's easting one ulp higher (`0x1.e7c562adf9522p+19` against
`...9521p+19`). On both machines the square of the threshold is one ulp below the pair's squared
distance as Python computes it. The Mac's KD-tree keeps the pair anyway and the replica's drops
it: 146 links and no island against 144 links and one. With stage 14's relative pad of 1e-9,
both give 146 links and no island.

### Stage S16.1 Production's NER path never saw the fallback's filters

`extract_locations_from_query` (`rag_pipeline/search/spatial.py`) turns a question into the
place names it geocodes. It has two paths. Where `en_core_web_sm` is installed, as in the agent
image, it offers spaCy's GPE, LOC and FAC entities. Where it is absent, as on the Mac and in CI,
it offers capitalized phrases filtered through `_NOT_PLACES`. The fallback produces exactly the
three tests' expectations: with the model uninstalled in the replica, all 15 tests in the file
pass. With the model, measured on spaCy 3.8.16 and `en_core_web_sm` 3.8.0, the NER path differs
in two ways.

* **It keeps the article.** "the Great Plains" (FAC), "the Chesapeake Bay" (LOC), "the United
  States", "the Gulf of Mexico", "the Rocky Mountains" and "the Great Lakes" all come back with
  the "the". The fallback has never offered it, so the same place reached the geocoder and
  `_BBOX_CACHE` under two spellings, depending on which path ran.
* **It labels formats, tools and agencies as places.** "GeoJSON" is a GPE in three of the
  queries below, and "NetCDF", "MODIS", "USGS", "LAS" and "Python" are GPEs elsewhere. Every
  one is in `_NOT_PLACES`, whose comment says these terms are never places, but only the
  fallback consults that set. So for "convert a GeoJSON to a COG with GDAL" the deployed
  extractor offers "GeoJSON" to Google's geocoder, which is paid and rate-limited, and a box that
  came back would scope the spatial search to wherever Google put it.

The tests stated the intended behaviour and the code was wrong. NER cannot know that a file
format is not a place, and the vocabulary that says so was skipped on the one path production
runs. The article is the weaker half. Whether it ever changed what Google returned was not
measured, because that takes paid calls, so dropping it is normalization: one spelling per place,
whichever path ran. The changes are these:

1. A leading **lowercase** "the" is dropped from an entity (`_without_article`). A capitalized
   one is kept. NER returns "The Hague" as a GPE, and at the start of a query a capital "The"
   cannot be told apart from a name.
2. An entity whose every word is in `_NOT_PLACES` is dropped (`_only_non_places`).
3. Dropping such an entity does not open the fallback, which is for text NER could not parse,
   such as the search peer's keyword form. Here NER parsed the sentence and found only a file
   format. Falling back would offer the capitalized words instead: "Convert" for "Convert a
   GeoJSON to a COG with GDAL", where spaCy reads "Convert" as a PERSON.

**Measured on 176 queries in the replica, with the real model.** The queries are the file's
test queries, 30 written around articles, 32 around technical terms, GeoAnalystBench's 44
distinct task titles and its 50 instructions, and the 10 prompts in
`geopathfinder_top10_tasks.csv`. 21 of them change. In 12 a leading "the" goes, and in 9 a
technical term goes (GeoJSON 3, Python 2, and one each of USGS, MODIS, LAS and NetCDF). No
candidate is added and no place is lost: where a technical term sat beside a place, Colorado and
Chicago remain. Six queries go from one non-place candidate to none. For scale, the deployed
container logged 9 candidates resolved to a box between its start (2026-10-01 15:15 UTC) and this
measurement, and none of them begins with an article or is a technical term.

The three tests pass as written, now on both paths. Ten new tests give the NER path the entities
the model returned, through a blank English pipeline with those spans set, so the path
production runs is tested on machines without the model. Against the old code, 8 of the 10 fail
on the Mac.

Found and **not** fixed, because each needs its own measurement:

* **The fallback offers a request's first word.** It takes any capitalized word that is not in
  `_NOT_PLACES`, and an imperative request starts with one. 30 of the 44 GeoAnalystBench task
  titles and 7 of the 10 geopathfinder prompts name no place that NER finds, and each offers its
  first word as a place: "Identify", "Use", "Find". Whether Google returns a box for these was not
  measured, because that takes paid calls.
* NER also labels "Kriging" (LOC), "Tsunami" (GPE) and "node" (GPE) as places, and no vocabulary
  covers them.
* CI as proposed on `claude/ci-deployed-constraints` runs the fallback path, not production's.
  The freeze's `en_core_web_sm @ <url>` line constrains a package that nothing requests, so pip
  installs no model. The new NER tests are what cover production's path there.

### Stage S16.2 A test that assumed QGIS was absent

`test_pyqgis_available_probes_worker_python` set `QGIS_PYTHON_BIN` to a nonexistent path and
expected `pyqgis_available()` to be False. But `qgis_python_candidates()` falls back to
`sys.executable` and then `/usr/bin/python3` on purpose, so that a developer's `.env` naming
QGIS.app does not disable PyQGIS inside the container. In the agent image `/usr/bin/python3` is
the distro Python 3.13.5 that `python3-qgis` installs into, in the deployed container as in a
fresh build, so the probe found QGIS, and the test failed in every image with QGIS. The code was
right. The test now pins the whole candidate list, and a fresh probe cache, instead of one
environment variable. No production behaviour changes.

In an image built from `rag_pipeline/Dockerfile` (linux/amd64, 2026-10-02),
`test_qgis_headless_tools.py` went from 1 failed and 20 passed to 21 passed.

### Stage S16.3 Verification, and what is not deployed

| where | tree | failed | passed | skipped |
|---|---|---|---|---|
| Mac (arm64, Python 3.13.5, pandas 2.2.3, no spaCy model) | `prototype` | 0 | 1645 | 4 |
| Mac | this branch | 0 | 1655 | 4 |
| Mac | this branch with the temporal and island fixes | 0 | 1662 | 4 |
| replica (amd64, the deployed versions) | `prototype` | 5 | 1640 | 4 |
| replica | this branch | 2 | 1653 | 4 |
| replica | this branch with the temporal and island fixes | **0** | 1662 | 4 |

The two failures on this branch alone are the temporal and island tests, which
`claude/temporal-numeric-code-columns` and `claude/distance-band-no-island` fix. "With the
temporal and island fixes" means with those two branches' code and tests applied. They touch no
file this stage touches. Each suite run mounted a fresh copy of the tree. An image built from
`rag_pipeline/Dockerfile` the same day fails exactly the six on `prototype` (6 failed, 1639
passed, 4 skipped). Its pip layer resolved that day's versions rather than the deployed ones, so
it is the reference only for the QGIS test.

Nothing was deployed. The running image still offers "GeoJSON" to the geocoder.

---

## Stage 17 — The image installs a list, not a laptop {#stage-17}

`reproject_vector` failed on every call in the deployed agent, and nobody had noticed. It writes
GeoParquet, which needs pyarrow, and the image had no pyarrow. Chasing that turned up more
packages the code reaches for and the image lacks, most of them latent, because only ingestion
uses them and ingestion does not run in the container yet.

### Stage S17.1 Why a missing declaration shows only in production

`rag_pipeline/Dockerfile` installs `requirements.txt` and nothing else. A development machine has
far more: anaconda's own packages, plus whatever `pip install --user` left in `~/.local`. Code
that reaches for a package the file never names therefore passes every local test and every
manual check, and fails only in the container.

Two properties of the code hid it further. None of these packages is imported at module scope,
and pyarrow is never imported *by name*: pandas and geopandas load it inside `to_parquet` and
`read_parquet`, where an import grep cannot see it. And where the absence does bite, most of the
code degrades instead of raising: a pickle instead of parquet, an empty string instead of a
PDF's text, a note instead of a NetCDF file's variables.

| package | reached from | in the deployed image, without it |
|---|---|---|
| pyarrow | `reproject_vector` on every call; `vector_spatial_join` above `AGENT_GEOJSON_MAX_FEATURES`; `read_vector`, the temporal tools and `extractors/geo_handles.py` reading parquet back | `reproject_vector` answered `Missing optional dependency 'pyarrow.parquet'` every time; `geo_handles` silently wrote pickles that only it can open |
| pypdf | `publication_extractor` | every PDF read as empty text, filed under the note `no_text_extracted`, which does not say why |
| python-docx | `publication_extractor` | every `.docx` read as empty text, the same way |
| xarray | `data_extractor` | every NetCDF, HDF or GRIB file answered `raster reader unavailable/failed` |

All four were confirmed inside the deployed `agent-api` container on 2026-10-01: each import
raises `ModuleNotFoundError`, and `GeoDataFrame.to_parquet` raises the error above. pyarrow was
never declared. No commit ever added it to `requirements.txt`, and a local image built on
2026-06-25 lacks it too. The parquet writes date from `421fc8da` (2026-06-12).

### Stage S17.2 Why the suite never caught one

A replica of the deployed environment was built by installing `requirements.txt` into
`python:3.11-slim` (amd64), with the deployed container's own `pip freeze` as constraints. Its
freeze matches production's in 175 of 176 packages; the one missing, py-spy, comes from a later
layer of the real Dockerfile. The full suite inside it gave **5 failed, 1640 passed, 4 skipped,
and not one failure was an import error.** Nothing in the suite reached any of the four:

- `test_spatial_join`'s three points stay under the GeoJSON limit and come back as GeoJSON, and
  no test called `reproject_vector`. So the vector tools' tests pass without pyarrow, in the
  replica as on dev. Dev's pyarrow was never what made them pass.
- No test touched `publication_extractor`, `data_extractor` or `geo_handles` at all.

A CI job pinned to the deployed freeze would therefore have caught none of them. Two tests in
`test_langchain_geo_tools.py` now cover the parquet round trip: `reproject_vector` writes it and
`inspect_vector` reads it back, and a spatial join over the limit does the same.
`test_declared_dependencies.py` covers the rest. **None of them uses `importorskip`,
deliberately:** a skip is exactly how a missing package passes. In the replica all six fail, and
none skips:

| test | in the replica |
|---|---|
| reproject round trip; spatial join over the limit | `ImportError: Missing optional dependency 'pyarrow.parquet'` |
| `geo_handles` frame passing | `assert '.pkl' == '.parquet'` |
| PDF text | `assert 'Flood exposure by census tract' in ''` |
| `.docx` text | `ModuleNotFoundError: No module named 'docx'` |
| NetCDF metadata | `raster reader unavailable/failed: ModuleNotFoundError: No module named 'xarray'` |

With the four packages added, all six pass.

The five replica-only failures are the same gap running the other way: there, dev is *older*
than production. They are not fixed here, and a CI job pinned to the deployed freeze will see all
five. Each was checked by changing one package in the replica to dev's version.

| failing test | dev | deployed | swap that makes it pass |
|---|---|---|---|
| `test_csv_with_coordinates_flows_through`, a stray `Beat` column | pandas 2.2.3 | pandas 3.0.5 | pandas 2.2.3 |
| three in `test_spatial_locations.py`, e.g. `'the Great Plains' == 'Great Plains'` | no spaCy model, so the regex fallback | `en_core_web_sm` | removing the model |
| `test_distance_band_without_a_threshold_leaves_no_island` | | | **none found.** Dev's pandas, numpy, scipy, esda, libpysal, geopandas, pyogrio and scikit-learn each still fail. Cause not established. |

### Stage S17.3 The pins

The four are pinned, unlike most of the file, because each version was checked against what
production runs. Installed on top of the replica, with the deployed freeze as constraints, they
add exactly four packages and move none of production's. pyarrow 25.0.1 is the sandbox image's
version (`iguide-codeexec`), which reads the same files. pypdf 6.6.2, python-docx 1.2.0 and
xarray 2026.7.0 are the versions `backend_swap`'s lock pinned when its Linux CI went green at
`4e8d327`.

### Stage S17.4 What this stage did not fix

- **xarray opens NetCDF3 and nothing newer.** Its only file engine in the image is scipy.
  NetCDF4/HDF5 needs `netCDF4` or `h5netcdf`, GRIB needs `cfgrib`, and neither dev nor the image
  has any of them, so this is a format `data_extractor` has never read rather than a missing
  declaration. rasterio's GDAL in the image does have netCDF, HDF5 and GRIB drivers, but
  `data_extractor` sends those extensions to xarray alone.
- **`data_extractor` reads `ds.dims` as a mapping**, which xarray 2026.7 warns will become a set
  of names. The pin holds it. A bump past that change would fail silently into the same
  `raster reader unavailable` note; `ds.sizes` is the fix.
- **pystac-client** is on dev and not in the image, but nothing reaches it: STAC is commented out
  of `_DEFAULT_PROVIDERS`, and neither caller of `get_opengeodata_results` passes providers.
- **colbert** is imported at module scope by `rag_pipeline/reranker.py`, which only
  `scripts/demo_reranker.py` imports, and `scripts/` is not copied into the image.
- **Twelve packages are imported directly but declared nowhere**, arriving only as somebody
  else's dependency: numpy, pyproj, pyogrio, pillow, scikit-learn, Werkzeug, uvicorn, PyYAML,
  anthropic, affine, langgraph-checkpoint and langgraph-prebuilt. None is missing today; each
  stays only as long as its parent keeps bringing it.

### Stage S17.5 Three more, found by scanning against the replica

The replica's suite raised no import errors, so the rest came from a static pass over every
import in the five packages the image ships, plus every pandas or xarray call that loads an engine
on demand (`to_parquet`, `read_parquet`, `read_excel`, `open_dataset`), each checked in the
replica. Three more packages are reached by the code, present on dev, and absent from the image:

| package | reached from | in the image, without it | on dev |
|---|---|---|---|
| openpyxl | `detect_time_column` and `time_series` on a `.xlsx` with no coordinate columns. GDAL opens the file, `read_vector` refuses a table without coordinates, and `_read_plain_table` falls back to `pd.read_excel`. | ``ImportError: `Import openpyxl` failed``. With it, the same upload gives three monthly periods. | 3.1.5, from anaconda |
| mapclassify | `choropleth_image(scheme=...)`, which the analysis peer binds whether or not files are attached | the scheme is dropped and a continuous ramp drawn, and nothing in the result says so | 2.10.0, from `~/.local` |
| IPython | `notebook_extractor`, at ingestion | a regex fallback. Of seven typical cells, `np.mean?` and a `!command` inside a loop fail to parse; IPython parses all seven. | 8.30.0, from anaconda |

openpyxl and mapclassify are live on the deployed agent's path; IPython is latent, like the
readers. openpyxl 3.1.5 and mapclassify 2.10.0 match the sandbox image. IPython 8.30.0 is dev's
version, the one the notebook front end was written against. Each has a test in
`test_declared_dependencies.py` that fails in the replica: the spreadsheet test cannot even write
its fixture without openpyxl, `choropleth_image` never passes `scheme` to the plot, and
`transform_cell` cannot parse either cell. With all seven pins on top of the replica, the install
adds 20 packages, the seven plus 13 dependencies (12 of them IPython's), and moves none of
production's.

Not declared, because it never worked on dev either: `.xls` needs xlrd, which neither dev nor the
image has, and the GDAL inside the pyogrio wheel has no XLS driver.

### Stage S17.6 Building it, and what the next deploy will change

The real `rag_pipeline/Dockerfile`, built from this `requirements.txt` on `python:3.11-slim` for
amd64, installs all seven, and a GeoParquet round trip works inside the result. The full suite
inside that image gives **6 failed, 1648 passed, 4 skipped.** Every new test passes and no failure
is an import error. The six are the replica's five plus `test_pyqgis_available_probes_worker_python`,
which fails in any image with QGIS installed, with or without this change: it points
`QGIS_PYTHON_BIN` at a missing interpreter and expects "unavailable", while
`qgis_python_candidates()` deliberately falls back to `/usr/bin/python3`, which has QGIS. It fails
the same way with the unmodified tree, and in a local image built on 2026-06-25.

**Deploying this changes more than these seven packages.** A changed `requirements.txt`
invalidates the image's pip layer, so the build that ships it re-resolves every unpinned name in
the file to whatever is newest that day. Against the deployed freeze, the fresh build changes 42
packages, none by a major version, and adds 21: the 20 above, plus opentelemetry-api, now pulled
in by an upgraded dependency. The moves most likely to change behaviour:

| package | deployed | fresh build |
|---|---|---|
| openai | 3.14.1 | 3.23.0 |
| anthropic | 1.6.0 | 1.11.0 |
| langsmith | 0.12.6 | 0.14.3 |
| langchain, langchain-core, langchain-openai | 1.4.1, 1.6.3, 1.6.2 | 1.4.3, 1.6.6, 1.6.7 |
| geopandas | 1.1.4 | 1.2.0 |
| sentence-transformers | 6.0.1 | 6.1.0 |

Installing with the deployed freeze as constraints ships only the additions, as the replica
shows. That is the job of a lock file, like the one on `backend_swap`; this stage does not add
one.

---

## Stage 18 — Testing what is deployed {#stage-18}

Until this stage the test suite ran in one place, the development Mac. It had never run in the
environment the deployment runs, which is Linux x86-64 with CPython 3.11
(`rag_pipeline/Dockerfile` is `python:3.11-slim`), and nothing recorded what that environment's
packages were.

### Stage S18.1 The deployed stack was an accident of build day

`requirements.txt` names 42 packages and pins none of them exactly. 35 are bare names, among them
geopandas, shapely, fiona, rasterio, libpysal and pandas; seven carry only a lower bound. numpy
and scipy are not named at all, because they arrive transitively. So the image's
`pip install -r requirements.txt` installed whatever was newest on the day it was built, and the
result was written down nowhere. Until this stage, "the deployed stack" existed only inside the
running container.

How fast that accident drifts, measured: the running image was built on 2026-09-22. Resolving
the same `requirements.txt` unpinned nine days later (2026-10-01, CPython 3.11.16, x86_64)
already differs from it on 42 of the 175 packages both contain, geopandas 1.1.4 → 1.2.0 and
pandas 3.0.5 → 3.0.6 among them, and adds a package the image does not have at all
(`opentelemetry-api`). A CI job that installed `requirements.txt` bare would have been testing
that stack, which nobody runs.

`constraints.txt` is now that record: the running `agent-api` container's `pip freeze`, 177
packages, taken 2026-10-01 from image `deeb331964f6` (built 2026-09-22; CPython 3.11.16,
x86_64), committed verbatim under a header that says how to retake it. Before committing it was
checked against the live container again, and the two freezes hash identically.

**The lock comes from the image, never from a development machine.** A lock frozen on the Mac
describes the Mac. `origin/backend_swap` has one, and it disagrees with the deployed image on 34
of the 40 packages the two share: pandas 2.2.3 against 3.0.5, numpy 2.1.3 against 2.4.6. Its
rasterio 1.5.0 pin needs Python 3.12 or later, so it could not install on any 3.11 build, and
nobody noticed until that branch's CI first ran (`51035b2`). The Mac also supplies pypdf,
python-docx and xarray from `~/.local` (`backend_swap` `4e8d327`), so a suite that is green
there says nothing about a build without them.

### Stage S18.2 CI installs through the lock

`.github/workflows/verify.yml` runs `python3 -m pytest rag_pipeline/tests/ -q` on
`ubuntu-latest` with CPython 3.11, after `pip install -r requirements.txt -c constraints.txt`. It
needs no secret, because `conftest.py` already replaces `load_dotenv` with a no-op (S9.7).

A constraint binds only what is requested. A requirement added after the freeze therefore
floats to latest-at-run-time while everything else stays pinned, and CI would test a mix that is
neither deployed nor latest without saying so. A step after the install prints every installed
`name==version` that the lock does not contain. It warns rather than fails, because until the
next deploy there is no deployed version to pin the newcomer to.

### Stage S18.3 What the first Linux run found

The first run (`36943761445`, on `ubuntu-24.04` with CPython 3.11.16, the deployed interpreter
exactly) installed all 175 packages at the deployed versions and reported:

| | passed | failed | skipped |
|---|---|---|---|
| development Mac, its own versions | 1645 | 0 | 4 |
| CI, the deployed versions | 1643 | 2 | 4 |

The four skips are the same opt-in live-service tests in both. Before the push, the workflow's
steps were run verbatim in a `python:3.11-slim` container on x86-64, which gave the same counts
once it had the system library described below.

**The Mac's baseline was measured on a stack that is not deployed.** Of ten version-sensitive
packages, eight differ: pandas 2.2.3 against 3.0.5, numpy 2.1.3 against 2.4.6, scipy 1.15.3
against 1.17.1, libpysal 4.15.0 against 4.14.1, and fiona is not installed there at all. One of
the two failures comes from that difference rather than from the platform, and it is the one
that matters in production:

* **pandas 3 parses years before 1677, so four-digit codes became dates.**
  `test_csv_with_coordinates_flows_through` expects `detect_time_column` to ignore `Beat`, a
  column of police beat numbers; on the deployed versions it lists it as a time candidate. Two
  things combine. GDAL reads a CSV's columns as text, so the guard in `_candidate_columns` that
  skips numeric columns without a time-like name never sees a number. And
  `pd.to_datetime(..., errors="coerce")` turns `"1234"` into NaT on pandas 2, whose nanosecond
  timestamps cannot reach before 1677, but into 1234-01-01 on pandas 3, which infers microsecond
  resolution. Ranking sorts by parse rate before the name hint, so the code column wins whenever
  the real date has a gap. Measured on the deployed versions: with one blank date in four rows,
  `detect_time_column` chooses `Beat` (parse rate 1.0) over `Date` (0.75); pandas 2.2.3 chooses
  `Date`. The deployed agent does this today. The fix is a separate change.
* **The distance band sits on a tie.** `test_distance_band_without_a_threshold_leaves_no_island`
  passes on macOS/arm64 and leaves one island on Linux x86-64 with identical libpysal 4.14.1,
  scipy 1.17.1 and numpy 2.4.6; it was reproduced inside the deployed container on 2026-10-01.
  `analysis_spatial_stats_tools.py` passes `min_threshold_distance` to `DistanceBand` exactly,
  and the margin on the test lattice is 0.0 m. The fix (pad the threshold by a relative 1e-9) is
  a separate change, and the test is deliberately not skipped in the meantime.
* **rasterio and fiona need a system library the slim base image lacks.** Their wheels bundle
  GDAL but link the system's `libexpat.so.1`, and `python:3.11-slim` has none, so
  `import rasterio` fails there. Under pytest 9 a module that is present but cannot load is a
  collection error, not a skip, so that session stopped at `test_raster_routing.py` with no test
  run. The deployed image has the library only as an automatic dependency of its GDAL and QGIS
  apt layers (`libgdal36`, `libqgis-core3.40.6`, the distro `python3.13-minimal`, among others).
  Dropping those layers looks safe, since the wheels bundle GDAL, and would leave rasterio and
  fiona unimportable. GitHub's runner has the library (both import there), so CI on the runner
  does not see this; a job built on the slim image would.
* **The missing extraction readers are invisible to the suite.** pypdf, python-docx and xarray
  are imported by `extractors/` and are absent from both `requirements.txt` and the image.
  Nothing fails, because the readers catch the ImportError and degrade quietly (empty text for a
  document, a "reader unavailable" note for a dataset), and no test in `rag_pipeline/tests` hands
  them a PDF, a .docx or a NetCDF file. A green run does not mean the deployment can read those
  formats. Declaring them is a separate change.

The workflow also imports every module the suite `importorskip`s before running it. That call
skips when a module is absent, so a package dropped from the install would turn whole spatial
modules into skips and leave the job green with far fewer tests.

### Stage S18.4 The images install through the lock

The first version of this stage left the image installing unpinned, because installing through
the lock changes what the next deploy installs. That decision is taken here.
`rag_pipeline/Dockerfile` (agent-api), `MCP_server/Dockerfile` and
`metadata-extraction-server/Dockerfile` now copy `constraints.txt` in beside `requirements.txt` and
pass `-c constraints.txt` to every `pip install`. A rebuild therefore reproduces the versions the
deployment runs. Before, each rebuild resolved that day's newest, and nine days after the
2026-09-22 build 42 of 175 packages had already moved (S18.1).

**What the next deploy installs changes in one direction: it stops upgrading.** On 2026-10-02 the
deployed agent-api container still matched the lock line for line, so the rebuild that ships this
change reinstalls the same 177 versions. Its pip layer reruns once, because the `COPY` above it
changed. Without this change, the next deploy that touches `requirements.txt` re-resolves the
whole stack. For PR #36's seven new packages that meant 42 moved packages; with it, the same
deploy adds the new packages and moves nothing else. The cost is that upstream fixes no longer
arrive by accident of build day. An upgrade is now a change to the lock, and CI tests it like any
other change.

**The spaCy model goes through the lock too.** `python -m spacy download en_core_web_sm` became
`pip install -c constraints.txt en_core_web_sm`. `spacy download` fetches spaCy's compatibility
table from GitHub at build time and installs whichever model version that table names for the
installed spaCy, with no lock and no hash check. A bare name constrained by the lock's direct-URL
line installs exactly that wheel instead, and pip checks the sha256 the URL carries. With one
digit of the hash changed, the image's pip 24.0 refused: `Expected sha256 0000… Got 1932…`. Today
both routes give the same wheel. The table lists only 3.8.0 for spaCy 3.8, and the wheel declares
no dependencies of its own. So the switch changes nothing now; it stops a future model release
from changing the image unannounced. py-spy likewise installs at its locked 0.4.2.

**MCP_server follows the same rule, because it installs the same file.** Its running container
was built in the same compose build as agent-api on 2026-09-22. Its `pip freeze` is the lock minus
exactly `en_core_web_sm` and `py-spy`: 175 packages, each at the locked version. So locking it
changes nothing it runs. It also keeps the two images on the same versions whenever either is
rebuilt, including the usual `up -d --build agent-api`, which rebuilds only one.
`metadata-extraction-server/Dockerfile` installs the same root `requirements.txt` and gets the
same change. It is not deployed: the VM has only a stopped container from 2026-06-12, and the
service sits outside compose's default profile.

**libexpat1 is named in all three apt layers.** `python:3.11-slim` for amd64 does not have the
package: not the tag cached here since 2026-09-19 (CPython 3.11.16, Debian 13.7), and not the one
the build pulled on 2026-10-02 (3.11.17). With the locked wheels installed on the first,
`import rasterio` and `import fiona` both fail with
`libexpat.so.1: cannot open shared object file`. The two libraries that need it are the GDALs
those wheels bundle,
`rasterio.libs/libgdal-c8c9c467.so.36.3.10.3` and `fiona.libs/libgdal-fiona-e8f6bdb0.so.35.3.9.2`.
`apt-get install libexpat1` alone fixes both. pyogrio 0.13.0 imports without it. The images keep
their GDAL and QGIS packages, which pull it in anyway, so nothing they contain changes. In the
deployed agent-api it was an automatic package that 16 others depend on, libgdal36,
libqgis-core3.40.6, python3.13-minimal and the mesa libraries among them. Naming it puts the
dependency where the next person to trim those layers will read it.

**How a new requirement is handled: it floats until the lock is retaken.** A constraint binds only
a name it lists. A package added to `requirements.txt` after the freeze resolves at build time to
its newest version that fits the pins, and so does any new dependency it brings. Nothing already
pinned moves to make room; if no version fits, the build fails with `ResolutionImpossible`. CI's
drift step (S18.2) names each floating package in a warning. After the deploy that ships it, the
lock is retaken from the running container with the command in its header. That pins the
newcomer and clears the warning. Otherwise the retake is a check: it should change nothing below
the marker, and any other difference means the image did not install what the lock says.

The newcomer is not pinned in the lock at once because its lines would be a guess. The lock's
lines come from an image's own `pip freeze`, and a newcomer usually brings dependencies of its
own: IPython brought twelve in PR #36. Writing that closure by hand predicts a resolution the
build performs anyway. Two things narrow the window instead:

* Pin the newcomer itself in `requirements.txt` when the version CI tests must be the version
  deployed, as PR #36 does. Then only its new dependencies float.
* To move a package the lock already lists, edit that one line in the same change. CI installs
  and tests it, and the retake after the deploy reproduces it.

**A package a Dockerfile installs by name must be in the lock already.** CI installs only what
`requirements.txt` asks for, so its drift step never sees py-spy or the model, and a floating one
would never be flagged. `rag_pipeline/tests/test_image_installs_through_lock.py` fails in that
case. It also fails for any `pip install` without `-c constraints.txt` or before the lock is
copied in, for `spacy download`, and for an image without `libexpat1` in an apt layer. It reads
the Dockerfiles as text and finds the images itself, so a new image is covered as soon as it
copies `requirements.txt`. Run against prototype's three Dockerfiles, it reports the five unlocked
installs (three in agent-api's) and the missing `libexpat1` in all three. Fourteen synthetic
Dockerfiles, and five for the apt reader, check that it rejects each wrong answer and accepts
each right one.

### Stage S18.5 Verified on local builds, not by a deploy

All three images were built from this change on the development Mac with
`docker build --platform linux/amd64 --pull`, emulated, on 2026-10-02. The agent-api build took
2,257 s, mcp-server 1,330 s and metadata-extraction-server 1,495 s, the three running at once.

| image | `pip freeze` inside it | against the deployment |
|---|---|---|
| agent-api | the lock's 177 lines, byte for byte, including `torch==2.14.0+cpu` and the model's URL line with its sha256 | identical to the running agent-api; the lock, the running container and the build hash alike (sha256 `ea8da91e6ce73d84…`) |
| mcp-server | 175 lines: the lock minus `en_core_web_sm` and `py-spy`, which it does not install | identical to the running mcp-server |
| metadata-extraction-server | the same 175 lines | not deployed |

Every line of each freeze is a line of the lock, so CI's drift step has nothing to report for
any of them; run inside the mcp-server image, it printed nothing. The model layer resolved the
bare name straight to the lock's URL. It read no compatibility table, installed 3.8.0 in 10 s,
and `spacy.load("en_core_web_sm")` works. `apt-mark` lists `libexpat1` as manually installed in
agent-api and metadata-extraction-server, and mcp-server names it on the same kind of install
line. In agent-api the same 16 packages depend on it as in the deployed container, and both
bundled GDALs resolve `libexpat.so.1` to `/lib/x86_64-linux-gnu/libexpat.so.1`.

The test suite was started inside the built agent-api image and did not finish. Partway through,
the Mac's disk filled and Docker Desktop stopped ("no space left on device"). The disk held
several sessions' amd64 images, these three among them, at 4 to 7 GB each. No result from that
run is recorded here. CI runs the suite on Linux under the same lock.

**The interpreter moved while the packages did not.** The build pulled CPython 3.11.17; the
deployed image runs 3.11.16, on Debian 13.6 rather than 13.7. The freeze is unaffected because
every compiled wheel here is built for cp311, not for a patch release. It is the float this stage
leaves, in S18.6.

### Stage S18.6 What this stage did not fix

* **The base image and the apt layers still float.** `FROM python:3.11-slim` is a moving tag, as
  S18.5 measured, and GDAL, QGIS and `docker-ce-cli` install whatever their archives serve on
  build day. None of that shows in `pip freeze`. Pinning the base by digest is a separate
  decision, with its own cost: security updates to the base stop arriving by themselves.
* **pip, setuptools and wheel come from the base image, not the lock.** `pip freeze` omits them,
  so the lock cannot pin them. Today they agree: 24.0, 79.0.1 and 0.46.3 in the deployed image and
  in both base tags above. torch requires `setuptools>=77.0.3`, which the base satisfies. A base
  with an older setuptools would let pip upgrade it unpinned and unreported.
* **The lock aligns versions, not the set of packages.** A requirement added later reaches only
  the images that are rebuilt. `up -d --build agent-api` leaves mcp-server without it.
* **Other images are outside this lock.** `embedding-server/` installs its own `requirements.txt`,
  and `sandbox/Dockerfile`, `Dockerfile.claude` and `Dockerfile.opencode` install bare names. The
  lock describes the root `requirements.txt` stack only.
* **CI tests the packages, not the image.** The runner is Ubuntu, not the image's Debian. It has
  no QGIS, no system GDAL, no spaCy model and no pre-downloaded embedding model, and its system
  libraries come from Ubuntu's base packages rather than the image's apt layers, which is how
  the libexpat dependency above stays hidden on it. The QGIS and Docker tests stub both out, so
  neither runs for real anywhere in CI.
* **The interpreter's patch version floats.** `setup-python` selects the newest 3.11.x it has.
  On the first run that was 3.11.16, the deployed version, but nothing holds it there; the
  install step prints which one it got.

> **Landed after the fixes it found.** Both failures recorded above were fixed before this stage merged: the island under the default distance band is [Stage 14](#stage-14), and `Beat` read as a date on pandas 3 is [Stage 15](#stage-15). Merged in that order on 2026-10-03, so the workflow's first run on `prototype` is against code that already contains both.

---

## Stage 19 — What runs in the agent's own process {#stage-19}

Every tool a peer is offered runs inside `agent-api`, and `docker-compose.yml` runs that container
as `user: root` with `/var/run/docker.sock` mounted: code executing there has root-equivalent
control of the host's Docker daemon. The sandbox in `agent_runtime/code_execution.py` (a fresh
container per run, `--network none`, read-only root filesystem, capabilities dropped) exists so
that untrusted code never runs there. One tool went around it.

### Stage S19.1 A tool that ran knowledge-base code with no sandbox

`kb_run_geofunction` (`extractors/geo_handles.py`) loaded a block with `get_kb_block(doc_id)`, cut
one function out of the block's stored source with `ast`, and `exec()`'d it in the agent process.
Block source is notebook code submitted to the I-GUIDE platform: third-party code. It was bound to
the analysis peer **unconditionally** — `default_analyze_fn` extended its tools with
`make_geo_analysis_tools()` inside a bare `try/except`, with no flag and no upload gate.

It was latent, not live. Verified 2026-10-01 in the deployed container, read-only: the model was
offered the tool, but the deployed knowledge base is the `local` backend and empty
(`agent_kb_search("flood inundation")` returned `count: 0`), so `get_kb_block` found nothing and
the `exec` was never reached. What made it urgent is the next step on the extraction branch
(`claude/extraction-integration`, "B1: one shared store"): pointing `get_kb_block` at the populated
Postgres record would have armed this `exec()` with that record's 3,830 notebook cells (the
extraction session's count) in the same commit. That branch held B1 until this landed.

| analysis-peer configuration | tools offered before | after | tools reaching an in-process `exec` |
|---|---|---|---|
| nothing attached | 38 | 37 | 1 → 0 |
| a file attached | 72 | 71 | 1 → 0 |
| unified search+analyze peer | 50 | 49 | 1 → 0 |

The search and code peers never had it; their lists (25 tools, and 35 or 65 for code) are
unchanged. Why it was bound in the first place is **not recorded**: the function and its binding
arrived in `af1ead0` ("Fix repeated search with no result"), whose message describes none of it.
It is the same commit the known-gaps list names as holding the docker-out-of-docker work.

### Stage S19.2 Dropped, not re-routed

There were two fixes: re-route the call through the sandbox, or stop offering it. It was dropped.
The extraction plan already reserves the sandbox as the execution path for generated tools, later
and behind its `AGENT_EXTRACTION` flag (its B7), so re-routing now would build that twice. The
route models are taught to use is `get_kb_block` + `execute_code`: `CODE_PEER_PROMPT` and the
`chicago-crime-analysis` skill say so, and no prompt ever named `kb_run_geofunction`.

What dropping it costs, stated now rather than discovered later:

* **The default analysis peer can no longer run KB code at all.** It has no `get_kb_block` — only
  the code peer and the unified peer do — so this tool was its one way to load a block. Reusing KB
  code is now the code peer's job, which is where the prompt that teaches it lives;
  `ANALYSIS_WORKFLOW_PROMPT` never mentioned KB code.
* **A KB loader that fetches from a URL loses its only working route in the deployment.**
  In-process code has the network and the sandbox does not, so such a loader could run through
  this tool and cannot run in `execute_code`. `CODE_PEER_PROMPT` already covers the case: when the
  data is not attached and no tool returns it as a file, the peer says so and asks for the file.

What changed:

* `make_geo_analysis_tools()` returns three tools — `kb_select_rows`, `heatmap_image`,
  `choropleth_image` — none of which executes stored source. `supervisor/graph.py` and
  `capabilities.py` are untouched, so the extraction branch's edits there do not collide.
* The function stays, for the hand-run `extractors/examples/geo_mixed_chain_demo.py` and as a
  starting point for B7, but it now **refuses** unless `AGENT_CODE_EXEC_BACKEND=local`, the switch
  that already means "run untrusted code on this host, unsandboxed". The deployment pins `docker`,
  so even a direct call fails closed before any block is read. It also left `__all__`.
* The capability atlas (`docs/spatial-toolkit.html`) lists 57 tools, not 58. Its standfirst said
  fifty-one while its footer said 58; the standfirst, strip and footer now all say 57.

### Stage S19.3 The guard tests the class, not the name

`rag_pipeline/tests/test_no_peer_execs_stored_code.py` captures the FINAL tool list each peer
hands to `create_agent` — search, analyze and code; with and without a file attached; unified;
and with MCP tools forced onto the local-import fallback — and walks each tool's code
transitively (module globals, function-local imports, closures, `functools.wraps` chains, nested
functions) for `exec`/`eval` or an importlib/runpy runner. A name check would only stop this
function coming back; this also fails for the next tool built the same way. Positive controls pin
what the walk sees, so an interpreter change that renamed an opcode cannot turn it into a silent
pass. Against the original `geo_handles.py` it fails 13 of its 42 tests; on this branch all 42
pass.

Its blind spot is an attribute call on an object it cannot type statically (`obj.run(src)`).
Running code in another process is out of scope on purpose: that is how `execute_code` reaches
its sandbox.

### Stage S19.4 What this stage did not fix

The guard found a second path on its first run. With MCP tools on and no module list — the API's
default for the search peer — and the MCP server unreachable when tools are built,
`make_langchain_mcp_tools` falls back, with nothing but a log warning, to importing the MCP tool
modules into `agent-api` (10 tools). `mcp_create_notebook_workflow_tool` is one of them: it
registers a generated tool whose body `exec()`s notebook-derived source, and the next tool build
binds that tool in-process.
Normally the same code runs in the `mcp-server` container, which has no Docker socket. The test
records it as a known gap that fails the moment the gap closes, rather than fixing it here: the
notebook workflow builder is live work, and whether to unbind it is not this change's call.

Still true: `agent-api` runs as root with the Docker socket. Every in-process tool inherits that
blast radius. This stage removes one way in, not the exposure.

### Stage S19.5 The MCP fallback stops importing tools that exec

S19.4's gap, closed on 2026-10-02. The local-import fallback in `make_langchain_mcp_tools` now
refuses three modules: `notebook_workflow_tools`, `generated_notebook_tools` and
`generic_executor_tools`. It refuses them whether they come from `DEFAULT_MCP_MODULES` (the first
two were on it) or from a request's `mcpModules`. Their tools still exist on the MCP server, and
the agent still binds them from there. Only the copies that ran inside `agent-api` are gone.

The path was reproduced before the fix. With the remote server forced unreachable, a one-cell
notebook wrote the id of the process that ran it to a file. The fallback bound
`mcp_create_notebook_workflow_tool`. Calling it registered `mcp_probe_nb` in the module globals
of `generated_notebook_tools`. The next build bound that tool (the 60 s cache TTL was cleared in
the probe), and calling it ran the notebook in the agent's own process: the id written was the
probe's own. The reproduction showed two things S19.4 had not said:

* The registered tool lives in module state. Every later request in that worker would have been
  offered it, whoever uploaded the notebook.
* `mcpModules` reaches the analysis peer's fallback unchanged, so that peer was exposed too, and
  `generic_executor_tools` could be named into either peer. Its `AGENT_ALLOW_WORKFLOW_EXEC` gate
  is an environment variable, not a sandbox.

| local-fallback configuration | tools offered before → after | MCP tools | tools reaching an in-process `exec` |
|---|---|---|---|
| search peer, MCP on, no module list (the API default) | 35 → 33 | 10 → 8 | 1 → 0 |
| search peer, `mcpModules` naming every MCP module | 39 → 35 | 14 → 10 | 3 → 0 |
| analysis peer, `mcpModules` naming every MCP module | 48 → 44 | 14 → 10 | 3 → 0 |

The other six configurations the guard captures are unchanged. The gap was latent in the
deployment. Its journal from 2026-09-22 to 2026-10-02 15:15 UTC records 35 remote MCP builds of
14 tools each and not one fallback build.

This fix was chosen over the two alternatives:

* **Unbinding by name** (`_DEFAULT_UNBOUND_MCP_TOOLS`) would also drop the remote builder. That
  builder is live work, and it runs in `mcp-server`, not here. A name list also cannot hold a
  generated tool in advance, because each one is named after its notebook. And
  `AGENT_MCP_UNBIND` replaces the default list wholesale, so `AGENT_MCP_UNBIND=none` would have
  bound everything back.
* **Routing generated tools through the sandbox** is where notebook-derived code should
  eventually run. The extraction plan already reserves that for generated tools (its B7). It does
  not fix the fallback, though. The same code runs in `mcp-server`, which has no Docker socket to
  reach a sandbox through.

The set is fixed in code and is not a setting, because an override is exactly what would bring
S19.4's path back. Whether to unbind the remote builder as well was put to the maintainer and
declined. What this costs: with the MCP server unreachable, the agent has no
`mcp_create_notebook_workflow_tool` or `mcp_list_generated_notebook_workflow_tools`, and a request
naming `generic_executor_tools` gets nothing from it. With the server reachable, which covers
every build in that journal, nothing changes.

Changes to the guard:

* **The gap left `KNOWN_IN_PROCESS_PATHS`.** On the fixed code, `test_the_known_gaps_are_still_real`
  failed until the entry was deleted, which is the exit S19.4 built in for it.
* **Two configurations were added.** A request naming every module under `MCP_server/tools` is
  now captured for the search and analysis peers. The path through a named module was outside
  what the guard captured before.
* **A provenance check was added.** No peer is offered code defined in a skipped module. It holds
  where the sink scan is blind, and it covers a generated tool by module, not by name.
* **Each skipped module has a witness.** One tool per module shows that it reaches `exec`. This
  keeps the list from growing into a general switch, and from keeping a module whose `exec` has
  moved into the sandbox.

Mutation-checked:

* Removing the request-name filter fails 4 tests.
* Restoring the original behaviour (filter off, both modules back on the default list) fails 6:
  the scan and the provenance check, each for the API default and for both request-named
  configurations.
* Putting the two modules back on the default list, with the filter in place, still passes,
  because the filter alone holds.
* An unexplained module on the list fails the witness test.

The file has 58 tests, up from 42.

Found, and not fixed here:

* **Scoping still imports named modules.** When the remote server answers, a request's
  `mcpModules` still makes the agent import the named modules (`_allowed_remote_tool_names`) to
  learn which remote tools to keep. Nothing imported there is bound or called in this process.
* **`mcp-server` is unchanged.** It runs the same tools, loads `.env` and mounts the shared file
  volume. S19.4's path is closed for `agent-api` only.
* **`AGENT_MCP_UNBIND` cannot unbind some tools.** It fails for any tool whose name begins with
  `c`, `m`, `p` or `_`, which includes this builder. `lstrip("mcp_")` strips characters, not a
  prefix, so `create_notebook_workflow_tool` becomes `reate_notebook_workflow_tool` and matches
  nothing. The default list never passes through that code, and neither of its two names begins
  with one of those letters, so only an operator's override is affected.
