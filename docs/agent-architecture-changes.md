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
