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
| 2 | [The agent becomes a package](#stage-2) | 2026-02 → 2026-05 | a 2048-line monolith split into `agent_runtime/` |
| 3 | [Supervisor over peers](#stage-3) | 2026-06 → 2026-07 | nesting replaced by peers sharing typed state |
| 4 | [Map-native delivery](#stage-4) | 2026-08 | one boundary every layer crosses; prompts stop issuing mandates |
| 5 | [The action ledger](#stage-5) | 2026-09 → `9e35950` | the agent records what tools *did*, not that they ran |
| 6 | [What the decider reads](#stage-6) | `claude/evidence-summary` | evidence described, capabilities generated, the ledger shared |
| 7 | [Who the caller is](#stage-7) | `claude/jwt-identity` | identity, ownership, server-owned history |

Stages 6 and 7 are **unmerged branches**, independent of each other. Deployment state is at the
end; do not infer it from the commits.

One thread runs through all eight: **the system repeatedly discovers that a component was
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
| `4b9eae8` | first rerank and hallucination-audit prompts, including *"Scores MUST show meaningful variance"* and a JSON verdict schema | reason not recorded; the schema shape becomes a measured defect in stage 4 |

### 1.6 Worth knowing

`api_server.py` was **unimportable for eight days** (2026-01-14 → 01-22). A commit titled
*"Revert to see the references"* left a literal `\n+` on line 20 — a module-level `SyntaxError` —
silently repaired later by an unrelated commit.

`f0cd457` raised `top_k` from 8 to 100 with no recorded reason. Given `merge_retrieval`'s
cumulative cap, a budget of 8 consumed by keyword hits would leave opengeodata zero slots — that
is a reading of the code, not a claim the commit makes.

---

## Stage 2 — The agent becomes a package {#stage-2}

*2026-02 → 2026-05. 79 commits.*

### 2.1 The monolith

`rag_pipeline/langchain_agent_executor.py` is created at `2cfcdf8` (266 lines) and grows
monotonically: 503 (`c22d79f`, MCP bridge) → 893 (`37b7f6a`, CodeAgent + intent classification)
→ 1231 (`deefd45`, LangGraph `StateGraph` + checkpointer) → 1672 (`cd031f3`) → **2048**
(`f3533e1`).

`f3533e1` is an explicit failure record kept on the mainline: *"Try to add an orchestrator agent…
These features are not workin as expected."* It is the peak that triggers the extraction.

### 2.2 Three agent designs in five weeks

- `37b7f6a` — keyword-hint intent classification picks an intent, a tool filter narrows the list,
  one executor runs.
- `deefd45` — a LangGraph `StateGraph`: `initialize → route → search → extract → analysis →
  finalize`, with `InMemorySaver` keyed on `thread_id`. `8c502d6` flattens it so search and
  analysis are siblings, with child threads `{thread}::search` / `{thread}::analysis` so the two
  stop sharing checkpoint state.
- `cd031f3` — **the StateGraph is removed entirely**. An orchestrator LLM is given the other
  agents *as tools* (`search_agent_evidence`, `analysis_agent_answer`, `code_agent_answer`). The
  route is reconstructed post-hoc from which tools were called. This is the "agents-as-tools"
  shape that stage 3 replaces.

### 2.3 The extraction, done twice, independently

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

### 2.4 Around the core

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

### 2.5 Prompt revisions in this stage

| commit | change | reason |
|---|---|---|
| `2cfcdf8` | the origin prompt: *"You are a retrieval-grounded assistant"* with *"Cite only doc_ids that appear in the tool response"* | the anti-fabrication core; survives the whole history |
| `37b7f6a` | split into `SEARCH_AGENT_PROMPT` / `ANALYSIS_AGENT_PROMPT` / `CODE_AGENT_PROMPT` | one role per prompt |
| `cd031f3` | SearchAgent gains *"Do not infer local file paths or use file tools unless the user explicitly provided attached/uploaded files"* | the agent was inventing file paths |
| `ad6361b` | the router's `graph` hint expands into a trigger list with worked examples | the LLM router was not enabling graph search for entity queries |
| `ad6361b` | `_CYPHER_SYSTEM_PROMPT`: *"READ-ONLY… Never use MERGE, CREATE, DELETE"*, with a regex sanitizer and a LIMIT injector | few-shot examples record specific observed failures, e.g. *"Do NOT use label union syntax like (r:A\|B\|C)"* |
| `7f71a90` | *"Never call `load_skill` twice in the same assistant turn"* across four prompts, enforced in code by per-run loaded sets | a code comment names it: *"Some models pass the skill directory as `resource_path` after the main skill is already loaded"* |
| `8dc7f25` | *"do not fake binary files with `write_output_file`"* | the model had been writing fabricated binaries through the text file tool |

### 2.6 A taxonomy that was never consumed

`722e4ed` adds `@mcp_tool(category=...)` validated at decoration time against six categories, and
says it is *"replacing the hardcoded tool-name sets in graph_state.py (consumption comes in the
next commit)"*. **That consumption never lands.** At the end of the era `tool_policy` still
switches on the name sets, and no commit in the range touches it with the string `category`. The
taxonomy is metadata-only.

### 2.7 A tool unreachable for three weeks

`9c45d82` is a one-line-per-call fix: `neo4j_search_tool` was calling the tier-3 function instead
of the 3-tier dispatcher added in `ad6361b`, so the hierarchy had been unreachable from the agent
since it was written.

---

## Stage 3 — Supervisor over peers {#stage-3}

*2026-06 → 2026-07. 42 commits. This is the pivot of the project.*

### 3.1 The model changes (`7adf7d1`, then `fb8bdfa` one day later)

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

### 3.2 Bounding the loop (`af1ead0`)

`AGENT_SUPERVISOR_MAX_SEARCHES` (2), `AGENT_SUPERVISOR_MAX_PEER_RUNS` (3), `_search_exhausted`,
`_is_unproductive_repeat` — which guards `analyze`/`code` but **deliberately not `search`**,
because search accumulates into evidence so a follow-up can add documents.

Same commit makes the grounding audit non-cosmetic: *"a flagged verdict changes the text the user
actually sees, rather than being computed and discarded."*

### 3.3 Deterministic short-circuits (`4624187`, `79cb450`, `e31272c`)

Three commits convert LLM tool-choice into code paths, all from one root cause: *"nothing steered
the SearchAgent to the wired `neo4j_explore_related_nodes` tool, so it fell back to
`semantic_search`"*. The symptom was a related-elements query *"returning a generic semantic
search of topically-similar papers presented as if they were curated relationships — which the
grounding audit correctly flagged HIGH."*

`e31272c` adds a subtlety worth keeping: recalling an element from conversation must be
**role-aware**, because *"a prior ANSWER embeds other elements' UUIDs in its citation URLs"*, so a
naive newest-first scan would recall a *cited* element instead of the user's subject —
reintroducing the bug *"in a form the grounding auditor can't catch."*

### 3.4 The packages split (`665db95`)

The two orchestrators move into `agent_runtime/supervisor/` and `agent_runtime/legacy/`, behind a
strategy registry, *"so neither's code/prompts can leak into the other"*. The packages never
import each other. Agents-as-tools survives only as an `AGENT_SUPERVISOR=0` fallback.

### 3.5 Prompt revisions in this stage

| commit | change | reason |
|---|---|---|
| `fb8bdfa` | `analyze` redefined from *"compose an answer"* to *"run a GIS/data analysis workflow"*; `done` becomes *"a grounded final answer is composed automatically"* | stop the decider treating analyze as the answer-writer now that `synthesize` exists |
| `af1ead0` | *"Each peer ITERATES INTERNALLY… do NOT pick it again to 'retry' or 'improve'"* | the observed loop the bounds also address |
| `665db95` | `SYNTHESIS_PROMPT` replaces the reused legacy prompt | the legacy one is *"a tool-calling AnalysisAgent persona whose rule 7 — 'call `code_agent_answer`' — is contradictory here"* |
| `b060d1a` | evidence rendering, not the prompt, is changed to show only `title:`/`url:` | rule 2 alone was insufficient: `_format_documents` *"still led each evidence item with `[<doc_id>] title`, which trained the synthesis LLM (esp. the small default model) to cite `[<uuid>]`" |
| `6e48d65` | the audit prompt: *"the execution record is FIRST-CLASS grounding"* | an answer presenting a real computed result was being flagged for lacking a document |
| `0dc93e8` | the audit gains a precision section: flag only contradictions and checkable specifics | *"A correct answer that adds non-contentious domain framing… was being flagged medium and surfacing a scary '⚠️ Grounding check' caveat"* |

### 3.6 A second code-peer runtime (`4758ea2`)

`AGENT_CODE_PEER=opencode` swaps the whole LangChain code peer for a container-per-run CLI. The
differences are deliberate and recorded: the container **keeps network access** (the CLI must
reach its LLM endpoint), unlike the `execute_code` sandbox which is `--network none`. This is the
boundary that still makes `analyze` and `code` genuinely distinct peers — not their toolsets,
which overlap almost entirely.

---

## Stage 4 — Map-native delivery {#stage-4}

*2026-08. 137 commits — the busiest month. Only 10 of them came through PRs; ~126 landed directly
on `prototype`, so the PR titles are not a useful index.*

There is a hard 12-day gap (Aug 6 → Aug 18) with no recorded reason, and the work either side is
qualitatively different. It is the real seam in the month.

### 4.1 The delivery contract (`30cae40`, `3eaaa2e`, `261772a`, `0456bf2`)

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

### 4.2 Prompt philosophy inverted — mandates become capability statements

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

### 4.3 The grounding audit, rewritten from measurement (`11490d6`)

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

### 4.4 The web, behind an SSRF guard (`f16ae4b`, `e0741a6`, `aa7105b`)

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

### 4.5 The spatial toolkit, and an absence that produced a false answer

`7cb9f47` adds seven PySAL/GeoDa tools with a stated engine policy: *"One engine per job so the
model is never choosing between two ways to compute the same number."*

`2212f8a` adds `select_by_attribute`, and its reason is the sharpest argument in the history for
minding gaps rather than only bugs:

> *"the analyze peer had 21 spatial tools and no way to isolate one feature… So it buffered all
> 708 grid cells — 4,504 overlapping polygons covering the whole city — and reported 'a 2 km
> buffer around the busiest grid cell'. Attribute selection is the plainest GIS operation there
> is, and its absence produced a false answer."*

### 4.6 Remote-sensing embeddings (`28832b5`, `b44a202`, `9b1001b`)

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

### 4.7 Model provider becomes a per-request choice

`81f6e9e` adds `GET /agent/models` and per-request model/provider/effort. `34d1fb5` replaces a
prefix heuristic with a **measured per-model table** after picking `gpt-5.6-luna` made every later
turn fail and, because the choice persisted to localStorage, *"the setting was unrecoverable from
the UI."* Key row: *"gpt-5.6-\* REFUSES tools unless reasoning_effort='none' is sent."*

Two commits **retract their own earlier claims** after re-measuring — `cd1adae` (*"I said twice
that CLAUDE_CODE_OAUTH_TOKEN 'cannot call the Messages API'… It is wrong"*) and `c0d5617`. And
`6198e7b` removes a liveness probe added an hour earlier, with the measurement that killed it:
*"Within the same minute on this credential, claude-haiku-4-5 answered with tool calls while
claude-sonnet-5 and claude-opus-5 both returned 429. The limits are PER MODEL."*

### 4.8 The first move toward collapsing the peers (`299e35d`, `3b7e181`, `daf5862`)

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

## Stage 5 — The action ledger {#stage-5}

*2026-09-01 → `9e35950` (2026-09-10). 106 commits.*

The stage's centre is one idea: **record what tools DID, not that they ran.** Eight commits build
it, and everything else in the stage either feeds it or reads it.

The justification is stated in `9e1201e` and proved by `7f0888f`, whose self-assessment is the
most useful sentence in the history:

> *"I diagnosed these two lines twice from the trace alone and was wrong both times… Two Earth
> Engine sweeps and two wrong commits, because the trace shows that a tool was called and never
> what it returned."*

### 5.1 Building it

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

### 5.2 One authority for "is it on the map?" (`8f9f24a`)

Four signals answered that question, combined with `or`, *"so the weakest won"*: a tool name in
`tool_calls` (never checking success), a bare nested `"on_map": true`, a recursive name match, and
a regex over the JSON blob. A **failed** `admin_boundary` tripped two of them; the supervisor then
*"suppressed its own corrective retry, wrote the conclusion into `result["on_map"]`, and RE-READ
that conclusion as evidence a layer existed."*

`map_layers.delivers_map_layer` becomes the single authority. Found in passing: `vector_spatial_join`
set `on_map` with no descriptor, *"so nothing has ever reached the map from it."*

### 5.3 Per-turn scoping (`2be2b83`)

Peer threads outlived the turn, so four verifiers asking *"what happened THIS turn?"* got the
whole conversation: *"last turn's `execute_code` made this turn's bare code fence report
executed=True"*; *"turn 1's documents came back as turn 2's evidence."* `PeerSession` scopes per
**invocation**, not per turn, because `default_analyze_fn` invokes the same thread three times and
concatenated slices made *"two failures of one tool render as four."*

### 5.4 The audit becomes a gate (`05fa222`, `e9801dd`, `76df5b9`, `29f24c2`)

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

### 5.5 Capability introspection becomes discovery (`6ba1bd3` … `cd083ad`)

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

### 5.6 Context budget from measurement, not assumption

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

### 5.7 Layer identity becomes a content digest

Seven commits over two days. `349a6f7` → labels carry a region tag; `895fb4a` → `slug[:40]` was
the id, and *"a one-character margin decided which region kept its layer"*; `3c73f3b` → the id
becomes a digest of everything that decides **content** and nothing else, with the caller's `name`
deliberately excluded (*"Renaming a layer now leaves it the same layer"*).

Two adversarial reviews found **twelve** and **fifteen** confirmed defects respectively, including
one the change itself introduced. `895fb4a` also records that the author's own earlier commit
caused the bug it fixes.

### 5.8 Terrain, and two misregistrations found by arithmetic

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

### 5.9 File-store session scoping (`9b83546`, `0fa5d25`, `db843f8`)

*"The file record had seven fields and none of them said who wrote it, so the store was one flat
space shared by every session."* A session id is stamped at creation and bound at the request edge
as a ContextVar — *"the same shape the streaming trace state uses… When JWT arrives it changes
where that id comes from, not what is done with it."* That sentence is the plan stage 7 executes.

`0fa5d25` is the follow-up worth remembering: **ContextVars do not cross threads**, and the agent
runs on a worker. *"Every unit test passed and the live path still wrote session=None, because the
tests all ran on one thread."*

`db843f8`: registering `list_conversation_files` was not enough — the file toolset attaches only
on an upload turn, so *"the analyse peer had no file tool, wrote `os.listdir('.')` in
`execute_code`, listed the sandbox working directory, and reported its own scratch script as one
of the artifacts."*

### 5.10 Prompt revisions in this stage

| commit | change | reason |
|---|---|---|
| `3d9587f` | `SYNTHESIS_PROMPT` rule 8: earlier-turn tool records are *"first-class grounding"*; *"A line marked FAILED means the tool did not work"* | worded **conditionally on purpose** — `default_compose_fn` reuses the prompt without that section |
| `9e33911` | the capability prompt gains the question: *"A user has asked what you can do. Answer THEIR question"* | it had been composing an identical grouped catalogue for every capability question |
| `05fa222` | `_REGROUND_DIRECTIVE`: *"downloading or inspecting a file is not the same as computing the answer"* | names the observed gazetteer trap |
| `5f828f4` → `4fcefd7` | two successive rewrites removing instructions the sandbox cannot satisfy — *"REUSE it verbatim — including real data-loading URLs/APIs"* under `--network none`, then its replacement naming `web_fetch`, which *"returns a page's on-topic passages, not the bytes of a dataset"* | a prompt that names an unavailable route teaches a failing habit |
| `ae813f6` | *"UNSURE OF AN API? Look it up before you write against it… a lookup is something you do BEFORE execute_code"* | *"a bound tool the prompt never mentions does not get used"* |
| `d1b171a` | `"Routed to {route}"` becomes a name table; the orchestrator label is renamed on the supervisor arm only | `fast`/`orchestrate` *"are node names in THIS graph"*, not facts about the request; legacy keeps the old label *"because there it is accurate"* |

---

## Stage 6 — What the decider reads {#stage-6}

*Branch `claude/evidence-summary`, 7 commits off `9e35950`. Unmerged.*

Three things the supervisor chose from turned out to be wrong the same way: each was **cheap and
available** rather than **what the decision needed**.

| | before | after |
|---|---|---|
| what the evidence is | counts, titles, `topical_coverage`, `top_score` — all lexical | plus `evidence_summary`, written by the model that read the documents |
| what peers can do | a hand-written paragraph, derived from nothing | generated from `capability_registry`, held against the peer builders by test |
| what this turn did | booleans and counts; the ledger of *previous* turns only | the same `this_turn` lines the answerer and auditor already read |

### 6.1 Evidence gains a description (`1044afb`)

Lexical signals cannot separate PySAL accessibility notebooks from DEM sources when both mention
"elevation". Measured on a self-hosted model: two full search rounds where the second added
nothing, because *"is this enough?"* was being answered from counts.

Two rules keep it from making things worse: it **describes and names gaps, it does not rule on
sufficiency** — that judgement belongs to the decider, and a summary announcing "this is enough"
would collapse two independent checks into one — and it sits **beside** the lexical signals rather
than replacing them, so a wrong summary can be disagreed with.

### 6.2 The capability paragraph becomes generated (`99d71ad`)

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

### 6.3 The decider joins the ledger's readers (`25c3e1e`)

Asked for a DEM, the supervisor ran `analyze` — which fetched and drew it — then routed to `code`,
which fetched the same DEM again. In a sweep that second pass cost **266 s and 16 `execute_code`
iterations** to redo work one tool call had done.

The rows were never missing. `_ledger_lines` had exactly **two** consumers — the answering model
and the grounding auditor — and the decider was not one of them. This is the same fix stage 5's
`29f24c2` made for the auditor, which had the identical blind spot.

**Revised during the work:** this began as a `map_layer_delivered` boolean. The boolean was kept
for the one question the prompt asks directly, but it was treating the symptom; the ledger is the
structural answer, and it is richer, already budgeted, and shared with the other two readers.

### 6.4 The tool surface absorbs how models actually call tools

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

## Stage 7 — Who the caller is {#stage-7}

*Branch `claude/jwt-identity`, 10 commits off `9e35950`. Unmerged, independent of stage 6.*

### 7.1 Named deployment modes (`152a537`, `f9b7081`)

`AGENT_MODE=dev|demo|token` replaces three booleans whose eight combinations included five
nonsensical ones (*"settings hidden AND a key required"* is a page demanding a credential it gives
you no way to enter). `PLATFORM_TIER=dev|prod` does the same for four platform URLs that must
agree. Both raise on an unknown value rather than falling back: this selects security behaviour,
and a typo silently resolving to a working mode is the failure nobody notices.

Deliberately, **the mode does not decide the API key** — that would make `AGENT_MODE=dev` mean one
thing on a laptop and something else on the public dev tier.

### 7.2 Identity (`69863b8`, `95a48de`, `c59a69d`)

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

### 7.3 Ownership (`d934cbe`, `e067055`, `1f880d3`)

`GET /agent/files/<id>/download` had never checked anything, and every answer publishes file ids
as links — so each was effectively a permanent public URL. A mismatch answers **404, not 403**,
because a 403 confirms the id exists and makes the endpoint an enumeration oracle.

`get_or_create_memory` fetched by bare UUID with no owner check; worse, the create half would have
**indexed over** a document it had just refused to read. That is why ownership is asserted at the
edge rather than guarded at each call.

`1f880d3` stores the client's own `StoredSession` rather than rebuilding it server-side, because
`sessionStore.ts` was written server-shaped on purpose and rebuilding would duplicate its
layer-descriptor rules in a second place where they would drift.

### 7.4 The browser stops owning the session (`84c60b2`, `4daf2e1`)

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

---

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
- **A capability taxonomy was added and never consumed** (`722e4ed`, stage 2.6).
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
