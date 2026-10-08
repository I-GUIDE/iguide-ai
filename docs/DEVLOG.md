# Development log — extraction restructure → deployable beta

One entry per step, appended in the **same commit** as the change it describes.
Plan: `~/.claude/plans/based-on-our-discussion-lazy-cray.md`.

Entry format:

```markdown
## YYYY-MM-DD · <milestone id> · <short title>
**Change** what changed, with file:line
**Why** what was wrong before — the defect, not the intention
**Measured** the number that moved: before -> after
**Surprised by** anything the change revealed that we did not expect (omit if nothing)
**Next** what this unblocks or what it forces
```

`Measured` makes each milestone's exit criterion auditable. `Surprised by` is where
findings get captured before they are forgotten — it is also the raw material for the
SIGSPATIAL methods and results sections.

Every LLM-touching entry records the model used. A recall number produced under `opus`
is not comparable to one produced under a self-hosted 7B.

---

## 2026-08-07 · M0.1 · Baseline measured before any change

**Change** Added this log. No code touched.

**Why** The plan's exit criteria are all expressed as numbers that must move. Those
numbers are worthless without a recorded starting point, and several of them were
measured during planning against live infrastructure that will drift.

**Measured** — baseline, all verified directly rather than inferred:

*Corpus and retrieval*
- `new-opensearch-index`: **619 docs** — publication 180, map 160, notebook 133,
  dataset 87, oer 32, code 27.
- Live platform catalog (`backend.i-guide.io/api/elements`): **750 elements**. The
  index is therefore **131 elements (17%) stale**. `iguide-feb12` exists on the
  cluster and is fresher.
- BM25 recall over the benchmark's **full** expected sets (37 ids across 11 tasks),
  each task's verbatim prompt as the query:
  | k | recall |
  |---|---|
  | 8 (current effective window) | **22/37 (59%)** |
  | 20 | **29/37 (78%)** |
  | 50 | 33/37 (89%) |
  | 100 | 34/37 (92%) |
- Three expected elements are unreachable by BM25 at **any** k — `afbee4bd` (T6,
  Open-Meteo notebook), `643aaea1` (T9), `de05a428` (T10). These are genuine indexing
  gaps, not window problems.
- Per-task at k=8 → k=20: T8 **3/7 → 7/7**, T7 1/3 → 2/3, T5 2/3 → 3/3, T1 2/3 → 2/3.
- Semantic search was **not** exercised: `FLASK_EMBEDDING_URL` in `.env` points at
  `149.165.159.254:5000`, which refuses connections. All figures above are BM25 only,
  so the real ceiling is higher.

*Agent-side*
- `iguide_agent_*` indices on the live cluster: **0 of 4 exist**.
- Agent KB: **290 blocks from 14 of 133 notebooks (10.5%)**, in a single local JSON at
  `agent_chat_files/agent_kb/iguide_agent_notebook_blocks.json`.
- Publications with reachable full text (`pdf_chunks`): **34 of 180**, and **no search
  tool queries them**.
- `opengeodata_search` calls across 11 recorded eval runs: **0**.
- MCP tools reaching a model in a default supervisor run: **4** of 16 native
  (`default_search_fn` defaults `include_mcp_tools=False` at `supervisor/graph.py:1428`;
  `default_analyze_fn` scopes to `spatial_analysis_tools` at `:1658`).

*Quality gates*
- Tests importing anything from `extractors/` or the emitters: **0**, across ~3,250 LOC.
- `_AUDIT_FLAG_SEVERITIES = {"high"}` (`supervisor/graph.py:418`); severities emitted
  across the 11 committed eval records are `low` ×4 and `none` ×40 — so the user-visible
  caveat never fired **on that corpus**, and "0 hallucinations in 21 runs" is the null
  output of a detector that never triggered there.
  > ⚠️ **REFINED (see M0.2b):** this must not be read as "the caveat cannot fire." Driving
  > the prototype live produced a `severity: high` caveat on the very first substantive
  > query — and it was a **false positive**. Scope the claim to the recorded corpus.
- `retrieval_success` is pinned at **10/10** by construction: `scripts/eval_common.py`
  scores it against `TASK_META[tid]["primary_ids"]` (a subset) while computing
  `full_exp` on the line above.
- Dev/prod dependency drift: dev runs langchain **1.2.10** / core **1.4.9** / langgraph
  **1.0.10**; `pip install --dry-run -r requirements.txt` resolves to
  **1.3.14 / 1.5.3 / 1.2.10**. The unbounded `>=1.0` pins are real and confirmed by the
  resolver report.
  > ⚠️ **RETRACTED (see M0.5):** this entry originally claimed "repo suite against the
  > prod-resolved set: 550 passed, 9 failed". That measurement was invalid — the venv it
  > ran in had no `pyvenv.cfg`, `sys.prefix` was `/opt/anaconda3`, and its interpreter
  > resolved langchain from `~/.local` at **1.2.10**. It measured the dev stack, not the
  > prod stack. M0.5 re-measures with a verified-isolated venv.

*Execution*
- Sandbox: `python:3.11-slim`, **512m / 1.0 cpu / 60s / 256 pids**, `--network none`.
- Workspace lifetime: **destroyed after every call** —
  `code_execution.py:279` `tempfile.mkdtemp()` and `:309` `shutil.rmtree(work)` in a
  `finally`. Step 2 cannot read step 1's output.
- Median recorded turn: **37s** (range 12–232s).

*Access*
- Reachable: platform API, OpenSearch, GitHub raw (8/12 notebooks on a naive
  `main`/`master` probe), MinIO `storage-dev.i-guide.io` (200).
- Not reachable from a dev machine: Neo4j `10.0.147.52:7687` (private subnet), the
  embedding server (stale `.env` host).

**Surprised by** Two findings inverted planned work. (1) `AGENT_ALLOW_WORKFLOW_EXEC=1`
routes ingested third-party notebook source into a bare
`exec(compile(...), namespace, namespace)` **in the MCP server process**
(`MCP_server/tools/generated_notebook_tools.py:43-47`), which has network, the shared
`agent_chat_files` volume, and cluster credentials — so the flag stays `0` and the
`mcp_run_nbwf_*` path gets deleted rather than enabled. (2) The prod mapping for
`spatial-bounding-box-geojson` is `{type: geo_shape, ignore_malformed: true}`, so the
dataset native-CRS bounds bug **silently drops the field** instead of failing the write;
that is why only 182/619 docs carry a bbox.

**Next** M0.2 (fail-closed auth) — nothing else is deployable until the unauthenticated
`/query` route is closed.

---

## 2026-08-07 · M0.2 · Auth fails closed; CORS scoped

**Change** `api/server.py`: `_require_agent_chat_api_key` now raises `RuntimeError`
when `AGENT_CHAT_API_KEY` is unset, unless `AGENT_CHAT_AUTH_OPTIONAL=1` is set
explicitly. Added a `require_api_key` decorator and applied it to the four routes that
had no auth at all: `/query`, `/query/batch`, `/agent/files/upload`,
`/agent/files/<id>/download`. Replaced `CORS(app)` with `CORS(app, origins=_cors_origins())`,
reading `AGENT_CORS_ORIGINS` (comma-separated) and falling back to the existing
`ALLOWED_DOMAIN_LIST` JSON array. New `rag_pipeline/tests/test_api_auth.py` (24 cases).

**Why** Two independent holes. `_require_agent_chat_api_key` returned early on an unset
key — so auth was disabled by *omission*, which is the failure mode that never shows up
in testing because the tests also omit the key. And `/query`, the full RAG pipeline, had
only two auth call sites in the whole file (`:1286`, `:1781`) and was not one of them:
it was reachable unauthenticated from any origin.

The fail-closed contract was already designed and just never implemented — the
agent-chat routes at `:1289` already catch `RuntimeError` and return
*"Server misconfiguration: API key not set"*. Making the helper raise it made the
existing handlers correct without touching them.

**Measured** Unauthenticated data-bearing routes: **4 → 0**. Wildcard CORS: **1 → 0**.
Auth tests: **0 → 24**, all passing.

`rag_pipeline/tests/` — two runs, because the environment changes the answer:
- Without `.env` (this worktree's default): 501 passed, **3 failed**.
- With the main checkout's `.env` exported: **503 passed, 1 failed** in 151s.

So two of the three were purely `Missing required environment variable:
OPENSEARCH_NODE`, not defects. The remaining failure is
`test_spatial_routing_e2e::test_spatial_routing_to_generation_e2e`, an e2e test against
real backends, in a file this change does not touch. It also failed in the pre-change
prod-pinned run recorded in M0.1.

**Surprised by** Three existing fixtures relied on the fail-open behavior:
`test_download_route.py:23` even documented it — `monkeypatch.delenv("AGENT_CHAT_API_KEY")
# no auth in test`. That is the clearest possible evidence the hole was load-bearing in
practice rather than theoretical. All three now set `AGENT_CHAT_AUTH_OPTIONAL=1`
explicitly, which is the behavior they actually wanted.

`/health` and `/agent/dashboard` are deliberately left open — the first is the container
healthcheck, the second a static HTML page with no data. Both are asserted in the new
suite so a future change to either has to be deliberate.

**Next** M0.3, the eval metric fix. Until `retrieval_success` scores against the full
expected sets, no later milestone can show an improvement.

---

## 2026-08-07 · M0.3 · retrieval_success scores the full expected set

**Change** `scripts/eval_common.py:342`: `retr` now scores against `full_exp` /
`full_cov` instead of `meta["primary_ids"]`. Added `retrieval_recall` ("N/M") and kept
the old value as `primary_retrieval_success`. `task_status` deliberately untouched, so
this change moves exactly one metric and nothing else.

**Why** `retrieval_success` was computed from a hand-picked subset of each task's
expected elements, and `full_exp`/`full_cov` were already computed on the two lines
above and simply unused. The metric therefore read "yes" on every recorded run and was
structurally incapable of registering a retrieval improvement — which is the metric all
of M1.2, M3 and M4 are supposed to move.

**Measured** Re-scoring all 11 committed records:

| | before | after |
|---|---|---|
| `retrieval_success == "yes"` | **11/11** | **5/11** |
| aggregate recall over full expected sets | not reported | **25/37 (68%)** |

Per task, worst first: T8 **2/7**, T5 1/3, T4 3/5, T1 2/3, T10 2/3, T9 4/5, then
T7 3/3, T2 4/4, T3 2/2, T6 1/1, CRIME_HEATMAP 1/1.

**Surprised by** 25/37 here versus the **22/37** BM25 figure recorded in M0.1. Both are
correct and they measure different things: 22/37 is a single BM25 query per task at k=8,
25/37 is what the full agent pipeline actually grounded on (several retrieval methods,
query refinement, and the direct sweep). Worth keeping distinct — the first is the
retrieval ceiling for one method, the second is end-to-end behavior. `scripts/eval_retrieval.py`
(M0.4) will report the first properly across methods and k.

Also: no test imports `eval_common`, so this metric had no regression net at all. The
new `retrieval_recall` field is what M1.2's exit criterion will be read from.

**Next** M0.4 — `scripts/eval_retrieval.py`, so recall@k is measurable per method
without an LLM in the loop.

---

## 2026-08-07 · M0.4 · scripts/eval_retrieval.py — the retrieval instrument

**Change** New `scripts/eval_retrieval.py`. Deterministic, no LLM: recall@k,
precision@k and MRR over the full expected sets, for arbitrary `--k` across arms
`keyword | semantic | agent_kb | union | union+agent_kb`. Union arms fuse with RRF
(k=60), matching the agent's own reranker so a union figure is comparable to what the
agent would actually see. Writes JSON with `--json`.

**Why** `run_eval_cases.py` measures end-to-end agent behavior, where a retrieval
regression can hide behind a good answer and a retrieval gain can be masked by the model
failing to use it. M1.2, M3 and M4 all claim to move retrieval, so they need an
instrument that isolates it and costs nothing to re-run.

**Measured** Baseline against the live cluster, `keyword` arm, all 11 tasks / 37 ids:

| k | recall | |
|---|---|---|
| 8 (current effective window) | **22/37** | 59.5% |
| 20 | **29/37** | 78.4% |
| 50 | 33/37 | 89.2% |
| 100 | 34/37 | 91.9% |

This **independently reproduces** the M0.1 baseline through a different code path,
which is the result I most wanted from this step — the 22/37 and 29/37 figures the whole
M1.2 case rests on are now produced by committed, re-runnable code rather than a
throwaway script.

Missed by every arm even at k=100 — genuine indexing gaps, not window problems:
`afbee4bd` (T6), `643aaea1` (T9), `de05a428` (T10). Exactly 3, matching M0.1.

**Surprised by** Two things worth keeping. First, `semantic` reports **unavailable**
rather than scoring 0/37, because `semantic_search` returns `[]` both when the embedder
is down and when nothing matched — collapsing those would have silently understated
every union arm and made the embedding outage look like a retrieval quality problem.
Second, my initial "unreachable at every k" label was wrong: it listed 8 elements when
only k=8 and k=20 had been tested, 5 of which are recoverable at k=50. Fixed to "missed
at every k *tested*", with a pointer to raise `--k` to separate a ranking problem from
an indexing gap. A metric that overstates the size of a problem is as unhelpful as one
that hides it.

Note `outputs/` is gitignored (`.gitignore:63`), so the JSON is not committed —
regenerate with:
`python scripts/eval_retrieval.py --k 8,20,50,100 --methods keyword --json outputs/retrieval_$(date +%F).json`

**Next** M0.5 — `constraints.txt`, so the dev/prod version drift stops making every
later measurement ambiguous.

---

## 2026-08-07 · M0.7 · Stop advertising a tool that cannot exist

**Change** `extractors/doc_ids.py`: `mcp_tool_name_for` now returns the real executor
name (`mcp_run_notebook_workflow` / `mcp_run_code_element`) instead of
`mcp_run_<workflow_id>`; added `run_invocation_for()` returning tool **and** argument,
since a name alone is not actionable for these executors. Both extractors' `contents`
markers changed from `[runnable: mcp_run_<wid>]` to `[workflow <wid>] … Not directly
callable; reuse the extracted functions.` `SkillSpec.allowed_tools` is now empty, and
`skill_emitter._render` emits an explicit "no single tool runs this" Run section instead
of silently omitting it. Hand-corrected the one shipped
`.agents/skills/ai-agent-for-chicago-crime-analysis/SKILL.md`.

**Why** `generic_executor_tools` registers exactly two fixed tools that take the workflow
id as an *argument*; it does not register one tool per workflow. So every
`mcp_run_nbwf_*` name ever emitted named a tool that could not exist — and one shipped
into a SKILL.md `allowed-tools` list, i.e. the model was explicitly told to invoke a
fiction. Worse, the docstring at `doc_ids.py:84` asserted the false belief
("The executor registers `run_<workflow_id>`"), so the bug was documented as intended
behavior.

The executors stay unreachable in beta regardless: `AGENT_ALLOW_WORKFLOW_EXEC` remains 0
and `generic_executor_tools` stays out of `DEFAULT_MCP_MODULES`, because the execution
body is a bare `exec()` of ingested source in the MCP server process. Hence "not directly
callable" rather than a corrected tool name — a truthful name for a gated tool would still
waste a turn.

**Measured** Emitted names that cannot resolve: **1 per promoted workflow → 0**.
Shipped SKILL.md files advertising a nonexistent tool: **1 → 0**. Suite unchanged
(501 passed, same 3 pre-existing failures).

**Next** M0.5, now materially larger — see the pyarrow finding below.

---

## 2026-08-07 · M0.5a · pyarrow is an undeclared dependency, and two geo tools are broken in prod

**Change** None yet — this entry records the finding, because it is a production defect
found while building the lock file and it deserves its own record.

**Why it matters** `pyarrow` appears in **no** requirements file, yet three code paths
call `to_parquet`:
- `agent_runtime/langchain_geo_tools.py:436` — `vector_reproject`
- `agent_runtime/langchain_geo_tools.py:476` — `vector_spatial_join`
- `extractors/geo_handles.py:42` — the GeoDataFrame file-handle mechanism (pickle fallback)

The dev machine has `pyarrow 24.0.0` from anaconda. A clean `pip install -r
requirements.txt` does not install it. So in the deployed image the two vector tools
return `{"ok": false, "error": "Missing optional dependency 'pyarrow.parquet'"}` and
`geo_handles` silently degrades to pickle — while both pass in dev.

**Measured** In a verified-isolated venv built only from `requirements.txt`:
`vector_spatial_join` → `ok: false`. That is the single extra failure in the prod-stack
suite run (**502 passed, 2 failed** vs dev's **503 passed, 1 failed**).

**Surprised by** How wrong my earlier framing was. I had recorded this as *LangChain*
version drift. It is not: the langchain triple moves nothing here. The real drift is the
scientific stack — prod resolves **pandas 3.0.5** against dev's **2.2.3** (a major
version), numpy 2.1.3 → 2.5.2, geopandas 1.1.2 → 1.1.4, and `fiona` appears from nowhere
at 1.10.1. For a geospatial platform a silent major pandas bump is a far larger risk than
a langchain minor, and neither was bounded. Also worth noting the resolver moved *within
this session*: the dry-run hours ago gave langchain 1.3.14, the real install gave 1.3.15.

**Next** M0.5 proper: add `pyarrow`, then build and *test* a constraints file rather than
freezing whatever resolves today.

---

## 2026-08-07 · M0.2b · Prototype told to match the new auth contract, verified in-browser

**Change** `examples/iguide_chat_prototype.html` and `examples/agent_chat_stream_demo.html`:
API-key label "(optional)" → "(required by default)" with a tooltip naming
`AGENT_CHAT_AUTH_OPTIONAL`; new `describeHttpError()` translating 403 / 500-misconfig /
CORS-block into actionable sentences; the upload path checks status *before* parsing JSON
(that route is now protected). New `scripts/run_agent_api_dev.sh` + a `.claude/launch.json`
entry so there is one supported way to start the API locally with the right env.

**Why** M0.2 changed the auth contract and I did not update the interface — so the
prototype still described the key as optional, and a blank key surfaced as a raw
`{"error":"Forbidden: invalid API key."}` body, which reads like a server bug rather than
a setting. A feature is not done when the server is right; it is done when the interface
tells the truth about it.

**Measured** — driven through the real prototype at `localhost:8132` against a live API on
`:5002`:

| Case | Before | After |
|---|---|---|
| no key | raw JSON body | "403 — this server requires an API key. Enter it in the API key field above." |
| wrong key | raw JSON body | "403 — the API key was rejected. Check the key matches AGENT_CHAT_API_KEY on the server." |
| correct key | — | full substantive answer, 4 notebooks, 8 sources |

Server-side matrix by curl: no key 403, wrong key 403, correct key 400
`user_input is required` (i.e. past auth into the handler), Bearer token also 400,
`/health` 200.

**Surprised by** three things, in increasing order of importance.

1. `scripts/run_agent_api_dev.sh` first used `set -a; . .env; set +a`, which **clobbered
   my explicit `PORT=5002` with the file's 3500 and replaced the API key I passed**. That
   is the shell twin of the `load_dotenv(override=True)` defect the plan already flagged:
   the file beating the environment makes per-run overrides impossible and the reason
   invisible. Rewrote it to skip any key already set in the environment.
2. Driving one query exercised both blocked backends and showed the cost: the embedding
   server is unreachable (stale `.env` host) so semantic retrieval is silently absent, and
   Neo4j returns *"Unable to retrieve routing information"* before falling back to keyword.
   The agent degrades correctly, but each unreachable backend is paid for in latency —
   which is the argument for the planned `NEO4J_ENABLED=0` fast-fail.
3. **The grounding audit fired `severity: high` on a well-grounded answer.** The answer
   named four notebooks, all four present in the retrieved sources
   (A2SFCA, Pysal Access Compute Example, SPASTC, 6.02 Thematic and Reference Mapping),
   and the audit appended: *"contains a hallucinated claim about the platform offering
   several notebooks for computing spatial accessibility to hospitals."* That claim is
   exactly what the evidence supports.

   This is a **false positive at the one severity that reaches the user**, and the gate's
   own comment at `graph.py:415` says `{"high"}` exists specifically to suppress false
   positives. So the caveat now degrades correct answers in the UI. It also corrects M0.1:
   "the caveat has fired zero times" is true of the 11 recorded runs, not of the system —
   it fired on the first live query. Logged as the top candidate for M6; it is also the
   publishable negative result the planning work identified.

**Next** M0.5 — the constraints file, which turned out to hide a worse problem than pandas.

---

## 2026-08-07 · M0.5 · constraints.txt — and two real production defects it exposed

**Change** New `constraints.txt` pinning the stack the passing suite and the eval baseline
were actually produced on. Added `pyarrow` to `requirements.txt`. The three Dockerfiles
that install the root requirements (`rag_pipeline/`, `MCP_server/`,
`metadata-extraction-server/`) now `pip install -r requirements.txt -c constraints.txt`.

**Why** `requirements.txt` carries unbounded pins and both entrypoints ran a plain
`pip install`, so "the deployed stack" was whatever PyPI happened to serve that hour. The
resolution moved *inside this session*: a dry-run gave langchain 1.3.14, the real install
an hour later gave 1.3.15.

**Measured** — four clean, verified-isolated builds, same suite, same `.env`:

| Build | Result |
|---|---|
| dev machine (anaconda) | **503 passed, 1 failed** |
| clean, unconstrained | 502 passed, **2** failed — `test_spatial_join` |
| constrained, but only `langgraph` pinned | 502 passed, **2** failed — `test_history_repair_middleware` |
| **fully pinned (this commit)** | **503 passed, 1 failed — matches dev** |

The one remaining failure is `test_spatial_routing_e2e`, pre-existing and unrelated.

**Surprised by** two defects, both of which were shipping:

1. **`pyarrow` was undeclared.** `langchain_geo_tools.py:436` (`vector_reproject`) and
   `:476` (`vector_spatial_join`) write GeoParquet, and `geo_handles.py:42` uses it to move
   (Geo)DataFrames between tools. Dev had pyarrow 24.0.0 via anaconda; a clean build did
   not. So both vector tools returned
   `{"ok": false, "error": "Missing optional dependency 'pyarrow.parquet'"}` **in every
   deployed image**, and `geo_handles` silently fell back to pickle. `requirements.txt:28`
   already carried a comment about exactly this failure mode for bs4/lxml/markdownify —
   the pattern had been recognised once and pyarrow missed. Fixed: `test_langchain_geo_tools.py`
   goes **12 passed / 1 failed → 13 passed**.

2. **Pinning `langgraph` alone is not enough, and the failure is total.** Its sub-packages
   version independently: a clean build took `langgraph-prebuilt` **1.0.13** against the
   pinned `langgraph` **1.0.10**, and 1.0.13 does
   `from langgraph.runtime import ExecutionInfo` — a name absent in 1.0.10. That raises
   `ImportError` while importing `langchain.agents`, so **`create_agent` cannot be
   constructed and the entire agent is dead on arrival**, not degraded. `langgraph-checkpoint`
   (4.0.1 → 4.2.0), `langgraph-sdk` (0.3.9 → 0.3.15) and `langsmith` (0.6.7 → 0.10.18)
   drifted the same way. All now pinned.

   This reframes the whole exercise: I had been treating the drift as a *quality* risk
   (untested versions) when it also contained an *availability* risk (a clean build that
   cannot start the agent at all).

Also worth recording: `fiona` is listed in `requirements.txt` but is **not installed in
dev**, so dev reads vectors through `pyogrio` while every container gets fiona. Left as-is
for now — a divergence to close deliberately, not silently.

**Next** M0.6 — `LLM_PROVIDER` with the `claude-cli` backend, so the extraction batches
from M2 onward cost nothing.

---

## 2026-08-07 · M0.6 · LLM_PROVIDER=claude-cli (dev/experiments only) · model: sonnet

**Change** New `rag_pipeline/llm_claude_cli.py` and a five-line dispatch in
`llm_utils.call_llm:58`. `LLM_PROVIDER=claude-cli` routes to `claude -p --output-format json
--model $CLAUDE_CLI_MODEL` (default **sonnet**). `last_model()` records the model per call;
`preflight()` + `python -m rag_pipeline.llm_claude_cli` diagnose setup in one command;
`check_not_deployed()` refuses to run where `AGENT_DEPLOYED` / `KUBERNETES_SERVICE_HOST` /
`ECS_CONTAINER_METADATA_URI` is present. 21 tests, subprocess stubbed so they need no
credentials.

**Why** `call_llm` is where the recurring cost of this project sits — the publication
extractor runs over ~180 elements and the rerank/audit/router paths fire every turn. Routing
it through the CLI during development makes the M2–M7 batches free. It is deliberately *not*
wired into `build_default_llm()` (the agent peers): experiments should run on models
comparable to what is deployed, and using a stronger model there would flatter the eval.

**Measured** 21/21 new tests pass; suite **501 → 522 passed**, same 3 pre-existing failures.

⚠️ **BLOCKED on a credential — the backend cannot authenticate on this machine.** Both paths
were tested and both are unavailable:

| Path | Result |
|---|---|
| `--bare` (needs `ANTHROPIC_API_KEY`) | `Not logged in · Please run /login` — no key in env |
| no `--bare` (uses the interactive login) | `401 OAuth access token has expired. Re-authenticate to continue.` |

To unblock, **one** of:
- `export ANTHROPIC_API_KEY=...` — preferred: works with `--bare`, reproducible, and the
  path Anthropic's terms require for automated use; or
- run `claude` interactively, `/login`, then set `CLAUDE_CLI_BARE=0`.

**Surprised by** a real design tension in `--bare` that changes how this must be configured.
`--bare` is what makes a run reproducible — it skips hooks, plugins, auto-memory and
CLAUDE.md auto-discovery, so this repo's own instructions are not silently prepended to every
extraction prompt. But its help text states that under `--bare` "Anthropic auth is strictly
ANTHROPIC_API_KEY … OAuth and keychain are never read". **So `--bare` and subscription auth
are mutually exclusive**, and the tool itself enforces the boundary the terms describe:
scripted use wants an API key. `use_bare()` therefore defaults from the credential that is
actually present rather than being hardcoded on, and the docstring records the trade-off
(subscription = free but leaks project context into prompts; API key = costs money but is
reproducible and compliant).

Second, smaller: the CLI **exits non-zero while still emitting the JSON that explains why**.
My first version judged `returncode` before parsing, which turned "not logged in" into an
unreadable dump of usage counters. Parsing now precedes the exit-code check, and
`test_auth_failure_is_actionable_even_on_nonzero_exit` pins it.

**Next** M0.8 — `_fan_out` in `ingest_from_github`, the last M0 item.

---

## 2026-08-07 · M0.8 · ingest_from_github now actually emits

**Change** `extractors/ingest.py`: `ingest_from_github` calls `_fan_out(manifest, ctx.targets)`
and takes `element_id` / `dry_run`. Emitting without an `element_id` now raises rather than
proceeding. `extractors/cli.py` grows `--element-id` and `--dry-run` and reports on stderr
which targets it emitted to. `MCP_server/tools/ingest_tools.py` mirrors both, returning
`{"ok": false, "error": ...}` as data rather than raising, matching this repo's MCP convention.

**Why** `_fan_out` was called only from `ingest_submission` (the webhook). So both other
entry points — `extractors.cli` and the MCP `ingest_github_repo` tool — extracted everything
and silently discarded it, while accepting `--targets` and `--reingest` arguments that
implied persistence. The `element_id` guard exists because without it every derived doc_id
anchors on `repo_id`, seeding docs into the agent KB that no platform element can ever claim.

**Measured** Ingesting one real notebook (`cca9b545`) through the CLI:

| | before | after |
|---|---|---|
| files persisted | **0** | **3** — `agent_kb/iguide_agent_notebook_blocks.json` (6 blocks), `generated_notebook_workflows/sources/nbwf_d01e717421c1b0ff.py`, and its manifest (12 keys) |
| emit without `element_id` | silently produced nothing | refuses with an actionable error |
| `--dry-run` | did not exist (was the only behavior) | prints the manifest, emits nothing |

Suite: 522 passed, same 3 pre-existing failures.

Bonus verification: the freshly generated SKILL.md came out with `allowed-tools: []` and the
new "no single tool runs this pipeline" Run section, and **zero** `mcp_run_nbwf_*` occurrences
— so M0.7 holds on generated output, not just on the one file I hand-edited.

**Surprised by** two things this test run exposed.

1. **The skill emitter writes into the working tree.** `skill_emitter._default_root()` is
   `REPO_ROOT/.agents/skills`, so my single test ingest created
   `.agents/skills/cca9b545-.../` inside the checkout — i.e. running ingestion mutates the
   repo, and any CI or test run would dirty it. `AGENT_GENERATED_SKILLS_ROOT` overrides it,
   but the *default* being the source tree is wrong for an emitter. Cleaned up; noted for the
   M3 test work, which will need that root pointed at a tmpdir.
2. **CLI-path ingestion produces badly named skills.** The generated skill was named
   `cca9b545-8416-45a3-9267-122ce6ce9991` — the raw UUID — because `slugify(title)` had no
   title to work from: `ctx.fields` carries the platform form metadata and the CLI path
   supplies none. The webhook path gets titles, tags and authors; the CLI path gets nothing.
   That is an argument for the M2 `sources.py` work fetching the element record from
   `backend.i-guide.io` rather than relying on whoever calls the CLI to pass fields.

**M0 complete.** Next: M1.1 (persistent workspace), the first of the three ceilings.

---

## 2026-08-07 · M1.2 · One retrieval window (AGENT_SEARCH_TOP_K, default 20)

**Change** New `rag_pipeline/search/utils.default_top_k()` as the single source of truth,
routed into every retrieval entry point: `keyword.py` (wrapper + `get_keyword_search_results`),
`semantic.py` (wrapper + `semantic_search`), `core.py`, `neo4j.py`, `spatial.py` ×2,
`opengeodata.py` ×3, `_safe_int` plus six tool signatures in `langchain_granular_tools.py`,
and `_direct_search_sweep`. `AGENT_SUPERVISOR_TOP_K` deliberately left at 8. 16 regression
tests in `test_retrieval_window.py`.

**Why** Recall over the benchmark's full expected sets was 22/37 at the effective window of
8 while 29/37 was available at 20 — the single largest measured improvement for the least
code in this whole plan.

**Measured**

| | recall |
|---|---|
| k=8 (old effective window) | **22/37 (59.5%)** |
| k=20 (new default) | **29/37 (78.4%)** |
| k=50 | 33/37 (89.2%) |

Verified live through `/agent/chat/stream` — the same endpoint the prototype uses —
`keyword_search count=8 → count=20`. Hardcoded windows remaining: **0**. Suite 522 passed,
same 3 pre-existing failures. `AGENT_SUPERVISOR_TOP_K` still 8, so answer-prompt cost is
unchanged.

**Surprised by** three things, and the first two are corrections to the plan.

1. **The window was in 16 places, not 6.** The plan (and the analysis behind it) listed
   `keyword.py`, `semantic.py`, four tool signatures and the sweep. The real count included
   `core.py`, `neo4j.py`, `spatial.py` ×2, `opengeodata.py` ×3, a sixth tool
   (`opengeodata_search_tool`), and — the one that actually mattered — **`size: int = 12`
   defaults on `get_keyword_search_results` and `semantic_search` themselves**, a *third*
   distinct window nobody had counted.
2. **My first attempt measured "no change" and was right to.** Recall came back 26/37 for
   both k=8 and k=20 because I had only fixed the state-machine wrapper, while my
   measurement called `get_keyword_search_results` directly and hit its own `12`. The
   instrument caught my incomplete change — which is precisely why M0.4 came before this.
3. **A near-regression worth recording.** Making `opengeodata`'s `limit` default to `None`
   would have *reduced* it to a single result: `_payload_from_context` does
   `int(limit or 1)`, so `None` collapses to 1, not to the window. Resolved before the call
   and pinned by `test_opengeodata_none_becomes_the_window_not_one`.

The general lesson, now encoded in the helper's docstring and a test: these values must
resolve at **call** time. `limit: int = default_top_k()` binds once at import and silently
ignores the environment forever — the same trap that made attempt 1 unmeasurable.

**Tooling note** The browser pane's viewport went to 0x0 partway through this step, so the
final prototype check was done against the SSE stream directly rather than the rendered UI.
Same endpoint, same payload, but recording it because it is weaker evidence than a rendered
page: it confirms the server behavior the prototype consumes, not the prototype's rendering
of it.

**Next** M1.1 — the persistent workspace.

---

## 2026-08-07 · M1.1 · Persistent per-session workspace, tiers, incremental artifacts

**Change** `agent_runtime/code_execution.py`: `execute()` takes `session_id` and `tier`. With
a session the workspace is `<work_root>/sessions/<safe_session_id>/` and **survives** the
call; without one the old throwaway-and-remove behaviour is kept exactly. Added
`resolve_tier()` / `EXEC_TIERS` (quick 60s/512m · standard 300s/2g · heavy 900s/6g, gated by
`AGENT_CODE_EXEC_ALLOW_HEAVY`), `sweep_workspaces()` TTL reclamation, a size cap, and
`_deps_satisfied`/`_record_deps` so a session installs a dependency once. `build_argv` honours
tier limits. `make_code_execution_tools(session_id=...)` threads it, and both peers in
`supervisor/graph.py` pass `child_thread_id(thread_id, "code_exec")`. 32 tests.

**Why** `:279` `mkdtemp()` plus `:309` `shutil.rmtree(work)` in a `finally` destroyed the
workspace after **every** call. Step 2 could not read step 1's output, so a multi-step
workflow was inexpressible no matter how capable the model — the hard ceiling this whole
milestone exists to remove. And 512m/1cpu/60s is a quick-tool budget: a county-level join
OOMs under it, which teaches the model to avoid real computation.

**Measured** A real 3-step workflow through the tool layer:

```
step1: ok=True  stdout='stage1 written'                    new_artifacts=['stage1.json']
step2: ok=True  stdout="read stage1: {'rows':3,'doubled':6}" new_artifacts=['stage2.json']
step3: ok=True  stdout="final: {'rows':3,'doubled':6}"       new_artifacts=[]
sessionless: stage1.json present: False
```

- Cross-call state survival: **0/N → 3/3**.
- Dependency installs across 3 calls in one session: **3 → 1** (and still 3 without a session).
- Incremental persistence works visibly: step 2 persisted only `stage2.json`, step 3 nothing.
- Default tier: **quick → standard** (60s/512m → 300s/2g). Suite **570 passed**, same 3
  pre-existing failures.

**Surprised by** four things, one of them a security bug.

1. **`_safe_session_id("..")` returned `".."`.** I copied the helper from
   `qgis_headless_tools.py:153`, whose allowlist `[^A-Za-z0-9_.-]` **permits dots** — so
   `..` survives sanitisation untouched, `<sessions_root>/..` is the work root itself, and
   `sweep_workspaces` would eventually `rmtree` it. Found by an adversarial test I wrote on
   principle, not by reading the code. Fixed here (reject any all-dots name, plus a
   resolved-path containment check as defence in depth); **the original in
   `qgis_headless_tools` is still vulnerable** and is spawned as its own task.
2. **The tier would have been dead on arrival.** `execute_code`'s `timeout_seconds` defaulted
   to `DEFAULT_TIMEOUT` (60), which is truthy, so it would have overridden every tier's
   timeout on every call and the tiers would have done nothing. Now `None`.
3. **Incremental persistence is a correctness requirement, not an optimisation.**
   `_persist_artifacts` walks sorted paths and stops at `MAX_ARTIFACTS=20`, so the moment a
   workspace persists, step 1's leftovers consume the budget and step 5's real output is
   never persisted at all. `test_late_output_is_not_crowded_out_by_early_leftovers` pins it.
4. **The exit criterion had to be measured differently than planned.** The plan wanted a
   timed "3 installs → 1". The host's anaconda pip cannot `--target` install at all — it
   raises `PermissionError` scanning an unreadable `sys.path` entry
   (`/Users/yfkang/Documents/New OpenCode Project/wildfire-agent/src`), a pre-existing
   environment fault unrelated to this change. Counting install subprocesses instead gives
   the same claim without measuring the wrong thing.

Also updated four `_Stub.execute` signatures in `test_code_execution.py` to accept
`**kwargs`: the executor contract genuinely gained two parameters, and having the tool
silently omit them for duck-typed executors would hide real wiring bugs.

**Next** M1.3 (agent indices) is blocked on the embedding server. Taking the grounding-audit
false positive (task #10) next instead, since it degrades every correct answer in the UI.

---

## 2026-08-07 · M6a · Grounding audit: verdict computed from the ledger, in code · model: gpt-4o

**Change** `agent_runtime/evidence_quality.py`: added `_is_rollup_claim()` and
`_recompute_verdict()`, applied to every audit result. Roll-up rows are reclassified as
supported when at least one genuine claim is supported, and
`hallucination_detected`/`severity`/`issues` are re-derived from the surviving ledger rather
than taken from the model. Also added an explicit roll-up rule to `_AUDIT_PROMPT`. 22 tests.

**Why** Driving the prototype produced a `severity: high` hallucination caveat on a
**fully-grounded** answer — four notebooks, all four present in the retrieved evidence. The
caveat then reached the user and undermined a correct answer.

**Measured**, 5 runs each against the live index:

| | false positives (well-grounded answer) | true positives (fabricated tail) |
|---|---|---|
| before | **5/5 high** | 5/5 high |
| prompt rule only | **4/5 high** | 5/5 high |
| + deterministic recomputation | **0/5 high** | **5/5 high** |

Confirmed end to end through `/agent/chat/stream`: the same query now returns a clean
1412-char answer with **no caveat**. Suite **592 passed**, same 3 pre-existing failures.

**Surprised by** three things, and the first invalidated my own first diagnosis.

1. **My initial reproduction was wrong, and I nearly filed a bug on it.** I passed raw
   OpenSearch hits to `audit_answer_grounding`, and `_normalize_document` does not understand
   `{"_source": {...}}` — it yields `doc_id="doc-0"`, `title="Untitled"`, `contents=""`. The
   auditor received *literally empty evidence* and correctly reported everything absent. The
   live path is fine because `_direct_search_sweep` normalises via `_hit_to_document`. Worth
   noting as a latent trap: any future caller passing raw hits gets an audit that flags every
   correct answer as fully hallucinated, silently.
2. **The offending row was always the same one, and it is unprovable by construction.** The
   answer's opening sentence, "The platform offers several notebooks that compute spatial
   accessibility to hospitals," cannot have a verbatim supporting span — no document says "the
   platform offers several." Its truth is carried by the items listed beneath it. Under the
   prompt's one-span-per-row rule it lands as "absent" and promotes to high severity.
3. **The prompt could not fix it.** Adding an explicit, emphatic rule not to open a row for
   roll-up sentences moved the rate only **5/5 → 4/5**: the instruction competes with the
   model's judgement and loses. `rollup_claims_reclassified` fires in 5/5 runs even *after*
   the prompt change, i.e. the model still rows the sentence every single time and the code is
   what corrects it.

That third point is the general lesson, and it is the same one the deterministic search
short-circuits already encode: **where an invariant is checkable, check it — do not ask a
model to respect it.** The verdict is now derived from the ledger in both directions: a clean
label over unsupported rows is corrected *upward* to high, and an issue raised against a row
the model itself marked supported is dropped.

The guard against this becoming a hole: `_HARD_COUNT_RE` keeps "three notebooks", "12
datasets", "the only notebook" out of the roll-up class, so a fabricated *count* stays
auditable. And with no usable ledger the model's verdict is left untouched, so this can never
invent a clean result for an audit that did not produce one.

**Next** M1.3 needs the embedding server. Proceeding to the extraction restructure (M2
prerequisites: `contracts.py` + the callability analyzer) which is unblocked.

---

## 2026-08-07 · M2.1 · Contracts + callability analyzer — and a finding that revises the plan

**Change** Three new pure modules (no I/O, no LLM, no execution): `extractors/contracts.py`
(`ParamSpec` / `Callability` / `InvariantSpec` / `UnitContract`),
`extractors/analysis/callability.py` (stdlib `symtable` free-variable analysis), and
`extractors/analysis/signatures.py` (full-fidelity signatures + type/unit/CRS inference). Plus
`scripts/measure_callable_units.py` to measure the corpus. 21 tests.

**Why** A function lifted from cell 12 that reads `gdf` from cell 4 imports fine and then
fails at call time — or silently uses a stale global. That risk is why the extractor promotes
one whole-notebook entry point instead of per-function units. The analyzer's job is to decide
which units are safe, and the key distinction is not "reads a global" but *what kind of
binding it is*: imports, sibling defs and literal consts can be copied into a slice; a value
produced by executing something cannot.

**Measured** — against the 14 real cached notebooks:

```
callable ratio: 40 of 41 functions (98%)
notebooks contributing >=1 callable unit: 9/14 (64%)
SUPPLY: 4/14 notebooks define ZERO functions (script-style, straight-line cells)
CONCENTRATION: the top 2 notebooks supply 26/41 units (63%)
top blockers (hidden-global class): 1x full_gdf, 1x linear_cm
```

Signature fidelity, on the case the old implementation mangled:
`def f(p, /, a: int=3, *args: str, k: float=1.0, **kw) -> 'gpd.GeoDataFrame'` — previously
emitted as `def f(a, *args, **kw)`. Suite **613 passed**, same 3 pre-existing failures.

**Surprised by — and this revises the plan.** I built the analyzer for the hidden-global
problem. **That problem is almost absent: 1 of 41 units is blocked by it.** The real limiting
factor is *supply*:

- **4 of 14 notebooks define no function at all.** They are straight-line cell scripts, so
  per-function promotion has nothing to promote regardless of how clean their globals are.
- **Two notebooks supply 26 of the 41 units (63%).** The distribution is extremely skewed, so
  "N callable units" across the corpus will be dominated by a handful of contributors.
- 9 of 14 notebooks yield at least one unit, so the method library is viable — but thin, and
  scaling to 133 notebooks will likely yield low hundreds of units, not thousands.

The plan anticipated exactly this branch: *"If that ratio is low, the analyzer is telling the
truth about the notebooks and the composition story needs the parameterisation work before it
needs more plumbing."* The ratio is **high** (98%) and the **supply** is low, which is a
different diagnosis than either branch predicted, and it points somewhere specific:

1. **Raise the priority of the code extractor.** `.py` files define functions by construction;
   notebooks often do not. 27 code elements are indexed and currently produce nothing callable.
   This was scheduled late (M7) on a "zero demand signal" reading; the supply number is a
   stronger argument than the demand one.
2. **Cell-to-function lifting is the way to reach script-style notebooks** — wrap a
   straight-line cell as a function whose free variables become parameters. That is real work
   and should be its own step, justified by this number rather than assumed.
3. Do **not** invest further in hidden-global handling. It is 1 case in 41.

I am not reordering the roadmap unilaterally on one 14-notebook sample — the measurement
should be repeated once more notebooks are ingested (M3) before the priority actually moves.
Recording it now so the decision is driven by the number rather than the original assumption.

**Next** the `blocked_by` histogram is doing its job. Continuing to M2's slice builder, then
per-function promotion in `notebook_extractor`.

---

## 2026-08-07 · M2.2 · Slice builder — and two bugs only *running* the output could find

**Change** `extractors/analysis/slices.py`: `build_unit_slice()` emits a standalone importable
module for one unit — provenance header, only the required imports verbatim, only the required
literal consts, the required sibling defs dependency-first, then the unit. Plus `slice_sha()`
(content address == version), `has_module_side_effects()`, `build_module_source()` moved here
from the notebook extractor, and framework-decorator stripping. 18 tests.

**Why** A slice must never execute anything at import: no data loads, no API calls, no
credential prompts. That is the whole safety argument for shipping extracted code, and it is
why a `needs_globals` unit is *refused* rather than patched — making it run would require
inlining exactly the statements this module excludes.

**Measured**, slicing every callable unit from the 14 real notebooks and importing each in a
clean subprocess:

| | |
|---|---|
| callable units | 40 |
| produced a slice | 40 |
| import side effects | **0** |
| **imports cleanly** | **39/40** (was 37/40 before the second fix) |

The one remaining failure is honest: `Simulation.run_simulation` needs a local `Viz` module
that is not installable. That is a requirements fact about the unit, not a slicing defect —
and the slice correctly declares what it needs rather than pretending.

**Surprised by** two bugs, both of which passed every static check and were caught only by
importing and calling the output.

1. **Requirements were collected for the target unit only, not its closure.** Slicing `good`
   (which calls `helper`, which reads the const `THRESH`) emitted `helper` but not `THRESH`,
   because `good` itself never reads it. The slice parsed, contained no side effects, satisfied
   every assertion I had written — and would have raised `NameError` on the first call. Fixed
   by unioning requirements over the transitive closure.
2. **Annotation-only names were never required.** `def filter_dataframe_by_value(df:
   pd.DataFrame, ...) -> pd.DataFrame` uses `pd` nowhere in its body, so symtable reported no
   global read — correctly, since annotations evaluate in the *enclosing* scope at `def` time.
   But the slice carries the annotation, so it died with `NameError: name 'pd' is not defined`
   **at import**. Fixed with `annotation_names()`, covering parameter annotations, the return
   annotation and decorators. Corpus import rate 37/40 → 39/40.

   Deliberately excluded: a *stringized* annotation (`x: 'gpd.GeoDataFrame'`) is not evaluated
   at def time and so cannot fail an import — requiring an import for it would add a dependency
   the unit does not actually need.

The methodological point is the one worth keeping. My static checks —
`has_module_side_effects`, "does it contain the runtime binding", "are dependencies ordered" —
all passed on both broken slices. **The only check that found either bug was importing the
artifact and calling it.** That is the same argument as the plan's artifact-re-runs criterion,
arriving a milestone early, and it is why the corpus import test is now the central test in
this file rather than a nice-to-have.

Suite **631 passed**, same 3 pre-existing failures.

**Next** per-function promotion in `notebook_extractor`, replacing the all-or-nothing
`all_parse_ok` gate at `:206`.

---

## 2026-08-07 · M2.3 · Per-function promotion — one bad cell no longer costs a notebook

**Change** `notebook_extractor` now emits one `MethodUnit` asset per top-level function, each
carrying a serialized `UnitContract`, assembled from the cells that **parsed**. Independent of
the whole-notebook workflow gate. Supporting wiring: `EMIT_LIBRARY` as a fourth target,
`KIND_METHOD_UNIT`, `AssetRecord.unit`, `doc_ids.method_unit_doc_id`, an
`iguide_agent_method_units` index, `DEFINES` edges, and an R-kernel guard. 13 tests.

**Why** `if assets and all_parse_ok:` meant a single unparseable cell — shell escapes, a
partial edit, notebook-only syntax — produced *nothing reusable from the entire notebook*.
Real notebooks routinely contain one.

**Measured**, through the real extractor over the 14 cached notebooks:

| | |
|---|---|
| block assets | 290 |
| **method units** | **41** |
| independently callable | **40** |

Matches the standalone M2.1 measurement exactly, which is the point — the extractor and the
analyzer agree. Three notebooks with unparseable cells (`21788323` 4 bad cells, `5278e805` 3,
`cca9b545` 1) now yield 1, 1 and 5 units respectively; under the old gate all three yielded
zero. Suite **644 passed**, same 3 pre-existing failures.

Design decisions worth recording:

- **`needs_globals` units are indexed but never shipped.** They keep `EMIT_OPENSEARCH` (so
  they stay discoverable and their blocker is visible in `contents`) and are denied
  `EMIT_LIBRARY`. Shipping one would require inlining the module-level statements the slice
  builder exists to exclude, so the two decisions have to agree.
- **`contents` is retrieval text, not the body** — signature plus doc summary. Raw code
  retrieves poorly against natural-language questions, which is the same reasoning
  `_embed_text` already applies to blocks.
- **Unit doc_ids are name-keyed** (`::unit::{qualified_name}`), not order-keyed. Inserting a
  cell renames every `::block::{order}` doc after it; a name-keyed id survives reordering,
  which matters for idempotent re-ingest. Pinned by a test that inserts a leading cell and
  asserts the ids are unchanged.
- **The R guard is not theoretical.** `nc <- st_read(...)` parses as valid Python (`<` then
  unary `-`), and r1 rewrites `%%R ...` to `_cellmagic('R ...')`, which also parses. Without
  the kernel check an R notebook would be promoted as runnable Python.

**Next** the library emitter (write the slices as an importable package), then mounting it
into the sandbox. `scripts/measure_callable_units.py` is now redundant with the extractor
itself and should be folded into a coverage report at M3.

---

## 2026-08-07 · M2.4 · The method library is real, importable code

**Change** `extractors/emitters/library_emitter.py` writes EMIT_LIBRARY units under
`storage_root()/method_library/` as an `iguide_methods` package — one content-addressed module
per unit (`v_<slice_sha>.py`), a per-element subpackage re-exporting the current version, a
`_registry.json`, and per-element `requirements.txt`. `AssetRecord.slice_source` carries the
emitted source; `_fan_out` gained a `library` branch. `code_execution` mounts the library
**read-only** at `/opt/iguide_methods` and puts it on `PYTHONPATH`. Plus `NEO4J_ENABLED`.
15 tests.

**Why** The agent composes extracted methods in Python, not by chaining tool calls. One tool
per unit would put every ingested function into every prompt — a 24-tool peer already spends
~3,900 tokens on schemas. A package keeps the tool surface O(1) in the number of units.

**Measured**, building the library from all 14 real notebooks:

| | |
|---|---|
| units offered | 40 |
| modules written | **40** |
| registry entries | **77** (40 qualified + 37 unique bare aliases) |
| skipped | 0 |

and, in a clean subprocess with only the library on `sys.path`:
`M.get('ke__21788323__21788323.get_url')` → `<function get_url>`, with `describe()` returning
the signature, source element and `slice_sha`. Suite **659 passed**, same 3 pre-existing
failures.

**Surprised by** a collision bug that only showed up at corpus scale. The first run reported
**40 modules written but a registry of 37**. I had namespaced the *module path* by element —
and then keyed the *registry* by bare symbol name, so `generate_random`,
`generate_random_loc` and `get_url` (each defined by two different notebooks) silently
overwrote each other. `get("get_url")` would have returned whichever element happened to be
ingested last.

Fixed by keying on the qualified `<element_pkg>.<symbol>` and adding a bare alias only when
unambiguous; a colliding bare name becomes an explicit `ambiguous` entry that `get()` raises
on, listing the candidates. A resolver that guesses is worse than one that refuses — and the
mismatch between 40 written and 37 registered is exactly the kind of thing a summary number
catches and a unit test on a two-element fixture would not.

**Neo4j** The updated IP (`149.165.155.195:7687`) is **reachable** — TCP open. But queries
fail with `Neo.ClientError.Security.Unauthorized`: the host moved and `NEO4J_PASSWORD` in
`.env` is stale for the new instance. Added `NEO4J_ENABLED` (default on) that short-circuits
before the driver connects. Honest measurement: with the host reachable this saves nothing
(0.42s vs 0.39s, since auth fails fast) — its value was the ~30s driver timeout when the host
was unroutable, which the IP fix already removed. Keeping it for dev machines off the network.

**Next** `kb_method_search` / `get_method_contract` so the agent can find units, then a real
end-to-end compose through the sandbox mount.

## 2026-08-12 · M4.0 · The graph arm was returning the wrong ids for the wrong nodes
**Change** `rag_pipeline/search/neo4j.py` — three defects in the keyword arm:
  (a) `neo4j_query_terms()` + `any(t IN $terms ...)` replacing a `CONTAINS $q` phrase match,
  ranked by match count (`2*title_hits + body_hits`);
  (b) `doc_id` now resolves to the platform UUID (`properties["id"]`), not Neo4j's internal
  `element_id`;
  (c) the match is scoped to `$labels` from the existing `_get_resource_labels()`.
  New `rag_pipeline/tests/test_neo4j_search.py` (22 tests, fake driver, no network).

**Why** With credentials fixed the arm connected and still returned nothing useful. The
Cypher bound `$q` to the *whole query* and tested `toLower(r.title) CONTAINS toLower($q)` — a
phrase-substring match, so any natural-language question missed:

    "spatial accessibility hospitals" -> 0      "spatial accessibility" -> 9
    "accessibility"                   -> 9      "flood"                 -> 10

Then, with terms fixed, hits came back with `doc_id = "4:f84f361b-...:532"`. The chain was
`properties.get("_id", node.element_id)`, and `_id` exists on **0 of 3205 nodes**, so it
*always* fell through to the internal id. Every other arm keys `doc_id` on the platform UUID,
so graph hits could never dedupe against them and cited links resolving to nothing.
Third, `MATCH (r)` swept all 3205 nodes; **2386 are :Alias/:Contributor**, which carry no
`visibility` — and `is_public_visibility(None)` is `True` — so a contributor's *name* matching
a query term was a returnable search result.

**Measured** recall@20 over the 37 full expected ids:

| arm | before | after |
|---|---|---|
| neo4j alone | 0/37 | **20/37** |
| keyword | 29/37 | 29/37 |

Suite **681 passed**, same 3 pre-existing live-cluster failures.

**Surprised by** the fix not being worth fusing in. Once the arm worked I added it to the
union — and recall@20 *fell* 29/37 → 26/37. Set analysis explains it: of the 37 expected
elements, keyword-only found 9, both found 20, and **neo4j-only found 0**. The graph's hits
are a strict subset of BM25's, so RRF into a fixed window can only evict correct hits. I
reverted the fusion and kept `union+neo4j` as a named arm so the negative result stays
reproducible. The graph earns its keep through *traversal* — related elements, collection and
contributor edges — not through lexical recall, and M4's "union the KB arm" plan should not be
read as "union every arm."

**Also** `.env`'s `NEO4J_PASSWORD` contains an unquoted `$`, so `set -a; source .env` silently
shell-expands it into a wrong password. Doing that during this step tripped
`Neo.ClientError.Security.AuthenticationRateLimit` and briefly looked like a credential
problem. Load `.env` with `python-dotenv` (or `scripts/run_agent_api_dev.sh`), never `source`.

**Next** `kb_method_search` / `get_method_contract` so the agent can find units, then a real
end-to-end compose through the sandbox mount.

## 2026-08-12 · M4.1 · The agent can find extracted methods — and 26 of 40 were unimportable
**Change** `agent_runtime/method_library.py` (new) + two tools `kb_method_search` /
  `get_method_contract`, registered and added to `graph_state.RAG_COMPONENT_TOOL_NAMES`.
  Then, forced by what the tools exposed: a new `NEEDS_INSTANCE` verdict in
  `extractors/analysis/callability.py`, and a `_defines_at_module_level()` guard in
  `library_emitter`.

**Why** M2 ended with 40 units written to an importable package the sandbox mounts — and no
way for the agent to learn any of them existed. Extracted, mounted, undiscoverable.

Both tools read the library's `_registry.json` rather than a `MethodUnit` OpenSearch index as
planned. The registry is written by the same `emit()` that writes the modules, so an import
line taken from it is guaranteed to resolve; an index doc and the mounted library drift
independently, and the failure mode of that drift is the worst kind — the agent is told to
import something that does not exist, inside a container with no network to check. The cost
is search quality (token overlap, no IDF), which is adequate at tens of units and is called
out in the docstring as the point to index properly.

**Measured** on the real 14-notebook corpus, before → after:

| | before | after |
|---|---|---|
| units advertised as callable | 40 | 16 |
| **advertised import lines that actually import** | **14/40** | **16/16** |
| verdicts | 39 callable, 1 needs_globals | 16 callable, 24 needs_instance, 1 needs_globals |

Suite **714 passed**, same 3 pre-existing live-cluster failures.

**Surprised by** how badly the library was broken, and by having measured it wrong before.
The first search returned `def download_all_samples(self, csv_files)` — a bound method offered
as a standalone callable. `iter_units` collects public methods as `Class.method` deliberately
(they are useful evidence), but a method reads no globals — `self` is a *parameter* — so it
passed every blocker check and was verdicted `callable`. **24 of 40 units were methods.**

Worse, the failure was not contained. The slice correctly emits the enclosing class, but the
registry advertised the bare method name, so `from .v_sha import build_api_url` raised
ImportError — and that line lives in the element's `__init__.py`, so **one bad method took
down every sibling unit in its element**. 26 of 40 import lines failed, of which only 24 were
themselves methods.

My earlier "39/40 slices import" was measuring the wrong thing: I imported the *module*, not
the *symbol*, and the symbol re-export is exactly where it breaks. Corrected here. This is
the plan's `0 false-callable` exit criterion failing in the field, and it is why that
criterion is asymmetric — a false-not-callable costs coverage, a false-callable ships a
broken unit.

Methods are still indexed for discovery, just never given `EMIT_LIBRARY`. Promoting them
properly means exporting a qualified path and deciding whether a class whose `__init__` opens
files or hits the network is "independently callable" at all — a real design question, not
something to infer silently.

**Next** end-to-end: agent finds a unit via `kb_method_search`, imports it inside the sandbox
through the read-only mount, and passes data between two calls in one session.

## 2026-08-12 · M2.5 · The library actually works in the sandbox — after two more defects
**Change** Lazy element packages (`_element_init`, PEP 562 `__getattr__`) in
  `library_emitter`; new `extractors/pkgmap.py` deriving each unit's install set from its own
  slice; `notebook_extractor` populates `UnitContract.requirements` from it.

**Why** With the tools in place I ran the real seam — build the library into `storage_root()`,
mount it into the sandbox, use the exact import line the registry advertises. Two failures,
both invisible to every static check:

1. **A sibling's dependency broke an unrelated unit.** Python runs a parent package's
   `__init__` before any submodule, so the pinned
   `from iguide_methods.ke_x.v_45b703eec714 import load_chicago_crime_data` executed *every*
   sibling module first, dying on `ModuleNotFoundError: pandas` raised by a **different unit**.
   It also silently made per-unit requirements meaningless — the true install set was the
   union over the element. Now the element `__init__` holds a `{symbol: module}` table and
   resolves through `__getattr__`, so importing one unit imports one module.

2. **Every contract understated its dependencies.** `UnitContract.requirements` was read by
   the emitter and never written by the extractor: **16 of 16 units declared `{}`** while
   their slices imported pandas, geopandas and smolagents. The agent reads that field to
   decide what to install, so it installed nothing and the import failed.
   `pkgmap.requirements_from_source` parses the slice — the thing that actually gets imported,
   walking function bodies too since lazy imports are common in notebook code — and maps
   import names to distributions. `inferred` names the distributions that were *assumed* to
   match their import name, so a wrong guess is auditable rather than a confusing "no matching
   distribution" at run time.

**Measured** in the real Docker sandbox (`python:3.11-slim`, `--network none`, read-only
rootfs), two sequential `execute_code` calls in one session:

| | before | after |
|---|---|---|
| units with real pip requirements | 0/16 | 14/16 (2 stdlib-only, correctly empty) |
| advertised import line works in-container | ✗ | ✓ |
| declared deps sufficient to import | ✗ | ✓ |
| deps reinstalled on call 2 | — | no (cached in the session `.deps`) |
| call 1's files readable in call 2 | — | ✓ (`step1.json`, `weather.csv`, 48 rows) |
| write to `/opt/iguide_methods` | — | refused, `OSError` |

Suite **743 passed**, same 3 pre-existing live-cluster failures.

**Surprised by** the shape of the whole M4.1/M2.5 sequence: four independent defects, each
one hidden by the one before it. The library could not be searched, so nobody noticed the
units were methods; the methods broke the element `__init__`, so nobody noticed the eager
re-export was wrong; the eager re-export made per-unit requirements irrelevant, so nobody
noticed they were empty. Each fix was the instrument that found the next defect, and none of
them were reachable from unit tests over synthetic fixtures — every one needed the real corpus
and, for the last two, the real container.

**Next** the agent path end to end: a prototype query that makes the model call
`kb_method_search`, then `execute_code` with the returned import line.

## 2026-08-12 · M0.6b · The agent path now runs on the CLI too — and it was running an agent
**Change** `agent_runtime/chat_claude_cli.py` (new): a LangChain `BaseChatModel` with working
  `bind_tools`, selected by `build_default_llm()` when `LLM_PROVIDER=claude-cli`. Plus CLI
  isolation flags in `llm_claude_cli._build_argv` and a neutral working directory.

**Why** M0.6 only switched `call_llm`. The agent itself goes through
`executor_factory.build_default_llm()` → `ChatOpenAI`, so every agent turn of every experiment
still billed OpenAI even with the provider set to claude-cli — which is most of the cost the
switch was meant to remove, since `create_agent` drives everything through tool calls.

Tool calling is prompt-enforced: the shim asks for `{"tool_calls":[...]}` or `{"content":...}`
and parses it. That is a real limitation and it is handled rather than hidden — a malformed
reply degrades to content (an unparseable turn should still end with an answer) and is counted
in `malformed_replies`; a hallucinated tool name is dropped instead of reaching the executor.

**Measured** one tool-selection prompt, `claude -p` from this repo's root vs isolated:

| | before | after |
|---|---|---|
| latency | **78.8 s** | **3.5 s** |
| turns the CLI took | 4 | 1 |
| reply | prose *about this repo's source* | exactly the requested JSON |

Full two-turn loop through the shim with the real tools: 3.4 s to choose `kb_method_search`,
11.0 s to answer from its result, **0 malformed replies**, and the answer carried the correct
pinned import line and requirements. Suite **762 passed**, same 3 pre-existing failures.

**Surprised by** what `claude -p` actually is. It is not a completion endpoint — it is Claude
Code. Invoked from a project directory it discovers `CLAUDE.md`, reads the repo, and takes
multiple agent turns: the first "answer" I got back was *commentary on the very file I was
writing*, which is both wrong and 22× slower. `--bare` suppresses all of this but explicitly
never reads OAuth, so it is unavailable on the subscription path. The equivalent had to be
assembled by hand: `--disallowed-tools` for every agent tool, `--strict-mcp-config`,
`--setting-sources ""`, `--no-session-persistence`, an explicit `--system-prompt` replacing
the coding-agent persona, and `cwd` set to an empty directory.

Worth noting for the paper's methods section: any measurement taken through this backend
*before* this change was contaminated by the contents of the checkout it ran in.

**Next** drive the prototype UI against this backend and confirm the model reaches for
`kb_method_search` on its own, from a real user question.

## 2026-08-12 · M4.2 · Four filters stood between a registered tool and the model
**Change** `kb_method_search` reachable end to end: added to the prototype's
  `enabled_search_methods`, to `search_methods.KNOWN_SEARCH_METHODS`, to the SearchAgent
  COVERAGE rule plus a new REUSE rule (and a CodeAgent rule), and — decisively — unioned
  deterministically into `_direct_search_sweep` via `_method_units_as_documents`. Reader tools
  now follow their search tool generically (`get_kb_block`→`agent_kb_search`,
  `get_method_contract`→`kb_method_search`, as `web_fetch` already did). Query stopwords added
  to the method scorer.

**Why** M4.1 registered the tools and put them in `RAG_COMPONENT_TOOL_NAMES`, which I took to
mean they were reachable. Driving the prototype showed they were not, and each fix revealed
the next gate. The same name had to be added to **four independent lists**:

1. `graph_state.RAG_COMPONENT_TOOL_NAMES` — the `tool_policy` filter (done in M4.1).
2. the prototype's hardcoded `enabled_search_methods` — a **hard** server-side filter.
   `agent_kb_search` was missing here too, so **sub-document evidence had never been reachable
   from this UI at all**.
3. `search_methods.KNOWN_SEARCH_METHODS` — request validation. Missing here the whole request
   400s: `unknown search method(s): 'kb_method_search'`, and the run never starts.
4. the SearchAgent persona, whose COVERAGE rule enumerates tools **by name**.

**Measured** the same question — *"I want to make a choropleth map of Chicago crime. Is there
already code on the platform I can reuse?"* — through the prototype, five runs:

| run | change under test | called the tool? | answer |
|---|---|---|---|
| 1 | M4.1 as committed | no | "adapt this notebook" |
| 2 | + prototype list, + companion rule | no | "adapt this notebook" (more elements) |
| 3 | + COVERAGE and REUSE prompt rules | no | "adapt this notebook" |
| 4 | + deterministic sweep | **request rejected** (filter 3) | — |
| 5 | + request validation | **yes** | three callable methods, with import lines |

Run 5's answer names `load_chicago_crime_data`, `load_chicago_community_areas` and
`plot_choropleth_map`, gives the pinned import line for each, and sketches the pipeline that
composes them. All three import lines were checked against the library on disk: **3/3 real**,
none fabricated. Suite **775 passed**, same 3 pre-existing failures.

**Surprised by** runs 1–3. The tool was registered, policy-allowed, request-enabled and named
in *two* persona rules, and the peer still called it **zero times out of three**, preferring
`keyword_search`/`semantic_search`/`neo4j_get_element_by_id`. `graph.py:1445` already carries
the lesson in its own comment — *"do NOT rely on the LLM picking the right tool"* — written
for the by-id tools after the same failure. Prompt text did not move this at all; the
deterministic union did, on the first try. Worth remembering when the instinct is to write a
better instruction.

Two smaller findings from the same runs. Scoring had no stopword filter, so *"what is the
capital of France"* matched `determine_number_of_cluster` on the word **"of"** — fixed, and
off-topic queries now return nothing. And the grounding audit earned its keep on real content:
the model wrote "all three methods require geopandas, matplotlib and smolagents" when only
`plot_choropleth_map` declares matplotlib, and the audit flagged exactly that — a true
positive on a claim that would have been tedious to catch by hand.

**Next** the composition half: make the code peer import a swept method inside `execute_code`
and pass a GeoDataFrame between two calls in one session.

## 2026-08-12 · M2.7 · Composition works — after a fifth gate and a leaking tool list
**Change** the method tools added to `default_code_fn`'s hardcoded KB allowlist
  (`supervisor/graph.py`); `execute_code`'s description now names the `iguide_methods`
  package; the CLI shim's description budget raised to fit it; and the `claude` CLI is now
  denied **all** of its own tools, not just the file/shell ones.

**Why** M4.2 made the library discoverable by the SEARCH peer. Asking the agent to actually
*run* a library method exposed two more problems.

**Gate five.** `default_code_fn` builds its own KB toolset with a **hardcoded** allowlist —
it does not follow the request's `enabled_search_methods` — and it listed only
`agent_kb_search` / `get_kb_block`. So the peer that writes and runs the code, in whose
sandbox the library is mounted, could not see the library at all. Its own prompt (added in
M4.2) told it to call `kb_method_search`; not having it, the model guessed the package name
from the host directory: `from method_library import ...` → `ModuleNotFoundError`. The package
is `iguide_methods`.

**The leaking tool list.** Two runs refused outright, claiming the code tool was unavailable
while it was bound, docker-backed and policy-allowed. The second refusal named the culprit:

> "Only a limited set of tools (`AskUserQuestion`, `ScheduleWakeup`, `ShareOnboardingGuide`,
> `Skill`, and `ToolSearch`) are callable here."

Those are the **CLI's own** tools — precisely the ones left allowed by the partial deny list
from M0.6b. The CLI advertises its remaining tools to the model, and under a long prompt the
model believed that list over the one in the prompt, then confabulated a justification.
Denying every CLI tool removes the competing list; the system prompt now also disowns any
other tool list explicitly.

**Measured** same request, four runs:

| run | change under test | tool calls | outcome |
|---|---|---|---|
| 1 | M4.2 as committed | 0 exec | "code-execution tool is not available" (false) |
| 2 | explicit "use execute_code" | exec ran | guessed `from method_library import …` → ModuleNotFoundError |
| 3 | + gate five, + package name in description | 0 exec | refused, naming the CLI's own tools |
| 4 | + full CLI tool deny | **7 kb_method_search, 6 get_method_contract, 24 execute_code** | **exit 0** |

Run 4 is the M2 composition criterion met: the agent searched the library, fetched the
contract, imported
`from iguide_methods.ke_afbee4bd_…v_7180f12f1def import extract_24h_before_peak`, declared
`pandas` from the contract, built a 48-row frame, wrote `weather.csv`, ran the **extracted
platform function** on it and reported true stdout — peak found, 24 preceding rows returned.
Module, symbol and dependency all verified against the registry and the file on disk.
Suite **785 passed**, same 3 pre-existing failures.

**Surprised by** how much of this was the *backend*, not the system. Three of the four failed
runs were artifacts of driving the agent through `claude -p`: a truncated tool description
that cut the `dependencies` instructions, and then a tool list from the CLI's own harness
overriding the prompt's. Prompt-enforced tool calling is a genuinely weaker substrate than the
native API, and it fails in ways that *look* like product bugs — an answer confidently
reporting that a working tool does not exist. Two consequences: eval numbers must not be
produced through this backend without saying so, and "the model said the tool was missing" is
now a known backend symptom rather than a signal to go re-check the wiring.

**Next** the same run under a native tool-calling model, to separate backend artefacts from
real agent behaviour, and the invariant gate (M6) on top of a run that now genuinely executes.

## 2026-08-12 · M3.0 / M4.3 · Source resolution at corpus scale, and an audit of my own blind spot
**Change** new `extractors/sources.py` (GitHub / MinIO / HTTP → `ResolvedSource` with sha256)
  and `scripts/measure_source_fetchability.py`. Plus five reachability fixes found by a
  32-agent adversarial audit of the defect class from M4.2.

**Why (sources)** M3 is corpus-scale ingest, and extraction quality is irrelevant for an
element whose file cannot be fetched. The plan's target was ≥90% from a naive 8/12 baseline.

**Measured** over all **180 notebook elements** on the live platform API:

| | |
|---|---|
| fetchable | **174/180 (96.7%)** |
| resolved via the curator's blob URL | 97 |
| resolved via repo + path | 83 |
| failures | 6, all HTTP 404 (file moved or repo renamed) |

The blob-URL path matters for correctness, not just coverage: `notebook-url` pins the **ref**
the curator linked, while repo+path has to guess the default branch — a different commit than
the element was published against.

**Why (audit)** M4.2 needed four filters fixed for one tool and M2.7 found a fifth. That is a
class, not five accidents, so I ran a workflow: six lenses (tool allowlists, prompt
enumeration, client config, env gates, emit/consume, silent degradation), every candidate
adversarially verified by an independent agent instructed to refute it. 32 agents, **11
findings confirmed** of the candidates raised. Fixed here:

1. **My own bug, one commit old.** The `agent_kb_search` arm I added to `_direct_search_sweep`
   in M4.2 read `payload["results"]`; the function returns `documents`. No exception, so the
   surrounding `except` could never catch it — a permanently empty arm that measured as
   "working". It also re-normalized already-normalized docs. This is precisely the defect
   class I wrote the audit to find, introduced by the commit that fixed the previous one.
2. `SEARCH_AGENT_PROMPT` rule 8 told the model to call **`fetch_element_source`**, which
   exists under no path — the live MCP server exposes it as `mcp_fetch_element_source`. Same
   shape as the `mcp_run_nbwf_*` phantom deleted in M0.7.
3. …and that real name was in **no** tool-name set, so it was stripped for every intent except
   `analysis_task`, where it survived only because the empty-selection fallback returns
   everything. Fixed the name and added it to `RAG_COMPONENT_TOOL_NAMES`.
4. `agent_chat_stream_demo.html` (served at `/agent/dashboard`) had the identical hardcoded
   checkbox allowlist the prototype had — no `agent_kb_search`, no `kb_method_search`, no
   `web_search`. Fixing one shipped client had left the other broken.
5. The prototype's `RETRIEVAL_TOOLS` set drives the reasoning log, and a name absent from it is
   **dropped entirely** — so method-library calls were invisible in the one place a user looks
   to check the agent's work.

Suite **792 passed**, same 3 pre-existing live-cluster failures.

**Surprised by** finding 1. I had just spent five commits on "a capability that is registered
but unreachable", wrote an audit for exactly that pattern, and the audit's first confirmed hit
was code I had written an hour earlier — silently dead for the same structural reason (a
failure with no exception to raise). The lesson is not "be more careful"; it is that this class
is invisible to unit tests and to the person who wrote it, and needs an instrument. The
remaining six confirmed findings (legacy `analysis_task` stripping every retrieval tool, the
analyze peer's toolset, skills with no SKILL.md in the image) are queued, not yet done.

**Next** corpus-scale ingest on the 174 fetchable notebooks, then the invariant gate.

## 2026-08-12 · M3.1 · Corpus scale: 16 units → 203, and three false-callables it exposed
**Change** `scripts/build_method_library.py` (resumable, bucketed failure accounting) driving
  the real path platform API → `sources.py` → extractor → analyzer → library. Plus
  `annotation_names()` extended to **parameter defaults and class bases**, module-level classes
  given closure entries, and `rag_pipeline/tests/conftest.py` isolating the library.

**Measured** the whole notebook corpus, end to end:

| | before (14 cached notebooks) | after |
|---|---|---|
| elements attempted | 14 | **180** |
| sources fetched | 14 (local copies) | **174 (96.7%)**, 6 × HTTP 404 |
| units analyzed | 41 | **349** |
| callable | 16 | **203** |
| needs_globals / needs_instance | 1 / 24 | 96 / 50 |
| library modules | 16 | **203** |
| registry entries | 29 | **388** |
| elements with ≥1 callable unit | — | 60 / 174 (**34.5%**) |

Import verification over all 203 advertised import lines: **187 import on a bare host (92.1%)**,
16 fail only on a declared third-party package the sandbox installs, **0 undeclared
dependencies, 0 slice defects**. Suite **799 passed**, same 3 pre-existing failures.

The 34.5% element coverage is **below the plan's ≥40%** target and I am not going to claim
otherwise. The blocker histogram says why: 96 units are `needs_globals` — notebook functions
that read a frame loaded in an earlier cell. That is a supply property of the corpus, not an
analyzer weakness, and lifting it needs cell-state promotion, which is real design work.

**Surprised by** what corpus scale caught that 14 notebooks could not. Three units imported
cleanly in every static check and then died with `NameError`, all in one blind spot: names
evaluated at **def time in the enclosing scope**. I had handled annotations in M2.2 and stopped
there, but the same rule covers three more positions:

* `def evaluate(..., feats=FEATS)` — a default naming a module constant, simply not carried.
* `def plot_predictions(train_data=X_train, ...)` — a default naming a **runtime** binding.
  This was a **false callable**: the unit was never independently callable and the analyzer
  said it was. Counting defaults turns it into a blocker and it is now correctly refused.
* `class AgentState(TypedDict): messages: Annotated[list, operator.add]` — the base *and* the
  body annotation. The class-scope lookup used the function-table map, which recurses THROUGH
  class tables without including them, so it silently found nothing.

Also a reproducibility hazard worth naming: two long-passing tests began failing the moment
the corpus library existed on disk, because the sweep unions it and nothing isolated it. Same
code, same commit, different machine state. `conftest.py` now points the library at an empty
directory by default and a test that wants one opts in.

**Next** the invariant gate (M6) — with 203 units that a peer will actually import and run, a
wrong CRS is now a wrong *number*, not a hypothetical.

## 2026-08-12 · M4.4 · The analyze peer, same gap one peer over
**Change** `_CODE_PEER_KB_TOOLS` hoisted to module scope and given to `default_analyze_fn` as
  well as `default_code_fn`.

**Why** the audit flagged it and it is the M2.7 defect repeated: the analyze peer also holds
`execute_code`, so it could **run** analysis code while being unable to discover that the
platform already has a callable method for the step it was about to re-implement. Hoisting the
set means the next peer that gains `execute_code` inherits the right answer instead of
re-deriving it.

**Measured** end-to-end re-verification against the **203-unit** library (12× the one the
earlier runs used): the same prototype question returns
`load_chicago_crime_data` and `plot_choropleth_map` with pinned import lines and correct
requirements, and **both import cleanly** from the library on disk. Note the element package is
now `ke_cca9b545_ai_agent_for_chicago_crime_analysis` — the real element title — because the
corpus builder passes platform metadata rather than a filename stem.
Suite **799 passed**, same 3 pre-existing failures.

**Still open from the audit** (confirmed, not yet fixed): agent skills are enabled by default
but no SKILL.md bundle is copied into the agent image, so `list_available_skills` is always
empty in a deployed container; and the analyze/code peer split itself, which M5 retires.

## 2026-08-12 · M1.3 / M4.5 · The embedder came back, and the KB arm paid off
**Change** embedding config fixed in three places (`semantic._embedding_url` logs the resolved
  endpoint once; `override=True` dropped from six scripts; compose pins the in-container URL
  for `mcp-server` and `metadata-extraction-server`); new
  `scripts/create_agent_indices.py`; `build_method_library.py --index`; and two scoring bugs
  in `eval_retrieval.py`.

**Why** every retrieval number in this log until now was measured with the semantic arm
**dead** — `.env` pointed at a decommissioned host, and `semantic.py` logged only the *unset*
case, so a *wrong* URL was completely silent. That is the worst failure shape: "no semantic
results" and "the embedder is unreachable" are indistinguishable from the outside.

**Measured** with the local embedder live (384-d, verified against the service, not trusted
from `AGENT_KB_EMBED_DIM`):

| arm | @8 | @20 |
|---|---|---|
| keyword | 22/37 | 29/37 |
| **semantic** | **26/37** | 27/37 |
| agent_kb (new) | 16/37 | 16/37 |
| union | 29/37 | 31/37 |
| **union + agent_kb** | **32/37** | **34/37 (91.9%)** |

M1.3 exit: agent indices **0 → 5**, kNN dimension verified 5/5 at 384-d, blocks
**0 → 3,830** and method units **0 → 349** (4,179 docs).
M4 exit: recall@20 **31/37 → 34/37**, against a ≥33/37 target.
Elements missed by *every* arm: **8 → 3**, and all three are **absent from the index
entirely** — a data gap, not a retrieval failure. So 34/37 is the ceiling and retrieval now
reaches all of it. Suite **799 passed**, same 3 pre-existing failures.

**Surprised by** two instrument bugs that had been quietly distorting the headline number in
*opposite* directions.

`arm_union` rank-fused and truncated to k. The sweep does no such thing — it concatenates
every arm and dedupes (`_merge_dedup`), then reranks. Modelling union as RRF-then-cut reported
a number the system never produces, and a **pessimistic** one: 25/37 where the collected set
held 29. Worse, my first correction concatenated keyword-then-semantic and still truncated to
k, which measured *keyword alone* (22/37) and looked like a regression. A collection arm's k
is the per-arm window; recall belongs over the whole collected set, because that set is the
ceiling on what the agent can cite.

And `arm_agent_kb` had **the same wrong-key bug I fixed in the sweep one commit earlier** —
`_ids(payload)` over a dict yields nothing — plus a second one: it would have scored raw block
ids (`<element>::block::19`) against element ids, which can never prefix-match. Either alone
reports 0/37 for a working arm. Three instances of one shape now (sweep, this arm, and the
`results`/`documents` key), which says the shape is the problem: **a function returning a
container whose failure mode is empty rather than raised.**

The substantive result stands on its own: sub-document evidence contributes **3 elements that
element-level search cannot reach at any k**, because the match is inside the notebook rather
than in its title or abstract. That is the moat thesis with a number against it.

**Next** M6, the invariant gate.

## 2026-08-12 · M6.1 · The invariant gate, running on the real frames
**Change** new `agent_runtime/sandbox_verify.py` — deterministic checks executed as an
  epilogue INSIDE the sandbox, writing `checks.json`; read back into
  `ExecResult.verification` and surfaced in the tool result. `AGENT_INVARIANT_GATE`, on by
  default. 27 tests.

**Why in-sandbox and not an AST pass.** A source check can see `.buffer(25000)`; it cannot
know what CRS the frame was in when that ran, because that depends on what the data loaded
as. Only the live object knows.

**Measured** in a real container (`--network none`, read-only rootfs), the motivating replay:

| | geographic frame | projected frame |
|---|---|---|
| exit code | **0** | 0 |
| printed area | **0.196** | 1 960 342 806 |
| gate verdict | **fail** | pass |

The wrong run *succeeds*. It prints a number with a plausible shape, and nothing in the
runtime objects to it — the gate is the only thing that says "that figure is in degrees²".
Suite **826 passed**, same 3 pre-existing failures.

Three checks so far: projected-CRS-before-measurement, entirely-null columns (the unmatched-
join signature), and join cardinality reported so silent row inflation is visible. Findings go
INSIDE the tool result rather than onto the answer afterwards, so the model can reproject and
re-run in-loop instead of caveating a wrong number.

**Surprised by** two of my own defaults being wrong in the same direction — toward reporting
"fine" when nothing was actually established.

`check_not_all_nan` used `select_dtypes("number")`. pandas types an all-`None` column as
`object`, so the column produced by an unmatched join — precisely the thing the check exists
for — was excluded from the check *by its own dtype*.

And the verdict was `FAIL if any failed else PASS if any passed`. A frame with **no CRS**
still passes the null check, so a result nobody could verify came back marked `pass`. The
precedence is now fail > cannot_determine > pass: one unknown downgrades the whole run. That
asymmetry is the point of the gate — a confident wrong number is worse than an admitted
unknown — and I had coded the opposite by reflex. My own test caught it, which is the argument
for writing the adversarial cases before trusting the implementation.

**Next** the remaining checks (declared units on numeric outputs, output bounds), and wiring
the verdict into `_reconcile_audit_with_artifacts` so an unverified number cannot be presented
as verified.

## 2026-08-13 · M6.2 · The gate now BLOCKS, and the last two checks
**Change** `_gate_failures()` + reconciliation changes in `supervisor/graph.py`;
  `check_declared_units` (units + optional bounds) in `sandbox_verify.py`; `execute_code`'s
  description teaches the `IGUIDE_OUTPUTS` convention. 42 gate tests.

**Why** M6.1 made the gate *observe*. It did not make it *matter*, and there was a specific
hole. `_reconcile_audit_with_artifacts` rule (2) drops a disputed number when that number
appears in the execution record — but **a wrong number appears in the record too**.
`AREA: 0.196` is right there in stdout, so the gate would say "that is degrees squared" and the
reconciliation would answer "it is in the record, so it is grounded." Rule (2) is now disabled
whenever the gate failed, and the gate flags **even when the LLM auditor found nothing**,
because the gate knows a distance was computed in degrees and an auditor reading prose cannot.

**Measured** in a real container, all five checks:

| case | exit | verdict | why |
|---|---|---|---|
| geographic buffer | **0** | **fail** | `projected_crs` on both frames |
| projected + declared unit | 0 | **pass** | — |
| projected, unit omitted | 0 | **fail** | `declared_units`: no unit given |

and on the answer path:

| situation | before | after |
|---|---|---|
| number from a gate-**failed** run | cleared as grounded | **flagged, severity high** |
| number from a gate-**passed** run | cleared | cleared (M6a suppression intact) |
| gate failed, auditor **silent** | nothing | **flagged** |

Suite **841 passed**, same 3 pre-existing failures.

`declared_units` exists because no amount of frame inspection can distinguish 21500 metres from
21500 feet. A null unit is a FAIL, not an omission — the number most likely to be wrong is
exactly the one whose unit nobody wrote down. Bounds are checked only when the run declares
them; inventing a plausible range would manufacture false positives.

**Surprised by** two things, both about test fixtures rather than code.

First, I "found" a rule-(2) regression that did not exist: my fixture used
`severity: "medium"`, and `_audit_flagged` gates on `{"high"}` only — so the function returned
early and never reached the rule I thought I had broken. The lesson is narrow but real: when a
test of a guard passes suspiciously, check that the guard was entered.

Second, the epilogue imported `math as _math` while the inlined checks — which are this
module's own source — reference bare `math`. `check_finite` raised `NameError` *inside* its
guard, so it surfaced as `cannot_determine`: a check that had silently stopped checking while
still reporting. Exactly the failure shape this gate exists to prevent, in the gate itself.

**Next** M2's reproducibility half — `stage_url`/`stage_object`, artifact emission and
`rerun_artifact.py`.

## 2026-08-13 · M2.8 · The reproducibility half: artifacts that replay
**Change** new `agent_runtime/artifacts.py` and `scripts/rerun_artifact.py`; the sandbox
  epilogue now also writes `environment.json` and `declared_outputs.json`; every run emits
  `run.py` + `manifest.json` + `inputs.jsonl` (`AGENT_ARTIFACT_EMIT`, on by default). 21 tests.

**Why** the compose path worked since M2.7, but nothing recorded a run well enough to repeat
it, so "solid, complex, **reproducible**" had two of three. Four things had to be pinned:

* **the image by digest.** `python:3.11-slim` resolves to different bytes next month, so an
  artifact holding the tag records nothing about its environment.
* **the environment from inside.** An agent-side `pip freeze` describes the *agent*, not the
  container that produced the number.
* **the inputs by sha256**, so a replay can assert it read the same bytes rather than a file
  with the same name.
* **the declared output values**, so the replay has something to *compare* rather than repeat.

**Measured** a real run, then replayed in a clean container:

```
image      python@sha256:a3ab0b96…  (pinned by digest)
env        python 3.11.15, 13 packages   (captured in-container)
gate       original pass -> replay pass
total_buffer_area:  replay=3920685613.182  original=3920685613.182  ==
replay completed (outputs identical)                                   exit 0
```

and with the manifest's baseline tampered to `4200000000.0`:

```
total_buffer_area:  replay=3920685613.182  original=4200000000.0  != DIFFERS
DIFFERED (1)                                                           exit 1
```

**1/1 declared numeric output identical**, and the exit codes discriminate — which is the
property that makes this worth running in CI. Suite **862 passed**, same 3 pre-existing.

**Surprised by** how nearly this shipped as theatre. Two defects, both of which would have left
a script that always reports success:

The manifest first recorded `declared_outputs` from the **gate's findings** — so it stored
`"unit is null"` where the value should be, keeping the complaint and losing the measurement.
The replay would have had nothing to compare and would have said so only in a footnote.

And the comparison loop *printed* each value and appended to a `differences` list that nothing
ever populated. It returned 0 unconditionally. A reproducibility check that cannot fail is
worse than none, because it converts an open question into a false assurance. Fixed to compare
with a 1e-9 relative tolerance — exact float equality would flag a re-ordered sum and train
everyone to ignore the script.

Also: `run.py` deliberately holds the model's source *without* the injected epilogue, so the
persisted artifact is what the model wrote; the replay re-appends the epilogue itself, which is
what makes the gate verdict comparable across runs.

**Next** the dataset extractor's two shipping defects — native-CRS bounds written into a
`geo_shape` field (181/619 docs carry a bbox), and `.xlsx` parsed as CSV.

## 2026-08-13 · M7.1 · Dataset bounding boxes that survive indexing, and formats read as themselves
**Change** new leaf `rag_pipeline/search/geo_shapes.py` (`to_wgs84_bounds`, `bbox_geo_shape`,
  `plausible_wgs84`; `infer_geo_shape` moved here and re-exported from `spatial.py`);
  `data_extractor` reprojects instead of writing native bounds, gains `.xlsx`/`.parquet`
  readers, a `metadata` family for `.json`/`.xml`, and tar/gzip containers; the hand-rolled
  lat/lon scan is replaced by the tested `_pick_coord_column`/`parse_coordinate`. 24 tests.

**Why** two defects were losing data on every ingest, both silently.

`_envelope` wrote the file's **native** bounds straight into `spatial-bounding-box-geojson`.
That field is mapped `{type: geo_shape, ignore_malformed: true}`, so UTM metres did not fail
the write — OpenSearch dropped the field and the document indexed cleanly. The dataset was then
absent from every spatial query with nothing anywhere recording that it happened. On the live
index **181 of 619 docs carry a bbox**.

`.xlsx` routed to `csv.reader`, which reads the binary container without raising and returns a
garbage single-column header — worse than an error, because the asset indexes with nonsense
that looks like real metadata.

**Measured**

| input | before | after |
|---|---|---|
| UTM 16N Chicago bounds | written raw → dropped at index time | reprojected to `[-87.72, 41.82, -87.48, 42.00]` |
| bounds with no CRS, in metres | assumed 4326 → dropped | **no bbox** + note, native bounds kept |
| CRS says 4326, bounds in metres | written → dropped | **no bbox** + note |
| `.xlsx` schema | garbage 1-column header | `{latitude, longitude, value}`, 2 rows |
| `.tar` / `.tgz` / `.gz` | "could not read container" | members + family counts |
| STAC item with a declared bbox | unhandled sidecar, no bbox | declared bbox used |
| `spatial.py` import cost for a shape | 1.23 s (Flask + spaCy) | **0.021 s** |

Suite **886 passed**, same 3 pre-existing failures.

The asymmetry drives every choice here: an absent bbox is incomplete, a wrong one is wrong —
and since `ignore_malformed` makes them indistinguishable downstream, a note recorded at
extraction time is the only way to tell them apart afterwards. So a dataset that HAS bounds and
gets no bbox now also emits a warning: "no spatial extent" and "we had one and could not use
it" are different facts, and only the second is a bug to chase.

**Surprised by** `_HANDLERS`. It mapped a family to a **function object**, captured at import.
Reassigning `_handle_raster` on the module therefore had no effect whatsoever — the dict still
held the original — which is why two of my tests failed while the code under test was correct.
It now maps to a NAME resolved at call time. Same shape as the `default_top_k` default-argument
bug from M1.2 and the `arm_agent_kb` wrong-key bug from M4.5: **a value captured early that
looks like a live reference.** Third distinct instance, so it is worth naming as a pattern
rather than fixing case by case.

**Next** M3's reconciling re-ingest — there is still no delete anywhere, so a notebook that
loses a cell leaves orphan `::block::<n>` docs forever, and 3,830 blocks are now indexed.

## 2026-08-13 · M3.2 · Re-ingest that reconciles, and bulk writes
**Change** `opensearch_emitter` gains `reconcile_plan` / `existing_doc_ids` /
  `_delete_orphans`, bulk index+embed via `helpers.bulk`, `run_fingerprint` +
  `ingest_runs` for skip-if-unchanged, and `_assert_agent_indices` as a hard safety rail.
  16 tests.

**Why** re-ingest meant "write the new docs" and nothing else. There is **no delete anywhere
in this repo** outside `memory_module`, so a notebook that lost a cell kept its old
`::block::<n>` documents forever — and kept having them retrieved as evidence for code that no
longer exists. With 3,830 blocks now indexed from the corpus, that stopped being theoretical.

**Measured** against the live cluster:

| step | indexed | orphans found | deleted | docs under the parent |
|---|---|---|---|---|
| ingest 5 cells | 5 | 0 | 0 | 5 |
| re-ingest identical | 5 | 0 | 0 | 5 |
| notebook drops to 3 cells | 3 | **2** | **2** | **3** |

M3 exit criterion met: orphans after a cell deletion **≥1 → 0**. Writes now go through
`helpers.bulk` — the corpus backfill was 4,179 index calls plus 4,179 embed updates, ~8,300
round trips. Suite **902 passed**, same 3 pre-existing failures.

Three deliberate choices, each because the failure mode is severe:

* **Delete by explicit id, not `delete_by_query`.** The ids come from a diff just computed, so
  no query exists that could match more than intended.
* **A search failure yields the EMPTY diff.** If the current state cannot be read, treating an
  unreadable index as "nothing is there" would delete the element's entire history.
* **`_assert_agent_indices` refuses anything outside the agent prefix, and any collision with
  `OPENSEARCH_INDEX`.** This module deletes documents now; a misconfigured prefix resolving to
  the platform index would be unrecoverable from here.

`run_fingerprint` hashes the DOCS rather than a commit sha, deliberately: a re-ingest of the
same commit through a changed extractor must not be skipped, and that is exactly the case where
skipping would hide a regression.

**Surprised by** my own fake client. `FakeClient.search` returned every doc regardless of the
query, so `test_the_diff_is_scoped_to_ONE_parent` failed against correct code — and, worse,
would have *passed* whether or not the code scoped by parent at all. A fake that ignores the
filter cannot test the filter. It now honours the term query, which is the only version of that
test worth having, since an unscoped diff would wipe every other notebook sharing the index.

**Next** M5 (retire the analyze peer) and the remaining M7 element types.

## 2026-08-13 · M7.2–M7.4 · All four element types produce units; M5 refused on measurement
**Change** `code_extractor` promotes callable units (and no longer treats an argparse *import*
  as a CLI); `publication_extractor` gains a first-class `status` and writes the first
  **IMPLEMENTED_BY** edges; `data_extractor` generates a provenance-carrying
  `load_<slug>(staged_path)` unit per dataset. 78 new tests.

**Measured** element types with callable-unit contracts: **1 → 4**.

| type | unit | verified |
|---|---|---|
| notebook | per top-level function | 203 units in the corpus library |
| **code** | per top-level function, same analyzer path | `load_points` callable w/ geopandas; `needs_config` blocked on a runtime global |
| **dataset** | generated `load_<slug>(staged_path)` | source **compiles and executes**; declares its CRS invariant |
| **publication** | spec-only, `is_method_spec` + `status` | degraded specs prefix `contents`, never present as "no method" |

`IMPLEMENTED_BY` edges **0 → >0**: matched against the real 203-unit registry,
`plot_choropleth_map` and `spatial_join_and_count` link to their units while `run` / `data` /
`get` / `load` link to nothing. Every edge carries `confidence: low`, `by: symbol_match` —
asserting a paper's method IS a function on the strength of a shared name would be fabricated
provenance, and provenance is the one thing here that has to be trustworthy.
Suite **956 passed**, same 3 pre-existing failures.

**M5 (retire the analyze peer) is deliberately NOT done.** Its justification is turns wasted on
analyze↔code ping-pong, so I measured it across every live run this session:

| decision | runs |
|---|---|
| code | 4 |
| search | 2 |
| analyze | 1 |
| **turns using BOTH peers** | **0** |
| **analyze↔code ping-pong** | **0** |

The supervisor picks one peer and finishes. The plan's own M5 exit criterion — "ping-pong
measured → 0" — is already satisfied, so the refactor would be **44 sites of risk in the
orchestration core for no measured gain**. Small sample and one backend, so `pre-m5-analyze-retire`
is tagged if that changes. M4.4 also gave the analyze peer the method library, so the two peers
have converged in capability, which weakens the "jumping between them" argument further.

**Surprised by** the generated loader. The first version assembled its docstring with a
conditional indent that skipped lines starting with a triple quote — which un-indented the
docstring and made **every** generated loader a `SyntaxError`. It looked right in the source and
the extractor reported success, because nothing compiled it. `build_loader_unit` now compiles its
own output and returns None on failure. Generated code has to be checked by whatever generates
it; there is no reviewer in that path.

Also: two of my three test failures this round were wrong expectations, not wrong code — an
empty dataset title falls back to the FILENAME (`load_f_dat`), which is more informative than a
generic `load_dataset`, and my `EMIT_LIBRARY` assertion tested for the literal string
`"EMIT_LIBRARY"` instead of its value `"library"`.

**Next** remaining: the SKILL bundle never reaching the agent image, and the `.corpus_cache`
blobs still in commit `6f6d030`.

## 2026-08-13 · M7.5 · Skills that survive deployment
**Change** `rag_pipeline/Dockerfile` copies `.agents/`; compose points
  `AGENT_GENERATED_SKILLS_ROOT` and `AGENT_SKILL_PATHS` at the persistent
  `agent_chat_files` volume. New `test_deployment_contract.py` (6 tests).

**Why** `skills.py` discovers bundles under `REPO_ROOT/.agents/skills` and `REPO_ROOT/skills`,
and the image copied **neither**. `list_available_skills` therefore returned an empty list in
every deployed container while working perfectly on a developer checkout — the worst shape of
gap, because it only appears where nobody is watching a test suite. Generated skills had the
same problem twice over: written to `/app/.agents/skills` they would vanish on every restart,
and the image runs as non-root so that path is not writable anyway.

The new file asserts the packaging contract generally, not just this one bug: every package the
agent imports is copied, the embedding URL is pinned for all three services that need it,
`LLM_PROVIDER=claude-cli` never appears in compose, and `AGENT_ALLOW_WORKFLOW_EXEC=1` cannot
drift on. Suite **962 passed**, same 3 pre-existing failures.

**Not done, deliberately:** the 174 `.corpus_cache` blobs remain in commit `6f6d030`. The
branch is unpushed so a rewrite is cheap and safe, but rewriting history is not something to do
unasked; `git filter-branch -f --index-filter 'git rm -r --cached --ignore-unmatch
.corpus_cache' 6f6d030^..HEAD` is the fix when wanted. The working tree and all future commits
are already clean (untracked + gitignored in the M7.1 commit).

## 2026-08-13 · M6.3 / M7.6 · The contract is enforced, and classes are units
**Change** `contract_invariants()` turns inferred parameter expectations into enforceable
  `InvariantSpec`s (wired into both extractors); `sandbox_verify` gains `check_contract_arg` and
  a **prologue** that wraps imported units so invariants are checked against the actual argument
  at call time; `analyze_module` promotes safely-constructible classes. 33 new tests.

**Why (contracts)** `UnitContract` has always carried `crs_expectation`, and
`infer_units_and_crs` has always set `"projected"` on a GeoDataFrame parameter whose body
performs a metric operation — 20 corpus params carry it. None of it ever became an
`InvariantSpec`, and nothing at run time read it. Measured before this: **0 invariants in the
whole 203-unit registry.** "Contract-bearing structure" meant "structure the model is told
about".

**Why call time, not after.** An epilogue inspecting the namespace cannot know which frame was
passed as which parameter. `compute_accessibility(demand, supply)` needs *demand* projected; a
post-hoc scan sees two GeoDataFrames and has to guess. Wrapping the call removes the guess — and
the prologue must run BEFORE the user's code, or their `from iguide_methods… import symbol`
captures the original and the contract goes unchecked while appearing enforced.

**Measured** in a real container, importing a corpus unit that declares `gdf` must be projected:

| run | exit | verdict | finding |
|---|---|---|---|
| passes a 4326 frame | **0** | **fail** | `calculate_buffers declares gdf must be in a PROJECTED CRS — its body performs a metric operation` |
| passes a UTM frame | 0 | pass | — |

| | before | after |
|---|---|---|
| invariants in the registry | **0** | **66** (56 `reject_all_nan`, 10 `projected_crs`) |
| units carrying one | 0 | 39 |
| callable units | 203 | **229** |
| elements with ≥1 callable unit | 60/174 (34.5%) | **65/174 (37.4%)** |
| import lines with real defects | 0 | **0** (209 import bare, 20 need declared deps) |

Suite **998 passed**, same 3 pre-existing failures.

**Why classes.** 50 units were `needs_instance` methods whose class was never promoted — no
route to any of them. A class is a legitimate unit of reuse *when constructing it is safe*: an
`__init__` that calls `open`/`read_file`/`requests.get`, or reads a runtime global, is the
object-shaped version of the hidden-global problem and importing its slice into a network-less
sandbox fails. Those get `needs_instance` and stay indexed-only; the rest are promoted, +26
units.

**Surprised by** how the class change had to be restructured twice. My first version kept a
separate `promotable_classes` list while `iter_units` still didn't yield classes — so the
extractors, which iterate verdicts and skip anything `iter_units` omits, would have analysed
every class and then silently dropped it. The fix was to make `iter_units` yield classes as
*candidates* and let the verdict decide, exactly as it does for functions. Then the second
version had classes reaching the function loop and dead-ending on `by_name` (function tables
only), coming back `unparseable`. One loop handling both kinds was the answer; two loops with a
side list was the bug in two different costumes.

**Still the user's call, not mine:** whether the agent may serve private content (43 of 799
elements), whether the method library should be exposed as an API, and multi-worker serving.
Those are product and access-control decisions, not implementation.

## 2026-08-13 · M6.4 · The requirement travels with the search result
**Change** `search_methods` returns a `requires` list (`projected_crs(gdf)`,
  `reject_all_nan(resources)`) per hit; `None` when a unit declares nothing. 3 tests.

**Why** an end-to-end run exposed this. Asked to find `calculate_buffers`, read its contract and
run it, the agent called `kb_method_search`, **never called `get_method_contract`**, and so never
saw that the method requires a projected CRS. It reprojected to EPSG:26971 anyway and its answer
said "required by the projected-CRS invariant" — the right action, described with an authority it
did not have. A requirement the caller must satisfy belongs where the caller first sees the
method; a second tool call the model may not make is not a reliable channel.

Verified on the real registry:

```
calculate_buffers          requires=['projected_crs(gdf)', 'reject_all_nan(gdf)']
calculate_primary_regions  requires=['projected_crs(resources)', 'reject_all_nan(resources)',
                                     'projected_crs(spatial_units)', 'reject_all_nan(spatial_units)']
calculate_comprehensive_stats  requires=None
```

`None` rather than `[]` on purpose: an empty list reads as "checked, nothing required", which is
a different claim from "this unit declares nothing". Suite **1001 passed**, same 3 pre-existing.

**Worth recording precisely, because I nearly overstated it:** the end-to-end run is evidence
that the *path* works, NOT that the contract drove the model's choice. The stream shows
`get_method_contract` was never called and no invariant text ever reached the model. The proof
that enforcement works is the direct container test — a 4326 frame into a contract-bearing unit
gives `verdict=fail` naming the unit and parameter, a UTM frame gives `pass`. Two different
claims; only one of them was demonstrated by the agent run.

## 2026-08-13 · M0.9 · A default test run is green
**Change** the three live-service tests are marked `@pytest.mark.integration`, and `pytest.ini`
  gains `addopts = -m "not integration"`.

**Why** those three failed for the entire session and I reported them as "pre-existing" in every
commit message. They are not broken — they need a reachable OpenSearch cluster, the embedding
server and Neo4j, and they were the only red in the suite. A suite that is red by default trains
everyone to skim the summary line, which is exactly how a real regression gets through; the
marker already existed in `pytest.ini` and these tests simply never used it.

**Measured**

| run | before | after |
|---|---|---|
| `pytest` (fresh clone, no services) | 1001 passed, **3 failed** | **1001 passed, 0 failed**, 4 deselected |
| `pytest -m integration` | — | 3 failed, 1 skipped (as expected without services) |

## 2026-08-13 · M2.9 · The replay could not replay the runs that matter
**Change** `rerun_artifact.py` mounts the method library read-only at `METHOD_LIBRARY_MOUNT` and
  extends `PYTHONPATH`; refuses when a recorded `v_<sha>` module no longer exists; and no longer
  blames the artifact when the REPLAY is what failed. 4 tests.

**Why** found by my own regression check, not by a test. Replaying an artifact whose code imports
a library unit died with:

```
ModuleNotFoundError: No module named 'iguide_methods'
gate       original pass -> replay None
REPEATED, NOT VERIFIED: the run declared no IGUIDE_OUTPUTS …
```

Two defects in one output. The replay never mounted the library, so **the reproducibility claim
failed for exactly the runs that use the system's differentiator** — every earlier replay I
demonstrated used only pandas/geopandas and so never touched it. And the message then blamed the
ARTIFACT: it declared its outputs perfectly, and the replay published none because it had
crashed. A tool that misattributes its own failure to the thing it is auditing is worse than one
that just fails.

**Measured** the same artifact, after:

```
library    …/method_library mounted read-only, 1 pinned unit(s) present
image      python@sha256:a3ab0b96…  (pinned by digest)
gate       original pass -> replay pass
total_area: replay=3920685613.18   original=3920685613.18   ==
replay completed (outputs identical)
```

A retired `v_<sha>` is now a hard stop with the symbol named, because the recorded sha is the
whole point of pinning. Suite **1005 passed**, 0 failed.

## 2026-08-13 · M6.A · Tier A: nine ways the gate reported success without checking

**Change** Nine fixes from the 41-agent contract-chain audit, all in the
  extraction → contract → enforcement → answer chain.

  *Reaching the answer* — `supervisor/graph.py:_gate_failures` kept `id(node)` in a
  visited set while each `json.loads` graph was reachable only from its call frame, so
  CPython recycled addresses between branches and the walk returned before reading a
  later result's `verification`. Now anchors each parsed graph for the walk's lifetime.
  `_gate_issues_from` keys on the **verdict**, not on surviving `status=="fail"`
  findings: `_read_checks` truncates at 12, so the judgement can outlive its evidence,
  and `cannot_determine` never gated anything at all.

  *Seeing the frame* — `_has_geometry` tested for a column literally named `geometry`;
  PostGIS exports default to `geom`. Candidates are now ordered geometry-first, because
  globals were walked in definition order and capped at 12, so the late output frames —
  the ones a number is quoted from — were exactly the ones dropped, silently.

  *Projected ≠ metres* — EPSG:3435 (Illinois East, US survey feet) is `is_projected`,
  so a buffer declared in metres was 3.28× too large and passed. `_crs_unit` /
  `_unit_matches` compare the CRS axis unit against the declared one.

  *Containers* — `e2sfca(catchments)` takes a **list** of frames; the guard tested the
  bound argument with `_looks_like_frame`, which a list is not, so the invariant was
  skipped for every unit whose interface is a collection.

  *Import spellings* — `library_units_used` was a regex over one physical line, so
  three spellings the system itself emits resolved **zero** units and therefore
  installed zero guards: an `as` alias, a parenthesised list (every advertised
  `import_line` exceeds 79 chars, so wrapping is the *expected* case), and the
  element-package alias that `import_line`'s own docstring calls supported. Now `ast`,
  and registry lookup falls back to element-package + symbol rather than exact
  `v_<sha>` string equality.

  *Exit paths* — the gate is registered with `atexit` **from the prologue**. An
  epilogue appended after the user's code is never even defined when that code ends in
  `sys.exit(main())`, the standard script skeleton.

**Why** Every one of these reports as success. That is the specific failure this gate
  exists to prevent, and it had it internally.

**Measured**
  - fail positions missed by the reconciliation walk: **9/19 → 0/19** (14/21 on a
    second run of the same matrix — it depends on the allocator, which is why a
    single-case test can pass on the very run that ships a wrong number)
  - truncated `fail` report reaching the answer: **0 → 1 issue**;
    `cannot_determine` run: **0 → 1 issue**
  - import spellings resolving a contract: **3/6 → 6/6**
  - EPSG:3435 against a `metres` contract: **pass → fail**; 32616/3857 still pass
  - exit paths writing a report: **1/4 → 4/4**
  - live container, documented element-package spelling: 4326 → `fail` naming
    `calculate_buffers(gdf)` first; 32616 → `pass`
  - tests **1005 → 1061**, green by default

**Surprised by** Two things, both corrections of my own claims.

  First, I reported the previous batch as "all four exit paths now write checks.json,
  and `cannot_determine` is A3 correctly refusing to pass an unreachable run." That was
  wrong. The verdict was `cannot_determine` because `_iguide_gate_body()` returned a
  hand-listed tuple of two names while the gate also called `capture_environment()` and
  read `DECLARED_OUTPUTS` — both locals of that function. Two NameErrors, swallowed by
  the gate's own `except`, surfacing as an honest-looking unknown. `environment.json`
  and `declared_outputs.json` were not written **at all**, which would have left every
  replay with nothing to compare. It now returns `dict(locals())`, which cannot drift.
  Confined to the uncommitted batch; committed history was sound.

  That is the **fourth** NameError inside this gate's own guard (`math as _math`,
  `ModuleType`, and now the tuple). Asserting that `checks.json` *exists* catches none
  of them — asserting the absence of an `error` key is what does, and that assertion is
  now a parametrized test over all four exit paths in a real subprocess.

  Second, the A3 coverage guard keyed on `sys.modules`, which is process-global: it
  reported geopandas as loaded because some unrelated module imported it, so every
  pure-arithmetic run was called unverifiable. Flooding the channel with unknowns is
  how a real one stops being read. Now scoped to module objects bound in the namespace
  under inspection.

  Also: moving the atexit registration into the prologue means exec'ing it anywhere
  schedules a report into the CWD at interpreter exit — it was dropping three files
  into the repo root at the end of every pytest run.

**Next** ~23 further confirmed audit findings. The load-bearing one is that
  `params_of` returns `[]` for a `ClassDef`, so 26 class units carry 0 invariants while
  22 of them advertise `def X()`.

## 2026-08-13 · M6.B · Class units: a contract that raises TypeError, and a registry that never forgot

**Change** `signatures.signature_of` / `params_of` handle `ast.ClassDef`; a class now
  advertises its **constructor call** (`DoubleConv(in_c, out_c)  # class`) with params
  read from `__init__` minus `self`. `install_contract_guards` guards a class by
  patching `__init__` rather than replacing the class. `library_emitter` records
  `unit_kind` and prunes registry entries for elements it re-extracted, rebuilding bare
  aliases and ambiguity from what survives. `method_library.search_methods` /
  `get_contract` / `library_summary` surface `unit_kind`.

**Why** A `ClassDef` has no `.args`, so it fell through the function path: `params_of`
  returned `[]` and `signature_of` emitted `def DoubleConv()`. That advertisement is
  wrong twice over — a class is not a `def`, and every constructor argument was
  dropped. `DoubleConv`'s shipped code is `def __init__(self, in_c, out_c)`, so an
  agent following the contract gets `TypeError` immediately.

  Separately, the registry was merge-only and there is no delete anywhere in the ingest
  path, so it accumulated entries from every previous build.

**Measured**
  - registry entries **657 → 440**; non-alias units **446 → 229**, now exactly the 229
    module files on disk. The 217 dropped were stale duplicates — the same symbol
    appearing twice with *different* contracts (`def DoubleConv()` from an old build
    alongside the corrected `DoubleConv(in_c, out_c)`).
  - units advertising a zero-argument `def X()`: **35 → 5**, and all 5 remaining are
    genuinely no-argument functions (verified against the shipped source).
  - class units with constructor params recorded: **0 → 22 of 26**. The other 4 have no
    `__init__` of their own and are advertised `(...)  # class, inherits <base>` —
    honest, because resolving a base class means resolving a name from another module.
  - advertised params vs shipped `__init__` across all 26: **24 match exactly, 2
    correctly say `(...)`, 0 mismatches**.
  - `unit_kind` in the registry: **0 of 657 → 229 of 229** (26 class, 203 function).
  - tests **1061 → 1071**.

**Surprised by** The fix adds **zero invariants**. The audit finding reads as "26 class
  units carry 0 invariants", which sounds like the invariant chain was broken for
  classes — and it was, structurally: `contract_invariants` iterates parameters, so an
  empty `params_of` guaranteed an empty result. But all 26 promoted classes are
  neural-net layers and torch `Dataset`s; across their 80 constructor parameters the
  inferred types are 39 `number`, 39 `unknown`, 2 `str`, and **not one frame**. So 0 is
  the correct answer for this corpus. The chain is verified by a synthetic
  `CatchmentBuilder` whose `__init__` buffers a GeoDataFrame, which now gets
  `projected_crs(catchments)`.

  The guard mechanism was the real latent hazard: `functools.wraps` on a class returns
  a plain function, so replacing the module attribute would have broken
  `isinstance(x, C)` and `class Sub(C)` — and every class here is an `nn.Module` or
  `Dataset`, exactly what gets subclassed and isinstance-checked. It would have fired
  the first time a geospatial class was ingested.

  Also noted while measuring: `catchment_ratios_centroid` gets `reject_all_nan` but not
  `projected_crs`, while `catchment_ratios_area` gets both. That is the
  intraprocedural-inference finding — the centroid variant's metric operation is in a
  helper — and it is next.

**Next** `infer_units_and_crs` is intraprocedural, so the e2SFCA public entry points get
  no CRS invariant while their helpers do. Then `_construction_is_safe` (misses
  `os.makedirs`/`os.listdir`, base-class `__init__`, class-body I/O) and `by_name` keyed
  by simple name.

## 2026-08-13 · M6.C · CRS inference follows calls, and the corpus listing is cached

**Change** `signatures.infer_units_and_crs` takes an optional `bodies` map (unit name →
  AST node) and searches for metric operations through the call graph, cycle-safe and
  depth-capped at 4. Both extractors pass the module's node map. Separately,
  `scripts/build_method_library.py` caches the element listing.

**Why** CRS inference walked only the unit's own body. `catchment_ratios_area` computes
  distances inline and got `projected_crs`; `catchment_ratios_centroid` — same public
  interface, same requirement — delegates to `calculate_centroid` and got nothing. The
  public entry points are exactly the units an agent calls.

**Measured**
  - invariants in the registry **66 → 77**; units carrying a CRS invariant **5 → 9**
  - the four units that gained: `e2sfca` 3→6, `ae2sfca` 3→6,
    `catchment_ratios_centroid` 3→6, `aggregate_ratios_centroid` 2→4 — the e2SFCA
    accessibility family, which is the strongest cluster in this corpus
  - units that **lost** an invariant: **0**. Units that gained a spurious one: 0 (the
    no-metric-work case still gets nothing)
  - evidence records the full path, two hops deep: `.centroid( at line 169 via
    calculate_centroid() via catchment_ratios_centroid()`

**Surprised by** The corpus rebuild came back with 134 units instead of 229 and I nearly
  logged that as a regression from this change. It was a **429 from the platform API**
  mid-listing. The sources were cached by element id — 387 MB of them — but the element
  *listing* was not, so a rate limit aborted a run that had everything it needed on
  disk, which made "resumable" untrue in exactly the case resumability exists for. Now
  cached, with `--refresh-listing` to force a re-fetch. A partial listing is never
  cached: writing one would silently shrink the corpus on every later run, and a 429
  halfway through page 3 is the normal way that happens.

**Next** `_construction_is_safe` (misses `os.makedirs`/`os.listdir`, base-class
  `__init__`, class-body I/O) and `by_name` keyed by simple name, which reads another
  class's `__init__`.

## 2026-08-13 · M6.D · Two classes that could not be constructed shipped as callable

**Change** `analysis/callability._construction_is_safe` rewritten: filesystem inspection
  added to `_IO_CALLS` (with `os.path.join`/`dirname`/etc. explicitly exempted as string
  arithmetic), calls followed into the class's own methods and module-level functions,
  the **class body** and `__new__` and class decorators checked, and the runtime-global
  lookup resolved from the class's own method table. It returns the *reason* instead of a
  bool, so the verdict names the disqualifying call and its line.

**Why** Four gaps, each of which let a class that cannot be constructed in the sandbox
  ship as `callable`:
  - the vocabulary listed readers and network calls but no filesystem *inspection*;
  - only `__init__`'s own body was walked, so I/O one call away was invisible;
  - class-body statements run at **import** of the slice — worse than an unsafe
    `__init__`, because the failure takes down every sibling unit in the module;
  - `by_name["__init__"]` was keyed by **simple name across the whole module**, so with
    more than one class defining `__init__` — 4 of 26 promoted slices — the
    runtime-global check answered about whichever table won the dict.

**Measured**
  - units **229 → 227**, needs_instance **56 → 58**. Exactly the two false-callables,
    no collateral: `TIFDataset` (`os.listdir` in `__init__`, then `raise ValueError` when
    the directory has no `.tif`) and `OptimizedWeatherDownloader` (`os.makedirs`
    directly, *and* via `setup_logging()`).
  - the demotion is verified by construction, not by inspection: the `TIFDataset` shape
    raises `FileNotFoundError: /work/images` on a fresh path and `ValueError` on an empty
    directory. **1/1 of the shipped contract's own call fails.**
  - no false positives: a constructor doing only `os.path.join(os.path.dirname(root))`
    stays `callable`, and 24 of 26 class units are unaffected.

**Surprised by** This milestone *reduces* the headline number, and that is the correct
  direction. The plan's own criterion says so — "a false-callable ships a broken unit; a
  false-not-callable only costs coverage" — but it is worth stating plainly that 229 was
  an overcount I had reported twice.

**Next** Remaining from the audit: name-substring type inference (113/137 params typed
  `number` off the letters 'n'/'k', 6 DataFrames typed `geodataframe`);
  `reject_all_nan` failing correct runs on input shape; contract taken from the LAST
  definition while the shipped slice comes from the FIRST; `_apply_grounding_caveat`
  appending only `summary` so the gate's remedy never reaches the user.

## 2026-08-13 · M6.E · The gate's remedy now reaches the user, and cannot_determine reaches them at all

**Change** `_apply_grounding_caveat` appends the audit's **issues** with their reasons,
  not only the summary; a deterministic gate verdict gets its own headline instead of the
  evidence-support wording; `_audit_flagged` returns true whenever `invariant_gate` is
  set, at any severity.

**Why** Two failures at the very last link, after everything upstream was fixed:

  1. The caveat appended only `summary`, so the user was told a check failed and never
     told *what* failed or what to do. The remedy lives in each issue's `reason` — it is
     the gate's own message that carries "reproject to a local projected CRS (a UTM or
     state-plane zone in metres) before calling". Computing a remedy and discarding it is
     worse than not computing one.
  2. `_audit_flagged` passed only severity `high`. A `cannot_determine` gate verdict is
     recorded as `medium` **on purpose** — an unverifiable number is not a detected
     error, and calling it high would train the reader to ignore the label — so every
     cannot_determine verdict was computed, reconciled into the audit, and then silently
     dropped before reaching the answer. The plan requires the opposite: "an explicit
     cannot-determine outcome that is reported, never swallowed."

  Also: "parts of this answer may not be fully supported by the retrieved evidence" is
  the wrong category for a geographic-CRS buffer. That is a wrong number, not an
  under-cited claim, and describing it as the latter understates it.

**Measured** Caveat text for a failing run **0 → 1 remedy** and now names the call
  (`calculate_buffers(gdf)`). `cannot_determine` caveats reaching the user **0 → 1**. A
  passing run still gets no caveat, and the LLM auditor's soft medium is still suppressed
  — the severity floor was raised for the deterministic gate only. Issue list capped at 4
  with an "…and N more" tail, so a 9-finding run does not flood the answer.
  Tests **1083 → 1089**.

**Surprised by** The severity floor and the gate's severity choice were each individually
  right and jointly wrong. `_audit_flagged`'s docstring correctly explains why it passes
  only `high` (the auditor sets `hallucination_detected` even for soft over-reach), and
  `_reconcile_audit_with_artifacts`'s comment correctly explains why cannot_determine is
  medium. Neither was aware of the other, and the two together deleted the outcome. That
  is the fifth instance this week of a check that stops checking while still reporting
  success — this time assembled out of two correct decisions.

**Next** Name-substring type inference (113/137 params typed `number` off the letters
  'n'/'k', 6 DataFrames typed `geodataframe`); `reject_all_nan` failing correct runs on
  input shape; contract taken from the LAST definition while the shipped slice comes from
  the FIRST.

## 2026-08-13 · M6.F · A name can say "frame"; only an annotation can say "GEOframe"

**Change** `infer_types` splits `_GEO_FRAME_HINTS` (`geodataframe`, `gdf`, `geo_df`,
  `_geo`, `geoframe`) from the general `_FRAME_HINTS`. A name-only match now types
  `dataframe`; an annotation still wins outright.

**Why** Every `_FRAME_HINTS` match claimed `geodataframe`, so `df`, `dataframe`,
  `scaled_df` and `metrics_df` were all typed geospatial. `geodataframe` is the type that
  earns a `projected_crs` invariant, and a plain DataFrame at the call site yields
  `cannot_determine`, which downgrades the whole run's verdict. A false unknown on a
  correct run is how the channel floods and the real signal stops being read.

**Measured**
  - `geodataframe` params **48 → 41**; `dataframe` **8 → 15**. The 7 retyped include
    `plot_histograms(df)`, `create_train_test_sets(scaled_df)`, `save_results(metrics_df)`
    and `extract_24h_before_peak(df)`.
  - invariants: `projected_crs` **21 → 21**, `reject_all_nan` **56 → 56**, total **77 →
    77**.
  - `agdf`/`ogdf` in the e2SFCA family stay geospatial, and they keep their
    `projected_crs` — narrowing the signal cost nothing that motivated it.

**Surprised by** **Zero invariants changed.** None of those 7 units performs a metric
  operation, so none had a `crs_expectation` to lose. So this is a latent-correctness fix
  and an *advertising* fix, not a measured enforcement gain — and the advertising is the
  part that bites today, because `params[].inferred_type` is what `get_method_contract`
  shows the agent when it chooses a method. Telling it `plot_histograms` takes a
  GeoDataFrame is wrong regardless of whether an invariant fires.

  **Two audit findings did not reproduce, and I am recording that rather than quietly
  dropping them:**
  - "113 of 137 params typed `number` on the letters 'n'/'k'" — actual: 195 of 631 params
    are `number`, and exactly **4** come from a name of 1–2 characters, all of them `k`
    in clustering functions (`kmeans_map`, `top_abs_corr_features`, `community_inequality`,
    `intra_inter_idx`) where `k` genuinely is a number. Not a defect at any scale.
  - "the contract comes from the LAST definition while the shipped slice comes from the
    FIRST" — actual: **0 of 227** shipped slices define their symbol more than once, so
    there is no divergence to fix. The concern is real in principle for notebooks that
    redefine a cell; it does not occur in this corpus.

**Next** `reject_all_nan` failing correct runs on input shape — the last of the ranked
  audit findings. Then the deferred plan items: publication paragraph chunking, the Neo4j
  graph emitter, the ablation arm, and the CI artifact re-run job.

## 2026-08-13 · M6.G · reject_all_nan failed correct runs; and the sixth NameError in the gate

**Change** `check_not_all_nan` now discriminates on evidence rather than on shape:
  join-result or every-column-null → `fail`; empty frame → `cannot_determine`; some
  columns null with others populated → `pass` **with the observation recorded**. The
  call-time guard in `_check_one_arg` uses the same rule instead of reusing the
  module-scope check wholesale. `_looks_like_join_result` added to the inlined helpers.

**Why** An all-null column is only provably a defect when the frame is a join result. In
  an input it is ordinary data — a dataset with an empty optional column (`apt_number`,
  `middle_name`) is not broken. Reproduced: a correct run over such a frame was verdicted
  `fail` and blocked, and an empty result from a filter that legitimately matched nothing
  was also `fail`. A gate that fails correct runs gets switched off, which costs the
  unmatched-join detection the check exists for.

**Measured** In a real container:
  - correct run, sparse optional column: **`fail` → `pass`**
  - unmatched `sjoin`: **`fail`, still** — naming `index_right, val` and "nothing matched"
  - `pd.merge` suffixes (`_left`/`_right`) also count as join evidence; keying only on the
    sjoin marker would have missed the commonest unmatched merge
  - call site: a sparse input is silent, an all-null input is `fail`, an empty input is
    `cannot_determine`
  - tests **1093 → 1101**

**Surprised by** Two things.

  First, the recalibration exposed that `_check_one_arg` returned `finding if status ==
  FAIL else None` — so downgrading `reject_all_nan` made the call-time check record
  **nothing at all**. Fixing one silent-noop created another, and the only reason I saw it
  is that I ran the live container.

  Second: `_looks_like_join_result` was not in `_inlined_helpers()`, so the check raised
  `NameError` inside the gate's own `except` and every run came back
  `cannot_determine: check errored`. That is the **sixth** instance — `math as _math`,
  `ModuleType`, `_GEO_MODULES`, `DECLARED_OUTPUTS`, `capture_environment`, and now this.
  Every previous fix was whack-a-mole found by whichever run happened to hit that path;
  the parametrized exit-path test added yesterday exercises only the *coverage* path, so
  it sailed through.

  Replaced with a structural test: parse the generated gate body, collect every `Name`
  load, and assert each is bound within it. That catches all six retroactively and the
  next one prospectively. It is the test I should have written the first time.

  Also worth recording so it is not mistaken for a regression: the suite reported 226s on
  one run and 37s on the next with no change in between — a cold-cache artefact, not the
  new subprocess tests. Slowest single test is 2.41s.

**Next** The ranked audit queue is now empty. Remaining plan items, none of them defect
  work: publication paragraph chunking, the Neo4j graph emitter with the `IMPLEMENTED_BY`
  edge, the ablation arm, and the CI artifact re-run job.

## 2026-08-13 · M6.H · Driving the prototype: the trace said "no results" for ten real hits

**Change** `kb_method_search` emits `count`, placed **before** the `results` array;
  `agent_kb_search` reports the `backend` it used and, when it read the local store while a
  cluster is configured, says so in a note (carried through the success path) plus a
  once-per-process warning; the prototype's row builder and its truncated-payload salvage
  both accept `symbol`; `run_agent_api_dev.sh` resolves `AGENT_KB_BACKEND` and
  `AGENT_METHOD_LIBRARY_DIR` and prints both.

**Why** Every one of these was found by driving the prototype, and none by the suite.

  The trace line read `kb_method_search → no results (log truncated)` for a call that
  returned **ten** methods. Two independent causes. The client builds a row from
  `title|name|doc_id|id|url` and a method unit has **none** of those — only `symbol` — so
  every row was skipped and the count fell through to zero. And the payload is truncated
  for display, with the client salvaging by regex from the fragment, so `count` placed
  after a long array is gone precisely when it is needed.

  `agent_kb_search → no results` was a different problem: the backend defaults to the
  **local** file-backed store, and neither `.env` nor the launcher set
  `AGENT_KB_BACKEND`. So the server read a token-overlap file store while the cluster with
  4,179 indexed docs sat reachable — the entire agent-KB indexing effort inert, with
  "fewer results" as the only symptom.

**Measured**
  - `kb_method_search` in the live trace: **"no results" → "10 results"**, with the method
    symbols now listed as document lines
  - `count` recoverable from an 80/150/400/2000-char truncation: **0/4 → 4/4**
  - launcher output now states `agent KB backend: opensearch` and the library path, where
    before it stated neither
  - the agent then selected a real library method (`calculate_primary_regions`, SPASTC) and
    reasoned about it — the retrieval→contract chain working through the UI
  - tests **1101 → 1115**

**Surprised by** How much of a detour the display bug caused. I read "no results", concluded
  the method library was broken, and went looking for the extraction defect — the library
  was fine the whole time. A trace that misreports is worse than one that says nothing,
  because it sends the reader somewhere specific and wrong. That is the same failure this
  whole milestone series has been about, and it had reached the developer-facing surface.

  A second instance in the same fix: `agent_kb_search`'s success path built a fresh return
  dict, so the note explaining *which store it read* was computed and dropped. The
  diagnostic existed in a variable and nowhere else.

  Also learned: werkzeug logs an SSE `POST` at stream **start**, so "POST … 200" is not a
  completion signal. I used it as one and twice concluded a turn had finished while it was
  still running.

**Next** Nothing defect-shaped is queued. Remaining plan items: publication paragraph
  chunking, the Neo4j graph emitter with `IMPLEMENTED_BY`, the ablation arm, the CI
  artifact re-run job.

## 2026-08-13 · M6.I · A subprocess crash destroyed a turn that had already done the work

**Change** `llm_claude_cli.call` retries a **signal death** only — `returncode < 0`, raised
  as an internal `_SignalDeath` and converted to the `RuntimeError` callers already
  handle. Bounded by `CLAUDE_CLI_SIGNAL_RETRIES` (default 2, clamped 0–5; 0 restores the
  old behaviour). Nothing else is retried. The message now says *"was killed by signal
  11 … with no diagnostic output"* instead of *"exited -11"*.

**Why** Reproduced from a live server log, twice in one turn:

  ```
  RuntimeError: claude CLI exited -11 (model=sonnet):
  During task with name 'model' ... 'search' ... 'orchestrate'
  ```

  `-11` is not an exit status — it is SIGSEGV. The process was killed, which is a crash of
  the tool rather than an answer about the request, and `call()` had no retry of any kind.
  The exception propagated out of the langgraph node, through `graph_runtime`'s
  `raise worker_error[0]`, and out of the SSE stream, **discarding a turn that had already
  completed two search sweeps (20 results), resolved a Chicago bounding box, selected
  `calculate_primary_regions` from the library and written an evidence summary.**

  The asymmetry is the design: retrying a signal death costs one subprocess, while
  retrying an auth failure or a budget refusal spends the user's quota to reach the same
  conclusion slower *and* delays the one message that says what to do. So each mode is
  asserted, not just the happy path.

**Measured** Against scripted subprocess results — 1 SIGSEGV → recovers in 2 calls;
  2 SIGSEGVs → recovers in 3; persistent → gives up after 3 and says so; SIGKILL treated
  the same; and **not retried** (1 call each): clean success, auth failure, structured API
  error, plain nonzero, empty output, timeout. `CLAUDE_CLI_SIGNAL_RETRIES=0` restores the
  previous behaviour exactly. Tests **1115 → 1134**.

**Surprised by** The message was actively misleading. `exited -11` invites the reader to
  look for an exit code of 11, and the returncode's **sign** is the only place the
  "killed by a signal" fact exists — it is gone by the time anyone reads the string. The
  empty stderr is itself informative (an abrupt kill leaves nothing) and read as a
  missing message.

  **Blocked, and reporting rather than working around it:** while reproducing this I found
  the `claude` CLI now returns `401 — OAuth access token has expired. Re-authenticate to
  continue.` So the `claude-cli` provider cannot serve a live turn until the user
  re-authenticates. That is a credential action, so it is theirs to perform. Everything
  in this entry is verified against scripted subprocess results and the recorded log, not
  a live call — stated explicitly because "verified" and "verified live" are not the same
  claim.

**Next** A workflow is auditing what else a single transient LLM failure destroys across
  `graph_runtime`, the supervisor, the SSE stream and the extraction batch path.

## 2026-08-13 · M6.J · The retry I had just added was unreachable, and discarded a real answer

**Change** `llm_claude_cli`: a complete result is returned regardless of exit status; the
  signal check runs in the payload branch too; transient upstream statuses
  (408/409/425/429/500/502/503/504/529) are retried, classified from `api_error_status`.
  `_SignalDeath` and `_TransientApi` share a `_Transient` base so the retry loop has one
  concept of "worth trying again".

**Why** An adversarial audit of **M6.I found two defects in M6.I**. The CLI writes its
  result object when the turn finishes; if it is then killed during teardown, the old
  ordering raised `RuntimeError: claude CLI error (exit=-11): REAL ANSWER` — putting a
  completed, paid-for answer inside the text of the exception complaining about it. And
  the signal check lived only in the no-payload branch, so the retry was unreachable
  whenever the dying process had flushed any JSON, which is the common case.

**Measured** 12 modes: complete-answer-then-SIGSEGV returns the answer in **1** call;
  SIGSEGV with a partial payload, with no output, and SIGKILL recover in **2**;
  408/429/500/502/503/504/529 recover in **2**; 400/403/404/422 and a clean success take
  exactly **1**; a persistent 429 gives up after 3 and names the status.
  Tests **1134 → 1163**.

**Surprised by** M6.I's own tests all passed. They scripted a bare SIGSEGV with empty
  stdout — the one signal case the code already handled — so the suite confirmed the
  claim while the common case was broken and the worst case destroyed an answer. I wrote
  those tests immediately after writing the code, from the same mental model, which is
  exactly when a test is least likely to be independent of it.

  Note on the audit itself: it reported `count: 0, "no findings survived adversarial
  verification"`. That was an **artifact** — 10 of its 12 agents died on a session limit,
  so the refute stage never ran and `confirmed` was empty by construction. Reading only
  the summary would have concluded the code was clean. The findings were in the journal.
  Same failure shape this whole milestone series is about, this time in my own tooling.

## 2026-08-13 · M6.K · Reading a whole paper instead of its first 12,000 characters

**Change** `publication_extractor.paragraph_chunks` + map/reduce over chunks, budgeted by
  `PUB_MAX_CHUNKS` (default 4, clamped 1–20). New `STATUS_PARTIAL`, deliberately **not**
  in `DEGRADED_STATUSES`. Coverage recorded (`chunks_used`/`chunks_total`/`chars_seen`/
  `chars_total`/`chunk_failures`) and **prefixed** onto `contents`. Separately,
  `llm_utils.call_llm` no longer swallows an exception from an injected callable.

**Why** `text[:12000]` cut at a fixed offset, mid-word and mid-sentence, discarding
  everything after it. In a standard paper layout the Methods section comes late, so the
  model answered honestly about the text it was shown and the result was recorded as *the
  method of the paper*. Measured on a 48,661-character document: the old path saw **25%**
  and reported `llm_extracted`.

**Measured** On that document — `PUB_MAX_CHUNKS=2` → 2/5 sections, 45% of characters,
  `llm_partial`, prefixed and warned; `=4` → 4/5, 91%, still partial; `=20` → 5/5, 99%,
  `llm_extracted`, no caveat. Steps keep document order and dedupe across chunks;
  datasets/tools union; one unparseable chunk costs that chunk, not the extraction (4 of 5
  steps survive, and *which* chunk failed is recorded). Tests **1163 → 1173**.

**Surprised by** `call_llm` caught exceptions from an **injected** callable and returned
  "I could not compose an answer due to a generation error." — while a real provider's
  exception propagated. So the test seam did not behave like production in the one respect
  a failure test cares about, and `llm_unavailable` (the call raised) was **unreachable**
  through it: a raising double always arrived as `llm_unparseable`. The plan's stated
  benefit — that the degradation path becomes cheap to exercise offline — was not true.
  Nothing depended on the canned string; removing it left all 1,173 tests green and the
  three statuses now discriminate.

**Next** The audit's remaining findings, none yet independently verified: a peer node
  raising discards the whole turn (no checkpointer, no error edge, no error budget, while
  the same loop tolerates 8 unproductive steps); and a failed turn leaves no record, so
  the user's own question is dropped from history. Both are orchestration-core changes.

## 2026-08-13 · M6.L · The chunking I wrote an hour ago reported full coverage for a half-failed read

**Change** `publication_extractor`: coverage split into `chunks_total` / `chunks_attempted`
  / `chunks_parsed`, with `chars_seen` counting only chunks that produced a spec. The
  prefix and the operator warning name the **actual** constraint — budget-capped and
  crash-lost are different problems with different remedies. `ClaudeCliUnavailable` is now
  batch-fatal: it stops the paper instead of being retried per chunk.

**Why** Two confirmed findings from the resumed audit, both in code committed in M6.K.

  `coverage` was built from `len(chunks)` — the chunks *launched* — before the loop that
  calls the model. Reproduced: 2 of 4 chunks killed by a transient crash rendered as
  **"Extracted from 4 of 4 sections"** with 39,007 of 39,061 characters seen, sitting next
  to a status of `llm_partial` that said the opposite. Every number contradicted the one
  field that was right, and `chunks_parsed` — which told the truth — was computed and then
  dropped before it reached `extracted`.

  Worse, the warning said *"raise PUB_MAX_CHUNKS to cover more"*. The budget was never the
  constraint; that sends an operator to change the one thing that would not have helped.

  Separately: a `ClaudeCliUnavailable` was caught per chunk like any other failure. An
  expired credential does not fix itself, so a batch would call the model once per
  remaining chunk of every remaining paper — turning one expired token into 180 empty
  specs that each read as "this paper describes no method", and burning the batch's whole
  runtime to produce them. Not hypothetical: **the CLI's OAuth token expired during this
  session.**

**Measured** Same 4-chunk paper, 2 chunks crashing: coverage now reads
  `attempted=4 parsed=2`, `chars_seen` **39,007 → 22,751**, prefix **"4 of 4" → "2 of 4"**,
  and the warning changed from *"raise PUB_MAX_CHUNKS"* to *"2 of 4 sections failed to
  extract; a re-run may recover them"*. A genuinely budget-capped read still says
  *"budget capped the read at 2 of 4 sections; raise PUB_MAX_CHUNKS"* and does **not**
  mention failures. An expired credential stops after **1** call, not 4; a transient crash
  still attempts all of them. Tests **1173 → 1177**.

**Surprised by** This is the second consecutive commit where an audit found the defect in
  the commit before it, and both times the shape was identical to what the commit was
  fixing. M6.K's whole point was "a partial read must not be indistinguishable from a
  complete one" — and it shipped a partial read that reported itself as complete.

  My own tests passed. They asserted coverage only on the paths where nothing crashed,
  because I wrote them from the same mental model as the code, minutes later. The audit's
  refuter found it by reading the assignment order — `coverage` built before the loop —
  which is visible in ten seconds to anyone not already convinced the code was right.

**Next** Four confirmed findings remain, all orchestration-core and all flagged for the
  user's decision rather than acted on: a peer node raising discards the turn (reproduced:
  `evidence in state: 10` → `RAISED` → `synthesize ran: False`); the supervisor compiles
  without a checkpointer; a failed turn leaves no record; and the user sees the raw Python
  exception string. Notably the refuter established that `synthesize_node` **already** has
  three graceful degradations — they are simply unreachable, because the exception aborts
  the graph before `synthesize` is routed to.

## 2026-08-13 · M6.M · One failed model call no longer destroys a turn

**Change** Five boundaries, all previously unguarded. `_run_peer` contains `search`,
  `analyze`, `code` **and the decider**; `synthesize_node` wraps `_synthesize_core` with a
  model-free fallback that hands back the retrieved documents; the supervisor gained a
  `peer_failures` list and an error budget (`AGENT_PEER_ERROR_BUDGET`, default 2, plus
  immediate stop on a fatal backend failure); the graph compiles **with** a checkpointer
  under a per-run namespace; `agent_chat_service` records the turn before re-raising; and
  `api/server.py` classifies the exception into an actionable message with a machine-readable
  code, keeping the raw text under `detail`.

**Why** The audit reproduced it: `search call 2 | evidence in state: 10` → `RAISED` →
  `synthesize ran: False`. One peer raising unwound the whole `graph.invoke`, and the user
  got a raw Python exception string.

  The capability to degrade already existed — `synthesize_node` can answer from partial
  evidence, from history, or state an honest insufficiency, and the sibling arm
  (`legacy/graph_nodes.py:265-283`) has caught sub-agent exceptions since it was written.
  Nothing was missing but a `try` at the boundary.

**Measured** Same reproduction, after: the turn produces an answer, the 20 documents
  survive, and the answer carries a partial-answer note. Model entirely unavailable
  (decider + every peer + synthesis all failing) still yields an answer. A broken peer runs
  **≤3** times instead of up to `max_steps`; a fatal backend failure is asked **once**. Two
  turns on one conversation thread get distinct checkpoints and the second does not inherit
  the first's `step`. A failed turn is recorded with its question and reason, and the
  exception still reaches the caller. Tests **1177 → 1207**.

**Surprised by** Containing the peers alone accomplished nothing, and I only found that
  because I re-ran the reproduction instead of trusting the change. The failure being
  survived is *the model being unreachable* — and `_synthesize_core` calls the model, so the
  crash simply moved one node later. The decider is worse: it calls the model too and runs
  **first**, so in a real outage it fails before any peer, which would have made peer
  containment moot in exactly the case it exists for.

  The honesty defect was the sharpest one. With search contained but its failure unlabelled,
  the run reached the no-grounding branch and told the user *"the knowledge base has no
  matching content"* — a factual claim about the corpus, made when nothing had been looked
  up. Evidence-found and evidence-unreachable are different answers, and only one of them is
  the system's to assert.

  A design note worth recording: binding the *conversation* thread to the checkpointer would
  have made a second turn resume the first, inheriting its `step`, `actions` and `evidence`
  so a follow-up would start at step 8 and route straight to `done`. Recoverability and
  cross-turn resumption look like the same feature and are not.

**Next** The plan's remaining M7 items: the Neo4j graph emitter, the ablation arm, and the
  CI artifact re-run job. Live prototype verification remains blocked on the expired CLI
  credential.

## 2026-08-13 · M6.N · The gate stamped ⛔ on a correct answer

**Change** A geographic frame still FAILS on its own, but `run_checks` relaxes it to `pass`
  when the run shows positive evidence of reproject-then-measure: some frame is projected
  **and** carries a computed measurement, and the geographic frame carries none. Areal
  units (`km²`, `m²`, `hectares`, `acres`, and their spellings) added to `_UNIT_ALIASES`,
  and `check_declared_units` now consults the alias table as well as the literal set.

**Why** The first end-to-end prototype run with a working credential produced a
  **completely correct answer** — the agent found `calculate_buffers`, used the
  version-pinned import, reprojected to EPSG:32616 before buffering, and computed
  1,960.34 km² per buffer against an analytic 1,963.50 (0.16%). It even noticed the
  buffers overlap and reported the union separately.

  The answer was stamped: **"⛔ A deterministic invariant check FAILED on this run, so its
  numeric results are not verified."**

  Two causes. The untouched 4326 *input* frame was still bound at module scope — and data
  arrives in 4326 and you reproject it, so an input frame in a geographic CRS is present in
  almost every correct geospatial script. And the declared unit `km²` was in neither the
  known-unit set nor the alias table, so a correctly declared unit came back "unrecognised;
  not checked".

**Measured** The exact live namespace now verdicts **`fail` → `pass`**. Preserved: a bare
  4326 frame **fails**; a 4326 frame buffered in place with no numeric column **fails**; a
  frame holding a measurement computed in degrees **fails even beside a projected frame**.
  9 areal unit spellings recognised, `bananas` still flagged. Tests **1207 → 1220**.

**Surprised by** My first attempt was wrong in a way the tests caught immediately. I keyed
  the verdict on "does this frame hold a measurement column" — which would have **missed the
  motivating case**: `gdf.buffer(25000)` on a 4326 frame produces a wrong *geometry* and no
  numeric column at all. Seven tests failed, and they were right. The correct rule keeps the
  strict per-frame FAIL and relaxes only on positive evidence of the right workflow — and it
  needed **zero** test changes, which is the better signal that it was right.

  Also the **seventh** inline-scope NameError (`_has_metric_column` not in
  `_inlined_helpers`). The structural binding test written in M6.G caught it before it could
  reach a run, which is the first time that class of bug has been caught by a test rather
  than by a live failure.

  Worth stating: a ⛔ on a correct answer is not a cosmetic problem. It is the same
  channel-flooding failure as `reject_all_nan` failing correct runs, and the end state is a
  reader who ignores the marker — at which point the gate is worse than absent, because it
  cost real work and buys nothing.

## 2026-08-13 · M7.1 · Sandbox image, deployment runbook, CI

**Change** `sandbox/Dockerfile` gains the geospatial stack (from wheels) plus `libexpat1` and a
  warmed pyproj CRS database. New `docs/DEPLOYMENT.md` and `.github/workflows/verify.yml`
  (three jobs: pure tests, deployment contract, artifact replay). 8 new deployment-contract
  assertions. `.corpus_cache` stripped from git history.

**Why** The image's geospatial block was commented out, so every session touching a
  GeoDataFrame paid a `pip install geopandas` first — and the library's strongest clusters
  are all geospatial.

**Measured**
  - image **1.25 GB**, builds in ~7s warm; the full stack imports with `--network none`
  - a 25 km buffer in EPSG:32616 inside the image gives **1960.34 km²** (analytic 1963.50)
  - end-to-end in the **digest-pinned** image: `calculate_buffers` imported via the registry's
    own `import_line`, contract resolved, **exit 0, verdict `pass`, total 5881.03 km²** —
    the same number the live prototype run produced independently
  - runtime `pip install`s for that run: **0**
  - history rewrite: commits **257 → 257**, HEAD tree **byte-identical**, commits containing
    `.corpus_cache` **2 → 0**, backup ref at `refs/backup/pre-corpus-cache-rewrite`
  - tests **1220 → 1228**

**Surprised by** The first build succeeded and the image was broken. `import rasterio` died
  with `libexpat.so.1: cannot open shared object file` — the wheel installs cleanly against
  `python:3.11-slim` and fails at import, so **a green build proved nothing**. Only running the
  import caught it. That is now a CI step and a contract test, because it is the exact shape
  of failure this project keeps producing: a step that reports success while the thing it was
  meant to establish is false.

  Also: I appended a second `_dockerfile()` helper to the contract tests, which **shadowed** the
  existing module-level one and silently redirected two passing tests at the wrong file. They
  still ran and still asserted — against something else. Caught only because the count went
  1228 → 1226 in the same run I was watching for a different reason.

  Retraction: the plan called for `AGENT_CODE_EXEC_IMAGE` pinned by digest, and I had described
  digest pinning as already done. `resolve_image_digest` was implemented and correct, but no
  pinned image existed to point it at — the tag `python:3.11-slim` was still the deployed value.

**Next** Neo4j write-back **dropped**, not deferred: the `IMPLEMENTED_BY` edges already reach
  OpenSearch, `kb_method_search` reads the registry rather than the graph, and no consumer
  exists for multi-hop traversal. It would have been the only write to production data in the
  plan, for no present benefit. Neo4j stays read-only; the read path was fixed earlier today.

## 2026-08-14 · M7.2 · One command that walks every seam

**Change** `scripts/smoke_end_to_end.py` — six stages (config, library, retrieval,
  sandbox+gate, live API, replay), each asserting an outcome and printing the number it
  measured. A check that cannot run reports **SKIP with its reason, listed separately at
  the end**; exit code is the failure count.

**Why** A checklist in a document inherits the exact failure this project keeps producing:
  you tick "library builds" without noticing the units it built cannot be imported. Each
  stage here asserts the outcome instead, and a skip is never a pass.

**Measured** Against the live system: 16 passed, 1 failed, 2 skipped. The pass rows carry
  their evidence — `227 units / 227 modules on disk`, `calculate_buffers at rank 1`,
  `correct run verdicts pass in 6s`, `degrees buffer verdicts fail`,
  `rejects an unauthenticated call: 403`, `one real turn answered in 8s`.

**Surprised by** Twice, by my own check.

  It reported 6 modules across ~30 units as undeclared dependencies. Every one was declared
  correctly — I was comparing the **import** name against the **distribution** name.
  `import ee` is satisfied by `earthengine-api`, `sklearn` by `scikit-learn`, `PIL` by
  `pillow`; translating between those namespaces is precisely what `pkgmap` exists for. So
  the check was wrong and the extraction was right, and I nearly filed it as a corpus defect
  because an independent audit had reported something adjacent. It now resolves through
  `pkgmap`, and a declared-but-not-installed dependency is reported as *expected* rather
  than as a failure.

  And the deliberately-wrong run printed **the same number as the correct one**: 5881.03.
  Buffering by 25,000 *degrees* gives π×25000²/1e6 = 1963.5 per point — the arithmetic is
  identical and only the units differ, so three points sum to ~5881 either way. I flagged it
  as a suspicious duplicate before working out that it is the single best illustration of why
  this gate exists. Magnitude gives a reader no signal at all; only the CRS does. The gate
  said `fail`.

**Next** The gap analysis's ranked list, unstarted: gate coverage (10 of 12 realistic
  wrong-number scenarios pass), data access for the code peer, dataset source validation
  (66 of 130 curated datasets return an HTML landing page written to disk as data), and the
  extraction driver's missing per-type dispatch.

## 2026-08-16 · M8.1 · The only edge extraction emits that a graph would want

**Change** `extractors/analysis/citations.py` (new, pure) parses platform element references
  out of notebook text, and `notebook_extractor` emits them as `CITES` / `USES` edges. The scan
  sits **above** the cell loop's two `continue`s so markdown cells are included — a citation
  usually lives in prose, and a markdown cell with no code cell after it never becomes anyone's
  `md_context`. Self-citations are dropped. `base.py`'s relation vocabulary gains `CITES` and a
  note recording what the other six actually are.

**Why** I have been designing a store for edges that did not need one. Measured over the 174
  cached corpus notebooks: 4,212 edges, of which **4,071 (96.7%) satisfy
  `dst.startswith(src + "::")`** — a pure function of the id, already recoverable by the shipped
  `doc_ids.parent_doc_id:73`, and already queryable via the prefix/parent term at
  `agent_kb.py:318-321`. `cross_element` was **0**. The extraction graph was a forest of stars,
  and no partition or traversal over it could say anything. Meanwhile authors *do* link elements
  to each other — in prose and in `wget` lines — and `_FILE_TOKEN_RE:53` only ever matched data
  file extensions, so nothing looked.

**Measured** 179 i-guide.io URLs in the corpus, 100 carrying an element UUID. Through the real
  extractor over all 174 notebooks, 0 failures: **99 cross-element edges (CITES 92, USES 7)**
  from 29 elements to 40 targets — 21 notebooks and **19 outside the notebook set**. Graph: 56
  nodes, 12 components, largest **28**. Cross-element edges 0 → 99. All 41 distinct targets
  resolve to a live platform element (41/41). Self-loops emitted: 0. Tests 1302 → 1330.

**Surprised by** Three things, in ascending order of how wrong I was.

  The two hosts do not mean the same thing. `platform.i-guide.io/<type>/<uuid>` is a link to a
  page; `storage.i-guide.io/<type>/<uuid>/<file>` is a **download**. The second is a data
  dependency, so it emits `USES` — and it is the first `USES` edge in this repo whose `dst` is a
  real node. The existing one at `publication_extractor.py:401` sets `dst=str(ds)` from an
  LLM-named dataset string, and there is no code anywhere that resolves such a name to an id.

  My own characterising test was wrong, in the project's signature way. I asserted that citations
  are "the only edges that leave the element", classifying by `not dst.startswith(element_id)`.
  `HAS_WORKFLOW` fails that too — but its dst is a sha1 handle that is the doc_id of *nothing*
  (141 such edges corpus-wide, 141 distinct dangling dsts). Two buckets hid a third, and the
  test would have let a dangling edge count as cross-element structure. Now three buckets.

  And a retraction. `docs/DEVLOG.md:2606-2609` says the Neo4j write-back was dropped partly
  because "no consumer exists for multi-hop traversal". **That is false.**
  `neo4j_explore_related_nodes_tool(element_id, depth=2, limit=50)` is registered at
  `langchain_granular_tools.py:143-149,522-526`, named in `graph_state.py:50` and
  `search_methods.py:31`, instructed in the persona at `prompts.py:50`, and backed by
  `[:RELATED*1..{depth}]` at `neo4j_graph_tools.py:490-506`. The defensible claim is narrower:
  *no consumer traverses the extraction edge types*. Dropping the write-back was still right —
  Neo4j is community edition (`CREATE DATABASE` → `UnsupportedAdministrationCommand`, so there
  is no spare database), and `USE_TEXT2CYPHER` defaults to `"true"` with `_sanitize_cypher`
  (`agents.py:621-630`) carrying no label allowlist, so anything written to prod is immediately
  reachable by LLM-authored Cypher against production. The conclusion held; one of its three
  reasons did not.

**Next** The citation layer is one signal. The graph the platform *already* curates is larger and
  was never read: `GET /api/elements/{id}` returns `related-elements` (the listing endpoint hides
  it) — 329 of 473 sampled elements carry one, giving 576 distinct undirected pairs, 774
  reciprocal, hop-2 productive for 296 of 391 connected nodes, and already covering
  dataset–notebook (142) and dataset–publication (100). Next is fusing that with entity and
  relationship extraction over element content, so elements connect *through* shared entities
  rather than only directly.

## 2026-08-16 · M8.2 · A dependency that arrives by accident

**Change** `networkx` declared in `requirements.txt` and pinned `==3.4.2` in `constraints.txt`,
  plus four contract tests: that it is declared, that it is pinned, that the *installed* version
  matches the pin, and that `louvain_communities` exists at that version with `seed`, `weight`
  and `resolution` parameters.

**Why** It is imported directly by the community-detection work now starting, but it was
  reachable here only as a transitive dependency of `torch` (`pip show networkx` → Required-by:
  intake, mapclassify, osmnx, scikit-image, torch). That is the `pyarrow` failure from M0.5
  exactly: present in dev via anaconda, absent from a clean build the moment the package that
  dragged it in changes. A CPU-only torch variant would have removed it silently.

**Measured** dev resolves **3.4.2**; a clean `pip install -r requirements.txt -c constraints.txt`
  resolves **3.6.1**. So every partition measured on this host would have run on a different
  implementation in CI. `pip install --dry-run --python-version 3.11 --only-binary=:all:
  networkx==3.4.2` → exit 0, one `py3-none-any` wheel, zero runtime deps. Deployment-contract
  tests 14 → 18.

**Surprised by** The seed assertion earning its place immediately. Louvain without `seed=` is
  nondeterministic, so community *ids* would change between two runs over an identical graph —
  which silently invalidates any cached community summary and any id written back onto an
  element document. The algorithm being present is not the property that matters; being
  *seedable* is, and only one of those two is obvious from a version number.

**Next** The two measurement workflows (metadata-only baseline, extraction-native treatment) are
  mid-flight. The number they exist to produce is the marginal contribution of extraction over
  what the public platform API already gives anyone.

## 2026-08-16 · M8.3 · The seed was not the thing that made it reproducible

**Change** Corrects M8.2's own contract test. `test_the_partition_algorithm_...is_seedable` is
  kept, but it no longer stands alone: a second test pins the property we actually depend on —
  that canonicalising (sorting) node and edge insertion order makes the partition independent of
  the order a caller supplies, and that insertion order genuinely perturbs the result at this
  networkx version.

**Why** M8.2 asserted `louvain_communities` accepts `seed=` and treated that as reproducibility.
  It is not. `seed=` pins Louvain's own randomness but not the order it visits nodes, which
  follows graph insertion order. Found by the fusion measurement, whose first two runs over one
  identical graph disagreed at Q=0.6463 vs 0.6504 because `PYTHONHASHSEED` reordered a `set`.
  A test that certifies a property the system does not have is worse than no test.

**Measured** Independently reproduced across three `PYTHONHASHSEED` values on one fixed graph:
  set-derived insertion gave Q **0.235526 / 0.244236 / 0.234232** and **7 / 7 / 8** communities;
  sorted insertion gave **0.243509** and a byte-identical partition every time. Order-sensitivity
  then measured across scales — 9 orderings produce **9 distinct partitions at every size
  tested**: 60/200, 120/700, 200/1500, 400/4000, and **750 nodes / 9,112 edges**, the last being
  the fused graph's real dimensions. Deployment-contract tests 18 → 19.

**Surprised by** Writing the same defect twice inside one hour, in two different disguises.

  First: `assert partition(forward) != partition(reverse) or True`. The `or True` makes it
  unconditionally pass. That is the exact shape — a check that stops checking while reporting
  success — that this log has recorded seven times.

  Second, subtler and the one worth keeping: after removing the `or True`, the assertion FAILED,
  and the reason was that I had built the fixture with modular arithmetic. That graph is regular
  enough that Louvain resolves it identically from any insertion order. So the honest reading of
  the failure was not "the hazard is not real" but "my fixture cannot see it" — a random graph
  with the same node count is order-sensitive 9 times out of 9. A structured fixture would have
  passed as soon as I weakened the assertion, and would have certified reproducibility on the
  strength of a graph nobody has.

**Next** Report the two measurement workflows. Headline, stated here because it is a result about
  this project's central claim: against the strongest honest public-metadata baseline, all
  extraction-native edge layers add **141 new element pairs on a 15,711-pair baseline (0.90%)**,
  12 of 15 sampled read as spurious, and the 99 CITES/USES edges from M8.1 add **exactly zero**.

## 2026-08-17 · M8.4 · The graph, built on the layers that were measured to carry relatedness

**Change** `agent_runtime/corpus_graph.py` + `scripts/build_corpus_graph.py` build the corpus
  knowledge graph — 750 element nodes, four edge layers, Louvain communities, per-community
  profiles — and emit a self-contained interactive visualisation of the whole graph.

**Why these layers and not extraction's.** Extraction's own edges are a forest: 4,071 of 4,212
  satisfy `dst.startswith(src + "::")`, recoverable by the shipped `doc_ids.parent_doc_id`,
  cross_element 0. Adding every extraction-native layer to the layers below moved the graph by
  141 pairs on a 15,711-pair baseline (0.90%), rescued 0 isolated elements, changed modularity by
  ≤0.0025, and 12 of 15 sampled new pairs read as spurious. The M8.1 citation edges are kept
  because they are exact and human-authored — not because they add reach; they add 0 pairs.

**Measured** 750 elements, **6,121 edges**, 1 component, **15 communities, modularity 0.7806**.
  Layer pairs: SIMILAR_TO 5,347 · IN_COLLECTION 667 · RELATED_TO 592 · CITES/USES 93.
  `--assert-stable` re-partitions and compares: **replays identically**. Held-out curated pairs
  land in the same community 82.9% vs 11.0% random (7.6x) and kNN alone reaches 74.2% without
  ever seeing curated data.

**Surprised by** Two things the build made visible that a number alone had not.

  **The corpus's largest bloc fragments rather than clusters.** Communities 3, 9, 12, 13 and 14
  are all `flood map / flood risk`, type purity 1.00, and **contributor concentration ≥0.97 on
  the same uploader** — 160 map elements, one batch, split five ways. Louvain slicing a dense
  near-identical blob arbitrarily reads as five research themes. So community colour was made to
  do structural work instead of identity work: communities sharing a dominant contributor share
  one hue family and are bracketed together, which makes one upload look like one thing.

  **91.4% of edges rest on a single layer.** The fused graph is a union of evidence, not a
  consensus — and "0 isolated of 750" is a property of the kNN layer, not of the corpus. Strip
  kNN and 312 of 750 have no edge at all. The visualisation therefore ships the layer toggles as
  a first-class control rather than a footnote: untick embedding similarity and 312 nodes go red.
  A caveat you can operate is harder to skip than one you can read past.

**Also measured, and it changes the coverage ceiling.** A concurrent session's `86a7b18` made the
  extraction driver type-aware. The library on disk is now **551 units over 95 elements** (was
  227/65): notebook 227 units/65 elements/39 with an invariant · **dataset 22/22/19** ·
  code 302/**8**/16. Datasets are the first non-notebook units with contracts and the highest
  invariant density in the library (86% vs notebook 17%, code 5%) — but all 19 carry the same
  `crs_equals EPSG:4326`, so as a graph layer it is one value over 19 elements, i.e. the same
  near-clique shape that made the invariant layer degenerate before. Code is 302 units from 8
  elements at 5% invariant density, which corroborates that session's own net-negative call.

**Next** Community reports (the GraphRAG move — an LLM summary per community, which is what
  retrieval actually reads) over the 15 communities. And the extracted dataset `spatial` payloads
  went to the cluster under another session's env; `OPENSEARCH_NODE` is unset in this worktree's
  `.env`, so computed CRS/bounds/schema — the one dataset signal that is not the degenerate
  invariant — remains unmeasured from here.

## 2026-08-17 · M8.5 · Community reports, and a grounding check that rejected 93% of good output

**Change** `scripts/summarize_communities.py` writes one grounded report per community (title,
  summary, themes, caveat) via `call_llm` — 15 calls for this corpus. The visualisation now shows
  them: clicking a community in the rail opens its report, and rail rows are labelled by report
  title instead of by their top three tags.

**Why** The partition was unreadable. Clicking a community *filtered the canvas and nothing else*,
  so the only way to reach a community profile was to click one of its nodes and read the block at
  the bottom of the element inspector. The summaries were the thing a person actually wants and
  they were unreachable. This is also the half of the GraphRAG pattern that matters: global search
  reads community reports, not the clustering.

**Measured** 15/15 communities, 15 LLM calls (`claude-cli`/`sonnet`), 199s. Themes kept 59,
  dropped as ungrounded 16 (21%). Reports are legible and specific — *"SIMPLE-G Model:
  Agriculture, Water, and Trade"*, *"Telecoupling and Metacoupling for Global Sustainability"*,
  *"R Spatial Analysis with IPUMS Vector Data"* — and 6 of 15 carry a required single-uploader
  caveat.

**Surprised by** My grounding check rejecting **93%** of the model's themes, and being wrong.

  The first version required a theme to appear as a verbatim substring of the prompt. It threw out
  `"flood risk mapping"` from a community whose tags are `flood risk` and `flood map`, and
  `"county-level flood hazard"` from members titled *County-Level FEMA Flood Risk Maps*. Those are
  paraphrases composed of grounded terms — which is what a summary phrase *is*. Same error as
  comparing an import name to a distribution name in M7.2: the check was wrong and the model was
  right. Token-level matching took it 93% → 33%; a three-line stemmer (so `mapping` matches `map`)
  took it 33% → 21%.

  And the 21% that remain are the real thing. Every flood community had the model adding FEMA-shaped
  domain detail it was never shown — `100-year and 500-year flood scenarios`, `floodplain
  assessment`, `watershed-level flood hazard`, `FEMA-style flood risk delineation`. For a corpus
  where those maps may not be FEMA products at all, that is exactly the invention worth catching.
  Rejected themes are rendered struck-through in the report rather than dropped silently, because
  what a summariser wanted to say and could not support is information about the summariser.

  Two process notes. Fixing the check without re-running the model needed a `--recheck` mode that
  re-scores saved output at 0 LLM calls — otherwise "the fix improved the rate" would have been
  confounded with new sampling. And when I added it, the summary line still printed
  `15 LLM calls` on a run that made none; a reporting bug in the same commit as a fix to a
  measurement bug.

**Next** The extracted dataset `spatial` payloads (computed CRS/bounds/schema for 22 dataset
  elements) remain unmeasured from this worktree — `OPENSEARCH_NODE` is unset in its `.env`. That
  is the one candidate layer left with a plausible shot at non-degenerate cross-element edges.

## 2026-08-17 · M8.6 · A daily rebuild, because the partition costs 41 ms

**Change** `corpus_graph.corpus_fingerprint` (content hash of the corpus) and
  `corpus_graph.reconcile_communities` (Jaccard-matched community ids across runs), plus
  `scripts/graph_daily.py` — an idempotent scheduled refresh that no-ops on an unchanged corpus,
  carries community ids, re-summarises only communities that moved, and fails on shrinkage.

**Why a daily rebuild and NOT online incremental.** Profiled the whole pipeline:

  | step | wall |
  |---|---|
  | louvain partition | **41 ms** |
  | whole rebuild compute | 2,628 ms |
  | embed 750 element texts | 2,003 ms (76% of compute) |
  | fetch 750 records over REST | **1,234,000 ms** |
  | 15 community reports | 199,000 ms |

  Incremental community detection would optimise **41 ms of a 2.6 second job**. Fetching is 470x
  the entire compute and the LLM reports are 76x it, so the things worth making incremental are
  the fetch and the summaries — never the clustering. Rebuilding is also strictly more correct: an
  incrementally-maintained partition drifts from the one a rebuild would produce and nothing tells
  you.

**Measured, over four runs.** Cold: 750 elements, 6,121 edges, 15 communities, 9.0s. Unchanged
  corpus: fingerprint match, **nothing rebuilt, 0 LLM calls**. Truncated corpus (200 of 750):
  **exit 1**, refused. Day-2 simulation (6 elements retagged, 1 added): fingerprint moved,
  751 elements / 6,126 edges / 16 communities, **15 of 15 ids carried**, 1 new, and only **6 of 16
  communities changed membership** — so 6 summaries to regenerate and 10 carried forward.

**Surprised by** How little it takes to move the partition, which is the whole argument for
  reconciliation rather than re-enumeration.

  The 29 citation-only pairs — **0.5% of the edges** — change this corpus from 15 communities to
  16. And in the day-2 run, retagging 6 elements out of 750 and adding one produced a **28-element
  new community** and shifted C1 by +9/−32 members (jaccard 0.696). A nightly job that renumbered
  from Louvain's output would rename most communities most nights, invalidating every cached
  report and churning every `community_id` on an element document, while the corpus had barely
  changed.

  Also caught myself reading a guard's exit code off `tail`: `python … | tail -4; echo $?`
  printed `exit=0` for a run that correctly returned 1. The guard worked; my verification of it
  did not. Re-checked without the pipe.

**Next** Wire the fetch to `extractors/platform_graph.py` (read-only Neo4j, one query per label)
  instead of 750 REST calls — that removes the 1,234s term and the rate-limit failure mode
  entirely, and is the only part of this job that is still slow.

## 2026-08-17 · M8.7 · Three node counts for one graph, none of which reconcile

**Change** `scripts/graph_census.py` — one read-only pass that reports total nodes, total
  relationships, nodes per label, relationships per type, and element counts public-vs-all, and
  reconciles them against the public REST API. Exits 2 with the exact `.env` keys to set when
  Neo4j is not configured, rather than printing something.

**Why** Asked for the graph's node and relationship totals and found I could not answer it, while
  three different figures were in circulation and no two agree:

  - `docs/DEVLOG.md:957-959` — 799 elements / 819 graph nodes. A later measurement pass reported
    this does not reproduce.
  - "3,205 nodes" — repeated in this session's notes, and I repeated it again in conversation. No
    query is recorded anywhere next to it. **Retracted; it is unverified.**
  - `extractors/platform_graph.py`'s docstring denominators (notebook 200, publication 200,
    dataset 141, map 161, oer 44, code 38 = 784). These cannot all be label totals: publication
    200 is *below* the 203 the public API returns, and the graph is the store the API is a view
    of, so it cannot hold fewer.

**Measured** Element counts, re-verified from the API just now and agreeing with its own
  `total-count` field: **750 total, all `visibility: public`** — publication 203, notebook 180,
  map 160, dataset 130, oer 41, code 36. 750 enumerated distinct ids, `total-count: 750`.

  Neo4j nodes and relationships: **NOT ESTABLISHED.** This worktree's `.env` contains **zero**
  `NEO4J_*` keys (checked by name, 0 of 6 present), so the census cannot run here. The 43 private
  elements referenced in `86a7b18` are that session's measurement, not mine.

**Surprised by** Having cited "3,205 nodes" in conversation earlier today as though it were
  measured. It came from a prior session's summary, survived into this one as background, and I
  repeated it without a query behind it — which is precisely the failure this log exists to
  catch, committed by me in the same session in which I criticised a stale figure in someone
  else's docstring. The census script is the fix: the number now has exactly one source.

  Also worth recording: `platform_graph.py` already had the right tool at `:133` — a `counts()`
  doing `MATCH (n:{label}) RETURN count(n)` rather than counting a sampled page. The gap was never
  capability, only credentials.

**Next** Set the six `NEO4J_*` keys in this worktree's `.env` and run
  `python scripts/graph_census.py --compare-api`. The same credentials unblock the two other open
  measurements: extracted dataset `spatial` payloads, and pointing `graph_daily.py`'s fetch at the
  graph to remove its 1,234s REST term.

## 2026-08-16 · M8.6 · Sixty-four datasets that were one number and are actually two problems

**Change** `extractors/dataset_repositories.py` (new) + a `landing_page` branch in
  `scripts/build_method_library.py:361`. `classify_link` sorts a URL into `deposit` / `portal` /
  `opaque` / `direct`; six resolvers (figshare, Zenodo, HydroShare, Dataverse + the Illinois Data
  Bank, Hugging Face, a GitHub tree) turn a deposit's landing page into `{name, url, bytes}` per
  file by reading a documented listing endpoint. A `doi.org` link is followed exactly once first.

**Why** 64 of the corpus's 130 datasets fetched HTML instead of data, and the driver reported all
  64 identically as unfetchable. They are not one problem:

  - **~36 are a deposit behind a human page.** The file is there and the host publishes an API for
    it. Nothing was broken; nothing was asked.
  - **20 are a data portal's front door** — `fao.org/faostat/en/#data`, `who.int/data/#data`,
    `data.worldbank.org/indicator`. There is no file and there never was one. "Download failed"
    sends the next person to hunt a bug that does not exist, *and* discards the true fact that
    the element is a usable pointer to a data source.
  - **8 are opaque** (Drive, Dropbox, SEDAC-behind-login): a file exists but cannot be enumerated
    without a browser session. A third outcome again, and a third fix.

  No resolver constructs a download URL from a pattern. A guessed link that 404s is
  indistinguishable from a dataset that was withdrawn, and the corpus contains both.

**Measured** Datasets described in some form: **41 → 65 of 130.** Of the 64 landing pages,
  **24 resolved to file listings** and 20 were correctly named as portals rather than failures.
  Content recovered that no previous run could see: 341 heat-stress rasters (20.3 GB), LocBench
  (15.7 GB), U-Net training data (4.9 GB), and a US-boundary set of 23 zipped shapefiles
  (468 MB). Tests 1302 → 1324.

**Surprised by** A standalone probe over the same 64 reported 25 resolved where the driver
  reported 24. I have not reproduced the difference and will not claim which is right — the
  likeliest cause is a transient API response on a single element, but "likeliest" is not
  measured, so the number above is the driver's, and the discrepancy stays recorded.

  The more useful surprise: the fix for the largest group was not a fix at all. Twenty of these
  elements were already correct, and the only defect was the extractor's *description* of them.
  Two-thirds of the remaining recovery came from reading a documented API — no new capability,
  just asking the right endpoint.

**Next** 18 landing pages remain on hosts with no resolver, and 32 publications are open-access
  but HTML-only. Per the standing instruction the schema revision comes before indexing, so
  neither of these blocks the next step.

## 2026-08-17 · M8.7 · The index document knew a function existed and not how to call it

**Change** `extractors/emitters/opensearch_emitter.py`: `_unit_payload` mirrors the contract into
  `extracted.unit` and computes `import_line` from `library_module` + `library_symbol`; a `unit`
  branch in `_embed_text`; platform form fields applied BEFORE the canonical keys with
  `RESERVED_KEYS` refused and recorded; `extracted.parent_doc_id` defaulted from the doc_id rule;
  `block`/`runnable`/`spatial` omitted when absent; the envelope written once. New
  `index_mapping()` declares every term-queried path as `keyword`, and `mapping_drift()` +
  `ensure_index` report a live index whose mapping contradicts it.

**Why** Measured on the live cluster: `iguide_agent_method_units` holds **349 documents and not
  one contract**. `extracted` carried `unit_name`, `callable`, three explicit nulls and a
  duplicate of the title. Signature, parameter types, declared units, CRS expectations,
  invariants, pip requirements, module and slice_sha — the entire reason a MethodUnit document
  exists — were dropped by `_build_source`, which mirrors `block`/`runnable`/`spatial` and was
  never taught about `unit`.

  Nothing crashed, because there are **two stores and only one of them is lossy**.
  `kb_method_search` reads the on-disk `_registry.json`, so it kept returning full contracts; the
  loss only bites the *other* path, where a unit surfaced by `agent_kb_search` tells the agent a
  function exists while withholding every fact needed to call it. The two stores also disagree on
  size: registry **551 units / 95 elements**, index **349 docs / 73 parents**.

  Verified against HEAD's emitter before changing it, rather than asserted: contract in the
  document `False`; `embed_text` `'calculate_buffers def calculate_buffers(gdf, b)'` with no
  units and no dependencies; a form field named `doc_id` yielding `doc_id: 'HIJACKED'`; a dataset
  document's `parent_doc_id` `None`; `['block','runnable','spatial']` all present as nulls.

**Measured** Fields an index hit gives the agent: **11 → 17**, of which the contract's 13 go from
  0 to 13. `embed_text` for a unit now contains its inferred types, declared units, CRS
  expectation and pip deps (0 → 4 kinds of term). Reserved-key clobbers possible: **6 → 0**.
  Documents carrying `parent_doc_id`: 2 of 4 types → **4 of 4**. Explicit nulls per unit doc
  3 → 0. Tests **1324 → 1359**; emitter tests specifically **0 → 35**.

**Surprised by** Two things, both about my own method.

  I predicted the reconciler was already broken — `existing_doc_ids` term-queries
  `extracted.parent_doc_id`, which the live mapping makes `text`, and a term query against
  analyzed text usually matches nothing. **Wrong: 73 of 73 parents match.** This corpus's element
  ids are 8-character lowercase hex, which the standard analyzer emits as a single token. The bug
  is real but *latent* — a full platform UUID would be split on its hyphens and silently match
  nothing — and the difference between latent and active is exactly what I would have
  misreported. Hence `index_mapping()` fixing it and this note refusing to claim a save.

  Second: my first pass at "how much of this library is plumbing" used a keyword list, matched
  `lat` inside `translate`, and classified a transformer decoder layer as a geospatial method
  with 948 of 1062 units "domain-relevant". I deleted the classifier rather than tuning it —
  usefulness is a retrieval question and retrieval measures it. A second probe then reported a
  2× registry/summary disagreement that turned out to be my own wrong key (`alias_of` for
  `alias_for`). Three probe bugs in one sitting, all of the same shape: a check that returns a
  confident number while measuring something else.

**Measured, not fixed** 38% of the 551 units are NAKED — no docstring, no annotation, no return
  type — so their contract adds nothing beyond the symbol name; 14% take no parameters at all and
  can only be called for a side effect; 13% carry an invariant, from only 3 distinct checks
  (reject_all_nan 72, projected_crs 22, crs_equals 19). Concentration is severe: the top 5 of 95
  elements contribute **296 of 551 units**, one element alone 92. The README gate is
  element-level, so an on-domain repository still floods the library with its internals.

**Next** Import names are injected into the platform `tags` field —
  `notebook_extractor.py:209`, `code_extractor.py:207`. 338 of 349 indexed unit documents (97%)
  carry at least one stdlib module name as a "tag"; 43 of 432 distinct tags are stdlib names; the
  platform's own tags contain **zero** such names across all 180 notebook elements, so extraction
  invents them. `agent_kb.py:33` queries `title`/`contents`/`extracted.embed_text` and NOT `tags`,
  so this is a data-honesty defect and not the retrieval defect it looks like — worth stating
  before someone measures a retrieval win that was never available.

## 2026-08-17 · M8.8 · The index advertised an import line the library had not assigned yet

**Change** `extractors/ingest.py` `_fan_out` now runs **library → mcp → opensearch**;
  `notebook_extractor` stops merging import names into the platform `tags` field (3 asset sites
  plus the SkillSpec front matter) and the now-dead `tags` accumulator is removed;
  `kb_store.store_dir` honours `AGENT_KB_STORE_DIR`; `conftest` isolates the local KB store as
  well as the method library.

**Why** Three findings, each uncovered by the previous one.

  1. **The fan-out order was backwards.** `library_emitter` is what assigns a unit its
     `library_module` (`library_emitter.py:304`, mutating the live unit dict via `__dict__`), and
     the index document advertises that module as the import line. OpenSearch emitted *first*, so
     every unit document was built before its module path existed. This was unfindable until
     M8.7 gave the document a field to hold the module: with no field, there was nothing to
     notice was empty. A defect and its detector in consecutive commits.

  2. **`tags` was not tags.** 338 of 349 indexed unit documents carried at least one stdlib module
     name as a topical tag; 43 of 432 distinct tags were stdlib names (`os`, `time`, `copy`,
     `html`, `gc`, `getpass`, `__future__`). The platform's own tags contain **zero** across all
     180 notebook elements, so extraction invented every one. The imports were already in
     `block.imports` and the unit's `requirements`, so the merge added nothing reachable — I
     dropped my first attempt at a replacement `extracted.imports` field for the same reason.
     Note what this is NOT: `agent_kb.py:33` queries `title`/`contents`/`extracted.embed_text` and
     not `tags`, so no retrieval win was available here. A data-honesty fix, stated as one.

  3. **The suite was scheduled to break on success.** Two probes of mine wrote into the real
     method library and the real local KB store, and three tests that fake every other retrieval
     arm failed: `test_sweep_adds_implied_methods`, `test_direct_search_sweep_drops_unlisted`,
     `test_the_every_turn_sweep_still_never_touches_the_web`. They were asserting "and nothing has
     been ingested on this machine". `conftest` already isolated the *library* half after this
     exact failure hit two tests once before — the KB half was left uncovered, and indexing the
     corpus is precisely what the extraction work is for.

**Measured** Unit documents carrying an importable, version-pinned line: **0 of 130 → 12 of 12**
  on the element re-run end to end, and the advertised dotted path resolves to a file on disk (a
  test asserts that, rather than trusting the string). Documents with a stdlib-name tag over 60
  corpus notebooks / 1,533 documents: **338 of 349 → 0**; distinct stdlib-name tags **43 → 0**;
  the 181 surviving tags are the submitters' (`A2SFCA`, `AmeriFlux`, `Census Shapefile`, `CyberGIS`).
  Suite with 9 documents in the developer's real KB store: **3 failed → 0 failed**. Tests
  **1359 → 1370**.

**Surprised by** Both of the day's remaining bugs were found by my own carelessness rather than by
  design. `order_effect.py` set `AGENT_STORAGE_ROOT`, which is not a variable this codebase reads —
  the real one is `AGENT_FILE_STORAGE_ROOT` — so it wrote 12 duplicate units into the real library
  under a second element package (`ke_b1fa548b_spastc` beside
  `ke_b1fa548b_spastc_a_spatial_partitioning_algorithm_for_scal`), taking the registry 1079 → 1091
  rows and 17 → 29 ambiguous names, and 45 documents into the real KB store. Both are restored and
  verified back to byte-equivalent counts (551 units / 95 elements / class 49 / function 502;
  store empty). I briefly wrote this up as a hazard in the library emitter before checking the
  variable name — it was my typo, and the override works. The test-isolation gap it exposed is
  real, and is the part worth keeping.

**Next** The remaining content questions are measured and unfixed: 38% of units are naked (no
  docstring, no annotation, no return type), 14% take no parameters, and the top 5 of 95 elements
  contribute 296 of 551 units. Those want a unit-level quality signal so a contract-bearing unit
  outranks a bare one — deranking, not deletion. The four live indices predate `index_mapping()`
  and `scripts/create_agent_indices.py` now reports the drift rather than silently keeping them.

## 2026-08-17 · M8.9 · Built the ruler for method ranking, then used it to reject my own fix

**Change** `scripts/eval_method_retrieval.py` — 12 natural-language questions paired with the
  symbol(s) that should answer them, reporting R@1 / R@3 / R@8 / MRR over the built library, with
  `--show-misses` printing what outranked the right answer. Plus
  `rag_pipeline/tests/test_method_retrieval_eval.py` (6 tests) so the harness cannot flatter a
  change: an empty library reports itself rather than scoring zero, a library of decoys scores
  exactly zero, and each case's rank is recorded individually.

**Why** "R@1 5 of 8" appears in this project's notes with no harness behind it. It was a
  recollection, not a measurement, and two ranking changes had already been judged against it —
  one of which (an IDF floor) had to be reverted. Before touching ranking again I needed a number
  I could reproduce.

  What the harness measures is **ranking among the units the library already holds**, not
  coverage: I wrote the questions after reading the library, so I chose ones I knew were
  answerable. That bias is stated in the module docstring, because a rising score here must never
  be read as "the library covers more".

**Measured** Baseline on the 551-unit library: **R@1 8/12, R@3 11/12, R@8 12/12, MRR 0.8021.**
  So the honest replacement for the remembered "5 of 8" is 8 of 12 on a different, larger set.

  Then the intended change. I had gone looking for a **contract-completeness** signal, expecting
  the 38%-naked units to be crowding out the good ones. `--show-misses` refuted it in one run:
  every unit outranking a correct answer is itself rich. `as_geodataframe` beats
  `calculate_buffers`; `team_era5_daily` beats `deaccumulate`; and `catchment_ratios_area` — whose
  summary is "Calculates the resource to population ratios for each catchment ..." — ranks
  **first for three different questions**, pushing `camels_geology_attrs` to rank 8. All have
  docstrings, annotations and parameters. The signal I was about to add would have moved nothing.

  The real shape was a missing length normalisation: `_score` was a bare sum, so a long summary
  accumulated common-token matches for free. I implemented BM25-style length normalisation
  (b=0.75) — the textbook fix — and it made things **worse**: R@1 8 → 7, R@3 11 → 9, R@8 12 → 11,
  MRR 0.8021 → 0.6925. Reverted, not tuned. The reason it hurts is visible in the data: the
  correct answers here tend to have the *longer, more specific* summaries
  (`camels_geology_attrs`: "Area-weighted log-permeability and porosity following CAMELS ..."), so
  dividing by length penalises exactly the specificity that makes them right.

**Surprised by** A third of the benchmark is reachable only because someone wrote a docstring.
  Stripping every summary and leaving symbol names alone makes four of twelve cases unfindable:
  three are acronym-named (`e2sfca`, `ebr_of`, and `camels_geology_attrs` for "permeability and
  porosity"), and the fourth is a tokenizer gap — the query "de-accumulate" splits into `de` +
  `accumulate` while the symbol `deaccumulate` stays one token, so they share nothing at all.
  Given that **38% of units have no docstring**, for those units the symbol name IS the whole
  index, and a unit named `e2sfca` with no summary is unreachable by anyone who does not already
  know the acronym. Pinned as a characterization test with the set named, so it is visible if it
  grows.

  That reframes the earlier finding usefully: docstring completeness matters as *retrieval text*,
  where it is decisive, and not as a *ranking* signal, where it measured nothing.

**Not done, deliberately** No further ranking changes. Two attempts have now been reverted on
  measurement (the IDF floor, and this length normalisation), and the third-order move — tuning b,
  or weighting fields differently until 12/12 — would be fitting the ranker to twelve questions I
  wrote myself. The next real improvement is semantic matching over `extracted.embed_text`, which
  now carries the parameter types and declared units M8.7 added, and which needs the cluster.

## 2026-08-17 · M8.8 · The census, and the retracted number was right

**Change** `scripts/graph_census.py` gained `--env-file` and a credential fallback: a git worktree
  does not receive the main checkout's untracked files, so `.env` is simply **absent** here. It now
  looks in this worktree, then in the sibling checkout, rather than requiring a developer to
  duplicate a credentials file into every worktree.

**Measured — the platform Neo4j graph, read-only:**

  ```
  TOTAL nodes 3205 · TOTAL relationships 2899

  nodes by label     Alias 1195 · Contributor 1191 · Publication 206 · Notebook 203 · Map 161
                     Dataset 141 · Oer 44 · Code 38 · Collection 19 · Documentation 6
                     Notification 1
  relationships      ALIAS_OF 1195 · CONTRIBUTED 812 · RELATED 605 · CAN_EDIT 114
                     BOOKMARKED 87 · BELONGS_TO 71 · SUB_COLLECTION_OF 14 · HAS 1
  elements           public 750 · all 793 · 43 private
                     (notebook 23, dataset 11, publication 3, oer 3, code 2, map 1)
  ```

  Public per-type reconciles with the REST API **exactly on all six types** — publication 203,
  notebook 180, map 160, dataset 130, oer 41, code 36.

  And the curated layer reconciles completely: RELATED 605 = 594 both-endpoints-public + 11
  touching a private element; reciprocal storage is **0**, so Neo4j holds each pair once while the
  API presents it on both endpoints (1,184 directed entries ≈ 2 × 592). 594 − 2 self-loops (which
  the API pass had already found and dropped) = **592**, exactly the curated pairs in the graph
  this branch builds. Nothing public is being silently lost, and 11 curated relations correctly
  stay out because they touch private elements.

**Surprised by** Three things.

  **The number I retracted was right.** "3,205 nodes" is exactly correct. Retracting it was still
  the right call — I had asserted it with no query behind it, and being accidentally right is not
  the same as knowing. But it is worth recording that the process cost nothing here and would have
  saved something if the number had been stale, which is the whole bet.

  **This is not mainly a knowledge graph.** Element nodes are **793 of 3,205 (24.7%)**; Alias
  (1,195) plus Contributor (1,191) are **74.4%**. On the relationship side, only RELATED (605) +
  BELONGS_TO (71) + SUB_COLLECTION_OF (14) = **690 of 2,899 (23.8%)** express structure between
  knowledge elements at all. The rest is people, permissions and alias plumbing — ALIAS_OF alone
  is 41% of all relationships. Any "write our edges into the platform graph" plan would have been
  adding to a store whose dominant purpose is identity, not knowledge.

  **`Documentation` has 6 nodes, all private.** An earlier pass reported the label as having "zero
  elements behind it" because the public API returns none. It has six; every one is non-public.
  Absent from the public view is not the same as absent.

  Also: `platform_graph.py`'s docstring denominators were right for four types (dataset 141, map
  161, oer 44, code 38 = the all-inclusive counts) and stale for two (notebook 200 vs 203,
  publication 200 vs 206). Partly-correct recorded numbers are the hardest kind to catch.

**Next** The same credentials unblock the two remaining measurements: the extracted dataset
  `spatial` payloads on the cluster, and pointing `graph_daily.py`'s fetch at `platform_graph`
  to remove its 1,234s REST term.

## 2026-08-17 · M8.10 · A CAPTCHA page counted as a downloaded open-access paper

**Change** `extractors/publication_extractor.py`: `sniff_kind` routes on the file's first bytes
  rather than its extension; `read_document` returns `(text, reason)` so an empty result explains
  itself; `_TextFromHTML` (stdlib `html.parser`) reads a full-text HTML article; `_interstitial_kind`
  names a bot-check / paywall / error / cookie page served instead of the document. `read_note` and
  `source_kind` reach `extracted`, and the reason is prefixed into `contents`. 22 tests.

**Why** The publication probe reported `status: no_text` on `03bc2865__oa.pdf` and pypdf said
  `invalid pdf header: b'<head'`. The file is HTML named `.pdf`, and its content is **IOP
  Publishing's bot-check page** — "please can you confirm you are a human by ticking the box
  below", 356 characters that the DOI fetcher counted as a downloaded open-access PDF. So the
  earlier "94 reachable OA PDFs" counts at least one CAPTCHA wall, and indexing it would have
  filed a bot-check notice as a paper's methods section.

  `_read_text` was `except Exception: return ""`, which collapsed three causes into one status:
  the file is not a PDF, `pypdf` is not installed, or the PDF is a scan with no text layer. Each
  needs a different response — refetch, install, OCR — and `no_text` names only the third.

  A bot check is a refusal. It gets recorded, never satisfied: the detector exists so the count of
  reachable publications means documents, and there is deliberately no attempt to get past it.

**Measured** Over the 8 cached OA documents: 7 parse as PDFs (24k–73k characters each) and 1 is
  the interstitial, now reported as `the server returned a bot-check page, not the document`
  instead of `no_text`. Distinct explanations available for an empty read: **1 → 8** (empty file,
  zip-under-pdf-name, missing pypdf, unparseable PDF, no text layer, HTML with no text,
  interstitial, no reader). HTML articles are now read rather than discarded — a synthetic
  full-text article yields its methods paragraph with `<script>`, `<style>` and `<nav>` stripped.
  Tests **1376 → 1398**.

**Surprised by** Two files in the corpus cache — `ee9ac06c__dataset.xhtml` and
  `f2c58ce4__dataset.xhtml` — are **0 bytes**. A download that failed and was saved anyway. They
  read as "this document contains nothing" rather than "the fetch produced nothing", which is a
  different sentence with a different fix, so the empty-file case is now named explicitly.

  Also notable: the interstitial check has to be length-gated. A real paper about web security can
  legitimately contain "are you a robot", and a test pins that a long document mentioning the
  phrase is still read — refusing it would be a worse bug than the one being fixed.

**Next** The count of reachable open-access publications should be re-derived with this reader
  before it is quoted again; the current figure was produced by a fetcher that trusted the URL.
  That needs the network and the 94-DOI batch, not another local read.

## 2026-08-17 · M8.11 · The contract had to survive four transfers and was dropped at three

**Change** `rag_pipeline/search/agent_kb.py`: `_method_payload` projects a unit's contract into
  the normalized hit `normalize_hit` returns — symbol, signature, doc_summary, params (name,
  annotation, inferred type, declared unit, CRS expectation), returns, invariant checks, pip
  requirements, the version-pinned import line, slice_sha, and a three-valued callability. Plus
  `rag_pipeline/tests/test_agent_kb_search.py`, which had been **committed empty** in `de8c0e4`.

**Why** I ran the agent's own retrieval path against a freshly-emitted element to confirm M8.7 and
  M8.8 had landed, and the hits came back with no contract. `normalize_hit` — the function that
  builds what the agent actually reads — carried `title`, `contents`, `resolved_tools`,
  `runnable_tool` and nothing about the unit.

  So the chain a contract must survive is four transfers, and it was broken at three:
  the emitter did not mirror `unit` (M8.7), the fan-out ran before `library_module` existed
  (M8.8), and the search normalizer discarded it (here). Any one left intact makes the other two
  worthless, and **none of them raises**: the agent simply sees a function it cannot call and
  re-implements it. That failure appears in no log — only as a worse answer.

**Measured** Queried through `agent_kb_search` against an isolated store, for
  "buffer geometries by a distance in metres": `calculate_buffers` now returns its signature, the
  line `from iguide_methods.ke_b1fa548b_spastc_a_spatial_partitioning_algorithm.v_c43f0727bb2a
  import calculate_buffers`, `gdf` annotated `gpd.GeoDataFrame` with `declared_unit: metres` and
  `crs_expectation: projected`, invariants `['projected_crs', 'reject_all_nan']` and
  `pip: ['geopandas']`. Contract fields reaching the agent: **0 → 11**. Notebook blocks carry no
  `method` key at all, so nothing is paid on hits that have no contract. Tests **1398 → 1418**.

**Surprised by** Two of my own tests failed and both were right about intent and wrong in the
  assertion, which turned out to be the useful part. `test_a_not_independently_callable_unit_says_so`
  asserted that `callable` was ABSENT — the name says "says so" and the assertion said "says
  nothing". And a contract with no `callability` block at all was being reported `callable: False`,
  because I had flattened a three-valued verdict to a bool. Absent verdict means nothing analyzed
  it; `needs_globals` means it cannot be imported as-is and the reason ("reads the module-level
  global PARAMS") is exactly what tells the agent to pass it as an argument instead. Now: absent
  emits nothing, negative emits `callable: False` plus `not_callable` and `not_callable_reason`.
  The same fail-versus-cannot-determine distinction the invariant gate already makes, which I had
  just re-broken one layer away.

  Also: `test_agent_kb_search.py` was a tracked, committed, zero-byte file — from a commit of mine
  titled "the trace said no results for ten real hits". An empty test file is worse than a missing
  one, because it reads as coverage.

**Next** Nothing further on the schema. The chain extractor → contract → library module → index
  document → agent payload is now closed and tested at each hop, so indexing to
  `iguide_agent_*` is the next step whenever that is wanted; the four live indices predate
  `index_mapping()` and `scripts/create_agent_indices.py` reports the drift.

## 2026-08-17 · M8.12 · Two smoke failures that were the checker's fault, not the system's

**Change** `scripts/smoke_end_to_end.py`: the library check now asserts that every registry unit's
  module EXISTS rather than that the counts are equal, and the gate check names a run that died
  before the gate instead of reporting the gate's verdict.

**Why** A full smoke run reported 5 failures. Three were honest; two were the checker.

  **"registry matches disk: 551 registry units vs 617 modules — a stale entry advertises a
  contract for code that is gone."** The assertion was `modules == units`, and the 66 extra
  modules are *superseded versions*: slices are content-addressed, so re-extracting an edited
  function mints a new `v_<sha>.py` and leaves the old one importable, which is the whole reason
  an artifact's recorded version is resolvable. The most-re-extracted element has 129 modules for
  92 units. Worse, the message described the OPPOSITE fault — a registry entry whose module is
  gone — which the equality test would have reported with the same words. So the check failed on
  health and would have been unreadable on the real problem. It now walks each unit's advertised
  module and asserts the file is there; 0 missing.

  **"correct run verdicts pass: verdict=cannot_determine"** sent me into `sandbox_verify`. The run
  never got there: `AGENT_CODE_EXEC_IMAGE` is unset, so the executor fell back to
  `python:3.11-slim` and the script died on `import geopandas`. `cannot_determine` was the gate
  being *right* — nothing had run for it to inspect. The check now reads stderr for the exception
  that stopped the script and says so, including the specific remedy when the image is unpinned.

**Measured** With `AGENT_CODE_EXEC_IMAGE=iguide-codeexec:latest` (the image is built and on this
  host, 1.25 GB): the gate returns `pass` for the reprojected run and `fail` for the
  degrees-buffer replay. Smoke totals **9 passed / 5 failed → 12 passed / 2 failed**, and the two
  remaining failures are the honest ones — the agent KB is on the local backend with an empty
  store while a cluster is configured, which is exactly the "not indexed yet" state that was asked
  for. Tests unchanged at 1418.

**Surprised by** The two gate runs print the *same number*: `total 5881.03` for both the correct
  run and the degrees-buffer replay. I went looking for a reporting bug and there isn't one — the
  buffer distance is numerically `25000` in both cases and the area is computed the same way, so
  only the CRS decides whether the answer is 5,881 km² or 5,881 square degrees. That is the most
  compelling argument for the invariant gate I have found: the two outputs are indistinguishable
  to a human reading stdout, and the only thing that separates a correct result from a meaningless
  one is the contract check.

**Next** The smoke script's remaining two failures clear on indexing. `AGENT_CODE_EXEC_IMAGE`
  wants pinning by digest rather than by the `:latest` tag before any artifact is treated as
  reproducible — the check already reports the tag as unverified rather than passing it.

## 2026-08-17 · M8.9 · What the relationship types actually connect

**Change** `graph_census.py` now reports, for every relationship type, the **endpoint labels** and
  their counts — not just the type name and a total. A type's name is not its meaning; what it
  joins is.

**Measured — all 8 types in the production graph, grouped by what they do:**

  *Element <-> element — knowledge structure. 690 of 2,899 (23.8%)*
  ```
  RELATED            605   Publication->Publication 125 · Notebook->Notebook 118
                           Notebook->Dataset 54 · Dataset->Notebook 43 · Oer->Notebook 38
                           Oer->Publication 36 · Dataset->Dataset 34 · Publication->Dataset 33
                           Notebook->Publication 24 · Code->Publication 22
                           Dataset->Publication 17 · Publication->Notebook 11
  BELONGS_TO          71   Notebook/Dataset/Publication/Code/Oer -> Collection
  SUB_COLLECTION_OF   14   Collection -> Collection
  ```
  *Person -> element — provenance and permissions. 1,013 (34.9%)*
  ```
  CONTRIBUTED        812   Contributor -> every element type, and Collection
  CAN_EDIT           114   Contributor -> Dataset 78 · Notebook 30 · Code 5 · Oer 1
  BOOKMARKED          87   Contributor -> Notebook 63 · Dataset 13 · Publication 5 · Oer 5 · Map 1
  ```
  *Identity plumbing. 1,196 (41.3%)*
  ```
  ALIAS_OF          1195   Alias -> Contributor
  HAS                  1   Contributor -> Notification
  ```

**Surprised by** Three properties that only the endpoint labels reveal.

  **`CONTRIBUTED` is exactly one per node, not an authorship relation.** Its endpoint counts are
  206/203/161/141/44/38/19 — identical to the per-label node counts, summing to 812 = 793 elements
  + 19 collections. So every element has exactly one `CONTRIBUTED` edge: it is an owner pointer.
  There is no co-authorship structure in the graph to mine, which is why projecting "shared
  contributor" through it yields a clean partition (contributor NMI 0.58 against the communities)
  rather than a graph — it cannot produce overlapping groups.

  **`RELATED`'s direction is an authoring artifact, not semantics.** Notebook->Dataset is 54 and
  Dataset->Notebook is 43. With reciprocal storage measured at 0, each pair exists once, so the
  arrow records who added the link rather than any asymmetry of meaning. Treating `RELATED` as
  directed would invent 605 one-way relationships that nobody asserted.

  **`Map` appears in zero `RELATED` edges.** 160 map elements — 21% of the public corpus — have no
  curated relation to anything. Their only graph attachment is `CONTRIBUTED` (161) plus a single
  `BOOKMARKED`. This is the same hole seen from the API side (0 of 160 maps connected) now
  confirmed at the source: it is not an API projection artifact, the relations do not exist.
  The five flood-map communities in this branch's graph are therefore held together *entirely* by
  embedding similarity, with zero human-asserted edges among them.

  Also: `HAS` has exactly one instance, Contributor -> Notification. A type that exists for one row.

**Next** Unchanged: extracted dataset `spatial` payloads, and `graph_daily.py` fetching through
  `platform_graph` instead of 750 REST calls.

## 2026-08-17 · M8.13 · Writing the extraction report found two more silent drops

**Change** `extractors/data_extractor.py`: `_describable` replaces the `extracted` field
  **whitelist** with a denylist, so a handler's whole measurement reaches the document and a
  reserved key is refused loudly (`extracted.handler_keys_refused`).
  `extractors/publication_extractor.py`: new `STATUS_NO_METHOD` (`no_method_described`), and
  `is_method_spec` now means "this holds steps" rather than "nothing went wrong". 8 tests.

**Why** Both were found by assembling a per-type report from real artifacts rather than from notes.

  **A whitelist that lost the same information twice.** The dataset asset's `extracted` payload
  originally carried eight hardcoded fields — a document said only "GeoJSON, vector, 566 bytes".
  That was fixed by writing a nineteen-field list, which still dropped everything the list's author
  had not personally needed: a GeoTIFF's `resolution`, `bands` and `dtypes`, and a NetCDF's
  `variables` and `dims`. The corpus's groundwater-policy dataset computes
  `variables: ['crs', 'WAT4_QWATGRD'], dims: {latitude: 288, longitude: 690}` and was indexing
  `{format: nc, family: raster, size_bytes: 803380}`. **For a NetCDF the variable list IS the
  schema**, so that whole format indexed no schema at all. A whitelist fails silently and in the
  same direction every time a handler learns something new; a denylist fails loudly.

  **Three of seven publications were method specs describing no method.** `is_method_spec` was
  defined as `not degraded`, which conflates "the extractor succeeded" with "there is a method
  here". Of the 7 readable open-access documents, three legitimately describe no computational
  method — `d78ceebb` is a PDXScholar citation cover page, `f94c3e60` an editorial on
  pharmaceutical waste, `fd728b4e` an argumentative paper on AI ethics. The LLM read all three
  correctly and returned `steps: []`; the extractor then stamped them
  `llm_extracted / degraded: False / is_method_spec: True`. Now `no_method_described`, not
  degraded — nothing failed — with the distinction from `no_text` (refetch or OCR) kept explicit
  and stated in `contents` so a hit does not read as a broken spec.

**Measured** NetCDF fields indexed: **3 → 5** (gains `variables`, `dims`). GeoTIFF: **4 → 7**
  (gains `resolution`, `bands`, `dtypes`). Publications correctly classified as carrying no method:
  **0 of 3 → 3 of 3**. Publication extraction over the 8 cached OA documents via
  `LLM_PROVIDER=claude-cli` (`CLAUDE_CLI_MODEL=sonnet`, $0 API spend): 1 bot-check wall,
  3 no-method, 3 `llm_extracted` with 12–16 steps, 1 `llm_partial` at 3 of 4 sections. Tests
  **1418 → 1426**.

**Surprised by** How much the report itself was worth. Both defects had been present through every
  green suite and every smoke run, and neither was findable by looking at code — only by putting
  the four types' real output side by side and asking what a reader would learn from each. The
  NetCDF one had also survived a commit whose message says it fixed exactly this class of bug.

  One more thing the report surfaced without needing a fix: `835ca51a`'s coverage accounting works
  end to end — `chunks parsed 3 of 3 attempted / 4 total`, `chars 55437 of 63177`, with
  `PUB_MAX_CHUNKS` named as the cap in `contents`. A spec built from three quarters of a paper says
  so in the text the agent reads.

**Next** Nothing outstanding on extraction content. The whitelist→denylist change means a new
  handler field reaches the index without an emitter edit, which removes the step that was being
  forgotten.

## 2026-08-26 · M8.16 · Re-derived the publication reach: 53 of 203, and the old 94 was real

**Change** Ran the fixed resolver over all 203 publications and audited the result with
  `scripts/audit_publication_reach.py`. Also corrected a stale error string that still named
  `external_link` — the wrong field, and the one that caused the branch to fire 195 times.

**Why** "94 reachable open-access PDFs" was quoted in this log and in `open_access.py`'s docstring
  and was produced by a fetcher that trusted the URL. The M8.10 reader changed what "reachable"
  can mean, so the figure needed re-deriving rather than re-quoting.

**Measured** The pipeline, stage by stage, over 203 publication elements:

  | stage | n |
  |---|---|
  | DOI extracted from a recorded source link | **177** (169 resolved + 8 already cached) |
  | no DOI anywhere | 26 |
  | DOI, but no open copy exists | 49 (44 closed + 5 not a registered Crossref DOI) |
  | open access, but Unpaywall names no direct PDF | 39 |
  | **Unpaywall named an open PDF** | **89** |
  | download failed | 35 |
  | file on disk | 54 |
  | **readable document** | **53** |
  | publisher wall | 1 |

  **Reach: 53 of 203 (26.1%).** Licences on what we hold: 39 cc-by, 3 cc-by-nc-nd, 4 unstated.

  The reconciliation is exact: **89 offered − 35 download failures − 1 wall = 53 readable.**

**Surprised by** The old number was not wrong, it was *mislabelled*. 89 is this run's count of DOIs
  for which Unpaywall names an open PDF, against the docstring's "94 of 176 resolvable DOIs" — the
  same quantity, reproduced within five. What it never meant was "we hold 94 readable papers". The
  36-document gap is entirely the two stages downstream of Unpaywall, and quoting the resolver's
  answer as the corpus's content skipped both.

  The dominant loss is publisher bot protection, and it is concentrated:
  **30 of the 35 download failures are HTTP 403** — MDPI 13, Wiley 6, Taylor & Francis 4, ACM 2,
  ScienceDirect 2, plus PNAS, JBC and a KCL repository. Unpaywall correctly reports these as
  legally open; the server simply refuses an automated GET. That is a refusal, recorded and not
  worked around — the same stance as the bot-check page in M8.10, and it is now the single largest
  recoverable category in the type. An institutional proxy or a publisher's own bulk-access
  programme would be the legitimate route, and neither is a code change.

**Two process failures of my own, both the shape this log exists to catch.** I set a monitor to
  wait for the batch and keyed it on "the output JSON has ≥200 elements" — which the **stale** file
  from the pre-fix run already satisfied, so it fired instantly and reported "DRIVER FINISHED"
  while the driver had barely started. And an interim audit reported a document as
  "unparseable by the reader" when it was simply half-downloaded; pypdf's "EOF marker not found" on
  a partial file is indistinguishable from corruption. Fixed in M8.15 by flagging any file written
  in the last 30 seconds. Both were checks that produced a confident number about the wrong thing.

**Next** Nothing outstanding on the count. The 39 "open access, no direct PDF" cases are the
  cheapest remaining recovery — Unpaywall records a landing URL for many of them, and the M8.10
  HTML reader can read a full-text article — which would test the reader against real publisher
  HTML rather than the synthetic fixtures it has now.

## 2026-08-26 · M8.17 · The HTML guard was refusing the landing pages the resolver asked for

**Change** `open_access.resolve` scans `oa_locations` for a PDF, not only `best_oa_location`, and
  records which location it used; the driver falls back to the landing URL and verifies
  readability BEFORE counting a fetch; `sources.fetch_url` takes `allow_html`;
  `opensearch_emitter.emit` finally calls its own skip-if-unchanged machinery. Plus
  `scripts/export_dataset_details.py` and `scripts/build_extraction_browser.py`.

**Why** Both were named as gaps in the previous report.

  **Skip-if-unchanged was written, tested, and unwired.** `run_fingerprint`, `previous_run` and
  `record_run` had no caller outside the tests, so every re-index rewrote the corpus and
  `iguide_agent_ingest_runs` did not exist on the cluster. Now wired — with a presence check,
  because a matching fingerprint means "this run would write the same documents", not "those
  documents are in the index". Without it, a recreated index stays empty forever while every
  re-ingest reports a clean skip.

  **The landing-page recovery was blocked by its own guard.** `fetch_url` refuses HTML as a
  landing page, which is right for a dataset. `_wants_html` only recognises a URL whose *path*
  ends in `.html`, and a publisher article URL (`mdpi.com/2073-4441/12/1/123`) does not — so the
  new fallback was refused **30 times** by the check protecting it. The caller's intent is the
  fact, not the URL's suffix.

**Measured** Publication reach, over three runs of the same 203 elements:

  | | fetched | readable | reach |
  |---|---|---|---|
  | before (M8.16) | 54 | 53 | 26.1% |
  | + alternate OA locations & landing fallback | 58 | 57 | 28.1% |
  | + `allow_html` | **73** | **72** | **35.5%** |

  So 53 → 72 readable documents, and "open access, but no direct PDF url" is gone as a category.
  What remains: 44 with no open copy, 36 publisher 403s, 26 with no DOI, 10 landing pages whose
  HTML carried no extractable text, 5 not in Crossref, 5 more bot-check pages. Tests
  **1455 → 1478**.

**Surprised by** The 403s went 30 → 36 while everything else improved. The extra six are landing
  URLs on hosts that also refuse an automated GET — the same publishers, refusing the second
  attempt too. Worth stating plainly: this run made no progress against bot protection and was
  never going to. It is now the single largest recoverable category and needs an institutional
  proxy, not code.

  Also five *new* bot-check pages surfaced, which is the fallback working as intended: they are
  landing pages that fetched with a 200 and were then rejected by the reader instead of being
  counted as papers. The driver checks readability before crediting a fetch for exactly this.

**A destructive mistake of mine** I overwrote `.claude/launch.json` — tracked, and holding three
  existing server configurations — by writing the file instead of appending to it. Restored from
  HEAD with my config appended. I should have read it first; the tool even told me the file already
  existed and I did not stop to check what was in it.

**Next** Nothing outstanding from the report. The browser page regenerates from
  `outputs/` in one command, so it stays current without hand-editing.

## 2026-08-26 · M8.23 · The unit count I reported was 551; the registry now holds 840

**Not my change.** A concurrent session committed `M8.21` (registry key collisions) and `M8.22`
(browser cross-links + publication specs) while I was finishing the publication run. This entry
exists because those commits invalidate numbers **I** published, and the correction belongs next to
the original claim rather than only in their commit message.

**What they found** The registry key was `{element_package}.{symbol}`, which is not unique when one
element spans several files — and a code element IS a whole repository (`2641203f` has 35 source
files). Two files defining `load` collide on one key; both modules get written, because they are
content-addressed with differing shas, and the second registry write drops the first. The module
stays on disk with nothing pointing at it. Their measurement: 58 of 362 callable code units, 16%,
unreachable. Proven directly — two same-named functions in two files produced 2 modules and 1
registry entry holding the second file's code. The cross-element form of this same bug had been
fixed once before, and the fix was to introduce this very key; the remedy is the same one level
down.

**Corrected figures, measured just now against the settled registry** (no extraction running,
written 12:55:45):

  | | reported | now |
  |---|---|---|
  | library units | 551 | **840** |
  | from the code extractor | 302 | **591** |
  | from notebooks / datasets | 227 / 22 | 227 / 22 (unchanged) |
  | naked (no docstring, annotation or return) | 38% | **44%** (373) |
  | zero-arg | 14% | **17%** (151) |
  | carrying an invariant | 13% | **10%** (84) |
  | top 5 of 95 elements | 296 of 551 (54%) | **573 of 840 (68%)** |
  | largest single element | 92 | **193** |

  The entire increase is in code units, which is consistent with a collision that only bites
  elements spanning many files. I have NOT reproduced the arithmetic from 302 to 591 — their fix
  accounts for 58, and the remainder is presumably the per-file keys now counting units that
  previously collapsed onto one key. Recording the gap rather than inventing a reconciliation.

**Why this matters beyond bookkeeping** Recovering the lost units made the library **bigger and
thinner at the same time**: naked 38% → 44%, invariants 13% → 10%, and the top five elements now
hold two-thirds of everything. That is the report's own "volume is the wrong objective" point,
sharpened — 289 more units bought no more documentation and no more enforceable contracts.

**Stale published artifact** `I-GUIDE Extraction Results`
(54f2b535-0571-41fc-9085-195bdbf76d5e) states 551 units, 302 code units, 38% naked, 296-of-551
concentration, and "8 cached documents, not 203" for publications. All six are now wrong. The
Extraction Browser (25250db2) is current — it regenerates from `outputs/` and reads 840. The report
needs a rebuild; its source HTML was not kept on disk, so that is a re-authoring job rather than a
substitution.

**Next** Nothing blocking. Whoever rebuilds the report should take the numbers from this entry
rather than from the previous one.

## 2026-08-26 · M8.23 · One field name blanked the page, and 25 tests said it was fine

**Change** `build_extraction_browser.py` renames a publication's declared parameters to
  `declared_params`; the template's search-index pass is defensive; new
  `rag_pipeline/tests/test_extraction_browser_runs.py` executes the page's JavaScript under node.
  Plus `export_publication_specs.py` run over all 73 readable documents.

**Why** The user opened the browser and nothing rendered. In M8.22 I gave a publication's declared
  parameters the key `params` — the same key a unit uses for its parameter **list**. The page
  builds its search index in one pass *before* rendering anything and calls `.map()` on every
  record's `params`. An object has no `.map`, so the first publication with parameters threw, the
  IIFE died, and not one row was built.

  Every test passed. They check the data contract — each record has an id, a kind, a renderable
  name — and each record *was* individually valid. The failure was in what the page DID with them.
  The docstring in that file claimed it covered "a page that comes up blank". It did not.

**Measured** Diagnosed by extracting the script and running it under node, not by reading it:
  `TypeError: (r.params || []).map is not a function`. After the fix the same harness reports
  `1224 of 1224 record(s)` and a populated tally.

  The new tests were verified against the *broken* page rather than assumed: stash the fix,
  regenerate, and **6 of 6 fail**; restore, and 6 of 6 pass.

  Publication specs over all 73 readable documents (claude-cli / sonnet, 22 min, $0 API spend):
  53 `llm_extracted`, 10 `llm_partial`, 8 `no_method_described`, 1 `no_text`, 1 `llm_unavailable`.
  **63 carry a method spec: 951 steps, median 13 per paper, 414 distinct datasets and 394 distinct
  tools named, 60 with declared parameters.** The richest is HISDAC-US at 34 steps / 22 datasets /
  24 parameters. Tests **1504 → 1510**.

**Surprised by** Nothing about the bug — it is an ordinary name collision. What matters is the
  pattern: this is the third time this session a check reported success while the thing it was
  meant to establish was false, and the first time I shipped it rather than catching it. All three
  had the same shape — asserting the artifact's *description* rather than its *behaviour*. The
  emitter tests checked that documents were built, not that they carried a contract. The layout
  tests checked that CSS rules were present, not that panes scrolled. These checked that records
  were well-formed, not that the page rendered.

  The remedy that generalises is the one added here: run the real thing and look at what comes
  out. A stub DOM and 20 lines of node were enough, and would have caught it before it shipped.

**Next** Nothing outstanding on the browser. `--max-files` still defaults to 60, which silently
  truncated two code elements before this session raised it; the driver now records
  `truncated_by_max_files`, but the default itself is worth revisiting against build time.

## 2026-08-26 · M8.26 · Staging: the 44 dataset loaders finally have something to load

**Change** `agent_runtime/staging.py` — `stage_url`, `stage_object`, `stage_element`,
  `staged_inputs`; three LangChain tools in `langchain_granular_tools.py`; the names added to
  `graph_state.RAG_COMPONENT_TOOL_NAMES` and the tools given to the code peer. 41 tests.

**Why** Extraction generates 44 loaders shaped `def load_x(staged_path)` and nothing could produce
  a `staged_path`: the only staging that existed resolved a user-uploaded `file_id`. An entire
  element type was **readable and not runnable** — the agent could describe a dataset's schema and
  could not open the file.

  The fetch runs agent-side, and that is the security property rather than a convenience. The
  sandbox runs `--network none`; MinIO keys, the OpenSearch credentials and a route into the
  platform's private subnet all live in the agent process. Fetching inside the container would mean
  putting credentials inside it. So only bytes cross the boundary, and the container learns nothing
  about where they came from.

**Measured** End to end against a real corpus dataset served over HTTP: staged **37,079,384 bytes**
  with sha256 `b8436072a7df645f…` to `/work/inputs/export.csv`, provenance appended to
  `inputs.jsonl`, and then `load_chicago_crime_data_2026(staged_path)` — the generated loader,
  imported from the real library — returned a **DataFrame of 128,886 rows and 22 columns**. That is
  the first time a platform dataset has gone from a search result to an open frame. Tests
  **1537 → 1578**.

**The load-bearing part is the SSRF guard.** The URL is chosen by a language model and this process
  can reach `169.254.169.254` (instance credentials), `127.0.0.1` (its own API) and
  `10.0.147.52:7687` (the platform's Neo4j). Every resolved address must pass, not just the first,
  because a host that resolves public once and private next is the whole DNS-rebinding trick. The
  refusal names the address, explains why, and names `AGENT_STAGING_ALLOW_PRIVATE` — a rejection a
  reader cannot act on gets worked around by disabling the check wholesale.

**Surprised by** How close the workspace key came to being silently wrong. I first wired the tools
  with a `session_id` variable that was not the one the execution tools use — they key on
  `child_thread_id(thread_id, "code_exec")`. Staging into a different key would have put the bytes
  in a directory the sandbox never mounts, and the failure would have surfaced inside a container
  with no network as "file not found". A test now reads `graph.py` and asserts the two calls agree,
  because nothing else in the system would notice.

**Next** `stage_element` resolves through `sources.source_link`, so a portal pointer is reported as
  having no file rather than as a broken download. What is still missing is the prompt work: the
  code peer now HAS the tools, and the measured lesson of this project is that a capability the
  model must elect is a capability you do not have. Whether it reaches for `stage_element` unprompted
  needs a live run to answer.

## 2026-08-26 · M8.27–28 · Four live turns on one dataset question

**Change** Completed the `claude-cli` tool denylist and added `native_tools_visible()`; a loader
  whose signature takes `staged_path` now carries its own staging instruction in the evidence view.

**Why / Measured** Four runs of: *"Using the Chicago crime dataset on the I-GUIDE platform, what
  are the five most common primary crime types, and how many records does each have? Compute the
  counts from the actual data file."*

  **Run 1 — the provider, not the agent.** Answer: "every tool returned *No such tool available*;
  the only tool that responds here is `ListAgents`." `ListAgents` belongs to the CLI *subprocess*.
  `_AGENT_TOOLS`, the `--disallowed-tools` list that stops `claude -p` behaving like an agent, was
  missing `Agent` and `ListAgents`, so the model had a real tool channel with one working tool and
  preferred it over the prompt-described JSON envelope the shim depends on. **The turn measured
  the denylist's hole, not the agent.** `--allowed-tools` is not the fix — it governs permission
  prompts, not availability, and a name matching nothing still leaves all 13 tools present
  (verified). List completed; `native_tools_visible()` now asks the running CLI and reports NONE.

  **Run 2 — staging elected.** `kb_method_search`, `get_method_contract`, `agent_kb_search`,
  `get_kb_block`, **`stage_element`** — and it staged the real file: **37,079,384 bytes, sha256
  `b8436072a7…`**, matching the end-to-end probe exactly. Then Docker was down, so no compute.

  **Run 3 — the right answer.** Docker restarted. THEFT 27,824 · BATTERY 23,885 · CRIMINAL DAMAGE
  14,047 · ASSAULT 11,575 · MOTOR VEHICLE THEFT 10,221, from 128,886 records. **Independently
  verified against the file: every number matches.** Staging was not re-elected because the
  workspace persisted from run 2 — correct behaviour, but it means run 3 does not demonstrate
  election.

  **Run 4 — fresh workspace, and the honest failure.** `stage_element` appears **zero** times in
  the entire run state. The agent read the loader's advertised signature
  `load_chicago_crime_data_2026(staged_path)`, called it without staging, got "block not found",
  and fell back to `mcp_count_crimes_per_community` — a different tool over a different population.
  It reported THEFT as **9,993** against the file's **27,824**, and said so: "only incidents
  successfully spatially joined … may not be included."

**Surprised by** How exactly run 4 reproduced this project's central lesson. The evidence view
  advertised a parameter and never said where it comes from. So the loader now carries
  `FIRST call stage_element("<id>") to obtain staged_path — this file is not in the sandbox until
  you do`, attached to the specific unit rather than stated as a general rule, because a general
  rule is what run 4 already had and skipped.

  Worth recording separately: the agent's *judgement* was good in all four runs. It refused to
  fabricate twice, diagnosed the Docker outage precisely, and volunteered the population caveat
  that made run 4's wrong number detectable. The failures were in what it was told and what it
  could reach, not in what it decided to do about them.

**Next** The invariant gate returned `cannot_determine` with all-zero counts and "findings were
  not retained" on the successful run, and reported `unrecognised unit 'records'`. So the number
  is right and the system cannot yet say so — the same gap this whole thread has been closing, one
  layer further in.

## 2026-08-26 · M8.29 · The gate can now verify a counting run

**Change** `agent_runtime/sandbox_verify.py`: count-unit aliases; `_count_finding` actually checks
  a declared count; new `check_count_population`; an explicit `not_applicable` finding when a run
  had nothing to check. `graph.py`'s synthesised message distinguishes "checked nothing" from
  "findings lost". 9 tests.

**Why** The live run that produced the right answer — THEFT 27,824 of 128,886 records — could not
  be verified, for three separate reasons:

  1. **`records` was not a unit.** `count` was in `_KNOWN_UNITS`; `records` was not, and nothing
     aliased it. The model wrote the natural word for what it was counting and got "unrecognised
     unit 'records'; not checked". One UNKNOWN downgrades the whole verdict, so a correct answer
     was reported unverified. Same shape as the `km²` case already recorded above it.
  2. **Recognising a unit is not checking it.** Even spelled `count`, the value was never
     inspected — `unit count` would have passed for `-3`. A count is the one unit whose value the
     gate can judge alone.
  3. **"Nothing to check" was reported as evidence lost.** A helper `execute_code` call with no
     frames and no declared outputs produced zero findings, and the reader got
     `cannot_determine (counts all zero) but its findings were not retained` — which sends someone
     looking for missing evidence that never existed.

**Measured** Run 3's actual code, re-run in the real sandbox against the real 37 MB file:

  | | before | after |
  |---|---|---|
  | verdict | `cannot_determine` | **`pass`** |
  | counts | `{pass 0, fail 0, unknown 0}` | **`{pass 10, fail 0, unknown 0}`** |
  | inspected | — | `['df']` |

  And the new population check on the failure run 4 actually made: a count of 128,886 declared
  from a frame of 40,000 now **fails** with *"declared count 128886 exceeds every frame in this
  run (df=40000), so it cannot have been counted from any of them"*. Tests **1580 → 1599**.

**Surprised by** The structural binding test caught my own change. `_count_finding` and
  `check_count_population` are referenced by the gate but the gate is INLINED into the sandbox
  script by an explicit function list, and I forgot to add them —
  `test_no_name_in_the_inlined_gate_is_unbound` named both, immediately. That test was written for
  exactly this after seven NameErrors were swallowed as `cannot_determine`, and it is the first
  time it has caught a fresh one rather than documenting an old one.

**Deliberately not done** `check_count_population` does not guess which frame is "the" population.
  Run 4's real error — counting from a 49,789-row spatial join while answering a question about
  128,886 records — is not detectable from inside the sandbox, because both numbers are legitimate
  counts of the frames present. What the check does is RECORD the population, so the discrepancy is
  visible to a reader rather than invisible. Judging it would need the question, which the gate
  deliberately cannot see.


## 2026-08-27 · M8.30–M8.31 · Names for the artifacts, and a page that shows them per notebook

**Change** Two commits.

  `_block_title` in `extractors/notebook_extractor.py` replaces the positional `<element> — cell
  12` with the name the author already wrote, in six tiers: markdown heading → first sentence of
  prose → the comment block atop the code → what the cell defines → what the code does → `cell N`.

  `scripts/export_notebook_blocks.py` (new) re-reads the 174 cached notebooks and writes every
  block and unit; `build_extraction_browser.py` gains a `notebook` record kind and the page gains a
  notebook detail view plus a **By element** mode that groups every record under its parent
  knowledge element.

**Why** A block titled `cell 12` identifies a cell and describes nothing — a retrieved block told
  the agent, and the browser reader, only its ordinal. The author's name for the step was one field
  away in `markdown_context` the whole time: `## 1c. Project All Files to the Same CRS` sat unused
  on the cell titled `cell 10`.

  And notebooks — the largest source in the corpus — were the only type with nothing to browse.
  Dataset, publication and code each have a per-element outcome file under `outputs/`; a notebook's
  blocks and units existed only inside a search index.

**Measured**

  | | before | after |
  |---|---|---|
  | blocks named `cell N` | 871 / 3,689 | **469 / 3,689** |
  | blocks with an authored name | 2,116 | **2,464** |
  | notebooks with a browsable record | 0 | **174** |
  | cells reachable in the page | 0 | **3,830** |
  | units shown per notebook | 0 | **382** (227 callable) |
  | records grouped under a parent element | — | 1,398 across **543** elements |
  | tests | 1,789 | **1,807** |

  The 227 callable units match the registry's notebook count exactly, which is the cross-check that
  the export and the library agree on what shipped.

**Surprised by** Three things, all the same shape — *a check that reports success while the thing
  it was meant to establish is false*.

  1. **Tiers 3–5 were guesswork and the corpus said so.** The first draft named cells from a
     substring list (`plot`, `buffer`, …) plus the classified constructs. Probing the 871 cells
     that actually reached that point: constructs fired on **15**. What those cells carry is a
     leading comment and `def` lines. Rebuilding the tiers around the measurement took `cell N` from
     871 to 469; the guess would have taken it to about 840.

  2. **The page's node harness only ever exercised the first render.** Every detail view sits behind
     a click, so the hostile-record test — the one written specifically to prove that a malformed
     field cannot blank the page — passed only because nothing was ever opened. Extending the
     harness to click a row made it fail immediately on `chips(42)`: the search index had been
     hardened after the original blanking, the detail views never had.

  3. **A blank column raises nothing.** The exporter asked blocks for `cell_order`; the extractor
     writes `order`. All 3,830 ordinals came out `None`, the column rendered empty, and every test
     passed. Same class: `file_io` is `{"referenced": [...]}` and passing it through as a list gave
     the page an object to iterate.

  Then a fourth that no test could have found: the page has **no `<meta charset>`**. Served over
  HTTP it decodes as windows-1252, and every em dash — including the separator in every block
  title — renders as `â€"`. Opening the file from disk guessed UTF-8, which is why it survived
  this long. Confirmed only by opening the page in a real browser, which is the standing rule here
  precisely because of cases like this.

**Deliberately not done** Units carry no cell order. A unit's provenance records the element and
  the source file, never the cell it came from, so the notebook view does not claim one — the field
  was emitting `None` for all 382. Adding `cell_order` to the unit contract would change
  `contracts.py` and need a full re-ingest to take effect; it is a real gap, recorded rather than
  faked with a blank column.


## 2026-08-29 · M8.32–M8.34 · A store the agent owns

**Change** Postgres 17 + pgvector as the system of record for extracted content
  (`extractors/kb_db.py`), one compose service bound to loopback. `scripts/backfill_kb_db.py`
  loads it from the built method library and the JSON exports. `AGENT_KB_DB=1` routes
  `kb_method_search` and `get_method_contract` at it, with the registry as the fallback.

**Why** The question was "can we put the extracted content into a database" — and it already is
  in two, both of which belong to the *website*. The agent writes into `iguide_agent_*` indices
  on the platform's OpenSearch cluster, so re-indexing its own knowledge base needs write access
  to production website infrastructure, and a mapping change means re-crawling the corpus:
  network fetches plus an LLM call for each of 203 publications, because nothing durable holds
  what extraction found. The two file stores that grew alongside them — `_registry.json`, 2.5 MB
  rewritten whole, and `outputs/*.json` — have no constraints at all.

  So the gap was never a search index. It was a record with constraints, transactions and joins,
  which the indices can be rebuilt from. Scale is explicitly **not** the argument: ~10k rows.

**Measured** Backfill, no re-extraction and no LLM spend: 543 elements, 989 units, 3,830 blocks,
  43 dataset files, 130 dataset outcomes, 203 publications. Idempotent.

  | | registry | postgres |
  |---|---|---|
  | known-item MRR | 0.944 | 0.966 |
  | known-item recall@1 | 0.900 | 0.936 |
  | right element in the top 5 | 88/95 | 93/95 |
  | **median latency, the agent's path** | **58.8 ms** | **7.8 ms** |
  | slices byte-identical to the mounted file | — | 840/840 |
  | duplicate unit / orphan unit / refused unit carrying code | all accepted | all rejected |

  Retrieval quality is close enough to call a wash at a 120-unit sample. The load-bearing
  results are the constraints and the fidelity, not the ranking.

**Surprised by** Five things, and four of them were my own claims failing on contact.

  1. **The registry can answer by element title, and I said it could not.** `element_title` is
     absent from all 840 provenance records, so the `element` field falls back to a hex id and
     its 1.0 weight is dead. I wrote in a docstring that it therefore "cannot answer at all".
     It scores **92.6%**: the registry KEY embeds a slug of the element title. The slug is
     truncated at ~50 characters — "…Mapping using Physics-Aware Spatial AI" is stored as
     "…mapping_using_phy" — so the limit is real and far smaller than I asserted. I then built a
     query to demonstrate the difference ("physics aware spatial AI") and **neither** store found
     the element, because a title indexed at weight C loses to symbols matching at weight A.
     The demonstration was abandoned rather than tuned into working.

  2. **The agreement test skipped silently, twice.** `conftest` points
     `AGENT_METHOD_LIBRARY_DIR` at an empty directory on purpose, so the one test justifying the
     whole change — does the Postgres contract match the registry's, field by field — compared
     nothing and passed. The fallback test had the same disease: it searched a library that was
     not there and passed for the wrong reason. Once it actually ran it immediately found the
     Postgres contract dropping `invariants`, which is where "requires a projected CRS" lives.
     Every other field matched, so one eyeballed result would never have shown it.

  3. **A tsvector caps at 1 MB, and the corpus holds a 4.9 MB markdown cell** — 99.8% base64
     image data, an image embedded in a notebook. Not a tuning question: an unbounded index
     expression made the row impossible to insert. Every index expression now bounds its input;
     the columns keep everything. That blob also goes into `contents` for the OpenSearch doc and
     gets embedded today, which is a separate cost nobody has looked at.

  4. **`websearch_to_tsquery` ANDs its terms.** "buffer geometries by a distance" required one
     unit to match buffer AND geometry AND distance, and returned **nothing** across all 840. The
     ranker being replaced sums the weights of whatever matched, so the query is built as an OR
     over the same lexemes the indexed side used.

  5. **The export was truncating cells.** `export_notebook_blocks` clipped code at 4,000
     characters — right for a browser row, wrong for the file a database is backfilled from: 79
     cells and 283 markdown blocks were stored truncated, so the record would have held cells
     that no longer compile. The clip moved to the page, which is the consumer that actually has
     the constraint.

**Deliberately not done** OpenSearch is untouched and still serves every search tier; nothing was
  migrated off it, and `AGENT_KB_DB` is off by default so every existing path is unchanged. The
  emitter (a fifth `EMIT_` target writing to Postgres during ingest) is not built — the backfill
  proves the schema against real data first. Making the mounted library an actual projection —
  regenerating `iguide_methods/` from `slice_source` — is the step the 840/840 fidelity result
  makes safe, and it is not taken yet.


## 2026-08-29 · M8.35–M8.40 · Ablating the knowledge base, and the four things that had to be
## fixed before the measurement meant anything

**Change** Four analysis problems over two real corpus datasets, run with and without the
  extracted KB (`scripts/ab_kb_problems.py`, `scripts/report_ab_kb.py`), graded against
  independently computed ground truth. To make that possible: `agent_kb_search` can read the
  Postgres record; `AGENT_ABLATE_KB` actually removes the KB; the claude-cli shim retries once
  when its envelope does not arrive; and `stage_element` accepts the short element id.

**Why** "Show the difference with and without the agent KB" needed four preconditions that did
  not exist. Each one, missing, produced a confident wrong answer rather than an error — which is
  the entry.

**Measured** Ground truth for the headline problem, with two DIAGNOSTIC wrong answers so a
  reported number names the mistake rather than merely failing:

  | crimes within 100 m of a major street | count |
  |---|---|
  | correct, EPSG:26916 metres | **72,658** of 128,855 |
  | 100 read as FEET in the streets' native EPSG:3435 | 47,634 |
  | buffered in degrees | 68,419 |

  Final sweep, after all four fixes, n=1 per cell:

  | problem | ablated | with KB |
  |---|---|---|
  | p1 · 100 m street proximity | 2/2 correct, 475 s | 2/2 correct, 1,081 s |
  | p3 · busiest month + top types | 2/2 correct, 492 s | 2/2 correct, **296 s** |
  | p4 · negative control (no such method) | refused correctly | refused correctly, after 8 searches |
  | | **4/4 checks** | **4/4 checks** |

  And the before/after that the staging fix produced on p1's with-KB arm:
  **0/2 in 7,612 s with 21 KB calls and no code run → 2/2 in 1,081 s.**

**Surprised by** Five things. Four were my own measurement, and every one of them looked like a
  result rather than a fault.

  1. **There was nothing to ablate.** `AGENT_KB_BACKEND=local` is the default and that store has
     never been written here, so `agent_kb_search` returned `count: 0`. The with-KB arm would have
     been inert while reporting numbers.

  2. **`enabled_search_methods` is not an ablation.** The code and analyze peers hold
     `_CODE_PEER_KB_TOOLS` "deliberately independent of the request's enabled_search_methods", and
     `_direct_search_sweep` unions the KB in without the model electing anything. The no-KB arm
     called `agent_kb_search` once and answered both checks correctly — a control that was really
     a treatment, reporting success.

  3. **The instrumentation broke the arm it measured.** First clean-looking result: no_kb 8/8,
     with_kb 0/8, every with-KB run under a minute and never reaching the sandbox. `_ToolCounter`
     wrapped the KB tools as `def counted(*a, **kw)`; LangChain infers tool schemas from the
     SIGNATURE, so `agent_kb_search` was advertised with no parameters and the peer turn failed on
     first use. Only the treatment arm calls those tools. What isolated it was the SHAPE, not the
     numbers: 24-46 s against 165-528 s, and a pre-instrumentation run that had passed 4/4.

  4. **The dev shim was the dominant noise source** — 0-6 malformed replies per run, and a
     malformed reply ENDS the agent loop, turning the turn that was going to call `execute_code`
     into "I don't have access to run that". Same arm, same problem, 2/2 then 0/2, on shim noise.

  5. Then the finding the experiment existed for. **The KB was steering the agent into a call that
     could not succeed.** `stage_element` resolved against `/api/elements/{id}`, which takes only
     the full UUID, while the 8-character short id is the canonical form everywhere in extraction
     — and the evidence view emits exactly that short form at the point of use: `FIRST call
     stage_element("265e6957")`. The with-KB arm found the right method, stated the right plan,
     and spent 2h07m and 21 KB calls unable to obtain a staged path, correctly refusing to invent
     a number. That instruction was added after watching an agent call a loader without staging;
     it had been pointing at a broken call ever since.

  And one genuine KB result, from the negative control. With 989 units to search, the KB surfaced
  a real near-miss — `s2_indices(lat, lon, date, epsg, shape, transform)`, which does compute NDVI
  from Sentinel-2 — and the agent still refused, naming why it does not answer the question: one
  scene at one point/date, no monthly compositing, no cloud masking, no basin aggregation. The
  risk the control was designed to catch did not materialise.

**Deliberately not done** No claim that either arm is better. n=1 per cell after four rounds of
  fixing the harness, and the same arm on the same problem has gone 2/2 and 0/2 on shim noise, so
  the responsible output is per-run with the degradation count beside it. What IS supported: both
  arms answer these problems correctly and reproducibly once the staging bug is fixed; the KB
  halves the time on p3 and doubles it on p1; and it does not induce false claims of reuse.

  The four problems also leave the publication half of the corpus — 203 elements with extracted
  method specs — entirely untested. The best composable candidate found while looking:
  publication `355786a5` (E2SFCA for COVID healthcare access, 31 steps carrying the actual
  parameters: 10/20/30-minute bands, weights 1/0.68/0.22) against notebook `3b45070e`, whose nine
  callable units include `e2sfca(..., distances: List[float], weights: List[float])` — the
  paper's numbers drop straight into the signature. The `IMPLEMENTED_BY` edge that would record
  that pairing is still never written.


## 2026-08-29 · M8.42–M8.48 · A strong case, a challenging case, and one defect three times

**Change** Two new A/B cases the earlier four could not reach — compose a workflow from a
  publication's method spec plus a separate element's callable implementation (p5), and resolve
  a symbol two elements define identically (p6) — and seven revisions to the knowledge
  *reference* strategy that running them forced.

**Why** The first four problems named their datasets and asked for counts, which the agent does
  well unaided; both arms scored 4/4 and the experiment said nothing about the knowledge base.
  A strong case has to need what only pre-computed cross-document structure provides.

**Measured** p5 asks for a paper's catchment bands, its distance-decay weights, whether the
  platform already implements the method, and what that implementation requires of its inputs:

  | | ablated | with KB |
  |---|---|---|
  | p5 · spec + implementation | **0/4** | **4/4** |
  | p6 · ambiguous symbol | picked one silently | refuses, names both elements |

  The with-KB answer states 0-10/10-20/20-30 minute bands, weights 1/0.68/0.22 mapped to the
  right bands, the pinned import line for `e2sfca`, and the projected-CRS precondition. The
  ablated arm, with the same web access and the paper publicly available, got none of it. This
  is the first decisive separation in the whole experiment, and it lands exactly where the
  thesis predicts: not on analysis the agent can do, but on structure it cannot cheaply
  rebuild.

  Retrieval, over six probe queries: actionable hits 24/48 -> 31/48, tokens per search
  3,818 -> 2,620, callable methods returned for three method-shaped questions 1/2/2 -> 4/4/4.

**Surprised by** One defect, three times, and it is architectural rather than incidental:
  **adding to the record is not publishing to the agent.**

  1. `search_kb` queried `block` and `unit`. 203 publication method specs — one LLM call each
     to produce — sat behind their own GIN index that nothing queried. Asked for the paper's
     weights, the agent answered "its abstract (the only text retrieved) does not state the
     specific minute-based catchment bands".
  2. With the channel added, a spec carried whole at 6,000 characters arrived at 1,858. Four
     independent truncation budgets sit between the record and the model, in three modules —
     producer 6,000, normaliser 1,800 (the CELL budget), evidence renderer 2,500, enrichment
     sub-render 900 — each reasonable alone, none aware of the others, so the producer's budget
     was decoration.
  3. With that fixed, `error`, `import_line_candidates` and `disambiguate_with` were all on the
     row and none reached the model, because `_method_units_as_documents` builds its contents
     from four fields chosen when it was written. The agent called `get_method_contract` eight
     times and still reported "the entries only give signature and dependency list".

  And three of my own fixes were wrong in instructive ways. A 50/50 **quota** capped units as
  well as guaranteeing them and pinned "actionable fraction" at 50% by construction — the same
  defect as an eval that cannot move; replaced by a floor. Reusing the **`ambiguous`** key, which
  already means "ambiguity stub, no import line", made an existing filter delete a real callable
  method from the evidence entirely. And **two budgets I guessed** (700 chars for cells, 2,600
  for specs) both cut through the thing being retrieved; both are now set from the corpus.

  The grader was wrong twice, and the second time is the one to remember. It reported **4/4**
  on an answer that read "the retrieved evidence does not state the actual numeric weight values
  (e.g., 1.0 / 0.68 / 0.22 or similar)" — the agent wrote those digits as an example of what it
  did NOT have. A substring match cannot tell assertion from negation, and the false pass came
  out in exactly the direction the experiment was hoping for. Earlier the same grader could not
  see two-digit numbers at all, so every check on p5 was ungradeable by construction and a
  correct "10, 20, and 30 minutes" was scored MISS.

  Also worth naming, because it is now three-for-three in this codebase: **absence is never a
  message.** A missing callability verdict meant "not callable" and hid the affirmative fact
  from every consumer — all 840 callable units carry `callability = {}` while all 149 refused
  ones carry the full dict. A missing bbox meant "no geometry" rather than "unknown CRS". And a
  withheld import line meant "we don't have it" rather than "you have not said which one".

**Deliberately not done** The `IMPLEMENTED_BY` edge is still never written, so the pairing p5
  depends on — publication `355786a5` and notebook `3b45070e` — has to be rediscovered by search
  on every turn. That is the obvious next change, and it is an EXTRACTION change rather than a
  reference one: the corpus holds the relation and nothing records it.

  Nor is the deeper structural problem fixed. Between the record and the model sit a search, a
  normaliser, an evidence renderer and a type-specific sub-renderer, each written against the
  fields that existed at the time and each silently lossy for anything added later. Every one of
  the three findings above is an instance. The fix is not another patch at a fourth site; it is
  a single typed evidence contract that every layer must carry forward or explicitly drop.

## 2026-08-30 · M8.49 · IMPLEMENTED_BY produced zero edges, and the parenthetical is why

*Written 2026-10-01 from the commit message (7b4b668) and the run records.*

**Change** `publication_extractor.implemented_by_edges` considers the parenthetical in a tool name
  as well as the text before it. Three tests: an acronym links; generic and short names still do
  not, so the widening cannot start linking papers to half the library; and the edge still declares
  `confidence: low`, because asserting that a paper's method IS a function on the strength of a
  shared name would be a fabricated provenance claim.

**Why** It was said three times on this branch, including in this log (M8.42–M8.48), that the
  IMPLEMENTED_BY edge "is still never written". That was wrong: the function exists and has tests.
  What is true is worse, and shows only when it runs over the corpus. It wrote **zero** edges from
  the 61 specs that reference any tool at all. It normalised a tool name with `rsplit("(", 1)[0]`,
  which discards the parenthetical, and the parenthetical is where the symbol lives. A paper writes
  "Enhanced Two-Step Floating Catchment Area (E2SFCA) method" and the callable unit is `e2sfca`, so
  the one step meant to make a name matchable threw the match away.

**Measured** With the parenthetical considered: exactly **1** edge across all 73 specs, `355786a5`
  → `3b45070e.e2sfca`, and no false positive anywhere in the corpus. That is the pairing p5 depends
  on, which until now the agent rediscovered by search on every turn. 1,869 tests passed.

**Surprised by** One edge is the honest ceiling of this approach, and the cause is a category
  mismatch, not tuning. `tools_referenced` records what SOFTWARE a paper used ("OSMnx", "Python",
  "Jupyter Notebook", "Census Bureau API"), while the method library is indexed by FUNCTION. Papers
  name libraries, libraries contain functions, and the two vocabularies barely meet.

**Next** Link a spec to an ELEMENT, on evidence other than a shared identifier.

## 2026-08-30 · M8.50 · Spec-to-element links from citation and author evidence

*Written 2026-10-01 from the commit message (7c48331), `outputs/spec_links.json` and the source.*

**Change** `extractors/analysis/spec_links.py`, `scripts/build_spec_links.py`, and a `spec_link`
  table in the record. The module is pure: the author index and the topical scorer are injected, so
  the ranking is testable without a corpus, a network or a database. Two routes, each carrying its
  confidence and the evidence for it:

  | route | evidence | confidence |
  |---|---|---|
  | `cited_doi` | the element's own text names the paper's DOI | high |
  | `shared_author_and_topic` | a shared author AND the element's text scores against the spec | medium |

  A medium link is drawn only when the best candidate clears `TOPIC_MIN = 0.05` and the runner-up
  scores below `TOPIC_MARGIN_RATIO = 0.80` of it, and a spec gets at most one link
  (`MAX_LINKS_PER_SPEC = 1`). The link travels in the spec's `contents`, not only in a field
  beside it, because M8.48 established three times over that a fact the model has to go looking
  for is a fact it does not use.

**Why** Symbol matching draws one edge and can draw no more (M8.49), so the link has to be drawn at
  the element level, from evidence a person left behind. Neither half of the second route works
  alone. Author overlap by itself gave `355786a5` **seventeen** candidates, including
  "CyberGIS-Compute Core" and "Data Collection": a co-authorship graph, not an implementation link.
  Topical score by itself would link any two papers about accessibility.

**Measured** 14 links → **7** after the two corrections below: 2 high (both `cited_doi`) and 5
  medium. For `355786a5` the real implementation, `3b45070e`, scores 0.736 against 0.443 for the
  runner-up and 0.010 for the co-authored bystander. The two high-confidence pairs have
  near-identical publication and notebook titles, the strongest validation available without a
  human. 1,889 tests passed.

**Surprised by** Both corrections came from reading the first fourteen links, not the code.
  - The citation route was gated behind "the publication has extracted steps", and both DOI-citing
    pairs in the corpus point at publications whose step extraction came back empty. The strongest
    evidence available produced nothing. A person writing a DOI into a notebook is saying "this
    implements that" whether or not an LLM got a step list out of the PDF.
  - A relative floor with a cap of three was too loose. `a182e493` scored 0.7011, 0.6971 and 0.6254,
    three candidates within 11%. That is the shape of "several of this author's notebooks are about
    this topic", not of one notebook implementing the paper. A wrong IMPLEMENTED_BY edge is
    fabricated provenance rather than a gap, so the rule became a margin and a cap of one.

**Known wrong** One of the five medium links is a false positive, found on review: `863cb37f` →
  `05269a1a` links a trend-analysis paper to an inundation-mapping notebook through a shared author
  and overlapping topic vocabulary. Medium precision is therefore at most 4 of 5 on what has been
  checked. It cleared the rule comfortably, 0.221 against a runner-up of 0.013 out of two
  candidates. A margin shows the best candidate stands apart from the alternatives, not that it is
  right; with two candidates from one shared author, it stood apart from almost nothing.

## 2026-08-30 · M8.51 · A generated skill becomes a procedure, and step-level tracing to see whether it helps

*Written 2026-10-01 from the commit message (893aab4) and `outputs/agent_traces.json`.*

**Change** Two things, both prompted by asking how the extracted artifacts actually get used.
  - `scripts/trace_agent_run.py` and `scripts/report_agent_traces.py` record every tool call in a
    turn, with arguments, result and timing, per arm. The tracer wraps `BaseTool.run`, the one hook
    every tool passes through. The granular tools are module-level functions, `execute_code` is a
    closure inside a factory, and the staging tools are built per session, so wrapping the
    factories would have missed two of the three.
  - The skill emitter renders a procedure instead of a table of contents. `SkillSpec` carries its
    methods (symbol, signature, summary, requirements, invariants, pinned import line). Steps lead
    with the author's own cell titles (`_block_title`, M8.30). `allowed-tools` names
    `kb_method_search`, `get_method_contract` and `execute_code`. The Run section is four steps
    naming the tools, the contract check, and the staging precondition a `staged_path` signature
    implies.

**Why** Every behavioural claim of the previous few days ("it searched the library and wrote its
  own code anyway") had been inferred from call counts and answer text, not observed. And the
  generated skill, set beside the hand-written one for the same notebook, read "(cell 3) tools: —
  — # AI Agent for Chicago Crime Analysis This notebook walks you through building a 'Code Agent'
  capable of performing geo", with empty `allowed-tools` and a Run section saying only "reuse the
  functions extracted from this element". The hand-written skill named the functions and the
  failure to avoid. The difference was raw material, not effort, and the material now existed.

**Measured** First traces, on the p5 question (`e2sfca`): without the KB, 3 calls and no values in
  142 s; with it, 5 calls and a correct answer in 179 s. With the KB the agent led with
  `kb_method_search` before any platform search, and its three KB calls took 0.1, 0.1 and 0.0 s,
  against 1.4 s (keyword) and 1.7 s (spatial) for the ablated arm's platform searches. The
  regenerated A2SFCA skill lists nine callable methods with signatures, pinned imports, pip
  requirements and CRS invariants, and its steps read "Travel-Time Polygons", "Reproject", "E2SFCA
  Implementation" instead of 140 characters of truncated prose. 1,897 tests passed.

**Surprised by**
  - Import lines have to be resolved at EMIT time. `library_emitter` assigns `library_module` by
    mutating the live unit dict, but `manifest.skill` is an `asdict` copy taken when the extractor
    returned, so a module path written afterwards never reached the skill. The emitter now reads
    the manifest's assets, which `_fan_out` already builds library-first for the same reason.
  - Invariants are recorded per parameter, so a three-argument method rendered "projected_crs,
    reject_all_nan" three times over. The reader needs the rules, not their multiplicity.

**Next** Both A/B harnesses set `AGENT_SKILLS_ENABLED=0`, so every measurement so far ran with the
  skill path switched off entirely. Turning it on is the experiment.

## 2026-08-30 · M8.52 · The generated skill was invisible to the registry that loads it

*Written 2026-10-01 from the commit message (f9c2082) and `outputs/agent_traces.json`.*

**Change** The skill emitter writes front matter with `yaml.safe_dump(..., width=10**6)`, so no
  scalar folds. `skill_emitter.emit` returns an `error` when the written skill is not discoverable,
  and ingest reports it as a failure instead of a footnote. Three tests: a long description
  round-trips through the REAL parser, an undiscoverable skill reports an error, a discoverable one
  does not. The tracer gains a `with_kb_skills` arm: the `with_kb` tools plus
  `AGENT_SKILLS_ENABLED=1` and the generated skill root.

**Why** `yaml.safe_dump` folds a long scalar across lines, and `agent_runtime.skills.parse_frontmatter`
  is, by its own docstring, a "simple YAML-style" line parser. The continuation line carries no
  `key:`, so the parser rejects the entire skill. The flood skill was generated, written to disk,
  and absent from the registry. Only long descriptions fold, so the one short-description skill
  already on disk loaded fine and hid the fault.

  The flood trace was the reason to look. With the KB, calls 15–26 (12 of its 33) went on reading
  the library's source and searching `/opt /data /mnt /home /tmp` for a dataset nothing had staged.
  That is exactly what the skill's Run section says to do instead: "If a signature takes
  `staged_path`, call `stage_element(...)` first; the file is not in the sandbox until you do."

**Measured** Flood: notebook `05269a1a` and its dataset `f49f395e`, against ground truth of 24 test
  dates, 314 rasters and EPSG:4326 read from the archive beforehand. One run per arm; all three
  answers correct.

  | arm | calls | time | `stage_element` at call |
  |---|---|---|---|
  | no KB | 21 | 635 s | 18 |
  | with KB | 33 | 638 s | 29 |
  | KB + skill | **11** | **331 s** | **6** |

  The skill arm imported `read_dates_list` from `iguide_methods` instead of reading its source: the
  only run that reused a function extracted from a notebook. (p3's KB run had already imported and
  called a GENERATED dataset loader, `load_chicago_crime_data_2026`, and answered correctly.) Peer
  routing shortened from search → analysis → search → analysis → code, to search → analysis → code,
  to search → code. Both arms without the skill opened with a long "cannot access the data" phase
  and then reported real numbers, so the grounding audit marked their correct answers as
  unverified; the skill arm's record audited clean. 1,900 tests passed.

**Surprised by** `_roundtrip_ok` was the right check, and caught this: it returned
  `discoverable: False`, which ingest then recorded inside a warning announcing a successful write.
  A check whose result nothing acts on is not a check. Separately, in all three arms
  `import rasterio` from the session's `/work/.deps` failed with `libexpat.so.1`, costing each arm a
  call; the skill arm took the CRS from a host-side QGIS call instead. Prototype fixed the image in
  `360a53dd`.

**Limits** One run per arm, on one case. The skill was generated for the element the question
  names, so finding the right skill is untested. Skills for the rest of the corpus have not been
  regenerated with the fixed emitter.

## 2026-10-01 · M8.53 · CI ran for the first time, and the lock had never installed on the images' Python

**Change** `constraints.txt` pins rasterio by interpreter: `1.5.0` on Python ≥3.12, `1.4.4` on
  3.11. The `deployment-contract` job now installs `python-dotenv`, and `networkx` through the
  lock, beside pytest and pyyaml.

**Why** `backend_swap` was pushed for the first time today, so `.github/workflows/verify.yml`
  ran for the first time, seven weeks after M7.1 wrote it. All three jobs failed, for two causes.

  rasterio 1.5.0 declares `requires_python >=3.12` and ships wheels for cp312+ only. The lock was
  frozen on this machine's CPython 3.13, while CI and all three Python images (agent-api, MCP,
  sandbox) use `python:3.11-slim`. So `pip install -r requirements.txt -c constraints.txt`, the
  exact line `rag_pipeline/Dockerfile` and `MCP_server/Dockerfile` run, could not succeed on any
  3.11 build from the day M0.5 introduced the lock (2026-08-12). Nobody built either image from
  this branch in that time, and the CI that would have said so had never run.

  The contract job installed only pytest and pyyaml. pytest imports
  `rag_pipeline/tests/conftest.py` as part of the `rag_pipeline` package, whose `__init__`
  imports dotenv at module scope, so the job died at collection before a single assertion. Behind
  that sat three contract tests that need networkx, one asserting the INSTALLED version equals
  the pin, which is why it goes through `-c constraints.txt` rather than loose.

**Measured** All 43 unconditional pins checked against PyPI metadata for CPython 3.11: rasterio
  is the only one that excludes it. A full resolve for 3.11 on manylinux (uv, with pip's
  merge-all-indexes behaviour): 171 packages, every pin held, rasterio 1.4.4. The same resolve
  for 3.13 still selects 1.5.0. `test_deployment_contract.py` in a bare venv holding only the
  job's packages: collection error → 3 failures (networkx) → **22/22**. The gate's CI step, run
  verbatim: degrees buffer → `fail`, reprojected → `pass`.

**Surprised by** The sandbox image built, and passed its offline geospatial import check, on the
  first CI run. `sandbox/Dockerfile` does not read `constraints.txt`, so the one image that ignores
  the lock was the only one that could build. The lock's own header says "both Dockerfiles"
  use it; that sentence was true and was the problem. A check that never runs is the limiting case
  of a check that reports success while the thing it establishes is false.

**Not yet written** Devlog entries for M8.49–M8.52 (IMPLEMENTED_BY, spec links, skills as
  procedures, the registry that could not read a generated skill). Their commit messages carry the
  reasoning; the entries are owed. *(Written later the same day and inserted above, in sequence.)*

**Next** The first green run, and whatever the full suite does on Linux / 3.11, which nothing has
  ever exercised.

## 2026-10-01 · M8.54 · The first full CI run executed zero tests

**Change** The `tests` job runs `python -m pytest -q --ignore=tests/live` instead of `pytest -q`.

**Why** With the lock fixed (M8.53) the suite was reached for the first time, and stopped at
  collection: `MCP_server/test_dual_transport.py` does `from MCP_server import server`, which
  resolves only when the repo root is on `sys.path`. `python -m pytest` puts it there; bare
  `pytest` does not. One collection error interrupts the session, so the run reported 1 error
  and **0 tests executed**. Every local measurement in this log used `python3 -m pytest`, which
  is why the difference never showed.

  A root-level run also collects `tests/live/`: 32 tests that call live services and carry no
  `integration` marker, so `pytest.ini`'s deselection does not reach them. Excluded here; marking
  them `integration` is the proper fix and belongs with whoever owns them.

**Measured** The exact CI command, locally: **1,947 passed**, 29 deselected, 62 s. That is the
  documented `rag_pipeline/tests/` suite (1,900) plus 47 tests in `MCP_server/` and `tests/` that
  were never in the documented command and all pass.

## 2026-10-01 · M8.55 · What a clean Linux build has that this machine does not, and the reverse

**Change** `pypdf`, `python-docx` and `xarray` declared in `requirements.txt` and pinned in
  `constraints.txt` at the dev machine's versions. The default distance band in
  `analysis_spatial_stats_tools.py` is padded by a relative 1e-9. Two trace-payload tests skip
  without a built library, and a new one pins the empty-library note on its own.

**Why** With collection fixed (M8.54) the suite ran on Linux / 3.11 for the first time: 1,925
  passed, 4 failed. All four were the gap between this machine and a clean install.

  - **Undeclared readers.** `pypdf` was never in `requirements.txt`; here it comes from
    `~/.local`. In a clean build `publication_extractor` reports "pypdf is not installed, so no
    PDF can be read" for every PDF. `python-docx` (DOCX publications) and `xarray`
    (NetCDF/HDF/GRIB datasets) had the same gap and no test to expose it. Every publication and
    dataset coverage number in this log was measured with all three present.
  - **A threshold with zero margin.** The spatial-weights tool promises that its default
    distance band leaves no island. `min_threshold_distance` returns the critical neighbour
    distance exactly, and `DistanceBand` recomputes it: on the test lattice the margin is
    `0.0` m. macOS/arm64 gives 0 islands and Linux/x86-64 gives 1, and reproducing with CI's exact
    libpysal 4.14.1 / scipy 1.17.1 here still gives 0, so the flip is the platform's rounding, not
    a library version. The deployed image is Linux/x86-64. This is prototype's tool; the fix is
    one line and should go there too.
  - **Tests that assumed local data.** Two tests needed the method library built under
    `agent_chat_files/`, which is not committed. One of them was meant to pin the difference
    between "nothing ingested" and "nothing matched" but only ever exercised the second. CI
    reached the first and failed for asserting the wrong note.

**Measured** 3.11 / manylinux resolve with the readers: every pin held (pypdf 6.6.2,
  python-docx 1.2.0, xarray 2026.7.0, rasterio 1.4.4). The trace-payload tests in a copy of the
  tree without the library: 6 passed, 4 skipped with reasons. The full CI command locally:
  **1,948 passed**.

**Surprised by** Three of the four failures were invisible here for the same reason: this
  machine has more than the project declares, whether that's packages in the user site, a library
  built from the corpus, or a CPU that rounds the tie the convenient way. The lock was written to
  stop that drift, and it could only cover what was declared.

**Not done** libpysal, esda, spreg and pygeoda are in `requirements.txt` but not in the lock, so
  CI resolves libpysal 4.14.1 / esda 2.9.0 where dev runs 4.15.0 / 2.10.0. Pinning them is a
  measured step by this file's own rule, not a CI fix.

## 2026-10-01 · M8.56 · Prototype merged into the extraction work, and what each conflict became

**Change** `claude/extraction-integration` = `backend_swap` (628d3b0) + `origin/prototype` (5ae6d92).
  Ten files conflicted (38 hunks). Every resolution below is a decision the agent designer and the
  maintainer should see before this reaches a PR.

  | area | resolution |
  |---|---|
  | `code_execution.py` | Prototype's file is the base: durable per-conversation workspace by copy-in/copy-out, read-only `.deps` cache, preinstalled probe, `entrypoint` runs (D3). Ported onto it: the read-only library mount, AHEAD of `/work/.deps` on `PYTHONPATH`; the invariant gate around inline code with `checks.json` never carried between runs or persisted as an output; `contracts_for_code`; run-artifact emission. **Dropped:** `backend_swap`'s execution tiers, deps marker and run-in-place session dir. |
  | staging | Writes into prototype's durable workspace (`session_workspace_dir`), keyed `"codeexec"` like both peers' `execute_code` — prototype's key, which deployed conversations already use. |
  | auth (`api/server.py`) | Prototype's model wins: verified user, service key or demo mode as alternatives; an unset key leaves the check open outside token mode. `/query` and `/query/batch` keep `backend_swap`'s decorator, now running prototype's checks. The file routes lose it: download is not key-gated by design (map sources cannot send a header). **Dropped:** fail-closed-when-unset. |
  | supervisor | Peer containment is `backend_swap`'s `_run_peer` (fatal detection, error budget, decider fallback) returning prototype's degraded results; prototype's re-grounding edge plus `backend_swap`'s checkpointer, now opt-in at build and supplied by `run_supervisor`; the code peer's KB tools are the union with web search; the partial-answer disclosure is kept. |
  | registry | Staging is described to the supervisor, and the retrieval clause names the method library (S2). |

**Why** The extraction work existed only on `backend_swap`, and nothing on prototype could reach it.
  Prototype's designs won wherever both branches had built the same thing, because prototype's
  are the deployed, reviewed ones. Extraction's pieces were ported onto them rather than kept
  beside them.

**Measured** Merged suite, local: **2,990 passed, 0 failed**. On the deployed stack (CPython 3.11,
  pandas 3.0.5, the agent-api freeze): 2 failures, both known — the temporal tool's `Beat`
  misdetection (prototype's, queued for the maintainer) and the networkx pin (next commit).

**Surprised by** The previous merge (2026-08-27) silently dropped `_artifacts.emit(...)` from
  `execute()`: no run has produced its reproducible record since, the comment describing the call
  stayed behind, and CI's replay step found nothing to replay. Restored here. The same merge kept
  the prototype tests it could not satisfy by retargeting them onto its own design; their
  originals are restored. And `method_library_dir()` returned any existing directory, empty or
  not, despite its docstring; it now requires the package.

**Not done** S3 (method tools as citable evidence), S4 (tier routing for extraction's OpenSearch
  reads and staging's backend URL), S5 (null-tolerant wrappers on the new tools), one extraction
  flag defaulting the bundle OFF, and the lock regenerated from the deployed freeze.

## 2026-10-01 · M8.57 · The extraction tools accept an explicit null (S5)

**Change** `stage_element`, `stage_url`, `list_staged_inputs`, `kb_method_search` and
  `get_method_contract` registered through prototype's `accept_null_defaults`. The null-argument
  scan in `test_null_tool_args.py` now covers the retrieval and staging FACTORIES.

**Why** Prototype found 141 optional parameters that rejected an explicit `null` — a model that
  fills every slot could not call them — and wrapped every tool it owned. The tools extraction
  adds were registered without the wrapper, which a merge cannot show because nothing conflicts.
  Listing the factories rather than five tool names keeps the next tool added to either inside the
  scan.

**Measured** The scan covers 54 tools, including all seven extraction tools; 9/9 pass.

## 2026-10-01 · M8.58 · A method the code peer found is evidence the answer can cite (S3)

**Change** `capability_registry.KB_CODE_PEER_TOOLS` is the one definition of the code peers' KB
  tools; the peer binding and `_RETRIEVAL_TOOLS` both read it. Method-tool results are rendered as
  evidence documents through `_method_hit_as_document`, the per-hit renderer lifted out of
  `_method_units_as_documents` so the sweep and the harvester share it.

**Why** The integration found those names hardcoded in three places that had drifted. The code
  peer's binding dropped the method tools (S1, fixed at the merge), and the evidence allowlist never
  had them, so a method a peer found could shape the code but never be cited. Adding the names
  alone would have been worse: method rows carry `symbol` and `import_line`, not `title` and
  `contents`, so the name-based harvester would have handed the answerer untitled, empty documents.

**Measured** 6 new tests: one definition; every KB tool counts as evidence; a search result and a
  contract each become a titled document carrying the pinned import line and citing the SOURCE
  element; a refused lookup is not cited; a geocoder's `results` still are not evidence. Full
  suite: **2,996 passed**.

## 2026-10-01 · M8.59 · Two things that would have broken the first deploy from this branch

**Change** `agent-db` sits behind `profiles: ["extraction"]` and its password is no longer a compose
  `:?` required variable. The agent-api image copies BOTH skill roots. Two contract tests guard the
  compose file, and the skills test requires every root instead of any one.

**Why** Both came from the agent designer's review of the merge.
  - The `:?` on `AGENT_DB_PASSWORD` failed `docker compose up` for every service on a host whose
    `.env` lacked it, for a feature that is meant to be off. The review proposed a profile. I
    checked it before relying on it, and it is not enough: compose v5.5.1 interpolates the WHOLE
    file before applying profiles, so the error stands with the service profiled out. The password
    stays mandatory where it matters, because the postgres image refuses an empty superuser password.
  - The image copied `.agents/` but not `skills/`, which is the first entry in `DEFAULT_SKILL_ROOTS`
    and holds two of the three curated skills. In the running agent-api neither directory existed
    and `SkillRegistry.discover()` returned 0. The contract test passed throughout, because it
    checked `any(...)` root.

**Measured** `docker compose config` with the password unset: before, an interpolation error for
  the whole file; after, three services by default and four under `--profile extraction`.
  Deployment contract: 24/24.

## 2026-10-01 · M8.60 · The lock pins what production runs, not what one laptop had

**Change** `constraints.txt` regenerated from `pip freeze` inside the running agent-api container:
  48 curated pins that keep their reasons, plus 134 generated pins for the rest of the image, so
  the lock reproduces it. Packages the image does not have yet (the extraction bundle's Postgres
  client and readers, plus pyarrow) are listed separately. `torch` is split by platform, because
  the image's `2.14.0+cpu` build has no macOS wheel.

**Why** The agent designer's review flagged it as the most consequential blocker. The agent-api
  `Dockerfile` installs `-c constraints.txt`, and the lock pinned a dev Mac's stack, so an image
  built from this branch would have downgraded production in 34 places. Among them: pandas
  3.0.5→2.2.3 (a major version), numpy 2.4.6→2.1.3, langchain 1.4.1→1.2.10, openai 3.14.1→2.45.0
  and langsmith 0.12.6→0.6.7. The deployed stack is the one the benchmark measured, so the lock
  adapts to it.

**Measured** Resolves to 182 packages for CPython 3.11 on manylinux, and for 3.13 on macOS 14+.
  Suite on the locked stack: **2,996 passed, 1 failed**. The failure is prototype's temporal `Beat`
  bug (queued for the maintainer). The networkx contract test now passes on 3.6.1.

**Surprised by** pyarrow is NOT in the deployed agent-api image: prototype never declared it, and
  only the sandbox image has it (25.0.1). Prototype's `reproject_vector` and `vector_spatial_join`
  write GeoParquet agent-side (`langchain_geo_tools.py:570`, `:620`), and `geo_handles.py` passes
  frames as parquet, so both tools should fail in production with "Missing optional dependency
  'pyarrow'". This is the defect M0.5 fixed on `backend_swap` and that never reached prototype.
  The test venv had pyarrow through `backend_swap`'s requirements, so the suite cannot show it; it
  needs a check inside the container. This machine still runs the old stack (networkx 3.4.2), so
  the pin test fails here until the local environment follows the lock.

## 2026-10-01 · M8.61 · One switch for the extraction bundle, off by default

**Change** `AGENT_EXTRACTION` (`agent_runtime/extraction_flag.py`) now gates everything the
  extraction work adds to the agent, and it is off unless set. Off, the agent has:
  - no `kb_method_search` / `get_method_contract` in the granular factory, and no staging tools
    (that factory returns none);
  - no library mount and no `PYTHONPATH` entry for it, in either executor;
  - no invariant gate and no artifact emission by default. An explicit `AGENT_INVARIANT_GATE` or
    `AGENT_ARTIFACT_EMIT` still wins, in both directions;
  - no method-library arm in the deterministic sweep;
  - no VERIFICATION / METHOD LIBRARY paragraphs in `execute_code`'s description, and no search
    rule 12 or code rule 1b. `search_agent_prompt()` and `code_agent_prompt()` add them when on;
  - no method-library clause and no staging toolset in the capability registry, so the
    supervisor is not told about either.
  `DEPLOYMENT.md` and `.env.example` document the switch. The suite runs with it on (conftest).
  `test_extraction_flag.py` pins the off state surface by surface. Its tool and mount tests also
  check the on half, so they cannot pass merely because nothing is built.

**Why** Blocker 2 of the agent designer's review: merging the extraction into prototype must
  change nothing in production until someone opts in. "Nothing" has two halves, and the second is
  the one that has already bitten. The tools must be absent, AND nothing may describe them. In
  M2.7 the code peer's prompt told it to call `kb_method_search`, which it did not have. It
  guessed the package from the host directory, wrote `from method_library import ...`, and got
  `ModuleNotFoundError`. So each description is gated with the tool it describes. The gate and
  artifact defaults change meaning with it: on `backend_swap` both defaulted on, and here they
  follow the bundle.

**Measured** `test_extraction_flag.py` 10/10, `test_method_library_tools.py` 49/49. CI-style suite
  on the deployed-stack venv: **3,006 passed, 1 failed, 1 skipped**, identical to before the flag.
  The failure is still prototype's temporal `Beat` bug.

## 2026-10-01 · M8.62 · The flag also covers what the evidence says

**Change** Three more surfaces follow `AGENT_EXTRACTION`:
  - `agent_kb._method_payload` keeps a unit's signature, summary and requirements, but drops
    `import_line`, `slice_sha` and the callability verdict. An `agent_kb_search` hit names a
    method without saying how to import it;
  - the sweep's per-turn agent-KB join (`attach_kb_to_documents`) does not run;
  - `_render_extracted` lists every unit as a reference: no RUNNABLE block, no `stage_element`
    instruction, no pointer at `kb_method_search`.
  The flag's docstring, `DEPLOYMENT.md` and `.env.example` now say what off actually means.

**Why** M8.61 gated what the tools and prompts say and missed what the EVIDENCE says. All three
  surfaces came with the merge. Prototype's `agent_kb.py` (297 lines) had no method payload and no
  join, and its evidence view had no `_render_extracted`. With the flag off, an `agent_kb_search`
  hit still carried `from iguide_methods... import ...`, and the evidence view still said the
  library was mounted read-only. That is M2.7 again: a model handed an import line writes the
  import. Found by grepping for the bundle's tool and package names outside the gated files,
  before sending the peer the coverage list.

  Also corrected: M8.61's docs said that off, the agent "behaves as prototype did before the
  integration". That was never true. The merge also brought retrieval and failure-handling
  changes that are not extraction capabilities and are not behind the switch. Off means none of
  the bundle is exposed, and nothing the model reads describes it.

**Measured** `test_extraction_flag.py` 13/13. Each new test checks its on half too, and the join
  test records calls instead of raising, because the call site swallows exceptions. CI-style
  suite: **3,009 passed, 1 failed** (prototype's `Beat`).

## 2026-10-01 · M8.63 · The library the agent lists is the library the sandbox imports

**Change**
  - One resolver for the library's location, `code_execution.method_library_root()`. The reader
    and both writers (`library_emitter.library_root()`, `build_method_library.py`) now use it.
  - `docker-compose.extraction.yml`, the opt-in override. It sets `AGENT_EXTRACTION=1` and
    bind-mounts `AGENT_METHOD_LIBRARY_DIR` at the IDENTICAL path, and it requires that path
    (`:?`). `COMPOSE_FILE` in `.env` is the switch; the main compose file is unchanged.
  - A failed `import iguide_methods` now names its cause: bundle off, nothing built, or built
    but invisible to the sandbox. The last one logs the host-path rule once for the operator and
    tells the model to write the function inline. The note goes FIRST in stderr, because `_clip`
    keeps the head.
  - The image copies `build_method_library.py` and `smoke_end_to_end.py`, and the runbook runs
    both inside agent-api. The smoke script's executor call is fixed: the merge had left it
    passing `session_id=`/`tier=`, so it raised TypeError. It now reports the flag, forces the
    bundle on for its own process (a pre-flight), and checks the library path is host-visible.
  - `DEPLOYMENT.md`: a new step 0 to turn the bundle on, the build and smoke steps moved into
    agent-api, a rewritten `AGENT_METHOD_LIBRARY_DIR` row, and a note that §6's gate rows need
    the bundle on.

**Why** Turning the bundle on in the deployed shape would have mounted an EMPTY library.
  agent-api starts sandboxes through the host's Docker daemon, which resolves `-v` sources on the
  host. The library's default location is on the `agent_chat_files` named volume, and the host
  has no such path: the agent designer checked the VM read-only, and `/app` does not exist there.
  The daemon creates the missing directory and mounts it. `kb_method_search` keeps listing
  methods with import lines, and every import fails.

  Two more defects blocked the fix:
  - Setting `AGENT_METHOD_LIBRARY_DIR` did nothing for the writers, which hardcoded the default.
  - The one check that would have caught it, the smoke script's sandbox import, was broken by the
    merge. Its runbook step ran `docker run` from the HOST, which passes while every turn fails.

  Rejected: binding the named volume's own `_data/method_library`. It works today, but it couples
  the deployment to Docker's internal volume layout (data-root, project-name prefix).

**Measured**
  - `docker compose config`: the main file alone renders. With the override and no path it
    fails, naming the variable. With the path set, agent-api has the library bind (source ==
    target) beside the work-root bind.
  - New tests: 8 in `test_method_library_location.py`, 5 in `test_deployment_contract.py`. The
    two compose tests drive compose's real merge. CI-style suite: **3,022 passed, 1 failed**
    (prototype's `Beat`).
  - Not reproduced live: Docker Desktop is not running here. The empty mount is inferred from
    bind semantics plus the VM check.

**Surprised by** `test_the_advertised_module_is_a_file_that_exists` failed after the resolver
  fix, because it had only passed BECAUSE of the split. Its fixture set the storage root alone;
  conftest points `AGENT_METHOD_LIBRARY_DIR` at an empty directory; the writers ignored that, so
  the library landed where the test looked. The fixture now names the library path.

## 2026-10-01 · M8.64 · Evidence describes only what its reader can act on

**Change** `capability_registry.EVIDENCE_CONSUMERS` declares what each reader of the evidence can
  do with an extracted method. There are three capabilities:
  - run the library (`execute_code` with the mount);
  - stage inputs (`stage_element` / `stage_url`);
  - search methods (`kb_method_search` / `get_method_contract`).

  The rows:
  - `answer` keeps every capability, which is the view as it was;
  - `code_peer` gets all three;
  - `analyze_peer` gets library and search, but not staging;
  - `cli_peer` gets none.

  `consumer_capabilities()` returns nothing while the bundle is off. `_render_extracted`,
  `_doc_block` and `_format_documents` render for a named consumer:
  - an import line only for one that can run the library;
  - a staging instruction only for one that can stage;
  - a `kb_method_search` pointer only for one that holds it.

  A `MethodUnit` document loses its `import:` line for a reader that cannot run the library. The
  analyse and code peers now name their rows. The CLI brief (`opencode_peer._build_peer_prompt`)
  names `cli_peer`, and when the analysis results it carries import `iguide_methods`, it says that
  package is not available in the container.

**Why** With the bundle on, every reader got one rendering: "the import line works verbatim inside
  execute_code (the library is mounted read-only)", and for a loader "FIRST call stage_element".
  Only the LangChain code peer has all of that.
  - The analyse peer binds no staging tools.
  - The CLI code peers have no `execute_code`, no staging and no mount, by design. The agent
    designer's review (D1) keeps submitter-authored library code out of a container that holds
    network access and the model credential. So the peer users reach by opting into Claude would
    have been handed imports that fail there.

  The rows are declared rather than read from a peer's name, as the designer asked, so a consumer
  that gains the mount gets the runnable view by editing its row. A test holds each peer's row to
  the factories its builder calls.

  There is a second route for the CLI peer, found while answering the designer's "is the evidence
  path the only one?": `analysis_results` carries the analyse peer's own tool calls verbatim. No
  skills are seeded into a CLI peer's `/work`.

**Measured** `test_evidence_consumers.py` 11/11, and the flag, join and method-evidence tests
  still pass (83 in all). The answerer's row keeps every capability, so its text is unchanged by
  construction. A test pins it equal to the code peer's full view, because which cut the
  answerer reads is B3's to settle. CI-style suite: **3,033 passed, 1 failed** (prototype's
  `Beat`).

## 2026-10-01 · M8.65 · The agent KB reads the cluster the platform's search reads (S4)

**Change**
  - `platform_endpoints.search_cluster()` returns `(node, user, password)` for the SEARCH tier's
    cluster.
    - The node resolves in platform search's own order: `OPENSEARCH_NODE_<SEARCH_TIER>`, then
      `OPENSEARCH_NODE`, then the tier's table entry.
    - The credential FOLLOWS THE NODE. A tiered node takes the tiered pair or none. A bare node
      takes `opensearch_credentials`' rule, now `_credentials_for(tier, node)`.
  - The agent KB (`_os_client` and its "is a cluster configured?" checks) and the extraction's
    emitter both use it, so an ingest lands where the agent reads.
  - The emitter's delete guard now protects the platform index under its tiered name
    (`search_index()`) as well as the bare one.
  - `platform_endpoints.backend_url()` follows `SEARCH_TIER`, and `IGUIDE_BACKEND_URL[_<TIER>]`
    still wins. Staging fetches element records from it. With nothing set anywhere it is prod's
    backend, as before.
  - A drift guard fails on any module in `search/` or `extractors/emitters/` that reads the bare
    cluster names. Its first catch was prototype's `search/agents.py`, which resolved its node tiered
    but its credential bare, sending the untiered pair to the tier's cluster. Fixed to match
    `keyword.py`.

**Why** The agent KB and its emitter read raw environment names while keyword and semantic search
  followed the tier. The developer `.env` this checkout runs against defines all three forms of the
  node (bare, `_DEV`, `_PROD`). So under `SEARCH_TIER=prod`, platform search used
  `OPENSEARCH_NODE_PROD` while the agent KB used the bare `OPENSEARCH_NODE`: possibly another
  cluster, with another credential. A refused agent-KB query reads as "no results". Staging also
  resolved element ids against prod's backend whatever tier the ids came from. The agent designer
  corrected the urgency: this is LATENT, not live, because the deployed agent KB backend is `local`.

  Two designs were weighed against each other:
  - The agent designer proposed `opensearch_url()` / `opensearch_credentials()`, as `memory_module`
    uses. That keeps the credential-follows-host rule. But those follow PLATFORM_TIER and return the
    BARE node, so the agent KB would follow the conversation store rather than the search its
    results are joined to.
  - Raw per-variable tiering (`search/utils.getenv`) follows the search, but falls back to the
    untiered pair beside a tiered node. That pair measured 401 against dev's cluster on 09-22.

  `search_cluster` keeps the node of the second and the credential rule of the first.

**Measured**
  - `test_tier_routing.py`: 15 tests.
    - The agent KB's client equals `keyword.py`'s, and the emitter's equals the agent KB's, under
      four configurations, the deployed shape among them.
    - Beside a tiered node holding only the bare pair, the agent KB sends no credential at all,
      where `keyword.py` sends the wrong one; that difference is left for the maintainer.
    - Also covered: the tiered-node-alone check, the emitter guard, the drift guard, and staging's
      resolver and request.
  - Also added: `test_a_failed_library_import_alone_also_brings_the_warning`, the agent designer's
    check on M8.64's second route, where the package name appears only in a traceback in
    tool_results. It passes, because the check scans the whole analysis text.
  - CI-style suite: **3,049 passed, 1 failed** (prototype's `Beat`).

**Surprised by** My first version failed four tests in the full run, and passed them alone. The
  anchored edit began at `def _os_client():`, so the `@lru_cache(maxsize=1)` above it moved onto the
  new helper. It cached a SETTING: one read with the variable unset pinned `""` for the next test,
  and the client lost its cache. Order-dependent, and invisible in a targeted run.

## 2026-10-01 · M8.66 · A server that writes no conversation anywhere

**Change** `AGENT_PERSISTENT_MEMORY=0` keeps every conversation in-process. It overrides the
  request's `usePersistentMemory` in `_normalize_agent_chat_request`, and
  `PUT /agent/conversations/<id>` refuses with 409 `persistence_disabled` rather than dropping the
  save silently. Session memory still carries context within a conversation. Unset honours the
  request, exactly as before. Documented in `.env.example`.

**Why** Verifying the extraction flag through the map UI means running this branch locally. The
  only persistence switch was the request flag, and the map UI always sends `true`
  (`agentClient.ts:380`). The developer `.env` the dev launcher loads names the PRODUCTION cluster,
  so every test turn would have written a conversation and a turn trace into production's
  `chat_memory` / `chat_traces`. The user chose a no-persist switch over writing to dev or prod.
  Two request paths reach the store: the chat turn (create, update and trace, all gated by the
  request flag) and the conversation PUT. The other writers (`delete_memory`, `update_rating`)
  have no route.

**Measured** `test_persistent_memory_switch.py`: 7 tests.
  - Unset honours the request.
  - Four spellings of off each beat a client that asks for persistence, in both casings of the
    field.
  - The PUT refuses with persistence off and still saves with it unset.
  - CI-style suite: **3,056 passed, 1 failed** (prototype's `Beat`).

## 2026-10-01 · M8.67 · A fetched KB doc follows the flag too

**Change** `get_kb_block` passes every stored doc it returns through `_reference_only`. With the
  extraction bundle off, a unit doc's `extracted.unit` loses `import_line`,
  `import_line_candidates`, `library_module`, `slice_sha` and `callability`, and keeps what the
  unit IS. This is the same rule as `_method_payload` (M8.62). The bundle-on path is unchanged.

**Why** Found while driving the map UI with the bundle off on qwen3.8. M8.62 gated what
  `agent_kb_search` HITS say, but `get_kb_block`, prototype's own reader, returns the RAW stored
  document. For a doc from `iguide_agent_method_units` that includes the unit's import line,
  whatever the flag says. It is not live today: a read-only count of prod's
  `iguide_agent_method_units` found 0 docs with `extracted.unit.import_line`. But the first ingest
  with this branch's emitter writes them, and a model handed an import line writes the import.

**Measured** `test_off_a_fetched_kb_doc_carries_no_import_line` checks both halves, off and on. CI-style suite: **3,057 passed, 1 failed** (prototype's `Beat`).

## 2026-10-01 · M8.68 · The gate judges the operation, not the frames left lying around

**Change**
  - The prologue now wraps `warnings.warn` on every gated run (`install_operation_tracker`; it
    imports nothing). It records each metric operation geopandas runs on a GEOGRAPHIC receiver:
    `area`, `length`, `buffer` (not `buffer(0)`), `distance`, `dwithin`, `hausdorff_distance`,
    `frechet_distance`, `interpolate`. geopandas itself warns on exactly these, and goes quiet once
    the frame is reprojected. Each record carries the line that ran it.
  - With the tracker live, `run_checks` FAILs `projected_crs` once per such operation, naming the
    line. Which frames happen to sit in EPSG:4326 at the end no longer decides the verdict:
    - a geographic frame that nothing measured is an input, and passes;
    - one holding a metric column that no tracked operation produced becomes a named
      cannot_determine, not a ⛔.
    Without the tracker (run_checks called on its own) the frame rules are unchanged.
  - Agent side, `_note_printed_outputs`: if stdout shows `IGUIDE_OUTPUTS` but nothing was assigned,
    the run gets a named cannot_determine ("printed, not assigned").
  - `environment.json` and `declared_outputs.json` are no longer offered as downloads. They stay
    in the workspace and the run record, like `checks.json`.
  - `execute_code`'s description now says to ASSIGN `IGUIDE_OUTPUTS`, and that printing it checks
    nothing.

**Why** Driving the map UI with the bundle on:
  - The agent imported the library's `calculate_buffers`, reprojected Champaign to 26916, and got
    165.04 km² (QGIS gave 164.99). The answer was stamped "⛔ A deterministic invariant check
    FAILED".
  - The untouched 4326 input was still bound, and the measurement was a SCALAR, so the
    metric-column rescue (added after the same false alarm earlier) never fired.
  - A failed gate says "fix and re-run", so the agent ran the correct code three times. That gave
    46 steps, three download sets and four map layers.

  My first fix was to record `to_crs` lineage and forgive frames that had been reprojected. The
  agent designer pointed out the false negative that keeps: reproject, then measure the ORIGINAL
  anyway. That is the mistake a model actually makes. Their proposal: projected-before-measuring is
  a property of the CALL. geopandas already emits the signal, as they verified in the deployed
  image: one warning each for `.area`, `.length` and `.buffer` on 4326, none after `to_crs`. The
  lineage draft was removed before commit.

  `warn` is wrapped rather than `showwarning` hooked, because agents routinely start with
  `filterwarnings("ignore")`, and that would blind a `showwarning` hook.

  Documented blind spots: shapely-level calls (`shapely.area(geoms)`) bypass geopandas, and
  `pyproj.Geod` measures geodesically in 4326, correctly, without warning.

**Measured** `test_invariant_gate.py` 108/108; CI-style suite **3,067 passed, 1 failed**
  (prototype's `Beat`).
  - The UI script passes, both in-process and through the assembled prologue and epilogue.
  - Measuring the original after reprojecting FAILs and names `boundary.area.sum()`.
  - A degree buffer FAILs even with warnings silenced.
  - Neither a centroid nor `buffer(0)` counts as a measurement.

  The exit-path test changed its expectation: a degree buffer inside `main()` used to be
  invisible (`cannot_determine` with a coverage note), and the tracker now sees it, so the verdict
  is FAIL. Its `sys.exit(main())` case had only "passed" because the body never imported `sys`,
  so `main()` never ran.

**Surprised by** An idempotent installer bound its record to the first namespace. A second gate
  in the same process found `warn` already wrapped, returned early, and recorded into a run that
  had ended. That is harmless in the sandbox (one gate per process), but in one test process it left
  the second run's record empty: two degree measurements went unrecorded and their tests failed.
  The wrapper now looks up its record at call time, and conftest restores `warn` around every test.

## 2026-10-01 · M8.69 · The answer offers what the agent can run; it does not hand out imports

**Change**
  - A new consumer capability, `OFFER_LIBRARY`: telling a HUMAN which library methods this agent
    can run. The `answer` row is now `{OFFER_LIBRARY}`, where it used to hold all three.
  - For that reader, `_render_extracted` lists importable units under "METHODS THIS AGENT CAN RUN
    for the user, in its own sandbox". The import lines work only there, so the model should offer
    to run a method rather than tell the user to import it. Each line is kept as a labelled
    `agent-sandbox import:` detail.
  - `_doc_block` relabels a `MethodUnit` document the same way: "method this agent can run", and
    its import line marked as working only in the agent's sandbox.
  - The code and analyse peers keep the runnable view.
  - The answerer no longer gets the staging instruction or the `kb_method_search` pointer. Both
    were instructions to call tools it does not have, aimed at a reader who cannot call them.
  - `_correct_artifact_claims` gains a backstop. An answer containing a `from iguide_methods`
    import that never says where it runs gets an APPENDED "⚠️ Correction: … runs only inside this
    agent's sandbox, not on your machine. Ask me to run it." It only fires with the bundle on.

**Why** The map UI with the bundle on: asked for reusable choropleth code, the answer gave the user
  two `from iguide_methods...` lines to run. They are correct code, but only inside the agent's
  sandbox; `iguide_methods` is not a package anyone can install.

  This was the case M8.64 deferred ("the answer row keeps everything; B3's question"). The agent
  designer's review settled it: a rendering change, not an evidence cut, so B3 is untouched. The
  method stays a REFERENCE (element, function, version, "runnable by this agent"), and the raw
  line is kept as a detail.

  The backstop appends rather than rewrites, following the delivered-layer precedent. The import
  IS correct code, so a rewritten snippet would be neither what the model said nor true, and it
  would hide that a correction happened.

**Measured** 300 tests across the consumer, flag, join, method-evidence, gate and supervisor files; CI-style suite **3,070 passed, 1 failed** (prototype's `Beat`).
  Four join and flag tests that had been checking the CODE peer's view through the default
  consumer now name `consumer="code_peer"`. The exit-path test's docstring records the call-site
  signal seeing inside `main()` as the design win it is.

## 2026-10-08 · M8.70 · Merge prototype (61 commits, #30–#81) into the extraction branch

**Change** `origin/prototype` at `defcc86` merged into `claude/extraction-integration` so the
  branch can go to prototype as one PR. The merge base is `5ae6d92`, the VM's deployed tree. 13
  files conflicted. Each resolution keeps both sides' intent:
  - `capability_registry.py`: `Toolset` carries both sides' gates. Ours is `requires_extraction`
    and `extraction_summary`; prototype's are `requires_skills` and `requires_flag`.
    `describe(capability, *, skill_roots=None)` applies all three. Prototype's
    `describe_code_peer` is kept, and the staging toolset is its only code-only entry, shown only
    while the bundle is on.
  - `api/server.py`: three things must each allow persistence before a turn writes to the store:
    the request's flag, prototype's `deployment_mode.persistent_memory_allowed()` (`AGENT_MODE=local`,
    #47), and our `AGENT_PERSISTENT_MEMORY=0` (M8.66). Either server-side switch keeps a laptop
    off the shared store.
  - `code_execution.py` imports both `extraction_enabled` and `fork_safe` (#57). `graph.py` keeps
    prototype's gated `public_data_tools` block (#77). `executor_factory.py` gets one docstring
    listing claude-cli, then anvilgpt/lumen, then OpenAI.
  - `requirements.txt` and `constraints.txt` are prototype's (#44 installs through constraints),
    plus only what extraction needs: `psycopg[binary]` pinned to 3.3.4, `pyogrio` and `networkx`.
  - Dockerfiles are prototype's, plus our lines. `metadata-extraction-server` copies
    `extractors/`, `rag_pipeline/` and `agent_runtime/`. `rag_pipeline` copies the library build
    and smoke scripts and creates `agent_chat_files/skills`.
  - CI keeps prototype's `tests` job plus our `deployment-contract` and `artifact-replay` jobs.
  - `.claude/launch.json` keeps both sides' configs.
  - `test_supervisor_knows_its_peers.py` maps both `make_skill_tools` and the staging factory.

**Why** The user asked for the extraction work on prototype so the deployed agent can try it.
  Merging prototype in first, instead of asking a reviewer to resolve 13 conflicts in a PR, keeps
  the decisions here, beside the reasons for each one.

  Nothing changes behaviour for a deployment that does not opt in. `AGENT_EXTRACTION` is off by
  default, the KB Postgres record is lazy and off unless `AGENT_KB_DB=1`, and the library mount
  needs `docker-compose.extraction.yml`.

  Three of prototype's guards caught extraction code the moment the two met. Each is fixed
  here, not exempted:
  - The fork guard (#57) found `agent_runtime/artifacts.py` (`docker image inspect`) and
    `rag_pipeline/llm_claude_cli.py` (both CLI calls) on `subprocess.run`. Both now use
    `fork_safe.run`. The dev claude-cli path runs on exactly the Mac where a fork after a
    reprojection kills the child. The retry tests now stub `llm_claude_cli.fork_safe`.
  - The bare-credential guard's `_STILL_BARE` listed `agent_kb.py` and `opensearch_emitter.py`
    as "moved by claude/extraction-integration", and both did move in M8.65. It is empty now, as
    the guard's own expiry assertion asked.
  - The loopback-port guard read `127.0.0.1:${AGENT_DB_PORT:-5544}:5432` as exposed, because
    the default's own colon shifted its `rsplit`. It now interpolates first, as compose does. A
    default stands in for its variable, and an undefaulted variable reads as empty, so an
    interpolated ADDRESS never passes. The three new cases agree with
    `docker compose config` (the `${BIND}` one renders with no `host_ip`).

**Measured** Deployed-replica image (`iguide-replica:deeb331-freeze`, amd64, `--network none`):
  **3,514 passed, 13 failed, 16 skipped**. After the three fixes: 121 of 121 pass in the touched files.
  The other 10 fail identically on a `git archive` of pure `origin/prototype` in the same image.
  The freeze predates the readers prototype now declares (xarray, pyarrow, pypdf, docx). CI
  installs `requirements.txt -c constraints.txt`, which has them.

## 2026-10-08 · M8.71 · The contract job runs without the agent, and so does the offline stub

**Change** `rag_pipeline/tests/conftest.py`'s autouse `_no_live_services` returns early when its
  imports fail on a missing third-party package. A missing module of our own still raises.

**Why** CI on PR #83 failed `deployment-contract` with 29 errors and no assertions. That job
  (ours) installs only pytest, pyyaml, python-dotenv and networkx, because it reads Dockerfiles
  and compose rather than the agent. The fixture (prototype's, #55) imports
  `langchain_mcp_tools`, which imports `requests`, so every test errored in setup. Neither side
  failed alone; only the merge put them together. The stub prevents live calls, and code that
  cannot be imported makes none, so skipping it there loses nothing. Re-raising on our own
  modules keeps a real breakage from silently disabling the guard in the full suite.

**Measured** Reproduced the job in clean `python:3.11-slim` (amd64), with its exact
  `pip install … -c constraints.txt`: **27 passed, 2 skipped** (was 29 errors in CI). In the
  replica, the contract tests and the stub's own consumers (web search, artifact claims, ports):
  91 passed, 2 skipped.

## 2026-10-08 · M8.72 · A label is not a measurement, and a gate unknown does not re-run the turn

**Change** `check_declared_units` gives a non-numeric declared value (str, bool, None, list,
  dict) no numeric checks and records it as `declared_value`, pass, `measurement: false`.
  `_UNIT_ALIASES` counts point(s), polygon(s), zone(s), tract(s), county/counties, cell(s),
  pixel(s), region(s), site(s) and building(s). In `graph.py`, `_unsupported_claims` leaves out
  a gate `cannot_determine`, so it no longer sends the turn back to analyze. A gate `fail` still
  re-runs, and the peer is now given the gate's message.

**Why** This was the first live turn with the bundle on (2026-10-08, Lumen deepseek-v4-flash,
  Champaign buffer). The answer was right: 4 points, 10 km in EPSG:26916, area/(πr²) = 0.9984.
  It still shipped "COULD NOT VERIFY" and re-ran everything once. Run 1 was marked unknown
  because of `unit: "points"` (and because its work was inside a function). Run 2 got 15 pass,
  0 fail and 2 cannot_determine, both on `output_crs: {"value": "EPSG:26916", "unit": "crs"}`.
  The re-run repeated kb_method_search, admin_boundary, get_method_contract, execute_code and
  add_map_layer, taking about 50 s and five extra LLM calls, and emitted both layers again. The
  gate is deterministic, so re-running cannot confirm what it could not. Reasoning is in
  architecture stage 36, "The gate stops flagging correct runs, and its unknowns stop re-running
  them".

**Measured** Run 2's declared outputs now get a pass verdict with 0 unknowns, both agent-side
  and through the assembled sandbox gate. In the old code, 33 of the new gate tests and all 4
  new graph tests fail. The guards (a number in an unknown unit stays unknown, a degree buffer
  still fails) pass on both. Full suite on the Mac: 3566 passed, 18 skipped, 1 failed. The failure is `test_the_installed_networkx_matches_the_pin`, which compares the anaconda networkx (3.4.2) with `constraints.txt`. It depends on the machine and not on this change.

**Not fixed** The run 1 `coverage` unknown for work done inside a function. It shows a caveat
  and no longer causes a re-run.

