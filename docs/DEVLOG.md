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
