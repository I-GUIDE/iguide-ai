# Working on this repo (any coding agent)

Instructions for AI coding agents. `CLAUDE.md` is a symlink to this file, so Claude Code,
opencode (which this repo runs as a code peer), Cursor and Codex all read the same text.

Everything below is a fact that cost someone time to discover, or a convention that is
deliberate and will look wrong if you don't know why. Structure, stack and file layout are
omitted on purpose — you can read those faster than a document can describe them. Every
claim names a file so you can verify it rather than trust it.

## Run it

Agent API (Flask + gunicorn in prod, `app.run` locally):

```bash
PYTHONPATH="$PWD" PORT=5055 AGENT_PUBLIC_BASE_URL= AGENT_CHAT_API_KEY= python3 api/server.py
```

All four matter, and each failure is silent or misleading:

- no `PYTHONPATH` → `ModuleNotFoundError: rag_pipeline` (`api/server.py` imports it, and running
  a script puts only `api/` on `sys.path`)
- `AGENT_PUBLIC_BASE_URL` inherited from a deployment `.env` → every `download_url` becomes
  absolute against the *remote* host, so local downloads 404 with `unknown file_id`
- `AGENT_CHAT_API_KEY` set → `/agent/chat*` answers 403 (`_get_agent_chat_api_key`); empty
  disables the gate
- `PORT` defaults to 5002; the compose deployment maps host 3500 → container 5002

Map UI prototype (`map-ui-prototype/`, Vite + React + MapLibre + deck.gl):

```bash
AGENT_TARGET=http://localhost:5055 npm run dev     # proxies /agent/* same-origin
```

The proxy in `vite.config.ts` is **dev-only**. A production build has no proxy: serve the
static bundle and the API from one origin, or set an absolute endpoint in the UI's Connection
panel. Node 18+ required.

## Tests

```bash
python3 -m pytest rag_pipeline/tests/ -q
```

Baseline is **649 passed, 1 skipped, 0 failed**. If something fails, it is yours.

That baseline was reached by fixing a test everyone had learned to ignore, and the way it hid
is worth knowing because it will happen again. `test_spatial_routing_e2e.py` suppresses the
non-spatial retrieval sources so it can assert every document came from the spatial one. It
patched `rag_pipeline.search.keyword.retrieve_keyword` — but `core.py` does `from .keyword
import retrieve_keyword`, a from-import that binds the function into `core`'s namespace at
import time, so patching the defining module afterwards changes nothing and the real function
still runs. **Patch where a symbol is used, not where it is defined.** Two further sources
(neo4j, opengeodata) had been added since the test was written, and the graph tier that
actually runs is `get_neo4j_agent_results`, not the `retrieve_neo4j` fallback beneath it.

The failure was misattributed for a long time to the missing spaCy model `en_core_web_sm`,
because that logs a loud warning on import of `rag_pipeline/search/spatial.py`. It is a red
herring for this test: with no model, `_extract_place_candidates` falls back to
`_capitalized_candidates`, which handles the test's query fine. Installing the model is still
worth doing — production entity extraction runs on a weaker regex path without it — but it
fixes nothing here.

## Which model answers

**A request with no `model` and no `provider` uses OpenAI `gpt-4o-2024-11-20`.** That is the
deliberate default: it is what the deployment has been validated against, and it is what every
client got before selection existed, so an older caller behaves identically.

Selection is per request — `model`, `provider` and `reasoning_effort` on `/agent/chat` and
`/agent/chat/stream`, resolved in `_llm_for_request` (`agent_runtime/agent_chat_service.py`)
and built by `build_llm` (`agent_runtime/executor_factory.py`). `GET /agent/models` lists what
may be selected; it queries each provider live, because a hardcoded id that the provider has
retired 404s at request time instead of being absent from the picker.

Two providers are wired:

- **OpenAI** — the default. The `gpt-5.x` family and the o-series accept `reasoning_effort`;
  the accepted values are `none`, `low`, `medium`, `high`, `xhigh`. **Not** `minimal`, which
  some docs list and `gpt-5.6-luna` rejects. `supports_reasoning_effort()` decides, and the
  argument is dropped for models that would refuse it, so a UI leaving the control set while
  switching to gpt-4o cannot break the request.
- **AnvilGPT** (Purdue RCAC, Open WebUI) — set `AGENT_LLM_PROVIDER=anvilgpt` for the
  process default, or select a model per request. Its ids look like `qwen3.6:27b`, NOT the
  HuggingFace `Qwen/Qwen3.6-27B` form a vLLM server uses; a wrong id 404s. Chat lives at
  `/api/chat/completions`, which `normalize_openai_base_url` reduces to the `/api` base.

**Do not set `max_tokens` for a reasoning model.** qwen3.6:27b and the gpt-5.x line spend
their first tokens on reasoning and only then write `content`, so a tight ceiling returns
`finish_reason="length"` with `content=None` — an EMPTY answer. `extract_final_answer` reads
blank as "no answer", so a truncated reasoning model looks like a failed peer rather than a
cut-off one. Measured on qwen3.6:27b: `max_tokens=20` produced no content at all; unset
completes normally.

**A reasoning model over an OpenAI-compatible shim loses its own thinking unless we keep
it.** `gpt-oss:120b` on AnvilGPT answers a tool-calling step with `content: None`,
`reasoning_content: "<the plan>"` and the tool call. `langchain_openai` 1.6 parses neither
reasoning field onto the message — it is absent from `additional_kwargs` AND from
`response_metadata` — so the assistant turn replayed on the next step is
`{role: assistant, content: None, tool_calls: [...]}`: a bare tool call with no record of why
it was made. The model re-derives its plan from the user request alone, reaches the same
conclusion, and issues the same call again.

Measured: one turn spent five `admin_boundary` calls across two peers — the first two with
IDENTICAL arguments — then cycled `execute_code` through guesses at a filename it had already
been given.

**This is a defect of OpenAI-compatible SHIMS, not of reasoning models generally.** OpenAI
proper never puts the chain of thought on the wire: measured against `api.openai.com` with
tools bound, the assistant message keys are exactly `['annotations', 'audio', 'content',
'function_call', 'refusal', 'role', 'tool_calls']` for `gpt-4o`, `gpt-5.6-luna`, `gpt-5.2` and
`o4-mini` alike — `reasoning_content` is *absent*, not null, even when the model demonstrably
reasoned (`gpt-5.2` at effort=high burned 128–214 reasoning tokens and returned none of the
text; only `usage.completion_tokens_details.reasoning_tokens` records it). So the subclass is a
permanent, harmless no-op on the OpenAI path, and no client-side fix exists there — only
`/v1/responses`, which we do not use, carries reasoning items across steps.

Two corrections to earlier wording that was wrong. gpt-4o is **not** immune because "its plan
lives in `content`" — measured, gpt-4o also returns `content: None` on a tool-calling step. It
is immune because it has no chain of thought to lose. And `gpt-5.x` is **OpenAI**, not a
different provider; `gpt-5.6-luna` in particular is immune for a third reason again — with
function tools OpenAI hard-rejects any `reasoning_effort` but `'none'` (HTTP 400, *"To use
function tools, use /v1/responses or set reasoning_effort to 'none'"*), `resolve_effort` forces
`'none'`, and at `'none'` luna reports `reasoning_tokens: 0`. **As this agent calls it, luna is
a non-reasoning tool-caller.** Unshackled (no tools) it reasons normally — 181 tokens at
`'high'` — so that is a real capability we trade away for tool use on this endpoint.

`_reasoning_preserving_chat_openai()` (`agent_runtime/executor_factory.py`) subclasses
`ChatOpenAI` and, in `_create_chat_result`, stashes the reasoning on the message and promotes
it into `content` when the provider left content empty **on a tool-calling step**. Gated on
`tool_calls` deliberately: an intermediate step is the only place this is safe, because
promoting deliberation onto a FINAL message would leak it into the user's answer. Both
construction paths use the subclass — including the per-request one, which is where the model
picker actually selects gpt-oss.

Which model actually answered is not visible in the transport log — it shows only the host,
and a dropped `reasoning_effort` looks identical to an applied one. So each per-request build
logs `per-request LLM: provider=… model=… reasoning_effort=…`, and
`active_llm_description()` reports the process default. Read those rather than inferring the
model from whether a call succeeded.

## The delivery contract

An analysis result reaches the user as an **interactive map layer**, not a file path in prose:

- `add_map_layer` (`agent_runtime/langchain_geo_tools.py`) is how a map gets delivered —
  heatmap / choropleth / points / shapes, plus a downloadable GeoJSON.
- It travels as a `map_layer` SSE event, forwarded verbatim by `api/server.py` (search
  `event_name == "map_layer"`). The event is **additive** — no existing SSE event changed —
  so a chat-only client that ignores unknown event types is unaffected.
- `render_map_image`, `heatmap_image`, `choropleth_image` and `qgis_map_image` are the other,
  separate route: a static PNG that cannot be panned, zoomed or clicked. Do not describe one
  as being "on the map".
- **Geometry never goes into the LLM-visible documents.** Evidence documents carry titles and
  abstracts; footprints and coordinates go to the map on the side channel. Widening the
  documents floods the context and gets truncated.

## Conventions that are deliberate

**Capability statements, not mandates.** Prompts (`agent_runtime/prompts.py`,
`agent_runtime/supervisor/prompts.py`) state what a tool can do and what each route costs. An
earlier version issued mandates and threats ("an answer that only pastes code is a FAILURE").
That shape backfires: a model told that not running code is a failure claims it ran when the
sandbox died. If you are tempted to add "you MUST", add a check instead.

**Verify structurally, then hand back the observation.** The supervisor checks whether the
work actually happened and gives the peer one retry carrying that fact —
`_has_execution_record` / `_called_tool` and the `*_OBSERVATION` constants in
`agent_runtime/supervisor/graph.py`. Two live examples: an answer shipping unrun code, and an
answer claiming an interactive map when no layer-emitting tool ran.

**Metric discipline.** Every distance, area and radius reprojects to an estimated UTM CRS.
`units='degrees'` is refused outright, and metre-based-but-distorted CRSs (EPSG:3857 and
friends) are re-measured — a 10 km buffer in 3857 at 40°N covers ~7.7 km on the ground. See
`_as_metric` in `agent_runtime/analysis_overlay_tools.py`. Never compute a ground distance in
degrees.

**Provenance travels with the number.** rs-embed's own service says it best, in the comment
above the line that forwards its embedder metadata verbatim: it "carries the provenance a
caller must not invent". A tool that passes on the vectors and drops how they were produced
does not stop the model answering *"what resolution was that?"* — it makes it answer from the
defaults, which is right up until a default moves. So `_provenance` in
`agent_runtime/rs_embed_tools.py` carries the keys that change what a number MEANS (imagery
source and collection, `scale_m`, compositing and cloud threshold, date range, model variant
and normalisation, grid shape and orientation, `nodata_fraction`) and drops the embedder's own
diagnostics (`param_*`, `device`, `batch_*`, `tokens_shape`), which crowd the context and
answer nothing a user asks. Two shapes caught only by running it against a live embedding:
`bands` is nested under `sensor` on the on-the-fly path but top-level on the precomputed one,
and a precomputed product lists its 64 embedding DIMENSIONS there rather than spectral bands —
so past a spectral-length list only the count is kept. Absent keys are OMITTED, never reported
as null: "the deployment does not send this" and "the run had no value for it" are different
facts. `/api/embed` does not attach `meta` yet (only `/api/zones` does), so `embed_region`
reports provenance only where the service supplies it.

**Errors name the alternatives.** A dead end costs a whole turn, so a failure returns the
next action: a wrong choropleth column returns the numeric columns; an unknown KB block
returns the nearest real ones; an image passed to `add_map_layer` returns the attached
datasets it *can* read; a filter matching nothing returns the column's actual range instead of
writing an empty layer.

**Artifacts are named for their purpose.** `artifact_name()` derives a name from what the file
is for, so a conversation doesn't accumulate `output_1.geojson`. Large outputs are reported
with their size (`AGENT_LARGE_ARTIFACT_MB`, default 25) because an invisible 89 MB intermediate
was being written every turn.

**And the answer is checked against the artifact, deterministically.** Provenance in the tool
result only helps if the answer uses it. Live, "show me the clay embedding of urbana at
2025/03/01-2025/05/01" ran `embed_region` with no `model`, embedded with the default (gse), and
then reported *"Here's the Clay v1.5 embedding … extracted from the global LGND Clay Embeddings
– Sentinel-2 collection … 2.56 km MajorTOM grid cell"* — provenance lifted from a web-search
hit for the word "clay" — while the legend beside it read `gse embedding (PCA-RGB)`. The same
answer told the user to add the layer from a URL, when it was already rendered. The LLM
grounding audit flagged none of it: the cited pages were real, they just were not where the
raster came from.

Two checks in `supervisor/graph.py`, both deterministic because the LLM audit is what missed
this. Upstream, the analyse peer retries once when the query names a model
(`_models_named_in`) that is not the one that ran (`_models_used`) — fixing the cause, since a
caveat on the wrong raster is not what was asked for. Downstream,
`_correct_artifact_claims` appends a marked correction when the answer names a model that did
not run, or tells the user to add a layer that was already delivered (`_DENIES_MAP_RE`). Model
ids are probed from the service, not hardcoded, so an unreachable service disables the model
check rather than making it wrong — the map-denial check needs no catalog and keeps working.

## The tool surface

36 tools in six families, plus `execute_code` and four file tools. Enumerate them from the
factories rather than trusting a list — that is the only trustworthy inventory:

`make_overlay_tools` · `make_aggregate_tools` · `make_temporal_tools` ·
`make_langchain_geo_tools` · `make_rs_embed_tools` · `make_langchain_qgis_tools` ·
`make_code_execution_tools` · `make_langchain_file_tools`

The analysis families load **only when files are attached** to the conversation
(`default_analyze_fn` in `agent_runtime/supervisor/graph.py`), so a bare chat session has none
of them. `rs_embed_tools` calls an external service at `RS_EMBED_URL` (default
`http://localhost:8077` — inside a container that means the container itself, not the host).

There is **no raster analysis**: no zonal statistics, band math, reclassify or terrain. Route
that through `execute_code` (rasterio is available) or a GDAL algorithm via
`qgis_processing_run`. The map client models vector layers only.

## Why this workload is a poor fit for multiple agents

Worth stating plainly, because the supervisor-over-peers shape invites the assumption that
adding agents adds capability. For THIS workload it mostly adds seams.

**The peers never run in parallel.** `decide()` returns ONE action per step and the graph
routes through conditional edges, so search/analyze/code execute strictly in sequence. The
usual justification for a multi-agent design — independent specialists working concurrently —
has never applied here. What we actually have is a sequential router over fragmented state.

**The work is one long chain over shared state, not separable subproblems.** A real request is
`admin_boundary` → `embed_zones` → `fit_zone_model` → `add_map_layer`: each step consumes the
previous step's artifact (a file_id, a GEOID column, a CSV of vectors) and its provenance
(which model, which date range, which resolution). Specialists pay off when subproblems are
independent and only their conclusions need to meet. Here almost every intermediate value is
needed downstream, so every peer boundary is a place for that value to be dropped — and it was.
Measured consequences, all from one exchange:

* The answer to "what resolution was that" (`scale_m: 10`) sat in the analyze peer's thread.
  The router sent the follow-up to the SEARCH peer, which has its own separate thread and had
  never seen it. (This bullet used to say the search peer "starts fresh every turn — started
  with 2 message(s)". That is wrong: each peer's thread is checkpointed on a stable child id
  and accumulates. The observation was real but came from a probe that sent no stable
  `thread_id`, so every turn minted `auto-thread-<uuid4>`. The fragmentation argument does not
  depend on it — state split across peers is the point, and separate threads are exactly that
  split.) It searched for 49 steps —
  drifting into SoilGrids *soil* clay — until the payload hit 66,275 tokens against a 65,536
  window and the turn died with a 400. One clay turn reached 199,605.
* `admin_boundary` must be registered in three peers under three different gating rules
  (25 `tools.extend` sites in `supervisor/graph.py`). One copy ended up nested inside
  `if input_file_ids:` — the exact gate its own comment said to avoid — so with nothing
  attached the tool did not exist, and every model fell back to embedding a rectangle around a
  city centroid. Two rounds of tool-description tuning could not fix a tool that was absent.
* The action ledger above exists ONLY to reunify what the peer split broke. It is a patch over
  the architecture, not a feature of it.

**What the split genuinely buys is not routing but a place to stand.** The structural checks —
`_MAP_LAYER_TOOLS`/`on_map`, `_has_execution_record`, the model-mismatch retry,
`_correct_artifact_claims` — work because they run OUTSIDE the loop and can force a retry. That
value is real and worth keeping: the in-loop LLM grounding audit read
*"the Clay v1.5 embedding … from the LGND Sentinel-2 collection"* over a gse raster as
well-supported, because the pages it cited were real; only the deterministic outside check
caught it. An agent grading its own trace misses what it got wrong, because the error and the
grader share a premise.

So the two jobs separate: **verification outside the loop is worth its cost; routing between
peers is not.** If this is ever revisited, collapse the peers into one loop with one context and
one tool list, keep the checks around it, and add token-budget trimming first — a single loop
reaches the context window sooner, not later. Design note:
https://claude.ai/code/artifact/a9fd7318-1125-4fa9-aa3d-0b2e45321f16

## The conversation remembers what it DID

`session_memory` stored `{userQuery, answer}` — prose. Everything a tool produced was discarded
at the end of the turn, and the supervisor's decision payload (`_distill`) is built from
per-turn state, so on turn 2 it reports `has_evidence=False` / `has_analysis=False` /
`artifacts_produced=[]` however much turn 1 did. Routing to `search` was the correct read of
what it was shown. Measured: "do you use the original resolution of clay or do you downsample
it" became forty-nine keyword searches (drifting into SoilGrids *soil* clay) and died at
66,275 tokens against a 65,536 window — while `scale_m` sat in the previous turn's tool result.

`_ledger_rows` records one row per tool INVOCATION — `{tool, curated args, curated facts,
file_id, map_layer}` — into `session_memory._ACTIONS` (process-local, bounded, same lifetime as
the turn store). Two consumers, and both were needed:

* `_distill(..., for_decision=True)` adds `prior_turns_in_this_conversation` so the supervisor
  can pick `done`. Decision-only — the client payload is a per-turn record and stays clean.
* Synthesis gets `_prior_actions_note()` as a system message. This is the one that fixed the
  answer. With the ledger in the routing payload alone the decider still chose `search` twice;
  the answer only became right when the ANSWERING model could see the facts.

**The auditor needs the ledger too.** For a while only the answerer got it, and the grounding
audit still compared the answer against this turn's evidence and execution record. A follow-up
that correctly read the previous turn's gse parameters off the ledger ("64 dimensions, 7.645 m
per pixel") was therefore audited against material that never mentioned them and the answer
shipped with *"parts of this answer may not be fully supported by the retrieved evidence
(severity: high)"* stapled to it — the feature accusing itself of hallucinating, in front of
the user. `_ledger_lines()` is now shared: the same rendered rows go into the synthesis note
AND into `execution_context["prior_actions"]`, which both the LLM auditor
(`_format_execution_context`) and the deterministic reconciliation read. The map is persistent
too, so a layer added in an earlier turn still counts as delivered when a later answer refers
to it.

Three things cost a deploy cycle each and are worth not rediscovering. `tool_results` entries
are `{name, tool_call_id, content}` with content a JSON **string** — reading `output`/`result`
yields empty payloads while every hand-written unit test passes, which is why these tests build
artifacts through `extract_search_artifacts` instead. `graph.py` had **no module logger**, so
the first diagnostic raised `NameError` inside the recorder. And the field names are the
service's, not the user's: given `scale_m=10` the model still answered *"the available evidence
does not specify the ground-resolution"* — `_FACT_PHRASES` spells it out as "10 m per pixel",
and that is what turned it into *"generated at a 10-meter ground resolution"* with zero
searches.

## Named areas without an upload

Every polygon tool starts from a file the user attached, so "the embeddings for Champaign
County" was unanswerable without one — the user had to bring a boundary and know the GEOID
inside it. `admin_boundary` (`agent_runtime/admin_boundary_tools.py`) takes a state, county or
city NAME and returns a WGS84 GeoJSON that is already on the map and already shaped for
`embed_zones` (`zone_id_field="GEOID"`); `subdivide="tracts"` returns the tracts inside a named
county, which is the many-zone input that tool is for.

It reads the Census **TIGERweb** REST API, not Earth Engine, for three reasons: the agent
container has no `ee` and no Earth Engine credential (that lives only in the rs-embed service,
under a personal Google account); TIGERweb needs no credential; and it has the one layer Earth
Engine lacks — incorporated places. `TIGER/*/Places` is not in the EE catalogue at any vintage,
and GAUL/geoBoundaries stop at district, so a *city* boundary is simply unavailable there
(geoBoundaries has no ADM2 named "Nairobi" at all — Kenya's ADM2 are sub-counties).

Two behaviours are load-bearing. It matches `BASENAME`, not `NAME`: the latter carries the
suffix ("Champaign County", "Champaign city"), so matching it loses every county a user names
without saying "County". The mirror of that also has to work, and originally did not: matching
BASENAME literally meant `area="Champaign County"` found nothing, and the LIKE fallback
searches BASENAME too, so the caller got a dead end with no candidates. `_name_variants()`
therefore tries the name as given AND the name with that level's own suffix removed (County,
Parish, Borough, city, town, …), in both the exact match and the fallback. Watched live:
gpt-oss:120b spent three tool calls guessing `"Champaign County"/Illinois` →
`"Champaign County"/IL` → `"Champaign"/IL` before it landed. And a name matching several places is REFUSED with the candidates
listed — 16 incorporated places are named Springfield and the first is in none of the states
anyone means, so picking one silently is the failure that matters.

Registering it forced a related change: the zonal tools were gated on `input_file_ids`, on the
premise that a polygon layer could only arrive by upload. That premise is gone, and leaving the
gate would have hidden the tool that consumes what `admin_boundary` produces — the model would
fetch a county and have nothing to embed it with.

Measured, and worth knowing before pairing them: Champaign County's 48 tracts planned **1140**
tiles. `embed_zones` is **uncapped by default**, so that sweep runs in full — 1140 requests to
the imagery provider — rather than returning a few per cent. Pass `max_tiles` for a bounded look
or `zone_ids` for a few zones; either way `admin_boundary` says it up front in `coverage_hint`,
while the budget can still be chosen.

## Code execution

`execute_code` runs per-run Docker: `--network none`, read-only root filesystem, dropped
capabilities (`agent_runtime/code_execution.py`). A **separate** install phase has network for
pip. Consequences: code cannot fetch anything at runtime — an API call belongs in a tool in
the agent process, not in generated code — and abnormal exits are translated
(`_diagnose_abnormal_exit`: 137 is the OOM kill, 139 a segfault) because the raw signal
surfaced as an empty stderr.

**A baked image saves nothing on its own.** `pip install --target` sets
`ignore_installed=True` — pip does not consult the image's site-packages — so a package baked
into `AGENT_CODE_EXEC_IMAGE` is reinstalled on every run regardless. `sandbox/Dockerfile`
promised that saving for a long time without delivering it. What delivers it is
`preinstalled()`: the executor runs the image once per process (`--network none`,
`--read-only`, same posture as a real run), asks `find_spec` which of `_IMPORT_TO_PIP` it can
import, and drops those from the install list. It is PROBED, never declared, because a
declared list drifts: an image that stopped shipping a package would suppress an install the
code needs and die on `ModuleNotFoundError` with nothing naming the cause. Every probe failure
returns the empty set, i.e. exactly the old behaviour. Only unconstrained specs are dropped —
`geopandas==0.14` must never be satisfied by whatever the image happens to carry.

**A conversation's installed packages persist, and are mounted rather than copied.** One run's
`pip install` is importable by the next (`workspace_deps_dir`). The cache is *bind-mounted* at
`/work/.deps`, not carried through `_copy_tree`, because a site-packages tree is tens of
thousands of files and copying it in and out of every run would cost more than the install it
replaces — which is also why `_copy_tree` still skips `.deps` unchanged. It mounts **read-only
for the run** and writable only during the install phase: the cache outlives the run, so code
that could write to it could leave a shadowing `numpy.py` for the next turn to import. Two
consequences that bit during the change: pip leaves an existing copy in a `--target` directory
alone and only warns, so the install phase needs `--upgrade` or a cached package silently pins
whatever arrived first; and a directory's mtime only moves when an entry is added or removed,
so a workspace must be `utime`d on use or the TTL sweep deletes a live conversation.

**Installs are pinned to the image's own versions, and that is not optional.** `--target` sets
`ignore_installed=True`, so pip re-resolves the FULL transitive closure of anything requested
and writes its own numpy and pandas into the cache — which precedes site-packages on
`PYTHONPATH`. Measured: with the image on numpy 2.4.6, `pip install --target … numpy==2.3.0`
put 2.3.0 in the cache and the NEXT run imported 2.3.0, for the rest of the conversation, while
the model was told `installed: []`. That is an ABI hazard, because the image's rasterio and
geopandas were compiled against the version it shipped. So the probe reports `{name: version}`,
not just names, and every install carries a constraints file built from it
(`_constraints_text`, `AGENT_CODE_EXEC_PIN_IMAGE=0` to disable): a genuine conflict now fails
with pip naming it, which beats a segfault three tools later.

**A cache entry is only trusted if its install finished.** `cached_dep_names` requires a
`RECORD` inside the `.dist-info`, and a failed or timed-out install evicts the cache
(`_evict_torn_cache`). pip moves the package tree and its metadata into `--target` as separate
steps, across a bind-mount boundary so it is a copy rather than a rename; a kill on the install
timeout leaves metadata with no package beside it, and trusting the directory name alone would
suppress the reinstall for the life of the conversation. Both evictions rename the directory
aside before deleting it, because another run of the same conversation may have it mounted into
a live container — the analysis peer and the code peer share the `::codeexec` session.

**A fix should be a patch, not a re-emitted program.** `write_workspace_file` /
`read_workspace_file` / `edit_workspace_file` act on the conversation's working directory
host-side, and `execute_code(entrypoint="main.py")` runs a file already in it — so changing
line 40 of a 200-line script is one exact-match replacement instead of the whole script again
through a full LLM round trip. `edit_workspace_file` refuses a snippet that does not appear
exactly once, because replacing one of three identical lines produces code that runs and is
wrong. These are registered only when there IS a durable workspace (`graph_runtime.py` wires
`execute_code` without a session), and the agent's general file tools cannot substitute: they
are rooted at the repo, the file store and `UPLOAD_FOLDER`, and turn a bare filename into a
managed store output, so they would appear to edit the workspace while never touching it.

Three guards on that surface were each a live escape before they were closed. `entrypoint` is
resolved through the same boundary as a written path and its failure is REPORTED, not
swallowed — validating only the copy read for dependency inference still ran `../outside.py`.
Reserved directories are matched case-INSENSITIVELY, because `Path.resolve()` does not
canonicalise case and `.DEPS/numpy.py` therefore walked straight past an exact-match check into
the dependency cache. And a workspace key is a slug PLUS a digest, since the slug maps every
unsafe character to `_` and truncates, so client-supplied thread ids `sess:42` and `sess_42`
shared one directory — and now would have shared a package cache.

**A polygon and a date range are no longer exclusive.** `embed_region` takes `start`/`end` and
embeds a RECTANGLE; `embed_zones` embeds the polygon but took only `year`. So "the clay
embedding of Urbana for 2025-03-01..2025-05-01" forced a silent trade — the agent kept the
dates and quietly returned a rectangle around the city centroid instead of the city. The zonal
service request (`ZonesReq`) now accepts `start`/`end` and prefers them over `year`
(`TemporalSpec.range` was already there, one line below the `year` branch), and `embed_zones`
passes them through. Both or neither: half a range would become a whole-year composite, which
is the same quiet substitution in a new place. NOTE that the service file
(`examples/webapp/server.py`) is UNTRACKED — rs-embed gitignores `examples/**` — so that half
of the change lives only on the deployment and in `~/webapp-backups/`.

**The code PEER is swappable, and that is a different thing.** `execute_code` is a tool the
LangChain peer calls; `AGENT_CODE_PEER` replaces the peer itself with an agentic CLI that
iterates inside its own container — write, run, read the error, retry — and returns the same
flat `answer`/`tool_calls`/`tool_results` shape, so synthesis and the trace pipeline cannot
tell which ran. Two are wired: `opencode` (`agent_runtime/opencode_peer.py`, pointed at the
deployment's OpenAI-compatible endpoint) and `claude` (`agent_runtime/claude_peer.py`,
Anthropic, `ANTHROPIC_API_KEY`). Each needs its sandbox image built first —
`Dockerfile.opencode` / `Dockerfile.claude`. Unlike `execute_code` these containers **keep
network**, because a CLI with no LLM endpoint does nothing; the hardening is the rest of the
flag set, not the network.

The Claude peer takes **either** credential, and they bill different accounts:
`CLAUDE_CODE_OAUTH_TOKEN` (from `claude setup-token`) authenticates as a Claude *subscription*
and counts against that person's plan; `ANTHROPIC_API_KEY` is metered API billing. The token
wins when both are set. This is not a free swap: `--bare` never reads OAuth, so it is passed
only on the API-key path, and the CLAUDE.md auto-discovery it would have suppressed is closed
instead by renaming any staged upload named `CLAUDE.md` — an uploaded file must not become a
brief for an agent running with permissions skipped.

**A Claude subscription token is an inference credential, and it still will not run the
agent.** `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token` is scoped `user:inference`; sent as
`Authorization: Bearer` plus `anthropic-beta: oauth-2025-04-20` it authenticates `/v1/messages`
and returns real tool calls (sent as `x-api-key` it is a 401 — that is the whole of the
"it is not an API key" story). `build_llm` accepts it, so Claude is selectable as an ANSWERING
model with no API key. What stops it is commercial, and the API says so itself: *"Third-party
apps now draw from extra usage, not plan limits."* The plan's included quota covers Claude Code,
not an app built on the same token, so a turn 400s until someone adds extra usage — which is
metered spending, i.e. an API key by another name. The CODE PEER is unaffected: it runs the CLI,
which is not a third-party app.

**A CLI peer's tool surface is deliberately wide.** Measured from inside the sandbox: it has
Bash/Read/Write/Edit, sub-agents, and WebFetch/WebSearch, and code it runs reaches the
internet (`urlopen` returned 200) — because unlike the `execute_code` sandbox it is NOT
`--network none`. So swapping the peer swaps the network posture of generated code, and its
web tools do not pass through this agent's own two-step caps or `AGENT_WEB_ALLOWED_PORTS`.
That is a decision, not an oversight: a peer that can install a package, read the traceback
and retry is the point of running one, and the CONTAINER carries the safety. Narrow it with
`AGENT_CLAUDE_ALLOWED_TOOLS` if a deployment wants that; it is unset on purpose.

A CLI peer has none of the AGENT's tools, so no `add_map_layer` — but its geodata still
reaches the map: `map_layers.layers_for_artifacts` turns any `.geojson` it wrote into a
descriptor and the peer wrapper emits it, from the request's trace context, through the same
`build_map_layers` boundary every tool's layer crosses.

**A conversation keeps its project directory between turns** (`claudesess_<thread>` under
`AGENT_CODE_EXEC_WORK_ROOT`), so the CLI resumes with `--continue` and its `pip install
--user` cache survives; the container is still fresh each run. Two things that forced
themselves on the design: artifact persistence walks the whole dir, so without a
size+mtime manifest every earlier file is re-uploaded every turn; and the CLAUDE.md guard
must be scoped to THIS TURN's uploads, or it renames the CLI's own `.claude` state and
`--continue` silently finds no history.

Do not assume a CLI's flags. Claude Code 2.1.x has **no** `--max-turns`, and an unknown flag
makes the CLI exit non-zero — which reads as "the peer failed", not "somebody guessed". Check
`--help` in the built image and pin the check in a test.

## The published capability atlas

`docs/spatial-toolkit.html` is the tool inventory as a reader-facing page, enumerated from the
running container and published as an artifact that is updated **in place at the same URL**.
It goes stale silently — it sat for eleven days missing `admin_boundary`, `embed_zones` and
`fit_zone_model`, and still told readers the embedding tools "need a region — draw one", which
`admin_boundary` had made false. Re-enumerate and update it whenever the tool surface, the
delivery contract or the stated limits change. `docs/README.md` says how.

## Where the pieces are, and where they might go

`docs/system-components.md` maps the running system — frontend, HTTP surface, agent runtime,
tools, sandbox, storage, external services — and then proposes (but does not build) a middleware
between the frontend and the agent, with what would move into it and what must not.

Read it before moving anything across that boundary. Two of its conclusions are easy to get
wrong from first principles: `api/server.py` is **already** a middleware fused into the agent's
process, so the work is mostly extraction; and `thread_id` is not a label — it hashes to the
sandbox workspace directory, so minting a fresh one per request silently hands every turn an
empty `/work`.

## Verifying UI and delivery changes

For anything the browser renders, load the running prototype and drive the real gesture before
calling it done. API-level checks reported success while the map was broken twice: MapLibre
mounted in a `display:none` container never fires `load` and freezes its canvas, and a layer
delivered three times buried a density surface under raw points. `"tests pass"` and `"the SSE
stream contains the event"` are necessary, not sufficient — count the layers on screen and
confirm the render mode.

## Deployment

Runs as four Docker Compose services behind nginx, which terminates TLS and needs
`proxy_buffering off` with a long `proxy_read_timeout` for the SSE stream. Host-specific
detail — addresses, the satellite-embedding service unit, dependency pins — is operational and
deliberately not in this file; ask the maintainer.

Never commit `.env`, API keys, or Earth Engine credentials.
