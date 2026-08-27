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

**Errors name the alternatives.** A dead end costs a whole turn, so a failure returns the
next action: a wrong choropleth column returns the numeric columns; an unknown KB block
returns the nearest real ones; an image passed to `add_map_layer` returns the attached
datasets it *can* read; a filter matching nothing returns the column's actual range instead of
writing an empty layer.

**Artifacts are named for their purpose.** `artifact_name()` derives a name from what the file
is for, so a conversation doesn't accumulate `output_1.geojson`. Large outputs are reported
with their size (`AGENT_LARGE_ARTIFACT_MB`, default 25) because an invisible 89 MB intermediate
was being written every turn.

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

## Code execution

`execute_code` runs per-run Docker: `--network none`, read-only root filesystem, dropped
capabilities (`agent_runtime/code_execution.py`). A **separate** install phase has network for
pip. Consequences: code cannot fetch anything at runtime — an API call belongs in a tool in
the agent process, not in generated code — and abnormal exits are translated
(`_diagnose_abnormal_exit`: 137 is the OOM kill, 139 a segfault) because the raw signal
surfaced as an empty stderr.

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
