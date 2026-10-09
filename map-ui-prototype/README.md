# I-GUIDE Map UI — Prototype

A **chat-first** map interface for the I-GUIDE agent. The map is the canvas; you
drive it by talking to the agent. Every result — knowledge-base hits, live
OpenStreetMap features, uploaded data, analysis outputs — is a `LayerArtifact`
rendered by one thin function, exactly the contract the real agent will emit.

## Stack (matches the architecture decision)

- **MapLibre GL JS** via **react-map-gl** — embeddable base map (OSM raster tiles, no key)
- **deck.gl** (`MapboxOverlay`) — GPU rendering of large/complex layers (points, polygons, heatmaps)
- **@turf/turf** — in-browser spatial analysis (stands in for the agent's sandboxed geoprocessing)
- **Overpass API** — live OpenStreetMap queries, bounded by the drawn region or map view

## Run

```sh
cd map-ui-prototype
npm install
npm run dev        # http://localhost:5173
```

Claude Code sessions can start this dev server directly via the Browser preview
tool instead of a manual terminal command — `.claude/launch.json` (repo root)
declares it as the `map-ui-prototype` launch configuration (`npm run dev` in
this directory, port 5173).

## Try it (chat)

- `find flood risk datasets` — semantic KB search
- `show cafés here` / `hospitals here` — live Overpass in the current map view
- `buffer the cafés by 2 km` · `heatmap of the cafés` · `clip the datasets to the region`
- Draw a region with **▭ Region** to bound searches; drop a **GeoJSON** file to add your own layer

## Where the real system plugs in

This prototype is deliberately structured so each mock swaps 1:1 for production:

| File | Prototype (mock) | Production |
|---|---|---|
| `src/agentBrain.ts` | keyword intent parser | the real I-GUIDE agent (LLM + tools) |
| `src/mockKb.ts` `searchKb()` | in-memory + turf filter | OpenSearch kNN + `geo_shape` filter (Contract 2) |
| `src/analysis.ts` | turf in the browser | agent's sandboxed Python (GeoPandas/rasterio) |
| `src/contracts.ts` | `LayerArtifact` / `SpatialFilter` | **unchanged** — the wire contract |
| `src/overpass.ts` | direct Overpass call | direct, or proxied via a platform service |

The KB spatial fields (`spatial-centroid`, `spatial-bounding-box`) use the same
GeoJSON shapes produced by the fixed `../embedding-server/reindex_wkt_spatial.py`.

## Running against a LOCAL agent

```sh
# from the repo root (this worktree), with the agent's .env present:
PYTHONPATH="$PWD" PORT=5055 AGENT_CHAT_AUTH_OPTIONAL=1 AGENT_PUBLIC_BASE_URL= python3 api/server.py
# then point the UI proxy at it:
AGENT_TARGET=http://localhost:5055 npm run dev
```

**`AGENT_PUBLIC_BASE_URL=` (empty) matters.** The platform `.env` sets it to the deployed
host; if you inherit that while running locally, every `download_url` is an absolute URL to
the deployed server, which does not have your locally-created files — downloads then fail with
`{"error":"unknown file_id: ..."}`. Empty keeps URLs host-relative so they resolve through the
Vite proxy.

## Replaying a recorded turn

The GIS harness (`gis_harness/`) records every turn's server events as `<task>.events.jsonl`.
A replay runs one of them through this page instead of a live stream: the recorded query is sent
as if typed, `/agent/chat/stream` answers with the recorded events, and each layer is served
from the file the harness captured with the run. `streamChat`, `collectDownloads` and the
`map_layer` drawing are the same code as live. That is the reason for doing it this way: a
re-implementation of what the UI shows would be a second opinion, and this is the UI.

```sh
# screenshots: <task>.png beside every events file under the paths (absolute paths)
npm run replay:shots -- /abs/path/to/gis_harness_runs_archive/p5-gate [--force] [--parallel 4]
# by hand: open http://localhost:5173/?replay=p5-gate/lumen_deepseek-v4-flash/T02L.events.jsonl
REPLAY_ROOT=/abs/path/to/gis_harness_runs_archive npm run dev
npm run check:replay     # the reducer, the server, and a recording through the real streamChat
```

- **Only dev and `VITE_REPLAY=1` builds have it.** `REPLAY_KEY` folds to `null` in a production
  build, so `npm run build` ships none of it.
- **A layer the replay cannot draw is said on the picture.** A strip above the page names the
  run, model and turn, counts the stream's map layers against the ones the page actually holds
  when the turn ends, and gives a line per layer that did not make it: its file was never
  captured, its inline GeoJSON arrived as a string, or a raster's bounds are projected metres so
  it lands off the map. The strip turns red when there is any such line.
- **The screenshot holds the whole transcript.** In replay the page grows to fit it and the map
  keeps a fixed 820 px height at the top, so its framing is the one a user saw.
- What the replay changes in a recording, each named on the strip: absolute agent-file urls
  become host-relative (nothing is fetched from the deployment), geojson stored as JSON text by
  `chat_traces` is parsed back, and a recording with no answer event gets its record's answer.
- A replay never saves itself to History.
- Headless Chrome is the installed Google Chrome (`playwright-core`, no browser download), or
  `REPLAY_CHROME=<binary>`. Basemap tiles come from tile.openstreetmap.org as they do live,
  through a persistent profile in `node_modules/.cache/replay-profile` so each area is fetched once.

## Two visualization routes

- **Interactive map** — vector data (GeoJSON) is plotted as a layer. Geometry streamed via the
  `map_layer` SSE event *and* any `.geojson` file artifact the agent writes are both auto-loaded.
- **Static image** — a PNG plot stays an attachment/download, for when an image is what was asked for.

## Switching back to the original prototype page

The pre-#20 page — I-GUIDE platform chrome, "Knowledge Elements", the generic starter prompts —
is kept in the tree, not just in git history, and is selected by a build flag:

```
VITE_UI_VARIANT=platform npm run dev      # or npm run build
```

Unset (the default) gives the rs-embed deployment: the I-GUIDE mark linking back to the
platform, "I-GUIDE AI", and remote-sensing prompts.

What makes up each variant:

| | default (`rsembed`) | `platform` |
|---|---|---|
| header | `src/components/TopNav.tsx` | `src/components/TopNav.platform.tsx` |
| prompts | `SUGGESTIONS_RSEMBED` | `SUGGESTIONS_PLATFORM` |

`src/uiVariant.ts` is the single flag both read, so the header and the prompts can never
disagree. It is a direct `import.meta.env` comparison on purpose: Vite inlines it, so the
variant you are not building is dropped from the bundle rather than shipped dead — verified by
grepping `dist/` for each variant's marker strings.

The platform copy is verbatim from commit `76dab0b`, including the CSS its header needs
(`.navsearch`, `.jpy`, `.avatar`, the caret and the narrow-screen rule). Do not tidy it: its
value is being an exact copy.
