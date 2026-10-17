# GIS task harness

Whole-task regression runs for the agent: twelve classic GIS problems, four on real data, one
live-data variant, and four questions that cannot be answered. Every task has an exact expected value, or a check
any analyst could repeat. Each run goes through the real HTTP API of a local server and is
scored mechanistically. There is no LLM judge.

It exists because on 2026-10-08 two live turns got every GIS number right and still showed the
user false alarms, a lost record, memorised figures, and a 27-call geocoding loop. Each was fixed
by a narrow patch that nothing re-ran against whole tasks. Architecture stage 41 has the record.

## Run it

```bash
# 4 tasks (T01 fetch+measure, T02 vector upload, T08 raster, U02 refusal), one model
python -m gis_harness.run --start-server 5301 --model lumen:deepseek-v4-flash --tasks sample

# everything, two models on one server, two tasks in flight per model
python -m gis_harness.run --start-server 5301 --tasks all --parallel 2 \
    --model lumen:deepseek-v4-flash --model openai:gpt-5.6-luna --label baseline-81fd7c8

# chosen tasks, three trials each (models vary run to run; one trial is an anecdote)
python -m gis_harness.run --start-server 5301 --model openai:gpt-5.6-luna --tasks T04,T09 --trials 3

# re-score a finished run after changing score.py (no model calls)
python -m gis_harness.run --summarise gis_harness/runs/baseline-81fd7c8
```

`--start-server` launches `api/server.py` from this checkout with `AGENT_MODE=local`, so the
run writes no conversation, snapshot or trace to shared infrastructure. It also blanks
`GOOGLE_MAPS_API_KEY` (KB spatial search geocodes through Google, a metered call no task needs),
and sets `AGENT_EXTRACTION=1` and `AGENT_INVARIANT_GATE=1`, as the deployment does. Either can be
overridden from the environment (`AGENT_EXTRACTION=0 python -m gis_harness.run ...`).

- `AGENT_EXTRACTION=1` binds the code peer's staging tools (`stage_url`, `stage_element`,
  `list_staged_inputs`) and the method-library tools (`kb_method_search`, `get_method_contract`).
  **Every run before 2026-10-09 ran without them**: `baseline-4b066f6` through `p5-fixed-gate`
  and `deployed-23cfd02-gate`. Their code peer bound 62–68 tools, none of them `stage_url`. A
  task with no input that needs public data was therefore measured on a toolset production
  does not have. (`stack-live-cases` is the one exception: a separate driver that set the
  flag by hand.)
- Before launching, the harness builds the staging tools under the server's environment and
  refuses to start if the flag is on and `stage_url` is missing. After the run it reads every
  code-peer `turn_instrumentation_toolset` line in `server.log` into `toolsets.json`. It exits 2
  if any of them lacks `stage_url` while the flag is on.
- `AGENT_INVARIANT_GATE=1` was added earlier, in `33426a5`. Runs before it (the stage 41
  baseline, and the first after-runs of stages 42 and 43) had the gate off, so their scores say
  nothing about the gate.
- The Postgres read path (`AGENT_KB_DB`) is a separate switch and stays off, so
  `kb_method_search` reads the on-disk method library, if one is present.

`--base-url` uses a server you started yourself. Start it the same way.

Output: `gis_harness/runs/<label>/<provider>_<model>/<task>.json` (score, answer, tool calls,
tokens, cost) and `<task>.events.jsonl` (every SSE event), plus `summary.json`. `runs/` is
gitignored. Baselines worth keeping are copied to `baselines/`.

## Screenshots

`--screenshots` replays every turn of the run through the map UI at the end and writes
`<task>.png` beside its events file. The picture shows what the user would have seen: the answer,
its banners, the Downloads panel, and the map with the turn's layers. The replay is the page's own
client reading the recorded stream (see "Replaying a recorded turn" in
`map-ui-prototype/README.md`). For older runs, backfill:

```bash
python -m gis_harness.screenshots /abs/path/to/gis_harness_runs_archive          # skips turns that have a .png
python -m gis_harness.screenshots gis_harness/runs/<label> --force               # redo them
```

A turn's map layers are fetched by url from a server that is gone by the time anyone replays
it, so the harness saves each one while the server is up, as `layers/<task>__NN__<label>.<ext>`,
NN being its place among the turn's `map_layer` events. That is the layout
`gis_harness_runs_archive/tools/dump_layers.py` gave the older runs, and the replay serves from
it. `<task>.json` counts them under `layer_files`. A layer with no file is named on the
screenshot, in red. Each directory also gets `screenshots.jsonl`: per turn, layers drawn of
total, why any were not, seconds and bytes. It is `.jsonl` because `--summarise` and
`--compare` read every `.json` under a run as a turn.

Cost, measured on the 460-turn archive on 2026-10-09 (Mac, 4 pages at once, warm tile cache):
168 s for all 460, about 0.37 s of wall time per turn, plus a 4–5 s build once per call. A
page takes 0.9 s at the median: 0.8 s with no map, 2.6 s with layers, which waits for the
basemap tiles and the fit's ease to finish. The p90 is 2.7 s and the worst is 3.3 s. A PNG is
0.27 MB at the median (0.22 MB with no map, 0.97 MB with one) and 1.9 MB at most, 239 MB for
the archive. The largest process peaked at 0.9 GB.

What a replay changes, each named on the strip: agent-file urls are made host-relative, so a
turn recorded against agent.i-guide.io is served from the capture and never asks production.
A layer's geojson stored as JSON text by `chat_traces` is parsed back. A recording with no
answer event (live rounds pulled from `chat_traces`) gets its record's answer as the final
`result`. Files other than layers (inline images, downloads) are served only if the run kept
its `server_files/`, so a live round's images show as broken.

## What is scored

| | from | means |
|---|---|---|
| **correct** | answer text | every expected value appears, within tolerance, in any unit of the right dimension |
| **refused** | answer text | (unsolvable tasks) it says it cannot, AND it states no value for the impossible quantity |
| **productive** | tool events | no call repeats an earlier identical (tool, arguments) call, and no call failed |
| **clean** | answer text | no ⚠️/ℹ️ banner, no "COULD NOT VERIFY". A banner on a correct answer is a false alarm |
| **sourced** | answer text | the answer names where the data came from (a file, Census TIGER, OpenStreetMap, USGS) |
| **strict** | all of the above | right, with nothing wasted, nothing false, and the source named |

Scores for different models are reported separately and never pooled. The number matcher is
lenient about where a value sits in the prose: the agent writes for a person. Every match
records the text it matched, so a reader can check it.

## The tasks

| id | problem | data | the trap |
|---|---|---|---|
| T01 | area of Champaign County + London–Paris great circle | fetched by the agent (TIGER) + pinned coordinates | area in degrees or Web Mercator (1.8x at 40 N) |
| T02 | schools within 1 mile, nearest | synthetic points around Millennium Park | a Web Mercator buffer is 1.34x too long at 42 N |
| T02L | same, live OpenStreetMap | the agent's own fetch; reference is the harness's Overpass query | the source must be named; count tolerance is wide |
| T03 | incidents per 1,000 by zone | 3x3 zones + points | highest count is not highest rate |
| T04 | fastest path + 2-minute isochrone | synthetic street grid with speeds | shortest length is not fastest |
| T05 | 2-median location-allocation | 40 demand points, 8 candidates | brute force over 28 pairs; optimum unambiguous by >1% |
| T06 | mean slope + watershed area | V-valley DEM with a cross ridge | basin is exactly 120 of 200 rows by construction |
| T07 | bathtub inundation at 5 m | coastal ramp with a NoData block | counting NoData (-9999) as flooded adds 1 ha |
| T08 | NDVI change | two 2-band uint16 scenes | uint16 NIR−red wraps around; NoData pixels |
| T09 | Moran's I + Gi* hot spots | 10x10 lattice | checked against esda in the tests |
| T10 | IDW + ordinary kriging, Meuse zinc | Meuse (fetched by SHA-256) | variogram given, so the answer is a linear solve |
| T11 | multi-criteria suitability | three aligned rasters | land cover is categorical |
| T12 | aftershocks within 7 days and 25 km | USGS ComCat snapshot, Ridgecrest 2019 | distance in degrees; time window |
| T13 | violent crime per 1,000 by community area, Moran's I, 6 SKATER regions | Chicago 2024 crimes (259,660 rows), 77 community areas, ACS 2023: a City of Chicago Data Portal **snapshot pinned by SHA-256** (not in the repo; see below) | Fuller Park (2,186 residents) has the highest rate; regions are scored by their constraints (6 regions, all 77 areas, each region contiguous) read from the turn's map layers, because equally valid partitions differ |
| T14 | serious crashes per 500 m hexagon, Gi* hot spots, the streets in them, night share, crashes per year | Chicago 2020–2024 fatal and incapacitating crashes (10,076) and a 3,028-cell hexagon grid: a **snapshot pinned by SHA-256** | 40 hexagons have z within 0.06 of 1.96, so the hot-spot count (331) holds only under the stated weights; 68 crashes have no usable coordinates |
| T15 | which 6 of 82 library branches minimise population-weighted distance (exact 6-median), and how many residents each serves | Chicago Public Library branches and 2020 Census block-group centres of population (2,158): a **snapshot pinned by SHA-256** | the best single swap is 0.42% worse, so a near-optimum from a heuristic fails the 0.2% check on mean distance |
| T16 | residents in Waikīkī's tsunami evacuation zone, the walk out along OSM streets, buildings in the 3.2 ft sea-level-rise area | Hawaii Statewide GIS, Census 2020 blocks, OpenStreetMap, Honolulu building footprints: a **snapshot pinned by SHA-256** | a block is in the zone by its internal point (a representative point gives 20,291, not 20,294); two walks end at 10.04 and 10.06 min, so the count over 10 min (846) holds only under the stated snapping |
| U01 | slope of a DEM that is not attached | a points file | — |
| U02 | NDVI without a near-infrared band | a 1-band red scene | — |
| U03 | kriging mercury, which Meuse lacks | meuse.csv | — |
| U04 | travel time to a point 7 km off the network | roads.geojson | — |

Expected values are computed in `datasets.py` from the same bytes that are uploaded, by brute
force, a closed form, or a second library.

T13's data is a snapshot of three live datasets that the portal revises daily, so the snapshot,
not the URL, is the task's input. It is 35 MB and not committed. `datasets._chicago_snapshot`
looks in `gis_harness/.cache/chicago_crime_2024/`, then in `$GIS_HARNESS_SNAPSHOTS`, and refuses
any file whose SHA-256 differs from `CHICAGO_FILES`. A copy is kept in the git-excluded
`gis_harness_runs_archive/datasets/`. `python -m gis_harness.datasets --snapshot chicago_crime`
downloads the current data, which will no longer match the pinned hashes: a new snapshot means
new hashes and re-checked expected values. `tests/test_gis_harness.py` holds them and checks
that each trap still changes the answer.

T14–T16 are the questions of three Spatial Analysis examples, on the same files, pinned the same
way (`CRASH_FILES`, `COOLING_FILES`, `WAIKIKI_FILES`; directories of the same names under
`datasets/` in the archive). Their files were derived (filtered, clipped, a hexagon grid built
for T14), so no download reproduces them; `datasets.py` says what was fetched and what was done
to it. Each asks every numbered question of its example's steps in one turn, and none of its map
requests. Their expected values were computed twice: on 2026-10-15 by separate scripts (esda,
networkx, a first MILP), and by `datasets.py` (shared-vertex contiguity, scipy's Dijkstra, a
second MILP). The two agree on every value, and the tests pin them. The tests that need a
snapshot skip where it is absent. Building T16 takes about 20 s and T15 about 40 s (the exact
6-median).

A count check reads only numbers written without a unit: T16 asks how many residents need more
than 10 minutes and how many hotels (10) are in the sea-level-rise area, and "10 minutes" must
not answer the second. Re-scored, the archive's 151 count checks over 427 turns keep their
verdicts.

## How good are the checks? (`validators_bench`)

```bash
# every scored turn in the archive (default: gis_harness_runs_archive/ in the main checkout)
python3 -m gis_harness.validators_bench

# against a previous full run: each check's precision and recall delta
python3 -m gis_harness.validators_bench --baseline OLD/report.json

# the committed fixture subset, as CI runs it; after an intended change, commit the new numbers
python3 -m gis_harness.validators_bench --fixtures --check
python3 -m gis_harness.validators_bench --fixtures --update-expected
```

It replays the agent's validation checks (gate, number scan, audit reconciliation, corrections,
peer note, banner policy) on each archived turn's recorded state, with no model call. It scores
them against the harness's labels: correct, wrong, or an honest refusal. It reports each check's
precision as a "this answer is wrong" signal, its recall over wrong answers, and false alarms by
cause. Architecture stage 52 has the measurements and what the replay can and cannot re-run.

The replay rebuilds each turn's lineage (`agent_runtime/lineage.py`, stage 56) from its recorded
results. A bridge run's code was also shown the bridge's host notes, which the archive does not
hold; recover them once from the commit each run names in its `.origin`:

```bash
python3 -m gis_harness.run_notes        # writes <run>/tool_notes.json for AGENT_CODE_BRIDGE=1 runs
```

## Data and licences

- Synthetic datasets are generated at run time from fixed seeds.
- `data/usgs_comcat_ridgecrest_2019-07.csv`: USGS ComCat, M≥3 within 100 km of 35.77 N,
  117.60 W, July 2019, fetched 2026-10-08. US Government work, public domain.
- Meuse: from scikit-gstat (MIT) at a pinned commit, checked by SHA-256 and cached in `.cache/`.
  Not committed.
- T01's area reference: Census TIGERweb `State_County/MapServer/1`, GEOID 17019, AREALAND +
  AREAWATER, read 2026-10-08.
- T02L's reference is an Overpass query made at run time. It is the one task whose expected
  value can change between runs.
- T13–T15: City of Chicago Data Portal (its terms of use) and the US Census Bureau (public
  domain). Not committed.
- T16: Hawaii Statewide GIS Program (tsunami evacuation zones; the 3.2 ft SLR exposure area by
  SOEST, UH CGG, PacIOOS and Tetra Tech), the US Census Bureau, City and County of Honolulu
  building footprints (DPP HoLIS), and OpenStreetMap (© OpenStreetMap contributors, ODbL: any
  published copy must say so). Not committed.

## Cost

Each `<task>.json` records every model call's provider token counts, including the
supervisor's decider, synthesis and audit, not only the peers (`llm_usage` trace events). Prices
come from `agent_runtime/model_pricing.py`, the table the server prices live turns with. A model with no per-token price (Lumen) reports `cost_usd: null`, not
0. See the stage 41 entry in `docs/agent-architecture-changes.md` for measured per-task costs.
