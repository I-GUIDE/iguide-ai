# GIS task harness

Whole-task regression runs for the agent: twelve classic GIS problems, one live-data variant,
and four questions that cannot be answered. Every task has an exact expected value, or a check
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
`GOOGLE_MAPS_API_KEY` (KB spatial search geocodes through Google, a metered call no task needs).
`--base-url` uses a server you started yourself. Start it the same way.

Output: `gis_harness/runs/<label>/<provider>_<model>/<task>.json` (score, answer, tool calls,
tokens, cost) and `<task>.events.jsonl` (every SSE event), plus `summary.json`. `runs/` is
gitignored. Baselines worth keeping are copied to `baselines/`.

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
| U01 | slope of a DEM that is not attached | a points file | — |
| U02 | NDVI without a near-infrared band | a 1-band red scene | — |
| U03 | kriging mercury, which Meuse lacks | meuse.csv | — |
| U04 | travel time to a point 7 km off the network | roads.geojson | — |

Expected values are computed in `datasets.py` from the same bytes that are uploaded, by brute
force, a closed form, or a second library. `tests/test_gis_harness.py` holds them and checks
that each trap still changes the answer.

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

## Cost

Each `<task>.json` records every model call's provider token counts, including the
supervisor's decider, synthesis and audit, not only the peers (`llm_usage` trace events). Prices
come from `pricing.json`. A model with no per-token price (Lumen) reports `cost_usd: null`, not
0. See the stage 41 entry in `docs/agent-architecture-changes.md` for measured per-task costs.
