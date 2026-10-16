"""Build the Demos tab: each case's input files and its catalogue entry.

    python3 map-ui-prototype/scripts/build_demos.py --chicago-snapshot DIR

A case either runs on a GIS harness task's own pinned data, or (task None) has no upload and
the agent fetches its own data (OpenStreetMap, USGS elevation). Its FIRST step states the task without
describing the data (user, 2026-10-15): the agent reads the files itself, and the "Explain files"
button asks it to describe them. `reliability` is from the scored archive, which ran the
harness's own, longer wording. Later steps are follow-ups that put the result on the map; the
harness does not score those.

Writes public/demos/<case>/<files> and src/demoCatalog.json. The data is regenerated, not edited:
the builders in gis_harness/datasets.py are deterministic (fixed seeds), so rerunning this gives
the same bytes. Chicago is the exception: its data is a snapshot of the City of Chicago Data
Portal, trimmed here to the violent incidents that have coordinates.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

UI = Path(__file__).resolve().parents[1]
REPO = UI.parent
sys.path.insert(0, str(REPO))

from gis_harness import datasets, tasks  # noqa: E402

# Measured over the scored archive (gis_harness_runs_archive, 2026-10-15): correct / runs, both
# models. Only tasks at 96% or better are demos; a demo that fails live teaches the wrong thing.
RELIABILITY = {
    "T01": "30 of 30 harness runs", "T02": "31 of 31 harness runs",
    "T03": "25 of 25 harness runs", "T04": "24 of 25 harness runs",
    "T05": "25 of 25 harness runs", "T07": "25 of 25 harness runs",
    "T08": "29 of 29 harness runs", "T11": "25 of 25 harness runs",
    "T12": "25 of 26 harness runs", "T13": "3 of 3 harness runs",
}

CASES = [
    {"id": "chicago-crime", "task": "T13",
     "title": "Chicago violent crime: rates, clusters and regions",
     "blurb": "Chicago recorded about 80,000 violent crimes in 2024, and they are far from evenly "
              "spread. This case compares the rate across the city's 77 community areas, asks whether "
              "high-rate areas sit next to each other, and groups the city into regions with similar "
              "rates.",
     "tags": ["spatial join", "rates", "autocorrelation", "regionalization", "choropleth"],
     "category": "Hypothesis testing",
     "steps": [
         {"label": "Rates by community area",
          "prompt": "Count the 2024 violent crimes in each Chicago community area, compute the rate"
                    " per 1,000 residents and map it as a choropleth. What is the citywide rate, "
                    "and which community area has the highest rate?"},
         {"label": "Are the rates clustered?",
          "prompt": "Test whether those rates are spatially clustered with global Moran's I "
                    "(queen contiguity, row-standardised weights), then map the local (LISA) "
                    "clusters: high-high, low-low and the outliers."},
         {"label": "Six crime regions",
          "prompt": "Group the 77 community areas into 6 contiguous regions with similar violent "
                    "crime rates using SKATER. Map the regions and give each one's community "
                    "areas, population and mean rate."},
     ]},
    {"id": "schools-buffer", "task": "T02",
     "title": "Schools within a mile of a site",
     "blurb": "A development site in the Chicago Loop and the schools around it. The case asks which "
              "schools are within a mile of the site and which one is closest.",
     "tags": ["buffer", "spatial selection", "distance"],
     "category": "Query and selection",
     "steps": [{"label": "Count and nearest",
                "prompt": "How many schools are within 1 mile of the site at 41.8827 N, 87.6233 W? "
                          "Which one is nearest, and how far is it in metres?"},
               {"label": "Show it on the map",
                "prompt": "Show the site, its 1-mile buffer and the schools inside it on the map."}]},
    {"id": "zone-rates", "task": "T03",
     "title": "Incidents per 1,000 residents by zone",
     "blurb": "Incidents recorded across nine zones with very different populations. The zone with the "
              "most incidents is not the one where an incident is most likely per resident.",
     "tags": ["spatial join", "rates", "choropleth"],
     "category": "Transformation",
     "steps": [{"label": "Rates by zone",
                "prompt": "Compute each zone's incident rate per 1,000 residents. Which zone has "
                          "the highest rate, and what is it?"},
               # Step 1 already draws the rates (a choropleth comes with the answer), so step 2
               # asks the question the case is built on instead of mapping it again.
               {"label": "Counts or rates?",
                "prompt": "Which zone has the most incidents? Why is it not the zone with the "
                          "highest rate, and how does the order of the zones change between "
                          "counts and rates?"}]},
    # No upload and no harness task: the agent fetches the Loop's streets from OpenStreetMap
    # (overpass_search feature='road network', PR #139; before it, 'highway' meant motorways only
    # and the network came back as 34 pieces). Replaces T04's synthetic 6x6 UTM grid, whose
    # correct route followed no real street. Live 2026-10-16 (deepseek-v4-flash): 1.71 min,
    # 1,378 m (Jackson Blvd east, Michigan Ave north); reference from the same 1,521 ways 1.713
    # min / 1,378.4 m; step 2: 262 intersections, reference 261. Overpass is a shared public
    # service and can be down for hours; the demo fails with it.
    {"id": "fastest-route", "task": None,
     "title": "Fastest drive across the Loop",
     "blurb": "Union Station to the Art Institute by car, on the Loop's real streets. With nothing "
              "to upload, the agent fetches the street network from OpenStreetMap, follows one-way "
              "streets and speed limits, and finds the fastest way across downtown Chicago.",
     "tags": ["network analysis", "routing", "OpenStreetMap", "no upload"],
     "category": "Optimisation",
     "steps": [{"label": "Fastest drive",
                "prompt": "What is the fastest driving time in minutes from Canal St & Jackson Blvd"
                          " (41.8780 N, 87.6393 W) to Michigan Ave & Adams St (41.8794 N, 87.6243 "
                          "W) in Chicago's Loop? Use OpenStreetMap's drivable streets inside "
                          "41.866–41.889 N, 87.645–87.618 W, follow one-way streets, and drive at "
                          "each street's posted maxspeed, or 30 mph where none is posted. Map the "
                          "route."},
               {"label": "Two minutes out",
                "prompt": "Following the same rules, how many street intersections (points where "
                          "three or more road segments meet) can be reached within 2 minutes of the"
                          " start? Map them."}]},
    {"id": "facility-siting", "task": "T05",
     "title": "Where to put two facilities",
     "blurb": "Forty neighbourhoods with known populations and eight possible sites for two new "
              "facilities. The case finds the pair of sites that keeps people closest to a facility "
              "overall.",
     "tags": ["location-allocation", "optimisation"],
     "category": "Optimisation",
     "steps": [{"label": "Choose the sites",
                "prompt": "Choose the 2 candidate sites that minimise the total population-weighted"
                          " straight-line distance from each demand point to its nearest chosen "
                          "site. Which two sites, and what is that total in person-kilometres?"},
               {"label": "Map the service areas",
                "prompt": "Map the demand points coloured by the chosen site that serves each "
                          "one, and mark the two chosen sites."}]},
    # No upload and no harness task: admin_boundary(city) -> dem_for_region(clip to the city) ->
    # inundation_at_level. Replaces T07's synthetic tile. The city boundary is ~80% open water,
    # which a bathtub calls flooded (asked about "the city", 98.7%); the question asks about
    # land on purpose and leaves the water to the agent (user, 2026-10-15). Reference from the
    # 3DEP frame the agent fetched (water <= 0 m): 110.7 km2 of 117.7 km2 land, 94.1%; 3 m 84.4%,
    # 2 m 60.4%. Live, the land question can run long (one run chased OSM water and coastline
    # for 14 minutes before it was stopped).
    {"id": "flood", "task": None,
     "title": "Galveston under a 5 m flood",
     "blurb": "Galveston, Texas sits on a low barrier island on the Gulf of Mexico and has been "
              "flooded by hurricane storm surges many times. With nothing to upload, the agent "
              "fetches the city's boundary and USGS elevation itself, and asks how much of the city"
              " lies at or below 5 m.",
     "tags": ["elevation", "inundation", "no upload"],
     "category": "Transformation",
     "steps": [{"label": "Land at or below 5 m",
                "prompt": "How much land in the city of Galveston, Texas lies at or below 5 m above"
                          " sea level? Use USGS elevation for the city boundary, count land only "
                          "(open water is not land), and treat it as a simple bathtub: every land "
                          "cell at or below 5 m counts. Give the area in km² and as a share of the "
                          "city's land."},
               {"label": "Lower water",
                "prompt": "How do the land area and the share change if the water reaches only 2 m,"
                          " or 3 m?"}]},
    {"id": "ndvi-change", "task": "T08",
     "title": "Vegetation change from two scenes",
     "blurb": "Two satellite scenes of the same area, from 2020 and 2024. The case measures how green "
              "the area was on each date and how much of it lost vegetation in between.",
     "tags": ["raster", "remote sensing", "change detection"],
     "category": "Transformation",
     "steps": [{"label": "NDVI and loss",
                "prompt": "Compute NDVI for 2020 and 2024. What is the mean NDVI on each date, and "
                          "how many hectares lost more than 0.2 NDVI between them?"},
               {"label": "Map the loss",
                "prompt": "Map where NDVI fell by more than 0.2 between the two dates."}]},
    {"id": "suitability", "task": "T11",
     "title": "Site suitability from three layers",
     "blurb": "Three aligned layers for one area: slope, distance to the nearest road, and land cover. "
              "The case finds the land that is flat, near a road and not water or wetland, and the "
              "single best spot.",
     "tags": ["raster", "multi-criteria"],
     "category": "Transformation",
     "steps": [{"label": "Score the cells",
                "prompt": "A cell is suitable if its slope is under 8 degrees, its road distance is"
                          " at most 500 m, and its land cover is not water or wetland. Score each "
                          "suitable cell as 0.5*(1 - slope/8) + 0.5*(1 - distance/500). What is the"
                          " total suitable area in hectares, and what are the centre coordinates of"
                          " the highest-scoring cell?"},
               {"label": "Map the score", "prompt": "Map the suitability score."}]},
    {"id": "aftershocks", "task": "T12",
     "title": "Aftershocks of the 2019 Ridgecrest earthquake",
     "blurb": "In July 2019 a magnitude 7.1 earthquake struck near Ridgecrest, California, followed by "
              "thousands of aftershocks. The case counts those in the following week within 25 km and "
              "maps them by magnitude.",
     "tags": ["spatio-temporal", "point data"],
     "category": "Query and selection",
     "steps": [{"label": "Count the aftershocks",
                "prompt": "What was the magnitude of the largest event in the July 2019 Ridgecrest "
                          "sequence? How many events occurred within 7 days after it and within 25 "
                          "km of its epicentre, and how many of those were magnitude 4 or greater?"},
               {"label": "Map them",
                "prompt": "Map the aftershocks within 25 km and 7 days, sized by magnitude, with "
                          "the mainshock marked."}]},
    {"id": "measure", "task": "T01",
     "title": "Measuring area and distance correctly",
     "blurb": "Two measurements with nothing to upload: the area of Champaign County, Illinois, whose "
              "boundary the agent fetches itself, and the distance from Trafalgar Square in London to "
              "Notre-Dame in Paris.",
     "tags": ["live data", "projections", "geodesic distance"],
     "category": "Measurement",
     "steps": [{"label": "Area and distance", "prompt": None}]},
]

CHICAGO_VIOLENT = ("HOMICIDE", "ASSAULT", "BATTERY", "ROBBERY", "CRIMINAL SEXUAL ASSAULT")


def chicago_files(snapshot: Path, dest: Path) -> list:
    import pandas as pd

    cr = pd.read_csv(snapshot / "crimes_2024.csv")
    v = cr[cr["primary_type"].isin(CHICAGO_VIOLENT)].dropna(subset=["latitude", "longitude"])
    v[["id", "date", "primary_type", "latitude", "longitude"]].to_csv(
        dest / "crimes_2024_violent.csv", index=False)
    for name in ("community_areas.geojson", "acs_2023_community_areas.csv"):
        shutil.copyfile(snapshot / name, dest / name)
    return ["crimes_2024_violent.csv", "community_areas.geojson", "acs_2023_community_areas.csv"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chicago-snapshot", type=Path, required=True,
                    help="the pinned snapshot directory (gis_harness/.cache/chicago_crime_2024)")
    a = ap.parse_args()
    out_root = UI / "public" / "demos"
    shutil.rmtree(out_root, ignore_errors=True)
    catalogue = []
    for case in CASES:
        dest = out_root / case["id"]
        dest.mkdir(parents=True)
        if case["task"] is None:
            # A live case: the agent fetches its own data, so there is nothing to build and
            # every step's question is written out here.
            names, first = [], None
        elif case["task"] == "T13":
            names = chicago_files(a.chicago_snapshot, dest)
            first = None
        else:
            task = tasks.BY_ID[case["task"]]
            files, expected = datasets.build(task.dataset, dest / "_build",
                                             REPO / "gis_harness" / ".cache")
            names = []
            for name, path in sorted(files.items()):
                shutil.copyfile(path, dest / name)
                names.append(name)
            shutil.rmtree(dest / "_build", ignore_errors=True)
            first = task.prompt(expected)
        steps = [{"label": s["label"], "prompt": s["prompt"] if s["prompt"] else first}
                 for s in case["steps"]]
        if not names:
            dest.rmdir()
        catalogue.append({
            "id": case["id"], "title": case["title"], "blurb": case["blurb"],
            "tags": case["tags"], "category": case["category"],
            **({"harnessTask": case["task"], "reliability": RELIABILITY[case["task"]]}
               if case["task"] else {}),
            "files": [{"name": n, "path": f"demos/{case['id']}/{n}",
                       "bytes": (dest / n).stat().st_size} for n in names],
            "steps": steps})
    (UI / "src" / "demoCatalog.json").write_text(json.dumps(catalogue, indent=1, ensure_ascii=False) + "\n")
    total = sum(f["bytes"] for c in catalogue for f in c["files"])
    print(f"{len(catalogue)} demos, {sum(len(c['files']) for c in catalogue)} files, {total / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
