"""Build the Demos tab: each case's input files and its catalogue entry.

    python3 map-ui-prototype/scripts/build_demos.py --chicago-snapshot DIR

Every case's FIRST step is a GIS harness task's own question, run on that task's own pinned data,
so a demo shows exactly what the harness measures and says how often it works (`reliability`,
from the scored archive). Later steps are follow-ups that put the result on the map; the harness
does not score those.

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
          "prompt": "The attached files are Chicago's 2024 violent crimes with coordinates "
                    "(crimes_2024_violent.csv: homicide, assault, battery, robbery and criminal "
                    "sexual assault), the 77 community areas (community_areas.geojson) and ACS "
                    "2023 population by community area (acs_2023_community_areas.csv). Count the "
                    "crimes in each community area by each incident's coordinates, compute the "
                    "rate per 1,000 residents and map it as a choropleth. What is the citywide "
                    "rate, and which community area has the highest rate?"},
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
     "steps": [{"label": "Count and nearest", "prompt": None},
               {"label": "Show it on the map",
                "prompt": "Show the site, its 1-mile buffer and the schools inside it on the map."}]},
    {"id": "zone-rates", "task": "T03",
     "title": "Incidents per 1,000 residents by zone",
     "blurb": "Incidents recorded across nine zones with very different populations. The zone with the "
              "most incidents is not the one where an incident is most likely per resident.",
     "tags": ["spatial join", "rates", "choropleth"],
     "category": "Transformation",
     "steps": [{"label": "Rates by zone", "prompt": None},
               {"label": "Map the rates",
                "prompt": "Map the rate per 1,000 residents for each zone as a choropleth."}]},
    {"id": "fastest-route", "task": "T04",
     "title": "Fastest route on a road network",
     "blurb": "A small grid of Chicago streets, each with its own speed. The shortest way between two "
              "intersections is not the fastest, and the case also asks how far you can get in two "
              "minutes.",
     "tags": ["network analysis", "routing"],
     "category": "Optimisation",
     "steps": [{"label": "Fastest travel time", "prompt": None},
               {"label": "Show the route", "prompt": "Show the fastest route on the map."}]},
    {"id": "facility-siting", "task": "T05",
     "title": "Where to put two facilities",
     "blurb": "Forty neighbourhoods with known populations and eight possible sites for two new "
              "facilities. The case finds the pair of sites that keeps people closest to a facility "
              "overall.",
     "tags": ["location-allocation", "optimisation"],
     "category": "Optimisation",
     "steps": [{"label": "Choose the sites", "prompt": None},
               {"label": "Map the service areas",
                "prompt": "Map the demand points coloured by the chosen site that serves each "
                          "one, and mark the two chosen sites."}]},
    {"id": "flood", "task": "T07",
     "title": "Land under a 5 m flood",
     "blurb": "An elevation model of a low-lying coastal tile, with a few cells of missing data. The "
              "case asks how much land lies at or below 5 m if the water rises that far.",
     "tags": ["raster", "inundation"],
     "category": "Transformation",
     "steps": [{"label": "Flooded area", "prompt": None},
               {"label": "Show the flood", "prompt": "Show the flooded cells on the map."}]},
    {"id": "ndvi-change", "task": "T08",
     "title": "Vegetation change from two scenes",
     "blurb": "Two satellite scenes of the same area, from 2020 and 2024. The case measures how green "
              "the area was on each date and how much of it lost vegetation in between.",
     "tags": ["raster", "remote sensing", "change detection"],
     "category": "Transformation",
     "steps": [{"label": "NDVI and loss", "prompt": None},
               {"label": "Map the loss",
                "prompt": "Map where NDVI fell by more than 0.2 between the two dates."}]},
    {"id": "suitability", "task": "T11",
     "title": "Site suitability from three layers",
     "blurb": "Three aligned layers for one area: slope, distance to the nearest road, and land cover. "
              "The case finds the land that is flat, near a road and not water or wetland, and the "
              "single best spot.",
     "tags": ["raster", "multi-criteria"],
     "category": "Transformation",
     "steps": [{"label": "Score the cells", "prompt": None},
               {"label": "Map the score", "prompt": "Map the suitability score."}]},
    {"id": "aftershocks", "task": "T12",
     "title": "Aftershocks of the 2019 Ridgecrest earthquake",
     "blurb": "In July 2019 a magnitude 7.1 earthquake struck near Ridgecrest, California, followed by "
              "thousands of aftershocks. The case counts those in the following week within 25 km and "
              "maps them by magnitude.",
     "tags": ["spatio-temporal", "point data"],
     "category": "Query and selection",
     "steps": [{"label": "Count the aftershocks", "prompt": None},
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
        if case["task"] == "T13":
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
            "tags": case["tags"], "category": case["category"], "harnessTask": case["task"],
            "reliability": RELIABILITY[case["task"]],
            "files": [{"name": n, "path": f"demos/{case['id']}/{n}",
                       "bytes": (dest / n).stat().st_size} for n in names],
            "steps": steps})
    (UI / "src" / "demoCatalog.json").write_text(json.dumps(catalogue, indent=1, ensure_ascii=False) + "\n")
    total = sum(f["bytes"] for c in catalogue for f in c["files"])
    print(f"{len(catalogue)} demos, {sum(len(c['files']) for c in catalogue)} files, {total / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
