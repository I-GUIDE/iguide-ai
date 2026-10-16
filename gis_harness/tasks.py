"""The tasks: twelve classic GIS problems, four on real data, one live-data variant, and four
unsolvable ones.

A task is a user's question plus what a correct answer must contain. The prompt reads the way
a user would write it: it names the files and the method where a GIS analyst would expect to
be told (a variogram, a weight scheme), and leaves the rest to the agent. It never names a tool.

Checks are mechanistic (see score.py): a value within tolerance, a named id present, a point
within half a cell. Expected values come from datasets.py, computed independently of the agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class Check:
    name: str                       # key into the task's expected values
    kind: str = "number"            # number | count | text | ids | point | regions
    unit: Optional[str] = None      # expected unit, for number checks (see score.UNITS)
    rel_tol: float = 0.0
    abs_tol: float = 0.0
    alternatives: Sequence[str] = ()  # extra expected keys for `point` (e.g. a UTM pair)


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    dataset: str
    prompt: Callable[[Dict[str, Any]], str]
    checks: Sequence[Check] = ()
    sources: Sequence[str] = ()     # naming any one of these counts as naming the source
    solvable: bool = True
    fabrication: Optional[str] = None  # regex: a value of the impossible quantity, if stated
    live: bool = False              # expected value is fetched at run time, not pinned
    tags: Sequence[str] = field(default_factory=tuple)


def _ll(p) -> str:
    lon, lat = p
    return f"{abs(lat):.5f}° {'N' if lat >= 0 else 'S'}, {abs(lon):.5f}° {'W' if lon < 0 else 'E'}"


LIVE_SCHOOL_SITE = (41.9231, -87.7093)  # Logan Square, Chicago

TASKS: List[Task] = [
    Task(
        "T01", "CRS-aware area + great-circle distance", "area_distance",
        lambda e: ("What is the area of Champaign County, Illinois, in square kilometres? And "
                   "what is the great-circle distance in km between Trafalgar Square, London "
                   "(51.5080 N, 0.1281 W) and Notre-Dame de Paris (48.8530 N, 2.3499 E)?"),
        checks=(Check("area_km2", unit="km2", rel_tol=0.01),
                Check("distance_km", unit="km", rel_tol=0.005)),
        sources=("Census", "TIGER"), tags=("fetch", "measure", "sample")),
    Task(
        "T02", "Buffer/overlay: schools within a mile (pinned upload)", "schools",
        lambda e: ("The attached schools.geojson has school locations. How many schools are "
                   "within 1 mile of the site at 41.8827 N, 87.6233 W? Which one is nearest, "
                   "and how far is it in metres?"),
        checks=(Check("count_within_mile", kind="count"),
                Check("nearest_school", kind="text"),
                Check("nearest_m", unit="m", rel_tol=0.01)),
        sources=("schools.geojson", "attached"), tags=("upload", "vector", "sample")),
    Task(
        "T02L", "Buffer/overlay: schools within a mile (live OpenStreetMap)", "live_osm_schools",
        lambda e: (f"How many schools does OpenStreetMap have within 1 mile of "
                   f"{LIVE_SCHOOL_SITE[0]} N, {abs(LIVE_SCHOOL_SITE[1])} W?"),
        checks=(Check("count_within_mile", kind="count", rel_tol=0.15, abs_tol=2),),
        sources=("OpenStreetMap", "OSM", "Overpass"), live=True, tags=("fetch", "vector")),
    Task(
        "T03", "Spatial join + rates", "zones_rates",
        lambda e: ("zones.geojson holds 9 zones with a population field, and incidents.csv "
                   "holds incident locations (lat/lon). Count the incidents in each zone and "
                   "compute the rate per 1,000 residents. Which zone has the highest rate, and "
                   "what is it?"),
        checks=(Check("top_zone", kind="text"),
                Check("top_rate_per_1000", rel_tol=0.01)),
        sources=("zones.geojson", "incidents.csv", "attached"), tags=("upload", "vector")),
    Task(
        "T04", "Network: fastest path + isochrone", "road_network",
        lambda e: ("roads.geojson is a road network: each line is an edge between two "
                   "intersections, with a speed_kph attribute. Travel time on an edge is its "
                   "length divided by its speed. What is the fastest travel time in minutes "
                   f"from the intersection at {_ll(e['origin'])} to the one at "
                   f"{_ll(e['destination'])}? How many intersections, counting the start, can "
                   "be reached within 2 minutes of it?"),
        checks=(Check("fastest_minutes", unit="min", rel_tol=0.015),
                Check("nodes_within_iso", kind="count")),
        sources=("roads.geojson", "attached"), tags=("upload", "network")),
    Task(
        "T05", "Location-allocation: 2-median", "p_median",
        lambda e: ("demand.csv has 40 demand points with populations, and candidates.csv has 8 "
                   "candidate facility sites. Choose the 2 sites that minimise the total "
                   "population-weighted straight-line distance from each demand point to its "
                   "nearest chosen site. Which two sites, and what is that total in "
                   "person-kilometres?"),
        checks=(Check("sites", kind="ids"),
                Check("objective_person_km", unit="person_km", rel_tol=0.01)),
        sources=("demand.csv", "candidates.csv", "attached"), tags=("upload", "optimisation")),
    Task(
        "T06", "DEM: slope + watershed", "dem_watershed",
        lambda e: ("dem.tif is a 30 m elevation model. What is its mean slope in degrees? And "
                   "what is the area, in km², of the watershed draining to the outlet at "
                   f"{_ll(e['outlet_lonlat'])} (UTM 16N x={e['outlet_utm'][0]:.0f}, "
                   f"y={e['outlet_utm'][1]:.0f})?"),
        checks=(Check("mean_slope_deg", unit="deg", rel_tol=0.03),
                Check("watershed_km2", unit="km2", rel_tol=0.03)),
        sources=("dem.tif", "attached"), tags=("upload", "raster", "hydrology")),
    Task(
        "T07", "Inundation (bathtub)", "inundation",
        lambda e: ("coast_dem.tif is an elevation model in metres, with the sea along its "
                   "southern edge. If water rises to 5 m, how many hectares are at or below that "
                   "level? Treat it as a simple bathtub: every cell at or below 5 m counts."),
        checks=(Check("flooded_ha", unit="ha", rel_tol=0.01),),
        sources=("coast_dem.tif", "attached"), tags=("upload", "raster")),
    Task(
        "T08", "NDVI change detection", "ndvi_change",
        lambda e: ("scene_2020.tif and scene_2024.tif cover the same area on two dates; band 1 "
                   "is red and band 2 is near-infrared. Compute NDVI for each date. What is the "
                   "mean NDVI on each date, and how many hectares lost more than 0.2 NDVI "
                   "between them?"),
        checks=(Check("mean_ndvi_2020", abs_tol=0.005),
                Check("mean_ndvi_2024", abs_tol=0.005),
                Check("loss_ha", unit="ha", rel_tol=0.02)),
        sources=("scene_2020.tif", "scene_2024.tif", "attached"),
        tags=("upload", "raster", "sample")),
    Task(
        "T09", "Spatial autocorrelation: Moran's I + Gi*", "lattice_autocorrelation",
        lambda e: ("lattice.geojson is a grid of cells with a 'value' field. Compute global "
                   "Moran's I for value using queen contiguity with row-standardised weights. "
                   "Then compute Getis-Ord Gi* with binary queen weights, each cell included in "
                   "its own neighbourhood: how many cells are hot spots with z > 1.96?"),
        checks=(Check("morans_i", abs_tol=0.005),
                Check("hot_spots", kind="count")),
        sources=("lattice.geojson", "attached"), tags=("upload", "statistics")),
    Task(
        "T10", "Interpolation: IDW + ordinary kriging (Meuse)", "meuse",
        lambda e: ("meuse.csv holds the Meuse river soil samples (x, y in metres, Dutch RD New, "
                   "EPSG:28992; zinc in ppm). Estimate zinc at x=179500, y=331500 two ways: "
                   "(a) inverse distance weighting with power 2 using all samples, and (b) "
                   "ordinary kriging of log(zinc) with a spherical variogram (nugget 0.05, "
                   "partial sill 0.59, range 897 m) using all samples, back-transformed with "
                   "exp()."),
        checks=(Check("idw_zinc", unit="ppm", rel_tol=0.01),
                Check("ok_zinc", unit="ppm", rel_tol=0.02)),
        sources=("meuse.csv", "Meuse", "attached"), tags=("upload", "statistics")),
    Task(
        "T11", "Multi-criteria suitability", "suitability",
        lambda e: ("slope_deg.tif, road_distance_m.tif and landcover.tif are aligned 10 m "
                   "rasters. A cell is suitable if its slope is under 8 degrees, its road "
                   "distance is at most 500 m, and its land cover is not 1 (water) or 2 "
                   "(wetland). Score each suitable cell as 0.5*(1 - slope/8) + "
                   "0.5*(1 - distance/500). What is the total suitable area in hectares, and "
                   "what are the centre coordinates of the highest-scoring cell?"),
        checks=(Check("suitable_ha", unit="ha", rel_tol=0.01),
                Check("best_cell_lonlat", kind="point", abs_tol=0.0002,
                      alternatives=("best_cell_utm:15",))),
        sources=("slope_deg.tif", "road_distance_m.tif", "landcover.tif", "attached"),
        tags=("upload", "raster")),
    Task(
        "T12", "Space-time: earthquake aftershocks", "earthquakes",
        lambda e: ("usgs_comcat_ridgecrest_2019-07.csv is a USGS earthquake catalog. What was "
                   "the largest event's magnitude? How many events occurred within 7 days after "
                   "it and within 25 km of its epicentre, and how many of those were magnitude "
                   "4 or greater?"),
        checks=(Check("mainshock_mag", abs_tol=0.05),
                Check("aftershocks_7d_25km", kind="count", abs_tol=1),
                Check("aftershocks_m4_7d_25km", kind="count")),
        sources=("USGS", "ComCat", "usgs_comcat_ridgecrest"), tags=("upload", "space-time")),
    # ------------------------------------------------------------------ unsolvable
    Task(
        "T13", "Chicago violent crime: rates, clustering, regionalization", "chicago_crime",
        lambda e: ("The attached files are Chicago's 2024 reported crimes (crimes_2024.csv), the 77 "
                   "community areas (community_areas.geojson) and ACS 2023 population by "
                   "community area (acs_2023_community_areas.csv). Count the violent crimes "
                   "(homicide, assault, battery, robbery and criminal sexual assault) in each "
                   "community area by each incident's coordinates, and compute the rate per 1,000 "
                   "residents. What is the citywide rate, and which community area has the "
                   "highest rate? Test whether the rates are spatially clustered with Moran's I "
                   "(queen contiguity, row-standardised weights). Then group the community areas "
                   "into 6 contiguous regions with similar violent crime rates (SKATER), and map "
                   "the rates and the regions."),
        checks=(Check("citywide_rate", rel_tol=0.01),
                Check("top_area", kind="text"),
                Check("top_rate", rel_tol=0.02),
                Check("morans_i", abs_tol=0.01),
                Check("regions", kind="regions")),
        sources=("crimes_2024.csv", "attached", "Chicago Data Portal", "City of Chicago"),
        tags=("upload", "vector", "regionalization")),
    # T14-T16 are the Spatial Analysis examples' questions on their real data, one turn each:
    # every numbered question of the example's steps, none of its map requests.
    Task(
        "T14", "Chicago serious crashes: counts, Gi* hot spots, streets, time", "serious_crashes",
        lambda e: ("chicago_serious_crashes_2020_2024.csv lists every crash in Chicago from 2020 to "
                   "2024 that killed or seriously injured someone (City of Chicago Data Portal), and "
                   "hex_grid_500m.geojson is a grid of 500 m hexagons over the city. Count the "
                   "crashes in each hexagon: how many crashes fall in the grid, and which hexagon "
                   "has the most, with how many? Run Getis-Ord Gi* on the counts with binary queen "
                   "contiguity, each hexagon in its own neighbourhood, and analytical z-scores: how "
                   "many hexagons are hot spots (z > 1.96), and which hexagon has the highest z, and "
                   "what is it? Using the street recorded with each crash (its street name; W and E "
                   "Madison St are one street), which three streets have the most crashes inside "
                   "the hot-spot hexagons, and how many each? What share of "
                   "all the crashes in the file happened at night, between 10 p.m. and 4 a.m.? And "
                   "over all the crashes in the file, how many were there per year on average in "
                   "2020–2022 and in 2023–2024?"),
        checks=(Check("crashes_in_grid", kind="count"),
                Check("top_count_hex", kind="text"),
                Check("top_count", kind="count"),
                Check("hot_cells", kind="count"),
                Check("top_z_hex", kind="text"),
                Check("top_z", abs_tol=0.01),
                Check("hot_streets", kind="ids"),
                Check("street_1_crashes", kind="count"),
                Check("street_2_crashes", kind="count"),
                Check("street_3_crashes", kind="count"),
                Check("night_pct", abs_tol=0.05),
                Check("per_year_2020_2022", abs_tol=0.1),
                Check("per_year_2023_2024", abs_tol=0.1)),
        sources=("chicago_serious_crashes_2020_2024.csv", "attached", "Chicago Data Portal",
                 "City of Chicago"),
        tags=("upload", "vector", "statistics", "real-data")),
    Task(
        "T15", "Library cooling centres: exact 6-median on census population", "cooling_centres",
        lambda e: ("cpl_branches.csv lists Chicago Public Library's 82 branches with coordinates, "
                   "and chicago_blockgroup_population_centres_2020.csv has the 2020 Census centre of "
                   "population and the population of each block group in the city. In a heat "
                   "emergency the city can keep only 6 branches open late as cooling centres. Which "
                   "6 minimise the total population-weighted straight-line distance from each block "
                   "group to its nearest open branch? Measure distance in metres in UTM zone 16N "
                   "(EPSG:26916) and solve it exactly (it is a 6-median problem). Report the 6 "
                   "branches, the average distance per resident in km, the share of residents more "
                   "than 2 km from their nearest open branch, and how many residents each of the 6 "
                   "branches serves."),
        # The best single swap away from the optimum is 0.42% worse (best_swap_ratio), so the
        # mean distance is held to 0.2%.
        checks=(Check("branches", kind="ids"),
                Check("mean_km", unit="km", rel_tol=0.002),
                Check("share_over_2km_pct", abs_tol=0.1),
                *(Check(f"served_{i}", kind="count") for i in range(1, 7))),
        sources=("cpl_branches.csv", "chicago_blockgroup_population_centres_2020.csv", "attached",
                 "Census", "Chicago Public Library"),
        tags=("upload", "optimisation", "real-data")),
    Task(
        "T16", "Waikīkī: tsunami zone population, walking out, sea-level rise", "waikiki_tsunami",
        lambda e: ("tsunami_evacuation_zone.geojson is Oʻahu's official tsunami evacuation zone "
                   "(Hawaii Statewide GIS Program), clipped to Waikīkī, the Ala Wai and McCully. "
                   "census_blocks_2020.geojson has the 2020 Census blocks there with population "
                   "(POP100), housing units (HU100) and each block's internal point (INTPTLAT, "
                   "INTPTLON). osm_hotels.geojson and osm_streets.geojson are OpenStreetMap's hotels "
                   "and streets, slr_exposure_3_2ft.geojson is the State of Hawaiʻi's 3.2 ft "
                   "sea-level-rise exposure area (SLR-XA), and buildings.geojson has the building "
                   "footprints. Counting a block as inside the zone when its internal point is, how "
                   "many residents and housing units are in the evacuation zone, and how many of the "
                   "hotels are in it? If the people in the zone walked out: use osm_streets.geojson "
                   "as an undirected walking network in UTM zone 4N (EPSG:32604); streets meet only "
                   "where they share a vertex, as OpenStreetMap ways do (a bridge does not meet the "
                   "road it crosses). Snap each "
                   "populated block inside the zone (by its internal point) to its nearest street "
                   "node and find the shortest walking distance along the streets to the nearest "
                   "street node outside the zone. At 1.1 m/s, what is the population-weighted median "
                   "walking time, which block takes longest and how long, and how many residents "
                   "need more than 10 minutes? Finally, how many of the buildings intersect the "
                   "sea-level-rise exposure area, how many have their centroid inside it, and how "
                   "many of the hotels are inside it?"),
        # Two walks end within 0.1 min of 10 (10.04, 10.06; see _why), so pop_over_10min is exact
        # only under the stated method: that is the point of the check. Noding every crossing
        # (unary_union) joins bridges to the roads under them and gives 731, with the same median
        # and maximum (run 2, 2026-10-16), hence the sentence on where streets meet. The intersect count is 909
        # in UTM 4N and 911 in WGS84; either is right.
        checks=(Check("zone_pop", kind="count"),
                Check("zone_hu", kind="count"),
                Check("hotels_in_zone", kind="count"),
                Check("walk_median_min", unit="min", abs_tol=0.05),
                Check("walk_max_block", kind="text"),
                Check("walk_max_min", unit="min", rel_tol=0.01),
                Check("pop_over_10min", kind="count"),
                Check("slr_buildings_intersect", kind="count", abs_tol=2),
                Check("slr_buildings_centroid", kind="count"),
                Check("hotels_in_slr", kind="count")),
        sources=("tsunami_evacuation_zone.geojson", "census_blocks_2020.geojson", "attached",
                 "Hawaii Statewide GIS", "OpenStreetMap", "Census"),
        tags=("upload", "vector", "network", "real-data")),
    Task(
        "U01", "Unsolvable: slope of a DEM that is not attached", "schools",
        lambda e: "What is the mean slope, in degrees, of the elevation model I attached?",
        solvable=False, sources=("schools.geojson", "attached"),
        fabrication=r"\b\d+(?:\.\d+)?\s*(?:°|deg(?:rees?)?\b)", tags=("refusal",)),
    Task(
        "U02", "Unsolvable: NDVI without a near-infrared band", "single_band_red",
        lambda e: "Compute the mean NDVI of the attached scene.",
        solvable=False, sources=("scene_red_only.tif", "attached"),
        fabrication=r"NDVI[^\n]{0,60}?(?:is|of|=|:)\s*\**\s*[-−]?0?\.\d{2,}",
        tags=("refusal", "sample")),
    Task(
        "U03", "Unsolvable: kriging a variable the data lack", "meuse",
        lambda e: ("Using the attached meuse.csv, krige the mercury concentration at "
                   "x=179500, y=331500."),
        solvable=False, sources=("meuse.csv", "attached"),
        fabrication=r"(?:mercury|Hg)[^\n]{0,80}?\b\d+(?:\.\d+)?\s*(?:ppm|mg)",
        tags=("refusal",)),
    Task(
        "U04", "Unsolvable: a destination off the network", "road_network",
        lambda e: ("roads.geojson is a road network (speed_kph per edge). What is the fastest "
                   f"travel time from the intersection at {_ll(e['origin'])} to the point "
                   "41.95000° N, 87.55000° W along this network?"),
        solvable=False, sources=("roads.geojson", "attached"),
        fabrication=r"\b\d+(?:\.\d+)?\s*min(?:utes?)?\b", tags=("refusal",)),
]

BY_ID = {t.id: t for t in TASKS}
SAMPLE = [t.id for t in TASKS if "sample" in t.tags]


def live_osm_schools(cache_dir=None) -> Dict[str, Any]:
    """The live variant's reference: an independent Overpass query, made by the harness.

    Counts distinct OSM objects tagged amenity=school (nodes, ways and relations) whose
    position (a way's centre) is within 1 mile. The agent may count campus polygons and their
    buildings differently, so the check's tolerance is wide. The purpose of this task is the
    source, not the exact count.
    """
    import requests

    lat, lon = LIVE_SCHOOL_SITE
    q = (f'[out:json][timeout:60];nwr["amenity"="school"](around:1609.344,{lat},{lon});'
         'out center tags;')
    import time

    last = None
    # The order stage 40 (#89) measured: z.overpass-api.de answered when the others did not.
    # Kept in step with rag_pipeline/search/overpass.py by hand: the reference must not depend
    # on the agent's own code.
    mirrors = ("https://z.overpass-api.de/api/interpreter",
               "https://overpass.kumi.systems/api/interpreter",
               "https://overpass.private.coffee/api/interpreter",
               "https://lz4.overpass-api.de/api/interpreter",
               "https://overpass-api.de/api/interpreter",
               "https://maps.mail.ru/osm/tools/overpass/api/interpreter")
    # Two passes with a pause: on 2026-10-08 all three refused once (504/500) and answered
    # minutes later.
    for attempt, url in enumerate([*mirrors, *mirrors]):
        if attempt == len(mirrors):
            time.sleep(20)
        try:
            r = requests.post(url, data={"data": q}, timeout=90,
                              headers={"User-Agent": "iguide-gis-harness/1"})
            r.raise_for_status()
            els = r.json()["elements"]
            return {"count_within_mile": len(els),
                    "_why": {"count_within_mile": f"Overpass {url} at run time, nwr amenity=school"}}
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"no Overpass mirror answered: {last}")
