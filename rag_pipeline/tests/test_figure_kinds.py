"""Stage 54: what a figure IS, read from how the code computed it, and checks that fire only
when they know.

The validators bench (stage 52) found every count-check and declared-unit finding to be a false
alarm, each a number whose kind the check guessed from a declaration or a frame inventory. The
cases below are the archived runs' own code, reduced: T08L's raster pixel count and NDVI mean,
T11L's patch list, T03L's tract population, T05L's population sum and `.area > 0`, T01's
exploratory `print(gdf.area)`, T02L's empty OpenStreetMap tags, and U04's degree distance, the
one projected_crs catch on a wrong answer, which must survive.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("pint")

from agent_runtime import declared_outputs, figure_kinds as FK, sandbox_verify as sv


def _kind(code: str, name: str = "x") -> FK.Kind:
    return FK.declared_kinds(code + f"\nIGUIDE_OUTPUTS = {{'{name}': {{'value': {name}}}}}\n")[name]


HEAD = "import numpy as np\nimport geopandas as gpd\nimport pandas as pd\n"


# --------------------------------------------------------------------------- the kinds

@pytest.mark.parametrize("code,kind,counts,of", [
    ("tracts = gpd.read_file('t.geojson')\nx = len(tracts)", "count", "rows", "tracts"),
    ("tracts = gpd.read_file('t.geojson')\nx = tracts.shape[0]", "count", "rows", "tracts"),
    ("df = pd.read_csv('c.csv')\nx = int((df['dist'] < 1000).sum())", "count", "rows", "df"),
    ("df = pd.read_csv('c.csv')\nx = df['dist'].isna().sum()", "count", "rows", "df"),
    ("df = pd.read_csv('c.csv')\nnear = df[df['dist'] < 1000]\nx = len(near)", "count", "rows",
     "near"),
    ("df = pd.read_csv('c.csv')\nx = df['type'].value_counts()['school']", "count", "rows", "df"),
    ("df = pd.read_csv('c.csv')\nx = df.groupby('tract').size().max()", "count", "rows", "df"),
    ("df = pd.read_csv('c.csv')\nx = df['type'].nunique()", "count", "distinct", "df"),
    ("valid = np.isfinite(ndvi)\nx = int(valid.sum())", "count", "elements", None),
    ("x = np.count_nonzero(slope < 2)", "count", "elements", None),
    ("from scipy import ndimage\nlab, x = ndimage.label(mask)", "count", "features", None),
    ("patches = list(qual.geoms)\nx = len(patches)", "count", "items", "patches"),
    ("x = sum(1 for f in feats if f['ok'])", "count", "items", None),
])
def test_a_count_says_what_it_counts(code, kind, counts, of):
    k = _kind(HEAD + code)
    assert (k.kind, k.counts) == (kind, counts), k
    if of:
        assert k.of == of, k


@pytest.mark.parametrize("code,kind", [
    ("x = float(np.nanmean(ndvi))", "mean"),
    ("x = float(np.nanmedian(ndvi))", "median"),
    ("x = np.nanstd(ndvi)", "std"),
    ("df = pd.read_csv('c.csv')\nx = df['pop'].mean()", "mean"),
    ("df = pd.read_csv('c.csv')\nx = float(df['pop'].sum())", "sum"),
    ("df = pd.read_csv('c.csv')\nx = df['pop'].max()", "max"),
    ("x = np.percentile(z, 90)", "quantile"),
    ("g = gpd.read_file('a.geojson').to_crs(26916)\nx = g.geometry.area.sum() / 1e6", "area"),
    ("g = gpd.read_file('a.geojson').to_crs(26916)\nx = round(g.length.sum(), 1)", "length"),
    ("g = gpd.read_file('a.geojson')\nx = g.geometry.distance(p).min()", "distance"),
    ("x = qual.area / 10000.0", "area"),
    ("t = gpd.read_file('t.geojson')\nx = t.loc[3, 'POP100']", "value"),
    ("t = gpd.read_file('t.geojson')\ntop = t.sort_values('r').iloc[0]\nx = int(top['population'])",
     "value"),
    ("t = gpd.read_file('t.geojson')\ntop = t.head(10)\nx = int(top.iloc[0]['population'])", "value"),
    ("t = gpd.read_file('t.geojson')\nx = len(t[t.a > 0]) / len(t) * 100", "ratio"),
    ("t = gpd.read_file('t.geojson')\nx = len(t) / 1000", "ratio"),
    ("x = (mask).mean() if False else np.mean(mask > 0)", "unknown"),
    ("x = helper(a, b)", "unknown"),
])
def test_a_statistic_a_measurement_or_a_value_is_not_a_count(code, kind):
    assert _kind(HEAD + code).kind == kind


def test_the_kind_follows_names_through_the_nearest_binding():
    code = HEAD + ("t = gpd.read_file('t.geojson')\n"
                   "n = len(t)\n"
                   "n = t['pop'].sum()\n")
    assert _kind(code, "n").kind == "sum"


def test_a_frame_that_keeps_its_rows_is_stable():
    a = FK.analyze(HEAD + "t = gpd.read_file('t')\nt = t.to_crs(26916)\nu = gpd.read_file('u')\n"
                          "u = u[u.a > 0]\n")
    assert a.stable("t") and not a.stable("u")


def test_code_that_does_not_parse_has_no_kinds():
    assert FK.declared_kinds("def (:\n") == {}
    assert FK.declared_kinds(None) == {}


# --------------------------------------------------------------------------- declared counts

def _evaluate(declared, code, frames=None):
    rep = {"declared": [{"name": k, **v} for k, v in declared.items()],
           "frame_sizes": frames or {}}
    findings, typed = declared_outputs.evaluate(rep, code=code)
    return findings, {t["name"]: t for t in typed}


T08L = HEAD + '''
z = np.load("ndvi_bbox.npz")
ndvi = z["ndvi"]
gdf = gpd.read_file("Champaign_County.geojson")
ndvi_county = np.where(mask, ndvi, np.nan)
valid = np.isfinite(ndvi_county)
IGUIDE_OUTPUTS = {
    "mean_ndvi": {"value": float(np.nanmean(ndvi_county)), "unit": "NDVI (dimensionless)"},
    "valid_pixels": {"value": int(valid.sum()), "unit": "pixels"},
}
'''


def test_a_mean_declared_with_a_count_unit_is_typed_by_the_computation():
    """T08L: "NDVI (dimensionless)" parses as a count of NDVI, and 0.569 failed as a fractional
    count; 20,680 pixels "exceeded every frame" (one county polygon). Both banners were on a
    turn whose answer was wrong for another reason, and the same two fired on the correct T08L."""
    findings, typed = _evaluate({"mean_ndvi": {"value": 0.5687, "unit": "NDVI (dimensionless)"},
                                 "valid_pixels": {"value": 20680, "unit": "pixels"}},
                                T08L, frames={"gdf": 1})
    assert not [f for f in findings if f["status"] != "pass"], findings
    assert typed["mean_ndvi"]["figure_kind"]["kind"] == "mean"
    assert typed["mean_ndvi"]["declared_as"] == "count"
    assert typed["valid_pixels"]["figure_kind"]["counts"] == "elements"
    assert not [f for f in findings if f["check"] == "count_population"]


def test_a_count_of_rows_is_held_to_its_own_frame():
    code = HEAD + ("tracts = gpd.read_file('t.geojson').to_crs(26916)\n"
                   "inside = tracts[tracts.intersects(city)]\n"
                   "n = len(inside)\nIGUIDE_OUTPUTS = {'n': {'value': n, 'unit': 'tracts'}}\n")
    ok, _ = _evaluate({"n": {"value": 27, "unit": "tracts"}}, code, {"tracts": 48, "inside": 27})
    assert [f["status"] for f in ok if f["check"] == "count_population"] == ["pass"]
    bad, _ = _evaluate({"n": {"value": 60, "unit": "tracts"}}, code, {"tracts": 48, "inside": 27})
    fail = [f for f in bad if f["check"] == "count_population"]
    assert [f["status"] for f in fail] == ["fail"] and fail[0]["frames"] == {"tracts": 48}


def test_a_count_of_a_rebound_frame_is_not_compared():
    """The frame's size is read when the run ends; a frame filtered after the count was taken
    is smaller then, and comparing would be a false alarm."""
    code = HEAD + ("df = pd.read_csv('c.csv')\nn = len(df)\ndf = df[df.a > 0]\n"
                   "IGUIDE_OUTPUTS = {'n': {'value': n, 'unit': 'records'}}\n")
    findings, _ = _evaluate({"n": {"value": 500, "unit": "records"}}, code, {"df": 20})
    assert not [f for f in findings if f["check"] == "count_population"]


def test_an_unknown_kind_never_fails():
    code = "n = helper()\nIGUIDE_OUTPUTS = {'n': {'value': n, 'unit': 'schools'}}\n"
    for value in (2.5, -1, 10 ** 9):
        findings, typed = _evaluate({"n": {"value": value, "unit": "schools"}}, code, {"df": 3})
        assert {f["status"] for f in findings} == {"pass"}, findings
        assert typed["n"]["figure_kind"] == {"kind": "unknown"}


# --------------------------------------------------------------------------- reach

def _fail(op, line):
    return {"check": "projected_crs", "status": "fail", "target": line, "op": op,
            "message": f"'{op}' ran on a GEOGRAPHIC CRS in `{line}`"}


T01 = '''import geopandas as gpd
gdf = gpd.read_file("file_c4bc7495541a")
print(gdf.crs)
print(gdf.geometry.area)  # area in degrees^2 if EPSG:4326
gdf_proj = gdf.to_crs("EPSG:26916")
area_m2 = gdf_proj.geometry.area.iloc[0]
print(f"Area: {area_m2 / 1e6:.2f} km^2")
'''

U04 = '''import geopandas as gpd
from shapely.geometry import Point
gdf = gpd.read_file('file_51e202ce053e')
dest_pt = Point(-87.55, 41.95)
min_dist = min(gdf.geometry.distance(dest_pt))
print("Min distance from dest to nearest edge (degrees):", min_dist)
'''

T05L = '''import geopandas as gpd
tracts = gpd.read_file('Champaign_County_tracts.geojson')
tracts_in_city = tracts[tracts.intersects(city_geom)].copy()
tracts_in_city['geom_in_city'] = tracts_in_city.geometry.intersection(city_geom)
tracts_in_city = tracts_in_city[tracts_in_city['geom_in_city'].area > 0].copy()
print(len(tracts_in_city))
'''


def test_an_exploratory_print_before_a_reprojected_measurement_does_not_reach():
    reaches, why = FK.measurement_reach(T01, _fail("area", "print(gdf.geometry.area)  # area in "
                                                           "degrees^2 if EPSG:4326"))
    assert not reaches and "measured again in a projected CRS on line 6" in why


def test_a_printed_degree_distance_with_nothing_measured_again_reaches():
    """U04: the run's only distance, in degrees; the answer's "9.04 km" came from it. The one
    projected_crs catch the bench credits, on two wrong turns. It must stay."""
    reaches, _ = FK.measurement_reach(U04, _fail("distance", "min_dist = min(gdf.geometry."
                                                              "distance(dest_pt))"))
    assert reaches


def test_a_sign_test_does_not_reach_but_a_threshold_does():
    line = "tracts_in_city = tracts_in_city[tracts_in_city['geom_in_city'].area > 0].copy()"
    assert FK.measurement_reach(T05L, _fail("area", line)) == (False, "only compared with zero")
    threshold = T05L.replace(".area > 0]", ".area > 1000]")
    assert FK.measurement_reach(threshold, _fail("area", line.replace("> 0]", "> 1000]")))[0]


@pytest.mark.parametrize("tail,reaches", [
    ("a = g.area\n", False),                               # never used
    ("a = g.area\nprint(a.sum())\n", True),                # printed, nothing measured again
    ("a = g.area\nIGUIDE_OUTPUTS = {'a': {'value': a.sum()}}\n", True),
    ("g['a'] = g.area\ng.to_file('out.geojson')\n", True),  # stored in a column
    ("a = g.area\nif (a > 0).all():\n    print('ok')\n", False),
    ("a = g.area\nb = len(a)\nprint(b)\n", False),          # only counted
    ("a = g.area\nb = helper(a)\n", True),                  # into code it cannot follow
])
def test_where_a_measurement_goes(tail, reaches):
    code = "import geopandas as gpd\ng = gpd.read_file('a')\n" + tail
    line = tail.splitlines()[0]
    assert FK.measurement_reach(code, _fail("area", line))[0] is reaches


def test_a_utm_crs_estimated_by_geopandas_counts_as_projected():
    code = T01.replace('gdf.to_crs("EPSG:26916")', "gdf.to_crs(utm)").replace(
        "gdf_proj = ", "utm = gdf.estimate_utm_crs()\ngdf_proj = ")
    reaches, _ = FK.measurement_reach(code, _fail("area", "print(gdf.geometry.area)  # area in "
                                                          "degrees^2 if EPSG:4326"))
    assert not reaches


def test_a_reprojection_of_another_frame_does_not_excuse_this_one():
    code = T01.replace('gdf_proj = gdf.to_crs("EPSG:26916")',
                       'other = gpd.read_file("x")\ngdf_proj = other.to_crs("EPSG:26916")')
    reaches, _ = FK.measurement_reach(code, _fail("area", "print(gdf.geometry.area)  # area in "
                                                          "degrees^2 if EPSG:4326"))
    assert reaches


def test_a_line_the_code_does_not_hold_is_assumed_to_reach():
    assert FK.measurement_reach(T01, _fail("area", "x = some_library_line.area"))[0]


# --------------------------------------------------------------------------- through the gate

def _run(tmp_path, monkeypatch, code):
    """Through the assembled prologue and epilogue, and the agent-side read, as a run goes."""
    from agent_runtime.code_execution import _read_checks

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "script.py"
    path.write_text(sv.prologue_source({}) + code + sv.epilogue_source(), encoding="utf-8")
    exec(compile(path.read_text(), str(path), "exec"), {"__name__": "__main__"})
    return _read_checks(tmp_path, code=code), json.loads((tmp_path / "checks.json").read_text())


LIVE_T01 = '''import geopandas as gpd
from shapely.geometry import box
gdf = gpd.GeoDataFrame({"name": ["c"]}, geometry=[box(-88.46, 39.88, -87.93, 40.40)], crs=4326)
print(gdf.geometry.area)
gdf_proj = gdf.to_crs("EPSG:26916")
area_km2 = gdf_proj.geometry.area.sum() / 1e6
print(f"Area: {area_km2:.2f} km^2")
'''


def test_the_exploratory_print_passes_through_the_whole_path(tmp_path, monkeypatch):
    pytest.importorskip("geopandas")
    report, raw = _run(tmp_path, monkeypatch, LIVE_T01)
    assert any(f["check"] == "projected_crs" and f["status"] == "fail" for f in raw["findings"])
    assert report["verdict"] == "pass", report
    assert report["kind_notes"][0]["check"] == "projected_crs"


def test_the_degree_distance_still_fails_through_the_whole_path(tmp_path, monkeypatch):
    pytest.importorskip("geopandas")
    code = '''import geopandas as gpd
from shapely.geometry import Point
gdf = gpd.GeoDataFrame({"n": [1, 2]}, geometry=[Point(-87.65, 41.87), Point(-87.63, 41.89)],
                       crs=4326)
min_dist = min(gdf.geometry.distance(Point(-87.55, 41.95)))
print("Min distance (degrees):", min_dist)
'''
    report, _ = _run(tmp_path, monkeypatch, code)
    assert report["verdict"] == "fail"
    assert any(f.get("op") == "distance" for f in report["findings"])


def test_an_empty_tag_after_an_inner_join_passes_through_the_whole_path(tmp_path, monkeypatch):
    pytest.importorskip("geopandas")
    code = '''import geopandas as gpd
from shapely.geometry import Point, box
schools = gpd.GeoDataFrame({"name": ["A", "B"], "old_operator": [None, None]},
                           geometry=[Point(0.1, 0.1), Point(0.2, 0.2)], crs=4326)
buffer = gpd.GeoDataFrame({"r": [1]}, geometry=[box(0, 0, 1, 1)], crs=4326)
joined = gpd.sjoin(schools, buffer, how="inner", predicate="intersects")
print("schools:", len(joined))
'''
    report, raw = _run(tmp_path, monkeypatch, code)
    assert any(f["check"] == "all_nan" and f["status"] == "fail" for f in raw["findings"])
    assert not any(f["check"] == "all_nan" for f in report["findings"]), report
    read = code + "print(joined['old_operator'].fillna(0).sum())\n"
    report, _ = _run(tmp_path, monkeypatch, read)
    assert any(f["check"] == "all_nan" and f["status"] == "fail" for f in report["findings"])
