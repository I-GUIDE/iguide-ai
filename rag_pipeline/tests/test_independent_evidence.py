"""Plan step 4: evidence the turn did not produce by checking itself.

4a, cross-run agreement: two runs in one turn that computed the same figure must agree.
4b, reference recomputation: a standard method recomputed with a reference library from the
run's own inputs must agree with what the run computed.

Each test holds a class, not an incident. The incidents are named where they came from: T10
(kriging, the nugget on the diagonal) and T09 (Gi*, row-standardised weights) in the scored
archive.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("pint")

from agent_runtime import agreement as AG
from agent_runtime import reference_checks as RC
from agent_runtime import verdict as V


# --------------------------------------------------------------------------- 4a names

@pytest.mark.parametrize("a,b", [("kriging_zinc", "krige_zinc"),
                                 ("kriging_logzinc", "krige_logz"),
                                 ("hot_spots", "hot_spot"),
                                 ("moransI", "morans_i")])
def test_names_that_spell_one_figure_two_ways_are_the_same_kind(a, b):
    assert AG.same_kind(AG.kind_key(a), AG.kind_key(b)), (a, b)


@pytest.mark.parametrize("a,b", [("idw_zinc", "kriging_zinc"), ("distance_m", "district_m"),
                                 ("zinc_mean", "zinc_mean_by_zone_west"),
                                 ("road_length", "river_length")])
def test_names_for_different_figures_are_not(a, b):
    ka, kb = AG.kind_key(a), AG.kind_key(b)
    assert not (ka and kb and AG.same_kind(ka, kb)), (a, b)


@pytest.mark.parametrize("name", ["total_area", "value", "result_2", "area_km2", "mean"])
def test_a_name_made_only_of_generic_words_names_no_kind(name):
    assert AG.kind_key(name) is None


# --------------------------------------------------------------------------- 4a the check

def _t10_facts(second=295.06):
    return [
        {"name": "idw_zinc", "value": 397.441, "unit": "ppm", "source": "execute_code",
         "call_id": "c1"},
        {"name": "kriging_zinc", "value": 309.531, "unit": "ppm", "source": "execute_code",
         "call_id": "c1"},
        {"name": "idw_zinc", "value": 397.44, "unit": "ppm", "source": "execute_code",
         "call_id": "c2"},
        {"name": "krige_zinc", "value": second, "unit": "ppm", "source": "execute_code",
         "call_id": "c2"},
    ]


def test_two_runs_that_disagree_on_one_figure_are_found():
    found = AG.disagreements(AG.figures(_t10_facts()))
    assert len(found) == 1
    assert {f.value for f in found[0].figures} == {309.531, 295.06}
    note = AG.reconcile_note(found)
    assert "309.531" in note and "295.06" in note and "DISAGREE" in note


def test_values_within_rounding_or_half_a_percent_agree():
    assert AG.disagreements(AG.figures(_t10_facts(second=309.5))) == []
    assert AG.disagreements(AG.figures(_t10_facts(second=310.9))) == []


def test_the_answer_that_reports_one_side_gets_a_finding_and_one_that_reports_both_does_not():
    found = AG.disagreements(AG.figures(_t10_facts()))
    one = AG.findings("(b) Ordinary kriging of log(zinc): **295.1 ppm**.", found, kind=V.PROBLEM)
    assert len(one) == 1 and one[0].check == "agreement" and "309.531" in one[0].message
    both = AG.findings("Kriging gave 295.1 ppm; an earlier run gave 309.5 ppm, which used "
                       "gamma(0)=0 and is right.", found, kind=V.PROBLEM)
    assert both == []
    neither = AG.findings("IDW gives 397.4 ppm.", found, kind=V.PROBLEM)
    assert neither == []


def test_measured_figures_and_disjoint_inputs_and_unit_mismatch_stay_silent():
    measured = [{"name": "area sum", "value": 10.0, "unit": "km**2", "source": "measured",
                 "call_id": "c1"},
                {"name": "area sum", "value": 20.0, "unit": "km**2", "source": "measured",
                 "call_id": "c2"}]
    assert AG.disagreements(AG.figures(measured)) == []
    facts = [{"name": "watershed_area", "value": 16.2, "unit": "km**2", "call_id": "c1"},
             {"name": "watershed_area", "value": 31.0, "unit": "km**2", "call_id": "c2"}]
    calls = [{"call_id": "c1", "args": {"input_files": ["file_a"]}},
             {"call_id": "c2", "args": {"input_files": ["file_b"]}}]
    assert AG.disagreements(AG.figures(facts, calls)) == []
    assert len(AG.disagreements(AG.figures(facts))) == 1
    units = [{"name": "watershed_area", "value": 16.2, "unit": "km**2", "call_id": "c1"},
             {"name": "watershed_area", "value": 16.2, "unit": "km", "call_id": "c2"}]
    assert AG.disagreements(AG.figures(units)) == []


def test_an_answer_that_sides_with_the_runs_consensus_is_not_flagged():
    """T06/T07L in the archive: early attempts computed 0 or a broken area, later runs agreed,
    and the answer reported what they agreed on."""
    facts = [{"name": "watershed_area_km2", "value": v, "unit": "km**2", "call_id": f"c{i}"}
             for i, v in enumerate([0.0666, 16.2, 16.2, 16.2])]
    found = AG.disagreements(AG.figures(facts))
    assert len(found) == 1                               # the writer is still told
    assert AG.findings("The watershed is 16.2 km².", found, kind=V.PROBLEM) == []
    assert len(AG.findings("The watershed is 0.0666 km².", found, kind=V.PROBLEM)) == 1
    zeros = [{"name": "flooded_area", "value": v, "unit": "hectare", "call_id": f"c{i}"}
             for i, v in enumerate([0.0, 0.0, 0.0, 654.11, 654.11])]
    found = AG.disagreements(AG.figures(zeros))
    assert AG.findings("654.11 ha would flood.", found, kind=V.PROBLEM) == []


def test_permutation_p_values_are_not_compared():
    facts = [{"name": "morans_i_p", "value": 0.001, "unit": "p-value", "call_id": "c1"},
             {"name": "morans_i_p", "value": 0.0001, "call_id": "c2"},
             {"name": "p_sim", "value": 0.02, "call_id": "c3"},
             {"name": "p_sim", "value": 0.03, "call_id": "c4"}]
    assert AG.figures(facts) == []


def test_units_are_converted_before_comparing():
    facts = [{"name": "basin_area", "value": 16.2, "unit": "km**2", "call_id": "c1"},
             {"name": "basin_area", "value": 1620.0, "unit": "hectare", "call_id": "c2"}]
    assert AG.disagreements(AG.figures(facts)) == []


def test_the_turn_log_carries_the_typed_facts_to_the_check():
    from agent_runtime.turn_log import TurnLog

    log = TurnLog()
    for cid, name, value in (("c1", "kriging_zinc", 309.531), ("c2", "krige_zinc", 295.06)):
        content = json.dumps({"ok": True, "stdout": "", "verification": {"outputs": [
            {"name": name, "value": value, "unit": "ppm", "source": "execute_code"}]}})
        log.record_call(peer="code", run=None, name="execute_code",
                        args={"input_files": ["file_m"]}, call_id=cid)
        log.record_result(peer="code", run=None, name="execute_code",
                          args={"input_files": ["file_m"]}, call_id=cid, content=content)
    assert len(AG.for_log(log)) == 1


# --------------------------------------------------------------------------- 4b in the sandbox

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

_KRIGING_WITH_NUGGET_ON_THE_DIAGONAL = '''
import numpy as np
rng = np.random.default_rng(3)
xs = rng.uniform(0, 2000, 60); ys = rng.uniform(0, 2000, 60)
vals = np.log(rng.uniform(100, 1500, 60))
nug, ps, rg = 0.05, 0.59, 897.0
def sph(h):
    h = np.asarray(h, float)
    return nug + np.where(h <= rg, ps * (1.5 * h / rg - 0.5 * (h / rg) ** 3), ps)
D = np.hypot(xs[:, None] - xs[None, :], ys[:, None] - ys[None, :])
n = len(xs)
A = np.ones((n + 1, n + 1)); A[:n, :n] = sph(D) {DIAG}; A[n, n] = 0.0
d0 = np.hypot(xs - 1000.0, ys - 900.0)
b = np.ones(n + 1); b[:n] = sph(d0)
lam = np.linalg.solve(A, b)
est = float(np.dot(lam[:n], vals))
print(f"kriged log value: {est:.4f}")
print(f"back-transformed: {np.exp(est):.2f} ppm")
'''


def _run_in_gate(tmp_path: Path, code: str) -> dict:
    from agent_runtime import sandbox_verify as SV

    script = tmp_path / "script.py"
    script.write_text(SV.prologue_source() + "\n" + code + SV.epilogue_source())
    proc = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True,
                          text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = json.loads((tmp_path / SV.CHECKS_FILENAME).read_text())
    return {"stdout": proc.stdout, "report": report}


def _correct_ok(code_diag_zero: bool = True) -> float:
    rng = np.random.default_rng(3)
    xs = rng.uniform(0, 2000, 60)
    ys = rng.uniform(0, 2000, 60)
    vals = np.log(rng.uniform(100, 1500, 60))

    def sph(h):
        return 0.05 + np.where(h <= 897.0, 0.59 * (1.5 * h / 897.0 - 0.5 * (h / 897.0) ** 3),
                               0.59)
    D = np.hypot(xs[:, None] - xs[None, :], ys[:, None] - ys[None, :])
    A = np.ones((61, 61))
    A[:60, :60] = sph(D)
    np.fill_diagonal(A[:60, :60], 0.0)
    A[60, 60] = 0.0
    b = np.ones(61)
    b[:60] = sph(np.hypot(xs - 1000.0, ys - 900.0))
    return float(np.dot(np.linalg.solve(A, b)[:60], vals))


def test_a_hand_written_kriging_with_the_nugget_on_the_diagonal_is_recomputed(tmp_path):
    out = _run_in_gate(tmp_path, _KRIGING_WITH_NUGGET_ON_THE_DIAGONAL.replace("{DIAG}", ""))
    (ref,) = out["report"]["reference"]
    assert ref["status"] == "recomputed", ref
    assert ref["model"] == "spherical"
    assert ref["target"] == pytest.approx([1000.0, 900.0], abs=1e-3)
    assert ref["diagonal_max"] == pytest.approx(0.05)
    (cand,) = [c for c in ref["candidates"] if c["variable"] == "vals"]
    assert cand["reference"] == pytest.approx(_correct_ok(), rel=1e-6)
    assert abs(cand["agent"] - cand["reference"]) > 1e-3
    # Agent side: the figure is tied to what the run printed, and its exp() too.
    from agent_runtime.code_execution import _compact_reference

    body = {"ok": True, "stdout": out["stdout"],
            "verification": {"reference": [_compact_reference(ref)]}}
    (r,) = RC.from_sandbox(json.dumps(body), "c7")
    labels = {f.label for f in r.disagreeing()}
    assert labels == {"ordinary kriging of vals", "exp() of ordinary kriging of vals"}
    shown = f"{math.exp(cand['agent']):.1f}"
    (f,) = RC.findings(f"Ordinary kriging gives **{shown} ppm**.", [r], kind=V.NOTE)
    assert f.check == "reference" and "gamma(0) must be 0" in f.message


def test_a_correct_hand_written_kriging_agrees_with_the_reference(tmp_path):
    code = _KRIGING_WITH_NUGGET_ON_THE_DIAGONAL.replace(
        "{DIAG}", "; np.fill_diagonal(A[:n, :n], 0.0)")
    out = _run_in_gate(tmp_path, code)
    (ref,) = out["report"]["reference"]
    (cand,) = [c for c in ref["candidates"] if c["variable"] == "vals"]
    assert cand["agent"] == pytest.approx(cand["reference"], rel=1e-6)
    body = {"ok": True, "stdout": out["stdout"], "verification": {"reference": [ref]}}
    assert all(not r.disagreeing() for r in RC.from_sandbox(body, "c1"))


def test_a_linear_solve_that_is_not_kriging_is_not_recorded(tmp_path):
    out = _run_in_gate(tmp_path, "import numpy as np\n"
                                  "x = np.linalg.solve(np.eye(8) * 2.0, np.ones(8))\n")
    assert not out["report"].get("reference")


# --------------------------------------------------------------------------- 4b Gi* and Moran

gpd = pytest.importorskip("geopandas")
pytest.importorskip("esda")
pytest.importorskip("libpysal")


@pytest.fixture()
def t09_lattice(tmp_path):
    from gis_harness import datasets as D

    files, expected = D.lattice_autocorrelation(tmp_path)
    return files["lattice.geojson"], expected


def test_esda_gi_star_on_row_standardised_weights_is_found_in_agent_code(tmp_path, t09_lattice):
    path, expected = t09_lattice
    code = textwrap.dedent(f'''
        import geopandas as gpd, libpysal, warnings
        from esda.getisord import G_Local
        warnings.simplefilter("ignore")
        g = gpd.read_file({str(path)!r})
        w = libpysal.weights.fill_diagonal(libpysal.weights.Queen.from_dataframe(g), 1.0)
        gi = G_Local(g["value"].values, w, star=True, permutations=0)
        print("hot spots:", int((gi.Zs > 1.96).sum()))
    ''')
    out = _run_in_gate(tmp_path, code)
    (ref,) = out["report"]["reference"]
    assert ref["counts"]["1.96"] == [1, expected["hot_spots"]]
    (r,) = RC.from_sandbox({"ok": True, "stdout": out["stdout"],
                            "verification": {"reference": [ref]}}, "c1")
    (f,) = RC.findings("Gi* hot spots with z > 1.96: 1 cell.", [r], kind=V.NOTE)
    assert "13" in f.message
    assert RC.findings("13 cells have z > 1.96.", [r], kind=V.NOTE) == []


def test_esda_gi_star_with_binary_weights_agrees(tmp_path, t09_lattice):
    path, _ = t09_lattice
    code = textwrap.dedent(f'''
        import geopandas as gpd, libpysal, esda, warnings
        warnings.simplefilter("ignore")
        g = gpd.read_file({str(path)!r})
        w = libpysal.weights.Queen.from_dataframe(g)
        gi = esda.G_Local(g["value"].values, w, transform="B", star=True, permutations=0)
        mi = esda.Moran(g["value"].values, libpysal.weights.Queen.from_dataframe(g),
                        permutations=0)
        print(int((gi.Zs > 1.96).sum()), round(mi.I, 4))
    ''')
    out = _run_in_gate(tmp_path, code)
    refs = RC.from_sandbox({"ok": True, "stdout": out["stdout"],
                            "verification": {"reference": out["report"]["reference"]}}, "c1")
    assert {r.method for r in refs} == {"Getis-Ord Gi*", "Moran's I"}
    assert all(not r.disagreeing() for r in refs)


@pytest.fixture()
def uploaded_lattice(monkeypatch, tmp_path, t09_lattice):
    from werkzeug.datastructures import FileStorage

    from agent_runtime.file_store import save_uploaded_file

    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    path, expected = t09_lattice
    with open(path, "rb") as fh:
        fid = save_uploaded_file(FileStorage(stream=fh, filename="lattice.geojson"))["file_id"]
    from agent_runtime.analysis_spatial_stats_tools import make_spatial_stats_tools

    tools = {t.name: t for t in make_spatial_stats_tools(default_input_file_ids=[fid])}
    return fid, tools, expected


def test_the_gi_star_tool_is_recomputed_with_binary_weights(uploaded_lattice):
    fid, tools, expected = uploaded_lattice
    args = {"file_id": fid, "column": "value", "weights": "queen", "star": True}
    ref = RC.from_tool("local_getis_ord", args, tools["local_getis_ord"].invoke(args), "c1")
    by = {f.label: f for f in ref.figures}
    assert by["cells with Gi* z > 1.96"].reference == expected["hot_spots"]
    assert by["cells with Gi* z > 1.96"].differs
    note = RC.reconcile_note([ref])
    assert "binary weights" in note


def test_the_moran_tool_agrees_with_the_reference(uploaded_lattice):
    fid, tools, expected = uploaded_lattice
    args = {"file_id": fid, "column": "value", "weights": "queen"}
    ref = RC.from_tool("global_spatial_autocorrelation", args,
                       tools["global_spatial_autocorrelation"].invoke(args), "c2")
    (f,) = ref.figures
    assert f.reference == pytest.approx(expected["morans_i"], abs=1e-9)
    assert not ref.disagreeing()


# --------------------------------------------------------------------------- 4b slope, zonal

def test_horn_slope_of_a_plane_is_its_gradient():
    x = np.arange(20, dtype=float)
    z = np.tile(0.1 * x * 30.0, (15, 1))       # rises 3 m per 30 m pixel eastwards
    s = RC.horn_slope_degrees(z, 30.0, 30.0)
    assert np.nanmean(s) == pytest.approx(math.degrees(math.atan(0.1)), abs=1e-9)
    assert np.isnan(s[0]).all() and np.isnan(s[:, 0]).all()


def _raster(tmp_path, z, crs="EPSG:32616", px=30.0):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path = tmp_path / "dem.tif"
    with rasterio.open(path, "w", driver="GTiff", height=z.shape[0], width=z.shape[1], count=1,
                       dtype="float64", crs=crs, transform=from_origin(500000, 4400000, px, px),
                       nodata=-9999.0) as dst:
        dst.write(z, 1)
    return path


def test_a_slope_from_a_projected_dem_read_as_degrees_disagrees(tmp_path):
    z = np.tile(0.1 * np.arange(40, dtype=float) * 30.0, (30, 1))
    path = _raster(tmp_path, z)
    resolve = {"dem": path}.__getitem__
    wrong = RC.from_tool("terrain_derivative", {"raster_file_id": "dem", "kind": "slope"},
                         {"ok": True, "mean": 0.0}, "c1", resolve)
    assert wrong.disagreeing()
    right = RC.from_tool("terrain_derivative", {"raster_file_id": "dem", "kind": "slope"},
                         {"ok": True, "mean": 5.71}, "c1", resolve)
    assert not right.disagreeing()


def test_overlapping_zones_are_each_averaged_over_their_own_pixels(tmp_path):
    from shapely.geometry import box

    z = np.zeros((40, 40))
    z[:, 20:] = 100.0                          # east half high
    path = _raster(tmp_path, z)
    zones = gpd.GeoDataFrame({"zone": ["west", "all"]},
                             geometry=[box(500000, 4398800, 500600, 4400000),
                                       box(500000, 4398800, 501200, 4400000)],
                             crs="EPSG:32616")
    zpath = tmp_path / "zones.geojson"
    zones.to_file(zpath, driver="GeoJSON")
    # What a rasterize-once tool reports for an overlap: the west zone's pixels all went to the
    # later zone, so the west zone got nothing; the last one is right.
    out = zones.copy()
    out["value_mean"] = [np.nan, 50.0]
    opath = tmp_path / "out.geojson"
    out.to_file(opath, driver="GeoJSON")
    resolve = {"dem": path, "zones": zpath, "out": opath}.__getitem__
    ref = RC.from_tool("zonal_stats_for_raster",
                       {"raster_file_id": "dem", "polygons_file_id": "zones", "prefix": "value",
                        "all_touched": False},
                       {"ok": True, "geojson": {"file_id": "out"}}, "c1", resolve)
    (f,) = ref.figures                         # the west zone has no reported mean to check
    assert f.reference == pytest.approx(50.0) and not f.differs
    out["value_mean"] = [0.0, 80.0]
    out.to_file(opath, driver="GeoJSON")
    ref = RC.from_tool("zonal_stats_for_raster",
                       {"raster_file_id": "dem", "polygons_file_id": "zones", "prefix": "value",
                        "all_touched": False},
                       {"ok": True, "geojson": {"file_id": "out"}}, "c1", resolve)
    assert [f.label for f in ref.disagreeing()] == ["zone 2 mean"]


def test_a_tool_that_failed_or_an_unknown_tool_is_not_recomputed():
    assert RC.from_tool("local_getis_ord", {}, {"ok": False, "error": "x"}) is None
    assert RC.from_tool("buffer_layer", {}, {"ok": True}) is None


# --------------------------------------------------------------------------- the bench's re-run

def test_only_code_that_can_produce_a_reference_record_is_re_run():
    from gis_harness.reference_reexec import TRIGGER

    assert TRIGGER.search("lam = np.linalg.solve(A, b)")
    assert TRIGGER.search("from scipy.linalg import solve\nw = solve(K, k)")
    assert TRIGGER.search("gi = G_Local(y, w, star=True)")
    assert TRIGGER.search("mi = esda.Moran(y, w)")
    assert not TRIGGER.search("df = pd.read_csv('a.csv')\nprint(df.zinc.mean())")


def test_a_task_file_is_staged_under_every_name_the_run_used(tmp_path):
    from gis_harness.reference_reexec import _stage

    src = tmp_path / "meuse.csv"
    src.write_text("x,y,zinc\n1,2,3\n")
    work = tmp_path / "work"
    work.mkdir()
    body = {"input_files": [{"file_id": "file_1e0c1a05d195", "filename": "meuse.csv",
                             "available_as": ["file_1e0c1a05d195", "meuse.csv"]}]}
    _stage(work, "pd.read_csv('file_1e0c1a05d195__meuse.csv')", body, {"meuse.csv": src})
    assert {p.name for p in work.iterdir()} == {"meuse.csv", "file_1e0c1a05d195",
                                                "file_1e0c1a05d195__meuse.csv"}
