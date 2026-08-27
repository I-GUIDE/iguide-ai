"""Tests for the vector / shapefile (TIGER) tools.

Build a tiny real shapefile with geopandas, register the components as uploads, and
exercise read / visualize / analyze across the zip and extracted-sidecars paths.
Skipped if the geo stack isn't installed (it is in the agent runtime image).
"""

from __future__ import annotations

import glob
import json
import zipfile
from pathlib import Path

import pytest

gpd = pytest.importorskip("geopandas")
pytest.importorskip("pyogrio")
from shapely.geometry import Point  # noqa: E402
from werkzeug.datastructures import FileStorage  # noqa: E402


def _tools():
    from agent_runtime.langchain_geo_tools import make_langchain_geo_tools
    return {t.name: t for t in make_langchain_geo_tools()}


def _upload(path):
    from agent_runtime.file_store import save_uploaded_file
    with open(path, "rb") as fh:
        return save_uploaded_file(FileStorage(stream=fh, filename=Path(path).name))["file_id"]


@pytest.fixture()
def shapefile(monkeypatch, tmp_path):
    """Write a 3-point shapefile (EPSG:4326) and return uploaded id variants."""
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    work = tmp_path / "shp"
    work.mkdir()
    gpd.GeoDataFrame(
        {"name": ["a", "b", "c"], "val": [1, 2, 3]},
        geometry=[Point(-88.24, 40.11), Point(-88.20, 40.12), Point(-88.18, 40.10)],
        crs="EPSG:4326",
    ).to_file(work / "pts.shp")
    comps = sorted(glob.glob(str(work / "pts.*")))
    zp = work / "pts.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for c in comps:
            z.write(c, Path(c).name)
    ids = {Path(c).suffix: _upload(c) for c in comps}
    return {
        "zip_id": _upload(zp),
        "shp_id": ids[".shp"],
        "siblings": [v for k, v in ids.items() if k != ".shp"],
        "work": work,
    }


def test_factory_shape():
    tools = _tools()
    assert set(tools) == {"inspect_vector", "render_map_image", "vector_to_geojson",
                          "reproject_vector", "vector_spatial_join", "add_map_layer"}
    assert all(getattr(t, "metadata", {}).get("category") == "geo" for t in tools.values())


def test_inspect_zip(shapefile):
    r = json.loads(_tools()["inspect_vector"].invoke({"file_id": shapefile["zip_id"]}))
    assert r["ok"] is True
    assert r["feature_count"] == 3
    assert r["crs"] == "EPSG:4326"
    assert {"name", "val"} <= {c["name"] for c in r["columns"]}
    assert len(r["bounds"]) == 4


def test_inspect_extracted_siblings(shapefile):
    """The hard case: .shp + sidecars uploaded as SEPARATE files must be re-staged."""
    r = json.loads(_tools()["inspect_vector"].invoke(
        {"file_id": shapefile["shp_id"], "sibling_file_ids": shapefile["siblings"]}))
    assert r["ok"] is True and r["feature_count"] == 3
    assert "val" in {c["name"] for c in r["columns"]}  # .dbf attributes resolved


def test_plot_creates_downloadable_png(shapefile):
    from agent_runtime.file_store import resolve_file_id
    p = json.loads(_tools()["render_map_image"].invoke({"file_id": shapefile["zip_id"], "column": "val"}))
    assert p["ok"] is True and p["download_url"]
    assert resolve_file_id(p["file_id"]).stat().st_size > 0  # a real PNG was written


def test_plot_downsamples(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    work = tmp_path / "many"; work.mkdir()
    pts = [Point(-88 + i * 0.001, 40 + i * 0.001) for i in range(100)]
    gpd.GeoDataFrame({"i": list(range(100))}, geometry=pts, crs="EPSG:4326").to_file(work / "m.shp")
    comps = glob.glob(str(work / "m.*"))  # capture BEFORE creating the zip
    zp = work / "m.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for c in comps:
            z.write(c, Path(c).name)
    p = json.loads(_tools()["render_map_image"].invoke({"file_id": _upload(zp), "max_features": 10}))
    assert p["ok"] is True and p["downsampled"] is True and p["plotted_features"] == 10


def test_vector_to_geojson_reprojects_to_wgs84(monkeypatch, tmp_path):
    from agent_runtime.file_store import resolve_file_id
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    work = tmp_path / "merc"; work.mkdir()
    gpd.GeoDataFrame({"v": [1]}, geometry=[Point(-9825000, 4880000)], crs="EPSG:3857").to_file(work / "m.shp")
    comps = glob.glob(str(work / "m.*"))  # capture BEFORE creating the zip
    zp = work / "m.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for c in comps:
            z.write(c, Path(c).name)
    g = json.loads(_tools()["vector_to_geojson"].invoke({"file_id": _upload(zp)}))
    assert g["ok"] is True
    out = gpd.read_file(resolve_file_id(g["file_id"]))
    assert out.crs.to_epsg() == 4326  # reprojected from 3857


def test_spatial_join(shapefile):
    r = json.loads(_tools()["vector_spatial_join"].invoke(
        {"left_file_id": shapefile["zip_id"], "right_file_id": shapefile["zip_id"]}))
    assert r["ok"] is True and r["feature_count"] == 3 and r["download_url"]


def test_graceful_failure_no_raise():
    r = json.loads(_tools()["inspect_vector"].invoke({"file_id": "file_does_not_exist"}))
    assert r["ok"] is False and r.get("error")


# --- wiring into the peers -------------------------------------------------

def test_geo_tools_wired_into_peers_only_with_files(monkeypatch):
    import agent_runtime.executor_factory as ef
    import agent_runtime.supervisor_graph as sg

    captured = {}

    def fake_build(**kwargs):
        captured["tools"] = [getattr(t, "name", "") for t in (kwargs.get("preloaded_tools") or [])]
        return object()

    monkeypatch.setattr(ef, "build_agent_executor", fake_build)
    monkeypatch.setattr(ef, "invoke_agent_with_payload_fallback", lambda *a, **k: {"messages": []})
    # avoid heavy QGIS/MCP imports in the analyze peer
    import agent_runtime.langchain_granular_tools as gt
    monkeypatch.setattr(gt, "make_langchain_qgis_tools", lambda **k: [])
    monkeypatch.delenv("AGENT_CODE_EXEC", raising=False)

    # analyze peer WITH files -> geo tools present
    sg.default_analyze_fn(include_mcp_tools=False, input_file_ids=["file_x"])("q", [], {"thread_id": None})
    assert "inspect_vector" in captured["tools"] and "render_map_image" in captured["tools"]

    # analyze peer WITHOUT files -> no geo tools
    captured.clear()
    sg.default_analyze_fn(include_mcp_tools=False, input_file_ids=None)("q", [], {"thread_id": None})
    assert "inspect_vector" not in captured["tools"]

    # code peer WITH files -> geo tools present
    captured.clear()
    sg.default_code_fn(input_file_ids=["file_x"])("q", [], {"thread_id": None})
    assert "inspect_vector" in captured["tools"]


# The overlay / aggregate / temporal registries ride the same "files are attached"
# gate as the vector tools above. Every name is listed explicitly: these sets are the
# contract the peers advertise to the model, so a factory that silently stops
# exporting a tool has to fail here rather than quietly shrink the toolset.
OVERLAY_TOOLS = {"clip_layer", "dissolve_layer", "intersect_layers", "erase_layer",
                 "buffer_layer", "simplify_layer", "geometry_summary"}
AGGREGATE_TOOLS = {"count_points_in_areas", "aggregate_to_grid", "nearest_distance",
                   "cluster_points", "summary_statistics", "select_by_attribute"}
TEMPORAL_TOOLS = {"detect_time_column", "filter_by_time", "time_series",
                  "compare_periods", "temporal_hotspots"}
ANALYSIS_TOOLS = OVERLAY_TOOLS | AGGREGATE_TOOLS | TEMPORAL_TOOLS

# The spatial-statistics family (libpysal/esda/spreg/pygeoda). Kept OUT of ANALYSIS_TOOLS and
# asserted separately: its factory is guarded like the others, so a deployment without the PySAL
# stack legitimately exposes none of these and must not fail the shared contract above.
SPATIAL_STATS_TOOLS = {"spatial_weights", "global_spatial_autocorrelation", "local_moran_lisa",
                       "local_getis_ord", "moran_scatterplot", "spatial_regression",
                       "regionalize"}


def _needs_pysal():
    pytest.importorskip("libpysal")
    pytest.importorskip("esda")


def test_analysis_tools_wired_into_peers_only_with_files(monkeypatch):
    pytest.importorskip("pandas")
    import agent_runtime.executor_factory as ef
    import agent_runtime.supervisor_graph as sg

    captured = {}

    def fake_build(**kwargs):
        captured["tools"] = [getattr(t, "name", "") for t in (kwargs.get("preloaded_tools") or [])]
        return object()

    monkeypatch.setattr(ef, "build_agent_executor", fake_build)
    monkeypatch.setattr(ef, "invoke_agent_with_payload_fallback", lambda *a, **k: {"messages": []})
    import agent_runtime.langchain_granular_tools as gt
    monkeypatch.setattr(gt, "make_langchain_qgis_tools", lambda **k: [])
    monkeypatch.delenv("AGENT_CODE_EXEC", raising=False)

    for label, run in (
        ("analyze", lambda fids: sg.default_analyze_fn(
            include_mcp_tools=False, input_file_ids=fids)("q", [], {"thread_id": None})),
        ("code", lambda fids: sg.default_code_fn(
            input_file_ids=fids)("q", [], {"thread_id": None})),
    ):
        captured.clear()
        run(["file_x"])
        names = captured["tools"]
        assert ANALYSIS_TOOLS <= set(names), (
            f"{label} peer is missing {sorted(ANALYSIS_TOOLS - set(names))}")
        # A duplicate name makes the model's tool choice ambiguous and can break
        # provider-side schema validation, so the assembled set must stay unique.
        assert len(names) == len(set(names)), (
            f"{label} peer has duplicate tool names: "
            f"{sorted(n for n in set(names) if names.count(n) > 1)}")

        captured.clear()
        run(None)
        assert not (ANALYSIS_TOOLS & set(captured["tools"])), (
            f"{label} peer exposed file-only analysis tools with nothing attached: "
            f"{sorted(ANALYSIS_TOOLS & set(captured['tools']))}")


def test_spatial_stats_tools_wired_into_peers_only_with_files(monkeypatch):
    """Same gate as the other analysis families: both peers get them when files are
    attached, neither does when nothing is."""
    _needs_pysal()
    import agent_runtime.executor_factory as ef
    import agent_runtime.supervisor_graph as sg

    captured = {}
    monkeypatch.setattr(ef, "build_agent_executor",
                        lambda **kw: captured.__setitem__(
                            "tools", [getattr(t, "name", "")
                                      for t in (kw.get("preloaded_tools") or [])]) or object())
    monkeypatch.setattr(ef, "invoke_agent_with_payload_fallback", lambda *a, **k: {"messages": []})
    import agent_runtime.langchain_granular_tools as gt
    monkeypatch.setattr(gt, "make_langchain_qgis_tools", lambda **k: [])
    monkeypatch.delenv("AGENT_CODE_EXEC", raising=False)

    for label, run in (
        ("analyze", lambda fids: sg.default_analyze_fn(
            include_mcp_tools=False, input_file_ids=fids)("q", [], {"thread_id": None})),
        ("code", lambda fids: sg.default_code_fn(
            input_file_ids=fids)("q", [], {"thread_id": None})),
    ):
        captured.clear()
        run(["file_x"])
        names = captured["tools"]
        assert SPATIAL_STATS_TOOLS <= set(names), (
            f"{label} peer is missing {sorted(SPATIAL_STATS_TOOLS - set(names))}")
        assert len(names) == len(set(names)), (
            f"{label} peer has duplicate tool names: "
            f"{sorted(n for n in set(names) if names.count(n) > 1)}")

        captured.clear()
        run(None)
        assert not (SPATIAL_STATS_TOOLS & set(captured["tools"])), (
            f"{label} peer exposed file-only spatial-stats tools with nothing attached")


def test_spatial_stats_tools_in_capability_inventory():
    _needs_pysal()
    from agent_runtime.capabilities import collect_capability_inventory

    names = {t["name"] for t in collect_capability_inventory()["tools"]}
    assert SPATIAL_STATS_TOOLS <= names, f"missing {sorted(SPATIAL_STATS_TOOLS - names)}"


def test_analysis_tools_are_tagged_geo_like_their_neighbours():
    """The peers mix these in with make_langchain_geo_tools, which tags every tool
    category=geo; an untagged tool would sort differently wherever category is read."""
    pytest.importorskip("pandas")
    from agent_runtime.analysis_aggregate_tools import make_aggregate_tools
    from agent_runtime.analysis_overlay_tools import make_overlay_tools
    from agent_runtime.analysis_temporal_tools import make_temporal_tools

    families = [(make_overlay_tools, OVERLAY_TOOLS), (make_aggregate_tools, AGGREGATE_TOOLS),
                (make_temporal_tools, TEMPORAL_TOOLS)]
    try:
        from agent_runtime.analysis_spatial_stats_tools import make_spatial_stats_tools
        import libpysal  # noqa: F401
        families.append((make_spatial_stats_tools, SPATIAL_STATS_TOOLS))
    except Exception:
        pass
    for factory, expected in families:
        tools = factory()
        assert {t.name for t in tools} == expected
        for t in tools:
            assert (getattr(t, "metadata", {}) or {}).get("category") == "geo", t.name


def test_capability_inventory_reports_the_analysis_tools():
    """The inventory answers 'what can you do' independently of what is attached
    right now, so it must probe these registries unfiltered like the geo one."""
    pytest.importorskip("pandas")
    from agent_runtime.capabilities import collect_capability_inventory

    names = {t["name"] for t in collect_capability_inventory()["tools"]}
    assert ANALYSIS_TOOLS <= names, f"missing {sorted(ANALYSIS_TOOLS - names)}"


# --- auto-discovery of extracted shapefile siblings ------------------------

def _tools_with(attached):
    from agent_runtime.langchain_geo_tools import make_langchain_geo_tools
    return {t.name: t for t in make_langchain_geo_tools(default_input_file_ids=attached)}


def test_inspect_auto_discovers_siblings_from_attached(shapefile):
    """No sibling_file_ids passed: the .shx/.dbf are found among the attached files,
    so the model only has to reference the .shp's file_id."""
    tools = _tools_with([shapefile["shp_id"], *shapefile["siblings"]])
    r = json.loads(tools["inspect_vector"].invoke({"file_id": shapefile["shp_id"]}))
    assert r["ok"] is True and r["feature_count"] == 3
    assert "val" in {c["name"] for c in r["columns"]}  # .dbf attributes resolved


def test_reference_any_component_resolves_shapefile(shapefile):
    """Pointing at a SIDECAR (not the .shp) still reconstructs the set by basename."""
    attached = [shapefile["shp_id"], *shapefile["siblings"]]
    tools = _tools_with(attached)
    r = json.loads(tools["inspect_vector"].invoke({"file_id": shapefile["siblings"][0]}))
    assert r["ok"] is True and r["feature_count"] == 3


def test_plot_auto_discovers_siblings(shapefile):
    from agent_runtime.file_store import resolve_file_id
    tools = _tools_with([shapefile["shp_id"], *shapefile["siblings"]])
    p = json.loads(tools["render_map_image"].invoke({"file_id": shapefile["shp_id"], "column": "val"}))
    assert p["ok"] is True and resolve_file_id(p["file_id"]).stat().st_size > 0


def test_lone_shp_without_sidecars_fails_clearly(shapefile):
    """A .shp with no sidecars (none attached, none passed) fails soft with a useful hint."""
    tools = _tools_with([shapefile["shp_id"]])  # only the .shp is "attached"
    r = json.loads(tools["inspect_vector"].invoke({"file_id": shapefile["shp_id"]}))
    assert r["ok"] is False and r.get("hint")


# --- add_map_layer: the interactive-map delivery path ----------------------------

def test_add_map_layer_describes_a_layer_for_the_client(shapefile):
    """The tool must return a map_layer descriptor build_map_layer can forward."""
    import json
    from agent_runtime.map_layers import build_map_layer

    out = _tools()["add_map_layer"].func(file_id=shapefile["zip_id"], render="auto", name="parcels")
    res = json.loads(out)
    assert res["ok"], res
    assert res["filename"].endswith(".geojson")     # GeoJSON, never parquet: the client reads it
    assert res["on_map"] is True
    ml = res["map_layer"]
    assert ml["url"] and ml["render"] in {"points", "shapes", "heatmap"}

    layer = build_map_layer("add_map_layer", out)   # -> the `map_layer` SSE event
    assert layer["kind"] == "map_layer"
    assert layer["url"] == ml["url"]
    assert layer["label"] == "parcels"


def test_choropleth_reports_the_numeric_columns_when_the_column_is_wrong(shapefile):
    """An arbitrary first-N column list hid the very column being looked for."""
    import json

    res = json.loads(_tools()["add_map_layer"].func(
        file_id=shapefile["zip_id"], render="choropleth", column="does_not_exist"))
    assert res["ok"] is False
    assert "numeric_columns" in res          # candidates, not a truncated slice of everything


def test_add_map_layer_redirects_an_image_to_the_real_datasets(tmp_path, monkeypatch):
    """Observed: the peer passed heatmap.png to add_map_layer, got 'unreadable vector source'
    plus an unrelated shapefile hint, and told the user an interactive map was impossible."""
    import json
    from agent_runtime.langchain_geo_tools import make_langchain_geo_tools

    png = tmp_path / "heatmap.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    csv = tmp_path / "incidents.csv"
    csv.write_text("lat,lon\n41.9,-87.6\n", encoding="utf-8")

    ids = {}
    for p in (png, csv):
        from agent_runtime.file_store import create_output_file_from_path
        ids[p.name] = create_output_file_from_path(p, filename=p.name)["file_id"]

    tools = {t.name: t for t in make_langchain_geo_tools(
        default_input_file_ids=[ids["heatmap.png"], ids["incidents.csv"]])}
    out = json.loads(tools["add_map_layer"].func(file_id=ids["heatmap.png"], render="heatmap"))

    assert out["ok"] is False
    assert "image" in out["error"].lower()
    # The point of the fix: it names what to pass INSTEAD.
    assert [c["filename"] for c in out["mappable_file_ids"]] == ["incidents.csv"]
    assert "shapefile sidecar" not in out["hint"]
