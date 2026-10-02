"""A repeat of the same step replaces its map layer; a different step adds one.

Observed 2026-10-01 in a real browser run: a 46-step turn ("buffer the Champaign boundary by
2 km") re-grounded once, re-running admin_boundary and the buffer. The map ended with the city
outline TWICE. admin_boundary keyed its layer on the file_id it had just written, and the
repeat wrote a byte-identical boundary under a new file_id. The buffer (qgis_metric_buffer,
drawn by add_map_layer) replaced itself only because the model happened to pass add_map_layer
the same `name` both times. Its id came from that label, so the next run could just as easily
have stacked it, and two different buffers sharing a name would have merged.

The rule tested here: a layer's id is a digest of the INPUTS that decided what it shows. A file
among those inputs counts by its content key, never by its file_id. Identical inputs give one
id, so putLayer replaces. Different inputs give different ids, so two analyses never collide.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime import admin_boundary_tools as ab  # noqa: E402


# --- fixtures -------------------------------------------------------------------------------

def _feature(props, ring):
    return {"type": "Feature", "properties": props,
            "geometry": {"type": "Polygon", "coordinates": [ring]}}


def _box(w, s, e, n):
    return [[w, s], [e, s], [e, n], [w, n], [w, s]]


CHAMPAIGN_CITY = _feature({"GEOID": "1712385", "NAME": "Champaign city",
                           "BASENAME": "Champaign", "STATE": "17"},
                          _box(-88.333, 40.065, -88.223, 40.164))
URBANA_CITY = _feature({"GEOID": "1777005", "NAME": "Urbana city",
                        "BASENAME": "Urbana", "STATE": "17"},
                       _box(-88.237, 40.079, -88.159, 40.132))
CHAMPAIGN_COUNTY = _feature({"GEOID": "17019", "NAME": "Champaign County",
                             "BASENAME": "Champaign", "STATE": "17"},
                            _box(-88.46, 39.88, -87.93, 40.40))


@pytest.fixture
def store(tmp_path, monkeypatch):
    root = tmp_path / "store"
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(root))
    monkeypatch.delenv("AGENT_PUBLIC_BASE_URL", raising=False)
    return root


@pytest.fixture
def boundary(store, monkeypatch):
    """admin_boundary with TIGERweb stubbed at its HTTP seam (as test_admin_boundary does)."""
    monkeypatch.setattr(ab, "_states_cache", [{"fips": "17", "name": "Illinois", "usps": "IL"}])

    def fake_query(layer, where, *a, **k):
        if "Tracts" in layer:
            return {"features": [
                _feature({"GEOID": f"170190001{i:02d}", "NAME": str(i), "STATE": "17",
                          "COUNTY": "019"}, _box(-88.3 + i / 100, 40.0, -88.29 + i / 100, 40.01))
                for i in range(6)]}
        if "MapServer/1" in layer and "Places" not in layer:   # counties
            return {"features": [CHAMPAIGN_COUNTY]}
        if "URBANA" in where:
            return {"features": [URBANA_CITY]}
        return {"features": [CHAMPAIGN_CITY]}

    monkeypatch.setattr(ab, "_query", fake_query)
    tool = {t.name: t for t in ab.make_admin_boundary_tools()}["admin_boundary"]
    return lambda **kw: json.loads(tool.func(**kw))


def _geo_tool(name):
    from agent_runtime.langchain_geo_tools import make_langchain_geo_tools
    return {t.name: t for t in make_langchain_geo_tools()}[name]


def _overlay_tool(name):
    from agent_runtime.analysis_overlay_tools import make_overlay_tools
    return {t.name: t for t in make_overlay_tools()}[name]


def _client_id(tool_name, output):
    """The id the CLIENT receives for a tool result's layer: the descriptor as it leaves
    build_map_layers, which fills in an id when the tool set none."""
    from agent_runtime.map_layers import build_map_layers

    layers = build_map_layers(tool_name, output if isinstance(output, str) else json.dumps(output),
                              qa=False)
    assert len(layers) == 1, layers
    return layers[0]["id"]


def _on_screen(*tool_outputs):
    """What the client shows after these results: map-ui-prototype's putLayer REPLACES a layer
    whose id matches and ADDS one whose id does not. Every result goes through the real
    boundary, build_map_layers, which is what turns a tool result into a `map_layer` event."""
    from agent_runtime.map_layers import build_map_layers

    shown = {}
    for tool_name, output in tool_outputs:
        for layer in build_map_layers(tool_name, output, qa=False):
            shown[layer["id"]] = layer["label"]
    return shown


# --- the shared rule ------------------------------------------------------------------------

# Taken from rs_embed_tools._layer_id BEFORE its digest moved into map_layers.content_layer_id.
# Every embedding layer already on someone's map keeps its id only if these stay exact.
GOLDEN_EMBED_IDS = [
    (("pca", "40.113,-88.231"),
     {"bbox": [-88.3, 40.0, -88.2, 40.1], "model": "gse", "period": ["2022-06-01", "2022-09-30"]},
     "embed-pca-40_113__88_231-66c04d7bf1"),
    (("segments", "Downtown Champaign"), {"k": 6, "bbox": [-88.3, 40.0, -88.2, 40.1]},
     "embed-segments-downtown_champaign-2a196ec597"),
    (("zonegroups", "file_abc123"),
     {"file": "file_abc123", "model": "gse", "clusters": 3, "zone_ids": None},
     "embed-zonegroups-file_abc123-4b8c0d3955"),
    (("dem", None), {"bbox": [1, 2, 3, 4], "size": 512}, "embed-dem-005e138f5c"),
    (("", ""), {}, "embed-region-bf21a9e8fb"),
    (("Zone Pixels!", "  Spaced  Hint  "), {"x": 1.5, "y": [3, 2, 1]},
     "embed-zone_pixels-spaced__hint-105b420042"),
    (("predicted", "file_p" * 12), {"column": "median_income", "blocks": 5},
     "embed-predicted-file_pfile_pfile_pfile_pfile_pfile_pfile-fb30f1691e"),
]


@pytest.mark.parametrize("args,content,expected", GOLDEN_EMBED_IDS)
def test_generalising_the_rule_moved_no_embedding_layer(args, content, expected):
    from agent_runtime.rs_embed_tools import _layer_id

    assert _layer_id(*args, **content) == expected


def test_a_layer_id_is_its_namespace_and_its_content_key():
    from agent_runtime.map_layers import content_key, content_layer_id

    assert content_layer_id("agent", "buffer", "2000m", distance_m=2000.0) \
        == "agent-" + content_key("buffer", "2000m", distance_m=2000.0)
    assert content_key("buffer", "2000m", distance_m=2000.0) \
        != content_key("buffer", "2000m", distance_m=3000.0)


# --- what a file holds ----------------------------------------------------------------------

def _write(store_root, name, text, key=None):
    from agent_runtime.file_store import create_output_file_from_path

    src = Path(store_root).parent / "src" / name
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(text, encoding="utf-8")
    recorded = {"content_key": key} if key else {}
    return create_output_file_from_path(src, filename=name, **recorded)["file_id"]


def _gdal_style(stem, features):
    """The layout GDAL's GeoJSON driver writes: the layer name, which is the output filename,
    sits in the file as a top-level member."""
    return json.dumps({"type": "FeatureCollection", "name": stem,
                       "crs": {"type": "name",
                               "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                       "features": features}, indent=None)


def test_identical_content_has_one_key_whatever_the_file_is_called(store):
    from agent_runtime.file_store import file_content_key

    text = json.dumps({"type": "FeatureCollection", "features": [CHAMPAIGN_CITY]})
    a = _write(store, "Champaign_city.geojson", text)
    b = _write(store, "champaign_again.geojson", text)
    c = _write(store, "Urbana_city.geojson",
               json.dumps({"type": "FeatureCollection", "features": [URBANA_CITY]}))
    assert a != b
    assert file_content_key(a) == file_content_key(b)
    assert file_content_key(a) != file_content_key(c)
    assert file_content_key(a).startswith("sha1-")


def test_the_layer_name_gdal_stamps_into_the_file_is_not_content(store):
    """The two 2 km buffers of the observed run were identical except for this member, which
    held their two output filenames."""
    from agent_runtime.file_store import file_content_key

    feats = [CHAMPAIGN_CITY]
    a = _write(store, "a.geojson", _gdal_style("champaign_city_2_km_buffer", feats))
    b = _write(store, "b.geojson", _gdal_style("Champaign_city_2km_buffer", feats))
    assert file_content_key(a) == file_content_key(b)


def test_a_name_inside_the_data_is_still_content(store):
    """Only GDAL's top-level layer-name member is ignored. A feature's own `name` is data."""
    from agent_runtime.file_store import file_content_key

    one = _feature({"name": "one"}, _box(0, 0, 1, 1))
    two = _feature({"name": "two"}, _box(0, 0, 1, 1))
    a = _write(store, "a.geojson", _gdal_style("same", [one]))
    b = _write(store, "b.geojson", _gdal_style("same", [two]))
    assert file_content_key(a) != file_content_key(b)


def test_a_recorded_key_wins_over_the_bytes(store):
    from agent_runtime.file_store import file_content_key

    a = _write(store, "x.geojson", "{}", key="city-1712385-abc")
    assert file_content_key(a) == "city-1712385-abc"


def test_a_file_rewritten_in_place_gets_a_new_key(store, tmp_path):
    """qgis_metric_buffer overwrites its output by name, so a cached digest must notice."""
    from agent_runtime.file_store import file_content_key

    path = tmp_path / "in_place.geojson"
    path.write_text('{"a": 1}', encoding="utf-8")
    before = file_content_key(str(path))
    path.write_text('{"a": 2}', encoding="utf-8")
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000_000))
    assert file_content_key(str(path)) != before


def test_an_unknown_reference_is_its_own_stable_key(store):
    from agent_runtime.file_store import file_content_key
    from agent_runtime.map_layers import boundary_layer_id

    assert file_content_key("  file_nowhere ") == "file_nowhere"
    assert file_content_key(None) == ""
    # Pinned in test_predict_from_package: an id that resolves to nothing is unchanged.
    assert boundary_layer_id("file_poly1") == "boundary-file_poly1"


# --- admin_boundary -------------------------------------------------------------------------

def test_asking_for_the_same_place_again_replaces_its_layer(boundary):
    """The re-grounding pass asked for Champaign again. Same place, same layer, even with a
    different wording and a different output name, and even though a new file was written."""
    first = boundary(area="Champaign", level="city", state="IL", output_name="first pass")
    again = boundary(name="Champaign city", level="city", state="Illinois",
                     output_name="Champaign_city")
    assert first["ok"] and again["ok"]
    assert first["file_id"] != again["file_id"]
    assert first["map_layer"]["id"] == again["map_layer"]["id"]
    assert first["map_layer"]["id"].startswith("boundary-city-1712385-")
    assert len(_on_screen(("admin_boundary", json.dumps(first)),
                          ("admin_boundary", json.dumps(again)))) == 1


def test_embed_zones_still_lands_on_the_boundary_it_redraws(boundary):
    """embed_zones knows only the file_id it is handed, and rebuilds the outline's id from it
    with boundary_layer_id to take that layer's place. It must still arrive at the same id
    from EITHER copy of the file."""
    from agent_runtime.map_layers import boundary_layer_id

    first = boundary(area="Champaign", level="city", state="IL")
    again = boundary(area="Champaign", level="city", state="IL")
    assert boundary_layer_id(first["file_id"]) == first["map_layer"]["id"]
    assert boundary_layer_id(again["file_id"]) == first["map_layer"]["id"]


def test_different_places_levels_and_subdivisions_stay_separate(boundary):
    ids = {
        boundary(area="Champaign", level="city", state="IL")["map_layer"]["id"],
        boundary(area="Urbana", level="city", state="IL")["map_layer"]["id"],
        boundary(area="Champaign", level="county", state="IL")["map_layer"]["id"],
        boundary(area="Champaign", level="county", state="IL",
                 subdivide="tracts")["map_layer"]["id"],
    }
    assert len(ids) == 4, ids


def test_a_cap_that_truncated_the_result_is_part_of_what_it_is(boundary, monkeypatch):
    whole = boundary(area="Champaign", level="county", state="IL", subdivide="tracts")
    monkeypatch.setattr(ab, "MAX_FEATURES", 3)
    capped = boundary(area="Champaign", level="county", state="IL", subdivide="tracts")
    assert capped.get("truncated") and capped["feature_count"] == 3
    assert whole["map_layer"]["id"] != capped["map_layer"]["id"]


# --- buffer_layer (the geopandas buffer) ----------------------------------------------------

def test_the_same_buffer_of_the_same_boundary_is_one_layer(boundary):
    """Two copies of one boundary (the re-ground's), different names, km vs m: one layer."""
    pytest.importorskip("geopandas")
    first = boundary(area="Champaign", level="city", state="IL")
    again = boundary(area="Champaign", level="city", state="IL", output_name="again")
    buffer = _overlay_tool("buffer_layer")
    a = json.loads(buffer.invoke({"file_id": first["file_id"], "distance": 2, "units": "km",
                                  "name": "Champaign city 2 km buffer"}))
    b = json.loads(buffer.invoke({"file_id": again["file_id"], "distance": 2000, "units": "m",
                                  "name": "Champaign_city_2km_buffer"}))
    assert a["ok"] and b["ok"]
    assert a["file_id"] != b["file_id"]
    assert _client_id("buffer_layer", a) == _client_id("buffer_layer", b)
    assert _client_id("buffer_layer", a).startswith("agent-buffer-2000m-")


def test_different_buffers_never_share_a_layer(boundary):
    """The opposite failure: the label-derived id merged two buffers that shared a name."""
    pytest.importorskip("geopandas")
    champaign = boundary(area="Champaign", level="city", state="IL")["file_id"]
    urbana = boundary(area="Urbana", level="city", state="IL")["file_id"]
    buffer = _overlay_tool("buffer_layer")

    def layer(**kw):
        out = json.loads(buffer.invoke({"name": "2 km buffer", **kw}))
        assert out["ok"], out
        return _client_id("buffer_layer", out)

    ids = [layer(file_id=champaign, distance=2, units="km"),
           layer(file_id=champaign, distance=3, units="km"),
           layer(file_id=urbana, distance=2, units="km"),
           layer(file_id=champaign, distance=2, units="km", dissolve=True)]
    assert len(set(ids)) == 4, ids


# --- qgis_metric_buffer -> add_map_layer: the chain the observed run took -------------------

@pytest.fixture
def fake_qgis(monkeypatch):
    """qgis_process stood in for by a copy, with GDAL's habit of stamping the output filename
    into the GeoJSON it writes. Real QGIS did exactly that in the observed run."""
    from rag_pipeline import qgis_headless_tools as q

    def fake_run(command, **kwargs):
        args = dict(a.split("=", 1) for a in command if "=" in a and a.split("=", 1)[0].isupper())
        src, dst = Path(args["INPUT"]), Path(args["OUTPUT"])
        if dst.suffix == ".geojson":
            data = json.loads(src.read_text(encoding="utf-8"))
            dst.write_text(_gdal_style(dst.stem, data["features"]), encoding="utf-8")
        else:
            shutil.copyfile(src, dst)
        return subprocess.CompletedProcess(command, 0, stdout='{"ok": true}', stderr="")

    monkeypatch.setattr(q.subprocess, "run", fake_run)
    return q.qgis_metric_buffer_tool


def test_the_qgis_buffer_records_what_it_holds_not_where_it_was_written(boundary, fake_qgis):
    from agent_runtime.file_store import file_content_key

    first = boundary(area="Champaign", level="city", state="IL")["file_id"]
    again = boundary(area="Champaign", level="city", state="IL")["file_id"]
    a = json.loads(fake_qgis(first, 2000, output_filename="Champaign_city_2km_buffer.geojson",
                             session_id="s"))
    b = json.loads(fake_qgis(again, 2000, output_filename="champaign_2km.geojson",
                             session_id="s"))
    c = json.loads(fake_qgis(first, 3000, output_filename="Champaign_city_3km_buffer.geojson",
                             session_id="s"))
    ka, kb, kc = (file_content_key(r["managed_output"]["file_id"]) for r in (a, b, c))
    assert a["managed_output"]["file_id"] != b["managed_output"]["file_id"]
    assert ka == kb and ka.startswith("qgis_buffer-2000m-")
    assert kc != ka


def test_the_regrounded_turn_ends_with_two_layers_not_four(boundary, fake_qgis):
    """The observed turn, end to end: boundary, QGIS buffer, add_map_layer, all twice. The
    second pass uses different output names and a different layer name, which is the variation
    the model actually produces. Before this change the boundary stacked."""
    pytest.importorskip("geopandas")
    add_map_layer = _geo_tool("add_map_layer").func
    results = []
    # The live check of this fix (2026-10-01) passed the boundary as a "sibling" of the GeoJSON
    # buffer on the first pass and nothing on the repeat. Both variations are reproduced here.
    passes = (("Champaign_city_2km_buffer.geojson", "Champaign city + 2 km buffer", True),
              ("champaign_city_buffer_2000m.geojson", "Champaign 2 km buffer", False))
    for out_name, layer_name, with_sibling in passes:
        city = boundary(area="Champaign", level="city", state="IL")
        buf = json.loads(fake_qgis(city["file_id"], 2000, output_filename=out_name,
                                   session_id="s"))
        drawn = add_map_layer(file_id=buf["managed_output"]["file_id"], name=layer_name,
                              sibling_file_ids=[city["file_id"]] if with_sibling else None)
        results.append([("admin_boundary", json.dumps(city)), ("qgis_metric_buffer", json.dumps(buf)),
                        ("add_map_layer", drawn)])

    first_pass = _on_screen(*results[0])
    both_passes = _on_screen(*results[0], *results[1])
    assert len(first_pass) == 2, first_pass
    assert set(both_passes) == set(first_pass), both_passes      # replaced, not added
    assert list(both_passes.values())[-1] == "Champaign 2 km buffer"   # the label did update


def test_add_map_layer_keeps_different_views_of_one_dataset_apart(boundary, store):
    """Replacing must not over-reach: the same data drawn two ways is two layers."""
    pytest.importorskip("geopandas")
    add_map_layer = _geo_tool("add_map_layer").func
    pts = json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"v": i}, "geometry": {"type": "Point",
                                                                  "coordinates": [-88.2 + i / 100, 40.1]}}
        for i in range(5)]})
    f = _write(store, "points.geojson", pts)
    same_again = _write(store, "points_copy.geojson", pts)

    def layer_id(**kw):
        out = add_map_layer(**kw)
        assert json.loads(out)["ok"], out
        return _client_id("add_map_layer", out)

    points = layer_id(file_id=f, render="points", name="sites")
    assert layer_id(file_id=same_again, render="points", name="the same sites") == points
    assert layer_id(file_id=f, render="heatmap", name="sites") != points
    assert layer_id(file_id=f, render="choropleth", column="v", name="sites") != points
    assert layer_id(file_id=boundary(area="Champaign", level="city", state="IL")["file_id"],
                    render="shapes", name="sites") != points


# --- what counts as the input: what was READ ------------------------------------------------

def test_a_sibling_that_changes_nothing_read_changes_nothing(boundary, fake_qgis):
    """Found by the live check, not by the first version of these tests. The model passed the
    boundary as a `sibling_file_ids` entry for a GeoJSON buffer, where siblings are never read,
    and the repeat passed none. Keyed on the list, the same buffer became two layers."""
    pytest.importorskip("geopandas")
    city = boundary(area="Champaign", level="city", state="IL")["file_id"]
    buf = json.loads(fake_qgis(city, 2000, session_id="s"))["managed_output"]["file_id"]
    add_map_layer = _geo_tool("add_map_layer").func
    plain = _client_id("add_map_layer", add_map_layer(file_id=buf, render="shapes"))
    listed = _client_id("add_map_layer", add_map_layer(file_id=buf, render="shapes",
                                                       sibling_file_ids=[city]))
    assert plain == listed

    buffer = _overlay_tool("buffer_layer")
    a = json.loads(buffer.invoke({"file_id": city, "distance": 2, "units": "km"}))
    b = json.loads(buffer.invoke({"file_id": city, "distance": 2, "units": "km",
                                  "sibling_file_ids": [buf]}))
    assert _client_id("buffer_layer", a) == _client_id("buffer_layer", b)


def _shapefile_parts(store, tmp_path, tag, values):
    """Write one shapefile and register each part as its own file, as an upload of the parts
    arrives. Returns (shp_id, [sibling ids])."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    from agent_runtime.file_store import create_output_file_from_path

    work = tmp_path / f"shp_{tag}"
    work.mkdir()
    gpd.GeoDataFrame({"v": values}, geometry=[box(-88.3, 40.0, -88.2, 40.1),
                                              box(-88.2, 40.0, -88.1, 40.1)],
                     crs="EPSG:4326").to_file(work / "tracts.shp")
    ids = {part.suffix: create_output_file_from_path(part, filename=part.name)["file_id"]
           for part in sorted(work.iterdir())}
    return ids.pop(".shp"), list(ids.values())


def test_a_shapefile_is_identified_by_all_of_its_parts(store, tmp_path):
    """Siblings DO count when they are what is read: the attributes of a shapefile live in its
    .dbf, so the same geometry with different attributes is a different layer, and the same
    parts uploaded again under new ids are the same layer."""
    add_map_layer = _geo_tool("add_map_layer").func

    def layer(shp, siblings):
        out = add_map_layer(file_id=shp, sibling_file_ids=siblings, render="choropleth",
                            column="v")
        assert json.loads(out)["ok"], out
        return _client_id("add_map_layer", out)

    first = layer(*_shapefile_parts(store, tmp_path, "a", [1, 2]))
    again = layer(*_shapefile_parts(store, tmp_path, "b", [1, 2]))
    other = layer(*_shapefile_parts(store, tmp_path, "c", [5, 9]))
    assert first == again
    assert first != other
