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


# =============================================================================================
# Stage 13.7: the rest of the toolkit. Every layer below took its id from its label (the clip,
# dissolve, intersect, erase, simplify and geometry-summary layers, the aggregate, spatial-
# statistics and temporal layers, a code peer's files) or from its input FILE ids (the zonal,
# prediction and shared-PCA layers), and the terrain layers from their raster's bbox alone.
# Each section reproduces both failures on the old ids. A re-ground re-fetches the inputs and
# lets the model rename the result, and it must REPLACE. A genuinely different analysis of the
# same inputs must ADD.
# =============================================================================================

@pytest.fixture
def fetched(boundary):
    """Every input fetched twice, as a first pass and its re-ground fetch them: two file_ids
    per place, one content per place."""
    def fetch(tag):
        return {
            "county": boundary(area="Champaign", level="county", state="IL",
                               output_name=f"county {tag}")["file_id"],
            "city": boundary(area="Champaign", level="city", state="IL",
                             output_name=f"city {tag}")["file_id"],
            "urbana": boundary(area="Urbana", level="city", state="IL",
                               output_name=f"urbana {tag}")["file_id"],
            "tracts": boundary(area="Champaign", level="county", state="IL", subdivide="tracts",
                               output_name=f"tracts {tag}")["file_id"],
        }
    return fetch("first"), fetch("again")


def _run_tool(tools, tool_name, **kw):
    out = json.loads(tools[tool_name].invoke(kw))
    assert out.get("ok"), out
    return out


def _replaces_and_adds(tool_name, kind, first, again, different):
    """The two assertions every section makes, on three results of one tool: the same analysis
    of re-fetched inputs under a new name, and a different analysis of the first inputs."""
    assert first["file_id"] != again["file_id"], "the re-ground must write a new file"
    ident = _client_id(tool_name, first)
    assert _client_id(tool_name, again) == ident, "a re-ground must REPLACE its layer"
    assert _client_id(tool_name, different) != ident, "a different analysis must ADD a layer"
    assert ident.startswith(f"agent-{kind}-"), ident
    shown = _on_screen(*((tool_name, json.dumps(r)) for r in (first, again, different)))
    assert len(shown) == 2, shown


# --- overlay: clip, dissolve, intersect, erase, simplify, geometry summary ------------------

# (tool, kind in the id, the analysis, a genuinely DIFFERENT analysis of the same places)
OVERLAY_CASES = [
    ("erase_layer", "erase",
     lambda f: {"target_file_id": f["county"], "erase_file_id": f["city"]},
     lambda f: {"target_file_id": f["county"], "erase_file_id": f["urbana"]}),
    ("clip_layer", "clip",
     lambda f: {"target_file_id": f["county"], "clip_file_id": f["city"]},
     lambda f: {"target_file_id": f["county"], "clip_file_id": f["urbana"]}),
    ("intersect_layers", "overlay",
     lambda f: {"left_file_id": f["county"], "right_file_id": f["city"]},
     lambda f: {"left_file_id": f["county"], "right_file_id": f["city"], "how": "union"}),
    ("dissolve_layer", "dissolve",
     lambda f: {"file_id": f["tracts"], "by": "STATE"},
     lambda f: {"file_id": f["tracts"], "by": "NAME"}),
    ("simplify_layer", "simplify",
     lambda f: {"file_id": f["county"], "tolerance_m": 50},
     lambda f: {"file_id": f["county"], "tolerance_m": 500}),
    ("geometry_summary", "centroids",
     lambda f: {"file_id": f["tracts"], "output": "centroids"},
     lambda f: {"file_id": f["tracts"], "output": "bbox"}),
]


@pytest.mark.parametrize("tool_name,kind,same,different", OVERLAY_CASES,
                         ids=[case[0] for case in OVERLAY_CASES])
def test_a_regrounded_overlay_replaces_its_layer_and_a_different_one_adds(
        fetched, tool_name, kind, same, different):
    """Two unnamed erases were both `agent-erased`, so the second replaced the first, and a
    re-run the model renamed stacked a copy. Every tool here drew without a key."""
    pytest.importorskip("geopandas")
    from agent_runtime.analysis_overlay_tools import make_overlay_tools

    tools = {t.name: t for t in make_overlay_tools()}
    first, again = fetched
    _replaces_and_adds(
        tool_name, kind,
        _run_tool(tools, tool_name, **same(first)),
        _run_tool(tools, tool_name, **same(again), name="the model's second name for it"),
        _run_tool(tools, tool_name, **different(first)))


def test_a_dissolve_drawn_as_it_would_be_drawn_anyway_is_one_layer(fetched):
    """dissolve_layer is the one overlay tool whose view is a parameter, so its key holds the
    view, taken AFTER 'auto' and the default shading are resolved. Asking for the choropleth it
    draws anyway is the same layer. Asking for plain shapes is a different one, and under the
    label-derived id it merged with the choropleth."""
    pytest.importorskip("geopandas")
    from agent_runtime.analysis_overlay_tools import make_overlay_tools

    tools = {t.name: t for t in make_overlay_tools()}
    first, _ = fetched

    def layer(**kw):
        return _client_id("dissolve_layer", _run_tool(tools, "dissolve_layer",
                                                      file_id=first["tracts"], by="STATE", **kw))

    auto = layer()
    assert layer(render="choropleth") == auto
    assert layer(render="choropleth", style_by="feature_count") == auto
    assert layer(render="shapes") != auto


# --- aggregate --------------------------------------------------------------------------------

def _incidents(store, name):
    """Eighteen points, three inside each of the six stub tracts, with a value and a date."""
    feats = [{"type": "Feature",
              "properties": {"v": i * 10 + j, "when": f"2026-0{6 + j % 2}-{10 + i:02d}"},
              "geometry": {"type": "Point",
                           "coordinates": [-88.2985 + i / 100 + j * 0.002, 40.002 + j * 0.003]}}
             for i in range(6) for j in range(3)]
    return _write(store, name, json.dumps({"type": "FeatureCollection", "features": feats}))


AGGREGATE_CASES = [
    ("count_points_in_areas", "points_in_areas",
     lambda p, f: {"points_file_id": p, "areas_file_id": f["tracts"]},
     lambda p, f: {"points_file_id": p, "areas_file_id": f["tracts"], "statistic": "sum",
                   "value_column": "v"}),
    ("aggregate_to_grid", "grid",
     lambda p, f: {"points_file_id": p, "cell_km": 1.0},
     lambda p, f: {"points_file_id": p, "cell_km": 0.5}),
    ("nearest_distance", "nearest",
     lambda p, f: {"from_file_id": p, "to_file_id": f["city"]},
     lambda p, f: {"from_file_id": p, "to_file_id": f["urbana"]}),
    ("cluster_points", "clusters",
     lambda p, f: {"file_id": p, "eps_km": 0.5, "min_samples": 2},
     lambda p, f: {"file_id": p, "eps_km": 0.2, "min_samples": 2}),
    ("select_by_attribute", "selected",
     lambda p, f: {"file_id": p, "column": "v", "top_n": 3},
     lambda p, f: {"file_id": p, "column": "v", "top_n": 5}),
]


@pytest.mark.parametrize("tool_name,kind,same,different", AGGREGATE_CASES,
                         ids=[case[0] for case in AGGREGATE_CASES])
def test_a_regrounded_aggregate_replaces_its_layer_and_a_different_one_adds(
        store, fetched, tool_name, kind, same, different):
    """These ids came from the label, and the unnamed label from the output filename, which
    comes from the input's name. So a hex grid at 1 km and at 0.5 km were one layer, and a
    renamed repeat was two. The kind prefix is asserted too: count_points_in_areas reuses the
    name `key` for its CSV's label column, and a key computed before that line would ship as
    `agent-area`."""
    pytest.importorskip("geopandas")
    if tool_name == "cluster_points":
        pytest.importorskip("sklearn")
    from agent_runtime.analysis_aggregate_tools import make_aggregate_tools

    tools = {t.name: t for t in make_aggregate_tools()}
    first, again = fetched
    points, points_again = _incidents(store, "incidents.geojson"), _incidents(store, "copy.geojson")
    _replaces_and_adds(
        tool_name, kind,
        _run_tool(tools, tool_name, **same(points, first)),
        _run_tool(tools, tool_name, **same(points_again, again), name="incidents again"),
        _run_tool(tools, tool_name, **different(points, first)))


def test_a_parameter_that_changes_nothing_drawn_does_not_split_a_layer(store, fetched):
    """Keys hold what DECIDES the layer, normalised. A value column that a count never reads is
    not a new layer, nor is m against km, which write the same columns. Miles add a column, so
    they are a new layer, and under the filename-derived id they were not."""
    pytest.importorskip("geopandas")
    from agent_runtime.analysis_aggregate_tools import make_aggregate_tools

    tools = {t.name: t for t in make_aggregate_tools()}
    first, _ = fetched
    points = _incidents(store, "incidents.geojson")

    def layer(tool_name, **kw):
        return _client_id(tool_name, _run_tool(tools, tool_name, **kw))

    counted = layer("count_points_in_areas", points_file_id=points, areas_file_id=first["tracts"])
    assert layer("count_points_in_areas", points_file_id=points, areas_file_id=first["tracts"],
                 value_column="v") == counted
    km = layer("nearest_distance", from_file_id=points, to_file_id=first["city"], units="km")
    assert layer("nearest_distance", from_file_id=points, to_file_id=first["city"], units="m") == km
    assert layer("nearest_distance", from_file_id=points, to_file_id=first["city"], units="mi") != km


# --- spatial statistics -----------------------------------------------------------------------

def _areas(store, name):
    """A 5x5 lattice of contiguous squares with a clustered rate and two covariates.

    Each edge comes from ONE expression, so neighbours share it bit for bit. Written as
    -88.3 + c/100 against -88.29 + c/100 they do not (-88.3 + 0.01 is -88.28999999999999), the
    columns stop touching, and pygeoda's SKATER segfaults on the disconnected graph."""
    def x(c):
        return -88.3 + c * 0.01

    def y(r):
        return 40.0 + r * 0.01

    feats = [_feature({"zid": f"z{r}{c}", "rate": float(r + c), "x1": float(r),
                       "x2": float((r * c) % 3)}, _box(x(c), y(r), x(c + 1), y(r + 1)))
             for r in range(5) for c in range(5)]
    return _write(store, name, json.dumps({"type": "FeatureCollection", "features": feats}))


SPATIAL_CASES = [
    ("local_moran_lisa", "lisa",
     lambda g: {"file_id": g, "column": "rate", "permutations": 99},
     lambda g: {"file_id": g, "column": "rate", "permutations": 99, "weights": "rook"}),
    ("local_getis_ord", "gistar",
     lambda g: {"file_id": g, "column": "rate", "permutations": 99},
     lambda g: {"file_id": g, "column": "rate", "permutations": 99, "star": False}),
    ("spatial_regression", "regression",
     lambda g: {"file_id": g, "y_column": "rate", "x_columns": ["x1", "x2"]},
     lambda g: {"file_id": g, "y_column": "rate", "x_columns": ["x1"]}),
    ("regionalize", "regions",
     lambda g: {"file_id": g, "columns": ["rate", "x1"], "n_regions": 3},
     lambda g: {"file_id": g, "columns": ["rate", "x1"], "n_regions": 4}),
]


@pytest.mark.parametrize("tool_name,kind,same,different", SPATIAL_CASES,
                         ids=[case[0] for case in SPATIAL_CASES])
def test_a_regrounded_spatial_statistic_replaces_its_layer_and_a_different_one_adds(
        store, tool_name, kind, same, different):
    """A LISA map under queen weights and under rook weights wrote one filename, so they were
    one layer. The prefix assertion guards regionalize, which loops `for key in result`
    between computing its key and building its layer."""
    pytest.importorskip("esda")
    if tool_name == "regionalize":
        pytest.importorskip("pygeoda")
    from agent_runtime.analysis_spatial_stats_tools import make_spatial_stats_tools

    tools = {t.name: t for t in make_spatial_stats_tools()}
    areas, areas_again = _areas(store, "areas.geojson"), _areas(store, "areas_copy.geojson")
    _replaces_and_adds(
        tool_name, kind,
        _run_tool(tools, tool_name, **same(areas)),
        _run_tool(tools, tool_name, **same(areas_again), name="rate clusters again"),
        _run_tool(tools, tool_name, **different(areas)))


def test_weights_a_scheme_never_reads_do_not_split_a_layer(store):
    """Queen contiguity has no k, so k=8 changes nothing it draws, while knn at k=4 and at k=6
    are different neighbourhoods and must stay two layers. The order the model lists
    explanatory variables in changes no residual."""
    pytest.importorskip("esda")
    from agent_runtime.analysis_spatial_stats_tools import make_spatial_stats_tools

    tools = {t.name: t for t in make_spatial_stats_tools()}
    areas = _areas(store, "areas.geojson")

    def layer(tool_name, **kw):
        return _client_id(tool_name, _run_tool(tools, tool_name, file_id=areas, **kw))

    queen = layer("local_moran_lisa", column="rate", weights="queen", permutations=99)
    assert layer("local_moran_lisa", column="rate", weights="queen", k=8,
                 permutations=99) == queen
    assert layer("local_moran_lisa", column="rate", weights="knn", k=4, permutations=99) != \
        layer("local_moran_lisa", column="rate", weights="knn", k=6, permutations=99)
    assert layer("spatial_regression", y_column="rate", x_columns=["x2", "x1"]) == \
        layer("spatial_regression", y_column="rate", x_columns=["x1", "x2"])


# --- temporal ---------------------------------------------------------------------------------

TEMPORAL_CASES = [
    ("filter_by_time", "time_window",
     lambda e, f: {"file_id": e, "start": "2026-06-01", "end": "2026-06-30"},
     lambda e, f: {"file_id": e, "start": "2026-07-01", "end": "2026-07-31"}),
    ("compare_periods", "period_change",
     lambda e, f: {"file_id": e, "areas_file_id": f["tracts"], "period_a": "2026-06",
                   "period_b": "2026-07"},
     lambda e, f: {"file_id": e, "areas_file_id": f["tracts"], "period_a": "2026-07",
                   "period_b": "2026-06"}),
    ("temporal_hotspots", "temporal_hotspots",
     lambda e, f: {"file_id": e, "freq": "month", "cell_km": 1.0},
     lambda e, f: {"file_id": e, "freq": "month", "cell_km": 0.5}),
]


@pytest.mark.parametrize("tool_name,kind,same,different", TEMPORAL_CASES,
                         ids=[case[0] for case in TEMPORAL_CASES])
def test_a_regrounded_temporal_layer_replaces_its_layer_and_a_different_one_adds(
        store, fetched, tool_name, kind, same, different):
    """The temporal labels describe the request when unnamed, so a renamed repeat stacked, and
    temporal_hotspots, labelled by its latest period alone, merged grids of every cell size."""
    pytest.importorskip("geopandas")
    from agent_runtime.analysis_temporal_tools import make_temporal_tools

    tools = {t.name: t for t in make_temporal_tools()}
    first, again = fetched
    events, events_again = _incidents(store, "events.geojson"), _incidents(store, "copy.geojson")
    _replaces_and_adds(
        tool_name, kind,
        _run_tool(tools, tool_name, **same(events, first)),
        _run_tool(tools, tool_name, **same(events_again, again), name="june events again"),
        _run_tool(tools, tool_name, **different(events, first)))


def test_a_window_written_another_way_is_the_same_slice(store):
    """The window and the time column are keyed as RESOLVED: a month and its first-to-last
    dates are one window, and naming the column detection would pick is no new layer."""
    pytest.importorskip("geopandas")
    from agent_runtime.analysis_temporal_tools import make_temporal_tools

    tools = {t.name: t for t in make_temporal_tools()}
    events = _incidents(store, "events.geojson")
    month = _client_id("filter_by_time", _run_tool(tools, "filter_by_time", file_id=events,
                                                   start="2026-06", end="2026-06"))
    assert _client_id("filter_by_time", _run_tool(
        tools, "filter_by_time", file_id=events, start="2026-06-01", end="2026-06-30",
        time_column="when")) == month


# --- a code peer's files (layers_for_artifacts) -----------------------------------------------

def _peer_layers(directory, filename, features):
    """What a CLI code peer's run delivers after writing one GeoJSON: the peer wrapper's
    payload, as build_map_layers receives it."""
    from agent_runtime.map_layers import layers_for_artifacts

    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text(
        json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
    layers = layers_for_artifacts(directory, [{"filename": filename,
                                               "download_url": f"/agent/files/{directory.name}/download"}])
    assert len(layers) == 1, layers
    return "claude_run", json.dumps({"ok": True, "map_layers": layers})


def test_a_code_peers_file_is_one_layer_whatever_it_called_it(tmp_path):
    """A CLI peer has no add_map_layer, and the id of what it wrote came from the label, which
    is the filename it chose. A re-run under another name stacked, and a different result under
    the same name took the first one's place."""
    pytest.importorskip("pyogrio")
    run = _peer_layers(tmp_path / "turn1", "hospitals.geojson", [CHAMPAIGN_CITY])
    rerun = _peer_layers(tmp_path / "turn2", "hospitals_within_2km.geojson", [CHAMPAIGN_CITY])
    other = _peer_layers(tmp_path / "turn3", "hospitals.geojson", [URBANA_CITY])
    assert len(_on_screen(run, rerun)) == 1
    assert len(_on_screen(run, other)) == 2


def test_a_peers_file_drawn_again_by_add_map_layer_is_the_same_layer(store, tmp_path):
    """Both routes key a drawn file with drawn_layer_key, so the analysis peer redrawing what
    the code peer wrote, the way it is drawn anyway, replaces the peer's layer."""
    pytest.importorskip("geopandas")
    text = json.dumps({"type": "FeatureCollection", "features": [CHAMPAIGN_CITY]})
    peer = _peer_layers(tmp_path / "peer", "city.geojson", [CHAMPAIGN_CITY])
    stored = _write(store, "city.geojson", text)    # the copy the peer wrapper persists
    drawn = _geo_tool("add_map_layer").func(file_id=stored, render="auto")
    assert _client_id(*peer) == _client_id("add_map_layer", drawn)


# --- embedding and prediction layers --------------------------------------------------------

def _png_data_uri():
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (4, 3), (10, 20, 30, 255)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture
def embed_zones(store, monkeypatch):
    """embed_zones over the zones service stubbed at _svc, as test_rs_embed_zonal stubs it,
    answering for the six stub tracts with vectors that fall into two groups."""
    import agent_runtime.rs_embed_tools as T

    rows = [{"zone_id": f"170190001{i:02d}", "pixels": 100 + i, "area_km2": 1.0,
             "e000": float(i % 2), "e001": float(i // 3), "e002": 0.5} for i in range(6)]
    reply = {"ok": True, "model": "gse", "year": 2022, "zones": 6, "dim": 3, "rows": rows,
             "image": {"png": _png_data_uri(), "bounds": [-88.3, 40.0, -88.23, 40.01],
                       "size_px": [3, 4], "pixels_shown": 600, "colour": "PCA to RGB"},
             "meta": {"model": "gse", "dims": 3, "bands": [], "scale_m": 10.0,
                      "pixel_ground_m": 7.44, "tiles_planned": 1, "tiles_fetched": 1,
                      "tiles_capped": False, "zone_id_field": "GEOID", "zones_total": 6,
                      "zones_with_pixels": 6, "tile_errors": [], "pixel_size_warnings": []}}
    monkeypatch.setattr(T, "_svc", lambda *a, **k: reply)
    tool = {t.name: t for t in T.make_rs_embed_zonal_tools()}["embed_zones"]

    def run(**kw):
        out = json.loads(tool.func(**{"zone_id_field": "GEOID", "model": "gse", "clusters": 2,
                                      **kw}))
        assert out.get("ok"), out
        return out
    return run


def test_a_regrounded_embedding_of_the_same_zones_replaces_both_its_layers(boundary, embed_zones):
    """zone_content["file"] was the polygons' file_id. A re-ground that fetched the tracts again
    wrote them under a new one, and the same sweep stacked a second raster and a second group
    layer beside the first."""
    pytest.importorskip("geopandas")
    first = boundary(area="Champaign", level="county", state="IL", subdivide="tracts")["file_id"]
    again = boundary(area="Champaign", level="county", state="IL", subdivide="tracts",
                     output_name="tracts again")["file_id"]
    a = embed_zones(file_id=first)
    b = embed_zones(file_id=again, name="Champaign tracts, embedded again")
    assert [layer["render"] for layer in a["map_layers"]] == ["raster", "categories"]
    shown = _on_screen(("embed_zones", json.dumps(a)), ("embed_zones", json.dumps(b)))
    assert len(shown) == 2, shown
    later_year = _on_screen(("embed_zones", json.dumps(embed_zones(file_id=first, year=2023))))
    assert set(later_year).isdisjoint(shown), "another year is another embedding"


def test_a_period_in_months_or_in_dates_is_one_embedding(boundary, embed_zones):
    """The service is sent _iso_date's dates, so "2025-03".."2025-05" and
    "2025-03-01".."2025-05-31" are one composite. The period was keyed on the raw strings."""
    pytest.importorskip("geopandas")
    tracts = boundary(area="Champaign", level="county", state="IL", subdivide="tracts")["file_id"]
    months = embed_zones(file_id=tracts, start="2025-03", end="2025-05")
    dates = embed_zones(file_id=tracts, start="2025-03-01", end="2025-05-31")
    assert set(_on_screen(("embed_zones", json.dumps(months)))) == \
        set(_on_screen(("embed_zones", json.dumps(dates))))


def test_the_vectors_csv_records_the_request_that_made_it(boundary, embed_zones):
    """fit_zone_model reads this CSV, so embed_zones records what it holds. The key is that of
    the sweep, without `clusters`, because the vectors do not depend on how many groups were
    asked for, while the group layer does."""
    pytest.importorskip("geopandas")
    from agent_runtime.file_store import file_content_key

    first = boundary(area="Champaign", level="county", state="IL", subdivide="tracts")["file_id"]
    again = boundary(area="Champaign", level="county", state="IL", subdivide="tracts",
                     output_name="again")["file_id"]
    a, b = embed_zones(file_id=first), embed_zones(file_id=again)
    three = embed_zones(file_id=first, clusters=3)

    def vectors(out):
        return file_content_key(out["vectors_csv"]["file_id"])

    assert vectors(a) == vectors(b) == vectors(three)
    assert vectors(a).startswith("zone_vectors-")
    assert vectors(embed_zones(file_id=first, year=2023)) != vectors(a)
    assert a["map_layers"][1]["id"] != three["map_layers"][1]["id"], "k is a different grouping"


def _dem_file(tmp_path, values, bbox, name):
    """A GeoTIFF in the file store, written the way dem_for_region writes one."""
    import rasterio
    from rasterio.transform import from_bounds

    from agent_runtime import terrain_tools
    from agent_runtime.file_store import create_output_file_from_path

    h, w = values.shape
    path = tmp_path / name
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32",
                       crs=terrain_tools._wgs84(), nodata=float("nan"),
                       transform=from_bounds(*bbox, w, h)) as dst:
        dst.write(values.astype("float32"), 1)
    return create_output_file_from_path(path, filename=name)["file_id"]


def _zone_lattice(bbox, n=4):
    """n x n square zones tiling bbox, with GEOIDs, as test_zonal_to_fit_zone_model builds."""
    x0, y0, x1, y1 = bbox
    dx, dy = (x1 - x0) / n, (y1 - y0) / n
    return {"type": "FeatureCollection", "features": [
        _feature({"GEOID": f"1701900{r}{c}00"},
                 _box(x0 + c * dx, y0 + r * dy, x0 + (c + 1) * dx, y0 + (r + 1) * dy))
        for r in range(n) for c in range(n)]}


def test_a_regrounded_chain_from_a_dem_to_a_prediction_replaces_its_layers(store, tmp_path):
    """zonal_stats_for_raster keyed its layer on the raster's and the polygons' file_ids, and
    fit_zone_model on the vectors' and the polygons'. A re-ground writes every one of them again
    under new ids, so each step of the chain stacked a copy of itself."""
    pytest.importorskip("geopandas")
    import numpy as np

    from agent_runtime import rs_embed_tools, terrain_tools
    from agent_runtime.file_store import resolve_file_id

    zonal = {t.name: t for t in terrain_tools.make_terrain_tools()}["zonal_stats_for_raster"]
    fit = {t.name: t for t in rs_embed_tools.make_rs_embed_zonal_tools()}["fit_zone_model"]
    bbox = (-88.40, 40.00, -88.00, 40.40)
    ramp = np.tile(np.linspace(100.0, 900.0, 80), (80, 1))

    def chain(tag, **zonal_kw):
        raster = _dem_file(tmp_path, ramp, bbox, f"dem_{tag}.tif")
        zones = _write(store, f"zones_{tag}.geojson", json.dumps(_zone_lattice(bbox)))
        summarised = json.loads(zonal.func(raster_file_id=raster, polygons_file_id=zones,
                                           zone_id_field="GEOID", prefix="elev", **zonal_kw))
        assert summarised["ok"], summarised
        summarised["file_id"] = summarised["geojson"]["file_id"]
        labelled = json.loads(Path(resolve_file_id(summarised["file_id"])).read_text("utf-8"))
        rows = ["zone_id,e0,e1,e2"] + [
            f"{f['properties']['GEOID']},{f['properties']['elev_mean'] / 1000.0},"
            f"{(f['properties']['elev_mean'] / 1000.0) ** 2},0.5" for f in labelled["features"]]
        vectors = _write(store, f"vectors_{tag}.csv", "\n".join(rows))
        predicted = json.loads(fit.func(vectors_csv_file_id=vectors,
                                        polygons_file_id=summarised["file_id"],
                                        label_column="elev_mean", zone_id_field="GEOID",
                                        blocks=2, name=f"prediction {tag}"))
        assert predicted["ok"], predicted
        predicted["file_id"] = predicted["predictions_file_id"]
        return summarised, predicted

    zonal_1, fit_1 = chain("first")
    zonal_2, fit_2 = chain("again")
    assert _client_id("zonal_stats_for_raster", zonal_1) == _client_id("zonal_stats_for_raster", zonal_2)
    assert _client_id("fit_zone_model", fit_1) == _client_id("fit_zone_model", fit_2)
    # Which pixels a zone summarises is part of the analysis.
    zonal_3, _ = chain("edge pixels", all_touched=False)
    assert _client_id("zonal_stats_for_raster", zonal_3) != _client_id("zonal_stats_for_raster", zonal_1)


def test_a_terrain_layer_is_keyed_on_its_raster_not_only_its_bounds(store, tmp_path):
    """terrain_derivative and inundation_at_level keyed their layers on the raster's bbox. A DEM
    clipped to a shape keeps the grid and the bounds of the unclipped DEM, so the slope of each
    was one layer and the second replaced the first. So were two hillshades under two suns."""
    import numpy as np

    from agent_runtime import terrain_tools

    tools = {t.name: t for t in terrain_tools.make_terrain_tools()}
    bbox = (-88.30, 40.05, -88.20, 40.15)
    ramp = np.tile(np.linspace(200.0, 260.0, 40), (40, 1))
    cut = ramp.copy()
    cut[:10, :10] = np.nan
    whole = _dem_file(tmp_path, ramp, bbox, "dem.tif")
    whole_again = _dem_file(tmp_path, ramp, bbox, "dem_again.tif")
    clipped = _dem_file(tmp_path, cut, bbox, "dem_clipped.tif")

    def layer(tool_name, **kw):
        out = json.loads(tools[tool_name].func(**kw))
        assert out["ok"], out
        return _client_id(tool_name, out)

    slope = layer("terrain_derivative", raster_file_id=whole, kind="slope")
    assert layer("terrain_derivative", raster_file_id=whole_again, kind="slope",
                 name="slope again") == slope
    assert layer("terrain_derivative", raster_file_id=clipped, kind="slope") != slope
    assert layer("terrain_derivative", raster_file_id=whole, kind="hillshade", azimuth=315) != \
        layer("terrain_derivative", raster_file_id=whole, kind="hillshade", azimuth=45)
    flood = layer("inundation_at_level", raster_file_id=whole, level_m=230)
    assert layer("inundation_at_level", raster_file_id=whole_again, level_m=230) == flood
    assert layer("inundation_at_level", raster_file_id=whole, depth_above_min_m=30) == flood
    assert layer("inundation_at_level", raster_file_id=clipped, level_m=230) != flood


def test_a_dem_cut_to_two_shapes_with_one_bbox_is_two_layers(store, monkeypatch, tmp_path):
    """dem_for_region keyed a clipped DEM on `clipped=True`, not on the shape it was cut to.
    A square and the diamond inscribed in it share a bounding box, cut different pixels, and
    drew as one layer. An unclipped DEM keeps the id it always had."""
    import numpy as np
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds

    from agent_runtime import terrain_tools
    from agent_runtime.rs_embed_tools import _layer_id, _region_tag, _round_bbox

    w, s, e, n = -88.30, 40.05, -88.20, 40.15
    cx, cy = (w + e) / 2, (s + n) / 2
    square = _write(store, "square.geojson", json.dumps(
        {"type": "FeatureCollection", "features": [_feature({}, _box(w, s, e, n))]}))
    diamond = _write(store, "diamond.geojson", json.dumps(
        {"type": "FeatureCollection",
         "features": [_feature({}, [[cx, s], [e, cy], [cx, n], [w, cy], [cx, s]])]}))
    values = np.tile(np.linspace(200.0, 260.0, 64), (64, 1))

    def served(bbox, size):
        with MemoryFile() as mem:
            with mem.open(driver="GTiff", height=64, width=64, count=1, dtype="float32",
                          crs=terrain_tools._wgs84(), nodata=-999999.0,
                          transform=from_bounds(*bbox, 64, 64)) as dst:
                dst.write(values.astype("float32"), 1)
            return mem.read()

    monkeypatch.setattr(terrain_tools, "_fetch_dem", served)
    dem = {t.name: t for t in terrain_tools.make_terrain_tools()}["dem_for_region"]

    def run(**kw):
        out = json.loads(dem.func(**kw))
        assert out["ok"], out
        return out

    assert run(file_id=square)["map_layer"]["id"] != run(file_id=diamond)["map_layer"]["id"]
    plain = run(bbox=[w, s, e, n])
    assert plain["map_layer"]["id"] == _layer_id(
        "dem", _region_tag(None, [w, s, e, n]), bbox=_round_bbox(plain["region_bbox"]),
        size=512, clipped=False)


def test_a_recolour_that_cannot_name_its_layer_survives_a_reground(store, tmp_path):
    """align_embedding_colors takes over embed_region's layer when a package's manifest dates
    say which one it re-colours, and otherwise keys a layer of its own on the packages. That
    fallback digested their file_ids, so re-saving the same vectors gave every layer a new id."""
    import numpy as np

    import agent_runtime.rs_embed_tools as T
    from agent_runtime.file_store import create_output_file_from_path

    def package(stem, seed, bbox):
        grid = np.random.default_rng(seed).normal(size=(6, 8, 8)).astype(np.float32)
        meta = json.dumps({"geometry": {"type": "bbox", "minlon": bbox[0], "minlat": bbox[1],
                                        "maxlon": bbox[2], "maxlat": bbox[3]},
                           "models": [{"model": "gse", "dim": 6, "grid_saved": True}]})
        path = tmp_path / f"{stem}.npz"
        np.savez(path, grid__gse=grid, pooled__gse=grid.reshape(6, -1).mean(1), meta=np.array(meta))
        return path

    packages = [package("a", 0, [-88.246, 40.110, -88.234, 40.121]),
                package("b", 1, [-88.212, 40.108, -88.200, 40.119])]
    align = {t.name: t for t in T.make_rs_embed_tools()}["align_embedding_colors"]

    def ids(tag):
        file_ids = [create_output_file_from_path(p, filename=f"{p.stem}_{tag}.npz")["file_id"]
                    for p in packages]
        out = json.loads(align.func(file_ids=file_ids))
        assert out["ok"], out
        return [layer["id"] for layer in out["map_layers"]]

    first = ids("first")
    assert all("sharedpca" in i for i in first) and len(set(first)) == 2
    assert ids("again") == first
