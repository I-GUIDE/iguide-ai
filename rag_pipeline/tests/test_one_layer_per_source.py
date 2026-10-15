"""One map layer per source file (user, 2026-10-15).

Live, Chicago demo step 3: regionalize drew the six SKATER regions, then the code peer called
add_map_layer on regionalize's own output file to rename it as a categorical layer, and the map
listed two identical SKATER layers. Each producer built its own id, and add_map_layer keyed on
the view (render, column) as well as the file. Now a layer's id is the content key of the file
it shows (map_layers.source_layer_id), applied where every descriptor becomes a `map_layer`
event, so drawing a file again restyles and renames its layer in place.
"""
from __future__ import annotations

import json

import pytest

gpd = pytest.importorskip("geopandas")
pytest.importorskip("libpysal")
from shapely.geometry import box  # noqa: E402


@pytest.fixture()
def lattice(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    monkeypatch.delenv("AGENT_PUBLIC_BASE_URL", raising=False)
    from agent_runtime.file_store import create_output_file_from_path

    cells = [box(-88 + c / 100, 40 + r / 100, -88 + (c + 1) / 100, 40 + (r + 1) / 100)
             for r in range(6) for c in range(6)]
    gdf = gpd.GeoDataFrame({"rate": [float((i % 6) + (i // 6) * 2) for i in range(36)]},
                           geometry=cells, crs="EPSG:4326")
    path = tmp_path / "areas.geojson"
    gdf.to_file(path, driver="GeoJSON")
    return create_output_file_from_path(path, filename="areas.geojson")["file_id"]


def _on_screen(*results):
    """What the map shows: the client replaces a layer whose id matches and adds one that does
    not (map-ui-prototype App.tsx), after every result crosses build_map_layers."""
    from agent_runtime.map_layers import build_map_layers

    shown = {}
    for tool, out in results:
        for layer in build_map_layers(tool, out, qa=False):
            shown[layer["id"]] = layer["label"]
    return shown


def _tool(name):
    from agent_runtime.analysis_spatial_stats_tools import make_spatial_stats_tools
    from agent_runtime.langchain_geo_tools import make_langchain_geo_tools

    tools = {t.name: t for t in [*make_spatial_stats_tools(), *make_langchain_geo_tools()]}
    return tools[name]


def test_redrawing_regionalize_output_replaces_its_layer(lattice):
    """The Chicago case: regionalize, then add_map_layer on its output with a new name."""
    pytest.importorskip("pygeoda")
    reg = _tool("regionalize").invoke({"file_id": lattice, "columns": ["rate"],
                                       "n_regions": 3, "method": "skater"})
    r = json.loads(reg)
    assert r["ok"], r
    regions_file = r["map_layer"]["url"].split("/files/")[1].split("/")[0]
    redraw = _tool("add_map_layer").invoke({"file_id": regions_file, "render": "categories",
                                            "column": "region",
                                            "name": "SKATER 3 regions by rate"})
    assert json.loads(redraw)["ok"], redraw
    shown = _on_screen(("regionalize", reg), ("add_map_layer", redraw))
    assert len(shown) == 1, shown
    # The model is told, so it does not report the replaced layer as still on the map (it did,
    # live, before this note).
    assert "REPLACED" in json.loads(redraw)["layer_identity"]
    assert list(shown.values()) == ["SKATER 3 regions by rate"], "the redraw renames it"


def test_a_restyle_is_the_same_layer_and_another_file_is_another(lattice, tmp_path):
    from agent_runtime.file_store import create_output_file_from_path

    add = _tool("add_map_layer")
    a = add.invoke({"file_id": lattice, "render": "choropleth", "column": "rate", "name": "Rate"})
    b = add.invoke({"file_id": lattice, "render": "shapes", "name": "Areas"})
    other = tmp_path / "other.geojson"
    gpd.GeoDataFrame({"v": [1.0, 2.0]}, geometry=[box(-87, 41, -86.9, 41.1),
                                                  box(-86.9, 41, -86.8, 41.1)],
                     crs="EPSG:4326").to_file(other, driver="GeoJSON")
    c = add.invoke({"file_id": create_output_file_from_path(other, filename="other.geojson")
                    ["file_id"], "render": "choropleth", "column": "v", "name": "Other"})
    shown = _on_screen(("add_map_layer", a), ("add_map_layer", b), ("add_map_layer", c))
    assert sorted(shown.values()) == ["Areas", "Other"], shown


def test_a_code_peer_file_and_add_map_layer_of_it_are_one_layer(lattice, tmp_path):
    """A toolless peer's file gets its layer from layers_for_artifacts; drawing the same file
    with add_map_layer later lands on that layer."""
    from agent_runtime.file_store import resolve_file_id
    from agent_runtime.map_layers import layers_for_artifacts

    path = resolve_file_id(lattice)
    peer = layers_for_artifacts(path.parent, [{"filename": path.name,
                                               "download_url": f"/agent/files/{lattice}/download"}])
    assert len(peer) == 1
    redraw = _tool("add_map_layer").invoke({"file_id": lattice, "render": "choropleth",
                                            "column": "rate", "name": "Rate"})
    shown = _on_screen(("code", json.dumps({"map_layers": peer})), ("add_map_layer", redraw))
    assert len(shown) == 1, shown


def test_a_tool_id_cannot_split_a_file_into_two_layers(lattice):
    """The rule is applied where the event is built, so a producer that sets its own id (here a
    made-up one) still lands on the file's layer."""
    from agent_runtime.map_layers import build_map_layer, file_layer_id

    built = build_map_layer("some_tool", {"map_layer": {
        "url": f"/agent/files/{lattice}/download", "label": "x", "render": "shapes",
        "id": "embed-something-else"}}, qa=False)
    assert built["id"] == file_layer_id(lattice)


def test_inline_and_external_layers_keep_their_ids():
    from agent_runtime.map_layers import build_map_layer

    built = build_map_layer("t", {"map_layer": {"url": "https://tiles.example/x.png",
                                                "label": "x", "render": "shapes",
                                                "id": "external-1"}}, qa=False)
    assert built["id"] == "external-1"
