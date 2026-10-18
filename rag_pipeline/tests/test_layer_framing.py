"""The model decides how a layer sits among the others (live, 2026-10-18).

Schools example, step 2: add_map_layer drew the 1-mile buffer in an asked-for blue at the colour
cap (190/255), on top of the 20 schools inside it, and the map fitted to it. The view was one
blue disc: no streets, the schools tinted under it. Whether a layer is the answer or a frame
around the answer is a judgement about what it is FOR, which only the model can make, so the
tool takes it: `outline`, `fill_opacity` and `beneath`.
"""
from __future__ import annotations

import json

import pytest

gpd = pytest.importorskip("geopandas")
from shapely.geometry import Point  # noqa: E402


@pytest.fixture()
def store(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    monkeypatch.delenv("AGENT_PUBLIC_BASE_URL", raising=False)
    from agent_runtime.file_store import create_output_file_from_path

    def put(gdf, name):
        path = tmp_path / name
        gdf.to_file(path, driver="GeoJSON")
        return create_output_file_from_path(path, filename=name)["file_id"]
    return put


def _add():
    from agent_runtime.langchain_geo_tools import make_langchain_geo_tools
    return {t.name: t for t in make_langchain_geo_tools()}["add_map_layer"]


def _buffer():
    site = gpd.GeoSeries([Point(-87.6233, 41.8827)], crs="EPSG:4326").to_crs(26916)
    return gpd.GeoDataFrame({"radius_m": [1609.344]}, geometry=site.buffer(1609.344),
                            crs=26916).to_crs(4326)


def test_a_framing_layer_carries_the_models_choice_to_the_map(store):
    from agent_runtime.map_layers import build_map_layers

    out = _add().invoke({"file_id": store(_buffer(), "buffer.geojson"), "render": "shapes",
                         "color": "blue", "fill_opacity": 0.15, "beneath": True})
    r = json.loads(out)
    assert r["ok"]
    assert r["map_layer"]["fill_opacity"] == 0.15 and r["map_layer"]["beneath"] is True
    (ml,) = build_map_layers("add_map_layer", out)
    assert ml["fill_opacity"] == 0.15 and ml["beneath"] is True and ml["outline"] is False


def test_outline_reaches_the_map_layer_event(store):
    from agent_runtime.map_layers import build_map_layers

    out = _add().invoke({"file_id": store(_buffer(), "buffer2.geojson"), "render": "shapes",
                         "outline": True})
    (ml,) = build_map_layers("add_map_layer", out)
    assert ml["outline"] is True and "fill_opacity" not in ml and ml["beneath"] is False


def test_left_unset_the_layer_is_as_before(store):
    from agent_runtime.map_layers import build_map_layers

    out = _add().invoke({"file_id": store(_buffer(), "buffer3.geojson"), "render": "shapes"})
    r = json.loads(out)
    assert not {"outline", "fill_opacity", "beneath"} & set(r["map_layer"])
    (ml,) = build_map_layers("add_map_layer", out)
    assert ml["outline"] is False and ml["beneath"] is False and "fill_opacity" not in ml


@pytest.mark.parametrize("bad", [1.5, -0.1])
def test_an_opacity_that_is_not_a_share_is_refused(store, bad):
    r = json.loads(_add().invoke({"file_id": store(_buffer(), "buffer4.geojson"),
                                  "render": "shapes", "fill_opacity": bad}))
    assert not r["ok"] and "0 to 1" in r["error"]


def test_outline_on_a_render_that_shows_values_in_its_fill_is_refused(store):
    gdf = _buffer().assign(rate=[3.2])
    r = json.loads(_add().invoke({"file_id": store(gdf, "rates.geojson"), "render": "choropleth",
                                  "column": "rate", "outline": True}))
    assert not r["ok"] and "fill" in r["error"] and "fill_opacity" in r["hint"]


def test_the_tool_tells_the_model_the_choice_is_its_own():
    doc = _add().description
    assert "outline=True" in doc and "fill_opacity" in doc and "beneath=True" in doc
