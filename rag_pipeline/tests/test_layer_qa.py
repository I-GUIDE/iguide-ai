"""Tests for agent_runtime/layer_qa.py — does a delivered visual actually show anything?

Every case here was observed as a "delivered" result that told the user nothing: a flat
choropleth, a palette that matched no data, geometry off the edge of the world, a blank PNG.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import pytest
from shapely.geometry import Polygon

from agent_runtime.layer_qa import inspect_geojson, inspect_image

SIDE = 4


def _grid(**cols):
    polys = [Polygon([(j, i), (j + 1, i), (j + 1, i + 1), (j, i + 1)])
             for i in range(SIDE) for j in range(SIDE)]
    return gpd.GeoDataFrame(cols, geometry=polys, crs="EPSG:4326")


def _write(tmp_path, gdf, name="layer.geojson"):
    p = tmp_path / name
    gdf.to_file(p, driver="GeoJSON")
    return str(p)


def test_a_choropleth_over_a_constant_column_is_flagged(tmp_path):
    path = _write(tmp_path, _grid(pop=[7] * SIDE * SIDE))
    qa = inspect_geojson(path, render="choropleth", style_by="pop")
    assert qa["ok"] is False
    assert "single distinct value" in " ".join(qa["problems"])


def test_a_choropleth_over_a_varying_column_passes(tmp_path):
    path = _write(tmp_path, _grid(pop=list(range(SIDE * SIDE))))
    assert inspect_geojson(path, render="choropleth", style_by="pop")["ok"] is True


def test_class_names_sent_down_a_numeric_ramp_are_flagged(tmp_path):
    """render='choropleth' on a class-name column: float() fails for every feature."""
    path = _write(tmp_path, _grid(cls=["High-High", "Low-Low"] * (SIDE * SIDE // 2)))
    qa = inspect_geojson(path, render="choropleth", style_by="cls")
    assert qa["ok"] is False
    assert "no numeric values" in " ".join(qa["problems"])


def test_a_legend_matching_no_data_value_is_flagged(tmp_path):
    path = _write(tmp_path, _grid(cls=["A", "B"] * (SIDE * SIDE // 2)))
    qa = inspect_geojson(path, render="categories", style_by="cls",
                         legend=[{"label": "High-High"}, {"label": "Low-Low"}])
    assert qa["ok"] is False
    assert "no legend label matches" in " ".join(qa["problems"])


def test_a_matching_legend_passes(tmp_path):
    path = _write(tmp_path, _grid(cls=["A", "B"] * (SIDE * SIDE // 2)))
    qa = inspect_geojson(path, render="categories", style_by="cls",
                         legend=[{"label": "A"}, {"label": "B"}])
    assert qa["ok"] is True


def test_projected_metres_labelled_as_lonlat_are_flagged(tmp_path):
    """The classic: a UTM layer written as EPSG:4326 lands thousands of degrees off-map."""
    polys = [Polygon([(x * 1000 + 400_000, y * 1000 + 4_600_000)
                      for x, y in [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]])
             for x in range(2) for y in range(2)]
    gdf = gpd.GeoDataFrame({"v": [1, 2, 3, 4]}, geometry=polys, crs="EPSG:4326")
    qa = inspect_geojson(_write(tmp_path, gdf), render="shapes")
    assert qa["ok"] is False
    assert "outside lon/lat range" in " ".join(qa["problems"])


def test_a_missing_style_column_is_flagged(tmp_path):
    path = _write(tmp_path, _grid(pop=list(range(SIDE * SIDE))))
    qa = inspect_geojson(path, render="choropleth", style_by="not_a_column")
    assert qa["ok"] is False
    assert "not in the written layer" in " ".join(qa["problems"])


def test_an_empty_layer_is_flagged(tmp_path):
    gdf = gpd.GeoDataFrame({"pop": []}, geometry=[], crs="EPSG:4326")
    path = tmp_path / "empty.geojson"
    gdf.to_file(path, driver="GeoJSON")
    qa = inspect_geojson(str(path))
    assert qa["ok"] is False
    assert "no features" in " ".join(qa["problems"])


def test_a_blank_png_is_flagged(tmp_path):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    out = tmp_path / "blank.png"
    fig.savefig(out, dpi=80)
    plt.close(fig)
    qa = inspect_image(str(out))
    assert qa["ok"] is False
    assert "effectively blank" in " ".join(qa["problems"])


def test_a_real_plot_passes(tmp_path):
    fig, ax = plt.subplots()
    ax.scatter(range(20), range(20), c=range(20), cmap="viridis")
    out = tmp_path / "real.png"
    fig.savefig(out, dpi=80)
    plt.close(fig)
    qa = inspect_image(str(out))
    assert qa["ok"] is True
    assert qa["distinct_colors"] > 10


def test_a_missing_file_never_raises():
    """A checker that breaks must not break the delivery it was inspecting."""
    assert inspect_geojson("/nope/missing.geojson")["ok"] is True
    assert inspect_image("/nope/missing.png")["ok"] is True


def test_inspect_artifacts_finds_the_blank_outputs_a_cli_peer_leaves(tmp_path):
    """A sandboxed CLI peer has no tools, so add_map_layer never sees what it wrote and
    nothing between it and the user looks at the output. Its own summary will not mention
    that the figure came out blank — it did not look either."""
    import json as _json

    from agent_runtime.layer_qa import inspect_artifacts

    (tmp_path / "empty.geojson").write_text(
        _json.dumps({"type": "FeatureCollection", "features": []}))
    (tmp_path / "notes.txt").write_text("nothing to check here")
    try:
        from PIL import Image
        Image.new("RGB", (40, 40), (255, 255, 255)).save(tmp_path / "plot.png")
        expect_image = True
    except ImportError:
        expect_image = False

    found = inspect_artifacts(str(tmp_path), ["empty.geojson", "notes.txt", "plot.png",
                                              "missing.geojson"])
    by_file = {f["file"]: f["problems"] for f in found}

    assert "empty.geojson" in by_file
    assert any("no features" in p for p in by_file["empty.geojson"])
    if expect_image:
        assert "plot.png" in by_file, "an all-white figure is the classic silent failure"
    assert "notes.txt" not in by_file, "a text file has nothing to check"
    assert "missing.geojson" not in by_file, "a name with no file is not a finding"


def test_inspect_artifacts_is_quiet_when_the_output_is_fine(tmp_path):
    """Empty means nothing detectable is wrong, and it must stay empty for real output —
    a checker that cries wolf gets ignored, which is worse than not having one."""
    import json as _json

    from agent_runtime.layer_qa import inspect_artifacts

    (tmp_path / "ok.geojson").write_text(_json.dumps({
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"v": 1},
             "geometry": {"type": "Point", "coordinates": [-87.6, 41.8]}},
            {"type": "Feature", "properties": {"v": 2},
             "geometry": {"type": "Point", "coordinates": [-87.5, 41.9]}},
        ]}))
    assert inspect_artifacts(str(tmp_path), ["ok.geojson"]) == []
    assert inspect_artifacts(str(tmp_path / "nope"), ["ok.geojson"]) == []


def test_the_llm_trace_names_the_model_not_the_langchain_class():
    """AnvilGPT, vLLM and every other OpenAI-compatible endpoint arrive as ChatOpenAI, so the
    trace said "ChatOpenAI started with 2 message(s)" while gpt-oss:120b or qwen3.6:27b was
    answering. The transport does not tell you who answered — the invocation params do."""
    from agent_runtime.streaming_trace import StreamingTraceCallbackHandler as H

    serialized = {"name": "ChatOpenAI"}
    assert H._model_label(serialized, {"invocation_params": {"model": "gpt-oss:120b"}}) \
        == "gpt-oss:120b"
    assert H._model_label(serialized, {"invocation_params": {"model_name": "qwen3.6:27b"}}) \
        == "qwen3.6:27b"
    assert H._model_label(serialized, {"metadata": {"ls_model_name": "claude-sonnet-5"}}) \
        == "claude-sonnet-5"
    # No params at all: the class name is still better than nothing.
    assert H._model_label(serialized, {}) == "ChatOpenAI"
    assert H._model_label(None, {}) == "chat_model"
