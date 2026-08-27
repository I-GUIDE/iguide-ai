"""Zonal embedding aggregation: the parts that must be right without touching Earth Engine."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime.rs_embed_zonal_worker import _to_lonlat, _to_merc  # noqa: E402


def test_mercator_roundtrip_is_exact_enough_to_place_a_pixel():
    for lon, lat in [(-87.65, 41.93), (121.5, 31.2), (-0.1, 51.5), (0.0, 0.0)]:
        x, y = _to_merc(lon, lat)
        lon2, lat2 = _to_lonlat(x, y)
        # 1e-9 degrees is ~0.1 mm; a 10 m pixel needs nothing like this precision.
        assert abs(lon - lon2) < 1e-9 and abs(lat - lat2) < 1e-9


def test_mercator_metres_are_not_ground_metres():
    """The whole reason the affine is derived in 3857: a 'metre' here is 1/cos(lat) long.
    Reading scale_m as ground metres predicted a 170x111 grid where 224x146 came back."""
    lat = 40.07
    x0, y0 = _to_merc(-88.00, lat)
    x1, _ = _to_merc(-87.98, lat)
    span_3857 = x1 - x0
    span_ground = 0.02 * 111320 * math.cos(math.radians(lat))
    assert span_3857 / span_ground == pytest.approx(1 / math.cos(math.radians(lat)), rel=0.01)


def test_unknown_zone_id_field_lists_the_real_columns(tmp_path):
    """A silent fallback to row numbers made the vectors key on 0..n while the label join
    expected geoids, and the two never met. The caller must be told."""
    import geopandas as gpd
    from shapely.geometry import box

    from agent_runtime.rs_embed_zonal_worker import run

    gdf = gpd.GeoDataFrame({"geoid": ["a", "b"], "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)]},
                           crs="EPSG:4326")
    p = tmp_path / "z.geojson"
    gdf.to_file(p, driver="GeoJSON")

    out = run({"polygons_path": str(p), "zone_id_field": "GEOID", "model": "gse"})
    assert out["ok"] is False
    assert "GEOID" in out["error"]
    assert "geoid" in out["available_columns"], "must name the column that DOES exist"


def test_sums_and_counts_roll_up_exactly_where_means_would_not():
    """Why the worker stores sum+pixels rather than only a mean: a mean of means is wrong
    across unequal zones, which is the whole reason cross-scale aggregation is possible."""
    zones = [{"sum": [10.0, 20.0], "pixels": 100}, {"sum": [30.0, 30.0], "pixels": 400}]
    total_sum = [sum(z["sum"][i] for z in zones) for i in range(2)]
    total_px = sum(z["pixels"] for z in zones)
    rolled = [v / total_px for v in total_sum]

    mean_of_means = [sum(z["sum"][i] / z["pixels"] for z in zones) / 2 for i in range(2)]
    assert rolled == pytest.approx([0.08, 0.1])
    assert mean_of_means != pytest.approx(rolled), "the naive average is a different number"


def test_zonal_tools_are_registered_with_their_map_layer_contract():
    from agent_runtime.rs_embed_tools import make_rs_embed_zonal_tools
    from agent_runtime.supervisor.graph import _MAP_LAYER_TOOLS

    names = {t.name for t in make_rs_embed_zonal_tools()}
    assert names == {"embed_zones", "fit_zone_model"}
    # Both deliver layers, so a turn that ran them has genuinely put something on the map.
    assert names <= set(_MAP_LAYER_TOOLS)


def test_missing_rs_embed_interpreter_is_reported_not_guessed(monkeypatch):
    import agent_runtime.rs_embed_tools as T

    monkeypatch.setattr(T, "RS_EMBED_PYTHON", "")
    monkeypatch.setattr(T, "_zonal_python", lambda: None)
    out = T.run_zonal_worker({"polygons_path": "/nope"})
    assert out["ok"] is False and "rs_embed" in out["error"]
    assert "RS_EMBED_PYTHON" in out["hint"]
