"""`max_tiles`: uncapped by default, and 0 means zero.

Two defects lived here. The tool defaulted to `max_tiles=24`, so every sweep was silently
partial; and the request builder gated on truthiness, so an explicit `max_tiles=0` was dropped
from the body, the service default of None applied, and the sweep ran COMPLETELY UNCAPPED --
the exact opposite of what 0 asks for.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _zone_layer(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box

    gdf = gpd.GeoDataFrame({"geoid": ["a", "b"]},
                           geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs="EPSG:4326")
    p = tmp_path / "zones.geojson"
    gdf.to_file(p, driver="GeoJSON")
    return p


def _reply(**meta_over):
    meta = {"model": "gse", "dims": 2, "bands": [], "scale_m": 10.0, "pixel_ground_m": 7.44,
            "tile_px": 256, "tiles_planned": 90, "tiles_needed": 2, "tiles_fetched": 2,
            "tiles_skipped_by_cap": 0, "tiles_capped": False, "zone_id_field": "geoid",
            "zones_total": 2, "zones_with_pixels": 2, "tile_errors": [],
            "pixel_size_warnings": []}
    meta.update(meta_over)
    return {"ok": True, "model": "gse", "year": 2022, "zones": 2, "dim": 2,
            "rows": [{"zone_id": z, "pixels": 100, "area_km2": 1.0, "e000": 0.1, "e001": 0.2}
                     for z in ("a", "b")],
            "meta": meta}


def _sent_body(tmp_path, monkeypatch, payload):
    import agent_runtime.rs_embed_tools as T

    sent = {}

    def _capture(path, body=None, **kw):
        sent["body"] = body
        return _reply()

    monkeypatch.setattr(T, "_svc", _capture)
    T.run_zonal_worker({"polygons_path": str(_zone_layer(tmp_path)),
                        "zone_id_field": "geoid", **payload})
    return sent["body"]


def test_embed_zones_sets_no_cap_by_default():
    import agent_runtime.rs_embed_tools as T

    tool = {t.name: t for t in T.make_rs_embed_zonal_tools()}["embed_zones"]
    default = inspect.signature(tool.func).parameters["max_tiles"].default
    assert default is None, "a default cap makes every sweep silently partial"


def test_no_cap_is_sent_when_none_was_asked_for(tmp_path, monkeypatch):
    assert "max_tiles" not in _sent_body(tmp_path, monkeypatch, {})
    assert "max_tiles" not in _sent_body(tmp_path, monkeypatch, {"max_tiles": None})


def test_a_cap_of_zero_reaches_the_service(tmp_path, monkeypatch):
    """0 is falsy: it used to be dropped, and the sweep then ran with NO cap at all."""
    assert _sent_body(tmp_path, monkeypatch, {"max_tiles": 0}).get("max_tiles") == 0


def test_a_positive_cap_reaches_the_service(tmp_path, monkeypatch):
    assert _sent_body(tmp_path, monkeypatch, {"max_tiles": 7})["max_tiles"] == 7


def test_the_coverage_counters_survive_the_hand_off(tmp_path, monkeypatch):
    import agent_runtime.rs_embed_tools as T

    monkeypatch.setattr(T, "_svc", lambda *a, **k: _reply(
        tiles_needed=35, tiles_fetched=24, tiles_skipped_by_cap=11, tiles_capped=True))
    out = T.run_zonal_worker({"polygons_path": str(_zone_layer(tmp_path)),
                              "zone_id_field": "geoid"})
    assert out["tiles_needed"] == 35
    assert out["tiles_skipped_by_cap"] == 11
    assert out["tiles_capped"] is True


def test_truncated_counts_the_tiles_the_zones_touch_not_the_whole_grid(tmp_path, monkeypatch):
    """"24 of 35" used to pair the CAP with the bounding grid, so it read as lost coverage
    even on sweeps that had fetched every tile they needed."""
    import geopandas as gpd
    from shapely.geometry import box

    import agent_runtime.file_store as FS
    import agent_runtime.langchain_geo_tools as G
    import agent_runtime.rs_embed_tools as T

    gdf = gpd.GeoDataFrame({"geoid": ["a"]}, geometry=[box(-87.61, 41.82, -87.60, 41.83)],
                           crs="EPSG:4326")
    src = tmp_path / "one.geojson"
    gdf.to_file(src, driver="GeoJSON")

    monkeypatch.setattr(G, "_stage_vector_source", lambda *a, **k: (str(src), None))
    monkeypatch.setattr(G, "_index_attached", lambda *a, **k: {})
    monkeypatch.setattr(FS, "create_output_file_from_path",
                        lambda p, filename=None: {"file_id": f"id_{filename}",
                                                  "filename": filename,
                                                  "download_url": f"/f/{filename}",
                                                  "size_bytes": 10})
    monkeypatch.setattr(T, "_svc", lambda *a, **k: {
        "ok": True, "model": "gse", "year": 2022, "zones": 1, "dim": 2,
        "rows": [{"zone_id": "a", "pixels": 500, "area_km2": 0.4, "e000": 0.1, "e001": 0.2}],
        "meta": {"model": "gse", "dims": 2, "bands": [], "scale_m": 10.0,
                 "pixel_ground_m": 7.44, "tiles_planned": 1140, "tiles_needed": 35,
                 "tiles_fetched": 24, "tiles_skipped_by_cap": 11, "tiles_capped": True,
                 "zone_id_field": "geoid", "zones_total": 1, "zones_with_pixels": 1,
                 "tile_errors": [], "pixel_size_warnings": []}})

    tool = {t.name: t for t in T.make_rs_embed_zonal_tools()}["embed_zones"]
    out = json.loads(tool.func(file_id="file_x", zone_id_field="geoid", max_tiles=24))

    msg = out["truncated"]
    assert "24 of the 35" in msg, msg
    assert "11" in msg, msg
    assert "1140" not in msg, "the bounding grid is not the denominator a reader should see"
