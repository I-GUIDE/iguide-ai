"""fit_zone_model joins each zone by its id as TEXT, so a leading zero survives.

embed_zones writes every zone id into the vectors CSV exactly as the polygons hold it, and a
TIGER GEOID is text: California's tract '06037100000' keeps its leading zero in the file. fit()
read that file with pandas' type inference. When every id in a file is all digits, pandas reads
zone_id as int64, and astype(str) then turns '06037100000' into '6037100000', which no GEOID in
the polygons equals. Measured on 2026-10-03, on pandas 2.2.3 and 3.0.5, through embed_zones
(service stubbed) -> fit_zone_model:

  * 30 California tracts, or tracts of any state with FIPS 01-09: "only 0 zones have both a
    vector and a label";
  * 15 California and 15 Illinois tracts: the fit ran on the 15 Illinois tracts alone, said ok,
    and put no California tract on the prediction map.

Ids without a leading zero, such as Illinois's 17..., read back the same either way, and have to
fit exactly as they did. Everything here is offline: the vectors are synthetic, and the one
rs-embed service call is stubbed, as in test_fit_zone_model_column_collisions.py.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json

import numpy as np
import pandas as pd
import pytest

from agent_runtime import rs_embed_tools

# Enough zones to clear the fitter's floor of 12, as in the bug report's repro.
N = 30
DIMS = 4

CALIFORNIA = [f"0603710{i:04d}" for i in range(N)]      # Los Angeles County, FIPS 06037
ILLINOIS = [f"1703110{i:04d}" for i in range(N)]        # Cook County, FIPS 17031
# Tract NAMEs ('101.10' style) lose a trailing zero the same way: pandas reads a column of them
# as float64, and astype(str) gives '8300.1' back.
TRACT_NAMES = [f"{8300 + i}.{i % 9 + 1}0" for i in range(N)]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENT_PUBLIC_BASE_URL", "")
    return tmp_path


def _square(i, cols=6, x0=-118.30, y0=33.95, side=0.01):
    a, b = x0 + (i % cols) * side, y0 + (i // cols) * side
    return {"type": "Polygon",
            "coordinates": [[[a, b], [a + side, b], [a + side, b + side], [a, b + side], [a, b]]]}


def _store_file(tmp_path, name, text):
    from agent_runtime.file_store import create_output_file_from_path

    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return create_output_file_from_path(path, filename=name)["file_id"]


def _zones_service(pixels, area, feats):
    """/api/zones over the zones actually SENT, keyed by the identifier the request names."""
    def reply(path, body=None, **_kwargs):
        assert path == "/api/zones"
        field = body["zone_id_field"]
        ids = [f["properties"][field] for f in body["zones_geojson"]["features"]]
        rows = [{"zone_id": z, "pixels": pixels[z], "area_km2": area[z],
                 **{f"e{d:03d}": float(feats[z][d]) for d in range(DIMS)}} for z in ids]
        return {"ok": True, "rows": rows,
                "meta": {"model": "gse", "dims": DIMS, "bands": [], "scale_m": 10.0,
                         "pixel_ground_m": 7.8, "tiles_planned": 1, "tiles_fetched": 1,
                         "tiles_capped": False, "zone_id_field": field, "zones_total": len(ids),
                         "zones_with_pixels": len(ids), "tile_errors": [],
                         "pixel_size_warnings": []}}
    return reply


def _embedded(tmp_path, monkeypatch, ids, name, field="GEOID"):
    """embed_zones over a tract layer keyed by `field` that already carries the label `truth`.
    Returns the vectors CSV's file id, the tracts' file id, the label per id, and the pixel
    count the service gave each id. No two zones share a count, so the `pixels` a zone carries
    onto the prediction map says which zone's vector it was paired with."""
    rng = np.random.default_rng(5)
    feats = {z: rng.normal(size=DIMS) for z in ids}
    pixels = {z: 300 + 17 * i for i, z in enumerate(ids)}
    area = {z: round(0.25 + 0.0123 * i, 6) for i, z in enumerate(ids)}
    # With noise, so the scores depend on which zones joined and in what order: a label the
    # features predict exactly scores r2 = 1 on any subset of the zones.
    truth = {z: round(3.0 + feats[z][0] - 0.5 * feats[z][1] + rng.normal(scale=0.3), 6)
             for z in ids}
    layer = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": _square(i), "properties": {field: z, "truth": truth[z]}}
        for i, z in enumerate(ids)]}
    tracts_id = _store_file(tmp_path, name, json.dumps(layer))

    monkeypatch.setattr(rs_embed_tools, "_svc", _zones_service(pixels, area, feats))
    embed = {t.name: t for t in rs_embed_tools.make_rs_embed_zonal_tools()}["embed_zones"]
    out = json.loads(embed.func(file_id=tracts_id, zone_id_field=field, clusters=3))
    assert out.get("ok") is True, out
    return out["vectors_csv"]["file_id"], tracts_id, truth, pixels


def _fit(vectors_id, tracts_id, field="GEOID"):
    fit = {t.name: t for t in rs_embed_tools.make_rs_embed_zonal_tools()}["fit_zone_model"]
    return json.loads(fit.func(vectors_csv_file_id=vectors_id, polygons_file_id=tracts_id,
                               label_column="truth", zone_id_field=field))


def _prediction_bytes(out):
    from agent_runtime.file_store import resolve_file_id

    return resolve_file_id(out["predictions_file_id"]).read_bytes()


def _answer(out):
    """The fit's reply without the name of the file it wrote, which is new on every call."""
    return {**{k: v for k, v in out.items() if k not in ("predictions_file_id", "download_url")},
            "map_layer": {k: v for k, v in out["map_layer"].items() if k != "url"}}


@contextlib.contextmanager
def _read_as_before(monkeypatch):
    """fit()'s read_csv call as it was, the path alone, so pandas types zone_id by inference.
    Whatever the new call adds, the emulated one drops."""
    real = pd.read_csv
    with monkeypatch.context() as m:
        m.setattr(pd, "read_csv", lambda path, *args, **kwargs: real(path))
        yield


@pytest.mark.parametrize("ids, field", [(CALIFORNIA, "GEOID"),
                                        (CALIFORNIA[:15] + ILLINOIS[:15], "GEOID"),
                                        (TRACT_NAMES, "NAME")],
                         ids=["california", "california_and_illinois", "tract_names"])
def test_every_zone_meets_its_polygon_whatever_pandas_would_type_its_id(store, tmp_path,
                                                                       monkeypatch, ids, field):
    from agent_runtime.file_store import resolve_file_id

    vectors_id, tracts_id, truth, pixels = _embedded(tmp_path, monkeypatch, ids, "tracts.geojson",
                                                     field)
    written = csv.DictReader(io.StringIO(resolve_file_id(vectors_id).read_text(encoding="utf-8")))
    assert sorted(row["zone_id"] for row in written) == sorted(ids), \
        "the premise: embed_zones writes each id as the polygons hold it"

    out = _fit(vectors_id, tracts_id, field)
    assert out.get("ok") is True, out
    assert out["zones_fitted"] == N, "every zone has both a vector and a label"

    layer = json.loads(_prediction_bytes(out))
    pred = {f["properties"]["zone_id"]: f["properties"] for f in layer["features"]}
    assert set(pred) == set(ids), "every zone is on the prediction map, under its own id"
    for z in ids:
        assert pred[z]["pixels"] == pixels[z], "each zone is paired with its own vector"
        assert pred[z]["observed"] == pytest.approx(round(truth[z], 4)), \
            "each zone's label is its own polygon's"


def test_ids_without_a_leading_zero_fit_exactly_as_before(store, tmp_path, monkeypatch):
    """Illinois's ids are all digits too, so pandas read them as int64 before, but with no zero
    to lose they came back unchanged. The fit must be the same to the last byte of its map."""
    ca_vectors, ca_tracts, *_ = _embedded(tmp_path, monkeypatch, CALIFORNIA, "california.geojson")
    vectors_id, tracts_id, *_ = _embedded(tmp_path, monkeypatch, ILLINOIS, "illinois.geojson")

    with _read_as_before(monkeypatch):
        # The old read has to reproduce the reported failure, or the comparison below would
        # only compare the new read with itself.
        assert _fit(ca_vectors, ca_tracts).get("error") == \
            "only 0 zones have both a vector and a label"
        before = _fit(vectors_id, tracts_id)
    after = _fit(vectors_id, tracts_id)

    assert before.get("ok") is True, before
    assert after.get("ok") is True, after
    assert after["zones_fitted"] == N
    assert _prediction_bytes(after) == _prediction_bytes(before)
    assert _answer(after) == _answer(before)


# --- the join report (decided 2026-10-07): a vector that found no polygon is said out loud ---

def _numeric_geoid_layer(tmp_path, ids, truth, name):
    """The same tracts in a layer that stores GEOID as a NUMBER, as a CSV-derived layer often
    does: California's '06037100000' is 6037100000 there."""
    layer = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": _square(i),
         "properties": {"GEOID": int(z), "truth": truth[z]}} for i, z in enumerate(ids)]}
    return _store_file(tmp_path, name, json.dumps(layer))


def test_a_vector_that_found_no_polygon_is_reported_not_dropped_silently(store, tmp_path,
                                                                         monkeypatch):
    """The pairing S6.10 changed: text-id vectors fitted against a numeric-GEOID layer. The 15
    California vectors meet no polygon; the fit still runs on Illinois, and now says so."""
    ids = CALIFORNIA[:15] + ILLINOIS[:15]
    vectors_id, _, truth, _ = _embedded(tmp_path, monkeypatch, ids, "tracts.geojson")
    numeric = _numeric_geoid_layer(tmp_path, ids, truth, "numeric.geojson")

    out = _fit(vectors_id, numeric)
    assert out.get("ok") is True and out["zones_fitted"] == 15, out
    join = out["join"]
    assert (join["vectors"], join["matched"], join["vectors_without_a_polygon"]) == (30, 15, 15)
    assert join["example_unmatched_vector_ids"][0].startswith("06"), "the id as the vectors hold it"
    assert join["example_polygon_ids"][0].startswith("6037"), "and as the polygons hold it"
    assert out["warning"].startswith("15 of 30 vectors found no polygon"), out.get("warning")


def test_a_failed_join_says_which_ids_missed(store, tmp_path, monkeypatch):
    vectors_id, _, truth, _ = _embedded(tmp_path, monkeypatch, CALIFORNIA, "tracts.geojson")
    numeric = _numeric_geoid_layer(tmp_path, CALIFORNIA, truth, "numeric.geojson")

    out = _fit(vectors_id, numeric)
    assert out["error"] == "only 0 zones have both a vector and a label"
    assert out["join"]["vectors_without_a_polygon"] == N and out["join"]["matched"] == 0
    assert "'06037100000'" in out["warning"] and "'6037100000'" in out["warning"]


def test_a_clean_join_reports_everything_matched_and_warns_nothing(store, tmp_path, monkeypatch):
    vectors_id, tracts_id, *_ = _embedded(tmp_path, monkeypatch, ILLINOIS, "tracts.geojson")

    out = _fit(vectors_id, tracts_id)
    assert out.get("ok") is True
    assert out["join"] == {"vectors": N, "polygons": N, "matched": N,
                           "vectors_without_a_polygon": 0, "polygons_without_a_vector": 0}
    assert "warning" not in out


def test_polygons_without_vectors_are_counted_but_not_warned_about(store, tmp_path, monkeypatch):
    """Embedding a subset, or a zone with no pixels, leaves polygons without a vector. That is
    ordinary, so it is reported and nothing more."""
    vectors_id, _, truth, _ = _embedded(tmp_path, monkeypatch, ILLINOIS[:20], "subset.geojson")
    rng = np.random.default_rng(11)
    truth.update({z: float(rng.normal()) for z in ILLINOIS[20:]})
    layer = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": _square(i), "properties": {"GEOID": z, "truth": truth[z]}}
        for i, z in enumerate(ILLINOIS)]}
    all_tracts = _store_file(tmp_path, "all.geojson", json.dumps(layer))

    out = _fit(vectors_id, all_tracts)
    assert out.get("ok") is True and out["zones_fitted"] == 20
    assert out["join"]["polygons_without_a_vector"] == 10
    assert out["join"]["vectors_without_a_polygon"] == 0
    assert "warning" not in out
