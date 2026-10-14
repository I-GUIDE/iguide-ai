"""fit_zone_model refuses a label that holds one value in every zone, instead of fitting it.

A label with one value everywhere leaves nothing to predict, and the fitter fitted it anyway.
Measured on 2026-10-08 on pandas 2.2.3 and 3.0.5, with 30 zones:

  * 5.0 in every zone: ok, rmse 0, and r2 written as a bare NaN (0/0). The reply was not JSON,
    and no verdict caught it, because NaN compares false;
  * 0.1 in every zone: ok, blocked r2 0.55 and 33% skill over the baseline. The mean of thirty
    0.1s rounds, so the spread is a few 1e-17 instead of 0, and the scores are ratios of noise.
    1/3 in every zone claimed r2 0.85 and 61% skill.

Now the fit compares the values themselves and refuses when there is only one, naming it and
the numeric columns that do vary. An infinite label, which dropna kept and which made every score
NaN, now counts as no label at all. Everything here is offline: the vectors are synthetic, and
the one rs-embed service call is stubbed, as in test_fit_zone_model_unmatched_zones.py.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from agent_runtime import rs_embed_tools

N = 30
DIMS = 4
# Keyed by a NUMBER, so the list of alternatives can be seen to leave the key out.
ZONES = [str(i + 1) for i in range(N)]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENT_PUBLIC_BASE_URL", "")
    return tmp_path


def _square(i, cols=6, x0=-87.70, y0=41.80, side=0.01):
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


def _tracts(tmp_path, truth, name):
    """30 zones keyed by the integer `ZNUM`: the label `truth`, a `canopy` that varies, and a
    `state` that does not, the kind of column that holds one value for a whole layer."""
    layer = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": _square(i),
         "properties": {"ZNUM": int(z), "truth": truth[i], "canopy": 20.0 + i, "state": 17}}
        for i, z in enumerate(ZONES)]}
    return _store_file(tmp_path, name, json.dumps(layer))


def _strict(text):
    """The tool's reply parsed as strict JSON: NaN and Infinity are not JSON."""
    def refuse(constant):
        raise ValueError(f"the reply carries {constant}, which is not JSON")
    return json.loads(text, parse_constant=refuse)


def _embed_and_fit(tmp_path, monkeypatch, truth, zone_ids=None):
    """embed_zones over the tracts (or only `zone_ids` of them), then fit_zone_model on `truth`."""
    rng = np.random.default_rng(5)
    feats = {z: rng.normal(size=DIMS) for z in ZONES}
    pixels = {z: 300 + 17 * i for i, z in enumerate(ZONES)}
    area = {z: round(0.25 + 0.0123 * i, 6) for i, z in enumerate(ZONES)}
    tracts_id = _tracts(tmp_path, truth, "tracts.geojson")

    monkeypatch.setattr(rs_embed_tools, "_svc", _zones_service(pixels, area, feats))
    tools = {t.name: t for t in rs_embed_tools.make_rs_embed_zonal_tools()}
    embedded = json.loads(tools["embed_zones"].func(file_id=tracts_id, zone_id_field="ZNUM",
                                                    clusters=3, zone_ids=zone_ids))
    assert embedded.get("ok") is True, embedded
    return _strict(tools["fit_zone_model"].func(
        vectors_csv_file_id=embedded["vectors_csv"]["file_id"], polygons_file_id=tracts_id,
        label_column="truth", zone_id_field="ZNUM"))


@pytest.mark.parametrize("value, shown", [(5.0, "5"), (0.1, "0.1"), (1 / 3, "0.333333")],
                         ids=["exact", "rounds_to_noise", "one_third"])
def test_a_label_with_one_value_is_refused_and_named(store, tmp_path, monkeypatch, value, shown):
    out = _embed_and_fit(tmp_path, monkeypatch, [value] * N)   # _strict: no NaN gets through

    assert out["ok"] is False
    assert out["error"] == (f"label_column 'truth' has the same value, {shown}, in all {N} zones "
                            "that have both a vector and a label, so there is nothing to predict")
    assert "spatial_block_cv" not in out, "no score is reported for a fit that was not made"
    assert out["numeric_columns_that_vary"] == ["canopy"], \
        "the alternatives leave out the label, the key, and a column that is constant too"
    assert out["join"]["matched"] == N


def test_one_value_among_the_zones_that_met_is_enough_to_refuse(store, tmp_path, monkeypatch):
    """Only the zones with a vector are fitted. 25 embedded zones labelled 5.0 leave nothing to
    predict, whatever the five zones without a vector hold."""
    truth = [5.0] * 25 + [1.0, 2.0, 3.0, 4.0, 6.0]
    out = _embed_and_fit(tmp_path, monkeypatch, truth, zone_ids=ZONES[:25])

    assert out["ok"] is False
    assert "in all 25 zones that have both a vector and a label" in out["error"]
    assert out["join"]["polygons_without_a_vector"] == 5


def test_a_label_that_varies_at_all_is_fitted_as_before(store, tmp_path, monkeypatch):
    """The check compares the values, not their spread: one zone a ten-millionth away from the
    rest is a label with two values, and it is fitted."""
    out = _embed_and_fit(tmp_path, monkeypatch, [1.0] * (N - 1) + [1.0000001])

    assert out["ok"] is True, out
    assert out["zones_fitted"] == N
    assert "numeric_columns_that_vary" not in out


def test_an_infinite_label_is_no_label(store, tmp_path, monkeypatch):
    """'Infinity' is what pd.to_numeric makes of the text, and dropna kept it: every score came
    back NaN and the reply was not JSON. Now that zone counts as unlabelled, and the other 29
    are fitted."""
    truth = [str(1.0 + 0.25 * i) for i in range(N)]
    truth[3] = "Infinity"
    out = _embed_and_fit(tmp_path, monkeypatch, truth)       # _strict: no NaN or Infinity

    assert out["ok"] is True, out
    assert out["zones_fitted"] == N - 1
    assert out["observed_range"] == [1.0, 8.25]


@pytest.mark.parametrize("scale", [1e-200, 1e306], ids=["underflows", "overflows"])
def test_a_label_too_small_or_too_large_to_score_is_refused(store, tmp_path, monkeypatch, scale):
    """The label varies, but its spread cannot be squared in double precision: around 1e-200 it
    underflows to 0, so r2 was NaN; near 1e308 it overflows. Either way the reply carried NaN."""
    out = _embed_and_fit(tmp_path, monkeypatch, [scale * (i + 1) for i in range(N)])

    assert out["ok"] is False
    assert "cannot be scored in double precision" in out["error"]
    assert out["hint"].startswith("Rescale the label")

