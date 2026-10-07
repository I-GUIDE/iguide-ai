"""fit_zone_model reads the label from the POLYGONS, whatever the label is called.

The vectors CSV that embed_zones writes has three columns besides the features: zone_id,
pixels and area_km2. The fitter merged all of it onto the polygon layer, and pandas does not
refuse a name both frames carry: it renames the two copies <name>_x and <name>_y, and the bare
name stops existing. That failed in two ways, one loud and one quiet:

  * a label called area_km2 or pixels raised KeyError. Seen live on 2026-10-03: asked to
    predict census-tract area from GSE embeddings, the agent added area_km2 to Arlington's 71
    tracts, fit_zone_model failed in 0.04 s, and the model wrote its own ridge regression in
    execute_code instead, so the user never saw the blocked-CV result;
  * the zone-groups layer embed_zones writes carries zone_id, pixels and area_km2 itself. Passed
    as the polygons with a numeric label of any other name, the fit went through, but the
    prediction map lost all three columns and support_pixels came back null.

Any other shared name failed the same way, such as a layer that already carried a copy of the
vectors' e000.. columns. Everything here is offline: the vectors are synthetic, and the one
rs-embed service call is stubbed.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from agent_runtime import rs_embed_tools

# Enough zones to clear the fitter's floor of 12. Thirty is the bug report's offline repro; the
# live case had 71 tracts.
N = 30
DIMS = 4


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENT_PUBLIC_BASE_URL", "")
    return tmp_path


def _square(i, cols, x0=-77.17, y0=38.83, side=0.01):
    a, b = x0 + (i % cols) * side, y0 + (i // cols) * side
    return {"type": "Polygon",
            "coordinates": [[[a, b], [a + side, b], [a + side, b + side], [a, b + side], [a, b]]]}


def _store_file(tmp_path, name, text):
    from agent_runtime.file_store import create_output_file_from_path

    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return create_output_file_from_path(path, filename=name)["file_id"]


def _geoids():
    return [f"5101310{i:04d}" for i in range(N)]


def _vectors_csv(tmp_path, geoids, feats, support=True):
    """The vectors CSV as embed_zones writes it: zone_id, the zones' SUPPORT (pixels and
    area_km2), then the features. Without `support`, the shape a hand-made CSV has."""
    pixels = {g: 400 + 37 * i for i, g in enumerate(geoids)}
    area = {g: round(0.2 + 0.0137 * i, 6) for i, g in enumerate(geoids)}
    header = (["zone_id"] + (["pixels", "area_km2"] if support else [])
              + [f"e{d:03d}" for d in range(DIMS)])
    rows = [",".join(header)]
    for i, g in enumerate(geoids):
        rows.append(",".join([g] + ([str(pixels[g]), f"{area[g]:.6f}"] if support else [])
                             + [f"{v:.6f}" for v in feats[i]]))
    name = "gse_zone_embeddings.csv" if support else "hand_made_vectors.csv"
    return _store_file(tmp_path, name, "\n".join(rows) + "\n"), pixels, area


def _tracts(tmp_path, geoids, properties, name="tracts.geojson"):
    """A polygon layer keyed by GEOID; `properties(i, geoid)` supplies the rest of each row."""
    layer = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": _square(i, cols=6),
         "properties": {"GEOID": g, **properties(i, g)}} for i, g in enumerate(geoids)]}
    return _store_file(tmp_path, name, json.dumps(layer))


def _predictions(out, columns=("zone_id", "pixels", "area_km2")):
    """The prediction layer fit_zone_model wrote, as {zone_id: properties}."""
    from agent_runtime.file_store import resolve_file_id

    layer = json.loads(resolve_file_id(out["predictions_file_id"]).read_text(encoding="utf-8"))
    props = [f["properties"] for f in layer["features"]]
    lost = set(columns) - set(props[0])
    assert not lost, f"the prediction map lost {sorted(lost)}"
    return {p["zone_id"]: p for p in props}


def _fit(**kwargs):
    fit = {t.name: t for t in rs_embed_tools.make_rs_embed_zonal_tools()}["fit_zone_model"]
    return json.loads(fit.func(**kwargs))


# What the answer IS, as opposed to how it is filed: the layer id and label carry the column
# name, so they are expected to differ between two fits of the same values.
_SCIENCE = ("zones_fitted", "features", "spatial_block_cv", "naive_random_split_cv",
            "baseline_predict_the_mean", "skill_vs_baseline_pct", "observed_range",
            "support_pixels")


@pytest.mark.parametrize("label", ["area_km2", "pixels"])
def test_a_label_named_like_a_vectors_column_is_the_polygons_label(store, tmp_path, label):
    rng = np.random.default_rng(7)
    feats = rng.normal(size=(N, DIMS))
    geoids = _geoids()
    # The CSV's pixels (400 and up) and area_km2 (0.2 to 0.6) are far from the labels below
    # (around 3), so each assertion can tell whose column it is reading.
    vectors_id, csv_pixels, csv_area = _vectors_csv(tmp_path, geoids, feats)

    # The tracts with the label the agent added, plus a twin of it under a name nothing else
    # uses, which is the control: the same values must give the same fit whatever they are called.
    truth = {g: round(3.0 + feats[i, 0] - 0.5 * feats[i, 1], 6) for i, g in enumerate(geoids)}
    polygons_id = _tracts(tmp_path, geoids, lambda i, g: {label: truth[g], "twin": truth[g]})

    out = _fit(vectors_csv_file_id=vectors_id, polygons_file_id=polygons_id,
               label_column=label, zone_id_field="GEOID", blocks=5)
    assert out.get("ok") is True, out
    assert out["zones_fitted"] == N and out["label_column"] == label

    pred = _predictions(out)
    assert set(pred) == set(geoids)
    for g in geoids:
        assert pred[g]["observed"] == pytest.approx(round(truth[g], 4)), \
            "the label must be the polygons' column, not the CSV's column of the same name"
        assert pred[g]["pixels"] == csv_pixels[g], "the CSV's support must survive under its name"
        assert pred[g]["area_km2"] == pytest.approx(csv_area[g])
    support = sorted(csv_pixels.values())
    assert out["support_pixels"] == {"min": support[0], "median": int(np.median(support)),
                                     "max": support[-1]}

    twin = _fit(vectors_csv_file_id=vectors_id, polygons_file_id=polygons_id,
                label_column="twin", zone_id_field="GEOID", blocks=5)
    assert twin.get("ok") is True, twin
    assert {k: out[k] for k in _SCIENCE} == {k: twin[k] for k in _SCIENCE}


def test_the_polygons_bring_only_their_shape_key_and_label(store, tmp_path):
    """Nothing else on the polygon layer reaches the join. A layer the vectors were joined onto
    earlier carries its own e000.. columns, which collided with the features (KeyError on the
    old fitter). A zone_id that is not the key, and the layer's own pixels and area_km2, must
    not reach the prediction map in place of the key and the CSV's support."""
    rng = np.random.default_rng(13)
    feats = rng.normal(size=(N, DIMS))
    stale = rng.normal(size=(N, DIMS))
    geoids = _geoids()
    vectors_id, csv_pixels, csv_area = _vectors_csv(tmp_path, geoids, feats)
    truth = {g: round(3.0 + feats[i, 0] - 0.5 * feats[i, 1], 6) for i, g in enumerate(geoids)}

    clean_id = _tracts(tmp_path, geoids, lambda i, g: {"truth": truth[g]}, name="clean.geojson")
    cluttered_id = _tracts(tmp_path, geoids, lambda i, g: {
        "truth": truth[g], "zone_id": f"Z{i}", "pixels": 9999, "area_km2": 99.0,
        **{f"e{d:03d}": float(stale[i, d]) for d in range(DIMS)}}, name="cluttered.geojson")

    clean = _fit(vectors_csv_file_id=vectors_id, polygons_file_id=clean_id,
                 label_column="truth", zone_id_field="GEOID")
    cluttered = _fit(vectors_csv_file_id=vectors_id, polygons_file_id=cluttered_id,
                     label_column="truth", zone_id_field="GEOID")
    assert clean.get("ok") is True, clean
    assert cluttered.get("ok") is True, cluttered
    assert {k: cluttered[k] for k in _SCIENCE} == {k: clean[k] for k in _SCIENCE}, \
        "the features are the CSV's; the layer's stale copy must not reach the fit"

    pred = _predictions(cluttered)
    assert set(pred) == set(geoids), "zone_id is the key, not the layer's own zone_id column"
    for g in geoids:
        assert pred[g]["pixels"] == csv_pixels[g]
        assert pred[g]["area_km2"] == pytest.approx(csv_area[g])


def test_support_is_the_vectors_own_or_none(store, tmp_path):
    """`pixels` and support_pixels say how many pixels each VECTOR averaged, which only the CSV
    knows. A CSV without them, which embed_zones did not write, used to borrow the polygon
    columns of the same names. Those describe the polygons, not the vectors, so now the
    prediction map carries no support and support_pixels is null."""
    rng = np.random.default_rng(17)
    feats = rng.normal(size=(N, DIMS))
    geoids = _geoids()
    vectors_id, _, _ = _vectors_csv(tmp_path, geoids, feats, support=False)
    polygons_id = _tracts(tmp_path, geoids, lambda i, g: {
        "truth": round(float(feats[i, 0]), 6), "pixels": 5000 + i, "area_km2": 2.5})

    out = _fit(vectors_csv_file_id=vectors_id, polygons_file_id=polygons_id,
               label_column="truth", zone_id_field="GEOID")
    assert out.get("ok") is True, out
    assert out["support_pixels"] is None
    pred = _predictions(out, columns=("zone_id",))
    assert not {"pixels", "area_km2"} & set(pred[geoids[0]])


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


@pytest.mark.parametrize("label", ["area_km2", "canopy_pct"])
def test_the_zone_groups_layer_embed_zones_writes_can_be_the_polygons(store, tmp_path,
                                                                      monkeypatch, label):
    """embed_zones -> fit_zone_model on its own two outputs. area_km2 is the groups layer's
    own column, which raised KeyError; canopy_pct is a label added to that layer, which fitted
    but put a prediction map with no zone_id, pixels or area_km2 on the map."""
    from agent_runtime.file_store import resolve_file_id

    rng = np.random.default_rng(11)
    geoids = _geoids()
    feats = {g: rng.normal(size=DIMS) for g in geoids}
    pixels = {g: 300 + 17 * i for i, g in enumerate(geoids)}
    # The groups layer rounds area_km2 to 4 decimals and the CSV keeps 6, so the prediction
    # map's area_km2 shows which copy the support came from. `observed` cannot show which copy
    # the label came from, because it is rounded to 4 decimals itself; the test above pins that.
    area = {g: round(0.250037 + 0.0123 * i, 6) for i, g in enumerate(geoids)}   # ...37 every time
    assert all(round(a, 4) != a for a in area.values())

    tracts_id = _tracts(tmp_path, geoids, lambda i, g: {})
    monkeypatch.setattr(rs_embed_tools, "_svc", _zones_service(pixels, area, feats))
    embed = {t.name: t for t in rs_embed_tools.make_rs_embed_zonal_tools()}["embed_zones"]
    embedded = json.loads(embed.func(file_id=tracts_id, zone_id_field="GEOID", clusters=3))
    assert embedded.get("ok") is True, embedded

    # The groups layer is reachable the way the agent reaches it: by the file id in its URL.
    layers = embedded.get("map_layers") or [embedded["map_layer"]]
    url = next(layer["url"] for layer in layers if layer["render"] == "categories")
    groups_id = re.search(r"/files/([^/]+)/download", url).group(1)
    groups = json.loads(resolve_file_id(groups_id).read_text(encoding="utf-8"))
    assert {"zone_id", "pixels", "area_km2"} <= set(groups["features"][0]["properties"]), \
        "the premise: the groups layer carries the CSV's non-feature columns itself"

    if label == "canopy_pct":
        # The agent's step in between: a value per zone, added to the layer and saved.
        for f in groups["features"]:
            f["properties"]["canopy_pct"] = round(
                40 + 10 * float(feats[f["properties"]["zone_id"]][0]), 4)
        groups_id = _store_file(tmp_path, "groups_with_canopy.geojson", json.dumps(groups))
    truth = {f["properties"]["zone_id"]: f["properties"][label] for f in groups["features"]}

    out = _fit(vectors_csv_file_id=embedded["vectors_csv"]["file_id"],
               polygons_file_id=groups_id, label_column=label, zone_id_field="zone_id")
    assert out.get("ok") is True, out
    assert out["zones_fitted"] == N

    pred = _predictions(out)
    assert set(pred) == set(geoids), "zone_id must survive onto the prediction map"
    for g in geoids:
        assert pred[g]["observed"] == pytest.approx(truth[g])
        assert pred[g]["pixels"] == pixels[g]
        assert pred[g]["area_km2"] == pytest.approx(area[g], abs=1e-9), \
            "the support is the CSV's six-decimal area, not the groups layer's rounded copy"
    support = sorted(pixels.values())
    assert out["support_pixels"] == {"min": support[0], "median": int(np.median(support)),
                                     "max": support[-1]}
