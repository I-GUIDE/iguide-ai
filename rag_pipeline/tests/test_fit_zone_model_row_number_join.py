"""fit_zone_model refuses a row-number join that the vectors cannot have been keyed by.

Without zone_id_field, both embed_zones and fit_zone_model key zones by row number: the service
returns zone "i" for the polygon in row i, and the fitter pairs polygon i with the vector keyed
"i". That holds for the layer embed_zones read. It does not hold for the zone-groups layer
embed_zones puts on the map, which keeps only the zones that received pixels: after a zone with
none, the layer's row numbers and its zones part ways, and every later zone was paired with
another zone's vector. Measured with 30 zones, the gap at row 3 and a label the vectors
determine exactly: 25 of the 28 zones fitted were mispaired, and the fit reported no skill.
Its prediction map carried each matched vector's zone_id, pixels and area_km2, so it looked
complete. Districts numbered 1 to 30, embedded by that column and fitted without naming it,
went the same way: all 28 zones fitted were mispaired.

The fitter now refuses whenever the vectors name a zone the polygons have no row for, and its
hint names the column that holds the vectors' ids. The layer the vectors were numbered by never
shows that sign, so it still joins by row number, beside a zone_id column of its own that means
something else. Everything here is offline: the vectors are synthetic, and the one rs-embed
service call is stubbed.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from agent_runtime import rs_embed_tools

N = 30
DIMS = 4
GAP = 3              # the zone no tile reached, as in the measured case
COLS, X0, Y0, SIDE = 6, -77.17, 38.83, 0.01
COEF = np.array([2.0, -1.0, 0.5, 0.25])


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENT_PUBLIC_BASE_URL", "")
    return tmp_path


def _square(i):
    a, b = X0 + (i % COLS) * SIDE, Y0 + (i // COLS) * SIDE
    return {"type": "Polygon",
            "coordinates": [[[a, b], [a + SIDE, b], [a + SIDE, b + SIDE], [a, b + SIDE], [a, b]]]}


def _row_of(geometry):
    """Which of the N input polygons a feature's shape is, from where it sits on the grid."""
    ring = geometry["coordinates"][0]
    cx = (min(p[0] for p in ring) + max(p[0] for p in ring)) / 2
    cy = (min(p[1] for p in ring) + max(p[1] for p in ring)) / 2
    return round((cy - Y0) / SIDE - 0.5) * COLS + round((cx - X0) / SIDE - 0.5)


def _store_file(tmp_path, name, obj):
    from agent_runtime.file_store import create_output_file_from_path

    path = tmp_path / name
    path.write_text(json.dumps(obj), encoding="utf-8")
    return create_output_file_from_path(path, filename=name)["file_id"]


def _layer(properties, rows=range(N)):
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": _square(i), "properties": properties(i)} for i in rows]}


class _World:
    """What the stubbed /api/zones answers from: a vector, a pixel count and an area for each of
    the N polygons. A polygon with no pixels comes back with pixels == 0 and no features, as a
    zone no tile reached does."""

    def __init__(self, gaps=(GAP,)):
        rng = np.random.default_rng(11)
        self.feats = rng.normal(size=(N, DIMS))
        self.pixels = [0 if i in gaps else 300 + 17 * i for i in range(N)]
        self.area = [round(0.250037 + 0.0123 * i, 6) for i in range(N)]

    def label(self, row):
        """A label the vectors determine exactly, so a correct join scores 1.0."""
        return round(float(3.0 + self.feats[row] @ COEF), 6)

    def zones(self, path, body=None, **_kwargs):
        """Keyed by the identifier the request names, or, with none, by the row each polygon
        was SENT in, which is what the service does."""
        assert path == "/api/zones"
        field = body["zone_id_field"]
        rows = []
        for i, feature in enumerate(body["zones_geojson"]["features"]):
            src = _row_of(feature["geometry"])
            row = {"zone_id": str(feature["properties"][field]) if field else str(i),
                   "pixels": self.pixels[src], "area_km2": self.area[src]}
            if self.pixels[src]:
                row.update({f"e{d:03d}": float(self.feats[src][d]) for d in range(DIMS)})
            rows.append(row)
        return {"ok": True, "rows": rows,
                "meta": {"model": "gse", "dims": DIMS, "bands": [], "scale_m": 10.0,
                         "pixel_ground_m": 7.8, "tiles_planned": 1, "tiles_fetched": 1,
                         "tiles_capped": False, "zone_id_field": field,
                         "zones_total": len(rows),
                         "zones_with_pixels": sum(1 for r in rows if r["pixels"]),
                         "tile_errors": [], "pixel_size_warnings": []}}


@pytest.fixture
def world(monkeypatch):
    def make(gaps=(GAP,)):
        w = _World(gaps)
        monkeypatch.setattr(rs_embed_tools, "_svc", w.zones)
        return w
    return make


def _tools():
    return {t.name: t for t in rs_embed_tools.make_rs_embed_zonal_tools()}


def _embed(**kwargs):
    out = json.loads(_tools()["embed_zones"].func(clusters=3, **kwargs))
    assert out.get("ok") is True, out
    return out


def _fit(**kwargs):
    return json.loads(_tools()["fit_zone_model"].func(**kwargs))


def _groups_layer(embedded):
    """The zone-groups layer, reached the way the agent reaches it: by the file id in its URL."""
    from agent_runtime.file_store import resolve_file_id

    layers = embedded.get("map_layers") or [embedded["map_layer"]]
    url = next(layer["url"] for layer in layers if layer["render"] == "categories")
    return json.loads(resolve_file_id(re.search(r"/files/([^/]+)/download", url).group(1))
                      .read_text(encoding="utf-8"))


def _hinted_field(refusal):
    match = re.search(r"pass zone_id_field='([^']+)'", refusal.get("hint", ""))
    assert match, f"the hint must name the column to pass: {refusal}"
    return match.group(1)


def _assert_paired(out, vector_row_of, fitted):
    """Every zone on the prediction map is the polygon its vector was computed for, and the
    label the vectors determine exactly is predicted exactly."""
    from agent_runtime.file_store import resolve_file_id

    assert out.get("ok") is True, out
    assert out["zones_fitted"] == fitted
    layer = json.loads(resolve_file_id(out["predictions_file_id"]).read_text(encoding="utf-8"))
    wrong = [f["properties"]["zone_id"] for f in layer["features"]
             if _row_of(f["geometry"]) != vector_row_of(f["properties"]["zone_id"])]
    assert not wrong, f"zones paired with another polygon's vector: {wrong}"
    assert out["spatial_block_cv"]["r2"] == pytest.approx(1.0, abs=1e-3)
    assert "verdict" not in out


def test_the_groups_layer_after_a_gap_is_refused_and_its_hint_is_the_fix(store, tmp_path,
                                                                         world):
    w = world()
    tracts_id = _store_file(tmp_path, "tracts.geojson", _layer(lambda i: {"NAME": f"T{i}"}))
    embedded = _embed(file_id=tracts_id)
    groups = _groups_layer(embedded)
    zone_ids = [f["properties"]["zone_id"] for f in groups["features"]]
    assert len(zone_ids) == N - 1 and zone_ids[GAP] == str(GAP + 1), \
        "the premise: after the gap, the layer's row numbers and its zones part ways"

    # The agent's step in between: a value per zone, added to the groups layer and saved.
    for f in groups["features"]:
        f["properties"]["canopy"] = w.label(int(f["properties"]["zone_id"]))
    groups_id = _store_file(tmp_path, "groups_with_canopy.geojson", groups)
    vectors_id = embedded["vectors_csv"]["file_id"]

    refused = _fit(vectors_csv_file_id=vectors_id, polygons_file_id=groups_id,
                   label_column="canopy")
    assert refused.get("ok") is False, \
        "paired by row number, 25 of the 28 zones fitted got another zone's vector"
    assert "predictions_file_id" not in refused and "on_map" not in refused
    assert f"1 of the {N - 1} vectors names a zone this {N - 1}-row layer" in refused["error"]
    assert f"'{N - 1}'" in refused["error"], "the one it names is the last zone, now past the end"
    assert "re-embed" in refused["hint"], "the vectors are fine; a new sweep costs tiles"

    out = _fit(vectors_csv_file_id=vectors_id, polygons_file_id=groups_id,
               label_column="canopy", zone_id_field=_hinted_field(refused))
    _assert_paired(out, int, fitted=N - 1)


@pytest.mark.parametrize("own_zone_id", [None, "a shuffle of the row numbers", "text"])
def test_the_layer_the_vectors_were_numbered_by_still_joins_by_row(store, tmp_path, world,
                                                                   own_zone_id):
    """A layer can carry a zone_id of its own that means something else. Embedded and fitted by
    row number, it must join as before. The shuffle is the worst case: those values name every
    vector too, just for other polygons."""
    w = world()
    shuffle = np.random.default_rng(5).permutation(N)
    own = {None: lambda i: {},
           "a shuffle of the row numbers": lambda i: {"zone_id": str(int(shuffle[i]))},
           "text": lambda i: {"zone_id": f"Z{i}"}}[own_zone_id]
    tracts_id = _store_file(tmp_path, "tracts.geojson",
                            _layer(lambda i: {"canopy": w.label(i), **own(i)}))
    embedded = _embed(file_id=tracts_id)

    out = _fit(vectors_csv_file_id=embedded["vectors_csv"]["file_id"],
               polygons_file_id=tracts_id, label_column="canopy")
    _assert_paired(out, int, fitted=N - 1)


@pytest.mark.parametrize("gaps", [(), (N - 1,)], ids=["no gap", "the gap is the last zone"])
def test_a_groups_layer_whose_rows_still_line_up_joins_by_row(store, tmp_path, world, gaps):
    """With no zone missing before the last one, the groups layer's rows are the vectors' keys,
    so there is nothing to refuse."""
    w = world(gaps)
    tracts_id = _store_file(tmp_path, "tracts.geojson", _layer(lambda i: {"NAME": f"T{i}"}))
    embedded = _embed(file_id=tracts_id)
    groups = _groups_layer(embedded)
    for f in groups["features"]:
        f["properties"]["canopy"] = w.label(int(f["properties"]["zone_id"]))
    groups_id = _store_file(tmp_path, "groups_with_canopy.geojson", groups)

    out = _fit(vectors_csv_file_id=embedded["vectors_csv"]["file_id"],
               polygons_file_id=groups_id, label_column="canopy")
    _assert_paired(out, int, fitted=N - len(gaps))


def test_ids_embedded_by_a_column_are_refused_by_row_number(store, tmp_path, world):
    """Districts numbered 1 to 30, embedded by that column and fitted without naming it: every
    row number but 0 is also a district, so 28 zones met a vector and all 28 were the wrong one.
    The column is not called zone_id, so this is caught by the vectors, not by the name."""
    w = world()
    districts_id = _store_file(tmp_path, "districts.geojson", _layer(
        lambda i: {"district": str(i + 1), "canopy": w.label(i)}))
    embedded = _embed(file_id=districts_id, zone_id_field="district")
    vectors_id = embedded["vectors_csv"]["file_id"]

    refused = _fit(vectors_csv_file_id=vectors_id, polygons_file_id=districts_id,
                   label_column="canopy")
    assert refused.get("ok") is False, refused
    assert _hinted_field(refused) == "district"

    out = _fit(vectors_csv_file_id=vectors_id, polygons_file_id=districts_id,
               label_column="canopy", zone_id_field="district")
    _assert_paired(out, lambda zone: int(zone) - 1, fitted=N - 1)


@pytest.mark.parametrize("polygons, column", [("the groups layer", "zone_id"),
                                              ("the tracts", "GEOID")])
def test_vectors_keyed_by_geoid_name_the_column_that_holds_them(store, tmp_path, world,
                                                                 polygons, column):
    """Embedded by GEOID and fitted without zone_id_field, no row number meets a vector. That
    failed loudly before, but with only "only 0 zones have both a vector and a label"; the
    groups layer calls the column zone_id, so the GEOID the embed was given is not there."""
    w = world()
    geoids = [f"5101310{i:04d}" for i in range(N)]
    tracts_id = _store_file(tmp_path, "tracts.geojson", _layer(
        lambda i: {"GEOID": geoids[i], "canopy": w.label(i)}))
    embedded = _embed(file_id=tracts_id, zone_id_field="GEOID")
    if polygons == "the groups layer":
        groups = _groups_layer(embedded)
        for f in groups["features"]:
            f["properties"]["canopy"] = w.label(geoids.index(f["properties"]["zone_id"]))
        polygons_id = _store_file(tmp_path, "groups_with_canopy.geojson", groups)
    else:
        polygons_id = tracts_id
    vectors_id = embedded["vectors_csv"]["file_id"]

    refused = _fit(vectors_csv_file_id=vectors_id, polygons_file_id=polygons_id,
                   label_column="canopy")
    assert refused.get("ok") is False and _hinted_field(refused) == column, refused

    out = _fit(vectors_csv_file_id=vectors_id, polygons_file_id=polygons_id,
               label_column="canopy", zone_id_field=column)
    _assert_paired(out, geoids.index, fitted=N - 1)


def test_a_column_is_named_only_if_it_could_key_every_polygon(store, tmp_path, world):
    """zone_id_field must name a column with no blank and no repeat, so only such a column is
    offered, and when several hold the ids, all of them are, for the caller to choose."""
    w = world()
    geoids = [f"5101310{i:04d}" for i in range(N)]
    repeat = geoids[:GAP] + [geoids[GAP + 1]] + geoids[GAP + 1:]     # every vector's id, twice
    tracts_id = _store_file(tmp_path, "tracts.geojson", _layer(lambda i: {
        "GEOID": geoids[i], "canopy": w.label(i), "GEOID_copy": geoids[i],
        "GEOID_repeat": repeat[i], "GEOID_blank": None if i == GAP else geoids[i]}))
    embedded = _embed(file_id=tracts_id, zone_id_field="GEOID")

    refused = _fit(vectors_csv_file_id=embedded["vectors_csv"]["file_id"],
                   polygons_file_id=tracts_id, label_column="canopy")
    assert refused.get("ok") is False, refused
    assert "'GEOID'" in refused["hint"] and "'GEOID_copy'" in refused["hint"]
    assert "GEOID_repeat" not in refused["hint"] and "GEOID_blank" not in refused["hint"]


def test_a_part_of_the_embedded_layer_is_refused_with_the_general_hint(store, tmp_path, world):
    """No column of these polygons holds the vectors' ids, so the hint says where keys come
    from. This is the one refusal that costs something: the first 20 rows of the embedded layer
    would pair correctly by row number. They line up only by the accident of order, which a
    filtered or re-sorted copy does not keep, so the check does not try to tell them apart."""
    w = world()
    tracts_id = _store_file(tmp_path, "tracts.geojson", _layer(lambda i: {"NAME": f"T{i}"}))
    embedded = _embed(file_id=tracts_id)
    first_20 = _store_file(tmp_path, "first_20.geojson", _layer(
        lambda i: {"NAME": f"T{i}", "canopy": w.label(i)}, rows=range(20)))

    refused = _fit(vectors_csv_file_id=embedded["vectors_csv"]["file_id"],
                   polygons_file_id=first_20, label_column="canopy")
    assert refused.get("ok") is False, refused
    assert "10 of the 29 vectors" in refused["error"] and "20-row layer" in refused["error"]
    assert "zone-groups layer" in refused["hint"] and "zone_id" in refused["hint"]
    assert set(refused["available_columns"]) == {"NAME", "canopy"}
