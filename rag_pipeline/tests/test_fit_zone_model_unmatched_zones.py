"""fit_zone_model's join report shows the unmatched ids so the cause can be read off them.

S6.10's `join` report counts who found no partner and gives example ids from each side, and its
`warning` puts one beside the other. The examples were the first few in file order, so when the
two layers listed their zones in different orders, the vectors' '06037100000' sat beside some
other tract's '6037102800', and nothing lined up. An id pandas read as missing (a zone keyed
'NA') showed as the text 'nan', which looks like a zone called nan. A column of long text ids put
whole descriptions into the reply. And a fit that met nothing because the two sides held the key
differently was told to embed more zones, buying tiles for vectors it already had.

Now the examples on both sides are ordered by value where an id reads as a number, an id read as
missing is null, each id is at most 80 characters, and the fewer-than-12 failure points at the
key when leftovers on both sides could have made up the shortfall. Everything here is offline:
the vectors are synthetic, and the one rs-embed service call is stubbed, as in
test_fit_zone_model_leading_zero_ids.py.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from agent_runtime import rs_embed_tools

N = 30
DIMS = 4

CALIFORNIA = [f"0603710{i:04d}" for i in range(N)]      # Los Angeles County, FIPS 06037
ILLINOIS = [f"1703110{i:04d}" for i in range(N)]        # Cook County, FIPS 17031
CA_AND_IL = CALIFORNIA[:15] + ILLINOIS[:15]
# TRACTCE: six digits, zero-padded. Sorted as text, '000100'.. would face '100', '1000', '1100'..
TRACTCE = [f"{100 * (i + 1):06d}" for i in range(15)] + [f"{100100 + 100 * i}" for i in range(15)]


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


def _tracts(tmp_path, ids, truth, name, as_numbers=False, rename=None, reverse=False):
    """A tract layer keyed by GEOID and labelled `truth`. `as_numbers` stores GEOID as a number,
    as a layer that went through a spreadsheet does; `rename` rewrites each id; `reverse` lists
    the zones in the opposite order, as a layer sorted another way does."""
    order = list(enumerate(ids))[::-1] if reverse else list(enumerate(ids))
    layer = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": _square(i),
         "properties": {"GEOID": int(z) if as_numbers else (rename(z) if rename else z),
                        "truth": truth[z]}}
        for i, z in order]}
    return _store_file(tmp_path, name, json.dumps(layer))


def _embedded(tmp_path, monkeypatch, ids, zone_ids=None):
    """embed_zones over a tract layer keyed by text GEOIDs. Returns the vectors CSV's file id, the
    tracts' file id and the label per id. `zone_ids` embeds only those zones."""
    rng = np.random.default_rng(5)
    feats = {z: rng.normal(size=DIMS) for z in ids}
    pixels = {z: 300 + 17 * i for i, z in enumerate(ids)}
    area = {z: round(0.25 + 0.0123 * i, 6) for i, z in enumerate(ids)}
    truth = {z: round(3.0 + feats[z][0] - 0.5 * feats[z][1] + rng.normal(scale=0.3), 6)
             for z in ids}
    tracts_id = _tracts(tmp_path, ids, truth, "tracts.geojson")

    monkeypatch.setattr(rs_embed_tools, "_svc", _zones_service(pixels, area, feats))
    embed = {t.name: t for t in rs_embed_tools.make_rs_embed_zonal_tools()}["embed_zones"]
    out = json.loads(embed.func(file_id=tracts_id, zone_id_field="GEOID", clusters=3,
                                zone_ids=zone_ids))
    assert out.get("ok") is True, out
    return out["vectors_csv"]["file_id"], tracts_id, truth


def _strict(text):
    """The tool's reply parsed as strict JSON: NaN and Infinity are not JSON."""
    def refuse(constant):
        raise ValueError(f"the reply carries {constant}, which is not JSON")
    return json.loads(text, parse_constant=refuse)


def _fit(vectors_id, polygons_id):
    fit = {t.name: t for t in rs_embed_tools.make_rs_embed_zonal_tools()}["fit_zone_model"]
    return _strict(fit.func(vectors_csv_file_id=vectors_id, polygons_file_id=polygons_id,
                            label_column="truth", zone_id_field="GEOID"))


@pytest.mark.parametrize("ids, unmatched", [(CA_AND_IL, CALIFORNIA[:15]),
                                            (TRACTCE, TRACTCE[:15])],
                         ids=["GEOID", "TRACTCE"])
def test_the_examples_line_up_when_the_layers_are_ordered_differently(store, tmp_path,
                                                                      monkeypatch, ids,
                                                                      unmatched):
    """Text ids embedded, the same ids stored as numbers in a layer listed in reverse. In file
    order the polygons' examples were the LAST zones that missed, beside the vectors' first."""
    vectors_id, _, truth = _embedded(tmp_path, monkeypatch, ids)
    numeric_id = _tracts(tmp_path, ids, truth, "numeric.geojson", as_numbers=True, reverse=True)

    out = _fit(vectors_id, numeric_id)
    assert out.get("ok") is True, out
    assert out["zones_fitted"] == 15
    join = out["join"]
    assert join["example_unmatched_vector_ids"] == unmatched[:3]
    assert join["example_polygon_ids"] == [str(int(z)) for z in unmatched[:3]], \
        "the same three zones, written as the polygons hold them"
    assert f"for example {unmatched[0]!r}, where the polygons have ids like " \
           f"{str(int(unmatched[0]))!r}" in out["warning"]


def test_an_id_read_as_missing_is_null(store, tmp_path, monkeypatch):
    """A zone keyed 'NA' is embedded and comes back from the CSV as missing. The vector side shows
    it as null, not as the text 'nan', and the reply stays strict JSON."""
    codes = ["NA"] + [f"C{i:02d}" for i in range(1, N)]
    vectors_id, tracts_id, _ = _embedded(tmp_path, monkeypatch, codes)

    out = _fit(vectors_id, tracts_id)
    assert out.get("ok") is True, out
    join = out["join"]
    assert (join["matched"], join["vectors_without_a_polygon"],
            join["polygons_without_a_vector"]) == (29, 1, 1)
    assert join["example_unmatched_vector_ids"] == [None]
    assert join["example_polygon_ids"] == ["NA"]
    assert "for example an id read as missing, where the polygons have ids like 'NA'" \
        in out["warning"]


def test_long_ids_are_cut_to_80_characters(store, tmp_path, monkeypatch):
    """Ids of long text, embedded from one layer and fitted against a copy that wrote them in
    capitals: no example, and nothing quoted in the warning, is longer than 80 characters. The
    ids hold no comma, which S6.10 found splits the CSV row."""
    long_ids = [f"Census Tract {i:04d} of the Los Angeles County planning area described "
                f"at length in the source layer's name field" for i in range(N)]
    assert min(len(z) for z in long_ids) > 100
    vectors_id, _, truth = _embedded(tmp_path, monkeypatch, long_ids)
    shouted_id = _tracts(tmp_path, long_ids, truth, "shouted.geojson", rename=str.upper)

    out = _fit(vectors_id, shouted_id)
    assert out["ok"] is False and out["join"]["matched"] == 0
    shown = out["join"]["example_unmatched_vector_ids"] + out["join"]["example_polygon_ids"]
    assert len(shown) == 6 and all(len(z) == 80 and z.endswith("…") for z in shown), shown
    assert shown[0] == long_ids[0][:79] + "…" and shown[3] == long_ids[0].upper()[:79] + "…"
    assert not any(z in out["warning"] for z in long_ids + [z.upper() for z in long_ids])


def test_a_join_that_meets_nothing_points_at_the_key(store, tmp_path, monkeypatch):
    """California alone, fitted against numeric GEOIDs, meets nothing. The zones exist, so
    "embed more zones" was the wrong advice: it sends the model to buy tiles for vectors it
    already has."""
    vectors_id, _, truth = _embedded(tmp_path, monkeypatch, CALIFORNIA)
    numeric_id = _tracts(tmp_path, CALIFORNIA, truth, "numeric.geojson", as_numbers=True)

    out = _fit(vectors_id, numeric_id)
    assert out["ok"] is False
    assert out["error"] == "only 0 zones have both a vector and a label"
    assert "Embed more zones" not in out["hint"]
    assert out["hint"].startswith("Only 0 zones met: 30 of the 30 vectors and 30 of the 30 "
                                  "polygons found no partner with the same zone id. Check "
                                  "zone_id_field")


# Guards: the two cases where "embed more zones" is still the right advice. Both pass without
# the change too; they pin the condition the new hint fires on.

def test_too_few_zones_embedded_keeps_asking_for_more(store, tmp_path, monkeypatch):
    """Ten of 30 tracts embedded: every vector met its polygon, there are just too few."""
    vectors_id, tracts_id, _ = _embedded(tmp_path, monkeypatch, ILLINOIS,
                                         zone_ids=ILLINOIS[:10])

    out = _fit(vectors_id, tracts_id)
    assert out["ok"] is False
    assert out["error"] == "only 10 zones have both a vector and a label"
    assert out["hint"].startswith("Embed more zones before fitting")
    assert out["join"]["polygons_without_a_vector"] == 20


def test_leftovers_too_few_to_make_up_the_shortfall_keep_asking_for_more(store, tmp_path,
                                                                          monkeypatch):
    """Ten zones meet and one is keyed two ways: fixing the key would give 11, still short of
    12, so the zone count is the cause, and the warning already names the key."""
    ids = ILLINOIS[:10] + CALIFORNIA[:1]
    vectors_id, _, truth = _embedded(tmp_path, monkeypatch, ids)
    numeric_id = _tracts(tmp_path, ids, truth, "numeric.geojson", as_numbers=True)

    out = _fit(vectors_id, numeric_id)
    assert out["ok"] is False
    assert out["error"] == "only 10 zones have both a vector and a label"
    assert out["hint"].startswith("Embed more zones before fitting")
    assert out["warning"].startswith("1 of 11 vectors found no polygon")
