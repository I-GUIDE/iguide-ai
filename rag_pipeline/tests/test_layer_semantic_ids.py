"""Different contents are different layers; the same content is one layer.

A layer's id is a digest of everything that decides what it SHOWS — region, model, period,
parameters, the inputs it was computed from. Not its label: the model picks a different name
for the same place between turns, so an id that moved with the wording would turn one layer
into two on every re-run, and renaming a layer would make it a different layer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CHAMPAIGN = [-88.246, 40.110, -88.234, 40.121]
URBANA = [-88.212, 40.108, -88.200, 40.119]

BASE = dict(bbox=CHAMPAIGN, model="gse", start="2022-06", end="2022-09")


def _id(**over):
    from agent_runtime.rs_embed_tools import _layer_id
    c = {**BASE, **over}
    return _layer_id("pca", "hint", **c)


# --- the two halves of the rule -----------------------------------------------------------

def test_the_same_content_is_the_same_layer():
    assert _id() == _id()


def test_every_content_input_makes_a_different_layer():
    """Each of these changes what the raster shows, so each must split the layer."""
    baseline = _id()
    for field, other in (("bbox", URBANA), ("model", "clay"),
                         ("start", "2018-06"), ("end", "2018-09")):
        assert _id(**{field: other}) != baseline, f"{field} does not reach the id"


def test_the_period_splits_the_layer():
    """Named explicitly because it was the open question: one region, two years, two layers."""
    a = _id(start="2018-06", end="2018-09")
    b = _id(start="2023-06", end="2023-09")
    assert a != b


# --- what must NOT change identity ---------------------------------------------------------

def test_renaming_a_layer_does_not_make_it_a_different_layer():
    """The model calls the same box 'Downtown Champaign — GSE' one turn and 'Champaign
    downtown 1km box' the next. Same request, same content, one layer."""
    from agent_runtime.rs_embed_tools import _layer_id

    a = _layer_id("pca", "Downtown Champaign - GSE", **BASE)
    b = _layer_id("pca", "Champaign downtown 1km box", **BASE)
    assert a.split("-")[-1] == b.split("-")[-1], "the digest must not move with the hint"


def test_float_noise_in_the_bbox_is_not_a_different_region():
    from agent_runtime.rs_embed_tools import _layer_id, _round_bbox

    a = _layer_id("pca", "h", bbox=_round_bbox([-88.2460000001, 40.11, -88.234, 40.121]))
    b = _layer_id("pca", "h", bbox=_round_bbox([-88.246, 40.11, -88.234, 40.121]))
    assert a == b


# --- the ways the previous scheme lost regions ---------------------------------------------

def test_non_latin_names_do_not_collapse_together():
    from agent_runtime.rs_embed_tools import _layer_id

    ids = {_layer_id("pca", n, bbox=CHAMPAIGN, model="gse", region=n)
           for n in ("北京", "上海", "Москва", "القاهرة")}
    assert len(ids) == 4, ids


def test_mirrored_hemispheres_are_different_regions():
    from agent_runtime.rs_embed_tools import _layer_id, _round_bbox

    west = _layer_id("pca", "h", bbox=_round_bbox([-88.28, 40.09, -88.20, 40.14]))
    east = _layer_id("pca", "h", bbox=_round_bbox([88.20, 40.09, 88.28, 40.14]))
    assert west != east


# --- per-tool content, the parameters each label already advertises -------------------------

def test_cluster_count_splits_the_zone_group_layer():
    from agent_runtime.rs_embed_tools import _layer_id

    zc = dict(file="f1", model="gse", year=2022)
    assert _layer_id("zonegroups", "f1", **zc, clusters=3) != \
           _layer_id("zonegroups", "f1", **zc, clusters=6)


def test_k_splits_the_segmentation_layer():
    from agent_runtime.rs_embed_tools import _layer_id

    assert _layer_id("segments", "h", bbox=CHAMPAIGN, model="gse", k=4) != \
           _layer_id("segments", "h", bbox=CHAMPAIGN, model="gse", k=8)


def test_the_shared_basis_is_part_of_the_shared_pca_layer():
    """The same region re-coloured against a different companion set is different pixels —
    that is the whole point of the tool."""
    from agent_runtime.rs_embed_tools import _layer_id

    a = _layer_id("sharedpca", "h", bbox=CHAMPAIGN, model="gse", basis=["a", "b"])
    b = _layer_id("sharedpca", "h", bbox=CHAMPAIGN, model="gse", basis=["a", "c"])
    assert a != b
    assert a == _layer_id("sharedpca", "h", bbox=CHAMPAIGN, model="gse", basis=["a", "b"])


def test_two_zones_of_one_file_and_one_zone_of_two_files_all_differ():
    from agent_runtime.rs_embed_tools import _layer_id

    ids = {_layer_id("zone", f, file=f, model="gse", zone=z)
           for f in ("fileA", "fileB") for z in ("0", "1")}
    assert len(ids) == 4


def test_the_prediction_layer_carries_its_inputs():
    from agent_runtime.rs_embed_tools import _layer_id

    base = dict(vectors="csvA", polygons="polyA", column="canopy_pct", blocks=5)
    for field, other in (("vectors", "csvB"), ("polygons", "polyB"),
                         ("column", "yield"), ("blocks", 10)):
        assert _layer_id("predicted", "polyA", **{**base, field: other}) != \
               _layer_id("predicted", "polyA", **base), field


# --- end to end through the real tool -------------------------------------------------------

def _embed_region_layer(monkeypatch, **kw):
    import agent_runtime.rs_embed_tools as T

    def _svc(path, body=None, method="POST", **k):
        if path == "/api/models":
            return {"models": [{"id": "gse", "type": "precomputed"}]}
        return {"results": [{"model": "gse", "ok": True, "type": "pooled", "dim": 64,
                             "grid_hw": [101, 101], "norm": 0.8,
                             "image": "data:image/png;base64,AAAA"}]}

    monkeypatch.setattr(T, "_svc", _svc)
    monkeypatch.setattr(T, "_save_png", lambda uri, stem: {
        "file_id": f"id_{stem}", "download_url": f"/files/{stem}.png"})
    tool = {t.name: t for t in T.make_rs_embed_tools()}["embed_region"]
    out = json.loads(tool.func(**kw))
    assert out["ok"] is True, out
    return out["map_layer"]


def test_two_periods_of_one_region_are_two_layers_end_to_end(monkeypatch):
    a = _embed_region_layer(monkeypatch, bbox=CHAMPAIGN, start="2018-06", end="2018-09",
                            name="downtown champaign")
    b = _embed_region_layer(monkeypatch, bbox=CHAMPAIGN, start="2023-06", end="2023-09",
                            name="downtown champaign")
    assert a["id"] != b["id"]


def test_the_same_request_named_differently_is_one_layer_end_to_end(monkeypatch):
    a = _embed_region_layer(monkeypatch, bbox=CHAMPAIGN, name="downtown champaign")
    b = _embed_region_layer(monkeypatch, bbox=CHAMPAIGN, name="Champaign downtown 1 km box")
    assert a["label"] != b["label"], "the fixture must actually change the wording"
    assert a["id"] == b["id"]


# --- what the adversarial review of the content-digest cut found ---------------------------

def test_two_packages_of_one_region_are_two_shared_pca_layers():
    """Regression from this change: bbox, model and basis are identical for every layer in one
    align call, so without the package itself two rasters took one id — and the per-call dedup
    drops the loser server-side, before it can even reach the map."""
    from agent_runtime.rs_embed_tools import _layer_id

    common = dict(bbox=CHAMPAIGN, model="gse", basis=["p1", "p2"])
    assert _layer_id("sharedpca", "h", package="p1", **common) != \
           _layer_id("sharedpca", "h", package="p2", **common)


def test_the_basis_actually_fitted_decides_the_shared_pca_layer():
    """Packages drop out when unreadable, grid-less or bbox-less, and the ones that remain set
    every layer's colours. Two renderings must not share an id because the REQUEST matched."""
    from agent_runtime.rs_embed_tools import _layer_id

    a = _layer_id("sharedpca", "h", package="p1", bbox=CHAMPAIGN, model="gse", basis=["p1", "p2"])
    b = _layer_id("sharedpca", "h", package="p1", bbox=CHAMPAIGN, model="gse", basis=["p1", "p3"])
    assert a != b


def _zone_content(clusters=5, **over):
    from agent_runtime.rs_embed_tools import _CLUSTER_COLORS
    c = dict(file="f1", model="gse", period=("year", 2022), tile_px=200, max_tiles=None,
             zone_id_field="GEOID", siblings=None, zone_ids=None,
             clusters=max(2, min(int(clusters), len(_CLUSTER_COLORS))))
    c.update(over)
    return c


def test_the_requested_cluster_count_decides_the_layer_not_the_realised_one():
    """It keyed on len(present) — the groups that came back. A 6-cluster request that k-means
    resolved to 5 groups took the 5-cluster layer's id, and a swallowed tile error moved the id
    of a request that had not changed."""
    from agent_runtime.rs_embed_tools import _layer_id

    assert _layer_id("zonegroups", "f1", **_zone_content(5)) != \
           _layer_id("zonegroups", "f1", **_zone_content(6))
    assert _layer_id("zonegroups", "f1", **_zone_content(5)) == \
           _layer_id("zonegroups", "f1", **_zone_content(5))


def test_the_effective_period_decides_the_layer_not_the_raw_arguments():
    """The service takes a range only when BOTH ends are given and falls back to `year`
    otherwise, so half a range is the same imagery as none — and `year` is dead once a real
    range is supplied. Digesting the arguments raw split one composite across two layers."""
    from agent_runtime.rs_embed_tools import _layer_id

    year_only = _layer_id("zonepixels", "f1", **_zone_content(period=("year", 2022)))
    assert year_only == _layer_id("zonepixels", "f1", **_zone_content(period=("year", 2022)))
    assert year_only != _layer_id("zonepixels", "f1",
                                  **_zone_content(period=("range", "2025-03", "2025-05")))


def test_sibling_files_reach_the_id():
    """_stage_vector_source reconstructs a shapefile from its siblings, so one file_id can name
    different geometry depending on what arrived with it."""
    from agent_runtime.rs_embed_tools import _layer_id

    assert _layer_id("zonepixels", "f1", **_zone_content()) != \
           _layer_id("zonepixels", "f1", **_zone_content(siblings=["s1"]))


def test_content_may_be_named_kind_or_hint_without_crashing():
    """kind and hint are positional-only: three sites spread a caller-built dict into
    **content, and a later key of either name would have been a TypeError at runtime."""
    from agent_runtime.rs_embed_tools import _layer_id

    assert _layer_id("zonepixels", "f1", kind="x", hint="y")
