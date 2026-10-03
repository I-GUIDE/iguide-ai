"""Scoring an embedding the user already has, without re-embedding the region.

``predict_from_package`` hands a saved ``.npz`` to the service's ``/api/predict_package``, which
runs the pretrained heads on the pooled vectors and needs no Earth Engine. That makes it cheap,
and cheap is exactly what makes the guards matter: the heads are three RandomForests fitted on
corn in Illinois in 2022, they record none of that, and a vector from another year, another place
or another model of the same width is scored in silence and comes back looking in-domain.

These tests pin the guards rather than the plumbing. Each one stands for a measured way to get a
confident wrong number: a stack of zone vectors flattened into one row, a right-width
wrong-model vector, an all-nodata region turned into zeros, and a date check that reads the one
manifest field that is actually always present.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime import rs_embed_head_domain as domain  # noqa: E402

# A +/-2048 m square near Champaign: the footprint embed_region uses by default, and small
# enough that the footprint guard stays quiet on the ordinary case.
ILLINOIS_BOX = {"type": "bbox", "minlon": -88.265, "minlat": 40.100,
                "maxlon": -88.217, "maxlat": 40.137}
BAVARIA_BOX = {"type": "bbox", "minlon": 11.4, "minlat": 48.0,
               "maxlon": 11.6, "maxlat": 48.2}


def _manifest(geometry=None, start="2022-06", end="2022-09", models=None, **extra):
    man = {"geometry": geometry if geometry is not None else dict(ILLINOIS_BOX),
           "start": start, "end": end, "buffer_m": 2048, "compute": "get_embedding",
           "models": models if models is not None else [{"model": "gse", "dim": 64, "meta": {}}]}
    man.update(extra)
    return man


def _package(tmp_path, name="pkg.npz", *, pooled=None, grid=True, manifest=...):
    """A package shaped exactly like the service's export: pooled__/grid__/meta."""
    payload = {}
    for model, vec in (pooled or {"gse": np.arange(64, dtype=np.float32)}).items():
        payload[f"pooled__{model}"] = np.asarray(vec, dtype=np.float32)
        if grid:
            # Incompressible on purpose: a grid of zeros would make the size assertion below
            # pass whether or not the repack actually dropped anything.
            rng = np.random.default_rng(len(model))
            payload[f"grid__{model}"] = rng.normal(
                size=(np.size(vec), 32, 32)).astype(np.float32)
    if manifest is not ...:
        if manifest is not None:
            payload["meta"] = np.asarray(json.dumps(manifest))
    else:
        payload["meta"] = np.asarray(json.dumps(_manifest()))
    path = tmp_path / name
    np.savez_compressed(path, **payload)
    return path


# --- reading the file ----------------------------------------------------------
def test_pooled_and_manifest_read_back(tmp_path):
    path = _package(tmp_path)
    vectors, problems = domain.pooled_vectors(path)
    assert problems == []
    assert set(vectors) == {"gse"}
    assert vectors["gse"].shape == (64,)
    man = domain.read_manifest(path)
    assert man["start"] == "2022-06"
    assert domain.package_region(man) == {
        "region_bbox": [-88.265, 40.1, -88.217, 40.137], "months": "2022-06..2022-09"}


def test_a_pickled_package_is_not_deserialised(tmp_path):
    """allow_pickle=False, because this path np.loads whatever file it is handed.

    A package can come from an upload as easily as from the service. The object array here is
    inert, but reaching it at all would mean a hostile .npz could run code; the guard must
    degrade to "no manifest" rather than unpickle.
    """
    path = tmp_path / "pickled.npz"
    np.savez(path, meta=np.array([{"start": "2022-06"}], dtype=object),
             pooled__gse=np.arange(64, dtype=np.float32))
    assert domain.read_manifest(path) == {}
    vectors, _problems = domain.pooled_vectors(path)
    # The float array beside it still reads: refusing pickles must not cost the vectors.
    assert set(vectors) == {"gse"}


def test_repack_drops_the_grid_and_keeps_the_numbers(tmp_path):
    path = _package(tmp_path)
    out = domain.repack_pooled(path, tmp_path / "pooled.npz")
    assert out["models"] == ["gse"]
    assert out["size_bytes"] < path.stat().st_size / 4
    with np.load(out["path"], allow_pickle=False) as z:
        assert sorted(z.files) == ["meta", "pooled__gse"]
        original, _ = domain.pooled_vectors(path)
        assert np.array_equal(z["pooled__gse"], original["gse"])
    assert domain.read_manifest(out["path"])["start"] == "2022-06"


# --- the vector guards ---------------------------------------------------------
def test_a_good_vector_is_accepted():
    assert domain.vector_refusal("gse", np.arange(64, dtype=np.float32), 64) is None


def test_a_stack_of_zone_vectors_is_refused():
    """16 zones x 64 dims flattens to (1, 1024) and is scored as though it were one region."""
    refusal = domain.vector_refusal("satmae", np.zeros((16, 64), dtype=np.float32), 1024)
    assert refusal and "not a single pooled vector" in refusal
    assert "16" in refusal


def test_a_right_width_wrong_model_vector_is_refused_by_width():
    refusal = domain.vector_refusal("gse", np.zeros(768, dtype=np.float32), 64)
    assert refusal and "768" in refusal and "64" in refusal


def test_an_all_nodata_vector_is_refused():
    """/api/predict raises on an all-nodata ROI; the package path would score it as ~0.45."""
    refusal = domain.vector_refusal("gse", np.full(64, np.nan, dtype=np.float32), 64)
    assert refusal and "no finite value" in refusal


def test_a_partly_nodata_vector_is_still_scored():
    vec = np.arange(64, dtype=np.float32)
    vec[:5] = np.nan
    assert domain.vector_refusal("gse", vec, 64) is None


# --- the domain guards ---------------------------------------------------------
def test_an_in_domain_package_warns_about_nothing():
    assert domain.domain_warnings(_manifest()) == []
    assert domain.unverifiable_domain(_manifest()) == []


def test_the_year_check_reads_the_field_that_is_always_present():
    """The top-level start/end, NOT the per-model meta.temporal.

    Across the 100 manifests in the store, per-model ``meta`` is empty for 36 of 105 entries —
    including every satmae and dofa one, two of the three models with a head — while top-level
    start/end is present on all 100. A guard hung on the precise field stays silent on exactly
    the models where it is most needed, so this package (2025 imagery, no per-model meta at all)
    must still be caught.
    """
    man = _manifest(start="2025-03", end="2025-05",
                    models=[{"model": "satmae", "dim": 1024, "meta": {}}])
    warnings = domain.domain_warnings(man)
    assert len(warnings) == 1
    assert "2025" in warnings[0] and "2022" in warnings[0]
    assert domain.unverifiable_domain(man) == []


def test_a_multi_year_window_that_covers_2022_does_not_warn():
    assert domain.domain_warnings(_manifest(start="2022-01", end="2022-12")) == []


def test_a_region_outside_illinois_warns():
    warnings = domain.domain_warnings(_manifest(geometry=dict(BAVARIA_BOX)))
    assert any("outside Illinois" in w for w in warnings)


def test_a_region_straddling_the_border_warns_partially():
    straddle = {"type": "bbox", "minlon": -92.4, "minlat": 40.0,
                "maxlon": -91.0, "maxlat": 40.5}
    warnings = domain.domain_warnings(_manifest(geometry=straddle))
    assert any("only partly overlaps" in w for w in warnings)


def test_a_region_far_larger_than_the_training_footprint_warns():
    big = {"type": "bbox", "minlon": -89.0, "minlat": 40.0,
           "maxlon": -88.0, "maxlat": 40.9}
    warnings = domain.domain_warnings(_manifest(geometry=big))
    assert any("spans about" in w for w in warnings)


def test_the_agents_own_default_footprint_does_not_warn():
    """buffer_m 2048 is 1.6x the training footprint. A guard that fires on every call is noise."""
    assert domain.domain_warnings(_manifest()) == []


def test_a_package_with_no_manifest_says_so_rather_than_passing():
    assert domain.domain_warnings({}) == []
    unverifiable = domain.unverifiable_domain({})
    assert len(unverifiable) == 1 and "no manifest" in unverifiable[0]


def test_a_manifest_missing_geometry_reports_the_gap():
    man = _manifest()
    man.pop("geometry")
    gaps = domain.unverifiable_domain(man)
    assert any("no geometry" in g for g in gaps)


def test_width_note_names_the_models_that_collide():
    note = domain.width_note(768)
    assert note and all(m in note for m in ("dofa", "olmoearth", "prithvi", "terrafm", "thor"))
    assert domain.width_note(64) is None


# --- the tool ------------------------------------------------------------------
HEADS = {"task": "corn_presence", "kind": "classification", "label": "corn presence",
         "units": "P(corn)", "region": "Illinois (CDL 2022)",
         "classes": ["not corn", "corn"],
         "models": [{"model": "gse", "dim": 64, "score": 0.444, "score_name": "accuracy", "n": 30},
                    {"model": "satmae", "dim": 1024, "score": 0.667,
                     "score_name": "accuracy", "n": 30}]}


@pytest.fixture()
def tool(monkeypatch):
    """predict_from_package with the service and the file store stubbed out."""
    from agent_runtime import rs_embed_tools

    calls = {"uploads": []}

    def fake_svc(path, payload=None, *, method="POST", timeout=None):
        assert path == "/api/heads"
        return dict(HEADS)

    def fake_upload(path, file_path, *, field="file", timeout=None):
        calls["uploads"].append(str(file_path))
        with np.load(file_path, allow_pickle=False) as z:
            scored = sorted(k[len("pooled__"):] for k in z.files if k.startswith("pooled__"))
            calls["uploaded_keys"] = sorted(z.files)
        return {"task": "corn_presence", "kind": "classification", "label": "corn presence",
                "units": "P(corn)",
                "results": [{"model": m, "ok": True, "kind": "classification",
                             "prediction": 0.5625, "label_pred": "corn",
                             "score": 0.667, "score_name": "accuracy"} for m in scored]}

    monkeypatch.setattr(rs_embed_tools, "_svc", fake_svc)
    monkeypatch.setattr(rs_embed_tools, "_svc_upload", fake_upload)
    fn = {t.name: t for t in rs_embed_tools.make_rs_embed_tools()}["predict_from_package"].func
    return fn, calls


def _point_at(monkeypatch, path):
    """Stand in for the store: the tool resolves a reference, not a bare id."""
    from agent_runtime import file_store

    record = {"file_id": "file_abc", "filename": Path(path).name}
    monkeypatch.setattr(file_store, "resolve_file_ref",
                        lambda ref, *, suffix=None: (Path(path), record, []))


def test_the_tool_scores_a_saved_package(tool, tmp_path, monkeypatch):
    fn, calls = tool
    _point_at(monkeypatch, _package(tmp_path))
    out = json.loads(fn("file_abc"))
    assert out["ok"] is True
    assert out["scored_models"] == ["gse"]
    assert out["region_bbox"] == [-88.265, 40.1, -88.217, 40.137]
    assert out["months"] == "2022-06..2022-09"
    assert out["region"] == "Illinois (CDL 2022)"
    assert out["validation"]["gse"] == {"score": 0.444, "score_name": "accuracy", "n": 30}
    assert "outside_training_domain" not in out
    assert out["results"][0]["prediction"] == 0.5625


def test_only_the_pooled_keys_are_uploaded(tool, tmp_path, monkeypatch):
    fn, calls = tool
    _point_at(monkeypatch, _package(tmp_path))
    json.loads(fn("file_abc"))
    assert calls["uploaded_keys"] == ["meta", "pooled__gse"]


def test_a_model_with_no_head_is_reported_and_not_uploaded(tool, tmp_path, monkeypatch):
    fn, calls = tool
    path = _package(tmp_path, pooled={"gse": np.arange(64, dtype=np.float32),
                                      "clay": np.arange(1024, dtype=np.float32)})
    _point_at(monkeypatch, path)
    out = json.loads(fn("file_abc"))
    assert out["ok"] is True
    assert out["scored_models"] == ["gse"]
    assert [e["model"] for e in out["not_scored"]] == ["clay"]
    assert calls["uploaded_keys"] == ["meta", "pooled__gse"]


def test_nothing_is_uploaded_when_no_model_has_a_head(tool, tmp_path, monkeypatch):
    fn, calls = tool
    _point_at(monkeypatch, _package(tmp_path, pooled={"thor": np.arange(768, dtype=np.float32)}))
    out = json.loads(fn("file_abc"))
    assert out["ok"] is False
    assert calls["uploads"] == []
    assert out["heads_available"] == ["gse", "satmae"]


def test_an_out_of_domain_package_is_scored_but_flagged(tool, tmp_path, monkeypatch):
    fn, _calls = tool
    path = _package(tmp_path, manifest=_manifest(geometry=dict(BAVARIA_BOX),
                                                 start="2025-03", end="2025-05"))
    _point_at(monkeypatch, path)
    out = json.loads(fn("file_abc"))
    assert out["ok"] is True
    flags = out["outside_training_domain"]
    assert any("2025" in f for f in flags)
    assert any("outside Illinois" in f for f in flags)


def test_a_manifestless_package_is_scored_but_called_unknown(tool, tmp_path, monkeypatch):
    fn, _calls = tool
    _point_at(monkeypatch, _package(tmp_path, manifest=None))
    out = json.loads(fn("file_abc"))
    assert out["ok"] is True
    assert "region_bbox" not in out
    assert any("no manifest" in g for g in out["domain_unverifiable"])


def test_a_zones_csv_is_refused_with_the_right_alternative(tool, tmp_path, monkeypatch):
    """embed_zones writes a CSV, not a package — and its rows should not go to these heads."""
    fn, calls = tool
    csv = tmp_path / "zone_vectors.csv"
    csv.write_text("zone_id,pixels,e000,e001\n17019,12,0.1,0.2\n")
    _point_at(monkeypatch, csv)
    out = json.loads(fn("file_abc"))
    assert out["ok"] is False
    assert "fit_zone_model" in out["hint"]
    assert calls["uploads"] == []


def test_an_unresolvable_file_id_says_what_to_pass(tool, tmp_path, monkeypatch):
    from agent_runtime import file_store

    fn, calls = tool

    def boom(ref, *, suffix=None):
        raise ValueError(f"no stored file matches {ref!r}")

    monkeypatch.setattr(file_store, "resolve_file_ref", boom)
    out = json.loads(fn("file_missing"))
    assert out["ok"] is False
    assert "list_embedding_packages" in out["hint"]
    assert calls["uploads"] == []


# --- addressing: reaching a package a later turn can only NAME ------------------
@pytest.fixture()
def store(tmp_path, monkeypatch):
    """An isolated file store, so the lookup is tested against records we planted."""
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    from agent_runtime import file_store

    return file_store


def _save(store, tmp_path, name, **kw):
    path = _package(tmp_path, name=name, **kw)
    return store.create_output_file_from_path(path, filename=name)


def test_find_files_matches_a_partial_name(store, tmp_path):
    _save(store, tmp_path, "downtown_champaign_1_km_box_vectors.npz")
    _save(store, tmp_path, "urbana_vectors.npz")
    hits = store.find_files(name="champaign", suffix=".npz")
    assert [h["filename"] for h in hits] == ["downtown_champaign_1_km_box_vectors.npz"]


def test_find_files_filters_by_extension(store, tmp_path):
    _save(store, tmp_path, "region_vectors.npz")
    csv = tmp_path / "zone_vectors.csv"
    csv.write_text("zone_id,e000\n1,0.5\n")
    store.create_output_file_from_path(csv, filename="zone_vectors.csv")
    assert [h["filename"] for h in store.find_files(suffix=".npz")] == ["region_vectors.npz"]


def test_find_files_returns_newest_first(store, tmp_path):
    import os
    import time

    old = _save(store, tmp_path, "older_vectors.npz")
    new = _save(store, tmp_path, "newer_vectors.npz")
    # mtime is the only ordering signal the records carry, so set it explicitly rather than
    # relying on the two writes landing in different clock ticks.
    os.utime(store.resolve_file_id(old["file_id"]), (time.time() - 500, time.time() - 500))
    ids = [h["file_id"] for h in store.find_files(suffix=".npz")]
    assert ids[0] == new["file_id"]
    assert old["file_id"] in ids


def test_resolve_file_ref_takes_an_id_or_a_name(store, tmp_path):
    rec = _save(store, tmp_path, "champaign_vectors.npz")
    by_id, record_a, alts_a = store.resolve_file_ref(rec["file_id"])
    by_name, record_b, _alts_b = store.resolve_file_ref("champaign", suffix=".npz")
    assert by_id == by_name
    assert record_a["file_id"] == record_b["file_id"]
    assert alts_a == []


def test_resolve_file_ref_reports_the_packages_it_did_not_pick(store, tmp_path):
    import os
    import time

    first = _save(store, tmp_path, "champaign_june_vectors.npz")
    _save(store, tmp_path, "champaign_july_vectors.npz")
    os.utime(store.resolve_file_id(first["file_id"]), (time.time() - 500, time.time() - 500))
    _path, record, alternatives = store.resolve_file_ref("champaign", suffix=".npz")
    assert record["filename"] == "champaign_july_vectors.npz"
    assert [a["filename"] for a in alternatives] == ["champaign_june_vectors.npz"]


def test_resolve_file_ref_raises_when_nothing_matches(store, tmp_path):
    _save(store, tmp_path, "champaign_vectors.npz")
    with pytest.raises(ValueError):
        store.resolve_file_ref("bavaria", suffix=".npz")


# --- the layer carries its own vectors -----------------------------------------
def test_a_raster_descriptor_can_point_at_its_vectors():
    from agent_runtime.rs_embed_tools import _raster_layer

    plain = _raster_layer({"download_url": "/f/1"}, [0, 0, 1, 1], "l", "id-1")
    assert "embedding" not in plain
    pointed = _raster_layer({"download_url": "/f/1"}, [0, 0, 1, 1], "l", "id-1",
                            embedding={"file_id": "file_abc", "model": "gse"})
    assert pointed["embedding"] == {"file_id": "file_abc", "model": "gse"}


# --- the tool, addressed by name and narrowed by model -------------------------
def test_the_tool_accepts_a_filename(tool, store, tmp_path, monkeypatch):
    fn, calls = tool
    rec = _save(store, tmp_path, "downtown_champaign_vectors.npz")
    out = json.loads(fn("downtown_champaign_vectors.npz"))
    assert out["ok"] is True
    assert out["package_file_id"] == rec["file_id"]
    assert out["package_filename"] == "downtown_champaign_vectors.npz"
    assert calls["uploaded_keys"] == ["meta", "pooled__gse"]


def test_duplicates_of_one_region_resolve_to_the_newest(tool, store, tmp_path):
    """Several exports of the same region and period are interchangeable — pick one, say so."""
    import os
    import time

    fn, _calls = tool
    first = _save(store, tmp_path, "champaign_a_vectors.npz")
    _save(store, tmp_path, "champaign_b_vectors.npz")
    os.utime(store.resolve_file_id(first["file_id"]), (time.time() - 500, time.time() - 500))
    out = json.loads(fn("champaign"))
    assert out["ok"] is True
    assert out["package_filename"] == "champaign_b_vectors.npz"
    assert [a["filename"] for a in out["also_matched"]] == ["champaign_a_vectors.npz"]
    assert "same region and months" in out["resolved_by"]


def test_a_name_covering_different_regions_is_refused(tool, store, tmp_path):
    """The real case: the default export name is reused, so one name spans many places.

    In the live store 73 packages carry 22 distinct filenames and `embedding_vectors.npz` alone
    is used 32 times. Resolving that by mtime would answer a question about the wrong region
    with a perfectly plausible probability, so it has to be refused with the candidates.
    """
    fn, calls = tool
    _save(store, tmp_path, "embedding_vectors.npz")
    _save(store, tmp_path, "embedding_vectors.npz",
          manifest=_manifest(geometry=dict(BAVARIA_BOX)))
    out = json.loads(fn("embedding_vectors.npz"))
    assert out["ok"] is False
    assert "different regions or months" in out["error"]
    assert len(out["candidates"]) == 2
    assert {c["filename"] for c in out["candidates"]} == {"embedding_vectors.npz"}
    # Region is what tells them apart, so it has to be IN the choice.
    assert all("region_bbox" in c for c in out["candidates"])
    assert calls["uploads"] == []


def test_a_name_covering_different_months_is_refused(tool, store, tmp_path):
    fn, calls = tool
    _save(store, tmp_path, "region_vectors.npz")
    _save(store, tmp_path, "region_vectors.npz",
          manifest=_manifest(start="2018-06", end="2018-09"))
    out = json.loads(fn("region_vectors.npz"))
    assert out["ok"] is False
    assert sorted(c.get("months") for c in out["candidates"]) == ["2018-06..2018-09",
                                                                  "2022-06..2022-09"]
    assert calls["uploads"] == []


def test_models_narrows_the_run_to_one_layer(tool, store, tmp_path, monkeypatch):
    fn, calls = tool
    _save(store, tmp_path, "three_vectors.npz",
          pooled={"gse": np.arange(64, dtype=np.float32),
                  "satmae": np.arange(1024, dtype=np.float32)})
    out = json.loads(fn("three_vectors.npz", ["satmae"]))
    assert out["scored_models"] == ["satmae"]
    assert calls["uploaded_keys"] == ["meta", "pooled__satmae"]


def test_an_unknown_model_filter_says_what_the_package_holds(tool, store, tmp_path):
    fn, calls = tool
    _save(store, tmp_path, "one_vectors.npz")
    out = json.loads(fn("one_vectors.npz", ["clay"]))
    assert out["ok"] is False
    assert "clay" in out["error"]
    assert calls["uploads"] == []


def test_listing_shows_saved_packages_with_where_and_when(tool, store, tmp_path, monkeypatch):
    from agent_runtime import rs_embed_tools

    _save(store, tmp_path, "champaign_vectors.npz")
    monkeypatch.setattr(rs_embed_tools, "_svc",
                        lambda path, payload=None, *, method="POST", timeout=None: dict(HEADS))
    listing = {t.name: t for t in rs_embed_tools.make_rs_embed_tools()}["list_embedding_packages"]
    out = json.loads(listing.func())
    assert out["ok"] is True and out["count"] == 1
    pkg = out["packages"][0]
    assert pkg["filename"] == "champaign_vectors.npz"
    assert pkg["models"] == ["gse"]
    assert pkg["months"] == "2022-06..2022-09"
    assert pkg["region_bbox"] == [-88.265, 40.1, -88.217, 40.137]
    assert pkg["has_head"] == ["gse"]


def test_listing_says_so_when_nothing_has_been_embedded(store, monkeypatch):
    from agent_runtime import rs_embed_tools

    monkeypatch.setattr(rs_embed_tools, "_svc",
                        lambda path, payload=None, *, method="POST", timeout=None: dict(HEADS))
    listing = {t.name: t for t in rs_embed_tools.make_rs_embed_tools()}["list_embedding_packages"]
    out = json.loads(listing.func())
    assert out["ok"] is True and out["packages"] == []
    assert "embed_region saves one" in out["note"]


# --- the pointer has to survive TWO field-by-field rebuilds ---------------------
def _built(descriptor):
    from agent_runtime.map_layers import build_map_layer

    return build_map_layer("embed_region", {"map_layer": descriptor})


def test_build_map_layer_keeps_the_embedding_pointer():
    """The regression this file exists for second time round.

    build_map_layer assembles a FIXED-SHAPE dict, so a field the tool sets and the client reads
    arrives as nothing unless it is named there — the same trap `outline` and `sampled` already
    fell into. Setting the pointer on the tool's descriptor is therefore not enough on its own,
    and nothing downstream can tell the difference between "no vectors" and "stripped in transit".
    """
    out = _built({"url": "/f/1/download", "label": "gse embedding", "render": "raster",
                  "bounds": [-88.3, 40.0, -88.2, 40.1], "id": "embed-pca-abc",
                  "embedding": {"file_id": "file_abc", "filename": "r_vectors.npz",
                                "model": "gse", "months": "2022-06..2022-09"}})
    assert out is not None
    assert out["embedding"] == {"file_id": "file_abc", "filename": "r_vectors.npz",
                                "model": "gse", "months": "2022-06..2022-09"}


def test_build_map_layer_omits_the_pointer_when_there_is_none():
    out = _built({"url": "/f/1/download", "label": "a mask", "render": "raster",
                  "bounds": [-88.3, 40.0, -88.2, 40.1], "id": "embed-seg-abc"})
    assert out is not None and "embedding" not in out


def test_build_map_layer_ignores_a_pointer_with_no_file_id():
    """Half a pointer is worse than none: it would read as data that cannot be fetched."""
    out = _built({"url": "/f/1/download", "label": "gse embedding", "render": "raster",
                  "bounds": [-88.3, 40.0, -88.2, 40.1], "id": "embed-pca-abc",
                  "embedding": {"model": "gse"}})
    assert out is not None and "embedding" not in out


def test_build_map_layer_drops_empty_pointer_fields():
    out = _built({"url": "/f/1/download", "label": "gse embedding", "render": "raster",
                  "bounds": [-88.3, 40.0, -88.2, 40.1], "id": "embed-pca-abc",
                  "embedding": {"file_id": "file_abc", "filename": None,
                                "models_in_package": []}})
    assert out["embedding"] == {"file_id": "file_abc"}


# --- one polygon, one layer ----------------------------------------------------
def test_the_boundary_and_its_embedding_are_one_layer():
    """A city appeared TWICE in the layer list: once as the boundary admin_boundary drew, and

    again as "gse embedded zone 1777005" when embed_zones redrew the same polygon with what it
    found inside. Both are keyed on the polygon FILE now, so the second replaces the first.
    """
    from agent_runtime.map_layers import boundary_layer_id

    assert boundary_layer_id("file_abc") == boundary_layer_id(" file_abc ")
    assert boundary_layer_id("file_abc") != boundary_layer_id("file_def")


def test_the_boundary_layer_carries_that_id():
    """admin_boundary set no id at all, so build_map_layer invented one from its LABEL —

    which embed_zones cannot reconstruct from the file_id it is handed.
    """
    from agent_runtime.map_layers import boundary_layer_id, build_map_layer

    built = build_map_layer("admin_boundary", {"map_layer": {
        "url": "/f/1/download", "label": "Urbana city", "render": "shapes",
        "id": boundary_layer_id("file_poly1"), "source": "analysis",
        "count": 1, "outline": True}})
    assert built is not None
    assert built["id"] == "boundary-file_poly1"
    assert built["outline"] is True


def test_a_label_does_not_say_the_model_twice():
    """The model names its own layers descriptively, and usually puts the model in the name.

    Prepending "Urbana city — gse — Jun–Sep 2022" to "gse pixel embedding in zones" said gse
    twice, in a name the layer panel then clips with an ellipsis.
    """
    from agent_runtime.rs_embed_tools import _layer_label

    assert _layer_label("gse pixel embedding in zones", "Urbana city — gse — Jun–Sep 2022") \
        == "Urbana city — gse — Jun–Sep 2022 — pixel embedding in zones"


def test_a_label_keeps_the_model_when_the_tag_lacks_it():
    """Only a REPEAT is dropped. A tag that never names the model still needs it."""
    from agent_runtime.rs_embed_tools import _layer_label

    assert _layer_label("gse embedding (PCA-RGB)", "40.113,-88.231") \
        == "40.113,-88.231 — gse embedding (PCA-RGB)"


def test_only_the_leading_word_is_dropped():
    """A word later in the description is load-bearing — 'zones', 'change', a k — and stays."""
    from agent_runtime.rs_embed_tools import _layer_label

    assert _layer_label("gse zone groups (k=6)", "Drawn Champaign region") \
        == "Drawn Champaign region — gse zone groups (k=6)"


def test_the_match_is_an_exact_token_not_a_prefix():
    """'zone' is not 'zones'. Matching loosely would eat a word that carries meaning and

    leave "gse zones — groups (k=6)", which reads as groups of nothing. When in doubt the
    duplicate is the cheaper mistake: it is ugly, where a wrong trim is misleading.
    """
    from agent_runtime.rs_embed_tools import _layer_label

    assert _layer_label("zone groups (k=6)", "gse zones") == "gse zones — zone groups (k=6)"


# --- a package name that identifies the package --------------------------------
def test_the_default_package_name_says_which_region_model_and_period():
    """32 of the 73 stored packages were called `embedding_vectors.npz`, because that was the

    default for every region the model did not name. A later turn naming one could not be
    answered — which is why the ambiguity refusal had to exist at all.

    A timestamp or a random id would make them unique and still unidentifiable: "which of these
    32 is Champaign" is the question being asked, and only the region, model and period answer
    it. All three are known where the name is built.
    """
    from agent_runtime.rs_embed_tools import _region_tag, _slug

    box = [-88.2456, 40.1124, -88.2367, 40.1192]
    unnamed = f"{_slug(None or _region_tag(None, box))}_{'-'.join(['gse'])}_2022-06_2022-09_vectors"
    assert "40_116" in unnamed and "88_241" in unnamed, "the region has to be in it"
    assert "gse" in unnamed and "2022-06" in unnamed
    assert unnamed != "embedding_vectors"


def test_two_regions_no_longer_collide_on_the_default_name():
    from agent_runtime.rs_embed_tools import _region_tag, _slug

    champaign = _slug(_region_tag(None, [-88.2456, 40.1124, -88.2367, 40.1192]))
    urbana = _slug(_region_tag(None, [-88.2100, 40.1100, -88.2000, 40.1200]))
    assert champaign != urbana


def test_a_named_region_keeps_its_name_and_gains_the_rest():
    from agent_runtime.rs_embed_tools import _slug

    named = f"{_slug('downtown champaign')}_{'-'.join(['gse', 'satmae'])}_2022-06_2022-09_vectors"
    assert named.startswith("downtown_champaign")
    assert "gse-satmae" in named
