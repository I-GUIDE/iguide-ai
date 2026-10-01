"""A shared PCA basis makes two embedding rasters comparable by eye.

embed_region fits a PCA and a contrast stretch on EACH region's own pixels, so identical ground
in two regions can come out different colours and different ground can come out the same. The
maps are individually meaningful and mutually incomparable — which is exactly the comparison a
reader makes as soon as both layers are on the map together.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODEL = "gse"
DIMS = 6
SHARED_VECTOR = np.arange(DIMS, dtype=np.float32) * 0.37 - 1.1

BBOX_A = [-88.246, 40.110, -88.234, 40.121]
BBOX_B = [-88.212, 40.108, -88.200, 40.119]


def _grid(seed: int, scale: float, h: int = 8, w: int = 8) -> np.ndarray:
    """A region's (d, h, w) grid, with the SHARED vector planted at pixel (0, 0)."""
    rng = np.random.default_rng(seed)
    g = (rng.normal(size=(DIMS, h, w)) * scale).astype(np.float32)
    g[:, 0, 0] = SHARED_VECTOR
    return g


def _package(tmp_path: Path, stem: str, grid: np.ndarray, bbox) -> Path:
    """Stand-in for the .npz embed_region saves."""
    p = tmp_path / f"{stem}.npz"
    meta = json.dumps({
        "geometry": {"type": "bbox", "minlon": bbox[0], "minlat": bbox[1],
                     "maxlon": bbox[2], "maxlat": bbox[3]},
        "models": [{"model": MODEL, "dim": DIMS, "grid_saved": True}],
    })
    np.savez(p, **{f"grid__{MODEL}": grid.astype(np.float32),
                   f"pooled__{MODEL}": grid.reshape(DIMS, -1).mean(1).astype(np.float32),
                   "meta": np.array(meta)})
    return p


def _run(monkeypatch, tmp_path, packages: dict, **kw):
    """Drive the tool with the file store stubbed to the packages on disk."""
    import agent_runtime.file_store as FS
    import agent_runtime.rs_embed_tools as T

    monkeypatch.setattr(T, "resolve_file_id", lambda fid: packages[fid], raising=False)
    monkeypatch.setattr(FS, "resolve_file_id", lambda fid: packages[fid])
    saved = {}

    def _save(path, filename=None):
        saved[filename] = Path(path).read_bytes()
        return {"file_id": f"id_{filename}", "filename": filename,
                "download_url": f"/files/{filename}", "size_bytes": len(saved[filename])}

    monkeypatch.setattr(FS, "create_output_file_from_path", _save)

    tool = {t.name: t for t in T.make_rs_embed_tools()}["align_embedding_colors"]
    return json.loads(tool.func(file_ids=list(packages), **kw)), saved


def _rgb_at(png_bytes: bytes, x: int, y: int):
    from io import BytesIO

    from PIL import Image

    return Image.open(BytesIO(png_bytes)).convert("RGB").getpixel((x, y))


def _two_regions(tmp_path):
    # Deliberately different spreads: independent stretches diverge hardest here.
    return {
        "file_a": _package(tmp_path, "region_a", _grid(0, 1.0), BBOX_A),
        "file_b": _package(tmp_path, "region_b", _grid(1, 4.0), BBOX_B),
    }


def test_the_same_vector_gets_the_same_colour_in_both_regions(monkeypatch, tmp_path):
    """The whole point. Pixel (0,0) holds an identical embedding vector in both regions."""
    out, saved = _run(monkeypatch, tmp_path, _two_regions(tmp_path))
    assert out["ok"] is True, out
    assert len(saved) == 2, saved.keys()

    a, b = (_rgb_at(v, 0, 0) for v in saved.values())
    assert a == b, f"identical ground rendered as {a} and {b}"


def test_per_region_rendering_would_not_have_matched(monkeypatch, tmp_path):
    """Guards the premise: with each region normalised on its own pixels — what embed_region
    does today — that same vector lands on two different colours, so the test above is
    measuring the fix and not a coincidence."""
    def solo(grid):
        feats = grid.reshape(DIMS, -1).T.astype(np.float64)
        mu = feats.mean(0)
        _u, _s, vt = np.linalg.svd(feats - mu, full_matrices=False)
        p = (feats - mu) @ vt[:3].T
        p *= np.where(p.sum(0) < 0, -1.0, 1.0)
        lo, hi = np.percentile(p, 2.0, axis=0), np.percentile(p, 98.0, axis=0)
        return tuple((np.clip((p[0] - lo) / (hi - lo + 1e-8), 0, 1) * 255).astype(np.uint8))

    assert solo(_grid(0, 1.0)) != solo(_grid(1, 4.0))


def test_both_regions_reach_the_map_with_their_own_bounds(monkeypatch, tmp_path):
    out, _ = _run(monkeypatch, tmp_path, _two_regions(tmp_path))
    layers = out.get("map_layers") or [out["map_layer"]]
    assert len(layers) == 2
    assert [l["bounds"] for l in layers] == [BBOX_A, BBOX_B]
    assert len({l["label"] for l in layers}) == 2, "the layers must not collide"
    assert all("shared PCA" in l["label"] for l in layers)


def test_names_label_the_layers_in_order(monkeypatch, tmp_path):
    out, _ = _run(monkeypatch, tmp_path, _two_regions(tmp_path),
                  names=["downtown champaign", "downtown urbana"])
    layers = out.get("map_layers") or [out["map_layer"]]
    assert layers[0]["label"].startswith("downtown champaign")
    assert layers[1]["label"].startswith("downtown urbana")


def test_variance_explained_is_reported(monkeypatch, tmp_path):
    """Three colours cannot carry 64 dimensions; say how much they do carry."""
    out, _ = _run(monkeypatch, tmp_path, _two_regions(tmp_path))
    ve = out["variance_explained"]
    assert len(ve) == 3
    assert all(0.0 <= v <= 1.0 for v in ve)
    assert sum(ve) <= 1.0 + 1e-6


def test_one_region_is_refused_because_a_basis_is_only_shared(monkeypatch, tmp_path):
    pkgs = {"file_a": _package(tmp_path, "region_a", _grid(0, 1.0), BBOX_A)}
    out, _ = _run(monkeypatch, tmp_path, pkgs)
    assert out["ok"] is False
    assert "at least two" in out["error"]


def test_a_package_without_a_grid_is_named_not_silently_dropped(monkeypatch, tmp_path):
    """The export cap keeps big grids out of the .npz; that has to be said, not hidden."""
    pooled_only = tmp_path / "pooled_only.npz"
    np.savez(pooled_only, **{f"pooled__{MODEL}": np.zeros(DIMS, dtype=np.float32),
                             "meta": np.array("{}")})
    pkgs = _two_regions(tmp_path)
    pkgs["file_c"] = pooled_only
    out, _ = _run(monkeypatch, tmp_path, pkgs)

    assert out["ok"] is True, "two good packages still produce a result"
    rejected = out["packages_rejected"]
    assert [r["file_id"] for r in rejected] == ["file_c"]
    assert "pooled vector only" in rejected[0]["error"]


def test_packages_with_no_model_in_common_are_refused_with_the_reason(monkeypatch, tmp_path):
    other = tmp_path / "other_model.npz"
    np.savez(other, **{"grid__clay": _grid(2, 1.0), "meta": np.array("{}")})
    pkgs = {"file_a": _package(tmp_path, "region_a", _grid(0, 1.0), BBOX_A), "file_b": other}
    out, _ = _run(monkeypatch, tmp_path, pkgs)

    assert out["ok"] is False
    assert "share no model" in out["error"]
    assert out["per_package"][1]["models"] == ["clay"]


def test_the_same_package_passed_twice_does_not_raise(monkeypatch, tmp_path):
    """`loaded.index(entry)` compared dicts holding numpy arrays, so a repeated file_id raised
    'truth value of an array is ambiguous' and took the whole tool down."""
    pkgs = _two_regions(tmp_path)
    pkgs["file_a_again"] = pkgs["file_a"]          # the same package, twice
    out, saved = _run(monkeypatch, tmp_path, pkgs)
    assert out["ok"] is True, out
    assert len(saved) == 3


# --- re-colouring replaces the layer it supersedes -----------------------------
def _dated_package(tmp_path, name, grid, bbox, start, end):
    """A package whose manifest records its months, as the service actually writes it."""
    p = tmp_path / f"{name}.npz"
    meta = json.dumps({
        "geometry": {"type": "bbox", "minlon": bbox[0], "minlat": bbox[1],
                     "maxlon": bbox[2], "maxlat": bbox[3]},
        "start": start, "end": end,
        "models": [{"model": MODEL, "dim": DIMS, "grid_saved": True}],
    })
    np.savez(p, **{f"grid__{MODEL}": grid.astype(np.float32),
                   f"pooled__{MODEL}": grid.reshape(DIMS, -1).mean(1).astype(np.float32),
                   "meta": np.array(meta)})
    return p


def _embed_region_layer_id(bbox, start, end):
    """The id embed_region gave the layer this re-colour supersedes."""
    from agent_runtime.rs_embed_tools import _layer_id, _region_tag, _round_bbox

    return _layer_id("pca", _region_tag(None, bbox), bbox=_round_bbox(bbox),
                     model=MODEL, start=start, end=end)


def _ids(out):
    layers = out.get("map_layers") or ([out["map_layer"]] if out.get("map_layer") else [])
    return [layer["id"] for layer in layers]


def test_a_recolour_takes_over_the_layer_it_replaces(monkeypatch, tmp_path):
    """The same embedding re-coloured is the SAME layer, not a second one.

    Emitting a fresh id put each year on the map twice, and the duplicate was the misleading
    half: the per-run colours this tool exists to replace, sitting beside the aligned ones with
    only a label to tell them apart.
    """
    packages = {
        "file_2018": _dated_package(tmp_path, "y2018", _grid(0, 1.0), BBOX_A, "2018-06", "2018-09"),
        "file_2022": _dated_package(tmp_path, "y2022", _grid(1, 4.0), BBOX_A, "2022-06", "2022-09"),
    }
    out, _saved = _run(monkeypatch, tmp_path, packages)
    assert out["ok"] is True
    assert _ids(out) == [_embed_region_layer_id(BBOX_A, "2018-06", "2018-09"),
                         _embed_region_layer_id(BBOX_A, "2022-06", "2022-09")]


def test_one_place_at_several_periods_stays_several_layers(monkeypatch, tmp_path):
    """The regression guard. Collapsing periods onto one id is the bug this once had."""
    packages = {
        "file_2018": _dated_package(tmp_path, "y2018", _grid(0, 1.0), BBOX_A, "2018-06", "2018-09"),
        "file_2020": _dated_package(tmp_path, "y2020", _grid(2, 2.0), BBOX_A, "2020-06", "2020-09"),
        "file_2022": _dated_package(tmp_path, "y2022", _grid(1, 4.0), BBOX_A, "2022-06", "2022-09"),
    }
    out, _saved = _run(monkeypatch, tmp_path, packages)
    assert len(set(_ids(out))) == 3


def test_a_package_that_cannot_say_what_it_supersedes_keeps_its_own_id(monkeypatch, tmp_path):
    """No months in the manifest means no safe replacement.

    An extra layer is a worse map; a WRONG replacement would overwrite a raster of somewhere
    else, so the undated case falls back to an identity of its own.
    """
    packages = _two_regions(tmp_path)  # this helper's manifests carry no start/end
    out, _saved = _run(monkeypatch, tmp_path, packages)
    ids = _ids(out)
    assert len(set(ids)) == 2
    assert all("sharedpca" in i for i in ids)
