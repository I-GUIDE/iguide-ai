"""T13: Chicago violent crime, pinned to a snapshot, and its constraint-checked regionalization."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gis_harness import datasets
from gis_harness.score import check_regions, score
from gis_harness.tasks import BY_ID

SNAP = Path(datasets.__file__).resolve().parent / ".cache" / datasets.CHICAGO_DIR


def _grid(tmp_path: Path, labels, name="regions.geojson") -> str:
    """A 3x3 grid of unit squares, cell i at row i // 3, column i % 3, labelled as given."""
    feats = []
    for i, lab in enumerate(labels):
        x, y = i % 3, i // 3
        ring = [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1], [x, y]]
        feats.append({"type": "Feature", "properties": {"region": lab, "rate": i * 1.5},
                      "geometry": {"type": "Polygon", "coordinates": [ring]}})
    p = tmp_path / name
    p.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    return str(p)


EXPECT = {"n_regions": 3, "n_areas": 9}


def test_contiguous_regions_pass(tmp_path):
    p = _grid(tmp_path, ["a", "a", "a", "b", "b", "b", "c", "c", "c"])
    r = check_regions(EXPECT, [p])
    assert r["matched"] and r["found"].endswith(":region")


def test_a_region_in_two_pieces_fails(tmp_path):
    # "a" holds the top-left and bottom-right corners only: two pieces.
    p = _grid(tmp_path, ["a", "b", "b", "b", "c", "b", "c", "c", "a"])
    r = check_regions(EXPECT, [p])
    assert not r["matched"] and "not contiguous" in r["found"]


def test_the_wrong_number_of_regions_or_areas_fails(tmp_path):
    assert not check_regions(EXPECT, [_grid(tmp_path, ["a"] * 4 + ["b"] * 5)])["matched"]
    assert not check_regions({"n_regions": 3, "n_areas": 77},
                             [_grid(tmp_path, ["a", "a", "a", "b", "b", "b", "c", "c", "c"])])["matched"]
    assert check_regions(EXPECT, [])["found"] == "no map layer"


def test_the_regions_layer_is_found_among_several(tmp_path):
    rates = _grid(tmp_path, list(range(9)), name="rates.geojson")
    regions = _grid(tmp_path, ["a", "a", "b", "a", "b", "b", "c", "c", "c"], name="regions.geojson")
    assert check_regions(EXPECT, [rates, regions])["matched"]


def test_score_reads_the_layers_the_harness_saved(tmp_path):
    task = BY_ID["T13"]
    exp = {"citywide_rate": 30.349, "top_area": "Fuller Park", "top_rate": 118.02,
           "morans_i": 0.5646, "regions": {"n_regions": 6, "n_areas": 9}}
    labels = ["r1", "r2", "r3", "r4", "r5", "r6", "r6", "r5", "r4"]
    turn = {"answer": "Citywide 30.35 per 1,000; Fuller Park is highest at 118.0 per 1,000. "
                      "Moran's I = 0.565. Source: the attached crimes_2024.csv.",
            "layer_paths": [_grid(tmp_path, labels)]}
    s = score(task, exp, turn)
    by = {c["name"]: c["matched"] for c in s.checks}
    assert by == {"citywide_rate": True, "top_area": True, "top_rate": True, "morans_i": True,
                  "regions": False}, by     # r4 and r5 are split in this grid


@pytest.mark.skipif(not SNAP.exists(), reason="the pinned Chicago snapshot is not in this checkout")
def test_the_pinned_snapshot_gives_the_reference_values(tmp_path):
    files, exp = datasets.build("chicago_crime", tmp_path, SNAP.parent)
    assert sorted(files) == sorted(datasets.CHICAGO_FILES)
    assert exp["citywide_rate"] == pytest.approx(30.349, abs=0.001)
    assert exp["top_area"] == "Fuller Park" and exp["top_rate"] == pytest.approx(118.02, abs=0.01)
    assert exp["morans_i"] == pytest.approx(0.5646, abs=0.0001)
    assert exp["regions"] == {"n_regions": 6, "n_areas": 77}
    assert "80353 counted" in exp["_why"]


def test_a_tampered_snapshot_is_refused(tmp_path):
    d = tmp_path / datasets.CHICAGO_DIR
    d.mkdir()
    for name in datasets.CHICAGO_FILES:
        (d / name).write_text("not the data")
    with pytest.raises(RuntimeError, match="do not match the pinned SHA-256"):
        datasets._chicago_snapshot(tmp_path)


def test_a_missing_snapshot_says_where_to_get_it(tmp_path, monkeypatch):
    monkeypatch.delenv("GIS_HARNESS_SNAPSHOTS", raising=False)
    with pytest.raises(FileNotFoundError, match="gis_harness_runs_archive/datasets"):
        datasets._chicago_snapshot(tmp_path)
