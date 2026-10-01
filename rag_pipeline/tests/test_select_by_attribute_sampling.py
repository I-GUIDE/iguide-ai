"""select_by_attribute must honour the invariant the ledger's features_total phrase depends on.

The phrase reads "features in the full set: N — a lower shown count means the map has a SAMPLE",
which is only true when feature_count is the count actually MAPPED. Three sampling producers set
it that way; select_by_attribute set it to the selection size, so a sampled layer read exactly
like a complete one — and adding features_total naively would have made the phrase positively
assert the layer was complete.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime.supervisor import graph as g

from rag_pipeline.tests.test_action_ledger import _pair

# --- select_by_attribute honours the invariant the features_total phrase depends on -------
#
# The phrase reads "features in the full set: N — a lower shown count means the map has a
# SAMPLE", which is only true if feature_count is the count actually MAPPED. Three sampling
# producers set it that way; select_by_attribute set it to the selection size, so a sampled
# layer read exactly like a complete one — and adding features_total naively would have made
# the phrase positively assert the layer was complete.

def _selected_layer(tmp_path, monkeypatch, n=200, threshold=0, ceiling=None):
    import json as _json

    import geopandas as gpd
    from shapely.geometry import Point

    monkeypatch.setenv("AGENT_CHAT_FILES_DIR", str(tmp_path))
    # No importlib.reload: storage_root() reads the env on every call, and reloading these
    # modules rebinds them for every OTHER test in the process. The sampling ceiling is a
    # module CONSTANT read at import, so setenv cannot move it — patch the constant.
    import agent_runtime.analysis_aggregate_tools as agg
    import agent_runtime.file_store as fs

    if ceiling is not None:
        monkeypatch.setattr(agg, "_MAP_LAYER_MAX_FEATURES", int(ceiling))

    gdf = gpd.GeoDataFrame({"val": list(range(n))},
                           geometry=[Point(-88 + i * 0.001, 40) for i in range(n)],
                           crs="EPSG:4326")
    src = tmp_path / "incidents.geojson"
    gdf.to_file(src, driver="GeoJSON")
    rec = fs.create_output_file("incidents.geojson", src.read_text())

    tools = {t.name: t for t in agg.make_aggregate_tools()}
    out = _json.loads(tools["select_by_attribute"].func(
        file_id=rec["file_id"], column="val", op=">=", value=threshold))
    assert out["ok"] is True, out
    return out


def test_a_sampled_selection_reports_the_mapped_count_and_the_full_one(tmp_path, monkeypatch):
    out = _selected_layer(tmp_path, monkeypatch, n=200, threshold=0, ceiling=50)
    assert out["feature_count"] == 50, "feature_count must be what is on the map"
    assert out["features_total"] == 200, "and features_total the full selection"
    assert out["sampled"] is True


def test_the_ledger_line_says_a_sampled_layer_is_a_sample(tmp_path, monkeypatch):
    out = _selected_layer(tmp_path, monkeypatch, n=200, threshold=0, ceiling=50)
    row = g._ledger_rows(_pair("select_by_attribute", {"column": "val"}, out))[0]
    line = g._ledger_lines([row])[0]
    assert "features: 50" in line
    assert "features in the full set: 200" in line and "SAMPLE" in line


def test_a_complete_selection_does_not_claim_to_be_a_sample(tmp_path, monkeypatch):
    """The other direction: the phrase must not fire when nothing was dropped."""
    out = _selected_layer(tmp_path, monkeypatch, n=20, threshold=0, ceiling=500)
    assert out["sampled"] is False
    assert out["feature_count"] == out["features_total"] == 20
    line = g._ledger_lines([g._ledger_rows(_pair("select_by_attribute", {"column": "val"}, out))[0]])[0]
    assert "features: 20" in line
    assert "features in the full set: 20" in line
