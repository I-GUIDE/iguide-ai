"""A layer id must identify the layer.

The id was the label's slug cut to 40 characters. Identity is what the client replaces on, so
two labels agreeing in their first 40 characters silently collapsed into one layer: no error, no
log, and the tool result still reported the layer delivered.

These use the real labels from the run where it fired — two 1 km boxes embedded and then
re-coloured on a shared basis. Champaign's stem is 41 characters and lost its per-run raster;
Urbana's is 38 and kept both. A one-character margin decided it.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CHAMPAIGN_RUN = "Downtown Champaign 1 km box — gse embedding (PCA-RGB)"
CHAMPAIGN_SHARED = "Downtown Champaign 1 km box — gse embedding (shared PCA)"
URBANA_RUN = "Downtown Urbana 1 km box — gse embedding (PCA-RGB)"
URBANA_SHARED = "Downtown Urbana 1 km box — gse embedding (shared PCA)"


def _layer_id(label: str) -> str:
    """The id the server actually ships for a raster whose tool set none."""
    from agent_runtime.map_layers import build_map_layer

    built = build_map_layer("embed_region", json.dumps({"map_layer": {
        "url": "/agent/files/f/download", "label": label, "render": "raster",
        "bounds": [-88.3, 40.1, -88.2, 40.2], "opacity": 0.85, "source": "analysis"}}))
    assert built is not None, label
    return built["id"]


def test_the_four_labels_from_the_incident_get_four_ids():
    ids = [_layer_id(l) for l in (CHAMPAIGN_RUN, CHAMPAIGN_SHARED, URBANA_RUN, URBANA_SHARED)]
    assert len(set(ids)) == 4, dict(zip(
        (CHAMPAIGN_RUN, CHAMPAIGN_SHARED, URBANA_RUN, URBANA_SHARED), ids))


def test_labels_differing_only_past_the_cut_are_still_distinct():
    """The general case: the discriminator sits beyond the readable prefix."""
    stem = "a very long region name that runs past the readable prefix"
    assert _layer_id(f"{stem} — variant one") != _layer_id(f"{stem} — variant two")


def test_a_short_label_keeps_the_id_it_always_had():
    """Existing layers must not be renamed by this change: only slugs that WOULD have lost
    information carry a digest."""
    assert _layer_id("gse embedding (PCA-RGB)") == "agent-gse_embedding_pca_rgb"
    assert "_" * 2 not in _layer_id("hospitals near Chicago")


def test_the_same_label_still_maps_to_the_same_id():
    """Stability is the point of an id — re-running a tool must REPLACE its own layer, not
    stack a second copy of it."""
    assert _layer_id(CHAMPAIGN_RUN) == _layer_id(CHAMPAIGN_RUN)
    assert _layer_id(CHAMPAIGN_SHARED) == _layer_id(CHAMPAIGN_SHARED)


def test_an_explicit_id_from_the_tool_still_wins():
    from agent_runtime.map_layers import build_map_layer

    built = build_map_layer("add_map_layer", json.dumps({"map_layer": {
        "url": "http://x/y.geojson", "label": CHAMPAIGN_RUN, "render": "shapes",
        "id": "chosen-by-the-tool"}}))
    assert built["id"] == "chosen-by-the-tool"


def test_two_layers_in_one_result_survive_the_per_call_dedup():
    """build_map_layers drops a second descriptor whose id matches the first. With the ids
    collapsed that discarded a real layer before it ever left the process."""
    from agent_runtime.map_layers import build_map_layers

    out = build_map_layers("embed_region", json.dumps({"ok": True, "map_layers": [
        {"url": "/agent/files/a/download", "label": CHAMPAIGN_RUN, "render": "raster",
         "bounds": [-88.3, 40.1, -88.2, 40.2]},
        {"url": "/agent/files/b/download", "label": CHAMPAIGN_SHARED, "render": "raster",
         "bounds": [-88.3, 40.1, -88.2, 40.2]},
    ]}))
    assert len(out) == 2, [l["label"] for l in out]
    assert len({l["id"] for l in out}) == 2


def test_genuinely_identical_descriptors_are_still_deduped():
    """The dedup exists for a reason; making ids finer must not disable it."""
    from agent_runtime.map_layers import build_map_layers

    one = {"url": "/agent/files/a/download", "label": CHAMPAIGN_RUN, "render": "raster",
           "bounds": [-88.3, 40.1, -88.2, 40.2]}
    out = build_map_layers("embed_region", json.dumps({"ok": True, "map_layers": [one, dict(one)]}))
    assert len(out) == 1


def test_the_id_stays_bounded():
    """Readability was the reason for the cut; keep the id short even with the digest."""
    long_label = "x" * 400
    assert len(_layer_id(long_label)) < 80
