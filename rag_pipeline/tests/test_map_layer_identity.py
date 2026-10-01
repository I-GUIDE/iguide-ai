"""Two regions must not collapse into one map layer.

The client derives a layer's id from its LABEL and REPLACES any layer with a matching id, so a
label built from the model alone made a second `embed_region` call silently overwrite the first:
two regions embedded in one turn left a single raster on the map, with no error and nothing in
the result to say a layer had been dropped.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CHAMPAIGN = [-88.246, 40.110, -88.234, 40.121]
URBANA = [-88.212, 40.108, -88.200, 40.119]


def _client_layer_id(label: str) -> str:
    """Mirrors map-ui-prototype/src/App.tsx: ``artifact-${label.toLowerCase().replace(...)}``.

    The client then replaces an existing layer whose id matches, so two labels that reduce to
    one id means one of the two rasters never reaches the map.
    """
    return "artifact-" + re.sub(r"[^a-z0-9]+", "_", label.lower())


def _label_for(monkeypatch, *, bbox, name=None):
    import agent_runtime.rs_embed_tools as T

    def _svc(path, body=None, method="POST", **kw):
        if path == "/api/models":
            return {"models": [{"id": "gse", "type": "precomputed"}]}
        return {"results": [{"model": "gse", "ok": True, "type": "pooled", "dim": 64,
                             "grid_hw": [101, 101], "norm": 0.8,
                             "image": "data:image/png;base64,AAAA"}]}

    monkeypatch.setattr(T, "_svc", _svc)
    monkeypatch.setattr(T, "_save_png", lambda uri, stem: {
        "file_id": f"id_{stem}", "download_url": f"/files/{stem}.png"})

    tool = {t.name: t for t in T.make_rs_embed_tools()}["embed_region"]
    out = json.loads(tool.func(bbox=bbox, name=name))
    assert out["ok"] is True, out
    return out["map_layer"]["label"]


def test_two_named_regions_keep_separate_layers(monkeypatch):
    """The reported failure: embed two places in one turn, see one raster."""
    a = _label_for(monkeypatch, bbox=CHAMPAIGN, name="downtown champaign")
    b = _label_for(monkeypatch, bbox=URBANA, name="downtown urbana")
    assert a != b
    assert _client_layer_id(a) != _client_layer_id(b), "same client id means one layer is lost"


def test_two_unnamed_regions_also_keep_separate_layers(monkeypatch):
    """`name` is optional, so distinctness cannot depend on the model having supplied one."""
    a = _label_for(monkeypatch, bbox=CHAMPAIGN)
    b = _label_for(monkeypatch, bbox=URBANA)
    assert _client_layer_id(a) != _client_layer_id(b)


def test_re_embedding_the_same_region_still_replaces_its_layer(monkeypatch):
    """Distinctness must not become duplication: a refetch of the same place is an UPDATE,
    and stacking identical rasters is its own bug."""
    a = _label_for(monkeypatch, bbox=CHAMPAIGN, name="downtown champaign")
    b = _label_for(monkeypatch, bbox=CHAMPAIGN, name="downtown champaign")
    assert _client_layer_id(a) == _client_layer_id(b)

    # …and unnamed, where the bbox centre is the tag.
    c = _label_for(monkeypatch, bbox=URBANA)
    d = _label_for(monkeypatch, bbox=URBANA)
    assert _client_layer_id(c) == _client_layer_id(d)


def test_the_label_still_says_what_the_layer_is(monkeypatch):
    """The tag disambiguates; it must not displace the description."""
    label = _label_for(monkeypatch, bbox=CHAMPAIGN, name="downtown champaign")
    assert "gse embedding (PCA-RGB)" in label
    assert "downtown champaign" in label


def test_region_tag_prefers_the_callers_name():
    from agent_runtime.rs_embed_tools import _region_tag

    assert _region_tag("downtown urbana", CHAMPAIGN) == "downtown urbana"
    assert _region_tag("  spaced  ", None) == "spaced"


def test_region_tag_falls_back_to_the_region_centre():
    from agent_runtime.rs_embed_tools import _region_tag

    assert _region_tag(None, CHAMPAIGN) == _region_tag("", CHAMPAIGN), "'' is not a name"
    assert _region_tag(None, CHAMPAIGN) != _region_tag(None, URBANA)
    assert _region_tag(None, CHAMPAIGN) == _region_tag(None, list(CHAMPAIGN)), "must be stable"


def test_region_tag_degrades_to_nothing_rather_than_raising():
    """A malformed or absent bbox must not take the whole tool down over a label."""
    from agent_runtime.rs_embed_tools import _layer_label, _region_tag

    for bad in (None, [], [1, 2], "nonsense", [None, None, None, None]):
        assert _region_tag(None, bad) == ""
    assert _layer_label("gse embedding (PCA-RGB)", "") == "gse embedding (PCA-RGB)"


def test_the_distinguishing_part_leads_so_truncation_cannot_eat_it(monkeypatch):
    """The layer panel clips names with an ellipsis at ~158px against ~350px of text, so a tag
    appended AFTER the description is exactly what gets thrown away: distinct ids, but two rows
    that both read "gse embedding (PCA-R…" and cannot be told apart."""
    a = _label_for(monkeypatch, bbox=CHAMPAIGN, name="downtown champaign")
    b = _label_for(monkeypatch, bbox=URBANA, name="downtown urbana")

    assert a.startswith("downtown champaign"), a
    assert b.startswith("downtown urbana"), b
    # Roughly what survives truncation: the visible prefixes already differ.
    assert a[:18] != b[:18]
