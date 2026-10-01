"""A map layer counts as delivered only when it would actually reach the user's screen.

Four signals used to answer this question and they were combined with `or`, so the weakest one
won: a tool NAME appearing in tool_calls (never checking success), a bare `"on_map": true`
anywhere in a nested payload, a tool name inside a recursive walk, and a regex over the JSON
blob. A FAILED `admin_boundary` — `{"ok": false}` with no descriptor on its ambiguity and error
paths — tripped two of them. The supervisor then suppressed its own corrective retry, wrote the
conclusion into `result["on_map"]`, and RE-READ that conclusion as evidence a layer existed,
which made the auditor drop every map issue and append "The layer is already on your
interactive map".

Both directions matter here: loosening this check is what caused the false positives the
reconciliation layers were built to suppress, so a genuinely delivered layer must still pass.
"""
from __future__ import annotations

import json

import pytest

from agent_runtime.map_layers import delivers_map_layer
from agent_runtime.supervisor import graph as g


def _result(name, payload, call_id="c1"):
    """A tool result in the shape every producer actually emits."""
    return {"tool_results": [{"name": name, "tool_call_id": call_id,
                              "content": json.dumps(payload)}]}


DESCRIPTOR = {"url": "/agent/files/file_a1/download", "label": "Urbana city",
              "render": "shapes", "count": 1}


# --- the failures that used to read as deliveries ----------------------------------------

def test_a_failed_admin_boundary_does_not_claim_a_layer():
    """The headline case. admin_boundary is in _MAP_LAYER_TOOLS, so the name matched."""
    ctx = _result("admin_boundary", {"ok": False, "error": "no county named 'Champaign County'",
                                     "did_you_mean": ["Champaign"]})
    assert not g._map_delivered_this_turn(ctx)
    assert not g._map_layer_was_delivered(ctx)


def test_an_ambiguous_admin_boundary_does_not_claim_a_layer():
    ctx = _result("admin_boundary", {"ok": False, "error": "16 places named 'Springfield'",
                                     "candidates": ["Springfield, IL", "Springfield, MO"]})
    assert not g._map_delivered_this_turn(ctx)


def test_an_empty_overpass_search_does_not_claim_a_layer():
    """Also in _MAP_LAYER_TOOLS: no features means build_map_layer returns None."""
    ctx = _result("overpass_search", {"ok": True, "count": 0, "features": []})
    assert not g._map_delivered_this_turn(ctx)


def test_a_descriptor_with_no_url_does_not_claim_a_layer():
    """The client fetches by url; without one there is nothing to draw."""
    ctx = _result("add_map_layer", {"ok": True, "map_layer": {"label": "x", "render": "shapes"}})
    assert not g._map_delivered_this_turn(ctx)


def test_a_bare_on_map_flag_is_not_enough():
    """`on_map: true` with no descriptor delivered nothing — vector_spatial_join's old bug."""
    ctx = _result("vector_spatial_join", {"ok": True, "on_map": True, "feature_count": 3})
    assert not g._map_delivered_this_turn(ctx)


def test_a_tool_name_alone_is_not_enough():
    """Only the call was recorded, with no result at all."""
    assert not g._map_delivered_this_turn({"tool_calls": [{"name": "add_map_layer", "args": {}}]})


# --- and the deliveries that must still count --------------------------------------------

def test_a_real_delivery_still_counts():
    ctx = _result("admin_boundary", {"ok": True, "map_layer": DESCRIPTOR})
    assert g._map_delivered_this_turn(ctx)
    assert g._map_layer_was_delivered(ctx)


def test_a_toolkit_layer_still_counts():
    """Not just add_map_layer — buffer_layer et al. deliver their own layers."""
    ctx = _result("buffer_layer", {"ok": True, "on_map": True, "map_layer": DESCRIPTOR})
    assert g._map_delivered_this_turn(ctx)


def test_multiple_layers_from_one_call_still_count():
    """embed_zones returns the raster AND the zones; either is a delivery."""
    ctx = _result("embed_zones", {"ok": True, "map_layers": [DESCRIPTOR]})
    assert g._map_delivered_this_turn(ctx)


def test_a_cli_peers_top_level_descriptor_still_counts():
    """claude/opencode return a plain dict with the descriptor at the top level."""
    assert g._map_delivered_this_turn({"map_layer": DESCRIPTOR, "answer": "done"})


def test_a_dict_content_is_handled_not_only_a_json_string():
    """The CLI peers put a live dict in `content`, not a serialized string."""
    ctx = {"tool_results": [{"name": "claude_run", "content": {"ok": True,
                                                               "map_layer": DESCRIPTOR}}]}
    assert g._map_delivered_this_turn(ctx)


def test_vector_spatial_join_now_emits_a_descriptor():
    """The tool the fix exposed: it set on_map with nothing for the client to fetch."""
    import inspect

    from agent_runtime import langchain_geo_tools as lgt

    src = inspect.getsource(lgt)
    assert '"source": "spatial_join"' in src, "vector_spatial_join must emit a map_layer"


# --- the delivery boundary itself ---------------------------------------------------------

@pytest.mark.parametrize("payload,expected", [
    ({"ok": True, "map_layer": DESCRIPTOR}, True),
    ({"ok": True, "map_layer": {"label": "no url"}}, False),
    ({"ok": True, "features": []}, False),
    ({"ok": False}, False),
    ({}, False),
])
def test_delivers_map_layer_is_the_single_authority(payload, expected):
    assert delivers_map_layer("add_map_layer", json.dumps(payload)) is expected


def test_the_authority_never_raises_on_junk():
    for junk in (None, "", "not json", 42, [], {"map_layer": "not a dict"}):
        assert delivers_map_layer("x", junk) is False


def test_skipping_qa_cannot_change_the_answer():
    """qa=False only skips a read that downgrades `render`; it must not change delivery."""
    from agent_runtime.map_layers import build_map_layers

    payload = {"ok": True, "map_layer": DESCRIPTOR}
    assert bool(build_map_layers("add_map_layer", payload, qa=True)) == \
           bool(build_map_layers("add_map_layer", payload, qa=False))


def test_the_supervisors_own_conclusion_is_not_re_read_as_evidence():
    """The feedback loop: result["on_map"] must not itself prove a layer was delivered."""
    assert not g._map_delivered_this_turn({"on_map": True, "summary": "I added the layer"})
