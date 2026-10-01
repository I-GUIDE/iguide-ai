"""The conversation remembers what it DID, not just what it said.

From a live failure: turn 1 computed a clay embedding whose result carried
`pixel_ground_m`; turn 2 asked "do you use the original resolution of clay or do you
downsample it". The supervisor's decision payload is built from per-turn state, so it reported
has_evidence=False / has_analysis=False / artifacts_produced=[] and routed to `search` —
correct, given what it was shown. Forty-nine keyword searches later the payload exceeded the
model's context window and the turn died with a 400. The answer had been in hand all along.
"""

from __future__ import annotations

import json

import pytest

from agent_runtime import session_memory as sm
from agent_runtime.supervisor import graph as g

CLAY_RESULT = {
    "model": "clay", "dim": 1024, "file_id": "file_d0bb",
    "provenance": {"model": "clay", "scale_m": 10, "pixel_ground_m": 10, "year": 2025},
    "map_layer": {"url": "/f/1", "label": "clay embedding (PCA-RGB)"},
}


def _real_artifacts():
    """Built through extract_search_artifacts, NOT hand-written.

    The first version of these tests invented the shape ({"name", "output"}) and passed while
    the feature was inert against production artifacts, which use
    {"name", "tool_call_id", "content"} with content as a JSON *string*. Going through the
    real extractor is what makes this a contract test instead of a mirror of my assumptions.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    from agent_runtime.runtime_utils import extract_search_artifacts

    call = AIMessage(content="", tool_calls=[{
        "name": "embed_region", "id": "c1",
        "args": {"lon": -88.192988, "lat": 40.115105, "model": "clay",
                 "start": "2025-03-01", "end": "2025-05-01"}}])
    result = ToolMessage(content=json.dumps(CLAY_RESULT), name="embed_region", tool_call_id="c1")
    return extract_search_artifacts({"messages": [call, result]})


CLAY_TURN = _real_artifacts()


@pytest.fixture(autouse=True)
def clean():
    sm.reset_all()
    yield
    sm.reset_all()


def test_a_call_and_its_result_become_one_row():
    rows = g._ledger_rows(CLAY_TURN)
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["tool"] == "embed_region"
    assert row["args"]["model"] == "clay" and row["args"]["start"] == "2025-03-01"
    # A list: one call can map several layers, and the row records all of them.
    assert row["map_layer"] == ["clay embedding (PCA-RGB)"]
    assert row["file_id"] == "file_d0bb"


def test_the_row_carries_the_answer_to_the_follow_up():
    """`pixel_ground_m` is what "original resolution or downsampled?" is asking for."""
    facts = g._ledger_rows(CLAY_TURN)[0]["facts"]
    assert facts["pixel_ground_m"] == 10
    assert facts["scale_m"] == 10 and facts["model"] == "clay" and facts["dim"] == 1024


def test_provenance_is_flattened_into_the_row():
    """rs-embed nests the fields that say what a number MEANS under `provenance`."""
    rows = g._ledger_rows({"tool_results": [{
        "name": "embed_region", "tool_call_id": "c1",
        "content": json.dumps({"provenance": {"scale_m": 30, "pixel_ground_m": 30}})}]})
    assert rows[0]["facts"] == {"scale_m": 30, "pixel_ground_m": 30}


def test_the_decider_is_told_what_the_conversation_already_did():
    g._record_actions({"thread_id": "t1"}, CLAY_TURN)
    state = {"thread_id": "t1", "query": "what resolution was that", "evidence": [],
             "analysis_results": None, "code_result": None, "step": 0}
    payload = g._distill(state, for_decision=True)
    # Per-turn state still reports an empty conversation — that is the bug being compensated.
    assert payload["has_evidence"] is False and payload["has_analysis"] is False
    prior = payload["prior_turns_in_this_conversation"]
    assert prior[0]["facts"]["pixel_ground_m"] == 10
    assert "done" in payload["available_actions"]     # …and it may act on it
    # The note must warn against blind reuse, or this becomes a staleness bug.
    assert "NOT as inputs to reuse blindly" in payload["prior_turns_note"]


def test_the_current_turn_is_not_in_its_own_prior_context():
    """_record_actions runs at synthesis; the decision runs before it, every step."""
    state = {"thread_id": "t1", "query": "q", "evidence": [], "step": 0}
    assert "prior_turns_in_this_conversation" not in g._distill(state, for_decision=True)
    g._record_actions(state, CLAY_TURN)
    assert g._distill(state, for_decision=True)["prior_turns_in_this_conversation"]


def test_the_client_payload_is_not_burdened_with_it():
    g._record_actions({"thread_id": "t1"}, CLAY_TURN)
    state = {"thread_id": "t1", "query": "q", "evidence": [], "step": 0}
    assert "prior_turns_in_this_conversation" not in g._distill(state)


def test_the_ledger_stays_small_enough_to_inject():
    """It is injected into a routing decision; a big one recreates the failure it prevents."""
    for _ in range(6):
        g._record_actions({"thread_id": "t1"}, CLAY_TURN)
    rows = sm.get_session_actions("t1")
    assert len(json.dumps(rows)) < 4000, len(json.dumps(rows))


def test_long_values_are_truncated():
    rows = g._ledger_rows({"tool_calls": [{"name": "x", "args": {"query": "z" * 500}}]})
    assert len(rows[0]["args"]["query"]) <= g._LEDGER_VALUE_CHARS + 1


def test_only_curated_fields_are_kept():
    """A dumped tool result would be the context problem again, in a new place."""
    rows = g._ledger_rows({"tool_results": [{
        "name": "embed_region", "tool_call_id": "c1", "content": json.dumps({
            "model": "clay", "scale_m": 10,
            "vectors": list(range(1000)), "debug": {"param_absmax": 3.2},
            "stdout": "x" * 5000})}]})
    assert set(rows[0]["facts"]) == {"model", "scale_m"}
    assert "vectors" not in json.dumps(rows[0]) and "stdout" not in json.dumps(rows[0])


def test_the_store_is_bounded_and_keeps_the_newest(monkeypatch):
    monkeypatch.setenv("AGENT_SESSION_MEMORY_MAX_ACTIONS", "5")
    sm.append_session_actions("t1", [{"tool": f"t{i}"} for i in range(20)])
    kept = sm.get_session_actions("t1")
    assert len(kept) == 5 and kept[-1]["tool"] == "t19"    # a follow-up is about recent work


def test_no_thread_id_is_a_no_op():
    g._record_actions({}, CLAY_TURN)
    assert sm.get_session_actions(None) == []
    assert g._distill({"query": "q", "evidence": [], "step": 0}, for_decision=True) is not None


def test_clearing_a_session_clears_its_actions():
    g._record_actions({"thread_id": "t1"}, CLAY_TURN)
    sm.clear_session("t1")
    assert sm.get_session_actions("t1") == []


def test_a_broken_ledger_never_breaks_routing(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("store exploded")

    monkeypatch.setattr(sm, "get_session_actions", boom)
    monkeypatch.setattr(sm, "append_session_actions", boom)
    state = {"thread_id": "t1", "query": "q", "evidence": [], "step": 0}
    g._record_actions(state, CLAY_TURN)          # must not raise
    assert g._distill(state, for_decision=True)["has_evidence"] is False


# --- the note the ANSWERING model reads -------------------------------------

def test_facts_are_phrased_in_the_words_a_user_would_use():
    """Given `scale_m=10` the model still answered "the available evidence does not specify
    the ground-resolution" — it did not connect the field name to the question. Spelling it
    out is what turned that into "generated at a 10-meter ground resolution"."""
    note = g._prior_actions_note([{"tool": "embed_region", "args": {"model": "gse"},
                                   "facts": {"model": "gse", "scale_m": 10, "year": 2022}}])
    assert "ground resolution" in note and "10 m per pixel" in note
    assert "do not re-derive it" in note          # and says not to go looking for it again


def test_an_unmapped_fact_still_appears():
    note = g._prior_actions_note([{"tool": "x", "facts": {"nodata_fraction": 0.0}}])
    assert "nodata_fraction=0.0" in note


def test_no_rows_means_no_note():
    assert g._prior_actions_note([]) is None


def test_facts_are_found_in_embed_regions_real_nesting():
    """embed_region returns {compute, embedding_package, map_layer, models:[{provenance}]} —
    the facts sit TWO levels down. Reading only the top level found nothing, so the ledger
    recorded a bare `embed_region` row and the follow-up still answered "the evidence does not
    provide specific details"."""
    payload = {
        "compute": "get_embedding", "ok": True,
        "map_layer": {"label": "clay embedding (PCA-RGB)", "url": "/x"},
        "models": [{"model": "clay", "dim": 1024, "grid": [52, 52],
                    "provenance": {"image_size": 256, "patch_size": 8,
                                   "source": "sentinel-2", "normalization": "raw"}}],
    }
    row = g._ledger_rows({"tool_results": [
        {"name": "embed_region", "tool_call_id": "c1", "content": json.dumps(payload)}]})[0]
    assert row["facts"]["model"] == "clay" and row["facts"]["dim"] == 1024
    # An on-the-fly model has no scale_m at all; this is what answers the resolution question.
    assert row["facts"]["image_size"] == 256 and row["facts"]["patch_size"] == 8
    note = g._prior_actions_note([row])
    assert "RESAMPLED" in note


# --- context robustness ------------------------------------------------------

def test_the_rendered_ledger_is_capped_in_characters_not_just_rows():
    """It exists because a turn overflowed the context window; it must not be able to cause
    that itself. 25 rows of long arguments would otherwise reach several thousand tokens."""
    fat = [{"tool": f"t{i}", "args": {"query": "z" * 78, "place": "y" * 78},
            "facts": {"source": "s" * 78}} for i in range(40)]
    kept = g._budgeted(fat)
    assert len(json.dumps(kept)) <= g._LEDGER_MAX_CHARS + 400   # one row may straddle the line
    assert len(kept) < len(fat)
    assert kept[-1]["tool"] == "t39"          # the NEWEST rows survive; oldest drop first


def test_a_small_ledger_is_untouched_by_the_budget():
    rows = [{"tool": "admin_boundary", "facts": {"geoid": "17019"}}]
    assert g._budgeted(rows) == rows


def test_one_huge_row_still_renders():
    """Never return nothing: a single oversized row is better than silence."""
    assert len(g._budgeted([{"tool": "x", "args": {"q": "z" * 5000}}])) == 1


def test_every_peer_and_the_router_see_the_ledger():
    """The search peer starts on a fresh thread each turn, so it was the one structurally
    incapable of knowing the answer was already in hand — it searched SoilGrids for soil clay
    while scale_m sat in the previous turn's result. Pin every exposure point."""
    import inspect

    from agent_runtime.supervisor import graph

    for fn in (graph.default_search_fn, graph.default_analyze_fn, graph.default_code_fn):
        assert "_prior_actions_note(_prior_actions(state))" in inspect.getsource(fn), fn.__name__
    # the router gets the structured rows; synthesis renders the note AND hands the same
    # lines to the grounding auditor as part of the execution record.
    src = inspect.getsource(graph)
    assert "_budgeted(_prior_actions(state))" in src
    syn = inspect.getsource(graph.build_supervisor_graph)
    # Synthesis and the auditor share ONE rendering. Delivery itself is asserted behaviourally
    # by test_the_note_reaches_the_answerer_* — a grep cannot tell "wired" from "wired to a
    # channel that discards it", which is exactly how the chat_history route shipped broken.
    assert "_ledger_text = " in syn
    assert '"prior_actions": _ledger_text' in syn
    assert "do_synthesize(q, evidence, ar, cr, _history, _note)" in syn


# --- the ledger must reach the grounding auditor, not just the answering model -------------
#
# Observed live (Champaign County tracts, unified peer): turn 2 embedded with gse, turn 3 asked
# "what parameters were used?". The answer correctly read 64 dims / 7.645 m per pixel off the
# ledger — and then the grounding caveat appended "may not be fully supported by the retrieved
# evidence (severity: high)". The answerer had the ledger; the auditor did not. The user saw the
# feature accuse itself of hallucinating.

def test_the_auditor_is_given_the_same_earlier_turn_records_as_the_answerer():
    from agent_runtime.evidence_quality import _format_execution_context

    lines = g._ledger_lines(g._ledger_rows(CLAY_TURN))
    assert lines, "ledger produced no lines to hand over"
    rendered = _format_execution_context({"prior_actions": lines})
    assert "earlier turns" in rendered.lower()
    # the values a follow-up would quote must survive into the auditor's record
    assert "1024" in rendered and "clay" in rendered


def test_a_cross_turn_number_is_not_flagged_as_unsupported():
    """The deterministic reconciliation must see ledger values, not only this turn's output."""
    audit = {"severity": "high", "hallucination_detected": True,
             "issues": [{"claim": "the embeddings have 1024 dimensions",
                         "reason": "not supported by the retrieved evidence"}]}
    lines = g._ledger_lines(g._ledger_rows(CLAY_TURN))
    # Without the ledger the issue survives...
    kept = g._reconcile_audit_with_artifacts(
        dict(audit), [], execution_context={"analysis_results": None, "code_result": None})
    assert (kept or {}).get("issues"), "control: should still be flagged with no ledger"
    # ...with it, the number is found in the execution record and the caveat is dropped.
    reconciled = g._reconcile_audit_with_artifacts(
        dict(audit), [], execution_context={"analysis_results": None, "code_result": None,
                                            "prior_actions": lines})
    assert not g._audit_flagged(reconciled), reconciled


def test_a_map_layer_from_an_earlier_turn_still_counts_as_delivered():
    """The map is persistent: a layer added in turn 2 is still on screen in turn 4.

    The cross-turn signal now reads the ledger row's `map_layer` FIELD, passed explicitly as
    prior_rows. It used to regex the rendered line for the literal "[on the map as " — two
    hand-synced strings in different functions, where reformatting one would silently switch
    the other off and staple a hallucination caveat onto a real layer.
    """
    rows = g._ledger_rows(CLAY_TURN)
    assert any(r.get("map_layer") for r in rows), "fixture must carry a map_layer row"
    assert g._map_layer_was_delivered(None, rows)
    assert g._map_delivered_earlier(rows)
    assert not g._map_delivered_earlier([{"tool": "keyword_search"}])
    assert not g._map_layer_was_delivered({"analysis_results": {"note": "nothing mapped"}})


def test_a_genuinely_invented_number_is_still_flagged():
    """The widened record must not become a blanket amnesty."""
    audit = {"severity": "high", "hallucination_detected": True,
             "issues": [{"claim": "the study covered 4096 counties",
                         "reason": "no evidence for this figure"}]}
    lines = g._ledger_lines(g._ledger_rows(CLAY_TURN))
    kept = g._reconcile_audit_with_artifacts(
        dict(audit), [], execution_context={"prior_actions": lines})
    assert g._audit_flagged(kept), "an invented figure must survive reconciliation"


# --- the note must actually ARRIVE at the answering model ---------------------------------
#
# It used to be prepended as chat_history item 0, and _format_chat_history renders `[-8:]` then
# tail-truncates at 4000 chars — both trims cut exactly where it sat. Measured: present at 2-7
# history items, ABSENT at 8+, and absent with 5 large items. The auditor still received it, so
# the two consumers that _prior_actions_note's own docstring says must agree disagreed on every
# conversation longer than seven exchanges. The browser check that passed had 2 history items.
#
# These tests assert DELIVERY, not source text. The previous grep-style assertion could not
# distinguish "wired" from "wired to a channel that discards it".

def _long_history(n, filler=60):
    return [{"role": "user" if i % 2 == 0 else "assistant",
             "content": f"turn {i} " + "x" * filler} for i in range(n)]


def _run_with_history(history, rows_turn):
    """Run a full supervisor turn on a thread that already has ledger rows, capturing synthesis."""
    captured = {}

    def _synth(q, ev, ar, cr, ch=None, pa=None):
        captured["chat_history"], captured["note"] = ch, pa
        return "answer"

    g._record_actions({"thread_id": "t-deliver"}, rows_turn)
    graph = g.build_supervisor_graph(
        search_fn=lambda q, s: [], analyze_fn=lambda *a: {"summary": "s"},
        synthesize_fn=_synth, decide_fn=lambda *a, **k: ("done", "why"),
        do_rerank=False, do_audit=False)
    graph.invoke({"query": "what resolution was that?", "thread_id": "t-deliver",
                  "chat_history": history, "evidence": [], "actions": []})
    return captured


@pytest.mark.parametrize("n_items", [2, 7, 8, 12, 20])
def test_the_note_reaches_the_answerer_however_long_the_conversation(n_items):
    """The regression that shipped: this fails at n_items >= 8 before the fix."""
    cap = _run_with_history(_long_history(n_items), CLAY_TURN)
    assert cap["note"], f"no ledger note delivered at {n_items} history items"
    assert "clay" in cap["note"], cap["note"]


def test_a_large_note_is_not_truncated_away_by_the_history_budget():
    cap = _run_with_history([{"role": "user", "content": "y" * 900} for _ in range(5)], CLAY_TURN)
    assert cap["note"] and "clay" in cap["note"]


def test_the_note_is_not_smuggled_through_chat_history():
    """It must travel as its own argument, so no history trim can reach it."""
    cap = _run_with_history(_long_history(4), CLAY_TURN)
    hist = cap["chat_history"] or []
    assert not any(g._LEDGER_HEADING in str(i.get("content", "")) for i in hist if isinstance(i, dict))


def test_the_answerer_and_the_auditor_see_the_same_lines():
    """One rendering, two consumers — the property the whole feature rests on."""
    from agent_runtime.evidence_quality import _format_execution_context

    # Span many TOOLS, not one repeated: _LEDGER_ROWS_PER_TOOL deliberately thins repeats of
    # a single tool to 3, so a x12 fixture of one tool no longer exceeds the old cap.
    rows = [{"tool": f"tool_{i}", "args": {"query": "q" * 100},
             "facts": {"results_returned": i}} for i in range(20)]
    text = "\n".join(g._ledger_lines(rows))
    assert len(text) > 2200, "make the fixture bigger; this must exceed the old cap"
    assert text in g._prior_actions_note(rows)
    assert text in _format_execution_context({"prior_actions": text})


def test_the_synthesizer_prompt_carries_the_note_and_names_the_section():
    calls = []

    class _Rec:
        def invoke(self, prompt):
            calls.append(prompt)
            return "ok"

    note = g._prior_actions_note(g._ledger_rows(CLAY_TURN))
    g.default_synthesize_fn(llm=_Rec())("q", [], None, None, None, note)
    assert g._LEDGER_HEADING in calls[0]
    assert "8. EARLIER-TURN TOOL RECORDS" in calls[0], "rule 8 missing from SYNTHESIS_PROMPT"


def test_a_custom_five_argument_synthesize_fn_still_works():
    """`lambda *a` doubles are the common shape; a 6th POSITIONAL arg keeps them working."""
    graph = g.build_supervisor_graph(
        search_fn=lambda q, s: [], synthesize_fn=lambda *a: "ans",
        decide_fn=lambda *a, **k: ("done", "why"), do_rerank=False, do_audit=False)
    out = graph.invoke({"query": "q", "thread_id": "t-arity", "actions": [],
                        "evidence": [{"doc_id": "d1", "title": "t", "contents": "c"}]})
    assert out["final_answer"] == "ans"


# --- an EARLIER turn's artifact is still downloadable --------------------------------------
#
# sanitize_answer_links verifies a download link only against the allowlist handed to it, and
# that list is built from THIS turn's analysis_results/code_result. Today an earlier turn's
# artifact survives by accident, because a peer's checkpointed thread replays its old tool
# results into this turn's payload. Scoping artifacts to their own turn — the correct fix for
# four verifiers currently fooled by the same replay — removes the accident, and a turn-4 answer
# offering a turn-1 CSV would have its link silently degraded to plain text.
#
# _refs_in_history reads those references out of the conversation instead. It is also strictly
# better than the replay TODAY: the claude/opencode peers return a plain dict and never had a
# checkpointed thread, so an artifact they produced has never been re-offerable.

HIST = [
    {"role": "user", "content": "embed the tracts"},
    {"role": "assistant",
     "content": "Done — [vectors](https://iguide.test/agent/files/file_7753f6d913c2/download)."},
]


def test_a_link_to_an_earlier_turns_file_is_recognised():
    refs = g._refs_in_history(HIST)
    assert "file_7753f6d913c2" in refs["file_ids"]
    assert "https://iguide.test/agent/files/file_7753f6d913c2/download" in refs["urls"]
    # the markdown "(" must not be swallowed into the URL
    assert not any(u.startswith("(") for u in refs["urls"])


@pytest.mark.parametrize("content,expected", [
    ("[a](https://h/agent/files/file_aa1111/download)", "file_aa1111"),
    ("bare https://h/agent/files/file_bb2222/download here", "file_bb2222"),
    ("<https://h/agent/files/file_cc3333/download>", "file_cc3333"),
    ("saved as file_dd4444ee", "file_dd4444ee"),
])
def test_every_form_a_reference_takes_in_an_answer_is_found(content, expected):
    assert expected in g._refs_in_history([{"role": "assistant", "content": content}])["file_ids"]


def test_an_invented_file_id_is_still_not_trusted():
    """The mitigation must not become blanket amnesty: only ids ALREADY shown to this user."""
    refs = g._refs_in_history(HIST)
    assert "file_deadbeef99" not in refs["file_ids"]


def test_an_empty_or_odd_history_is_harmless():
    for hist in (None, [], [("user", "hi")], ["a bare string"], [{"role": "user"}]):
        refs = g._refs_in_history(hist)
        assert refs == {"file_ids": [], "urls": []} or not refs["file_ids"]


def test_the_sanitizer_allowlist_includes_the_history_references():
    """Pin the wiring, since the helper alone protects nothing."""
    import inspect

    src = inspect.getsource(g.build_supervisor_graph)
    assert 'hist_refs = _refs_in_history(state.get("chat_history"))' in src
    assert '*hist_refs["file_ids"]' in src
    assert '*hist_refs["urls"]' in src


def test_an_earlier_turn_link_survives_sanitation_end_to_end():
    """The real contract, and the exact shape the risk takes.

    `sanitize_answer_links` only verifies when the allowlist is NON-EMPTY
    (`verifying = bool(ids or urls)`) — an empty list means "do not verify", not "verify against
    nothing". So the regression needs a turn that produced its OWN artifact: the allowlist is
    then non-empty, verification switches on, and an earlier turn's link is the one thing in the
    answer that cannot be vouched for. That is a normal answer shape — "here is the new layer,
    plus the CSV from before".
    """
    from agent_runtime.runtime_utils import sanitize_answer_links

    old_url = "https://iguide.test/agent/files/file_7753f6d913c2/download"   # turn 1
    new_url = "https://iguide.test/agent/files/file_99newnew9999/download"   # this turn
    answer = f"New layer [here]({new_url}); the earlier vectors are [still here]({old_url})."
    this_turn = {"file_ids": ["file_99newnew9999"], "urls": [new_url]}

    # WITHOUT the history references: verification is on, and turn 1's link is stripped.
    without = sanitize_answer_links(answer, allowed_file_ids=this_turn["file_ids"],
                                    allowed_urls=this_turn["urls"])
    assert new_url in without, "control: this turn's own link must always survive"
    assert old_url not in without, (
        "control failed — the regression this mitigation exists for is not reproducible")

    # WITH them: both survive.
    hist = g._refs_in_history(HIST)
    with_hist = sanitize_answer_links(
        answer,
        allowed_file_ids=[*this_turn["file_ids"], *hist["file_ids"]],
        allowed_urls=[*this_turn["urls"], *hist["urls"]])
    assert old_url in with_hist and new_url in with_hist


def test_verification_stays_strict_for_a_file_the_conversation_never_saw():
    """The mitigation widens the allowlist; it must not switch verification off."""
    from agent_runtime.runtime_utils import sanitize_answer_links

    fake = "https://iguide.test/agent/files/file_deadbeef99/download"
    answer = f"Here is [the data]({fake})."
    hist = g._refs_in_history(HIST)
    out = sanitize_answer_links(answer, allowed_file_ids=hist["file_ids"],
                                allowed_urls=hist["urls"])
    assert fake not in out, "an invented artifact link must still be defused"


# --- the ledger can now answer the questions people actually ask ---------------------------
#
# It could not say what was searched, what code computed, what Moran's I came back, which
# distance a buffer used, or whether anything FAILED. Worse, a failed call and a successful one
# of the same tool were paired by position within the tool name, so the failed call wore the
# good run's result — including its file_id and map_layer, which then read as a delivered layer.

def _pair(name, args, payload, call_id="c1", result_id=None):
    """A call/result pair in the shape extract_search_artifacts emits."""
    from langchain_core.messages import AIMessage, ToolMessage

    from agent_runtime.runtime_utils import extract_search_artifacts

    return extract_search_artifacts({"messages": [
        AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}]),
        ToolMessage(content=json.dumps(payload), name=name,
                    tool_call_id=result_id or call_id),
    ]})


def _merge(*artifact_sets):
    out = {"tool_calls": [], "tool_results": []}
    for a in artifact_sets:
        out["tool_calls"] += a["tool_calls"]
        out["tool_results"] += a["tool_results"]
    return out


def test_a_failed_call_does_not_wear_the_next_calls_result():
    """The headline pairing bug, with the fail-then-succeed pair that produced it."""
    bad = _pair("embed_zones", {"model": "clay", "file_id": "BAD"},
                {"ok": False, "error": "no such file: BAD"}, call_id="c1")
    good = _pair("embed_zones", {"model": "gse", "file_id": "GOOD"},
                 {"ok": True, "dim": 64, "file_id": "OUT",
                  "map_layer": {"url": "/f/1", "label": "gse zones"}}, call_id="c2")
    rows = g._ledger_rows(_merge(bad, good))

    failed = [r for r in rows if r.get("args", {}).get("file_id") == "BAD"]
    assert len(failed) == 1
    assert failed[0].get("failed") is True
    assert "facts" not in failed[0], "a failed call has no facts"
    assert "map_layer" not in failed[0], "and must not inherit the good run's layer"

    ok = [r for r in rows if r.get("args", {}).get("file_id") == "GOOD"]
    assert len(ok) == 1 and ok[0]["facts"]["dim"] == 64


def test_a_failure_reads_as_a_failure():
    rows = g._ledger_rows(_pair("embed_zones", {"model": "clay"},
                                {"ok": False, "error": "TypeError: bad geometry"}))
    note = g._prior_actions_note(rows)
    assert "FAILED embed_zones" in note
    assert "DID NOT RUN" in note and "bad geometry" in note
    assert "never describe it as completed" in note


def test_execute_code_leaves_a_row():
    """It produced ZERO rows before: none of its args or result keys were curated."""
    rows = g._ledger_rows(_pair(
        "execute_code",
        {"code": "print(1)", "language": "python", "label": "sum_1_to_100"},
        {"ok": True, "exit_code": 0, "stdout": "5050",
         "artifacts": [{"filename": "out.csv"}]}))
    assert len(rows) == 1
    assert rows[0]["args"]["label"] == "sum_1_to_100"
    assert "code" not in rows[0]["args"], "80 chars of a program is noise"
    assert rows[0]["outputs"] == "out.csv"


def test_a_morans_i_survives_its_nesting():
    """results.morans_i.statistic is two levels down; a flat key caught a truncated dict."""
    rows = g._ledger_rows(_pair(
        "global_spatial_autocorrelation", {"file_id": "f1", "column": "income", "weights": "queen"},
        {"ok": True, "results": {"morans_i": {"statistic": 0.42, "p_value": 0.001,
                                              "significance": "significant"}}}))
    assert "0.42" in str(rows[0]["facts"]["morans_i"])
    assert rows[0]["args"]["column"] == "income" and rows[0]["args"]["weights"] == "queen"
    assert "Moran's I: 0.42" in g._prior_actions_note(rows)


def test_a_buffer_distance_is_recorded():
    rows = g._ledger_rows(_pair("buffer_layer", {"file_id": "f1", "distance": 2, "units": "km"},
                                {"ok": True, "feature_count": 1}))
    line = g._prior_actions_note(rows)
    assert "distance=2" in line and "units=km" in line


def test_a_search_result_is_not_reported_as_imagery():
    """`source` means the search METHOD for retrieval and the IMAGERY for rs-embed.

    Rendered naively the ledger asserted "imagery source: keyword" as a fact about imagery.
    """
    rows = g._ledger_rows(_pair("keyword_search", {"query": "flood risk"},
                                {"ok": True, "source": "keyword", "count": 2}))
    note = g._prior_actions_note(rows)
    assert "imagery source" not in note
    assert "retrieval method: keyword" in note and "documents returned: 2" in note


def test_partial_coverage_is_not_silently_dropped():
    rows = g._ledger_rows(_pair("embed_zones", {"model": "gse", "max_tiles": 24},
                                {"ok": True, "zones_total": 48, "zones_with_pixels": 3,
                                 "tiles_planned": 1140, "truncated": "24 of 1140 tiles"}))
    note = g._prior_actions_note(rows)
    assert "PARTIAL COVERAGE" in note
    assert "zones that actually got pixels: 3" in note


def test_every_curated_fact_key_has_a_phrase():
    """Mechanically prevents the four bare key=value gaps from recurring."""
    missing = set(g._LEDGER_FACTS) - set(g._FACT_PHRASES)
    assert missing == set(), f"facts with no phrase: {sorted(missing)}"


def test_an_id_less_peer_still_pairs():
    """claude/opencode build {"name","args"} / {"name","content"} with no ids at all."""
    ctx = {"tool_calls": [{"name": "claude_run", "args": {"label": "x"}}],
           "tool_results": [{"name": "claude_run",
                             "content": {"ok": True, "file_id": "f9"}}]}
    rows = g._ledger_rows(ctx)
    assert len(rows) == 1 and rows[0]["file_id"] == "f9" and rows[0]["args"]["label"] == "x"


def test_a_nested_name_is_not_mistaken_for_a_tool():
    """admin_boundary's `matched` entries are {"name": "Champaign County", ...}."""
    rows = g._ledger_rows(_pair(
        "admin_boundary", {"area": "Champaign", "state": "Illinois", "level": "county"},
        {"ok": True, "matched": [{"name": "Champaign County", "geoid": "17019"}],
         "file_id": "f1"}))
    assert [r["tool"] for r in rows] == ["admin_boundary"]


def test_one_tool_cannot_crowd_out_the_others():
    searches = [{"tool": "keyword_search", "args": {"query": f"q{i}"}} for i in range(40)]
    kept = g._budgeted([*searches, {"tool": "embed_zones", "facts": {"dims": 64}}])
    assert any(r["tool"] == "embed_zones" for r in kept), "the analysis row must survive"
    assert sum(1 for r in kept if r["tool"] == "keyword_search") <= g._LEDGER_ROWS_PER_TOOL


def test_an_internal_knob_is_not_amnesty_for_an_invented_figure():
    """Why permutations/tile_px/timeout_seconds are deliberately NOT curated."""
    rows = g._ledger_rows(_pair("global_spatial_autocorrelation",
                                {"file_id": "f1", "permutations": 999},
                                {"ok": True, "results": {}}))
    blob = json.dumps(rows)
    assert "999" not in blob, "a 3-digit knob in the record excuses a fabricated 999"


# --- the search peer leaves a trace too ----------------------------------------------------
#
# search_node returned only evidence/attempts/streak/queries and discarded the peer's tool
# calls, and _record_actions was passed only analysis_results/code_result. So every retrieval
# method left NO trace: the ledger could not answer "what did we search for?", the single most
# common follow-up. Three producers are involved and only one leaves an artifact to extract —
# the deterministic sweep, the open-web fallback and the short-circuits all call the backends
# directly, so they need rows built for them.

def _decides(seq):
    """decide_fn(state, distilled) -> action string; 'done' once the script runs out."""
    box = {"i": 0}

    def decide(state, distilled):
        i = box["i"]
        box["i"] += 1
        return seq[i] if i < len(seq) else "done"

    return decide


def test_the_search_node_carries_its_rows_to_the_ledger():
    """End to end: search_fn -> action_rows -> state -> _record_actions -> the ledger."""
    row = g._search_row("keyword_search", "flood risk", "keyword", 2)
    graph = g.build_supervisor_graph(
        search_fn=lambda q, s: {"documents": [{"doc_id": "d1", "title": "t", "contents": "c"}],
                                "action_rows": [row]},
        synthesize_fn=lambda *a: "ans",
        decide_fn=_decides(["search", "done"]), do_rerank=False, do_audit=False)
    graph.invoke({"query": "flood risk", "thread_id": "t-search", "evidence": [], "actions": []})

    rows = sm.get_session_actions("t-search")
    assert any(r["tool"] == "keyword_search" for r in rows), rows
    note = g._prior_actions_note(rows)
    assert "retrieval method: keyword" in note and "documents returned: 2" in note


def test_a_list_returning_search_fn_still_works():
    """~30 existing doubles return a plain list; that must keep working."""
    graph = g.build_supervisor_graph(
        search_fn=lambda q, s: [{"doc_id": "d1", "title": "t", "contents": "c"}],
        synthesize_fn=lambda *a: "ans",
        decide_fn=_decides(["search", "done"]), do_rerank=False, do_audit=False)
    out = graph.invoke({"query": "q", "thread_id": "t-list", "evidence": [], "actions": []})
    assert out["final_answer"] == "ans"


def test_the_client_payload_is_not_burdened_with_the_turn_rows():
    """The state ships to the client verbatim, and the rows are already in the ledger."""
    graph = g.build_supervisor_graph(
        search_fn=lambda q, s: {"documents": [{"doc_id": "d1", "title": "t", "contents": "c"}],
                                "action_rows": [g._search_row("keyword_search", "q", "keyword", 1)]},
        synthesize_fn=lambda *a: "ans",
        decide_fn=_decides(["search", "done"]), do_rerank=False, do_audit=False)
    out = graph.invoke({"query": "q", "thread_id": "t-clean", "evidence": [], "actions": []})
    assert not out.get("action_rows"), "synthesize must clear them on the way out"


def test_rows_from_several_search_steps_accumulate():
    """SupervisorState has no reducers, so a plain overwrite would lose the earlier step."""
    calls = {"n": 0}

    def _search(q, s):
        calls["n"] += 1
        return {"documents": [{"doc_id": f"d{calls['n']}", "title": "t", "contents": "c"}],
                "action_rows": [g._search_row(f"search_{calls['n']}", q, "keyword", 1)]}

    graph = g.build_supervisor_graph(
        search_fn=_search, synthesize_fn=lambda *a: "ans",
        decide_fn=_decides(["search", "search", "done"]), do_rerank=False, do_audit=False)
    graph.invoke({"query": "q", "thread_id": "t-accum", "evidence": [], "actions": []})

    tools = {r["tool"] for r in sm.get_session_actions("t-accum")}
    assert {"search_1", "search_2"} <= tools, tools


def test_a_search_row_records_the_method_and_count_but_not_the_documents():
    """Titles and doc_ids are the payload bloat this module exists to avoid."""
    row = g._search_row("baseline_sweep", "flood risk in illinois", "keyword+semantic", 8)
    blob = json.dumps(row)
    assert row["facts"] == {"search_method": "keyword+semantic", "results_returned": 8}
    assert "title" not in blob and "contents" not in blob
    assert row["args"]["query"] == "flood risk in illinois"


def test_every_retrieval_producer_is_in_the_rename_set():
    """Otherwise its `source`/`count` render as imagery provenance again."""
    for tool in ("baseline_sweep", "web_fallback", "related_elements", "element_lookup",
                 "popularity_ranking", "keyword_search", "overpass_search"):
        assert tool in g._LEDGER_SEARCH_TOOLS, tool


# --- what the user is looking at -----------------------------------------------------------
#
# A projection of the ledger rows, not new state: it regroups the map_layer and outputs fields
# _ledger_rows already sets. Worth stating separately because the per-row form was not usable as
# an answer — two mechanisms had to reconstruct exactly this by hand. The map-delivery predicate
# walked rows hunting for a layer, and _refs_in_history regexed the CONVERSATION TEXT to recover
# download links, because nothing carried them forward. And the map is persistent, so "no map was
# produced" is a false statement the answerer previously had no way to check.

DELIVERED = [
    {"tool": "admin_boundary", "args": {"area": "Champaign"},
     "facts": {"feature_count": 48}, "map_layer": "48 tracts"},
    {"tool": "embed_zones", "args": {"model": "gse"}, "facts": {"dims": 64},
     "map_layer": "gse zone groups (k=3)", "outputs": "gse_zone_embeddings.csv"},
]


def test_the_note_says_what_is_still_on_the_map():
    note = g._prior_actions_note(DELIVERED)
    assert "WHAT THE USER IS LOOKING AT" in note
    assert "48 tracts" in note and "gse zone groups (k=3)" in note
    assert "gse_zone_embeddings.csv" in note


def test_it_is_derived_from_the_rows_not_stored_separately():
    """So it costs almost nothing on top of a note that is being sent anyway."""
    lines = g._visible_state_lines(DELIVERED)
    assert len(lines) == 2
    assert len("\n".join(lines)) < 200


def test_nothing_visible_means_no_section():
    assert g._visible_state_lines([{"tool": "keyword_search", "args": {"query": "q"}}]) == []
    note = g._prior_actions_note([{"tool": "keyword_search", "args": {"query": "q"}}])
    assert "WHAT THE USER IS LOOKING AT" not in note


def test_a_failed_call_delivers_nothing_to_look_at():
    """The consistency that matters: §2 and §3 are built from the same rows, so a failed row
    must not claim a layer in one and be excluded from the other."""
    rows = [*DELIVERED,
            {"tool": "regionalize", "args": {"n_regions": 5}, "failed": True,
             "error": "weights required", "map_layer": "phantom", "outputs": "phantom.csv"}]
    note = g._prior_actions_note(rows)
    assert "FAILED regionalize" in note
    assert "phantom" not in note, "a failed call must not claim a layer or an output anywhere"
    assert "48 tracts" in note, "and the real deliveries must survive"


def test_duplicates_are_collapsed():
    rows = [*DELIVERED, dict(DELIVERED[0])]
    lines = g._visible_state_lines(rows)
    assert lines[0].count("48 tracts") == 1


def test_the_auditor_sees_the_visible_state_too():
    """"the layer is on your map" is exactly the claim it used to flag as unsupported."""
    import inspect

    src = inspect.getsource(g.build_supervisor_graph)
    assert "*_visible_state_lines(_rows)" in src, "the auditor's rendering must include it"


def test_a_malformed_row_is_harmless():
    for junk in ([None], ["x"], [{}], [{"map_layer": None}], None):
        assert isinstance(g._visible_state_lines(junk), list)


# --- a tool can map SEVERAL layers in one call -------------------------------------------
#
# embed_zones puts the pixel raster AND the zone-group choropleth on the map; embed_region
# emits one raster per model; align_embedding_colors one per region. All three set the plural
# `map_layers`, with `map_layer` holding only the first. The ledger read the singular, so it
# recorded one layer and lost the rest — the mechanism whose whole purpose is to stop the
# answerer saying "no map was produced" was itself under-reporting the map.

TWO_LAYERS = {
    "ok": True, "model": "gse",
    "map_layer": {"url": "/f/1", "label": "gse pixel embedding in zones"},
    "map_layers": [
        {"url": "/f/1", "label": "gse pixel embedding in zones"},
        {"url": "/f/2", "label": "gse zone groups (k=5)"},
    ],
}


def _row(payload, tool="embed_zones", args=None):
    rows = g._ledger_rows(_pair(tool, args or {"model": "gse"}, payload))
    assert len(rows) == 1, rows
    return rows[0]


def test_both_layers_of_one_call_reach_the_ledger():
    assert _row(TWO_LAYERS)["map_layer"] == [
        "gse pixel embedding in zones", "gse zone groups (k=5)"]


def test_the_first_layer_is_not_recorded_twice():
    """`map_layer` is layers[0] and `map_layers` contains it, so the two keys overlap."""
    labels = _row(TWO_LAYERS)["map_layer"]
    assert len(labels) == len(set(labels))


def test_a_single_layer_row_is_unchanged():
    row = _row({"ok": True, "map_layer": {"url": "/f/1", "label": "hospitals near Chicago"}})
    assert row["map_layer"] == ["hospitals near Chicago"]


def test_the_rendered_line_still_reads_the_same_for_one_layer():
    """The single-layer wording is what the synthesis prompt and the auditor were tuned on."""
    line = g._ledger_lines([_row({"ok": True, "map_layer": {"url": "/f/1", "label": "48 tracts"}})])[0]
    assert "[on the map as '48 tracts']" in line


def test_the_rendered_line_names_every_layer():
    line = g._ledger_lines([_row(TWO_LAYERS)])[0]
    assert "gse pixel embedding in zones" in line and "gse zone groups (k=5)" in line


def test_the_note_lists_every_layer_still_on_the_map():
    note = g._prior_actions_note([_row(TWO_LAYERS)])
    assert "gse pixel embedding in zones" in note
    assert "gse zone groups (k=5)" in note


def test_the_same_layer_from_two_turns_is_listed_once():
    """Per-label dedupe. The old check compared whole rows, so it could not see inside a
    multi-layer one; the per-row ledger lines still mention each layer once per call, which is
    why this asserts on the still-on-the-map projection rather than the whole note."""
    lines = g._visible_state_lines([_row(TWO_LAYERS), _row(TWO_LAYERS)])
    onmap = [l for l in lines if "still on the user's map" in l]
    assert len(onmap) == 1
    assert onmap[0].count("gse zone groups (k=5)") == 1
    assert onmap[0].count("gse pixel embedding in zones") == 1


def test_a_multi_layer_row_still_counts_as_a_delivered_layer():
    assert g._map_layer_was_delivered(None, [_row(TWO_LAYERS)])


def test_a_flood_of_layers_is_capped_and_says_how_many_it_dropped():
    """One tool mapping forty zones separately must not push the ledger out of budget."""
    many = {"ok": True, "map_layers": [{"url": f"/f/{i}", "label": f"zone {i}"} for i in range(40)]}
    labels = _row(many)["map_layer"]
    assert len(labels) == g._LEDGER_LAYERS_PER_ROW + 1
    assert labels[-1] == f"+{40 - g._LEDGER_LAYERS_PER_ROW} more"


def test_a_row_carrying_a_bare_string_still_renders():
    """Rows are process-local and outlive no restart, but the one path that tells the model
    what is on screen should not crash on an unexpected shape."""
    legacy = {"tool": "admin_boundary", "args": {}, "map_layer": "48 tracts"}
    assert "[on the map as '48 tracts']" in g._ledger_lines([legacy])[0]
    assert "48 tracts" in g._prior_actions_note([legacy])
