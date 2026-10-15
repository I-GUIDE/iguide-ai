"""gis_harness.clarify: the scorer and the conversation loop, offline (no server, no model)."""
from __future__ import annotations

import json

import pytest

from gis_harness import clarify as C

H3_RES5 = 599551849401417727            # 0x852093affffffff
H3_RES9 = int("892093ac19bffff", 16)    # inside it
S2_LVL6 = 4096868286023598080           # 0x38db000000000000
S2_LEAF = int("38dbcb2ea52155e9", 16)   # inside it


def test_grid_parents_are_computed_as_h3_and_s2_define_them():
    assert C._h3_res(H3_RES5) == 5 and C._h3_parent(H3_RES9, 5) == H3_RES5
    assert C._s2_level(S2_LVL6) == 6 and C._s2_parent(S2_LEAF, 6) == S2_LVL6


@pytest.mark.parametrize("answer, strict, lenient", [
    (f"H3 res 5: {hex(H3_RES5)[2:]}; S2 level 6: {S2_LVL6}", True, True),
    (f"H3 (res 9): {hex(H3_RES9)[2:]}; S2 leaf: 38dbcb2ea52155e9", False, True),
    ("H3: 8a2093ac19a7fff; S2: 1234567890123456789", False, False),
])
def test_h3s2_strict_is_the_gold_cell_lenient_is_a_cell_inside_it(answer, strict, lenient):
    task = {"check": "h3s2", "gold": {"h3": str(H3_RES5), "s2": str(S2_LVL6)}}
    assert C.score_answer(task, answer) == {"correct": strict, "lenient": lenient}


def test_distance_uses_the_benchmark_tolerance_and_the_headline_figure():
    task = {"check": "distance_km", "gold": {"km": 232.0}}
    assert C.score_answer(task, "About 231.5 km in a straight line.")["correct"]
    first_wrong = C.score_answer(task, "By road 310 km; straight line 232.4 km.")
    assert first_wrong == {"correct": False, "lenient": True}
    assert C._km_tolerance(19.0) == pytest.approx(0.95) and C._km_tolerance(10) == 0.5


def test_direction_reads_compound_words_and_the_first_one_counts():
    task = {"check": "direction", "gold": {"direction": "East"}}
    assert C.score_answer(task, "Travel east, slightly south of east.")["correct"]
    assert C.score_answer(task, "Head north-east.") == {"correct": False, "lenient": False}


def test_geohash_strict_and_lenient():
    task = {"check": "geohash", "gold": {"geohash": "d4fvty"}}
    assert C.score_answer(task, "The geohash is d4fvty.")["correct"]
    assert C.score_answer(task, "Either d4fvtz or d4fvty.") == {"correct": False, "lenient": True}


class _Sim:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return type("M", (), {"content": json.dumps(self.replies.pop(0))})()


def test_a_conversation_answers_the_question_then_stops(tmp_path, monkeypatch):
    answers = iter(["Which H3 resolution do you want?", f"At resolution 5: {H3_RES5}."])
    sent = []

    def fake_turn(base, message, **kw):
        sent.append((message, kw.get("thread_id"), kw.get("file_ids")))
        return {"answer": next(answers), "error": None, "seconds": 1.0, "tool_calls": []}

    monkeypatch.setattr(C.client, "run_turn", fake_turn)
    sim = _Sim([{"done": False, "reply": "Resolution 5, please."}, {"done": True}])
    task = {"id": "X", "ambiguous": True, "check": "h3", "gold": {"h3": str(H3_RES5)},
            "query": "Which grid cell contains both A and B?", "user_knows": "resolution 5"}
    rec = C.converse("http://x", task, provider="p", model="m", sim=sim, out=tmp_path)
    assert [m for m, _, _ in sent] == [task["query"], "Resolution 5, please."]
    assert sent[0][1] == sent[1][1], "one thread for the whole conversation"
    assert rec["asked"] == 1 and rec["n_turns"] == 2 and rec["correct"]
    assert rec["needless_ask"] is False and "resolution 5" in sim.prompts[0]


def test_a_question_on_a_specified_task_is_needless_and_turns_are_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(C.client, "run_turn", lambda *a, **k: {
        "answer": "Which one?", "error": None, "seconds": 1.0, "tool_calls": []})
    sim = _Sim([{"done": False, "reply": "No preference."}] * 5)
    task = {"id": "Y", "ambiguous": False, "check": "geohash", "gold": {"geohash": "d4fvty"},
            "query": "Encode it.", "user_knows": ""}
    rec = C.converse("http://x", task, provider="p", model="m", sim=sim, out=tmp_path,
                     max_followups=2)
    assert rec["n_turns"] == 3 and rec["needless_ask"] is True and not rec["correct"]


def test_summary_splits_ambiguous_and_specified():
    recs = [{"ambiguous": True, "correct": True, "lenient": True, "asked": 1, "needless_ask": False,
             "n_turns": 2, "seconds": 10.0},
            {"ambiguous": False, "correct": False, "lenient": True, "asked": 1, "needless_ask": True,
             "n_turns": 2, "seconds": 20.0}]
    s = C.summarise(recs)
    assert s["all"]["correct"] == 1 and s["ambiguous"]["asked"] == 1
    assert s["specified"]["needless_asks"] == 1 and s["all"]["turns_mean"] == 2


def test_an_unanswerable_task_is_right_when_it_refuses_without_a_value():
    refused = {"correct": None, "refusal": {"refused": True, "fabricated": None}}
    invented = {"correct": None, "refusal": {"refused": True, "fabricated": "12.3 degrees"}}
    assert C._catalogue_correct(False, refused) is True
    assert C._catalogue_correct(False, invented) is False
    assert C._catalogue_correct(True, {"correct": True}) is True
