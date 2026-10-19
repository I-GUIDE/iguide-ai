"""Stage 87: a value the user states reaches the peer brief, and so the stated-parameter rule,
however long the request.

The brief (supervisor/graph.py `_turn_brief`) carried the request cut at 1,500 characters and
called it verbatim. GeoAnalystBench task 6 states its kernel bandwidth at character 1,567 of
2,011 and lost it from the brief; the rule reads the brief (langchain_exec_tools._task_text),
so a distance stated past the cut was never checked.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime import code_watch, turn_log  # noqa: E402
from agent_runtime.langchain_exec_tools import _task_text  # noqa: E402
from agent_runtime.supervisor import graph as g  # noqa: E402

# A long request whose stated value comes after the first 1,500 characters, as in the
# benchmark prompts: the file listing and the background come first, the parameters last.
FILLER = "Background: " + "Fire stations in Etobicoke serve the surrounding wards. " * 30
LONG = FILLER + "\nParameters fixed by this task: buffer each station by 2500 meters.\n"


def _rule_text(query: str) -> str:
    """What the stated-parameter rule reads during a code run of this turn."""
    with turn_log.bind(turn_log.new_log(), "code", g._turn_brief({"query": query})):
        return _task_text()


def _flags(code: str, brief: str) -> bool:
    return any(f["rule"] == "stated_parameter" for f in code_watch.precheck(code, brief=brief))


def test_a_short_request_is_carried_whole_and_unchanged():
    q = "buffer the schools by 500 m and count homes inside"
    brief = g._turn_brief({"query": q})
    assert f"Task (the user's request, verbatim): {q}" in brief
    assert "after the cut" not in brief


def test_the_long_request_used_to_lose_its_stated_value_at_the_cut():
    assert len(FILLER) > g._BRIEF_REQUEST_CHARS
    assert code_watch.stated_parameters(LONG[:g._BRIEF_REQUEST_CHARS]) == []


def test_a_value_stated_after_the_cut_is_quoted_in_the_brief():
    brief = g._turn_brief({"query": LONG})
    assert f"first {g._BRIEF_REQUEST_CHARS:,} of {len(LONG):,} characters" in brief
    assert "Values the request states after the cut, verbatim:" in brief
    assert "buffer each station by 2500 meters" in brief


def test_the_rule_checks_a_value_stated_after_the_cut():
    text = _rule_text(LONG)
    assert [w for _, _, w in code_watch.stated_parameters(text)] == ["2500 meters"]
    assert _flags("cover = stations.buffer(1000)\n", text)
    assert not _flags("cover = stations.buffer(2500)\n", text)


def test_a_value_inside_the_kept_part_is_not_quoted_again():
    q = "Buffer each station by 2500 meters. " + FILLER
    brief = g._turn_brief({"query": q})
    assert "after the cut" not in brief
    assert _flags("cover = stations.buffer(1000)\n", _rule_text(q))


def test_figures_with_no_parameter_word_are_not_carried():
    q = FILLER + "\nThe ward is 12 km long and the report runs 30 minutes.\n"
    assert "after the cut" not in g._turn_brief({"query": q})
