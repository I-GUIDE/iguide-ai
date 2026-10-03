"""What the grounding auditor is shown.

Two defects, both making a CORRECT answer look invented:

1. Silent truncation. `_render_execution_record` marked its cut; `prior_actions` and the
   environment lines were blind slices, and the prompt never mentioned truncation at all. The
   auditor's rule for "absent" is "you searched and found NO span" — so a claim supported at
   char 9000 of an 8000-char cap came back as an invented specific at high severity.

2. The turn under audit was described worst. `_prior_actions` is documented as excluding the
   current turn, so earlier turns arrived as `tool(args) -> facts` / `FAILED tool(args) -> DID
   NOT RUN`, while this turn arrived only as a JSON dump of the peers' results. A tool's
   arguments were quotable for every turn except the one being judged.
"""

from __future__ import annotations

from agent_runtime.evidence_quality import (_AUDIT_PROMPT, _elided,
                                            _format_execution_context)


# --- 1. no cut is silent -------------------------------------------------------------------

def test_a_short_record_is_untouched():
    assert _elided("abcdef", 10) == "abcdef"


def test_a_cut_record_says_it_was_cut():
    out = _elided("abcdefghijklmno", 6)
    assert out.startswith("abcdef")
    assert "[9 chars elided]" in out


def test_the_earlier_turn_ledger_is_no_longer_sliced_blind():
    ctx = {"prior_actions": ["x" * 20000]}
    rendered = _format_execution_context(ctx)
    assert "chars elided" in rendered, "an unmarked cut reads as a claim that never occurred"


def test_the_environment_lines_are_marked_too():
    ctx = {"environment": ["the map is interactive " * 5000]}
    assert "chars elided" in _format_execution_context(ctx)


def test_a_bare_string_context_is_marked():
    assert "chars elided" in _format_execution_context("y" * 9000, max_chars=100)


def test_the_prompt_says_an_elision_is_not_an_absence():
    """The marker is useless if the auditor reads it as "nothing there". It has to be told,
    and told what to do instead — downgrade, not flag."""
    assert "ELIDED RECORDS" in _AUDIT_PROMPT
    assert "chars elided" in _AUDIT_PROMPT, "the exact marker, so it is recognisable"
    assert "TRUNCATED" in _AUDIT_PROMPT


# --- 2. this turn is described like every other turn ---------------------------------------

def test_this_turn_reaches_the_auditor():
    ctx = {"this_turn": ["admin_boundary(area=Savoy, state=Illinois) -> 1 feature, GEOID 1767860"]}
    rendered = _format_execution_context(ctx)
    assert "THIS TURN" in rendered
    assert "state=Illinois" in rendered, "an argument grounds a claim about what was asked for"
    assert "GEOID 1767860" in rendered


def test_it_comes_before_everything_that_might_be_elided():
    """Ordering is the point: the turn under audit must not be the section that gets cut."""
    ctx = {"analysis_results": {"a": 1}, "this_turn": ["tool(x=1) -> ok"],
           "prior_actions": ["older"]}
    rendered = _format_execution_context(ctx)
    assert rendered.index("THIS TURN") < rendered.index("analysis_results")
    assert rendered.index("THIS TURN") < rendered.index("EARLIER TURNS")


def test_a_failed_call_is_marked_as_having_produced_nothing():
    ctx = {"this_turn": ["FAILED web_fetch(url=https://example.gov/x) -> DID NOT RUN: HTTP 403"]}
    rendered = _format_execution_context(ctx)
    assert "FAILED" in rendered
    assert "produced NOTHING" in rendered, (
        "the heading has to say what FAILED means, or the auditor reads the line as a result")


def test_an_empty_turn_adds_no_section():
    """A turn with no tool calls must not gain a heading promising grounding it does not have."""
    rendered = _format_execution_context({"this_turn": [], "analysis_results": {"a": 1}})
    assert "THIS TURN" not in rendered


def test_the_supervisor_builds_the_rows_with_the_shared_renderer():
    """Not a second hand-rolled rendering: the comment on _ledger_lines says its consumers must
    agree, and until now they agreed about every turn except the current one."""
    import inspect

    from agent_runtime.supervisor import graph

    src = inspect.getsource(graph.build_supervisor_graph) if hasattr(
        graph, "build_supervisor_graph") else inspect.getsource(graph)
    assert '"this_turn": _ledger_lines(_turn_rows)' in src
    assert "_turn_rows = [*_ledger_rows(ar, cr)" in src
