"""The ruler for method ranking, and the properties that keep it honest.

``scripts/eval_method_retrieval.py`` exists because two ranking changes to
``agent_runtime.method_library`` were already judged by throwaway scripts and quoted from memory
("R@1 5 of 8" appears in this project's notes with no harness behind it), and one of them had to be
reverted after the number turned out to have moved the wrong way.

These tests do not assert a score — the score depends on a library that is not in the repo, and
``conftest`` deliberately points the library at an empty directory. They assert that the harness
cannot flatter a change: that a miss is a miss, that an empty library reports itself instead of
scoring zero, and that the accept-any-of cases are the ones that genuinely have several right
answers.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "eval_method_retrieval.py"


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("eval_method_retrieval", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_empty_library_scores_nothing_and_says_why(harness):
    """conftest points the library at an empty directory, so this is the state the suite runs in.
    A harness that reported 0/12 here would look like a ranking regression; it has to report that
    there is nothing to rank."""
    result = harness.evaluate()
    assert result["library"]["units"] == 0
    assert result["recall_at_1"] == 0
    assert all(row["rank"] is None for row in result["rows"])


#: Cases whose answer CANNOT be reached from its symbol name — only from its docstring.
#: Discovered by asserting the opposite and being wrong. Three are acronym-named
#: (``e2sfca``, ``ebr_of``, ``camels_geology_attrs`` for "permeability and porosity"), and the
#: fourth is a tokenizer gap: the query "de-accumulate" splits into ``de`` + ``accumulate`` while
#: the symbol ``deaccumulate`` stays one token, so they share nothing.
DOCSTRING_ONLY = {
    "compute two-step floating catchment area accessibility",
    "energy balance ratio from uncorrected flux measurements",
    "de-accumulate cumulative ERA5-Land hourly fluxes",
    "area-weighted permeability and porosity catchment attributes",
}


def test_which_cases_depend_entirely_on_the_docstring(harness, monkeypatch):
    """Strip every summary, leave only symbol names, and see what becomes unfindable.

    This is a characterization test, and the fact it characterizes matters: a third of the
    benchmark is reachable ONLY because someone wrote a docstring. 38% of the library's units have
    no docstring at all (measured), and for those the symbol name is the entire index — so a unit
    named ``e2sfca`` with no summary is effectively unreachable by anyone who does not already
    know the acronym. That is a coverage limit of the extraction, not of the ranker, and it is the
    reason a docstring-completeness signal is worth having as *retrieval text* even though it lost
    as a *ranking* signal.

    If this set shrinks, something improved (a summary was added, or the tokenizer learned to
    split ``deaccumulate``). If it grows, a symbol became less findable.
    """
    registry = {}
    for _question, expected in harness.CASES:
        for name in expected:
            registry[f"pkg.{name}"] = {
                "library_symbol": name, "signature": f"def {name}()", "doc_summary": "",
                "module": "iguide_methods.pkg.v_abc", "slice_sha": "abc",
                "provenance": {"element_id": "e1"}}

    from agent_runtime import method_library

    monkeypatch.setattr(method_library, "load_registry", lambda: registry)
    monkeypatch.setattr(method_library, "library_summary",
                        lambda: {"units": len(registry), "elements": 1, "root": "x"})
    result = harness.evaluate(limit=len(registry))
    assert result["cases"] == len(harness.CASES)
    unfindable = {r["question"] for r in result["rows"] if r["rank"] is None}
    assert unfindable == DOCSTRING_ONLY, (
        f"newly unfindable by name: {sorted(unfindable - DOCSTRING_ONLY)}; "
        f"newly findable by name: {sorted(DOCSTRING_ONLY - unfindable)}")


def test_a_library_of_decoys_scores_zero_not_something(harness, monkeypatch):
    """The complement of the test above. A harness that credited near-misses would report progress
    for a library containing none of the answers."""
    registry = {f"pkg.decoy_{i}": {
        "library_symbol": f"decoy_{i}", "signature": "def decoy()",
        "doc_summary": "geospatial data processing helper for rasters and vectors",
        "provenance": {"element_id": "e1"}} for i in range(20)}

    from agent_runtime import method_library

    monkeypatch.setattr(method_library, "load_registry", lambda: registry)
    monkeypatch.setattr(method_library, "library_summary",
                        lambda: {"units": 20, "elements": 1, "root": "x"})
    result = harness.evaluate()
    assert result["recall_at_8"] == 0
    assert result["mrr"] == 0.0


def test_multi_answer_cases_are_only_where_the_library_is_genuinely_ambiguous(harness):
    """Accepting several symbols is how the benchmark avoids measuring an arbitrary tie-break —
    ``spatial_join_and_count`` exists twice under different elements with the same docstring, and
    ``e2sfca``/``ae2sfca`` are the plain and adjusted forms of one method. It must not become a
    way to make a case easier: each alternative set stays small, and each name distinct."""
    for question, expected in harness.CASES:
        assert 1 <= len(expected) <= 2, f"{question!r} accepts {len(expected)} answers"
        assert len(set(expected)) == len(expected), f"{question!r} repeats a symbol"


def test_the_questions_are_natural_language_not_symbol_names(harness):
    """A benchmark of bare symbol names would measure exact-match lookup, which already works, and
    would say nothing about the query a user actually types."""
    for question, expected in harness.CASES:
        assert " " in question, question
        for name in expected:
            assert name not in question, (
                f"{question!r} contains its own answer {name!r} verbatim")


def test_rank_is_recorded_per_case_so_a_regression_is_locatable(harness, monkeypatch):
    """An aggregate that drops by one is useless without knowing which case moved — that is how
    the reverted IDF-floor change went unexplained for a while."""
    from agent_runtime import method_library

    monkeypatch.setattr(method_library, "load_registry", lambda: {
        "pkg.calculate_buffers": {"library_symbol": "calculate_buffers",
                                  "signature": "def calculate_buffers(gdf, buffer)",
                                  "doc_summary": "Replace geometry with buffers.",
                                  "provenance": {"element_id": "e1"}}})
    monkeypatch.setattr(method_library, "library_summary",
                        lambda: {"units": 1, "elements": 1, "root": "x"})
    result = harness.evaluate()
    buffered = [r for r in result["rows"] if "buffer a GeoDataFrame" in r["question"]]
    assert buffered and buffered[0]["rank"] == 1
    assert buffered[0]["returned"] == ["calculate_buffers"]
