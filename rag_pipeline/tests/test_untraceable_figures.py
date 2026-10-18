"""An answer whose figures no result holds is not shipped as if it were computed.

Live 2026-10-17, production with validation off (T16 Waikiki): five of six code runs failed and
the one that succeeded printed only an unreachable-block count, yet the answer reported nine
figures, all wrong and none in any result. The guard is deterministic and always on.
"""
from __future__ import annotations

import json

from agent_runtime import facts as turn_facts
from agent_runtime.supervisor.graph import _code_run_summary, _guard_untraceable_figures

QUERY = ("How many residents and housing units are in the evacuation zone, and how many hotels? "
         "At 1.1 m/s, what is the median walking time, and how many residents need more than "
         "10 minutes?")


def _run(stdout: str = "", *, ok: bool = True, timed_out: bool = False, error: str = "") -> dict:
    payload = {"ok": ok and not timed_out, "exit_code": None if timed_out else (0 if ok else 1),
               "stdout": stdout, "timed_out": timed_out, "error": error}
    return {"name": "execute_code", "content": json.dumps(payload)}


FAILED_RUNS = [
    _run(ok=False, error="Traceback ...\nIndexError: index 185 is out of bounds"),
    _run("blocks whose snapped node not reachable from outside: 3\ngraph connected components: 67"),
    _run(timed_out=True, error="execution timed out"),
    _run(ok=False, error="IndexError: index -1 is out of bounds for axis 0 with size 0"),
    _run(timed_out=True, error="execution timed out"),
]

FABRICATED = (
    "Residents in the zone: 17,205. Housing units in the zone: 10,352. Hotels in the zone: 24.\n"
    "Population-weighted median walking time: 7.9 minutes. Longest walking time: 18.6 minutes. "
    "Residents needing more than 10 minutes to exit: 2,341 people.\n"
    "Buildings intersecting the SLR exposure area: 1,847. Buildings with their centroid inside: "
    "1,512. Hotels inside the SLR exposure area: 18.")


def _facts(results):
    return turn_facts.build(results=results, query=QUERY)


def test_a_fabricated_answer_is_replaced_by_a_plain_account():
    answer, info = _guard_untraceable_figures(FABRICATED, _facts(FAILED_RUNS), FAILED_RUNS)
    assert info is not None and info["untraceable"] >= 8
    for invented in ("17,205", "10,352", "7.9 minutes", "2,341", "1,847", "1,512"):
        assert invented not in answer
    assert "could not compute this reliably" in answer
    assert "4 of the 5 code runs failed" in answer and "2 timed out" in answer
    assert "IndexError" in answer


def test_a_computed_answer_is_untouched():
    runs = [_run("zone pop: 20294 HU: 17067 hotels: 65\nmedian walk 3.91 min, max 14.38 min, "
                 "over 10 min: 846 residents\nbuildings intersect 909, centroid 324, hotels 10")]
    good = ("Residents in the zone: 20,294; housing units 17,067; 65 hotels. The median walk is "
            "3.91 minutes, the longest 14.38 minutes, and 846 residents need more than 10 "
            "minutes. 909 buildings intersect the area, 324 have their centroid in it.")
    answer, info = _guard_untraceable_figures(good, _facts(runs), runs)
    assert info is None and answer == good


def test_mostly_invented_keeps_what_was_computed():
    runs = [_run("zone pop: 20294\nhotels in zone: 65")]
    mixed = ("Residents in the zone: 20,294.\nHotels in the zone: 65.\n"
             "Median walking time: 7.9 minutes.\nLongest walk: 18.6 minutes.\n"
             "Residents over 10 minutes: 2,341.\nBuildings intersecting: 1,847.\n"
             "Centroids inside: 1,512.")
    answer, info = _guard_untraceable_figures(mixed, _facts(runs), runs)
    assert info is not None
    assert "20,294" in answer and "65" in answer, "computed figures stay"
    assert "7.9 minutes" not in answer and "1,847" not in answer


def test_an_answer_with_few_figures_is_left_alone():
    two = "The zone holds about 17,205 residents and 24 hotels."
    answer, info = _guard_untraceable_figures(two, _facts([]), [])
    assert info is None and answer == two


def test_figures_from_the_question_count_as_traced():
    echo = ("At 1.1 m/s, residents needing more than 10 minutes were counted; the median walk is "
            "3.91 minutes and the longest 14.38 minutes, for 846 residents.")
    runs = [_run("median 3.91\nmax 14.38\nover 10 min 846")]
    answer, info = _guard_untraceable_figures(echo, _facts(runs), runs)
    assert info is None


def test_code_run_summary_counts_successes_and_names_failures():
    ok, failures = _code_run_summary(FAILED_RUNS)
    assert ok == 1
    assert failures.count("timed out") == 2 and failures.count("IndexError") == 2


# Production, 2026-10-18 (zone rates, step 2): a follow-up restated the table the previous turn
# computed and ran no tool. Its figures were in the conversation, not in this turn's results,
# and the guard replaced a correct answer. The earlier answer counts as a source.
ZONES_STEP1 = ("Zone Z2 has the highest incident rate: 22.89 per 1,000 residents (19 incidents, "
               "population 830).\nZ2 830 19 22.89\nZ5 7,400 66 8.92\nZ1 5,200 41 7.88\n"
               "Z4 2,600 18 6.92\nZ6 3,900 25 6.41\nZ8 6,100 37 6.07\nZ7 1,500 9 6.00\n"
               "Z3 4,100 22 5.37\nZ9 2,800 14 5.00\n251 of the 256 incidents fell inside a zone.")
ZONES_STEP2 = ("Z5 has the most incidents, 66, but with 7,400 residents its rate is 8.92 per "
               "1,000. Z2 has only 19 incidents among 830 residents, a rate of 22.89. By count "
               "the order is Z5 (66), Z1 (41), Z8 (37), Z6 (25), Z3 (22), Z2 (19), Z4 (18), "
               "Z9 (14), Z7 (9); by rate Z2 (22.89), Z5 (8.92), Z1 (7.88), Z4 (6.92), Z6 (6.41), "
               "Z8 (6.07), Z7 (6.00), Z3 (5.37), Z9 (5.00).")


def test_a_follow_up_restating_the_previous_answer_is_untouched():
    from agent_runtime.supervisor.graph import _raw_history_text

    history = [{"role": "user", "content": "Compute each zone's incident rate per 1,000 residents."},
               {"role": "assistant", "content": ZONES_STEP1}]
    facts = turn_facts.build(results=[], query="Which zone has the most incidents?",
                             history=_raw_history_text(history))
    answer, info = _guard_untraceable_figures(ZONES_STEP2, facts, [])
    assert info is None and answer == ZONES_STEP2


def test_without_the_history_the_same_follow_up_would_be_cut():
    facts = turn_facts.build(results=[], query="Which zone has the most incidents?")
    _answer, info = _guard_untraceable_figures(ZONES_STEP2, facts, [])
    assert info is not None


def test_history_does_not_launder_a_fabricated_answer():
    history = [{"role": "user", "content": "Tell me about Waikiki."},
               {"role": "assistant", "content": "Waikiki is on Oahu's south shore."}]
    from agent_runtime.supervisor.graph import _raw_history_text
    facts = turn_facts.build(results=FAILED_RUNS, query=QUERY, history=_raw_history_text(history))
    _answer, info = _guard_untraceable_figures(FABRICATED, facts, FAILED_RUNS)
    assert info is not None and info["untraceable"] >= 8
