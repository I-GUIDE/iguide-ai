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
