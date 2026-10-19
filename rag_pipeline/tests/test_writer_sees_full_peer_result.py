"""The answer writer sees a peer's whole summary, not a 2,000-char prefix of its JSON dump.

Live, 2026-10-18, the Chicago SKATER step (6 regions naming all 77 community areas) ended
"Region 6 — 2 areas, pop 23,5[truncated]" or "Region 6 ... cut off in the analysis output" in 4 of
6 runs. The analysis peer's summary was complete; default_synthesize_fn sent the writer
`json.dumps(analysis_results)[:2000]`, which spent the budget on JSON quoting and the tool calls,
and the writer reported the cut it was given.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime.supervisor import graph  # noqa: E402

AREAS = ["Rogers Park", "West Ridge", "Uptown", "Lincoln Square", "North Center", "Lake View",
         "Lincoln Park", "Near North Side", "Edison Park", "Norwood Park", "Jefferson Park",
         "Forest Glen", "North Park", "Albany Park", "Portage Park", "Irving Park", "Dunning",
         "Montclare", "Belmont Cragin", "Hermosa", "Avondale", "Logan Square", "Humboldt Park",
         "West Town", "Austin", "Near West Side", "South Lawndale", "Lower West Side", "Loop",
         "Near South Side", "Armour Square", "Garfield Ridge", "Archer Heights", "Brighton Park"]


def _skater_summary() -> str:
    lines = ["## SKATER Regionalization — 6 Contiguous Regions by Violent Crime Rate",
             "I grouped the 77 community areas into 6 contiguous regions using SKATER (queen "
             "contiguity). Between-region share of variance: 79.1%."]
    for i, (n, pop, rate) in enumerate([(44, "1,902,011", 19.4), (13, "421,330", 44.2),
                                        (10, "212,904", 71.1), (5, "98,214", 60.3),
                                        (3, "66,555", 85.5), (2, "23,512", 109.1)], start=1):
        lines.append(f"### Region {i} — {n} areas, pop {pop}, mean rate {rate}/1,000")
        lines.append(", ".join(AREAS))
    return "\n".join(lines)


def _analysis_results(summary: str) -> dict:
    big = {"type": "FeatureCollection",
           "features": [{"geometry": {"coordinates": [[[-87.6 + i * 1e-4, 41.8]] * 50]}}
                        for i in range(40)]}
    return {"summary": summary,
            "tool_calls": [{"name": "execute_code", "args": {"code": "x = 1\n" * 300}}],
            # Tool content arrives as a JSON STRING, the shape the compaction parses.
            "tool_results": [{"name": "execute_code", "content": json.dumps(big)}],
            "executed": True}


def _writer_prompt(analysis_results=None, code_result=None) -> str:
    seen = []
    fn = graph.default_synthesize_fn(llm=lambda p: (seen.append(p), "ok")[1])
    fn("Group the 77 community areas into 6 regions with SKATER.", [], analysis_results,
       code_result)
    return seen[-1]


def test_a_long_summary_reaches_the_writer_whole():
    summary = _skater_summary()
    assert len(summary) > 2000          # the old cap would have cut it
    sent = _writer_prompt(_analysis_results(summary))
    assert "Region 6 — 2 areas, pop 23,512, mean rate 109.1/1,000" in sent
    assert "Analysis results:" in sent
    # The summary comes first, before the record that used to eat the budget.
    assert sent.index("Region 6") < sent.index("Execution record:")
    assert graph._WRITER_SHORTENED_NOTE not in sent


def test_an_oversized_summary_is_cut_at_a_line_and_says_not_to_mention_it(monkeypatch):
    monkeypatch.setattr(graph, "_WRITER_SUMMARY_MAX_CHARS", 900)
    sent = _writer_prompt(_analysis_results(_skater_summary()))
    summary_part = sent.split("Summary (the peer's own words):\n", 1)[1].split(
        "\n\nExecution record:", 1)[0]
    body, note = summary_part.rsplit("\n", 1)
    assert note == graph._WRITER_SHORTENED_NOTE
    # Cut at a line boundary: the last kept line is a whole line of the original.
    assert body.splitlines()[-1] in _skater_summary().splitlines()
    assert "Do not mention this shortening" in note


def test_the_record_is_compacted_like_the_auditors_and_bounded(monkeypatch):
    monkeypatch.setattr(graph, "_WRITER_RECORD_MAX_CHARS", 3000)
    sent = _writer_prompt(_analysis_results("Short summary."))
    record = sent.split("Execution record:\n", 1)[1]
    assert len(record) <= 3000 + len(graph._WRITER_SHORTENED_NOTE) + 400   # + execution note
    assert "Short summary." in sent


def test_the_code_answer_is_sent_whole_and_alone():
    answer = _skater_summary()
    sent = _writer_prompt(code_result={"answer": answer, "code": "print(1)\n" * 500,
                                       "executed": True})
    assert "Region 6 — 2 areas, pop 23,512" in sent
    assert "Code result:" in sent
    assert "print(1)" not in sent       # the code branch still shows the answer, not the dict


def test_a_code_result_without_an_answer_falls_back_to_the_record():
    sent = _writer_prompt(code_result={"stdout": "rows: 77", "executed": True})
    assert "Code result:" in sent and "rows: 77" in sent
