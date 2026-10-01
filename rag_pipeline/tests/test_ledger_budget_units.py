"""The ledger budget must be measured in the unit its consumer pays.

`_LEDGER_MAX_CHARS`'s own comment calls it "a hard ceiling on the rendered ledger", but
`_budgeted` sized every row by `json.dumps`. The rendered form goes through the phrase book,
where one fact can expand 3.7x, so the rendered ledger ran far past the ceiling — by the one
mechanism that exists BECAUSE a turn overflowed the context window.

The two consumers genuinely differ: the router receives the raw rows and pays for their JSON;
the answering model and the auditor receive the rendered lines.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime.supervisor import graph as g


def _fat_rows(n=25):
    """Rows carrying only facts that predate any of this work, so the breach is not caused by
    a recently added one."""
    return [{"tool": f"tool_{i}",
             "args": {"file_id": f"file_{i:012x}", "column": "median_income"},
             "facts": {"image_size": 224, "scale_m": 10, "patch_size": 16,
                       "input_size_hw": "(224, 224)", "pixel_ground_m": 7.87,
                       "zone_id_field": "GEOID", "tiles_planned": 240,
                       "level": "county", "crs": "EPSG:4326", "dim": 64}}
            for i in range(n)]


def test_the_rendered_ledger_respects_the_ceiling_it_documents():
    rendered = "\n".join(g._ledger_lines(_fat_rows()))
    assert len(rendered) <= g._LEDGER_MAX_CHARS, (
        f"{len(rendered)} chars against a documented ceiling of {g._LEDGER_MAX_CHARS}")


def test_json_sizing_would_have_let_it_through():
    """Guards the premise: the same rows pass a JSON-sized budget, so the test above is
    measuring the fix rather than a fixture that was never over."""
    kept_by_json = g._budgeted(_fat_rows())
    rendered = "\n".join(g._ledger_line(r) for r in kept_by_json)
    assert len(rendered) > g._LEDGER_MAX_CHARS


def test_the_row_consumer_still_pays_in_json():
    """prior_turns_in_this_conversation receives the rows themselves, where JSON is the right
    unit — the fix must not impose the rendered unit on it."""
    kept = g._budgeted(_fat_rows())
    assert sum(g._json_size(r) for r in kept) <= g._LEDGER_MAX_CHARS


def test_a_ledger_under_budget_is_untouched():
    rows = _fat_rows(3)
    assert len(g._ledger_lines(rows)) == 3


def test_the_rendered_line_is_unchanged_by_the_refactor():
    """_ledger_line was extracted from _ledger_lines; the wording the prompts were tuned on
    must be byte-identical."""
    row = {"tool": "add_map_layer", "args": {"column": "income"},
           "facts": {"feature_count": 801}, "outputs": "income.geojson",
           "file_id": "file_abc", "map_layer": ["Income by tract"]}
    line = g._ledger_line(row)
    assert line == ("- add_map_layer (column=income) -> features: 801 "
                    "[produced income.geojson, file_id file_abc] "
                    "[on the map as 'Income by tract']")
    assert g._ledger_lines([row]) == [line]


def test_the_newest_rows_are_the_ones_kept():
    """Oldest-drops-first must survive the unit change."""
    rows = _fat_rows(25)
    lines = g._ledger_lines(rows)
    assert "tool_24" in lines[-1]
    assert "tool_0" not in "\n".join(lines)


def test_the_per_tool_cap_still_applies():
    rows = [{"tool": "keyword_search", "args": {"q": f"query {i}"}, "facts": {"count": i}}
            for i in range(10)]
    assert len(g._ledger_lines(rows)) == g._LEDGER_ROWS_PER_TOOL


# --- the ceiling has to bound the whole note, not half of it -------------------------------
#
# _LEDGER_MAX_CHARS is documented as "a hard ceiling on the rendered ledger" and exists BECAUSE
# a turn overflowed the context window. _ledger_lines honours it, but _prior_actions_note then
# appended the visible-state section from the FULL row list, outside the budget — so the note
# actually shipped to the answerer and the auditor could run far past the ceiling however
# tightly the lines were trimmed.

def _many_rows(n, label_len):
    return [{"tool": f"tool{i}", "args": {"i": i},
             "map_layer": ["L" * label_len + f"-{i}"]} for i in range(n)]


def test_the_whole_note_is_bounded_however_many_layers_there_are():
    from agent_runtime.supervisor import graph as g

    for rows in (_many_rows(60, 60), _many_rows(200, 120), _many_rows(400, 200)):
        note = g._prior_actions_note(rows) or ""
        body = note.split("completed.\n", 1)[-1]
        assert len(body) <= g._LEDGER_MAX_CHARS, (
            f"{len(body)} chars of ledger+visible against a {g._LEDGER_MAX_CHARS} ceiling")


def test_the_unbudgeted_concatenation_really_did_overflow():
    """Guards the premise: without the fix this is an order of magnitude over."""
    from agent_runtime.supervisor import graph as g

    rows = _many_rows(400, 200)
    unbudgeted = "\n".join(g._ledger_lines(rows)) + g._visible_state_note(rows)
    assert len(unbudgeted) > g._LEDGER_MAX_CHARS * 5


def test_the_visible_state_section_is_trimmed_not_dropped():
    """Telling the answerer what is on the user's screen is what it was added for."""
    from agent_runtime.supervisor import graph as g

    note = g._prior_actions_note(_many_rows(400, 200)) or ""
    assert "still on the user's map" in note
