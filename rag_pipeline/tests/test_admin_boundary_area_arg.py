"""`area` holds the place NAME, but it reads like the KIND of place.

Measured live with gpt-oss:120b, asked for the DEM of Urbana:

    admin_boundary({'state':'Illinois','level':'city','name':'Urbana','area':'city'})  failed
    admin_boundary({'name':'Urbana','area':'city','state':'Illinois','level':'city'})  failed
    admin_boundary({'area':'city','state':'Illinois'})                                  failed
    admin_boundary({'area':'Urbana','state':'Illinois','level':'city'})                 worked

Four calls, and the model had the right answer in `name` from the very first one — `name` is the
output FILENAME stem, so it had the two slots exactly inverted. `level` already carries the kind
of place, which makes an `area` holding a level word unambiguous rather than merely wrong: there
is one sensible reading and the tool should take it.

Same class as opengeodata_search's session_context_json — a parameter whose name invites the
wrong value. The fix is the tool's, not the model's.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime.admin_boundary_tools import make_admin_boundary_tools  # noqa: E402


@pytest.fixture
def boundary():
    return {t.name: t for t in make_admin_boundary_tools()}["admin_boundary"]


def call(boundary, **kwargs):
    return json.loads(boundary.func(**kwargs))


# --- recovery --------------------------------------------------------------------

def test_the_inverted_call_is_understood(boundary, monkeypatch):
    """The exact first call from the trace. It should not need three more."""
    seen = {}

    def fake_query(*args, **kwargs):
        seen["args"] = (args, kwargs)
        raise RuntimeError("stop here — the argument reading is what is under test")

    monkeypatch.setattr("agent_runtime.admin_boundary_tools._tigerweb_query", fake_query,
                        raising=False)
    out = call(boundary, state="Illinois", level="city", name="Urbana", area="city")
    # Either it got far enough to query (arguments read correctly), or it failed later for an
    # unrelated reason — what must NOT happen is the "no incorporated place named 'city'" dead
    # end, which means it took the level as the name.
    assert "named 'city'" not in json.dumps(out)


def test_an_unrecoverable_swap_says_exactly_what_to_do(boundary):
    """area='city' with no name to fall back on. Three of the four live calls looked like this,
    and the old error ('no incorporated place named city') described the symptom, not the fix."""
    out = call(boundary, area="city", state="Illinois")
    assert out["ok"] is False
    assert "place NAME" in out["error"] and "city" in out["error"]
    assert "area='Urbana'" in out["hint"]           # shows the shape of a correct call
    assert "level" in out["hint"] and "filename" in out["hint"]


@pytest.mark.parametrize("word", ["city", "County", "STATE", "cdp", "town", "tracts"])
def test_every_level_word_is_caught_not_just_city(boundary, word):
    out = call(boundary, area=word, state="Illinois")
    assert out["ok"] is False and "place NAME" in out["error"]


# --- and does not break the ordinary call ----------------------------------------

def test_a_real_place_name_is_left_alone(boundary):
    """The fix must not touch a correct call. 'Urbana' is not a level word."""
    out = call(boundary, area="Urbana", state="Illinois", level="city")
    # Whatever the network does, it must not be refused for looking like a level.
    assert "place NAME" not in json.dumps(out)


def test_a_place_actually_named_like_a_level_still_needs_care(boundary):
    """There are real places called 'Town' and 'State' — but `level` disambiguates, and an
    explicit name is what the caller gets to use. Documents the known limit rather than
    pretending it does not exist."""
    out = call(boundary, area="town", state="Illinois")
    assert out["ok"] is False          # refused, with instructions, not silently mis-resolved
    assert "hint" in out
