"""Tests for `admin_boundary` — a named US area to a boundary file, with no upload.

The Census service is stubbed at the HTTP seam (`_query`), so the suite neither needs the
network nor breaks when TIGERweb is slow. The live shape these stubs imitate was captured
from the real service.
"""

from __future__ import annotations

import json

import pytest

from agent_runtime import admin_boundary_tools as ab


def _feature(props, coords=None):
    return {"type": "Feature", "properties": props,
            "geometry": {"type": "Polygon",
                         "coordinates": coords or [[[-88.4, 39.9], [-87.9, 39.9],
                                                    [-87.9, 40.4], [-88.4, 40.4], [-88.4, 39.9]]]}}


COUNTY = _feature({"GEOID": "17019", "NAME": "Champaign County",
                   "BASENAME": "Champaign", "STATE": "17"})
CITY = _feature({"GEOID": "1712385", "NAME": "Champaign city",
                 "BASENAME": "Champaign", "STATE": "17"})


@pytest.fixture
def tool(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setattr(ab, "_states_cache",
                        [{"fips": "17", "name": "Illinois", "usps": "IL"},
                         {"fips": "06", "name": "California", "usps": "CA"}])
    return {t.name: t for t in ab.make_admin_boundary_tools()}["admin_boundary"]


def call(tool, **kw):
    return json.loads(tool.func(**kw))


# --- state qualifier --------------------------------------------------------

def test_state_accepts_name_code_or_fips(monkeypatch):
    monkeypatch.setattr(ab, "_states_cache",
                        [{"fips": "17", "name": "Illinois", "usps": "IL"}])
    assert ab.resolve_state("IL")[0] == "17"
    assert ab.resolve_state("illinois")[0] == "17"
    assert ab.resolve_state("17")[0] == "17"
    assert ab.resolve_state("7")[0] == "07"          # zero-padded to a real FIPS
    fips, near = ab.resolve_state("Atlantis")
    assert fips is None and near == []


def test_an_unknown_state_is_refused_before_any_lookup(tool, monkeypatch):
    monkeypatch.setattr(ab, "_query", lambda *a, **k: pytest.fail("should not query"))
    out = call(tool, area="Champaign", state="Atlantis")
    assert out["ok"] is False and "unknown state" in out["error"]


# --- the happy paths --------------------------------------------------------

def test_county_by_name_becomes_a_file_a_layer_and_a_zone_field(tool, monkeypatch):
    monkeypatch.setattr(ab, "_query", lambda *a, **k: {"features": [COUNTY]})
    out = call(tool, area="Champaign", state="IL")
    assert out["ok"] and out["matched"] == [
        {"geoid": "17019", "name": "Champaign County", "state_fips": "17"}]
    assert out["file_id"] and out["zone_id_field"] == "GEOID"
    assert out["bbox"] == [-88.4, 39.9, -87.9, 40.4]
    # A boundary frames what is analysed inside it; a filled polygon would hide the raster
    # embed_zones drapes underneath.
    assert out["map_layer"]["outline"] is True
    assert out["map_layer"]["url"] == out["download_url"]
    assert "embed_zones" in out["next_step"]


def test_it_matches_the_bare_name_not_the_suffixed_one(tool, monkeypatch):
    """NAME is 'Champaign County' / 'Champaign city'; matching it would lose every county a
    user names without saying 'County'."""
    seen = {}

    def fake(layer, where, *a, **k):
        seen["where"] = where
        return {"features": [COUNTY]}

    monkeypatch.setattr(ab, "_query", fake)
    call(tool, area="Champaign", state="IL")
    assert "BASENAME" in seen["where"] and "NAME=" not in seen["where"].replace("BASENAME=", "")


def test_city_level_reaches_a_layer_earth_engine_does_not_have(tool, monkeypatch):
    monkeypatch.setattr(ab, "_query", lambda layer, *a, **k: {"features": [CITY]})
    out = call(tool, area="Champaign", state="IL", level="city")
    assert out["ok"] and out["level"] == "city"
    assert out["matched"][0]["geoid"] == "1712385"


def test_an_unincorporated_place_falls_back_to_a_cdp(tool, monkeypatch):
    """Incorporated vs census-designated is a Census detail, not the user's problem."""
    calls = []

    def fake(layer, *a, **k):
        calls.append(layer)
        return {"features": [] if "MapServer/4" in layer else [CITY]}

    monkeypatch.setattr(ab, "_query", fake)
    out = call(tool, area="Champaign", state="IL", level="city")
    assert out["ok"] and out["level"] == "cdp"


# --- ambiguity --------------------------------------------------------------

def test_several_places_of_one_name_are_refused_with_the_candidates(tool, monkeypatch):
    """Verified live: 16 incorporated places are named Springfield, and the first is in none
    of the states anyone means. Picking one silently is the failure that matters here."""
    many = [_feature({"GEOID": f"{i:07d}", "NAME": f"Springfield {i}",
                      "BASENAME": "Springfield", "STATE": f"{i:02d}"}) for i in range(1, 6)]
    monkeypatch.setattr(ab, "_query", lambda *a, **k: {"features": many})
    out = call(tool, area="Springfield", level="city")
    assert out["ok"] is False
    assert "say which state" in out["error"]
    assert len(out["candidates"]) == 5
    assert "state=" in out["hint"]


def test_a_name_with_no_match_suggests_what_does(tool, monkeypatch):
    def fake(layer, where, *a, **k):
        return {"features": [] if "LIKE" not in where else [COUNTY]}

    monkeypatch.setattr(ab, "_query", fake)
    out = call(tool, area="Champaig", state="IL")
    assert out["ok"] is False and out["did_you_mean"] == ["Champaign County"]


# --- subdivision, the form embed_zones wants --------------------------------

def test_subdivide_returns_the_tracts_inside_the_county(tool, monkeypatch):
    tracts = [_feature({"GEOID": f"170190001{i:02d}", "NAME": str(i),
                        "STATE": "17", "COUNTY": "019"}) for i in range(12)]
    seen = []

    def fake(layer, where, *a, **k):
        seen.append((layer, where))
        return {"features": tracts if "Tracts" in layer else [COUNTY]}

    monkeypatch.setattr(ab, "_query", fake)
    out = call(tool, area="Champaign", state="IL", subdivide="tracts")
    assert out["ok"] and out["feature_count"] == 12
    assert out["zone_id_field"] == "GEOID"
    assert out["geoids"][0] == "17019000100"
    # scoped by the resolved county's own FIPS, not by re-matching the name
    assert "STATE='17'" in seen[-1][1] and "COUNTY='019'" in seen[-1][1]
    # The default tile budget covers a few per cent of a county; say so while it can still
    # be changed, rather than only afterwards in embed_zones' `truncated`.
    assert "max_tiles" in out["coverage_hint"]


def test_subdividing_a_city_is_refused_because_tracts_do_not_nest_in_one(tool, monkeypatch):
    monkeypatch.setattr(ab, "_query", lambda layer, *a, **k: {"features": [CITY]})
    out = call(tool, area="Champaign", state="IL", level="city", subdivide="tracts")
    assert out["ok"] is False and "county or a state" in out["error"]


def test_an_unknown_subdivision_names_the_ones_that_work(tool, monkeypatch):
    monkeypatch.setattr(ab, "_query", lambda *a, **k: {"features": [COUNTY]})
    out = call(tool, area="Champaign", state="IL", subdivide="parishes")
    assert out["ok"] is False and "block_groups" in out["hint"]


# --- failure modes ----------------------------------------------------------

def test_a_name_is_escaped_into_the_query(tool, monkeypatch):
    """The area name arrives from the model and lands in a SQL-ish where clause."""
    seen = []

    def fake(layer, where, *a, **k):
        seen.append(where)
        return {"features": []}

    monkeypatch.setattr(ab, "_query", fake)
    call(tool, area="O'Brien", state="IL")
    assert "UPPER(BASENAME)='O''BRIEN'" in seen[0]        # the exact-match query
    assert all("''BRIEN" in w for w in seen)              # and the did-you-mean fallback


def test_an_unreachable_service_says_what_to_do_instead(tool, monkeypatch):
    monkeypatch.setattr(ab, "_query", lambda *a, **k: {
        "error": "could not reach the Census TIGERweb service", "hint": "attach a file"})
    out = call(tool, area="Champaign", state="IL")
    assert out["ok"] is False and "TIGERweb" in out["error"]


def test_an_empty_area_is_refused(tool):
    assert call(tool, area="  ")["ok"] is False


def test_an_unknown_level_names_the_known_ones(tool):
    out = call(tool, area="Champaign", level="continent")
    assert out["ok"] is False and "county" in out["hint"]


def test_admin_boundary_is_registered_outside_the_attached_files_gate():
    """It was accidentally nested INSIDE `if input_file_ids:` in the analyse peer, so with
    nothing attached the tool did not exist — and "show me the boundary of Urbana city limits"
    fell through to geocode_places + embed_region (a rectangle) on every model tried. Needing
    no upload is the entire point, so this pins the indentation, not just the presence."""
    import inspect

    from agent_runtime.supervisor import graph

    for fn in (graph.default_analyze_fn, graph.default_code_fn):
        src = inspect.getsource(fn)
        assert "make_admin_boundary_tools" in src, fn.__name__
        for line in src.splitlines():
            if "tools.extend(make_admin_boundary_tools())" in line:
                indent = len(line) - len(line.lstrip())
                # 12+ spaces means it sits inside a nested `if` — the gate we must avoid.
                assert indent <= 12, f"{fn.__name__}: registered at indent {indent}"
                break
        # …and it must appear BEFORE the attached-files gate. Compare CODE lines only:
        # the explanatory comment above the registration quotes the gate verbatim.
        lines = src.splitlines()
        gate = next((i for i, l in enumerate(lines)
                     if l.strip().startswith("if input_file_ids:")), None)
        reg = next(i for i, l in enumerate(lines)
                   if "tools.extend(make_admin_boundary_tools())" in l)
        if gate is not None:
            assert reg < gate, f"{fn.__name__}: registered after the input_file_ids gate"


# --- people say "Champaign County", TIGER stores "Champaign" -------------------------------
#
# BASENAME is the bare name, so a literal match on the full English name found nothing AND the
# LIKE fallback (which also searches BASENAME) found nothing, so the caller got a dead end with
# no candidates to try. Watched live: gpt-oss:120b spent three tool calls guessing
# "Champaign County"/Illinois -> "Champaign County"/IL -> "Champaign"/IL before it landed.

def test_the_level_suffix_is_stripped_so_the_full_english_name_matches():
    from agent_runtime.admin_boundary_tools import _name_variants

    assert _name_variants("Champaign County", "county") == ["Champaign County", "Champaign"]
    assert _name_variants("Orleans Parish", "county") == ["Orleans Parish", "Orleans"]
    assert _name_variants("St. Louis city", "city") == ["St. Louis city", "St. Louis"]


def test_the_name_as_given_is_always_tried_first():
    """A place genuinely named with the suffix must not be broken by the trim."""
    from agent_runtime.admin_boundary_tools import _name_variants

    assert _name_variants("Champaign", "county") == ["Champaign"]
    # 'Township of Washington' does not END with a level suffix, so nothing is trimmed
    assert _name_variants("Township of Washington", "city") == ["Township of Washington"]


def test_a_suffix_inside_the_name_is_not_trimmed():
    from agent_runtime.admin_boundary_tools import _name_variants

    assert _name_variants("County Line", "county") == ["County Line"]


def test_both_variants_reach_the_query():
    """Pin that the variants are actually used in the WHERE clause, not just computed."""
    import inspect

    from agent_runtime import admin_boundary_tools as abt

    src = inspect.getsource(abt.make_admin_boundary_tools)
    assert "_name_variants(area_text, lvl)" in src
    # both the exact match and the LIKE fallback must iterate the variants
    assert src.count("for v in variants") >= 2
