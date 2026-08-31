"""A generated SKILL.md has to be followable, not merely accurate.

Side by side, the extraction-generated skill and the hand-written one for the same notebook:

    generated     "(cell 3) tools: — — # AI Agent for Chicago Crime Analysis This notebook walks
                  you through building a 'Code Agent' capable of performing geo"
                  allowed-tools: []
                  Run: "reuse the functions extracted from this element"

    hand-written  "Call agent_kb_search(...); note the cited element_id. Call get_kb_block(...)
                  to read the FULL source of load_chicago_crime_data, load_chicago_community_
                  areas, filter_dataframe_by_value, spatial_join_and_count."
                  allowed-tools: agent_kb_search, get_kb_block, execute_code

The difference was raw material, not effort. Two things exist now that did not when the emitter
was written: `_block_title` gives every cell the author's own name for it, and the method library
gives every promoted unit a signature and a version-pinned import line. So the skill can name its
steps and name the code that performs them.
"""

from __future__ import annotations

import pytest

from extractors.emitters import skill_emitter

SKILL = {
    "name": "a-skill",
    "description": "Do a thing.",
    "allowed_tools": [],
    "tags": ["geospatial"],
    "ordered_steps": [
        {"order": 3, "title": "1c. Project All Files to the Same CRS", "tools": ["geopandas"],
         "summary": "## 1c. Project All Files to the Same CRS\nBefore measuring distance we"},
        {"order": 9, "title": "cell 9", "tools": [], "summary": ""},
    ],
    "methods": [
        {"symbol": "e2sfca", "signature": "def e2sfca(catchments, distances, weights)",
         "doc_summary": "Calculates Enhanced Two-Step Floating Catchment Area (E2SFCA).",
         "requirements": ["geopandas", "pandas"],
         # Recorded per PARAMETER, so a three-argument method repeats each check three times.
         "invariants": ["projected_crs", "reject_all_nan", "projected_crs", "reject_all_nan",
                        "projected_crs", "reject_all_nan"],
         "import_line": "from iguide_methods.ke_x.v_abc import e2sfca"},
    ],
}


def test_a_step_leads_with_its_name_not_with_truncated_prose():
    md = skill_emitter._render(SKILL)
    assert "**1c. Project All Files to the Same CRS**" in md
    assert "tools: — —" not in md, "the old table-of-contents rendering is back"


def test_the_positional_fallback_survives_for_an_unnamed_cell():
    """`cell 9` is what a cell nobody named is called, and it still beats an empty line."""
    assert "**cell 9**" in skill_emitter._render(SKILL)


def test_the_methods_are_named_with_their_import_lines():
    """"Reuse the functions extracted from this element" is true and unfollowable: it does not
    say which functions, or how to import them, both of which extraction knows."""
    md = skill_emitter._render(SKILL)
    assert "def e2sfca(catchments, distances, weights)" in md
    assert "from iguide_methods.ke_x.v_abc import e2sfca" in md
    assert "requires: geopandas, pandas" in md


def test_invariants_are_a_set_not_a_multiplicity():
    """Invariants are recorded per parameter, so a three-argument method rendered
    "projected_crs, reject_all_nan" three times. The reader needs the rules, not their count."""
    md = skill_emitter._render(SKILL)
    line = next(l for l in md.splitlines() if "enforced:" in l)
    assert line.count("projected_crs") == 1, line
    assert line.count("reject_all_nan") == 1, line


def test_a_skill_that_grants_nothing_cannot_be_followed():
    """Empty `allowed-tools` was right when the only candidate was a fictional per-workflow
    executor. It is wrong now that the Run section names three real tools."""
    front = skill_emitter._render(SKILL).split("---")[1]
    for tool in ("kb_method_search", "get_method_contract", "execute_code"):
        assert tool in front, front


def test_the_run_section_names_the_staging_precondition():
    """A loader advertises `load_x(staged_path)` and nothing said where a staged_path comes
    from. That instruction belongs in the procedure, not only in the evidence view."""
    md = skill_emitter._render(SKILL)
    assert "staged_path" in md and "stage_element" in md


def test_an_element_with_no_callable_unit_says_so_rather_than_naming_nothing():
    """Most elements promote nothing. Telling that reader to import something would send them
    looking for a module that does not exist."""
    barren = {**SKILL, "methods": []}
    md = skill_emitter._render(barren)
    assert "promoted no independently callable unit" in md
    assert "agent_kb_search" in md and "get_kb_block" in md
    assert "iguide_methods" not in md


def test_the_import_line_is_resolved_at_emit_time_not_in_the_extractor():
    """`library_emitter` assigns `library_module` by mutating the live unit dict, but
    `manifest.skill` is an `asdict` COPY taken when the extractor returned — so a module path
    written afterwards never reaches it. The enrichment has to read the assets."""
    import inspect

    src = inspect.getsource(skill_emitter.emit)
    assert 'd.get("assets")' in src
    assert "library_module" in src


# ------------------------------------- a skill the registry cannot read is not a skill

def test_a_long_description_does_not_fold_the_front_matter(tmp_path):
    """`yaml.safe_dump` wraps a long scalar across lines and
    `agent_runtime.skills.parse_frontmatter` is, by its own docstring, a "simple YAML-style"
    line parser — the continuation carries no `key:`, so it rejects the WHOLE skill.

    The flood skill was generated, written to disk, and silently absent from the registry.
    """
    from agent_runtime.skills import parse_frontmatter

    long_desc = ("Run the Probabilistic Flood Inundation Mapping using Physics-Aware Spatial "
                 "AI workflow extracted from "
                 "05269a1a__flood_mapping_using_remotesensing_deeplearning.ipynb.")
    md = skill_emitter._render({**SKILL, "description": long_desc})
    front = md.split("---")[1]
    assert all(":" in line for line in front.strip().splitlines()
               if line.strip() and not line.strip().startswith("-")), front
    parsed, _ = parse_frontmatter(md, skill_file=tmp_path / "SKILL.md")
    assert parsed["description"] == long_desc


def test_an_undiscoverable_skill_is_reported_as_an_error(tmp_path, monkeypatch):
    """`_roundtrip_ok` already caught this and returned `discoverable: False`; ingest recorded
    it in a warning string beside a successful write and carried on. A check whose result
    nothing acts on is not a check."""
    monkeypatch.setattr(skill_emitter, "_roundtrip_ok", lambda root, name: False)
    out = skill_emitter.emit({"skill": SKILL}, skills_root=str(tmp_path))
    assert out["discoverable"] is False
    assert "NOT discoverable" in out.get("error", ""), out


def test_a_discoverable_skill_reports_no_error(tmp_path, monkeypatch):
    monkeypatch.setattr(skill_emitter, "_roundtrip_ok", lambda root, name: True)
    out = skill_emitter.emit({"skill": SKILL}, skills_root=str(tmp_path))
    assert out["discoverable"] is True
    assert "error" not in out
