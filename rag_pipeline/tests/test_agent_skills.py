from __future__ import annotations

import json

import pytest

from agent_runtime import skills as skills_module
from agent_runtime.skills import (
    SkillError,
    SkillRegistry,
    default_skill_roots,
    make_skill_tools,
    parse_frontmatter,
)
from agent_runtime.tool_policy import select_allowed_tools


def _write_skill(root, name: str, description: str = "Test workflow skill.") -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"""---
name: {name}
description: {description}
allowed-tools: keyword_search, semantic_search
tags: [test, docs]
---

# {name}

Use this skill for test workflows.
""",
        encoding="utf-8",
    )
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "guide.md").write_text("Reference details.", encoding="utf-8")


def test_parse_frontmatter_requires_name_and_description(tmp_path):
    skill_file = tmp_path / "SKILL.md"
    text = """---
name: sample-skill
description: Useful for sample tasks.
---

# Sample
"""
    frontmatter, body = parse_frontmatter(text, skill_file=skill_file)

    assert frontmatter["name"] == "sample-skill"
    assert frontmatter["description"] == "Useful for sample tasks."
    assert body == "# Sample"


def test_parse_frontmatter_supports_block_lists(tmp_path):
    skill_file = tmp_path / "SKILL.md"
    text = """---
name: list-skill
description: Useful for list frontmatter.
allowed-tools:
  - keyword_search
  - spatial_search
tags:
  - reports
  - geospatial
---

# List Skill
"""
    frontmatter, _body = parse_frontmatter(text, skill_file=skill_file)

    assert frontmatter["allowed-tools"] == ["keyword_search", "spatial_search"]
    assert frontmatter["tags"] == ["reports", "geospatial"]


def test_skill_registry_discovers_loads_and_lists_resources(tmp_path):
    _write_skill(tmp_path, "geo-workflow", "Use for geospatial workflows.")

    registry = SkillRegistry.discover([tmp_path])

    assert [skill.name for skill in registry.list()] == ["geo-workflow"]
    loaded = registry.load_skill("geo-workflow")
    assert loaded["status"] == "ok"
    assert loaded["skill"]["description"] == "Use for geospatial workflows."
    assert "Use this skill for test workflows." in loaded["instructions"]
    assert loaded["resources"] == [
        {
            "path": "references/guide.md",
            "size_bytes": len("Reference details."),
            "loadable": True,
        }
    ]

    resource = registry.load_resource("geo-workflow", "references/guide.md")
    assert resource["content"] == "Reference details."


def test_skill_registry_rejects_path_escape(tmp_path):
    _write_skill(tmp_path, "safe-skill")
    registry = SkillRegistry.discover([tmp_path])

    with pytest.raises(SkillError):
        registry.load_resource("safe-skill", "../secret.txt")


def test_skill_registry_reports_invalid_skill_files(tmp_path):
    skill_dir = tmp_path / "bad-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        """---
name: Bad_Skill
description: Invalid because the name is not normalized.
---

# Bad
""",
        encoding="utf-8",
    )

    registry = SkillRegistry.discover([tmp_path])

    assert registry.catalog() == []
    assert registry.errors
    assert "invalid skill name" in registry.errors[0]["error"]


def test_make_skill_tools_returns_loader_tools(tmp_path):
    _write_skill(tmp_path, "analysis-skill")

    tools = make_skill_tools(skill_roots=[tmp_path])
    tool_names = [tool.name for tool in tools]

    assert tool_names == ["list_available_skills", "load_skill"]
    load_tool = next(tool for tool in tools if tool.name == "load_skill")
    payload = json.loads(load_tool.invoke({"skill_name": "analysis-skill"}))
    assert payload["status"] == "ok"
    assert payload["skill"]["name"] == "analysis-skill"
    assert payload["already_loaded"] is False
    assert "Do not call load_skill" in payload["next_action"]


def test_load_skill_is_single_use_per_tool_instance(tmp_path):
    _write_skill(tmp_path, "analysis-skill")

    load_tool = next(tool for tool in make_skill_tools(skill_roots=[tmp_path]) if tool.name == "load_skill")

    first = json.loads(load_tool.invoke({"skill_name": "analysis-skill"}))
    second = json.loads(load_tool.invoke({"skill_name": "analysis-skill"}))

    assert first["status"] == "ok"
    assert "instructions" in first
    assert second == {
        "status": "already_loaded",
        "skill_name": "analysis-skill",
        "allowed_tools": ["keyword_search", "semantic_search"],
        "next_action": (
            "Do not call load_skill for this skill again in this request. "
            "Use the loaded instructions and call the relevant allowed tool, "
            "or produce the final answer if the required tool output is already available."
        ),
    }


def test_load_skill_treats_skill_directory_resource_as_repeat_load(tmp_path):
    _write_skill(tmp_path, "analysis-skill")
    skill_dir = tmp_path / "analysis-skill"

    load_tool = next(tool for tool in make_skill_tools(skill_roots=[tmp_path]) if tool.name == "load_skill")
    first = json.loads(load_tool.invoke({"skill_name": "analysis-skill"}))
    repeated = json.loads(
        load_tool.invoke(
            {
                "skill_name": "analysis-skill",
                "resource_path": str(skill_dir),
            }
        )
    )

    assert first["status"] == "ok"
    assert repeated["status"] == "already_loaded"
    assert repeated["skill_name"] == "analysis-skill"


def test_make_skill_tools_empty_when_no_skills(tmp_path):
    assert make_skill_tools(skill_roots=[tmp_path]) == []


def test_skill_tools_survive_intent_filtering():
    allowed = select_allowed_tools(
        "analysis_task",
        ["keyword_search", "load_skill", "list_available_skills"],
    )

    assert "load_skill" in allowed
    assert "list_available_skills" in allowed


# ------------------------------------------------------------------ the default roots
#
# Every production caller — graph_runtime, tool_policy, the supervisor, capabilities — passes
# ``skill_roots=None`` unless a request supplies ``skillPaths``, so what the agent sees is
# exactly ``DEFAULT_SKILL_ROOTS`` plus whatever ``AGENT_SKILL_PATHS`` adds. The tests above
# all pass explicit roots and never exercise that path. These do, against two roots laid out
# the way the defaults resolve inside the agent-api image, where REPO_ROOT is /app.


def _image_shaped_roots(tmp_path, monkeypatch):
    app = tmp_path / "app"
    roots = (app / "skills", app / ".agents" / "skills")
    monkeypatch.setattr(skills_module, "DEFAULT_SKILL_ROOTS", roots)
    for name in ("AGENT_SKILL_PATHS", "AGENT_SKILLS_PATHS", "AGENT_SKILLS_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    return roots


def test_discovery_with_no_explicit_roots_reads_every_default_root(tmp_path, monkeypatch):
    curated, generated = _image_shaped_roots(tmp_path, monkeypatch)
    _write_skill(curated, "curated-skill")
    _write_skill(generated, "generated-skill")

    registry = SkillRegistry.discover()

    assert registry.errors == []
    found = {skill.name: skill.source_root.resolve() for skill in registry.list()}
    assert found == {"curated-skill": curated.resolve(), "generated-skill": generated.resolve()}


def test_a_missing_default_root_is_skipped_without_an_error(tmp_path, monkeypatch):
    """The behaviour that let the packaging defect hide: a root that is not there is not an
    error, so a registry that degrades to one root (or none) reports nothing. Pinned here so
    the Dockerfile test in test_deployment_contract.py is understood as the only guard."""
    curated, generated = _image_shaped_roots(tmp_path, monkeypatch)
    _write_skill(curated, "curated-skill")
    assert not generated.exists()

    registry = SkillRegistry.discover()

    assert [skill.name for skill in registry.list()] == ["curated-skill"]
    assert registry.errors == []


def test_extra_roots_from_the_environment_are_added_not_substituted(tmp_path, monkeypatch):
    """A deployment that points AGENT_SKILL_PATHS at a writable volume for generated skills
    must keep the curated roots baked into the image."""
    curated, generated = _image_shaped_roots(tmp_path, monkeypatch)
    volume = tmp_path / "agent_chat_files" / "skills"
    _write_skill(curated, "curated-skill")
    _write_skill(generated, "generated-skill")
    _write_skill(volume, "volume-skill")
    monkeypatch.setenv("AGENT_SKILL_PATHS", str(volume))

    assert default_skill_roots() == [curated, generated, volume]
    names = [skill.name for skill in SkillRegistry.discover().list()]
    assert names == ["curated-skill", "generated-skill", "volume-skill"]
