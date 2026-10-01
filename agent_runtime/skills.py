"""Filesystem-backed agent skill discovery and loader tools.

Skills are lightweight instruction bundles stored as directories with a
``SKILL.md`` file.  The runtime exposes only skill metadata up front and
loads the full skill body or referenced resources only when an agent calls
the loader tool.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from agent_runtime.tool_args import accept_null_defaults


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SKILL_ROOTS = (
    REPO_ROOT / "skills",
    REPO_ROOT / ".agents" / "skills",
)
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
DEFAULT_MAX_RESOURCE_BYTES = 64 * 1024
MAX_RESOURCE_LIST_ITEMS = 200


class SkillError(ValueError):
    """Raised for invalid skill definitions or unsafe resource access."""


@dataclass(frozen=True)
class SkillMetadata:
    """Validated metadata for a discovered skill."""

    name: str
    description: str
    skill_path: Path
    skill_file: Path
    source_root: Path
    allowed_tools: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    frontmatter: Mapping[str, Any] | None = None

    def catalog_entry(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "path": str(self.skill_path),
            "allowed_tools": list(self.allowed_tools),
            "tags": list(self.tags),
        }


def _split_paths(raw: str) -> List[str]:
    if "," in raw:
        parts = raw.split(",")
    else:
        parts = raw.split(os.pathsep)
    return [part.strip() for part in parts if part.strip()]


def default_skill_roots(skill_roots: Optional[Sequence[str | Path]] = None) -> List[Path]:
    """Return configured skill roots.

    Explicit ``skill_roots`` take precedence.  Otherwise the runtime scans
    the project defaults and appends any paths from ``AGENT_SKILL_PATHS`` or
    ``AGENT_SKILLS_PATHS``.  Non-existent roots are ignored during discovery.
    """
    if skill_roots is not None:
        return [Path(root).expanduser() for root in skill_roots]

    roots = [Path(root) for root in DEFAULT_SKILL_ROOTS]
    env_value = os.getenv("AGENT_SKILL_PATHS") or os.getenv("AGENT_SKILLS_PATHS") or ""
    roots.extend(Path(item).expanduser() for item in _split_paths(env_value))
    return roots


def skills_enabled() -> bool:
    raw = str(os.getenv("AGENT_SKILLS_ENABLED", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if not text:
        return ""
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in inner.split(",")]
    return text


def _parse_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_items = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [value]
    return tuple(str(item).strip() for item in raw_items if str(item).strip())


def parse_frontmatter(text: str, *, skill_file: Path) -> tuple[Dict[str, Any], str]:
    """Parse simple YAML-style front matter from a ``SKILL.md`` file."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillError(f"skill file is missing YAML front matter: {skill_file}")

    closing_index: Optional[int] = None
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing_index = idx
            break
    if closing_index is None:
        raise SkillError(f"skill file front matter is not closed: {skill_file}")

    frontmatter: Dict[str, Any] = {}
    front_lines = lines[1:closing_index]
    idx = 0
    while idx < len(front_lines):
        raw_line = front_lines[idx]
        line = raw_line.strip()
        if not line or line.startswith("#"):
            idx += 1
            continue
        if ":" not in line:
            raise SkillError(f"invalid front matter line in {skill_file}: {raw_line!r}")
        key, raw_value = line.split(":", 1)
        key = key.strip()
        if raw_value.strip():
            frontmatter[key] = _parse_scalar(raw_value)
            idx += 1
            continue

        values: List[Any] = []
        idx += 1
        while idx < len(front_lines):
            nested_raw = front_lines[idx]
            nested = nested_raw.strip()
            if not nested or nested.startswith("#"):
                idx += 1
                continue
            if nested.startswith("- "):
                values.append(_parse_scalar(nested[2:]))
                idx += 1
                continue
            break
        frontmatter[key] = values if values else ""

    body = "\n".join(lines[closing_index + 1 :]).strip()
    return frontmatter, body


def _find_skill_file(skill_dir: Path) -> Optional[Path]:
    if not skill_dir.exists() or not skill_dir.is_dir():
        return None
    matches = [
        candidate
        for candidate in skill_dir.iterdir()
        if candidate.is_file() and candidate.name.lower() == "skill.md"
    ]
    if len(matches) > 1:
        raise SkillError(f"skill directory has multiple SKILL.md files: {skill_dir}")
    return matches[0] if matches else None


def _candidate_skill_files(root: Path) -> Iterable[Path]:
    root = root.expanduser()
    if not root.exists():
        return []
    if root.is_file() and root.name.lower() == "skill.md":
        return [root]
    if root.is_dir():
        direct_skill = _find_skill_file(root)
        if direct_skill:
            return [direct_skill]
        skill_files: List[Path] = []
        for child in sorted(root.iterdir(), key=lambda item: item.name):
            if child.is_dir():
                skill_file = _find_skill_file(child)
                if skill_file:
                    skill_files.append(skill_file)
        return skill_files
    return []


def _load_skill_metadata(skill_file: Path, source_root: Path) -> SkillMetadata:
    text = skill_file.read_text(encoding="utf-8")
    frontmatter, _body = parse_frontmatter(text, skill_file=skill_file)
    name = str(frontmatter.get("name") or "").strip()
    description = str(frontmatter.get("description") or "").strip()
    if not name:
        raise SkillError(f"skill file is missing required 'name': {skill_file}")
    if not SKILL_NAME_PATTERN.match(name):
        raise SkillError(
            f"invalid skill name {name!r} in {skill_file}; use lowercase letters, numbers, and hyphens"
        )
    if not description:
        raise SkillError(f"skill file is missing required 'description': {skill_file}")

    allowed_tools = _parse_list(frontmatter.get("allowed-tools") or frontmatter.get("allowed_tools"))
    tags = _parse_list(frontmatter.get("tags"))
    return SkillMetadata(
        name=name,
        description=description,
        skill_path=skill_file.parent.resolve(),
        skill_file=skill_file.resolve(),
        source_root=source_root.resolve(),
        allowed_tools=allowed_tools,
        tags=tags,
        frontmatter=frontmatter,
    )


def _resource_size_limit() -> int:
    raw = os.getenv("AGENT_SKILL_MAX_RESOURCE_BYTES", str(DEFAULT_MAX_RESOURCE_BYTES))
    try:
        return max(1024, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_MAX_RESOURCE_BYTES


def _safe_relative_resource_path(resource_path: str) -> Path:
    raw = str(resource_path or "").strip()
    if not raw:
        raise SkillError("resource_path is required")
    path = Path(raw)
    if path.is_absolute():
        raise SkillError("resource_path must be relative to the skill directory")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise SkillError("resource_path cannot contain empty, current, or parent path segments")
    if any(part.startswith(".") for part in path.parts):
        raise SkillError("resource_path cannot read hidden files or directories")
    return path


def _read_text_resource(path: Path) -> str:
    limit = _resource_size_limit()
    size = path.stat().st_size
    if size > limit:
        raise SkillError(f"resource is too large to load ({size} bytes; limit {limit} bytes)")
    return path.read_text(encoding="utf-8")


class SkillRegistry:
    """Discover and load local skill bundles."""

    def __init__(self, skills: Mapping[str, SkillMetadata], errors: Optional[List[Dict[str, str]]] = None) -> None:
        self._skills = dict(skills)
        self._errors = list(errors or [])

    @classmethod
    def discover(cls, skill_roots: Optional[Sequence[str | Path]] = None) -> "SkillRegistry":
        if not skills_enabled():
            return cls({})

        discovered: Dict[str, SkillMetadata] = {}
        errors: List[Dict[str, str]] = []
        for root in default_skill_roots(skill_roots):
            expanded = root.expanduser()
            try:
                candidate_files = list(_candidate_skill_files(expanded))
            except Exception as exc:
                errors.append({"path": str(expanded), "error": str(exc)})
                continue

            for skill_file in candidate_files:
                try:
                    metadata = _load_skill_metadata(skill_file, expanded)
                except Exception as exc:
                    errors.append({"path": str(skill_file), "error": str(exc)})
                    continue
                # Later roots or later files with the same name intentionally win.
                discovered[metadata.name] = metadata
        return cls(discovered, errors)

    @property
    def errors(self) -> List[Dict[str, str]]:
        return list(self._errors)

    def __bool__(self) -> bool:
        return bool(self._skills)

    def list(self) -> List[SkillMetadata]:
        return [self._skills[name] for name in sorted(self._skills)]

    def catalog(self) -> List[Dict[str, Any]]:
        return [skill.catalog_entry() for skill in self.list()]

    def get(self, name: str) -> SkillMetadata:
        normalized = str(name or "").strip()
        if normalized not in self._skills:
            available = ", ".join(sorted(self._skills)) or "none"
            raise SkillError(f"unknown skill {normalized!r}; available skills: {available}")
        return self._skills[normalized]

    def list_resources(self, skill: SkillMetadata) -> List[Dict[str, Any]]:
        resources: List[Dict[str, Any]] = []
        for path in sorted(skill.skill_path.rglob("*"), key=lambda item: item.as_posix()):
            if len(resources) >= MAX_RESOURCE_LIST_ITEMS:
                break
            if not path.is_file() or path.resolve() == skill.skill_file:
                continue
            rel = path.relative_to(skill.skill_path).as_posix()
            if any(part.startswith(".") for part in Path(rel).parts):
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            resources.append(
                {
                    "path": rel,
                    "size_bytes": size,
                    "loadable": size <= _resource_size_limit(),
                }
            )
        return resources

    def load_skill(self, name: str) -> Dict[str, Any]:
        skill = self.get(name)
        text = skill.skill_file.read_text(encoding="utf-8")
        frontmatter, body = parse_frontmatter(text, skill_file=skill.skill_file)
        return {
            "status": "ok",
            "skill": skill.catalog_entry(),
            "frontmatter": frontmatter,
            "instructions": body,
            "resources": self.list_resources(skill),
            "usage": (
                "Use these instructions as task-specific workflow guidance, adapted to the tools "
                "YOU actually have. Only call tools from the skill's allowed_tools that appear in "
                "your own tool list. If the skill references a tool you do NOT have, do not import "
                "its name as a Python module and do not claim you cannot proceed — instead "
                "reconstruct that step from the knowledge base: call agent_kb_search / get_kb_block "
                "to read the relevant function or notebook source and reuse it verbatim inside "
                "execute_code. To inspect a listed resource, call load_skill again with resource_path."
            ),
        }

    def load_resource(self, name: str, resource_path: str) -> Dict[str, Any]:
        skill = self.get(name)
        rel_path = _safe_relative_resource_path(resource_path)
        resource = (skill.skill_path / rel_path).resolve()
        if skill.skill_path.resolve() not in resource.parents:
            raise SkillError("resource_path escapes the skill directory")
        if not resource.exists() or not resource.is_file():
            raise SkillError(f"resource does not exist: {resource_path}")
        text = _read_text_resource(resource)
        return {
            "status": "ok",
            "skill": skill.catalog_entry(),
            "resource_path": rel_path.as_posix(),
            "content": text,
            "size_bytes": resource.stat().st_size,
        }

    def list_available_skills_json(self) -> str:
        return json.dumps({"skills": self.catalog(), "errors": self.errors}, ensure_ascii=True)

    def load_skill_json(self, skill_name: str, resource_path: str = "") -> str:
        try:
            payload = (
                self.load_resource(skill_name, resource_path)
                if str(resource_path or "").strip()
                else self.load_skill(skill_name)
            )
        except Exception as exc:
            payload = {
                "status": "error",
                "error": str(exc),
                "skill_name": skill_name,
                "resource_path": resource_path,
            }
        return json.dumps(payload, ensure_ascii=True, default=str)


def skill_catalog_for_prompt(registry: SkillRegistry, *, max_chars: int = 3000) -> str:
    """Return a compact skill catalog suitable for a tool description."""
    entries = []
    for skill in registry.list():
        allowed = f"; allowed tools: {', '.join(skill.allowed_tools)}" if skill.allowed_tools else ""
        tags = f"; tags: {', '.join(skill.tags)}" if skill.tags else ""
        entries.append(f"- {skill.name}: {skill.description}{allowed}{tags}")
    catalog = "\n".join(entries)
    if len(catalog) > max_chars:
        return catalog[: max_chars - 80].rstrip() + "\n- ... skill catalog truncated"
    return catalog


def make_skill_tools(*, skill_roots: Optional[Sequence[str | Path]] = None) -> List[Any]:
    """Build LangChain tools for discovering and loading configured skills."""
    registry = SkillRegistry.discover(skill_roots)
    if not registry:
        return []

    try:
        from langchain_core.tools import StructuredTool
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "LangChain is not installed. Add `langchain-core` (or langchain) to use skill tools."
        ) from exc

    catalog = skill_catalog_for_prompt(registry)
    loaded_skills: set[str] = set()
    loaded_resources: set[tuple[str, str]] = set()

    def list_available_skills() -> str:
        """List local skills available to this agent run."""
        return registry.list_available_skills_json()

    def load_skill(skill_name: str, resource_path: str = "") -> str:
        """Load a skill's instructions or one resource inside that skill."""
        normalized_name = str(skill_name or "").strip()
        normalized_resource = str(resource_path or "").strip()

        # Some models pass the skill directory as resource_path after the main
        # skill is already loaded. Treat that as a repeat main-skill load rather
        # than an unsafe resource lookup.
        if normalized_resource:
            try:
                skill = registry.get(normalized_name)
                candidate = Path(normalized_resource).expanduser()
                if candidate.is_absolute() and candidate.resolve() in {
                    skill.skill_path.resolve(),
                    skill.skill_file.resolve(),
                }:
                    normalized_resource = ""
            except Exception:
                pass

        if not normalized_resource:
            if normalized_name in loaded_skills:
                try:
                    skill = registry.get(normalized_name)
                    allowed_tools = list(skill.allowed_tools)
                except Exception:
                    allowed_tools = []
                return json.dumps(
                    {
                        "status": "already_loaded",
                        "skill_name": normalized_name,
                        "allowed_tools": allowed_tools,
                        "next_action": (
                            "Do not call load_skill for this skill again in this request. "
                            "Use the loaded instructions and call the relevant allowed tool, "
                            "or produce the final answer if the required tool output is already available."
                        ),
                    },
                    ensure_ascii=True,
                )
            payload = json.loads(registry.load_skill_json(skill_name=normalized_name))
            if isinstance(payload, dict):
                if payload.get("status") == "ok":
                    loaded_skills.add(normalized_name)
                payload["next_action"] = (
                    "This skill is now loaded. Do not call load_skill for this skill again in this request. "
                    "Proceed to the relevant allowed tool or final answer."
                )
                payload["already_loaded"] = False
                return json.dumps(payload, ensure_ascii=True, default=str)
            return json.dumps(payload, ensure_ascii=True, default=str)

        resource_key = (normalized_name, normalized_resource)
        if resource_key in loaded_resources:
            return json.dumps(
                {
                    "status": "already_loaded",
                    "skill_name": normalized_name,
                    "resource_path": normalized_resource,
                    "next_action": (
                        "Do not call load_skill for this resource again in this request. "
                        "Use the already loaded resource content."
                    ),
                },
                ensure_ascii=True,
            )
        loaded_resources.add(resource_key)
        return registry.load_skill_json(skill_name=normalized_name, resource_path=normalized_resource)

    return [
        StructuredTool.from_function(func=accept_null_defaults(list_available_skills),
            name="list_available_skills",
            description=(
                "List local agent skills available for this run. "
                "Use this when you need to inspect the skill catalog."
            ),
            metadata={"category": "skill"},
        ),
        StructuredTool.from_function(func=accept_null_defaults(load_skill),
            name="load_skill",
            description=(
                "Load instructions for a relevant local agent skill by skill_name — ONLY when a "
                "listed skill's description clearly matches THIS request. Do NOT load an unrelated "
                "or place/topic-specific skill that doesn't fit the task (e.g. a Chicago-crime "
                "skill for a non-crime or non-Chicago request); if none fit, don't call this. "
                "You may also load a listed skill resource by passing resource_path. "
                "Call this at most once per skill per user request; after it returns status ok "
                "or already_loaded, use the loaded instructions and do not call load_skill for "
                "the same skill again. "
                "Available skills:\n"
                f"{catalog}"
            ),
            metadata={"category": "skill"},
        ),
    ]


__all__ = [
    "SkillError",
    "SkillMetadata",
    "SkillRegistry",
    "default_skill_roots",
    "make_skill_tools",
    "parse_frontmatter",
    "skill_catalog_for_prompt",
]
