"""SKILL emitter — write the overall pipeline as a discoverable SKILL.md bundle.

From ``manifest.skill`` (a SkillSpec dict): validate the front matter against
agent_runtime/skills.py rules, render ``SKILL.md`` (front matter + ordered-steps
body), and write it to a discovered skills root (default ``REPO_ROOT/.agents/skills``
— one of SkillRegistry's default roots — overridable via ``AGENT_GENERATED_SKILLS_ROOT``).
A malformed skill would land silently in ``SkillRegistry.errors``, so we validate
(and best-effort round-trip through the real registry) before/after writing.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..manifest import UnifiedManifest

REPO_ROOT = Path(__file__).resolve().parents[2]
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def _default_root() -> Path:
    return Path(os.getenv("AGENT_GENERATED_SKILLS_ROOT", str(REPO_ROOT / ".agents" / "skills")))


def _validate(name: str, description: str) -> None:
    if not name or not _NAME_RE.match(name):
        raise ValueError(f"invalid skill name '{name}' (must match ^[a-z0-9][a-z0-9-]{{0,63}}$)")
    if not (description or "").strip():
        raise ValueError("skill description is required")


def _render(skill: Dict[str, Any]) -> str:
    front = {
        "name": skill["name"],
        "description": " ".join(str(skill.get("description") or "").split()),
        # The tools this skill's own Run section tells the reader to call. Empty front matter
        # was correct when the only candidate was a fictional per-workflow executor; it is wrong
        # now that the procedure below names three real tools, because a skill that grants
        # nothing cannot be followed.
        "allowed-tools": (list(skill.get("allowed_tools") or [])
                          or (["kb_method_search", "get_method_contract", "execute_code"]
                              if skill.get("methods") else
                              ["agent_kb_search", "get_kb_block", "execute_code"])),
        "tags": list(skill.get("tags") or []),
    }
    fm = yaml.safe_dump(front, sort_keys=False, default_flow_style=False, allow_unicode=True).strip()

    title = skill["name"].replace("-", " ").title()
    lines: List[str] = [f"---\n{fm}\n---", "", f"# {title}", "",
                        front["description"], "", "## Pipeline steps", ""]
    for i, step in enumerate(skill.get("ordered_steps") or [], 1):
        # The step's NAME leads, because a reader following a procedure needs to know what the
        # step is before what it uses. Previously every line began "(cell 3) tools: — —" and
        # then 140 characters of the notebook's prose, which is a table of contents.
        name = " ".join(str(step.get("title") or step.get("summary") or "").split())[:90]
        tools = ", ".join(step.get("tools") or [])
        suffix = f"  ·  uses {tools}" if tools else ""
        lines.append(f"{i}. **{name}**  *(cell {step.get('order')})*{suffix}")

    methods = [m for m in (skill.get("methods") or []) if m.get("symbol")]
    if methods:
        lines += ["", "## Callable methods extracted from this element", "",
                  "Import these rather than re-implementing them. Each import line is pinned to "
                  "the exact slice this contract describes, and the library is mounted read-only "
                  "inside `execute_code`.", ""]
        for m in methods:
            lines.append(f"- `{m.get('signature') or m['symbol']}`")
            if m.get("doc_summary"):
                lines.append(f"    - {' '.join(str(m['doc_summary']).split())[:150]}")
            if m.get("import_line"):
                lines.append(f"    - `{m['import_line']}`")
            if m.get("requirements"):
                lines.append(f"    - requires: {', '.join(map(str, m['requirements']))}")
            # A precondition the caller has to satisfy or get a plausible wrong number, so it
            # belongs beside the method rather than one tool call away.
            # Deduplicated: invariants are recorded PER PARAMETER, so a three-argument method
            # with a CRS and a NaN check on each rendered "projected_crs, reject_all_nan"
            # three times over. The reader needs the set of rules, not their multiplicity.
            checks = sorted({str(c) for c in (m.get("invariants") or []) if c})
            if checks:
                lines.append(f"    - enforced: {', '.join(checks)}")
    if skill.get("allowed_tools"):
        lines += ["", "## Run", "",
                  f"Invoke `{skill['allowed_tools'][0]}` (the extracted workflow) to reproduce this pipeline."]
    elif skill.get("methods"):
        # Name the tools and the order. "Reuse the functions extracted from this element" was
        # true and unfollowable — it did not say which functions, how to import them, or in
        # what order, all of which extraction knows.
        lines += ["", "## Run", "",
                  "No single tool runs this pipeline; compose it in `execute_code`.", "",
                  "1. `kb_method_search` for the method you need, or import it directly using "
                  "an import line above.",
                  "2. `get_method_contract(<symbol>)` before calling an unfamiliar one — the "
                  "invariants say what it assumes about its inputs, and violating one produces "
                  "a plausible wrong number rather than an error.",
                  "3. If a signature takes `staged_path`, call `stage_element(\"<element id>\")` "
                  "first; the file is not in the sandbox until you do.",
                  "4. Compose the calls in `execute_code` in the step order above."]
    else:
        lines += ["", "## Run", "",
                  "This element promoted no independently callable unit, so there is nothing to "
                  "import. Read the cells with `agent_kb_search` and `get_kb_block`, then adapt "
                  "them in `execute_code`; the steps above give the order."]
    return "\n".join(lines).rstrip() + "\n"


def _roundtrip_ok(root: Path, name: str) -> Optional[bool]:
    """Best-effort: confirm the real SkillRegistry discovers it without error."""
    try:
        from agent_runtime.skills import SkillRegistry
    except Exception:
        return None
    try:
        reg = SkillRegistry.discover([str(root)])
        names = {s.get("name") for s in reg.catalog()} if hasattr(reg, "catalog") else set()
        return name in names
    except Exception:
        return None


def emit(manifest: UnifiedManifest, *, skills_root: Optional[str] = None,
         dry_run: bool = False) -> Dict[str, Any]:
    d = manifest.to_dict() if isinstance(manifest, UnifiedManifest) else dict(manifest)
    skill = d.get("skill")
    if not skill:
        return {"written": None, "reason": "no skill in manifest"}

    _validate(skill.get("name"), skill.get("description"))
    # Import lines are resolved HERE, not in the extractor.
    #
    # `library_emitter` is what assigns each unit its `library_module`, and it does so by
    # mutating the live unit dict during emit — but `manifest.skill` is an `asdict` COPY taken
    # when the extractor returned, so a module path written afterwards never reaches it. The
    # manifest's own assets do carry it, and by the time this emitter runs the library has
    # already been written (see `_fan_out`, where the order is load-bearing for exactly this
    # reason). So the skill's methods are enriched from the assets rather than trusted as copied.
    by_symbol = {}
    for asset in (d.get("assets") or []):
        unit = (asset or {}).get("unit") or {}
        symbol = unit.get("library_symbol")
        module = unit.get("library_module") or unit.get("module")
        if symbol and module:
            by_symbol[symbol] = f"from {module} import {symbol}"
    for method in (skill.get("methods") or []):
        line = by_symbol.get(method.get("symbol"))
        if line:
            method["import_line"] = line

    md = _render(skill)
    if dry_run:
        return {"dry_run": True, "name": skill["name"], "bytes": len(md), "preview": md[:300]}

    root = Path(skills_root) if skills_root else _default_root()
    dest = root / skill["name"] / "SKILL.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(md, encoding="utf-8")
    return {"written": str(dest), "name": skill["name"], "root": str(root),
            "discoverable": _roundtrip_ok(root, skill["name"])}


__all__ = ["emit"]
