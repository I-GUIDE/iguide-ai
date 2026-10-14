"""The notes an archived run's code runs were shown, recovered from the commit the run ran on.

A lineage replay (agent_runtime/lineage.py, stage 56) cites, for each call, the notes it was
shown. A live turn records them as it runs. An archived turn does not hold them: `bound_tool_docs`
keeps 200 characters of each tool description, and the code bridge's host notes (PR #104,
`public_data_tools.BRIDGE_HOST_NOTES`, appended to execute_code's description) are not in it at
all. The notes are code, so the commit a run names in its `.origin` determines them. This reads
them from that commit, without importing it: the module's constant assignments are evaluated in
order in an empty namespace, which is enough for string and tuple constants.

    python3 -m gis_harness.run_notes [ARCHIVE]   # writes <run>/tool_notes.json for bridge runs

`tool_notes.json` is `{tool: {note_id: text}}`, the shape `lineage.register_tool_notes` and
`lineage.from_results(notes=...)` take. Only runs whose `.origin` says AGENT_CODE_BRIDGE=1 get one:
without the bridge the notes were not in the description.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

_MODULE = "agent_runtime/public_data_tools.py"
_NAMES = ("S2_L2A_VALUES", "DEFAULT_HOSTS", "BRIDGE_EXTRA_HOSTS", "BRIDGE_HOST_NOTES")


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def _show(sha: str, path: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(_repo()), "show", f"{sha}:{path}"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def bridge_notes(source: str) -> Dict[str, str]:
    """`BRIDGE_HOST_NOTES` as the module at that commit defines it, or {} when it has none."""
    ns: Dict[str, object] = {}
    for node in ast.parse(source).body:
        targets = ([t.id for t in node.targets if isinstance(t, ast.Name)]
                   if isinstance(node, ast.Assign) else
                   [node.target.id] if isinstance(node, ast.AnnAssign)
                   and isinstance(node.target, ast.Name) and node.value is not None else [])
        if not any(t in _NAMES for t in targets):
            continue
        try:
            value = eval(compile(ast.Expression(node.value), "<notes>", "eval"),  # noqa: S307
                         {"__builtins__": {}}, ns)
        except Exception:  # noqa: BLE001 - a constant this cannot evaluate is left out
            continue
        for t in targets:
            ns[t] = value
    notes = ns.get("BRIDGE_HOST_NOTES")
    return {str(h): str(v) for h, v in notes.items()} if isinstance(notes, dict) else {}


def notes_for_run(run: Path) -> Optional[Dict[str, Dict[str, str]]]:
    origin = run / ".origin"
    try:
        text = origin.read_text()
    except OSError:
        return None
    if "AGENT_CODE_BRIDGE=1" not in text:
        return None
    for sha in re.findall(r"\b[0-9a-f]{8,40}\b", text):
        source = _show(sha, _MODULE)
        notes = bridge_notes(source) if source else {}
        if notes:
            return {"execute_code": {f"bridge:{h}": f"{h}: {v}" for h, v in notes.items()},
                    "_origin": {"commit": sha}}
    return None


def main(argv: Optional[List[str]] = None) -> int:
    from gis_harness.validators_bench import default_archive

    args = argv if argv is not None else sys.argv[1:]
    root = Path(args[0]) if args else default_archive()
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        notes = notes_for_run(run)
        if notes is None:
            continue
        (run / "tool_notes.json").write_text(json.dumps(notes, indent=1, ensure_ascii=False))
        print(f"{run.name}: {len(notes['execute_code'])} notes from {notes['_origin']['commit']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
