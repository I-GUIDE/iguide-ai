"""Screenshot recorded turns as the map UI shows them, and keep the layers that needs.

    python -m gis_harness.screenshots <dir-or-events.jsonl>... [--force] [--parallel N]

Writes <task>[.t<n>].png next to each <task>[.t<n>].events.jsonl, and screenshots.jsonl in each
directory (one line per turn: layers drawn of total, why any were not, seconds, bytes). The page is the real map UI in a replay build: the recorded events go through its own
SSE client, so the picture is what a user would have seen, not a reconstruction of it. The work
is done by `map-ui-prototype/scripts/replay-shots.ts`. This wraps it so the harness (`--screenshots`)
and a backfill of the archive call it the same way.

A turn's map layers are fetched from the agent server by url, and that server is gone by the time
anyone looks. `capture_layers` saves them while it is still up, as
<dir>/layers/<task>[.t<n>]__NN__<label>.<ext>, NN being the layer's place among the turn's
map_layer events. That is the layout `gis_harness_runs_archive/tools/dump_layers.py` gave the
older runs, and the replay serves from it. A layer that was never captured is reported on its
screenshot, not left out quietly.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests

REPO = Path(__file__).resolve().parent.parent
UI = REPO / "map-ui-prototype"


def _safe(label: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(label or "layer")).strip("_")[:60] or "layer"


def map_layer_events(events_path: Path) -> List[Dict[str, Any]]:
    """The turn's map_layer descriptors, in stream order, as the map UI's client takes them
    (src/replay.ts :: replayLayers): an `event: map_layer` block, or an agent_trace of that type,
    the descriptor being the payload when it carries geojson or a url and its `detail` otherwise."""
    out = []
    for line in events_path.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        name, p = e.get("event"), e.get("data") if isinstance(e.get("data"), dict) else {}
        if name == "agent_trace" and p.get("type"):
            name, p = p["type"], {**(p.get("detail") if isinstance(p.get("detail"), dict) else {})}
        if name != "map_layer":
            continue
        out.append((p if (p.get("geojson") or p.get("url")) else p.get("detail")) or {})
    return out


def capture_layers(base_url: str, events_path: Path, timeout: float = 120.0) -> Dict[str, int]:
    """Save every map_layer of a finished turn under <dir>/layers/, from the server that is still
    up. Inline geojson is written out as sent. Counts what was saved and what could not be."""
    stem = events_path.name[: -len(".events.jsonl")]
    lay = events_path.parent / "layers"
    counts = {"saved": 0, "inline": 0, "failed": 0}
    for n, d in enumerate(map_layer_events(events_path), 1):
        prefix = f"{stem}__{n:02d}__{_safe(d.get('label'))}"
        url = d.get("url")
        try:
            if url:
                r = requests.get(base_url.rstrip("/") + url if url.startswith("/") else url,
                                 timeout=timeout)
                r.raise_for_status()
                ctype = r.headers.get("content-type", "")
                ext = (".png" if "png" in ctype else ".jpg" if "jpeg" in ctype
                       else ".tif" if "tiff" in ctype else ".geojson")
                lay.mkdir(exist_ok=True)
                (lay / f"{prefix}{ext}").write_bytes(r.content)
                counts["saved"] += 1
            elif d.get("geojson") is not None:
                lay.mkdir(exist_ok=True)
                g = d["geojson"]
                (lay / f"{prefix}.geojson").write_text(g if isinstance(g, str) else json.dumps(g))
                counts["inline"] += 1
        except Exception:  # noqa: BLE001 — a lost layer is reported by the replay, not fatal here
            counts["failed"] += 1
    return counts


def render(paths: Iterable[Path], force: bool = False, parallel: int = 4) -> int:
    """Run the replay screenshots over `paths`. Returns the exit code."""
    if not shutil.which("npm"):
        print("screenshots: npm is not on PATH; skipped", file=sys.stderr)
        return 1
    if not (UI / "node_modules").exists():
        subprocess.run(["npm", "install", "--no-audit", "--no-fund"], cwd=UI, check=True)
    cmd = ["npm", "run", "-s", "replay:shots", "--",
           *[str(Path(p).resolve()) for p in paths], "--parallel", str(parallel)]
    if force:
        cmd.append("--force")
    return subprocess.run(cmd, cwd=UI).returncode


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--force", action="store_true", help="redo turns that already have a .png")
    ap.add_argument("--parallel", type=int, default=4)
    a = ap.parse_args(argv)
    return render(a.paths, a.force, a.parallel)


if __name__ == "__main__":
    sys.exit(main())
