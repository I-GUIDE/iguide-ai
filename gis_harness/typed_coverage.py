"""How many figures in correct recorded answers reach the gate as typed numbers. No model calls.

    python -m gis_harness.typed_coverage ARCHIVE                          # what was printed
    python -m gis_harness.typed_coverage ARCHIVE --reexec OUT [--jobs 3]  # + measuring calls

ARCHIVE holds `<run>/<provider>_<model>/<turn>.json` and `<turn>.events.jsonl` (the harness's
`runs/` or `gis_harness_runs_archive/`). Only turns scored correct are counted.

Per run and model it reports the stated figures (`facts.is_claim`), how many are lengths or
areas, and how many resolve to a typed fact:
- **today**: the typed values the record already carries (`verification.outputs`, tool
  `outputs`);
- **printed**: `measured_outputs.evaluate` over each recorded execute_code stdout, with the live
  cap of `MAX_OUTPUTS` per run;
- **re-executed** (`--reexec OUT`): every execute_code call re-run in `iguide-codeexec:latest`
  with no network, through the current gate prologue, against the task's regenerated dataset.
  That is the only way to see what the measuring calls returned, since older records have no
  values. Each turn's calls share one working directory, as a session's workspace does. Fetched
  inputs (TIGER, OpenStreetMap) are not regenerated, so those calls fail and type nothing.
  `with crs` counts the figures that inherit a measuring call's CRS.

The code comes from the CALL events: the recorded result's `code` has its whitespace collapsed,
and indented code would not parse.
"""
from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from agent_runtime import facts as turn_facts
from agent_runtime import measured_outputs, sandbox_verify, turn_log
from gis_harness import datasets, tasks
from gis_harness.replay_number_scan import tool_results

IMAGE = "iguide-codeexec:latest"
_DRIVER = ('for f in $(ls script_*.py 2>/dev/null | sort); do i=${f#script_}; i=${i%.py}; '
           'rm -f checks.json; timeout 150 python $f > out_$i.txt 2> err_$i.txt; '
           'echo $? > rc_$i.txt; [ -f checks.json ] && mv checks.json checks_$i.json; done; '
           'chmod -R a+rw /work')


def _body(content: Any) -> Dict[str, Any]:
    s = str(content or "").strip()
    if s.startswith("content="):
        quote = s[8]
        end = s.find(f"{quote} name=", 9)
        try:
            s = ast.literal_eval(s[8:end + 1] if end > 0 else s[8:])
        except Exception:  # noqa: BLE001
            return {}
    try:
        b = json.loads(s)
    except (TypeError, ValueError):
        return {}
    return b if isinstance(b, dict) else {}


def correct_turns(archive: Path) -> Iterator[Tuple[Path, Path, Dict[str, Any]]]:
    for tj in sorted(archive.rglob("*.json")):
        if {"layers", "server_files"} & set(tj.parts) or tj.name == "summary.json" \
                or tj.name.endswith(".bounds.json"):
            continue
        ev = tj.with_name(tj.stem + ".events.jsonl")
        if not ev.exists():
            continue
        try:
            turn = json.loads(tj.read_text())
        except ValueError:
            continue
        if (turn.get("score") or {}).get("correct") and turn.get("answer"):
            yield tj, ev, turn


def call_codes(events: Path) -> List[Optional[str]]:
    """The code each execute_code call sent, in order."""
    out: List[Optional[str]] = []
    for line in events.read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        d = e.get("data") or {}
        det = d.get("detail") if isinstance(d, dict) else None
        if not (e.get("event") == "analysis" and d.get("type") == "tool_call"
                and isinstance(det, dict) and det.get("name") == "execute_code"):
            continue
        args = det.get("args")
        if isinstance(args, str):
            try:
                args = ast.literal_eval(args)
            except Exception:  # noqa: BLE001
                args = {}
        out.append(args.get("code") if isinstance(args, dict) else None)
    return out


# --------------------------------------------------------------------------- re-execution

def reexecute(archive: Path, out: Path, jobs: int = 3) -> Dict[str, Any]:
    """`{"<turn path>#<call>": {"report", "stdout"}}` for every call of every correct turn."""
    built: Dict[str, Dict[str, Path]] = {}
    by_id = {t.id: t for t in tasks.TASKS}

    def dataset(name: str) -> Dict[str, Path]:
        if name not in built:
            try:
                files, _ = datasets.build(name, out / "_datasets" / name, out / "_cache")
                built[name] = {Path(p).name: Path(p) for p in files.values()}
            except Exception:  # noqa: BLE001 - a live task has no builder
                built[name] = {}
        return built[name]

    items = []
    for tj, ev, turn in correct_turns(archive):
        rel = str(tj.relative_to(archive))
        task = by_id.get(str(turn.get("task") or tj.stem.split(".")[0]))
        files = dataset(task.dataset) if task and task.dataset else {}
        work = out / rel.replace("/", "__")
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        codes, calls = call_codes(ev), []
        for res in tool_results(ev):
            if res.get("name") != "execute_code":
                continue
            i, b = len(calls), _body(res.get("content"))
            code = codes[i] if i < len(codes) else None
            calls.append(i if code else None)
            if not code:
                continue
            for spec in b.get("input_files") or []:
                src = files.get(str(spec.get("filename") or ""))
                for name in spec.get("available_as") or []:
                    if src is not None and "/" not in name:
                        shutil.copy(src, work / name)
            (work / f"script_{i:03d}.py").write_text(
                sandbox_verify.prologue_source({}) + code + sandbox_verify.epilogue_source())
        items.append((rel, work, [c for c in calls if c is not None]))

    def run(item):
        rel, work, calls = item
        subprocess.run(["docker", "run", "--rm", "--network", "none", "-v", f"{work}:/work",
                        "-w", "/work", "-e", "MPLBACKEND=Agg", IMAGE, "sh", "-c", _DRIVER],
                       capture_output=True, timeout=3600)
        found = {}
        for i in calls:
            cj, so = work / f"checks_{i:03d}.json", work / f"out_{i:03d}.txt"
            found[f"{rel}#{i}"] = {
                "report": json.loads(cj.read_text()) if cj.exists() else {},
                "stdout": so.read_text(errors="replace") if so.exists() else ""}
        return found

    result: Dict[str, Any] = {}
    with ThreadPoolExecutor(jobs) as pool:
        for found in pool.map(run, items):
            result.update(found)
    (out / "reexec.json").write_text(json.dumps(result))
    return result


# --------------------------------------------------------------------------- coverage

def coverage(archive: Path, reexec: Optional[Dict[str, Any]] = None) -> Dict[Tuple[str, str], Counter]:
    by: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for tj, ev, turn in correct_turns(archive):
        rel = str(tj.relative_to(archive))
        results = tool_results(ev)
        fs = turn_facts.build(results=results, query=turn.get("query") or "")
        claims = [r.quantity for r in turn_facts.resolve(turn["answer"], fs)
                  if turn_facts.is_claim(r.quantity)]
        today, printed, rerun = turn_facts.FactSet(), turn_facts.FactSet(), turn_facts.FactSet()
        n = 0
        for res in results:
            for t in turn_log.typed_outputs(_body(res.get("content"))):
                try:
                    today.add(float(t["value"]), unit=t.get("unit"), typed=True)
                except (TypeError, ValueError):
                    pass
            if res.get("name") != "execute_code":
                continue
            stdout = _body(res.get("content")).get("stdout") or ""
            for o in measured_outputs.evaluate({}, stdout)[1]:
                printed.add(float(o["value"]), unit=o["unit"], typed=True)
            rx = (reexec or {}).get(f"{rel}#{n}")
            if rx is not None:
                for o in measured_outputs.evaluate(rx["report"], rx["stdout"])[1]:
                    rerun.add(float(o["value"]), unit=o["unit"], typed=True,
                              measured_in_crs=o.get("measured_in_crs"))
            n += 1
        c = by[(tj.parent.name.split("_", 1)[-1], tj.relative_to(archive).parts[0])]
        c["turns"] += 1
        for q in claims:
            def hit(s):
                return next((f for f in s.facts if turn_facts._matches(q, f)), None)
            measure = bool(q.unit) and turn_facts.parse_unit(q.unit).dimension in (
                "[length]", "[length] ** 2")
            p, r = hit(printed), hit(rerun)
            c["figures"] += 1
            c["lengths/areas"] += measure
            c["today"] += hit(today) is not None
            c["printed"] += p is not None
            c["printed l/a"] += p is not None and measure
            if reexec is not None:
                c["re-executed"] += r is not None
                c["with crs"] += bool(r is not None and r.measured_in_crs)
    return by


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--reexec", help="re-execute every call into this directory first")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    archive = Path(a.archive)
    rx = None
    if a.reexec:
        out = Path(a.reexec)
        out.mkdir(parents=True, exist_ok=True)
        cached = out / "reexec.json"
        rx = json.loads(cached.read_text()) if cached.exists() else reexecute(archive, out, a.jobs)
    by = coverage(archive, rx)
    keys = ["turns", "figures", "lengths/areas", "today", "printed", "printed l/a"]
    keys += ["re-executed", "with crs"] if rx is not None else []
    print("model / run".ljust(40), *[k.rjust(13) for k in keys])
    totals: Dict[str, Counter] = defaultdict(Counter)
    for (model, run), c in sorted(by.items()):
        print(f"{model} / {run}"[:40].ljust(40), *[str(c[k]).rjust(13) for k in keys])
        for k in keys:
            totals[model][k] += c[k]
    for model, c in sorted(totals.items()):
        print(f"TOTAL {model}".ljust(40), *[str(c[k]).rjust(13) for k in keys])


if __name__ == "__main__":
    main()
