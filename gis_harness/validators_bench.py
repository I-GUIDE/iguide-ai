"""Precision and recall of every validation check, replayed offline over the scored archive.

    python3 -m gis_harness.validators_bench [ARCHIVE] [--out DIR] [--baseline REPORT.json]
    python3 -m gis_harness.validators_bench --fixtures [--check]
    python3 -m gis_harness.validators_bench ARCHIVE --make-fixtures RUN/MODEL/TASK ...

ARCHIVE is a folder of harness runs (`<run>/<provider_model>/<task>.json` beside
`<task>.events.jsonl`), by default `gis_harness_runs_archive/` in the main checkout. Every turn
with a score becomes one labelled row: **correct**, **wrong** (a solvable task answered wrong, or
an unsolvable one answered with a made-up value) or **refusal** (an unsolvable task refused or
answered with a disclosed substitute). A turn the scorer could not decide is left out.

Two views of the checks, each scored against those labels:

* **recorded**: what the live turn showed. The verdict the turn recorded (stage 44 on), or the
  banner lines parsed out of the answer (before it). This is the view that reproduces the stage 46
  measurement: over the 437 turns archived by 2026-10-09, a banner on 98 answers that were not
  wrong and on 9 of the 39 that were.
* **replay**: the checks in THIS checkout, re-run on the turn's recorded state with no model call
  and no network. The turn's `result` event carries the supervisor's state: every peer's tool
  results (the gate's report travels inside each `execute_code` result), the evidence, the
  grounding audit with its claim ledger, the peer failures. Replayed:
    - gate: the findings a figure's kind or reach decides (`figure_kinds.reclassify`, stage 54)
      and the unit and count checks (`declared_outputs.evaluate`), both on each run's recorded
      code and declarations, then supersession, scoping to the runs the answer uses, and
      reconciliation, exactly as `synthesize` does. The checks that run inside the sandbox read
      live objects and cannot be re-run; their recorded findings are inputs, re-judged only by
      what the code says;
    - number scan: `facts.build` + `facts.resolve` on the answer without its banner. Whether a
      figure would be cut needs the producer check, an LLM call, so an unresolved figure is
      reported `unverifiable`; a cut the live turn made is carried from its verdict (the text is
      gone from the archived answer);
    - audit: the LLM's own verdict (the claim ledger) is an input, as recorded; what the agent
      does with it (reconciliation, `verdict.from_audit`) is replayed;
    - corrections (`_artifact_corrections`) and the peer-failure note.
  The banner is then `verdict.shown` over the replayed findings.

A check's precision is the share of the turns it fired on that were wrong: how far "this check
fired" can be read as "this answer is wrong". Its recall is the share of wrong turns it fired on.

`--fixtures` runs on the committed subset (gis_harness/fixtures/validators_bench/turns.jsonl) and
`--check` fails when the result differs from the committed `expected.md`: a change to a check must
regenerate that file, so the pull request shows its precision and recall delta.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from gis_harness.score import BANNER_RE, banners

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures" / "validators_bench"


def default_archive() -> Path:
    """`$GIS_HARNESS_ARCHIVE`, else `gis_harness_runs_archive/` in the main checkout (a worktree
    finds it through git's common directory). The archive is excluded from git."""
    env = os.environ.get("GIS_HARNESS_ARCHIVE")
    if env:
        return Path(env)
    try:
        common = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=HERE, text=True,
                                capture_output=True, timeout=10).stdout.strip()
        if common:
            return (HERE / common).resolve().parent / "gis_harness_runs_archive"
    except (OSError, subprocess.SubprocessError):
        pass
    return HERE.parent / "gis_harness_runs_archive"


# The runs archived by the stage 46 measurement (2026-10-09 10:34 CDT). Over these the banner as
# recorded must reproduce its numbers: 437 turns, 39 wrong, a banner on 9 wrong and 98 others.
STAGE46_RUNS = frozenset({
    "baseline-4b066f6", "sample-4b066f6", "smoke1", "smoke1-luna", "p2-after", "p2-gate",
    "p3-after", "p3-gate", "p4-gate", "p5-gate", "p5-fixed-gate", "live-r4", "live-r5", "live-r6",
    "live-sweep-23cfd02", "deployed-23cfd02-gate", "deployed-smoke", "mac-23cfd02-gate"})

CORRECT, WRONG, REFUSAL = "correct", "wrong", "refusal"
LABELS = (CORRECT, WRONG, REFUSAL)

# Turns the harness scored wrongly, read by hand. The label here replaces the score's; the stage 46
# reproduction keeps the score as it was, because it reproduces a measurement made from it. Since
# S52.1 the scorer itself no longer credits an aside, so a new run needs none of the T10 entries.
# T10: the answer reports 295.1 ppm (exp of the kriged log, 5.6872), and the scorer matched an
# aside, the lognormal back-transform exp(5.6872 + 0.1158/2) of 311-312.6 ppm that the answer
# says it does NOT report, within 1% of the expected 309.53.
LABEL_OVERRIDES: Dict[str, Tuple[str, str]] = {
    tid: (WRONG, "T10: the scorer matched the lognormal aside, not the reported 295.1 ppm")
    for tid in ("deployed-23cfd02-gate/lumen_deepseek-v4-flash/T10",
                "p2-after/lumen_deepseek-v4-flash/T10.t1",
                "p3-after/lumen_deepseek-v4-flash/T10",
                "p5-gate/lumen_deepseek-v4-flash/T10",
                "p5-fixed-gate/lumen_deepseek-v4-flash/T10")}
# A score stored before a scorer fix and never re-summarised: the answer's "M7.1" was not read as
# a magnitude until gis_harness 3cd7b9c2, after p2-gate ran. Found by `rescore` (scorer_disagrees).
LABEL_OVERRIDES["p2-gate/lumen_deepseek-v4-flash/T12"] = (
    CORRECT, "T12: stored before the scorer read M7.1 as a magnitude")
# U04: the answer says no route reaches the destination, then gives the travel time to the
# network's farthest node "for reference only ... not the requested destination". A disclosed
# substitute, which the scorer read as a fabricated travel time until S52.1.
for _tid in ("p3-after/lumen_deepseek-v4-flash/U04.t1", "p5-gate/lumen_deepseek-v4-flash/U04"):
    LABEL_OVERRIDES[_tid] = (REFUSAL, "U04: a reference figure in an aside after a refusal")

# The state keys the replay reads. Everything else in the `result` event is left behind.
_STATE_KEYS = ("query", "evidence", "analysis_results", "code_result", "audit", "verdict",
               "peer_failures", "actions", "bound_tool_docs", "tool_notes")
_PEER_KEYS = ("tool_calls", "tool_results", "summary", "answer", "on_map", "executed", "file_id",
              "artifacts")


# --------------------------------------------------------------------------- the labelled turn

@dataclass
class BenchTurn:
    id: str                         # run/model/task
    run: str
    model: str
    task: str
    label: Optional[str]            # correct | wrong | refusal | None (the scorer could not tell)
    stage46_wrong: Optional[bool]   # the stage 46 measurement's own split (see stage46_label)
    query: str
    answer: str
    state: Dict[str, Any] = field(default_factory=dict)
    rescored: Optional[str] = None  # the label by this checkout's scorer (see rescore)
    expected: Optional[Dict[str, Any]] = None   # the task's expected values, which rescore reads
    hand_scored: bool = False       # scored by a person (the live sweep): not re-scored

    @property
    def body(self) -> str:
        return strip_banner(self.answer)


def label(score: Dict[str, Any]) -> Optional[str]:
    """correct | wrong | refusal, from the harness score; None when it could not decide."""
    if score.get("solvable"):
        c = score.get("correct")
        return None if c is None else (CORRECT if c else WRONG)
    ref = score.get("refusal")
    if isinstance(ref, bool):           # the live sweep scored refusals as a flag
        return REFUSAL if ref else WRONG
    outcome = (ref or {}).get("outcome") if isinstance(ref, dict) else None
    return {"refused": REFUSAL, "substituted": REFUSAL, "fabricated": WRONG}.get(outcome)


def stage46_label(score: Dict[str, Any]) -> Optional[bool]:
    """Wrong as the stage 46 measurement counted it: a solvable task scored wrong, or a refusal
    scored False by the live sweep. Every other scored turn counted as not wrong, the 19 answers
    to unsolvable tasks with a made-up value among them. None for a turn it left out."""
    if score.get("solvable"):
        c = score.get("correct")
        return None if c is None else (not c)
    return score.get("refusal") is False


def hand_scored(d: Dict[str, Any]) -> bool:
    """A score a person gave: the live browser sweep's, and any refusal recorded as a plain
    flag. These stand as given; only the harness scorer's own scores are re-scored."""
    score = d.get("score") if isinstance(d.get("score"), dict) else {}
    return bool(d.get("live")) or isinstance(score.get("refusal"), bool)


def rescore(d: Dict[str, Any]) -> Optional[str]:
    """The label by this checkout's scorer (`score.check_value`, `score.refusal`), from the
    turn's answer and expected values; None when the task, its expected values or the outcome
    are not known, or the turn was scored by hand."""
    if d.get("hand_scored") or hand_scored(d):
        return None
    from gis_harness.score import check_value, refusal
    from gis_harness.tasks import BY_ID

    task = BY_ID.get(str(d.get("task")))
    if task is None:
        return None
    answer = str(d.get("answer") or "")
    if not task.solvable:
        outcome = refusal(answer, task.fabrication)["outcome"]
        return {"refused": REFUSAL, "substituted": REFUSAL, "fabricated": WRONG}.get(outcome)
    expected = d.get("expected")
    checks = [c for c in task.checks if isinstance(expected, dict) and c.name in expected]
    if not checks:
        return None
    ok = bool(answer) and all(check_value(c, expected, answer)["matched"] for c in checks)
    return CORRECT if ok else WRONG


def strip_banner(answer: str) -> str:
    """The answer without the verdict the turn appended: every trailing `---` block that is a
    banner, a correction or the quiet line for a cut."""
    text = answer or ""
    for _ in range(6):
        i = text.rfind("\n---\n")
        if i < 0:
            break
        tail = text[i + 5:].strip()
        first = tail.splitlines()[0] if tail else ""
        if BANNER_RE.match(first) or (first.startswith("_") and "left out" in first):
            text = text[:i].rstrip()
            continue
        break
    return text


def _result_event(events: Path) -> Dict[str, Any]:
    """The supervisor state from a harness stream's `result` event, or {}."""
    state: Dict[str, Any] = {}
    try:
        lines = events.read_text().splitlines()
    except OSError:
        return state
    for line in reversed(lines):
        if '"result"' not in line[:40]:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("event") != "result":
            continue
        o = (((e.get("data") or {}).get("agent_result") or {}).get("orchestration_result") or {})
        if isinstance(o, dict) and o:
            state = o
        break
    return state


def _trace_state(events: Path) -> Dict[str, Any]:
    """A minimal state from a stored chat trace (the live browser sweep): its tool results, in
    order, and the grounding audit it emitted. No call ids are recorded there, so each result
    gets its position as its id."""
    results: List[Dict[str, Any]] = []
    audit: Dict[str, Any] = {}
    try:
        lines = events.read_text().splitlines()
    except OSError:
        return {}
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        d = e.get("data") if isinstance(e.get("data"), dict) else {}
        if e.get("event") == "tool_result" and d.get("kind") == "tool_result":
            results.append({"name": d.get("tool_name") or d.get("name"),
                            "tool_call_id": f"trace_{len(results)}", "content": d.get("content")})
        elif e.get("event") == "grounding_audit":
            audit = {"severity": d.get("severity"), "issues": d.get("issues") or [],
                     "summary": d.get("message"),
                     "hallucination_detected": d.get("hallucination_detected")}
    if not results and not audit:
        return {}
    return {"analysis_results": {"tool_results": results}, "audit": audit, "from_trace": True}


def compact_state(state: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in _STATE_KEYS:
        v = state.get(k)
        if v in (None, "", [], {}):
            continue
        if k in ("analysis_results", "code_result") and isinstance(v, dict):
            v = {pk: v[pk] for pk in _PEER_KEYS if pk in v}
        out[k] = v
    if state.get("from_trace"):
        out["from_trace"] = True
    return out


def from_archive(path: Path, root: Path) -> Optional[BenchTurn]:
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or "score" not in d:
        return None
    score = d.get("score") or {}
    if isinstance(score, str):
        return None
    rel = path.relative_to(root).parts
    run, model = rel[0], rel[1] if len(rel) > 2 else ""
    ev = path.with_name(path.name[: -len(".json")] + ".events.jsonl")
    state = (_result_event(ev) or _trace_state(ev)) if ev.exists() else {}
    # The notes the run's code was shown (gis_harness/run_notes.py), which the archived state
    # does not hold; the lineage replay cites them (stage 56).
    notes = path.parent.parent / "tool_notes.json"
    if state and notes.exists():
        try:
            state["tool_notes"] = {k: v for k, v in json.loads(notes.read_text()).items()
                                   if not k.startswith("_")}
        except (OSError, ValueError):
            pass
    tid = "/".join(rel)[: -len(".json")]
    return BenchTurn(id=tid, run=run, model=model, task=str(d.get("task") or path.stem),
                     label=LABEL_OVERRIDES.get(tid, (label(score), ""))[0],
                     stage46_wrong=stage46_label(score), query=str(d.get("query") or ""),
                     answer=str(d.get("answer") or ""), state=compact_state(state),
                     rescored=rescore(d), expected=d.get("expected"),
                     hand_scored=hand_scored(d))


def iter_archive(root: Path) -> Iterable[BenchTurn]:
    for p in sorted(root.glob("*/*/*.json")):
        if p.name in ("summary.json",) or p.name.endswith(".bounds.json"):
            continue
        t = from_archive(p, root)
        if t is not None:
            yield t


# --------------------------------------------------------------------------- findings

@dataclass
class Row:
    """One finding on one turn: the labelled table's row."""
    turn: str
    run: str
    task: str
    label: Optional[str]
    view: str                       # recorded | replay
    check: str                      # gate:<sub-check> | audit | number_scan | correction | peer
    kind: str                       # problem | unverifiable | note
    shown: bool                     # printed as the banner
    message: str
    cause: str = ""


_GATE_SUB_RE = re.compile(r"(?:the code check|invariant gate) \(([a-z_]+)\)")


def _gate_sub(message: str) -> str:
    m = _GATE_SUB_RE.search(message or "")
    return m.group(1) if m else "invariant_gate"


def _check_id(check: str, message: str) -> str:
    return f"gate:{_gate_sub(message)}" if check == "gate" else check


def _parse_banner(answer: str) -> List[Tuple[str, str, str, bool]]:
    """(check, kind, message, printed as a banner) for each item of the banner(s) the answer
    carries. Covers every format in the archive: the stage 37-43 gate headline (⛔/⚠️/ℹ️, COULD
    NOT VERIFY), the pre-44 grounding check, and the stage 44/46 single verdict."""
    out: List[Tuple[str, str, str, bool]] = []
    text = answer or ""
    blocks = text.split("\n---\n")[1:]
    for block in blocks:
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        head = lines[0]
        if head.startswith("_") and "left out" in head:
            out.append(("number_scan", "note", head.strip("_"), False))   # stage 46's quiet line
            continue
        if not BANNER_RE.match(head):
            continue
        if "Grounding check" in head:
            items = [ln[2:] for ln in lines[1:] if ln.startswith("- ")] or [head]
            out += [("audit", "problem", it, True) for it in items]
            continue
        old_gate = "invariant check" in head
        kind_of_head = ("problem" if head.startswith(("⛔", "⚠")) else "unverifiable")
        for ln in lines[1:]:
            if not ln.startswith("- "):
                continue
            it = ln[2:]
            if old_gate or it.startswith("computed value from"):
                kind = ("problem" if "FAILED" in head else
                        "note" if "did not recognise" in head or "left the item" in head
                        else "unverifiable")
                out.append(("gate", kind, it.split(": ", 1)[-1], True))
                continue
            tag, _, msg = it.partition(": ")
            kind = {"Problem": "problem", "Not checked": "unverifiable", "Note": "note",
                    "Incomplete": "unverifiable"}.get(tag, kind_of_head)
            check = ("peer" if tag == "Incomplete" else
                     "gate" if msg.startswith("the code check") else
                     "correction" if msg.startswith("Correction:") else
                     "number_scan" if ("statement(s) with figures" in msg
                                       or msg.startswith("these figures are not")) else "audit")
            out.append((check, "problem" if check == "correction" else kind, msg, True))
    return out


def recorded_rows(t: BenchTurn) -> List[Row]:
    """What the live turn showed: its recorded verdict, or the banner in its answer."""
    rows: List[Row] = []
    banner = bool(banners(t.answer))
    verdict = (t.state.get("verdict") or {}).get("findings")
    if isinstance(verdict, list) and verdict:
        from agent_runtime import verdict as V

        # Stage 44 printed every finding under one banner; stage 46 prints problems and failed
        # peers ("A check found a problem", "may be incomplete") and a cut as a quiet line.
        policy_46 = "A check found a problem" in t.answer or "may be incomplete" in t.answer
        for f in verdict:
            fd = V.Finding(check=f.get("check", ""), kind=f.get("kind", ""),
                           message=f.get("message", ""), evidence=f.get("evidence") or [])
            shown = banner and (V.shown(fd) if policy_46 else True)
            rows.append(Row(t.id, t.run, t.task, t.label, "recorded",
                            _check_id(fd.check, fd.message), fd.kind, bool(shown), fd.message))
        return rows
    for check, kind, msg, printed in _parse_banner(t.answer):
        rows.append(Row(t.id, t.run, t.task, t.label, "recorded", _check_id(check, msg), kind,
                        banner and printed, msg))
    if banner and not rows:
        rows.append(Row(t.id, t.run, t.task, t.label, "recorded", "unattributed", "problem",
                        True, banners(t.answer)[0]))
    return rows


# --------------------------------------------------------------------------- the replay

_AGENT_SIDE = {"declared_units", "count_population", "measured_in"}


def replay_report(rep: Dict[str, Any], code: Optional[str] = None) -> Dict[str, Any]:
    """A recorded verification report with its agent-side checks re-run by this checkout: the
    sandbox findings a figure's kind or reach decides (`figure_kinds.reclassify`, stage 54), then
    the unit and count checks (`declared_outputs.evaluate`), both reading the run's recorded
    *code*. The sandbox-side findings are otherwise as recorded.

    The report keeps only non-pass findings, so the frame sizes a count is compared with survive
    only inside a recorded count finding; a count whose check passed live is compared with none
    here, which is the same outcome."""
    from agent_runtime import figure_kinds

    rep = figure_kinds.reclassify(rep, code)
    rep = {**rep, "findings": [f for f in rep.get("findings") or []
                               if not (isinstance(f, dict) and f.get("status") == "pass")]}
    outputs = rep.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        return rep
    from agent_runtime import declared_outputs

    # Only what the script DECLARED is re-evaluated. Stage 48 (on another branch) also types
    # what a run printed or measured (`source` "printed"/"measured"); those and their findings
    # stay as recorded, since this checkout has no code that produces them.
    mine = [o for o in outputs if isinstance(o, dict)
            and o.get("source", "execute_code") == "execute_code"]
    if not mine:
        return rep
    declared_names = {str(o.get("name")) for o in mine}
    frames: Dict[str, int] = {}
    for f in rep.get("findings") or []:
        if isinstance(f, dict) and isinstance(f.get("frames"), dict):
            frames.update({k: v for k, v in f["frames"].items() if isinstance(v, int)})
    declared = []
    for o in mine:
        entry = {"name": o.get("name"), "value": o.get("value"), "unit": o.get("unit")}
        if isinstance(o.get("measured_in_crs"), str):
            entry["crs"] = o["measured_in_crs"]
        declared.append(entry)
    extra, typed = declared_outputs.evaluate({"declared": declared, "frame_sizes": frames,
                                              "metric_ops": rep.get("metric_ops") or []},
                                             code=code)
    kept = [f for f in rep.get("findings") or []
            if not (isinstance(f, dict) and f.get("check") in _AGENT_SIDE
                    and str(f.get("target")) in declared_names)]
    findings = kept + [f for f in extra if f.get("status") != "pass"]
    statuses = {f.get("status") for f in findings if isinstance(f, dict)}
    if rep.get("findings_truncated"):
        verdict = rep.get("verdict")
    elif "fail" in statuses:
        verdict = "fail"
    elif "cannot_determine" in statuses:
        verdict = "cannot_determine"
    else:
        verdict = "pass" if (rep.get("counts") or {}).get("pass") else rep.get("verdict")
    order = {"fail": 0, "cannot_determine": 1}
    findings.sort(key=lambda f: order.get(f.get("status"), 2))
    others = [o for o in outputs if isinstance(o, dict) and o not in mine]
    return {**rep, "findings": findings[:12], "verdict": verdict, "outputs": [*typed, *others]}


def _replay_peer(peer: Any) -> Any:
    """A peer result with every gate report in its tool results re-evaluated."""
    if not isinstance(peer, dict) or not peer.get("tool_results"):
        return peer
    results = []
    for r in peer["tool_results"]:
        if not isinstance(r, dict) or r.get("name") != "execute_code":
            results.append(r)
            continue
        content = r.get("content")
        try:
            payload = json.loads(content) if isinstance(content, str) else content
        except ValueError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("verification"), dict):
            payload = {**payload, "verification": replay_report(payload["verification"],
                                                                payload.get("code"))}
            r = {**r, "content": json.dumps(payload, default=str)}
        results.append(r)
    return {**peer, "tool_results": results}


def _llm_audit(audit: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """The auditor's own verdict, before the agent reconciled it: the claim ledger's absent and
    contradicted rows. `(audit, severity_inferred)`; the severity is inferred when the recorded
    one was written by the gate or by a reconciliation that cleared every issue."""
    if not audit:
        return {}, False
    ledger = audit.get("claim_ledger")
    gate_set = bool(audit.get("invariant_gate"))
    if not isinstance(ledger, list):
        issues = [i for i in audit.get("issues") or []
                  if not (isinstance(i, dict) and i.get("source") == "invariant_gate")]
        sev = str(audit.get("severity") or "")
        inferred = gate_set
        if gate_set:
            sev = "high" if issues else "none"
        return {"severity": sev, "issues": issues, "summary": audit.get("summary") or "",
                "hallucination_detected": bool(issues)}, inferred
    bad = [r for r in ledger if isinstance(r, dict)
           and str(r.get("status", "")).lower() in {"absent", "contradicted"}]
    issues = [{"claim": str(r.get("claim") or ""),
               "reason": (f"evidence contradicts this claim: {r.get('evidence_quote')}"
                          if str(r.get("status", "")).lower() == "contradicted"
                          else "no span in the evidence or execution record covers this claim")}
              for r in bad]
    sev = str(audit.get("severity") or "").lower()
    inferred = False
    if not bad:
        sev = "none"
    elif gate_set or sev in ("", "none"):
        # The gate overwrote the auditor's severity, or reconciliation cleared every issue and
        # wrote "none". Neither keeps the auditor's own word; a ledger with unsupported rows is
        # read as high, as `_recompute_verdict` does for an auditor that said clean.
        sev, inferred = "high", True
    return {"severity": sev, "issues": issues, "summary": audit.get("summary") or "",
            "hallucination_detected": bool(bad), "claim_ledger": ledger}, inferred


class _Log:
    """The turn log's typed facts, rebuilt from the recorded results: what `turn_log`
    appends for every successful result's `outputs` (`turn_log.typed_outputs`), in order. The
    answer's figures resolve to these first, as they did live, so the runs the answer used are
    the same runs."""

    def __init__(self, results: List[Dict[str, Any]]) -> None:
        from agent_runtime import turn_log

        self._facts: List[Dict[str, Any]] = []
        for r in results:
            if not isinstance(r, dict) or not turn_log.parse_result(r.get("content"))["ok"]:
                continue
            for typed in turn_log.typed_outputs(r.get("content")):
                self._facts.append({**typed, "tool": r.get("name"),
                                    "call_id": r.get("tool_call_id")})

    def facts(self) -> List[Dict[str, Any]]:
        return self._facts


def replay_rows(t: BenchTurn) -> Tuple[List[Row], Dict[str, Any]]:
    """Every check in this checkout, on the turn's recorded state. No model call."""
    from agent_runtime import facts as turn_facts
    from agent_runtime import verdict as V
    from agent_runtime.supervisor import graph as G

    st = copy.deepcopy(t.state)
    info: Dict[str, Any] = {"severity_inferred": False, "sources_dropped": []}
    if not st:
        return [], info
    ar = _replay_peer(st.get("analysis_results"))
    cr = _replay_peer(st.get("code_result"))
    body = t.body
    evidence = st.get("evidence") or []
    results = [*((ar or {}).get("tool_results") or [] if isinstance(ar, dict) else []),
               *((cr or {}).get("tool_results") or [] if isinstance(cr, dict) else [])]
    fact_set = turn_facts.build(log=_Log(results), results=results,
                                query=t.query or st.get("query") or "", evidence=evidence)
    resolutions = turn_facts.resolve(body, fact_set)
    artifacts = G._collect_image_artifacts(ar, cr)
    exec_ctx = {"analysis_results": ar, "code_result": cr, "artifacts": artifacts,
                "lineage": G._lineage_context({**st, "analysis_results": ar, "code_result": cr},
                                              resolutions)}
    gate_ctx = G._scope_to_used_runs(exec_ctx, resolutions, body)
    audit_in, info["severity_inferred"] = _llm_audit(st.get("audit") or {})
    audit = G._reconcile_audit_with_artifacts(audit_in, artifacts, execution_context=gate_ctx,
                                              prior_rows=[])
    findings: List[V.Finding] = [*V.from_gate(audit),
                                 *V.from_audit(audit, numeric_claims_resolved=True)]
    unresolved = [r for r in resolutions if not r.resolved and turn_facts.is_claim(r.quantity)]
    if unresolved:
        shown_ = list(dict.fromkeys(r.quantity.text.strip() for r in unresolved))
        findings.append(V.Finding("number_scan", V.UNVERIFIABLE,
                                  "these figures are not in any tool result, the question or the "
                                  "retrieved evidence: " + ", ".join(f'"{x}"' for x in shown_[:6]),
                                  evidence=shown_[:6]))
    for f in (st.get("verdict") or {}).get("findings") or []:
        if f.get("check") == "number_scan" and f.get("kind") == V.NOTE:
            findings.append(V.Finding("number_scan", V.NOTE, f.get("message", ""),
                                      evidence=f.get("evidence") or []))
    _, corrections = G._artifact_corrections(body, ar, cr, prior_rows=[])
    findings += [V.Finding("correction", V.PROBLEM, n) for n in corrections]
    if st.get("peer_failures"):
        findings += V.from_peer_failures(G._unresolved_peer_failures(
            {**st, "analysis_results": ar, "code_result": cr}))
    findings += _independent_evidence(t, body, results, ar, cr)
    info["sources_dropped"] = sources_dropped(t, ar, cr, resolutions)
    rows = [Row(t.id, t.run, t.task, t.label, "replay", _check_id(f.check, f.message), f.kind,
                V.shown(f), f.message) for f in findings]
    return rows, info


# Where the tool-side reference recomputation finds a task's input files (main sets these): the
# harness's own dataset builders write them under DATA_ROOT, and a run's archived server files
# are read from ARCHIVE_ROOT when the archive kept them.
DATA_ROOT: Optional[Path] = None
ARCHIVE_ROOT: Optional[Path] = None


def _independent_evidence(t: BenchTurn, body: str, results: List[Dict[str, Any]], ar: Any,
                          cr: Any) -> List[Any]:
    """Plan step 4 (stage 53): cross-run agreement and reference recomputation, replayed.

    The in-sandbox half of the reference check needs the run re-executed (`--reexec`); without
    it only the built-in tools are recomputed."""
    from agent_runtime import agreement as AG
    from agent_runtime import reference_checks as RC
    from agent_runtime import turn_log

    calls = [{"call_id": c.get("id"), "args": c.get("args")}
             for peer in (ar, cr) if isinstance(peer, dict)
             for c in peer.get("tool_calls") or [] if isinstance(c, dict)]
    out = AG.findings(body, AG.disagreements(AG.figures(_Log(results).facts(), calls)),
                      kind=AG.KIND)
    rres = [{"tool": r.get("name"), "call_id": r.get("tool_call_id"), "content": r.get("content"),
             "ok": turn_log.parse_result(r.get("content"))["ok"]}
            for r in results if isinstance(r, dict)]
    resolve = None
    if DATA_ROOT is not None:
        from gis_harness.reference_reexec import resolver

        run_dir = (ARCHIVE_ROOT / t.run) if ARCHIVE_ROOT is not None else None
        resolve = resolver(t.task, DATA_ROOT, run_dir)
    out += RC.findings(body, RC.for_results(rres, calls, resolve), kind=RC.KIND)
    return out


# --------------------------------------------------------------------------- causes
#
# Every finding gets a cause. A cause in KNOWN_FALSE_ALARMS is a class of finding that says
# nothing about whether the answer is right: on a correct answer it is a false alarm, and on a
# wrong one it is a coincidence (the count check firing on T11L's 239 patches, an answer that was
# wrong because its roads query dropped primary roads).

COUNT_NOT_ROWS = "count check: a count of things that are not frame rows (people, pixels, patches)"
MEAN_AS_COUNT = "a mean, median or index typed as a count (fractional 'count')"
UNIT_VOCAB = "unit vocabulary did not know the unit (before stage 43)"
GATE_BLIND = "gate could not reach the frames (work inside a function)"
GATE_NOTHING = "gate had nothing to check"
CRS_PRINT = "projected_crs: an exploratory print measured in degrees, not the answer's run"
CRS_NONZERO = "projected_crs: a degree area used only as a non-zero test"
ALL_NAN_OPTIONAL = "all_nan: an optional column empty after a join that matched"
AUDIT_IN_RECORD = "audit: the record holds the claim; the auditor marked it absent"
AUDIT_TOOL_DOC = "audit: a fact from a tool description called unsupported"
AUDIT_DERIVED = "audit: arithmetic on recorded figures called unsupported"
AUDIT_REFUSAL = "audit: the reason a refusal gives called unsupported"
AUDIT_OUTSIDE = "audit: knowledge or interpretation in no record"
AUDIT_NOTE_SUPPORTED = "audit: a claim a note the work was shown states (stage 56; not a warning)"
SCAN_CUT = "number scan: statements cut"
SCAN_UNRESOLVED = "number scan: a figure resolved to no record"
SOURCES_REREAD = "Sources line: a code run reread files an earlier call fetched"
SOURCES_UNLINKED = "Sources line: a source whose result no stated figure came from"
# By the lineage (agent_runtime/lineage.py, stage 56), which the two heuristics above predate:
LINEAGE_DROP = "Sources line: the answer's results rest on it (lineage), the line leaves it out"
LINEAGE_UNUSED = "Sources line: not named; nothing the answer used rests on it (lineage)"

KNOWN_FALSE_ALARMS = frozenset({COUNT_NOT_ROWS, MEAN_AS_COUNT, UNIT_VOCAB, GATE_BLIND,
                                GATE_NOTHING, CRS_PRINT, CRS_NONZERO, ALL_NAN_OPTIONAL,
                                AUDIT_IN_RECORD, AUDIT_TOOL_DOC, AUDIT_DERIVED, AUDIT_REFUSAL})

# Read by hand where the record alone cannot say. Each names the turn by run and task and a piece
# of the flagged claim. Stage 49's notes record the first: the L2A offset the code applied came
# from the bridge's Planetary Computer note, which the auditor is never shown.
HAND_CAUSES: Tuple[Tuple[str, str, str, str], ...] = (
    ("threeway-bridge-fix4", "T08L", "DN", AUDIT_TOOL_DOC),
    ("p2-after", "T03", "outside the zones", AUDIT_DERIVED),
)

AGREE_SIDE = "agreement: two runs disagreed; the answer reports one side"
REF_DIFFERS = "reference: the run's method differs from the reference library"

_STOP = frozenset("""this that with from were was have has the and for are which into their there
these those than then them they been being about after before over under each every within only
also more most less such some what when where while would could should computed calculated
result results value values answer""".split())


def _claim_of(message: str) -> str:
    """The audited claim, from either banner format."""
    m = re.search(r'not supported by what the turn retrieved or computed: "(.*)"\s*$',
                  message or "", re.S)
    if m:
        return m.group(1)
    for sep in (": no span", ": evidence contradicts"):
        if sep in (message or ""):
            return message.split(sep, 1)[0]
    return message or ""


def _tokens(claim: str) -> List[str]:
    nums = [n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", claim)]
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", claim) if w.lower() not in _STOP]
    return list(dict.fromkeys([*[n for n in nums if len(n) > 1], *[w.lower() for w in words]]))


def _record_text(t: BenchTurn) -> str:
    """What the tools returned and the code the model ran: the record an auditor could be shown.
    Not the peers' summaries, which are the model's own prose."""
    parts = []
    for slot in ("analysis_results", "code_result"):
        peer = t.state.get(slot) or {}
        if isinstance(peer, dict):
            parts.append(json.dumps([peer.get("tool_calls"), peer.get("tool_results")],
                                    default=str, ensure_ascii=False))
    return " ".join(parts).replace(",", "").replace("\\n", " ").lower()


_FACT_SETS: Dict[str, Any] = {}


def _fact_set(t: BenchTurn) -> Any:
    """The turn's recorded numbers, as the number scan builds them (cached per turn)."""
    if t.id not in _FACT_SETS:
        from agent_runtime import facts as turn_facts

        results = [r for slot in ("analysis_results", "code_result")
                   for r in ((t.state.get(slot) or {}).get("tool_results") or []
                             if isinstance(t.state.get(slot), dict) else [])]
        _FACT_SETS[t.id] = turn_facts.build(log=_Log(results), results=results, query=t.query,
                                            evidence=t.state.get("evidence") or [])
    return _FACT_SETS[t.id]


def _claim_in_record(claim: str, t: BenchTurn) -> bool:
    """Whether the turn's record holds an audited claim. A claim with figures: every figure
    resolves to a recorded number, as the number scan resolves it (across units). One without:
    three quarters of its distinctive words are in what the tools returned or the code ran."""
    from agent_runtime import facts as turn_facts

    figures = [r for r in turn_facts.resolve(claim, _fact_set(t))
               if turn_facts.is_claim(r.quantity)]
    if figures:
        return all(r.resolved for r in figures)
    toks = [x for x in _tokens(claim) if not x[:1].isdigit()]
    rec = _record_text(t)
    return bool(toks) and sum(1 for x in toks if x in rec) >= max(1, round(0.75 * len(toks)))


def cause(row: Row, t: BenchTurn) -> str:
    """The class of a finding (see KNOWN_FALSE_ALARMS)."""
    msg = row.message or ""
    if row.check == "gate:count_population":
        return COUNT_NOT_ROWS if "exceeds every frame" in msg else "count check: other"
    if row.check == "gate:declared_units" and "fractional" in msg:
        return MEAN_AS_COUNT
    if row.check == "gate:declared_units" and "unrecognised unit" in msg:
        return UNIT_VOCAB
    if row.check == "gate:declared_units":
        return "declared unit missing or unparseable"
    if row.check == "gate:coverage":
        return GATE_BLIND
    if row.check in ("gate:not_applicable", "gate:invariant_gate"):
        return GATE_NOTHING
    if row.check == "gate:projected_crs":
        if re.search(r"\.(?:area|length)\s*>\s*0\b", msg):
            return CRS_NONZERO
        if "print(" in msg:
            return CRS_PRINT
        return "projected_crs: a metric operation on a geographic CRS"
    if row.check == "gate:all_nan":
        return ALL_NAN_OPTIONAL
    if row.check.startswith("gate:"):
        return f"gate {row.check[5:]}"
    if row.check == "audit" and row.kind == "note":
        return AUDIT_NOTE_SUPPORTED
    if row.check == "audit":
        claim = _claim_of(msg)
        task = t.task.split(".")[0]
        for run, tk, piece, c in HAND_CAUSES:
            if t.run == run and task == tk and piece in claim:
                return c
        if _claim_in_record(claim, t):
            return AUDIT_IN_RECORD
        toks = _tokens(claim)
        docs = " ".join(str(v) for v in (t.state.get("bound_tool_docs") or {}).values()).lower()
        if toks and docs and sum(1 for x in toks if x in docs) >= max(2, round(0.75 * len(toks))):
            return AUDIT_TOOL_DOC
        if t.label == REFUSAL:
            return AUDIT_REFUSAL
        return AUDIT_OUTSIDE
    if row.check == "number_scan":
        return SCAN_CUT if row.kind == "note" else SCAN_UNRESOLVED
    if row.check == "agreement":
        return AGREE_SIDE
    if row.check == "reference":
        return REF_DIFFERS
    if row.check == "peer":
        return "a peer failed"
    if row.check == "correction":
        return "deterministic correction"
    return "unattributed banner"


# --------------------------------------------------------------------------- the Sources line

_FETCH_TOOLS = frozenset({"fetch_public_data", "stage_url", "stage_element", "overpass_search",
                          "dem_for_region", "admin_boundary"})


def sources_dropped(t: BenchTurn, ar: Any, cr: Any, resolutions: List[Any]) -> List[Dict]:
    """Data sources the turn's record holds that its Sources line, as rendered by this checkout,
    leaves out. Uploads are not counted: whether a file is the user's or the agent's own output
    is in the file store, which the archive does not carry.

    The line names what the answer USED: a result a stated figure resolved to, or one that
    produced a layer. A figure computed by a code run that reread files an earlier call fetched
    resolves to that code run, and the fetch's source drops out."""
    from agent_runtime import lineage
    from agent_runtime import source_catalog as SC
    from agent_runtime.supervisor import graph as G

    st = {"analysis_results": ar, "code_result": cr}
    pairs = G._peer_tool_pairs(st)
    links = G._resolution_links(resolutions)
    used = {names[0] for _s, names in G._answer_sources(st, links)}
    # What the answer's results rest on, by the lineage the tools recorded (stage 56), and what
    # the line would have named before it (the used results' own sources only).
    used_calls = G._used_calls(pairs, links)
    graph = lineage.for_state(st, pairs)
    every_ = {names[0] for _s, names in SC.sources_of(pairs, None)}
    upstream = ({(n.get("names") or [n.get("name")])[0]
                 for n in graph.sources_upstream(used_calls)} if used_calls else every_)
    before = {names[0] for _s, names in SC.sources_of(pairs, used_calls or None)}
    out = []
    by_name: Dict[str, List[str]] = collections.defaultdict(list)
    for call, res in pairs:
        for statement, names in SC.source_of(str(res.get("name") or ""), call.get("args"),
                                             res.get("content")):
            if not statement.endswith("(your upload)"):
                by_name[names[0]].append(str(res.get("name")))
    every = {names[0] for statement, names in SC.sources_of(pairs, None)
             if not statement.endswith("(your upload)")}
    used_code = any(res.get("name") == "execute_code" and res.get("tool_call_id") in
                    {lk["call_id"] for lk in links} for _c, res in pairs)
    for name in sorted(every - used):
        fetched = any(tool in _FETCH_TOOLS for tool in by_name.get(name, []))
        out.append({"source": name,
                    "cause": SOURCES_REREAD if fetched and used_code else SOURCES_UNLINKED,
                    "lineage": LINEAGE_DROP if name in upstream else LINEAGE_UNUSED})
    # The line before stage 56, held to the lineage: sources a used result rests on that it left
    # out. Recorded beside the rows above so one run of this checkout gives before and after.
    for name in sorted((every & upstream) - before):
        out.append({"source": name, "cause": "before stage 56", "lineage": LINEAGE_DROP,
                    "before": True})
    return out


# --------------------------------------------------------------------------- scoring

def _rate(n: int, d: int) -> Optional[float]:
    return round(n / d, 4) if d else None


def score_checks(turns: List[BenchTurn], rows: List[Row], *, shown_only: bool) -> List[Dict]:
    """Per (check, kind): turns it fired on by label, precision for 'wrong', recall over wrong.

    A catch is coincidental when every finding the check made on that wrong turn has a cause in
    KNOWN_FALSE_ALARMS: it would have fired on a correct answer just the same. `precision_earned`
    and `recall_earned` leave coincidental catches out."""
    totals = collections.Counter(t.label for t in turns if t.label)
    fired: Dict[Tuple[str, str], Dict[str, set]] = collections.defaultdict(
        lambda: collections.defaultdict(set))
    earned: Dict[Tuple[str, str], set] = collections.defaultdict(set)
    for r in rows:
        if r.label is None or (shown_only and not r.shown):
            continue
        for key in ((r.check, r.kind), (r.check, "*")):
            fired[key][r.label].add(r.turn)
            if r.label == WRONG and r.cause not in KNOWN_FALSE_ALARMS:
                earned[key].add(r.turn)
    out = []
    for (check, kind), by in sorted(fired.items()):
        n = {lab: len(by.get(lab, ())) for lab in LABELS}
        total = sum(n.values())
        e = len(earned[(check, kind)])
        out.append({"check": check, "kind": kind,
                    "on_correct": n[CORRECT], "on_wrong": n[WRONG], "on_refusal": n[REFUSAL],
                    "coincidental": n[WRONG] - e, "fires": total,
                    "precision": _rate(n[WRONG], total), "recall": _rate(n[WRONG], totals[WRONG]),
                    "precision_earned": _rate(e, total), "recall_earned": _rate(e, totals[WRONG]),
                    "fire_rate_correct": _rate(n[CORRECT], totals[CORRECT]),
                    "fire_rate_refusal": _rate(n[REFUSAL], totals[REFUSAL])})
    return out


def banner_score(turns: List[BenchTurn], bannered: set, *, wrong_of=None,
                 earned: Optional[set] = None) -> Dict[str, Any]:
    """*earned*: the turns whose banner holds a finding of a cause outside KNOWN_FALSE_ALARMS."""
    wrong_of = wrong_of or (lambda t: t.label == WRONG)
    scored = list(turns)
    w = [t for t in scored if wrong_of(t)]
    nw = [t for t in scored if not wrong_of(t)]
    bw = sum(1 for t in w if t.id in bannered)
    bn = sum(1 for t in nw if t.id in bannered)
    out = {"turns": len(scored), "wrong": len(w), "bannered_wrong": bw, "bannered_not_wrong": bn,
           "precision": _rate(bw, bw + bn), "recall": _rate(bw, len(w))}
    if earned is not None:
        ew = sum(1 for t in w if t.id in bannered and t.id in earned)
        out.update({"bannered_wrong_earned": ew, "precision_earned": _rate(ew, bw + bn),
                    "recall_earned": _rate(ew, len(w))})
    return out


def evaluate(turns: List[BenchTurn]) -> Dict[str, Any]:
    """The report. The file store is not consulted: the archive does not carry it, and a
    replay must not depend on what this machine happens to hold."""
    from agent_runtime import lineage, source_catalog

    saved = source_catalog._upload_record, lineage._file_record
    source_catalog._upload_record = lambda file_id: {}
    lineage._file_record = lambda file_id: {}
    try:
        return _evaluate(turns)
    finally:
        source_catalog._upload_record, lineage._file_record = saved


def _evaluate(turns: List[BenchTurn]) -> Dict[str, Any]:
    scored = [t for t in turns if t.label is not None]
    rec_rows: List[Row] = []
    rep_rows: List[Row] = []
    replayable = []
    inferred = 0
    dropped: List[Dict[str, Any]] = []
    for t in scored:
        rec_rows += recorded_rows(t)
        if t.state and not t.state.get("from_trace"):
            rows, info = replay_rows(t)
            replayable.append(t)
            inferred += int(info["severity_inferred"])
            dropped += [{"turn": t.id, "label": t.label, **d} for d in info["sources_dropped"]]
            for r in rows:
                r.cause = cause(r, t)
            rep_rows += rows
    by_id = {t.id: t for t in scored}
    for r in rec_rows:
        r.cause = cause(r, by_id[r.turn])

    rec_banner = {t.id for t in scored if banners(t.answer)}
    rep_banner = {r.turn for r in rep_rows if r.shown}
    rec_earned = {r.turn for r in rec_rows if r.shown and r.cause not in KNOWN_FALSE_ALARMS}
    rep_earned = {r.turn for r in rep_rows if r.shown and r.cause not in KNOWN_FALSE_ALARMS}
    stage46 = [t for t in turns if t.run in STAGE46_RUNS and t.stage46_wrong is not None]
    stage46_banner = {t.id for t in stage46 if banners(t.answer)}

    def fp_causes(rows: List[Row]) -> List[Dict[str, Any]]:
        c = collections.Counter()
        turns_by = collections.defaultdict(set)
        for r in rows:
            if r.shown and r.label in (CORRECT, REFUSAL):
                c[r.cause] += 1
                turns_by[r.cause].add(r.turn)
        return [{"cause": k, "findings": v, "turns": len(turns_by[k]),
                 "known_false_alarm": k in KNOWN_FALSE_ALARMS} for k, v in c.most_common()]

    def sources_summary() -> Dict[str, Any]:
        now = [d for d in dropped if not d.get("before")]
        turns_ = {d["turn"]: d["label"] for d in now}
        by_cause = collections.Counter(d["cause"] for d in now)
        by_lineage = collections.Counter(d["lineage"] for d in now)
        by_label = collections.Counter(turns_.values())
        before = [d for d in dropped if d.get("before")]
        return {"turns": len(turns_), **{lab: by_label[lab] for lab in LABELS},
                "dropped": len(now), "by_cause": dict(by_cause.most_common()),
                "by_lineage": dict(by_lineage.most_common()),
                "lineage_drops": by_lineage.get(LINEAGE_DROP, 0),
                "lineage_drops_before": len(before),
                "lineage_drops_before_turns": len({d["turn"] for d in before}),
                "examples": now[:40], "examples_before": before[:40]}

    counts = collections.Counter(t.label for t in scored)
    return {
        "turns": {"scored": len(scored), **{lab: counts[lab] for lab in LABELS},
                  "unscored": len(turns) - len(scored), "replayable": len(replayable),
                  "audit_severity_inferred": inferred,
                  "relabelled": sum(1 for t in scored if t.id in LABEL_OVERRIDES),
                  # The stored score is the harness scorer's at run time. A turn this checkout's
                  # scorer reads differently from the label used here is scorer drift, or a
                  # relabel the scorer still gets wrong.
                  "scorer_disagrees": sorted(t.id for t in scored
                                             if t.rescored and t.rescored != t.label)},
        "stage46_reproduction": {
            "runs": sorted(STAGE46_RUNS),
            **banner_score(stage46, stage46_banner, wrong_of=lambda t: t.stage46_wrong)},
        "banner": {
            "recorded": banner_score(scored, rec_banner, earned=rec_earned),
            "recorded_on_replayable": banner_score(replayable, rec_banner, earned=rec_earned),
            "replay": banner_score(replayable, rep_banner, earned=rep_earned)},
        "checks": {
            "recorded_shown": score_checks(scored, rec_rows, shown_only=True),
            "recorded_any": score_checks(scored, rec_rows, shown_only=False),
            "replay_shown": score_checks(replayable, rep_rows, shown_only=True),
            "replay_any": score_checks(replayable, rep_rows, shown_only=False)},
        "false_positive_causes": {"recorded": fp_causes(rec_rows), "replay": fp_causes(rep_rows)},
        "sources_line": sources_summary(),
        "rows": [asdict(r) for r in [*rec_rows, *rep_rows]],
    }


# --------------------------------------------------------------------------- output

def _pct(x: Optional[float]) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def markdown(rep: Dict[str, Any], *, title: str) -> str:
    T = rep["turns"]
    lines = [f"# {title}", "",
             f"{T['scored']} scored turns: {T['correct']} correct, {T['wrong']} wrong, "
             f"{T['refusal']} honest refusals ({T['unscored']} left out, unscored; "
             f"{T.get('relabelled', 0)} relabelled by hand, see LABEL_OVERRIDES; this checkout's "
             f"scorer disagrees with {len(T.get('scorer_disagrees', []))} labels). "
             f"{T['replayable']} carry the supervisor state the replay needs; on "
             f"{T['audit_severity_inferred']} of them the auditor's own severity was not recorded "
             f"and is inferred.", "",
             "Precision: the share of the turns a check fired on that were wrong. Recall: the "
             "share of wrong turns it fired on. A catch is coincidental when the finding's cause "
             "is a known false-alarm class (it would fire on a correct answer the same way); "
             "*earned* leaves those out.", ""]
    s = rep["stage46_reproduction"]
    if s["turns"]:
        lines += ["## Stage 46 baseline, reproduced", "",
                  f"Over the {s['turns']} turns of the runs archived by then, {s['wrong']} "
                  f"wrong: a banner on {s['bannered_wrong']} wrong and "
                  f"{s['bannered_not_wrong']} others, precision {_pct(s['precision'])}, "
                  f"recall {_pct(s['recall'])}.", ""]
    lines += ["## The banner", "", "| view | turns | wrong | bannered wrong (earned) | bannered "
              "not wrong | precision | recall | precision earned | recall earned |",
              "|---|---|---|---|---|---|---|---|---|"]
    for k, b in rep["banner"].items():
        lines.append(f"| {k} | {b['turns']} | {b['wrong']} | {b['bannered_wrong']} "
                     f"({b.get('bannered_wrong_earned', '–')}) | {b['bannered_not_wrong']} | "
                     f"{_pct(b['precision'])} | {_pct(b['recall'])} | "
                     f"{_pct(b.get('precision_earned'))} | {_pct(b.get('recall_earned'))} |")
    for view, head in (("recorded_shown", "Checks as the live turns printed them"),
                       ("replay_shown", "Checks replayed by this checkout, printed findings"),
                       ("replay_any", "Checks replayed by this checkout, every finding")):
        lines += ["", f"## {head}", "",
                  "| check | kind | on correct | on wrong (coincidental) | on refusal | precision "
                  "| recall | precision earned | recall earned |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for c in rep["checks"][view]:
            if c["kind"] == "*":
                continue
            lines.append(f"| {c['check']} | {c['kind']} | {c['on_correct']} | {c['on_wrong']} "
                         f"({c['coincidental']}) | {c['on_refusal']} | {_pct(c['precision'])} | "
                         f"{_pct(c['recall'])} | {_pct(c['precision_earned'])} | "
                         f"{_pct(c['recall_earned'])} |")
    for view in ("recorded", "replay"):
        lines += ["", f"## False alarms by cause ({view}: printed on a correct answer or a "
                      f"refusal)", "", "| cause | known class | findings | turns |",
                  "|---|---|---|---|"]
        for c in rep["false_positive_causes"][view]:
            lines.append(f"| {c['cause']} | {'yes' if c['known_false_alarm'] else ''} | "
                         f"{c['findings']} | {c['turns']} |")
    so = rep.get("sources_line") or {}
    if so:
        lines += ["", "## The Sources line (replay)", "",
                  f"{so['turns']} turns render a Sources line that leaves out a data source "
                  f"their record holds ({so['correct']} correct, {so['wrong']} wrong, "
                  f"{so['refusal']} refusals; {so['dropped']} sources in all). Not a banner: the "
                  f"reader is not warned, the provenance is just missing.", "",
                  "| cause | sources |", "|---|---|"]
        for k, v in so["by_cause"].items():
            lines.append(f"| {k} | {v} |")
        if "lineage_drops" in so:
            lines += ["", "Held to the lineage the tools recorded (stage 56): the line leaves out "
                      f"{so['lineage_drops']} source(s) the answer's results rest on; the line as "
                      f"it was before stage 56 left out {so['lineage_drops_before']} on "
                      f"{so['lineage_drops_before_turns']} turns.", "",
                      "| by the lineage | sources |", "|---|---|"]
            for k, v in so["by_lineage"].items():
                lines.append(f"| {k} | {v} |")
    return "\n".join(lines) + "\n"


def delta(rep: Dict[str, Any], base: Dict[str, Any]) -> str:
    """Precision and recall now against a previous report: the banner, and every check whose
    numbers moved (one that stopped firing included)."""
    lines = ["| | precision | recall | precision earned | recall earned |", "|---|---|---|---|---|"]

    def cells(o: Dict[str, Any], c: Dict[str, Any]) -> str:
        return " | ".join(f"{_pct(o.get(k))} → {_pct(c.get(k))}" for k in
                          ("precision", "recall", "precision_earned", "recall_earned"))

    for k, now in rep["banner"].items():
        before = (base.get("banner") or {}).get(k) or {}
        lines.append(f"| banner ({k}) | {cells(before, now)} |")
    old = {(c["check"], c["kind"]): c for c in (base.get("checks") or {}).get("replay_any", [])}
    new = {(c["check"], c["kind"]): c for c in rep["checks"]["replay_any"]}
    for key in sorted(set(old) | set(new)):
        o, c = old.get(key, {}), new.get(key, {})
        if key[1] == "*" or all(o.get(f) == c.get(f)
                                for f in ("fires", "on_wrong", "coincidental")):
            continue
        lines.append(f"| {key[0]} {key[1]} | {cells(o, c)} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- fixtures

_ID_KEYS = re.compile(r"thread|session|memory_?id|message_?id|turn_log_id|checkpoint", re.I)
_ID_TEXT = re.compile(r"\b(?:harness|bridge3w|net3w|sess|thread)[-_][\w.:-]{6,}|"
                      r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b|"
                      r"/(?:Users|home)/[^\s\"'\\]+|"
                      r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", re.I)


def scrub(obj: Any) -> Any:
    """No session or thread id, local path or email address in a committed fixture."""
    if isinstance(obj, dict):
        return {k: ("<scrubbed>" if _ID_KEYS.search(str(k)) and isinstance(v, (str, int))
                    else scrub(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    if isinstance(obj, str):
        if obj[:1] in "{[" and len(obj) > 2:
            try:
                return json.dumps(scrub(json.loads(obj)), default=str)
            except ValueError:
                pass
        return _ID_TEXT.sub("<scrubbed>", obj)
    return obj


def load_fixtures(path: Path) -> List[BenchTurn]:
    """The committed turns, re-scored by this checkout's scorer, so CI sees a scorer change too."""
    turns = []
    for line in path.read_text().splitlines():
        if not line:
            continue
        t = BenchTurn(**json.loads(line))
        t.rescored = rescore({"task": t.task, "expected": t.expected, "answer": t.answer,
                              "hand_scored": t.hand_scored})
        turns.append(t)
    return turns


def make_fixtures(root: Path, ids: List[str], out: Path) -> None:
    want = set(ids)
    rows = [t for t in iter_archive(root) if t.id in want]
    missing = want - {t.id for t in rows}
    if missing:
        sys.exit(f"not in the archive: {sorted(missing)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for t in rows:
            d = asdict(t)
            d["state"] = scrub(d["state"])
            d["answer"], d["query"] = scrub(d["answer"]), scrub(d["query"])
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} turns to {out} ({out.stat().st_size // 1024} KiB)")


# --------------------------------------------------------------------------- main

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("archive", nargs="?",
                    help="default: $GIS_HARNESS_ARCHIVE, else gis_harness_runs_archive/ in the "
                         "main checkout")
    ap.add_argument("--fixtures", action="store_true",
                    help="run on the committed fixture subset instead of an archive")
    ap.add_argument("--out", help="folder for report.json and report.md (default: "
                                  "<archive>/validators_bench; none with --fixtures)")
    ap.add_argument("--baseline", help="a previous report.json to print the delta against")
    ap.add_argument("--check", action="store_true",
                    help="with --fixtures: fail if the report differs from expected.md")
    ap.add_argument("--update-expected", action="store_true",
                    help="with --fixtures: write the report as the new expected.md")
    ap.add_argument("--reexec", action="store_true",
                    help="re-run each turn's kriging/esda code under the current gate (docker, "
                         "no network) so the in-sandbox reference check is measured too")
    ap.add_argument("--data", help="where the tasks' input files are built (default: "
                                   "<out>/_data, a temporary folder with --fixtures)")
    ap.add_argument("--make-fixtures", nargs="+", metavar="RUN/MODEL/TASK",
                    help="write these archived turns, scrubbed, as the fixture subset")
    a = ap.parse_args(argv)
    os.environ.setdefault("AGENT_EXTRACTION", "1")      # the VM's setting, which the harness copies

    archive = Path(a.archive) if a.archive else default_archive()
    if a.make_fixtures:
        make_fixtures(archive, a.make_fixtures, FIXTURES / "turns.jsonl")
        return 0
    if a.fixtures:
        turns = load_fixtures(FIXTURES / "turns.jsonl")
        title, out = "Validator benchmark (fixture subset)", Path(a.out) if a.out else None
    else:
        root = archive
        if not root.is_dir():
            sys.exit(f"no archive at {root}")
        turns = list(iter_archive(root))
        title, out = "Validator benchmark", Path(a.out) if a.out else root / "validators_bench"
    global DATA_ROOT, ARCHIVE_ROOT
    if a.data:
        DATA_ROOT = Path(a.data)
    elif out is not None:
        DATA_ROOT = out / "_data"
    else:
        import tempfile

        DATA_ROOT = Path(tempfile.mkdtemp(prefix="validators_bench_data_"))
    ARCHIVE_ROOT = None if a.fixtures else archive
    if a.reexec:
        from gis_harness.reference_reexec import augment_state

        totals = collections.Counter()
        for t in turns:
            if t.state:
                t.state, info = augment_state(t.state, t.task, DATA_ROOT)
                totals.update(info)
        print(f"re-executed {totals['rerun']} runs ({totals['failed']} failed); "
              f"{totals['recorded']} wrote a reference record")
    rep = evaluate(turns)
    md = markdown(rep, title=title)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
        (out / "report.md").write_text(md)
        print(f"wrote {out / 'report.json'} and {out / 'report.md'}")
    print(md)
    if a.baseline:
        print(delta(rep, json.loads(Path(a.baseline).read_text())))
    if a.update_expected:
        (FIXTURES / "expected.md").write_text(md)
        print(f"wrote {FIXTURES / 'expected.md'}")
    elif a.check:
        exp = FIXTURES / "expected.md"
        expected = exp.read_text() if exp.exists() else ""
        if md != expected:
            import difflib

            sys.stdout.writelines(difflib.unified_diff(
                expected.splitlines(True), md.splitlines(True), "expected.md", "now"))
            print("\nThe checks' precision or recall on the fixtures changed. If the change is "
                  "intended, commit the new numbers with it:\n  python3 -m "
                  "gis_harness.validators_bench --fixtures --update-expected")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
