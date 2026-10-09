"""One statement about an answer's reliability, from every check that ran.

Before stage 44, thirteen post-processing steps each appended their own banner behind their own
`---`, none of them seeing the others' verdicts (docs/design-review-2026-10.md, flaw 7):
- the gate's ⛔/⚠️/ℹ️ headline;
- ⚠️ Grounding check;
- three kinds of ⚠️ Correction;
- ⚠️ Partial answer;
- the routing note.

"Could not check" was loud in one producer (the gate's COULD NOT VERIFY) and silent in another
(a failed audit returned severity "unknown" and raised nothing). Every pair of producers was a
possible contradiction, and two correct answers on 2026-10-08 showed several at once.

Every check now contributes `Finding`s, and `render` maps all of them to one banner. The rules:

* a **problem** (a wrong number, a failed run the answer still depends on, a false claim about
  the map) is reported with its evidence;
* an **unverifiable** item says what was not checked and why. It is never called a detection;
* a **note** is information, not a warning;
* an answer whose checks all ran and passed gets no banner at all.

**What reaches the answer text (stage 46).** Only what changes how the reader should treat the
answer: a problem, or a peer that failed (the answer is incomplete). A statement the number scan
cut gets one quiet line, because the reader is owed knowing that text was removed. Everything
else, every "not checked" and every note, stays in the verdict the client and the trace receive
(`findings`, `status`) and no longer prints as a banner.

The measurement behind it. Across 437 scored turns in the harness archive (every phase, both
models, plus the live browser sweep), a banner landed on 98 correct answers and 9 wrong ones, so
it marked a wrong answer 8% of the time, and it caught 9 of the 39 wrong answers. By kind, in the
stack's own runs (p4-gate, p5-gate, p5-fixed-gate): the gate's "could not check" (coverage,
nothing applicable) was on 22 correct answers and 0 wrong; the number scan's unresolved figures
on 3 and 0. A banner that says nothing about correctness most of the time teaches the reader to
skip it, and then the rare real problem is skipped too.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

PROBLEM, UNVERIFIABLE, NOTE = "problem", "unverifiable", "note"
# The banner lists this many findings and counts the rest: a wall of findings is not read.
MAX_LINES = 6
_ORDER = {PROBLEM: 0, UNVERIFIABLE: 1, NOTE: 2}


@dataclass
class Finding:
    check: str                      # which check: gate, number_scan, audit, peer, correction, ...
    kind: str                       # problem | unverifiable | note
    message: str
    evidence: List[str] = field(default_factory=list)   # fact ids, call ids, quoted text

    def to_dict(self) -> Dict[str, Any]:
        return {"check": self.check, "kind": self.kind, "message": self.message,
                "evidence": self.evidence}


def status(findings: List[Finding]) -> str:
    """verified | unverified | problem"""
    kinds = {f.kind for f in findings}
    if PROBLEM in kinds:
        return "problem"
    if UNVERIFIABLE in kinds:
        return "unverified"
    return "verified"


def shown(f: Finding) -> bool:
    """Whether a finding is printed as the banner: a problem, or a failed peer (an incomplete
    answer). Unverifiable items and notes stay in the verdict payload only."""
    return f.kind == PROBLEM or f.check == "peer"


def _quiet(f: Finding) -> bool:
    """The number scan's cut: one plain line, not a banner. Text was removed and the reader is
    owed knowing that, but it is not a warning about what remains."""
    return f.check == "number_scan" and f.kind == NOTE


def render(answer: str, findings: List[Finding]) -> str:
    """The answer with at most one banner, listing problems and peer failures once, problems
    first, then the quiet line for any cut. Nothing else is printed (see the module docstring)."""
    seen, items = set(), []
    for f in sorted(findings or [], key=lambda f: _ORDER.get(f.kind, 3)):
        key = (f.check, f.kind, f.message)
        if key in seen:
            continue
        seen.add(key)
        items.append(f)
    loud = [f for f in items if shown(f)]
    quiet = [f for f in items if _quiet(f)]
    if not loud and not quiet:
        return answer
    lines: List[str] = []
    if loud:
        if any(f.kind == PROBLEM for f in loud):
            lines.append("⚠️ **Check this answer.** A check found a problem:")
        else:
            lines.append("ℹ️ **This answer may be incomplete.**")
        for f in loud[:MAX_LINES]:
            tag = "Problem" if f.kind == PROBLEM else "Incomplete"
            lines.append(f"- {tag}: {f.message}")
        if len(loud) > MAX_LINES:
            lines.append(f"- …and {len(loud) - MAX_LINES} more")
    for f in quiet:
        lines.append(("\n" if loud else "") + f"_{_cut_sentence(f)}_")
    body = (answer or "").rstrip()
    return f"{body}\n\n---\n\n" + "\n".join(lines) if body else "\n".join(lines)


def _cut_sentence(f: Finding) -> str:
    m = re.match(r"\s*(\d+)", f.message or "")      # the producer writes the full count first;
    n = int(m.group(1)) if m else (len(f.evidence) or 1)   # evidence is capped at six
    return ("One statement with a figure no tool here can produce was left out."
            if n == 1 else
            f"{n} statements with figures no tool here can produce were left out.")


# --------------------------------------------------------------------------- producers

def from_gate(audit: Optional[Dict[str, Any]]) -> List[Finding]:
    """The invariant gate's verdict (already scoped to the runs the answer uses)."""
    out: List[Finding] = []
    verdict = str((audit or {}).get("invariant_gate") or "")
    for issue in (audit or {}).get("issues") or []:
        if not (isinstance(issue, dict) and issue.get("source") == "invariant_gate"):
            continue
        if issue.get("check") == "not_applicable":
            # Nothing in the run was geospatial work, so the gate's checks had nothing to apply
            # to. That is not "could not check": the numbers are still held to the record by the
            # number scan. A run whose geospatial work the gate could not reach ("coverage") is.
            continue
        status_ = issue.get("status") or ("fail" if verdict == "fail" else "cannot_determine")
        reason = str(issue.get("reason") or "").replace("invariant gate ", "the code check ")
        target = str(issue.get("claim") or "").replace("computed value from ", "")
        if status_ == "fail":
            out.append(Finding("gate", PROBLEM, f"{reason} ({target})"))
        elif issue.get("advisory"):
            out.append(Finding("gate", NOTE, f"{reason} ({target})"))
        else:
            out.append(Finding("gate", UNVERIFIABLE, f"{reason} ({target})"))
    return out


def from_audit(audit: Optional[Dict[str, Any]], *, numeric_claims_resolved: bool) -> List[Finding]:
    """The LLM audit, for what only it can judge: claims that are not numbers.

    A claim that states a number is the number scan's, which is deterministic; the audit's
    sampled verdict on it is not used. A failed audit is reported as not checked, never as
    clean.
    """
    out: List[Finding] = []
    severity = str((audit or {}).get("severity") or "").lower()
    if severity == "unknown":
        out.append(Finding("audit", UNVERIFIABLE,
                           "the qualitative claims were not checked: the grounding audit did not "
                           "run to completion"))
        return out
    if severity != "high":
        return out
    from agent_runtime.facts import quantities, is_claim

    issues = [i for i in (audit or {}).get("issues") or []
              if not (isinstance(i, dict) and i.get("source") == "invariant_gate")]
    if not issues and not (audit or {}).get("invariant_gate"):
        summary = str((audit or {}).get("summary") or "").strip()
        out.append(Finding("audit", PROBLEM, summary or "the grounding audit found statements "
                                                        "the turn's record does not support"))
        return out
    for issue in (audit or {}).get("issues") or []:
        if isinstance(issue, dict) and issue.get("source") == "invariant_gate":
            continue
        claim = str(issue.get("claim") if isinstance(issue, dict) else issue or "").strip()
        if not claim:
            continue
        if any(is_claim(q) for q in quantities(claim)):
            continue                  # the number scan decides numeric claims
        out.append(Finding("audit", PROBLEM,
                           f"this statement is not supported by what the turn retrieved or "
                           f"computed: \"{claim[:200]}\""))
    return out


def from_peer_failures(failures: List[Dict[str, Any]]) -> List[Finding]:
    names = []
    for f in failures:
        n = str(f.get("peer") or "a step")
        if n not in names:
            names.append(n)
    if not names:
        return []
    label = {"search": "search", "analyze": "analysis", "code": "code execution",
             "synthesize": "answer composition"}
    pretty = ", ".join(label.get(n, n) for n in names)
    return [Finding("peer", UNVERIFIABLE,
                    f"{pretty} failed during this turn, so this reply is based on what completed "
                    f"before the failure; re-running may produce a fuller answer")]
