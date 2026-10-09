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
"""
from __future__ import annotations

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


def render(answer: str, findings: List[Finding]) -> str:
    """The answer with at most one banner, listing every finding once, problems first."""
    if not findings:
        return answer
    seen, items = set(), []
    for f in sorted(findings, key=lambda f: _ORDER.get(f.kind, 3)):
        key = (f.kind, f.message)
        if key in seen:
            continue
        seen.add(key)
        items.append(f)
    st = status(items)
    head = {"problem": "⚠️ **Check this answer.** A check found a problem:",
            "unverified": ("ℹ️ **Not everything here could be checked.** Unchecked is not the "
                           "same as wrong:"),
            "verified": "ℹ️ **Note:**"}[st]
    lines = [head]
    for f in items[:MAX_LINES]:
        tag = {PROBLEM: "Problem", UNVERIFIABLE: "Not checked", NOTE: "Note"}.get(f.kind, "Note")
        lines.append(f"- {tag}: {f.message}")
    if len(items) > MAX_LINES:
        lines.append(f"- …and {len(items) - MAX_LINES} more")
    body = (answer or "").rstrip()
    return f"{body}\n\n---\n\n" + "\n".join(lines) if body else "\n".join(lines)


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
