"""Mechanistic scoring of one turn. No LLM judge.

Five questions per task, each answered from the answer text and the turn's own event stream:

  correct       every expected value appears in the answer, within tolerance
  productive    no tool call repeats an earlier identical call, and none failed
  clean         no warning banner on the answer (a banner on a CORRECT answer is a false alarm)
  sourced       the answer names where its data came from
  refused       (unsolvable tasks) the answer says it cannot, and states no value for it

A number matches if it is within tolerance of the expected value after converting its unit.
This is deliberately lenient about WHERE the number sits in the prose: the agent writes for a
person, not a parser, and a scorer that demands one format measures formatting. The cost of
that leniency is a wrong answer that happens to contain the right number elsewhere; with
expected values like 2,586.0 km^2 or 309.5 ppm, that is rare enough to accept, and every match
records the text it matched so a reader can check.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .tasks import Check, Task

# Units the SCORER recognises, by dimension, with the factor to the dimension's base unit.
# This is the answer key's vocabulary for twelve fixed tasks, not the agent's: the agent's own
# unit handling is under test, and none of this reaches it.
UNITS: Dict[str, Tuple[str, float]] = {
    "km": ("length", 1000.0), "m": ("length", 1.0), "mi": ("length", 1609.344),
    "km2": ("area", 1e6), "m2": ("area", 1.0), "ha": ("area", 1e4), "mi2": ("area", 2589988.110336),
    "min": ("time", 1.0), "h": ("time", 60.0), "s": ("time", 1 / 60),
    "deg": ("angle", 1.0),
    "ppm": ("conc", 1.0),
    "person_km": ("weighted_length", 1.0), "person_m": ("weighted_length", 0.001),
}

# Longest first, so "km²" is not read as "km".
_UNIT_WORDS: List[Tuple[str, str]] = [
    (r"person[- ]?(?:kilomet(?:re|er)s?|km)", "person_km"),
    (r"person[- ]?(?:met(?:re|er)s?|m)\b", "person_m"),
    (r"square kilomet(?:re|er)s?|sq\.? ?km|km²|km\^2|km2", "km2"),
    (r"square met(?:re|er)s?|sq\.? ?m\b|m²|m\^2|m2\b", "m2"),
    (r"square miles?|sq\.? ?mi(?:les?)?|mi²|mi2", "mi2"),
    (r"hectares?|\bha\b", "ha"),
    (r"kilomet(?:re|er)s?|\bkm\b", "km"),
    (r"miles?\b|\bmi\b", "mi"),
    (r"met(?:re|er)s?\b|\bm\b", "m"),
    (r"minutes?\b|\bmins?\b", "min"),
    (r"hours?\b|\bhrs?\b|\bh\b", "h"),
    (r"seconds?\b|\bs\b", "s"),
    (r"°|degrees?\b|\bdeg\b", "deg"),
    (r"ppm\b|mg/kg\b|mg kg-1", "ppm"),
]
_UNIT_RE = re.compile(r"\s*(" + "|".join(f"(?:{p})" for p, _ in _UNIT_WORDS) + ")", re.I)
_NUM_RE = re.compile(
    r"(?<![\w.])([-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[-−]?\.\d+)(?![\d])")

BANNER_RE = re.compile(r"^\s*(?:>\s*)?(?:[*_]{1,2})?\s*(?:⚠️?|ℹ️?)|COULD NOT VERIFY", re.M)
REFUSAL_RE = re.compile(
    r"\b(?:cannot|can ?not|can't|unable|not possible|isn't possible|impossible|no way to|"
    r"does(?: not|n't) (?:contain|include|have|provide)|do(?: not|n't) (?:contain|include|have)|"
    r"(?:is|are)(?: not|n't) (?:present|available|included|in the|on the|connected|part of|"
    r"reachable)|not (?:found|present|available)|lacks?|missing|only (?:one|a single) band|"
    r"no (?:near[- ]infrared|NIR|elevation|DEM|mercury|Hg|path|route|road|edge|data)|"
    r"unreachable|off the network|outside the network)\b", re.I)


@dataclass
class Number:
    value: float
    unit: Optional[str]
    text: str


def numbers(text: str) -> List[Number]:
    """Every number in the text, with the unit written right after it, if one is."""
    out: List[Number] = []
    for m in _NUM_RE.finditer(text):
        raw = m.group(1).replace("−", "-").replace(",", "")
        try:
            val = float(raw)
        except ValueError:
            continue
        unit = None
        um = _UNIT_RE.match(text, m.end())
        if um:
            word = um.group(1)
            for pat, key in _UNIT_WORDS:
                if re.fullmatch(pat, word, re.I):
                    unit = key
                    break
        out.append(Number(val, unit, text[m.start(): (um.end() if um else m.end())]))
    return out


def _clean(text: str) -> str:
    return (text or "").replace("**", "").replace("__", "").replace(" ", " ").replace(
        " ", " ")


def _within(v: float, e: float, rel: float, abs_: float) -> bool:
    return abs(v - e) <= max(abs_, rel * abs(e)) + 1e-12


def match_number(text: str, expected: float, unit: Optional[str], rel: float,
                 abs_: float) -> Optional[str]:
    for n in numbers(text):
        v = n.value
        if unit and n.unit and n.unit in UNITS and unit in UNITS:
            dim_e, f_e = UNITS[unit]
            dim_n, f_n = UNITS[n.unit]
            if dim_e != dim_n:
                continue
            v = v * f_n / f_e
        if _within(v, expected, rel, abs_):
            return n.text
    return None


def _text_variants(value: str) -> List[re.Pattern]:
    # "School 08" is also "School 8"; "Z2" is also "Zone 2".
    m = re.fullmatch(r"([A-Za-z ]*?)\s*0*(\d+)", value.strip())
    if not m:
        return [re.compile(rf"\b{re.escape(value)}\b", re.I)]
    prefix, num = m.group(1).strip(), m.group(2)
    pats = [rf"\b{re.escape(prefix)}\s*0*{num}\b"] if prefix else [rf"\b0*{num}\b"]
    if prefix.upper() == "Z":
        pats.append(rf"\bzone\s*0*{num}\b")
    return [re.compile(p, re.I) for p in pats]


def check_value(check: Check, expected: Dict[str, Any], answer: str) -> Dict[str, Any]:
    text = _clean(answer)
    exp = expected.get(check.name)
    found: Optional[str] = None
    if check.kind in ("number", "count"):
        found = match_number(text, float(exp), check.unit, check.rel_tol, check.abs_tol)
    elif check.kind == "text":
        for p in _text_variants(str(exp)):
            m = p.search(text)
            if m:
                found = m.group(0)
                break
    elif check.kind == "ids":
        hits = []
        for ident in exp:
            m = next((p.search(text) for p in _text_variants(ident) if p.search(text)), None)
            hits.append(m.group(0) if m else None)
        found = ", ".join(hits) if all(hits) else None
    elif check.kind == "point":
        pairs = [(exp, check.abs_tol)]
        for alt in check.alternatives:
            key, _, tol = alt.partition(":")
            pairs.append((expected[key], float(tol)))
        vals = [n for n in numbers(text)]
        for (a, b), tol in pairs:
            ha = next((n.text for n in vals if _within(abs(n.value), abs(a), 0, tol)), None)
            hb = next((n.text for n in vals if _within(abs(n.value), abs(b), 0, tol)), None)
            if ha and hb:
                found = f"{ha} / {hb}"
                break
    return {"name": check.name, "kind": check.kind, "expected": _jsonable(exp),
            "matched": found is not None, "found": found}


def _jsonable(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, tuple):
        return list(v)
    return v


# --------------------------------------------------------------------------- the trace


def _canonical_args(args: Any) -> str:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:  # noqa: BLE001
            return args.strip()
    try:
        return json.dumps(args, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001
        return str(args)


_FAILED_RE = re.compile(r'"ok"\s*:\s*false|\'ok\'\s*:\s*False|^\s*(?:content=[\'"])?Error\b|'
                        r'"error"\s*:\s*"(?!null)', re.I)


def tool_steps(calls: Sequence[Dict[str, Any]], results: Sequence[Dict[str, Any]],
               errors: Sequence[Dict[str, Any]] = ()) -> Dict[str, Any]:
    """Duplicate calls (same tool, same arguments as an earlier call) and failed calls."""
    seen: Dict[Tuple[str, str], int] = {}
    duplicates = []
    for i, c in enumerate(calls):
        key = (str(c.get("name")), _canonical_args(c.get("args")))
        if key in seen:
            duplicates.append({"index": i, "name": key[0], "first": seen[key]})
        else:
            seen[key] = i
    failed = [{"name": r.get("name"), "excerpt": str(r.get("content"))[:200]}
              for r in results if _FAILED_RE.search(str(r.get("content") or ""))]
    failed += [{"name": e.get("name") or e.get("tool_name"), "excerpt": str(e.get("message"))[:200]}
               for e in errors]
    return {"tool_calls": len(calls), "duplicates": duplicates, "failed": failed,
            "unproductive": len(duplicates) + len(failed)}


def banners(answer: str) -> List[str]:
    lines = []
    for m in BANNER_RE.finditer(answer or ""):
        start = answer.rfind("\n", 0, m.start()) + 1
        end = answer.find("\n", m.end())
        lines.append(answer[start: end if end >= 0 else None].strip()[:240])
    return lines


def source_named(answer: str, sources: Iterable[str]) -> Optional[str]:
    low = (answer or "").lower()
    return next((s for s in sources if s.lower() in low), None)


def refusal(answer: str, fabrication: Optional[str]) -> Dict[str, Any]:
    text = _clean(answer)
    said = REFUSAL_RE.search(text)
    fab = re.search(fabrication, text, re.I) if fabrication else None
    return {"refused": bool(said) and not fab,
            "refusal_text": said.group(0) if said else None,
            "fabricated": fab.group(0) if fab else None}


@dataclass
class TaskScore:
    task: str
    solvable: bool
    correct: Optional[bool]
    checks: List[Dict[str, Any]] = field(default_factory=list)
    steps: Dict[str, Any] = field(default_factory=dict)
    banners: List[str] = field(default_factory=list)
    source: Optional[str] = None
    refusal: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def productive(self) -> bool:
        return self.steps.get("unproductive", 0) == 0

    @property
    def strict(self) -> bool:
        """Everything right at once: the bar a user actually experiences."""
        ok = self.correct if self.solvable else bool(self.refusal and self.refusal["refused"])
        return bool(ok) and self.productive and not self.banners and self.source is not None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["productive"] = self.productive
        d["strict"] = self.strict
        return d


def score(task: Task, expected: Dict[str, Any], turn: Dict[str, Any]) -> TaskScore:
    answer = turn.get("answer") or ""
    s = TaskScore(task=task.id, solvable=task.solvable, correct=None, error=turn.get("error"))
    s.steps = tool_steps(turn.get("tool_calls", []), turn.get("tool_results", []),
                         turn.get("tool_errors", []))
    s.banners = banners(answer)
    s.source = source_named(answer, task.sources)
    if task.solvable:
        s.checks = [check_value(c, expected, answer) for c in task.checks]
        s.correct = bool(answer) and all(c["matched"] for c in s.checks)
    else:
        s.refusal = refusal(answer, task.fabrication)
    return s
