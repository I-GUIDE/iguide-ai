"""The turn's facts, and a deterministic scan of the numbers an answer states.

Before stage 44, whether a number in the answer was supported was decided three ways:
- an LLM audit that quoted "verbatim" spans nobody checked;
- reconciliation rule (2), a substring match restricted to numbers of three or more digits,
  so `21.5` was never seen and `997.93` was matched as `997`;
- stage 39's scan for travel figures, one claim class.

Each covered the class it was written for (docs/design-review-2026-10.md, flaw 2). This module
is the general version. Every number the answer states is resolved against what the turn
recorded:
- the typed facts tools reported (stage 43);
- every number in a tool result;
- the user's own question;
- the retrieved evidence;
- earlier turns' records.

A number resolves when a recorded number equals it within the precision the answer shows,
after converting units with the unit library. Its fact id is the link.

Precision is read from how the number is written: "343.9" is right for 343.89; "about 340" is
right for 343.7, because trailing zeros of a whole number are rounding; "2,585" is right for
2,584.62. A number that resolves nowhere is a finding. What happens to it is the verdict's
business (agent_runtime/verdict.py), not this module's.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from agent_runtime.units import convert, parse_unit

# A number, with thousands separators, and the unit-ish token right after it.
_NUM_RE = re.compile(r"(?<![\w.\-/])(-|−)?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\d])")
_UNIT_AFTER_RE = re.compile(
    r"[\s-]?((?:square |sq\.? ?|cubic )?[A-Za-z°µ%][A-Za-z0-9°²³µ%/^.\-]*(?:\s?(?:per|/)\s?[A-Za-z0-9,]+)?)")
_FENCE_RE = re.compile(r"```.*?```", re.S)
_LINK_TARGET_RE = re.compile(r"\]\([^)]*\)|https?://\S+|\bfile_[0-9a-f]{6,}\b|`[^`]*`")
_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+[.)]|[-*+])\s", re.M)
_HEADING_RE = re.compile(r"^\s*#+.*$", re.M)
# A number that NAMES something rather than measures it: right after an all-caps label
# ("EPSG:26916", "GEOID 17019", "UTM 16N", "FIPS 17"). Grammar, not a list of labels.
_IDENTIFIER_BEFORE_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,}[:\s#-]\s?$")


@dataclass
class Fact:
    id: str
    value: float
    unit: Optional[str] = None
    label: str = ""
    source: str = ""               # the tool, "query", "evidence", "earlier turn"
    call_id: Optional[str] = None
    measured_in_crs: Optional[str] = None
    typed: bool = False


@dataclass
class Quantity:
    text: str
    value: float
    unit: Optional[str]
    decimals: int                  # digits after the point; negative = trailing zeros of a whole
    sentence: str
    start: int
    identifier: bool = False       # names something (EPSG:26916, GEOID 17019), not a measure
    clause: str = ""               # the smallest piece of the sentence that holds it


@dataclass
class Resolution:
    quantity: Quantity
    fact: Optional[Fact] = None

    @property
    def resolved(self) -> bool:
        return self.fact is not None


@dataclass
class FactSet:
    facts: List[Fact] = field(default_factory=list)

    def add(self, value: float, **kw: Any) -> Fact:
        f = Fact(id=f"F{len(self.facts) + 1}", value=float(value), **kw)
        self.facts.append(f)
        return f

    def typed(self) -> List[Fact]:
        return [f for f in self.facts if f.typed]

    def render(self, limit: int = 40) -> str:
        """The facts a synthesizer is shown, typed first."""
        rows = []
        for f in (self.typed() or [])[:limit]:
            where = f" (measured in {f.measured_in_crs})" if f.measured_in_crs else ""
            rows.append(f"[{f.id}] {f.label} = {_fmt(f.value)} {f.unit or ''}{where} "
                        f"— {f.source}".rstrip())
        return "\n".join(rows)


def _fmt(v: float) -> str:
    return f"{v:,.6g}" if abs(v) < 1e15 else str(v)


# --------------------------------------------------------------------------- building

def _numbers_in(text: str) -> Iterable[Tuple[float, Optional[str]]]:
    for m in _NUM_RE.finditer(text or ""):
        raw = (m.group(2) or "").replace(",", "") + (m.group(3) or "")
        try:
            value = float(raw) * (-1 if m.group(1) else 1)
        except ValueError:
            continue
        unit = None
        um = _UNIT_AFTER_RE.match(text, m.end())
        if um:
            candidate = um.group(1).rstrip(".,;:")
            if parse_unit(candidate).kind == "unit":
                unit = candidate
        yield value, unit


def _walk_strings(node: Any, out: List[str], depth: int = 0) -> None:
    if depth > 12:
        return
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append(f"{k} {v}")
            else:
                _walk_strings(v, out, depth + 1)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _walk_strings(v, out, depth + 1)
    elif isinstance(node, str):
        s = node.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                _walk_strings(json.loads(s), out, depth + 1)
                return
            except ValueError:
                pass
        if s.startswith("content='") or s.startswith('content="'):
            # A ToolMessage's repr: the content is a Python string literal. Read it as one
            # (unicode_escape would garble "km²").
            import ast

            quote = s[8]
            end = s.find(f"{quote} name=", 9)
            literal = s[8:end + 1] if end > 0 else s[8:]
            try:
                inner = ast.literal_eval(literal)
                try:
                    _walk_strings(json.loads(inner), out, depth + 1)
                except ValueError:
                    out.append(inner)
                return
            except Exception:  # noqa: BLE001
                pass
        out.append(s)


def build(*, log: Any = None, results: Sequence[Dict[str, Any]] = (), query: str = "",
          evidence: Sequence[Any] = (), prior_rows: Sequence[Any] = ()) -> FactSet:
    """Every number the turn recorded. Typed facts carry a unit and a source; the rest carry
    the unit written beside them, if any."""
    fs = FactSet()
    for f in (log.facts() if log is not None else []):
        try:
            fs.add(float(f["value"]), unit=f.get("unit"), label=str(f.get("name") or ""),
                   source=str(f.get("tool") or f.get("source") or ""), call_id=f.get("call_id"),
                   measured_in_crs=f.get("measured_in_crs"), typed=True)
        except (TypeError, ValueError):
            continue
    for res in results:
        if not isinstance(res, dict):
            continue
        strings: List[str] = []
        _walk_strings(res.get("content"), strings)
        for s in strings:
            for value, unit in _numbers_in(s):
                fs.add(value, unit=unit, source=str(res.get("name") or "tool"),
                       call_id=res.get("tool_call_id"), label=s[:60])
    for value, unit in _numbers_in(query):
        fs.add(value, unit=unit, source="query")
    for doc in evidence or []:
        text = json.dumps(doc, default=str) if not isinstance(doc, str) else doc
        for value, unit in _numbers_in(text[:20000]):
            fs.add(value, unit=unit, source="evidence")
    for row in prior_rows or []:
        for value, unit in _numbers_in(json.dumps(row, default=str)):
            fs.add(value, unit=unit, source="earlier turn")
    return fs


# --------------------------------------------------------------------------- scanning

def quantities(answer: str) -> List[Quantity]:
    """The numbers an answer STATES. Not those inside code, links, file ids, list markers or
    headings: those are not claims."""
    text = answer or ""
    mask = list(text)
    for rx in (_FENCE_RE, _LINK_TARGET_RE, _LIST_MARKER_RE, _HEADING_RE):
        for m in rx.finditer(text):
            for i in range(m.start(), m.end()):
                mask[i] = " "
    masked = "".join(mask)
    out: List[Quantity] = []
    for m in _NUM_RE.finditer(masked):
        whole, frac = m.group(2) or "", m.group(3) or ""
        raw = whole.replace(",", "") + frac
        try:
            value = float(raw) * (-1 if m.group(1) else 1)
        except ValueError:
            continue
        if frac:
            decimals = len(frac) - 1
        else:
            digits = whole.replace(",", "")
            decimals = -(len(digits) - len(digits.rstrip("0"))) if digits != "0" else 0
        unit = None
        um = _UNIT_AFTER_RE.match(masked, m.end())
        if um:
            candidate = um.group(1).rstrip(".,;:")
            if parse_unit(candidate).kind == "unit":
                unit = candidate
        s0 = max(masked.rfind(".", 0, m.start()), masked.rfind("\n", 0, m.start())) + 1
        identifier = bool(_IDENTIFIER_BEFORE_RE.search(masked[max(0, m.start() - 12):m.start()]))
        e1 = min([i for i in (masked.find(". ", m.end()), masked.find("\n", m.end())) if i >= 0]
                 or [len(masked)])
        # The clause: inside parentheses, the ';'-separated segment; otherwise the sentence.
        c0 = max(masked.rfind("(", s0, m.start()), masked.rfind(";", s0, m.start()))
        c1_candidates = [i for i in (masked.find(";", m.end(), e1), masked.find(")", m.end(), e1))
                         if i >= 0]
        if c0 >= 0 and c1_candidates:
            clause = text[c0 + 1:min(c1_candidates)].strip()
        else:
            clause = text[s0:e1].strip()
        out.append(Quantity(text=text[m.start():(um.end() if um and unit else m.end())],
                            value=value, unit=unit, decimals=decimals,
                            sentence=text[s0:e1].strip(), start=m.start(),
                            identifier=identifier, clause=clause))
    return out


def _tolerance(q: Quantity, target: float) -> float:
    shown = 0.5 * 10 ** (-q.decimals)
    return max(shown, 1e-9 * abs(target))


def _matches(q: Quantity, f: Fact) -> bool:
    candidates = [f.value]
    if q.unit and not f.unit:
        # A figure with a unit against a bare recorded number: small numbers coincide
        # ("a 5-hour drive" against `"n": 5`), so a bare number grounds a unit-bearing figure
        # only when it is distinctive (>= 100) or its own context names that dimension.
        dim = parse_unit(q.unit).dimension
        named = any(parse_unit(w).dimension == dim for w in re.findall(r"[A-Za-z²]+", f.label)
                    if len(w) > 1 and parse_unit(w).kind == "unit")
        if abs(f.value) < 100 and not named:
            return False
    if q.unit and f.unit:
        converted = convert(f.value, f.unit, q.unit)
        if converted is None:
            pq, pf = parse_unit(q.unit), parse_unit(f.unit)
            if pq.dimension and pf.dimension and pq.dimension != pf.dimension:
                return False
        else:
            candidates = [converted]
    return any(abs(q.value - c) <= _tolerance(q, c) for c in candidates)


def resolve(answer: str, facts: FactSet) -> List[Resolution]:
    """Each stated number and the fact that supports it, typed facts first."""
    ordered = sorted(facts.facts, key=lambda f: (not f.typed,))
    out = []
    for q in quantities(answer):
        hit = next((f for f in ordered if _matches(q, f)), None)
        out.append(Resolution(quantity=q, fact=hit))
    return out


def is_claim(q: Quantity) -> bool:
    """Whether a stated number is a quantitative claim worth holding to the record: it has a unit,
    or it is not a small whole number. "2 schools", "step 3" and "1 of the 4" are not what this
    pass is for; a figure like 460 km or 12.5% is."""
    if q.unit:
        return True
    if q.identifier:
        return False
    whole = float(q.value).is_integer()
    if whole and 1800 <= q.value <= 2100 and "," not in q.text:
        return False                 # a year
    return not (whole and abs(q.value) < 10)
