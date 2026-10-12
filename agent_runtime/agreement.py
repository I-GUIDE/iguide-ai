"""Two runs in one turn that compute the same figure must agree (plan step 4a).

The miss this exists for. T10 (ordinary kriging of the Meuse zinc samples), stack-live-cases run:
the first code run declared `kriging_zinc = 309.531 ppm`, the reference value. A later run, which
put the nugget on the diagonal of the kriging matrix, declared `krige_zinc = 295.06 ppm`. The
answer reported 295.1 ppm and nothing in the turn said that an earlier run had computed something
else. Every check that ran looked at one run at a time, and each run was internally consistent.

A disagreement between two of the turn's own runs is evidence the turn itself produced: no
reference value and no model judgement is needed to see that 309.531 and 295.06 cannot both be
the kriged zinc at one point.

**What counts as "the same figure".** Only typed facts (stage 43, "A number carries its unit and
where it was measured"; stage 48, "Measured numbers reach the gate without being declared") and
only when their kind is known on both sides:

* the names agree word for word, after spelling is folded: `kriging_zinc` and `krige_zinc`,
  `kriging_logzinc` and `krige_logz`. A name made only of generic words (`total_area`,
  `value`, `result`) names no kind, so it is never compared;
* the units have the same dimension (or neither run gave one) and the values are converted
  before comparing;
* the two values come from different calls, and the calls did not read disjoint inputs.

A measurement the sandbox recorded on its own (`area sum`, `length max`) is named after the
operation, not the thing measured, so it is never compared either. Anything else stays silent.

**What happens.** Before the answer is written the writer is told which figures disagree and is
asked to say which is right (`reconcile_note`). After it is written, an answer that states one
side of a disagreement without the other gets a finding (`findings`). An answer that states
both has told the reader; one that states neither does not use the figure.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from agent_runtime.units import convert, parse_unit

# A difference this large, relative to the larger value, is a disagreement. Rounding is allowed
# for separately: "397.4" and "397.441" agree.
REL_TOL = 0.005
CHECK = "agreement"
# What a finding is. A `problem` prints a banner, and the plan admits a check to the banner only
# when, measured over the scored archive, the answers it marks are wrong at least 80% of the time.
# Until that measurement is recorded here, it is a `note` (in the verdict, not printed).
KIND = "note"

# Words that say how a figure was produced or summarised, not what it is. A name made only of
# these (and stop words) names no kind: two runs' `total_area` may be two different areas.
_GENERIC = frozenset("""
    value values result results output outputs total sum mean avg average median min max minimum
    maximum count number num n estimate estimated est final answer computed calc calculated
    area length distance perimeter size amount score stat stats statistic metric measure
    measured new old tmp temp test check res out x y z val var data first second third
    """.split())
_STOP = frozenset("of the a an in for with by to and at on per from is as via using".split())
# A permutation result is not one figure: a p-value floors at 1/(permutations+1) and moves with
# the seed, so 0.001 and 0.0001 from 999 and 9,999 shuffles are both right (T09L, threeway-net).
_PERMUTATION = frozenset("p pval pvalue psim sim zsim".split())
_WORD_RE = re.compile(r"[A-Za-z][a-z]*|[A-Z]+(?![a-z])|\d+")


@dataclass
class Figure:
    name: str
    value: float
    unit: Optional[str]
    call_id: Optional[str]
    order: int                       # the run's position in the turn, from 1
    inputs: Tuple[str, ...] = ()

    def shown(self) -> str:
        v = f"{self.value:,.6g}"
        return f"{v} {self.unit}".strip() if self.unit else v


@dataclass
class Disagreement:
    figures: List[Figure] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.figures[0].name

    def describe(self) -> str:
        parts = [f"`{f.name}` = {f.shown()} (run {f.order})" for f in self.figures]
        return " vs ".join(parts)


# --------------------------------------------------------------------------- names

def _tokens(name: str) -> List[str]:
    words = [w.lower() for w in _WORD_RE.findall(name or "")]
    return [w for w in words if not w.isdigit() and w not in _STOP]


def _same_word(a: str, b: str) -> bool:
    """`kriging`/`krige`, `logzinc`/`logz`, `hotspots`/`hotspot`; not `distance`/`district`."""
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    if len(short) >= 3 and long_.startswith(short):
        return True
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    return common >= max(4, len(short) - 1)


def _is_unit_word(token: str) -> bool:
    try:
        return parse_unit(token).kind == "unit"
    except Exception:  # noqa: BLE001
        return False


def kind_key(name: str) -> Optional[List[str]]:
    """The words that say what a figure IS, or None when the name says nothing about that. A
    unit word says how it is measured (`area_km2`, `distance_m`), not what it is."""
    toks = [t for t in dict.fromkeys(_tokens(name)) if not _is_unit_word(t)]
    if not toks or all(t in _GENERIC for t in toks):
        return None
    return toks


def same_kind(a: Sequence[str], b: Sequence[str]) -> bool:
    """Word for word, in any order, every word matched once."""
    if len(a) != len(b):
        return False
    pool = list(b)
    for w in a:
        hit = next((i for i, v in enumerate(pool) if _same_word(w, v)), None)
        if hit is None:
            return False
        pool.pop(hit)
    return True


def _printed_label(line: str) -> str:
    """A printed figure's name is the text in front of its number: "Kriging zinc (exp
    back-transform): 309.531 ppm" is called "Kriging zinc (exp back-transform)"."""
    m = re.search(r"[-−]?\d", line or "")
    return (line[:m.start()] if m else line or "").strip(" :=\t-")


# --------------------------------------------------------------------------- the turn's figures

def _inputs_of(args: Any) -> Tuple[str, ...]:
    if not isinstance(args, dict):
        return ()
    found: List[str] = []
    for k in ("input_files", "file_id", "file_ids", "path", "dem_file_id", "raster_file_id"):
        v = args.get(k)
        for item in (v if isinstance(v, (list, tuple)) else [v]):
            if isinstance(item, str) and item:
                found.append(item)
    return tuple(sorted(set(found)))


def figures(facts: Iterable[Dict[str, Any]],
            calls: Iterable[Dict[str, Any]] = ()) -> List[Figure]:
    """The typed facts whose kind is known, one per (call, name)."""
    args_by_call = {c.get("call_id"): c.get("args") for c in calls or [] if c.get("call_id")}
    order: Dict[Any, int] = {}
    out: List[Figure] = []
    seen: Set[Tuple[Any, str]] = set()
    for f in facts or []:
        if not isinstance(f, dict):
            continue
        if f.get("source") == "measured":
            continue                       # named after the operation, not the thing
        name = str(f.get("name") or "")
        if f.get("source") == "printed":
            name = _printed_label(name)
        if kind_key(name) is None or _PERMUTATION & set(_tokens(name)) \
                or "p-value" in str(f.get("unit") or "").lower():
            continue
        try:
            value = float(f.get("value"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        cid = f.get("call_id") or f.get("result_seq")
        if cid not in order:
            order[cid] = len(order) + 1
        key = (cid, name.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(Figure(name=name, value=value, unit=(f.get("unit") or None),
                          call_id=f.get("call_id"), order=order[cid],
                          inputs=_inputs_of(args_by_call.get(f.get("call_id")))))
    return out


def _comparable_units(a: Optional[str], b: Optional[str]) -> Optional[float]:
    """The factor from b's unit to a's, 1.0 when neither has one, None when not comparable."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return None
    if a.strip().lower() == b.strip().lower():
        return 1.0
    pa, pb = parse_unit(a), parse_unit(b)
    if pa.kind != "unit" or pb.kind != "unit" or pa.dimension != pb.dimension:
        return None
    k = convert(1.0, b, a)
    return k if k is not None and math.isfinite(k) and k != 0 else None


def _decimals(v: float) -> int:
    text = repr(float(v))
    if "e" in text or "E" in text:
        return 0
    frac = text.split(".")[1] if "." in text else ""
    return 0 if frac == "0" else len(frac)


def differ(a: Figure, b: Figure) -> Optional[bool]:
    """True when the two values disagree, False when they agree, None when not comparable."""
    k = _comparable_units(a.unit, b.unit)
    if k is None:
        return None
    bv = b.value * k
    rounding = 0.5 * 10 ** (-min(_decimals(a.value), _decimals(b.value)))
    tol = max(REL_TOL * max(abs(a.value), abs(bv)), rounding, 1e-12)
    return abs(a.value - bv) > tol


def disagreements(figs: Sequence[Figure]) -> List[Disagreement]:
    """Groups of the same figure, from different runs, whose values disagree."""
    out: List[Disagreement] = []
    used: Set[int] = set()
    for i, a in enumerate(figs):
        if i in used:
            continue
        ka = kind_key(a.name)
        group = [a]
        for j in range(i + 1, len(figs)):
            b = figs[j]
            if j in used or b.call_id == a.call_id and b.order == a.order:
                continue
            if a.inputs and b.inputs and not set(a.inputs) & set(b.inputs):
                continue                   # two runs on different data: two different things
            kb = kind_key(b.name)
            if kb is None or not same_kind(ka, kb):
                continue
            if _comparable_units(a.unit, b.unit) is None:
                continue
            group.append(b)
            used.add(j)
        runs = {g.order for g in group}
        if len(runs) < 2:
            continue
        if any(differ(x, y) for x in group for y in group if x.order != y.order):
            out.append(Disagreement(figures=group))
    return out


def for_log(log: Any) -> List[Disagreement]:
    if log is None:
        return []
    try:
        return disagreements(figures(log.facts(), log.calls()))
    except Exception:  # noqa: BLE001 - a check must never break the turn
        return []


# --------------------------------------------------------------------------- what the turn sees

def reconcile_note(found: Sequence[Disagreement]) -> str:
    """What the answer's writer is told before writing."""
    if not found:
        return ""
    lines = ["Runs this turn computed the same figure and DISAGREE:"]
    for d in found[:6]:
        lines.append(f"- {d.describe()}")
    lines.append("Do not report one of these values silently. Say which run is right and why "
                 "(what differs between the runs: data, parameters, formula), or report both "
                 "and say the turn could not settle it.")
    return "\n".join(lines)


def _stated(answer: str, fig: Figure) -> bool:
    from agent_runtime import facts as F

    target = F.Fact(id="agreement", value=fig.value, unit=fig.unit, source="agreement")
    return any(F.is_claim(q) and F._matches(q, target) for q in F.quantities(answer or ""))


def _support(d: Disagreement, fig: Figure) -> Set[int]:
    """The runs that computed *fig*'s value."""
    return {g.order for g in d.figures if differ(g, fig) is False}


def _settled(d: Disagreement, fig: Figure) -> bool:
    """Whether the turn's own runs settled on *fig*: more runs computed its value than any value
    it disagrees with. A zero does not compete with non-zero values: it is how a failed attempt
    (an empty selection, a raster read in the wrong unit) looks, and in the archive's correct
    turns the early zeros were exactly that (T06 slope 0, T07L area 0, before the runs that
    worked)."""
    mine = _support(d, fig)
    rivals = [g for g in d.figures if differ(g, fig)]
    if fig.value != 0:
        rivals = [g for g in rivals if g.value != 0]
    return all(len(mine) > len(_support(d, g)) for g in rivals)


def findings(answer: str, found: Sequence[Disagreement], *, kind: str) -> List[Any]:
    """A finding for each disagreement the answer reports one side of, when the turn's runs did
    not settle on that side."""
    from agent_runtime import verdict as V

    out = []
    for d in found:
        stated = [f for f in d.figures if _stated(answer, f)]
        if not stated:
            continue                       # the answer does not use this figure
        values = {round(f.value, 9) for f in d.figures}
        if {round(f.value, 9) for f in stated} == values:
            continue                       # it reports every value; the reader has been told
        if any(_settled(d, f) for f in stated):
            continue                       # the runs agree on what the answer reports
        others = [f for f in d.figures if differ(f, stated[0])]
        out.append(V.Finding(
            CHECK, kind,
            f"two runs in this turn computed {d.label} and disagree "
            f"({d.describe()}); the answer reports {stated[0].shown()} and does not mention "
            f"{', '.join(o.shown() for o in others)}",
            evidence=[str(f.call_id) for f in d.figures if f.call_id][:6]))
    return out
