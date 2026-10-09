"""Units, parsed by a unit library instead of looked up in a list.

The invariant gate used to decide what a declared unit meant by looking it up in two
hand-kept vocabularies (`_UNIT_ALIASES`, 113 spellings; `_KNOWN_UNITS`, 25) that disagreed
on 9 tokens. Every spelling a model chose that the lists lacked (`km²`, `records`, `points`,
`square_miles`, `schools`) came back "unrecognised unit; not checked" and put COULD NOT VERIFY
on a correct answer, each found by a user and fixed by adding the word
(docs/design-review-2026-10.md, flaw 1). This module parses with pint, so any unit pint knows
parses, in any spelling pint accepts, and the dimension comes from the unit.

Three rules, none of them a vocabulary:

* **Grammar, not spellings.** A unit token followed directly by a digit is an exponent (`m2`,
  `km2`, `mi2`), and underscores are spaces (`square_kilometers`). pint does the rest:
  `km^2`, `km**2`, `km²`, `sq km`, `square miles`, `hectares`.
* **A count is a count of something.** A single word pint does not know (`schools`,
  `records`, `index`) is the name of what is counted or indexed: a dimensionless quantity
  labelled by that word. It is not an error, and it is not a measurement that a frame
  check could judge.
* **This registry measures ground, not paper.** pint's own `Printer` group (point, pica,
  pixel, dot, ...) would read "points" as a length of 1/72 inch. Units pint itself files
  under that group are read as counted things instead. The set is pint's, not ours.

Anything else that does not parse is reported as unparseable: loudly, as a unit nobody can
check, never silently as a pass.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

_EXPONENT_RE = re.compile(r"(?<=[A-Za-z])(\d)(?![\d.])")
_BARE_WORD_RE = re.compile(r"^[A-Za-z]+$")
_HYPHEN_PRODUCT_RE = re.compile(r"(?<=[A-Za-z])-(?=[A-Za-z])")
_PER_RE = re.compile(r"(?:^|\s+)per\s+", re.I)
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
# What a unit expression is made of. Anything else (a ':' in "EPSG:4326") is not a unit, and
# its unknown words are not counted things.
_UNIT_CHARS_RE = re.compile(r"^[\w\s*/^.()²³°%-]+$")


@dataclass(frozen=True)
class ParsedUnit:
    text: str                      # as declared
    kind: str                      # "unit" | "label" | "unparseable" | "missing"
    canonical: Optional[str] = None  # pint's spelling, for "unit"
    dimension: Optional[str] = None  # e.g. "[length]", "[length] ** 2", "dimensionless"
    counted: Optional[str] = None    # for "label": what the number counts or names
    error: Optional[str] = None

    @property
    def is_length(self) -> bool:
        return self.dimension == "[length]"

    @property
    def is_area(self) -> bool:
        return self.dimension == "[length] ** 2"

    @property
    def is_count(self) -> bool:
        return self.kind == "label" or self.canonical in ("count", "dimensionless")

    @property
    def scale(self) -> float:
        return self._scale

    _scale: float = 1.0


@lru_cache(maxsize=1)
def registry():
    import pint

    return pint.UnitRegistry(on_redefinition="ignore")


@lru_cache(maxsize=1)
def _printer_units() -> frozenset:
    try:
        return frozenset(registry().get_group("Printer").members)
    except Exception:  # noqa: BLE001 - an older pint without groups: no printer units to drop
        return frozenset()


def _normalise(text: str) -> str:
    t = text.strip().replace("_", " ")
    t = _THOUSANDS_RE.sub("", t)
    t = _PER_RE.sub(" / ", t).strip()
    if t.startswith("/"):
        t = "1 " + t                              # "per 1000 residents"
    t = _HYPHEN_PRODUCT_RE.sub("*", t)           # person-km is person times km
    t = _EXPONENT_RE.sub(r"**\1", t)
    return t


@lru_cache(maxsize=512)
def parse_unit(declared: Any) -> ParsedUnit:
    text = "" if declared is None else str(declared)
    if not text.strip():
        return ParsedUnit(text=text, kind="missing")
    ureg = registry()
    if _BARE_WORD_RE.match(text.strip()):
        try:
            ureg.parse_units(_normalise(text))
        except Exception:  # noqa: BLE001 - a word pint does not know names what is counted
            return ParsedUnit(text=text, kind="label", dimension="dimensionless",
                              counted=text.strip())
    expr, counted = _normalise(text), []
    for _ in range(4):
        try:
            # parse_expression, not parse_units: a rate's unit carries a scale ("per 1000
            # people"), and the number is part of the unit, not a value.
            parsed = ureg.parse_expression(expr)
            unit = getattr(parsed, "units", None)
            if unit is None:                 # a bare number: no unit at all
                raise ValueError("a number, not a unit")
            magnitude = float(getattr(parsed, "magnitude", 1.0))
            if abs(magnitude - 1.0) > 1e-12 and "/" not in expr:
                # A scale belongs to a rate's denominator ("per 1000 residents"). "25 m" in a
                # unit field is a value that ended up in the wrong place.
                return ParsedUnit(text=text, kind="unparseable",
                                  error=f"a quantity ({parsed}), not a unit")
            break
        except Exception as exc:  # noqa: BLE001 - pint raises several kinds
            # A word pint does not know inside a compound ("person*km", "schools per km2") is
            # the counted factor: read it as a count and keep its name.
            unknown = [n for n in getattr(exc, "unit_names", None) or []
                       if isinstance(n, str) and n.isalpha()]
            if not unknown or not _UNIT_CHARS_RE.match(expr):
                return ParsedUnit(text=text, kind="unparseable",
                                  error=f"{type(exc).__name__}: {exc}"[:200])
            for name in unknown:
                expr = re.sub(rf"\b{re.escape(name)}\b", "count", expr)
                counted.append(name)
    else:
        return ParsedUnit(text=text, kind="unparseable", error="too many unknown names")
    names = {str(n) for n, _ in unit._units.items()} if hasattr(unit, "_units") else set()
    if names & _printer_units():
        return ParsedUnit(text=text, kind="label", dimension="dimensionless",
                          counted=text.strip())
    dim = str(unit.dimensionality)
    return ParsedUnit(text=text, kind="unit", canonical=str(unit),
                      dimension="dimensionless" if dim == "dimensionless" else dim,
                      counted=" ".join(counted) or None)


def convert(value: float, from_unit: Any, to_unit: Any) -> Optional[float]:
    """value in *from_unit* expressed in *to_unit*, or None when either does not parse or the
    dimensions differ."""
    a, b = parse_unit(from_unit), parse_unit(to_unit)
    if a.kind != "unit" or b.kind != "unit":
        return None
    try:
        ureg = registry()
        return float((value * ureg.parse_units(a.canonical)).to(
            ureg.parse_units(b.canonical)).magnitude)
    except Exception:  # noqa: BLE001
        return None


def typed_value(name: str, value: Any, unit: str, *, measured_in_crs: Optional[str] = None,
                source: Optional[str] = None, op: Optional[str] = None) -> dict:
    """A number as a tool reports it: `{name, value, unit, dimension, measured_in_crs, source}`.

    Built where the number is made, while the code still knows what it is. Raises on a unit
    that does not parse, so a tool that writes a wrong unit fails in its own tests, not in a
    user's answer.
    """
    pu = parse_unit(unit)
    if pu.kind in ("missing", "unparseable"):
        raise ValueError(f"typed_value {name!r}: unit {unit!r} does not parse ({pu.error})")
    out = {"name": name, "value": value, "unit": unit, "dimension": pu.dimension}
    if pu.counted:
        out["counted"] = pu.counted
    if measured_in_crs:
        out["measured_in_crs"] = str(measured_in_crs)
    if source:
        out["source"] = source
    if op:
        out["op"] = op
    return out
