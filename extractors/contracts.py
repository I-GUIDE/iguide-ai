"""Typed contracts for extracted callable units. Pure: no I/O, no heavy imports.

Why this module exists
----------------------
The unit of reuse for extracted knowledge is a **callable function with an explicit
contract**, not a frozen whole notebook. A whole-notebook workflow freezes the composition
its original author chose; users ask adjacent questions, so the composition is the part that
must stay free and the primitives are the part that must be fixed.

``notebook_extractor`` already parses every top-level function
(``_top_level_functions``) and then discards all but one entry point.
``code_extractor`` already emits per-symbol ``signature``/``docstring`` and then marks
non-entry-point assets "index-only". The contract below is what those discarded units become.

Everything here is JSON-round-trippable via ``dataclasses.asdict`` so it can ride inside an
``AssetRecord.unit`` payload and land in an OpenSearch document unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Bumped when the analyzer's classification rules change, so a stored verdict can be
# distinguished from one produced by a newer analyzer rather than silently trusted.
ANALYZER_VERSION = 1

# --- callability verdicts ---------------------------------------------------------------
CALLABLE = "callable"
NEEDS_GLOBALS = "needs_globals"
NEEDS_INSTANCE = "needs_instance"  # a bound method: calling it requires constructing its class
UNPARSEABLE = "unparseable"
SPEC_ONLY = "spec_only"          # publications: a declared contract with no body


@dataclass
class ParamSpec:
    """One parameter of an extracted unit.

    ``declared_unit`` and ``crs_expectation`` are what make the invariant gate possible: a
    metre-vs-degree error produces a *plausible* wrong answer (a 25 km buffer that is really
    21.5 km), so the expectation has to be recorded at extraction time to be checkable later.
    """
    name: str
    kind: str = "positional_or_keyword"   # positional_only|positional_or_keyword|var_positional|keyword_only|var_keyword
    annotation: str = ""                  # ast.unparse of the annotation, verbatim
    default: str = ""                     # ast.unparse of the default expression; "" == required
    required: bool = True
    inferred_type: str = "unknown"        # geodataframe|dataframe|path|url|number|str|bool|unknown
    declared_unit: str = ""               # metres|feet|degrees|count|percent|...
    crs_expectation: str = ""             # "EPSG:4326" | "projected" | "any" | ""
    schema: List[str] = field(default_factory=list)   # required columns, for frame params
    evidence: str = ""                    # why an inference was made, so a wrong guess is auditable


@dataclass
class Callability:
    """Whether a unit can be called on its own, and if not, exactly what blocks it.

    ``blocked_by`` is the actionable field: aggregated across a corpus it becomes a histogram
    telling you which extraction limitation to fix next, instead of guessing.
    """
    verdict: str = UNPARSEABLE
    reason: str = ""
    free_names: List[str] = field(default_factory=list)      # unresolved reads
    global_reads: List[str] = field(default_factory=list)    # resolved to module-level RUNTIME values
    global_writes: List[str] = field(default_factory=list)   # impurity, not a blocker
    requires_imports: List[str] = field(default_factory=list)
    requires_consts: List[str] = field(default_factory=list)
    requires_units: List[str] = field(default_factory=list)  # sibling defs it calls, transitively
    analyzer_version: int = ANALYZER_VERSION

    @property
    def is_callable(self) -> bool:
        return self.verdict == CALLABLE

    @property
    def blocked_by(self) -> List[str]:
        return sorted(set(self.global_reads) | set(self.free_names))


@dataclass
class InvariantSpec:
    """A checkable property derived from the unit's body, enforced at run time."""
    check: str                        # projected_crs|crs_equals|join_cardinality|reject_all_nan|required_columns
    target: str = ""                  # parameter name, or "return"
    args: Dict[str, Any] = field(default_factory=dict)
    evidence: str = ""                # the AST evidence, e.g. ".distance( at line 14"


@dataclass
class UnitContract:
    """The full contract for one extracted callable unit."""
    qualified_name: str
    unit_kind: str = "function"       # function|method|class|loader|spec
    signature: str = ""               # full fidelity, including annotations and return type
    params: List[ParamSpec] = field(default_factory=list)
    returns: str = ""
    return_kind: str = "unknown"      # geodataframe|dataframe|figure|path|scalar|none|unknown
    docstring: str = ""
    doc_summary: str = ""             # first line: the text worth embedding
    callability: Callability = field(default_factory=Callability)
    invariants: List[InvariantSpec] = field(default_factory=list)
    requirements: Dict[str, List[str]] = field(default_factory=dict)   # {"pip": [...], "system": [...]}
    library_module: str = ""
    library_symbol: str = ""
    slice_sha: str = ""               # content address of the emitted slice == the version
    provenance: Dict[str, Any] = field(default_factory=dict)
    fast_path: Optional[Dict[str, Any]] = None   # the validated whole-notebook workflow, if any

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = [
    "ANALYZER_VERSION", "CALLABLE", "NEEDS_GLOBALS", "UNPARSEABLE", "SPEC_ONLY",
    "ParamSpec", "Callability", "InvariantSpec", "UnitContract",
]
