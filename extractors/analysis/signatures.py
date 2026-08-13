"""Full-fidelity signatures and parameter contracts. Pure AST, no execution.

Replaces ``code_extractor._signature``, which rebuilt signatures by hand and silently dropped
positional-only markers, keyword-only arguments, every default, and every annotation —
including the return type. For a function like::

    def f(p, /, a: int = 3, *args: str, k: float = 1.0, **kw) -> 'gpd.GeoDataFrame':

it emitted ``def f(a, *args, **kw)``. An agent choosing a method from that signature cannot
tell what is required, what a parameter means, or what comes back.

``ast.unparse(node.args)`` reproduces all of it faithfully, so the hand-rolled version is
replaced rather than patched.
"""

from __future__ import annotations

import ast
from typing import Any, List, Optional

from ..contracts import InvariantSpec, ParamSpec

# A (Geo)DataFrame parameter is the hook the invariant gate hangs on: CRS and unit checks
# only apply to frames. Matched by NAME so stringized PEP-563 annotations work, the same
# technique as ``extractors/geo_handles.py:26 _is_frame_type``.
_FRAME_HINTS = ("geodataframe", "gdf", "dataframe", "df")
_PATH_HINTS = ("path", "file", "filename", "filepath", "shp", "csv", "src", "dest")
_URL_HINTS = ("url", "uri", "endpoint", "link")
_NUM_HINTS = ("count", "n", "k", "limit", "size", "buffer", "distance", "radius", "threshold")

# Distance/area operations that are WRONG in a geographic CRS: this is the degrees-vs-metres
# class of error, which produces a plausible number (21.5 km for a requested 25 km) rather
# than an exception.
_PROJECTED_OPS = ("buffer", "distance", "sjoin_nearest", "length", "area", "centroid")


def _unparse(node: Optional[ast.AST]) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _init_of(node: ast.AST) -> Optional[ast.AST]:
    """A class's ``__init__``, searched only in its OWN body.

    Not inherited: resolving a base class means resolving a name that may come from another
    module entirely, and guessing a superclass's constructor is worse than declining to.
    """
    if not isinstance(node, ast.ClassDef):
        return None
    for stmt in node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == "__init__":
            return stmt
    return None


def signature_of(node: ast.AST) -> str:
    """The call signature a caller would write, with annotations and defaults intact.

    For a class this is the CONSTRUCTOR call, ``DoubleConv(in_c, out_c)`` — not ``def
    DoubleConv()``, which is what a ClassDef produced when it fell through to the function path
    with no ``.args`` attribute. That advertisement was wrong twice over: it called a class a
    ``def``, and it dropped every constructor argument. Measured on the live registry, 35 units
    advertised ``def X()``; following the contract for ``DoubleConv``, whose shipped code is
    ``def __init__(self, in_c, out_c)``, raises TypeError immediately.
    """
    name = getattr(node, "name", "<anonymous>")
    if isinstance(node, ast.ClassDef):
        init = _init_of(node)
        if init is None:
            bases = ", ".join(_unparse(b) for b in (node.bases or []) if _unparse(b))
            # No __init__ of its own. Whether it takes arguments depends on a base class this
            # analyser deliberately does not resolve, so say that rather than imply zero args.
            return f"{name}(...)  # class{f', inherits {bases}' if bases else ''}"
        args = _strip_self(_unparse(getattr(init, "args", None)))
        return f"{name}({args})  # class"
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = _unparse(getattr(node, "args", None))
    returns = _unparse(getattr(node, "returns", None))
    sig = f"{prefix} {name}({args})"
    return f"{sig} -> {returns}" if returns else sig


def _strip_self(args: str) -> str:
    """Drop the leading ``self``/``cls`` — a caller never passes it."""
    text = (args or "").strip()
    for first in ("self", "cls"):
        if text == first:
            return ""
        if text.startswith(first + ","):
            return text[len(first) + 1:].strip()
        if text.startswith(first + ":") or text.startswith(first + " "):
            head, _, tail = text.partition(",")
            return tail.strip() if _ else ""
    return text


def _mk(name: str, kind: str, annotation: Any, default: Any, required: bool) -> ParamSpec:
    return ParamSpec(name=name, kind=kind, annotation=_unparse(annotation),
                     default=_unparse(default), required=required)


def params_of(node: ast.AST) -> List[ParamSpec]:
    """Every parameter, across all five categories.

    The previous implementation read only ``node.args.args``, so positional-only and
    keyword-only parameters were invisible and no default was ever recorded.

    A ``ClassDef`` has no ``.args`` at all, so it returned ``[]`` and the whole downstream chain
    went quiet: ``contract_invariants`` iterates parameters, so a class unit carried ZERO
    invariants no matter what its constructor did with a GeoDataFrame, and a run passing a
    geographic frame to one was never checked. Class parameters come from ``__init__``, minus
    ``self``.
    """
    if isinstance(node, ast.ClassDef):
        init = _init_of(node)
        if init is None:
            return []
        return [p for p in params_of(init) if p.name not in ("self", "cls")]
    a = getattr(node, "args", None)
    if a is None:
        return []
    out: List[ParamSpec] = []

    posonly = list(getattr(a, "posonlyargs", []) or [])
    positional = list(a.args or [])
    defaults = list(a.defaults or [])
    # defaults align to the RIGHT of posonly + positional
    all_pos = posonly + positional
    pad = len(all_pos) - len(defaults)
    for i, arg in enumerate(all_pos):
        d = defaults[i - pad] if i >= pad else None
        kind = "positional_only" if i < len(posonly) else "positional_or_keyword"
        out.append(_mk(arg.arg, kind, arg.annotation, d, d is None))

    if a.vararg:
        out.append(_mk(a.vararg.arg, "var_positional", a.vararg.annotation, None, False))

    for arg, d in zip(a.kwonlyargs or [], a.kw_defaults or []):
        out.append(_mk(arg.arg, "keyword_only", arg.annotation, d, d is None))

    if a.kwarg:
        out.append(_mk(a.kwarg.arg, "var_keyword", a.kwarg.annotation, None, False))
    return out


def _looks_like(name: str, hints: tuple) -> bool:
    low = name.lower()
    return any(h in low for h in hints)


def infer_types(params: List[ParamSpec], docstring: str = "") -> None:
    """Fill ``inferred_type`` from the annotation first, then the parameter name.

    Annotation wins because it is a declaration; the name is a heuristic and every guess
    records its ``evidence`` so a wrong one is auditable rather than mysterious.
    """
    for p in params:
        ann = (p.annotation or "").lower()
        if ann:
            if "geodataframe" in ann:
                p.inferred_type, p.evidence = "geodataframe", f"annotation {p.annotation!r}"
                continue
            if "dataframe" in ann:
                p.inferred_type, p.evidence = "dataframe", f"annotation {p.annotation!r}"
                continue
            if "path" in ann or "str" in ann and _looks_like(p.name, _PATH_HINTS):
                p.inferred_type, p.evidence = "path", f"annotation {p.annotation!r}"
                continue
            for token, kind in (("int", "number"), ("float", "number"),
                                ("bool", "bool"), ("str", "str")):
                if token in ann:
                    p.inferred_type, p.evidence = kind, f"annotation {p.annotation!r}"
                    break
            if p.inferred_type != "unknown":
                continue
        if _looks_like(p.name, _FRAME_HINTS):
            p.inferred_type, p.evidence = "geodataframe", f"parameter name {p.name!r}"
        elif _looks_like(p.name, _URL_HINTS):
            p.inferred_type, p.evidence = "url", f"parameter name {p.name!r}"
        elif _looks_like(p.name, _PATH_HINTS):
            p.inferred_type, p.evidence = "path", f"parameter name {p.name!r}"
        elif _looks_like(p.name, _NUM_HINTS):
            p.inferred_type, p.evidence = "number", f"parameter name {p.name!r}"


def infer_units_and_crs(params: List[ParamSpec], node: ast.AST, docstring: str = "") -> None:
    """Record a CRS expectation when the body performs a distance/area operation.

    Only ``projected`` is asserted here, and only from real AST evidence. An undetermined
    unit stays empty rather than being guessed — the invariant gate treats an undeclared unit
    as "must be declared before this number is presented", which is safe; a wrong guess is not.
    """
    ops_found: List[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and sub.attr in _PROJECTED_OPS:
            ops_found.append(f".{sub.attr}( at line {getattr(sub, 'lineno', 0)}")
    if not ops_found:
        return
    evidence = "; ".join(sorted(set(ops_found))[:3])
    for p in params:
        if p.inferred_type == "geodataframe" and not p.crs_expectation:
            p.crs_expectation = "projected"
            p.evidence = (p.evidence + " | " if p.evidence else "") + \
                         f"body performs a metric operation ({evidence})"
            if not p.declared_unit:
                p.declared_unit = "metres"


def contract_params(node: ast.AST, docstring: str = "") -> List[ParamSpec]:
    """params_of + type inference + unit/CRS inference, in one pass."""
    params = params_of(node)
    infer_types(params, docstring)
    infer_units_and_crs(params, node, docstring)
    return params


def contract_invariants(params: List[ParamSpec], node: ast.AST) -> List[InvariantSpec]:
    """Turn inferred parameter expectations into ENFORCEABLE invariants.

    The missing link. ``infer_units_and_crs`` has always set ``crs_expectation="projected"`` on
    a GeoDataFrame parameter whose body performs a metric operation — 20 params across the
    corpus carry it — and nothing ever converted that into an ``InvariantSpec``. Measured
    before this: **zero** invariants in the whole 203-unit registry. The expectation was
    displayed to the model in ``get_method_contract`` and enforced nowhere, so "contract-bearing
    structure" meant "structure the model is told about".

    Each invariant names the PARAMETER it constrains, which is what lets the runtime wrapper
    check the actual argument rather than guessing which frame in the namespace was meant.
    """
    out: List[InvariantSpec] = []
    for p in params:
        if p.crs_expectation == "projected":
            out.append(InvariantSpec(check="projected_crs", target=p.name,
                                     args={"unit": p.declared_unit or "metres"},
                                     evidence=p.evidence))
        elif p.crs_expectation and p.crs_expectation.lower().startswith("epsg:"):
            out.append(InvariantSpec(check="crs_equals", target=p.name,
                                     args={"crs": p.crs_expectation}, evidence=p.evidence))
        if p.inferred_type in {"geodataframe", "dataframe"}:
            # A frame argument that is entirely null is a failed upstream join; checking it at
            # the CALL means the caller learns which step broke, not just that the end was NaN.
            out.append(InvariantSpec(check="reject_all_nan", target=p.name,
                                     evidence=f"parameter inferred as {p.inferred_type}"))
    return out


__all__ = ["signature_of", "params_of", "infer_types", "infer_units_and_crs",
           "contract_params", "contract_invariants"]
