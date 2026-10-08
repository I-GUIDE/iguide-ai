"""Emit a standalone, importable slice for one extracted unit. Pure text, no execution.

What a slice is, and what it deliberately is not
------------------------------------------------
A slice is the *transitive closure* of one function: its own source, the sibling functions it
calls, the literal constants it reads, and the import lines that bind the names it uses.
Nothing else.

In particular a slice NEVER contains:

* module-level **runtime** bindings — ``gdf = gpd.read_file("x.shp")`` and friends, and
* module-level **side-effect statements** — bare calls, loops, ``print``, plotting setup.

That omission is the entire safety argument. Importing a slice must not download a file, hit
an API, block on a credential, or spend a minute rebuilding a frame nobody asked for. It is
also exactly why a unit whose verdict is ``needs_globals`` is not shipped at all: the only way
to make it run would be to inline the very statements this module refuses to emit.

This generalises ``NotebookExtractor._build_module_source``, which concatenates *every* cell,
and reuses the single-function isolation technique already proven in
``extractors/geo_handles.py:132 _extract_function_source`` — which exists precisely because
exec'ing a whole block was unsafe.
"""

from __future__ import annotations

import ast
import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..contracts import CALLABLE, Callability
from .callability import ModuleScope

_DECORATORS_TO_STRIP = {"tool", "mcp_tool", "staticmethod", "classmethod"}


def build_module_source(title: str, cells: Sequence[Tuple[int, str]]) -> str:
    """Concatenate transformed notebook cells into one module, with cell markers.

    Moved here from ``NotebookExtractor._build_module_source`` so the notebook extractor and
    the slice builder share one definition of "the module a notebook denotes".
    """
    header = f'"""Module synthesized from notebook: {title}"""'
    parts = [header]
    for order, code in cells:
        parts.append(f"# Cell {order}\n{code.rstrip()}")
    return "\n\n".join(parts) + "\n"


def _strip_decorators(node: ast.AST) -> None:
    """Drop framework decorators that would fail to resolve inside a slice.

    ``@tool``/``@mcp_tool`` come from the agent runtime, which a slice must not depend on:
    the point is a plain callable. Same reasoning as
    ``geo_handles._extract_function_source``.
    """
    kept = []
    for dec in getattr(node, "decorator_list", []) or []:
        name = ""
        if isinstance(dec, ast.Name):
            name = dec.id
        elif isinstance(dec, ast.Attribute):
            name = dec.attr
        elif isinstance(dec, ast.Call):
            f = dec.func
            name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", "")
        if name not in _DECORATORS_TO_STRIP:
            kept.append(dec)
    node.decorator_list = kept  # type: ignore[attr-defined]


def _def_source(node: ast.AST, source: str) -> str:
    seg = ast.get_source_segment(source, node)
    if seg:
        # Re-parse so decorators can be stripped without disturbing the rest of the text.
        try:
            mod = ast.parse(seg)
            if mod.body and isinstance(mod.body[0], (ast.FunctionDef, ast.AsyncFunctionDef,
                                                     ast.ClassDef)):
                _strip_decorators(mod.body[0])
                return ast.unparse(mod)
        except SyntaxError:
            pass
        return seg
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _order_units(names: Sequence[str], scope: ModuleScope,
                 verdicts: Dict[str, Callability]) -> List[str]:
    """Dependency-first ordering of sibling defs, cycle-safe."""
    ordered: List[str] = []
    seen: set = set()

    def visit(name: str, stack: frozenset) -> None:
        if name in seen or name in stack or name not in scope.defs:
            return
        c = verdicts.get(name)
        for dep in (c.requires_units if c else []):
            visit(dep, stack | {name})
        if name not in seen:
            seen.add(name)
            ordered.append(name)

    for n in names:
        visit(n, frozenset())
    return ordered


def _closure(qualified_name: str, verdicts: Dict[str, Callability]) -> Callability:
    """Requirements of a unit UNIONED over its whole transitive dependency closure.

    Taking only the target unit's own requirements produces slices that import cleanly and
    then die on the first call. Measured: slicing ``good`` (which calls ``helper``, which
    reads the const ``THRESH``) emitted ``helper`` but not ``THRESH``, because ``good`` itself
    never reads it — a guaranteed ``NameError`` at call time. ``has_module_side_effects``
    cannot catch that; only importing and calling the slice can.
    """
    root = verdicts.get(qualified_name)
    if root is None:
        return Callability()

    merged = Callability(verdict=root.verdict, reason=root.reason,
                         analyzer_version=root.analyzer_version)
    simple_of = {n.split(".")[-1]: n for n in verdicts}
    seen: set = set()
    stack = [qualified_name]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        c = verdicts.get(name) or verdicts.get(simple_of.get(name, name))
        if c is None:
            continue
        for src, dst in ((c.requires_imports, merged.requires_imports),
                         (c.requires_consts, merged.requires_consts),
                         (c.requires_units, merged.requires_units)):
            for item in src:
                if item not in dst:
                    dst.append(item)
        stack.extend(c.requires_units)
    return merged


def build_unit_slice(module_source: str, qualified_name: str, *,
                     scope: ModuleScope,
                     verdicts: Dict[str, Callability],
                     provenance: Optional[Dict[str, Any]] = None) -> str:
    """Return importable source for *qualified_name*, or "" when it cannot be sliced.

    Refuses non-callable units by design: shipping one would require inlining the runtime
    bindings this function exists to exclude.
    """
    c = _closure(qualified_name, verdicts)
    if c.verdict != CALLABLE:
        return ""
    try:
        tree = ast.parse(module_source)
    except SyntaxError:
        return ""

    simple = qualified_name.split(".")[-1]
    node = None
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == simple:
            node = stmt
            break
        if isinstance(stmt, ast.ClassDef) and stmt.name == qualified_name.split(".")[0]:
            node = stmt  # a method: emit the whole (small) class so `self` resolves
            break
    if node is None:
        return ""

    prov = provenance or {}
    lines: List[str] = ['"""Extracted callable unit — do not edit; regenerate by re-ingesting.',
                        ""]
    lines.append(f"unit          : {qualified_name}")
    for key in ("element_id", "parent_doc_id", "source_rel_path", "cell_order",
                "commit_sha", "extractor", "analyzer_version", "extracted_at"):
        if prov.get(key) not in (None, ""):
            lines.append(f"{key:<14}: {prov[key]}")
    lines += ["",
              "Emitted as the transitive closure of this unit only. Module-level statements",
              "that EXECUTE (data loads, API calls, plotting setup) are deliberately absent,",
              'so importing this module has no side effects.', '"""', ""]

    # 1. imports the unit actually needs, verbatim, in module order
    for name in scope.order:
        if name in c.requires_imports and scope.imports.get(name):
            stmt = scope.imports[name]
            if stmt not in lines:
                lines.append(stmt)
    if c.requires_imports:
        lines.append("")

    # 2. literal constants it reads
    for name in scope.order:
        if name in c.requires_consts and scope.consts.get(name):
            stmt = scope.consts[name]
            if stmt not in lines:
                lines.append(stmt)
    if c.requires_consts:
        lines.append("")

    # 3. sibling defs, dependencies first
    for dep in _order_units(c.requires_units, scope, verdicts):
        if dep == simple:
            continue
        src = _def_source(scope.defs[dep], module_source)
        if src:
            lines += [src, ""]

    # 4. the unit itself
    lines.append(_def_source(node, module_source))
    return "\n".join(lines).rstrip() + "\n"


def slice_sha(slice_source: str, *, length: int = 12) -> str:
    """Content address of a slice — this IS the unit's version.

    An unchanged notebook therefore re-ingests to a byte-identical module (an idempotent
    no-op), while an edited function mints a new one and older versions stay resolvable.
    """
    return hashlib.sha1((slice_source or "").encode("utf-8")).hexdigest()[:length]


def has_module_side_effects(slice_source: str) -> bool:
    """True if a slice would execute anything at import time. Should always be False."""
    try:
        tree = ast.parse(slice_source)
    except SyntaxError:
        return True
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            value = getattr(stmt, "value", None)
            # An assignment is only safe if its right-hand side executes nothing.
            if value is not None and any(isinstance(n, (ast.Call, ast.Await))
                                         for n in ast.walk(value)):
                return True
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue   # docstring
        return True
    return False


__all__ = ["build_module_source", "build_unit_slice", "slice_sha", "has_module_side_effects"]
