"""Can an extracted function be called on its own? Static analysis, no execution.

The problem this solves
-----------------------
A function lifted out of notebook cell 12 that references ``gdf`` defined in cell 4 will
import fine and then either fail at call time or — worse — silently use a stale global. That
is the single hardest part of making extracted units trustworthy, and it is why
``notebook_extractor`` currently promotes at most one whole-notebook entry point instead of
per-function units.

The check is a free-variable analysis built on stdlib ``symtable`` rather than a hand-rolled
``ast.walk``, because symtable already resolves comprehension scopes, the walrus operator,
``global``/``nonlocal``, closures and class bodies correctly. Verified: for a module with
``THRESH = 0.5`` (a literal), ``gdf_missing = gpd.read_file(...)`` (a call), and functions
reading each, symtable reports ``globals={'THRESH'}`` and ``globals={'gdf_missing'}``
respectively, with closure variables correctly reported as *free* rather than *global*.

The distinction that matters is not "does it read a global" but **what kind of binding the
global is**:

  imports  -> satisfiable: copy the import line into the slice
  defs     -> satisfiable: copy the dependency function, transitively
  consts   -> satisfiable: copy the literal
  runtime  -> BLOCKER: the value came from executing something (``gpd.read_file(...)``)
  unbound  -> BLOCKER: nothing in the module defines it

Only the last two make a unit uncallable. Everything else can be carried along.
"""

from __future__ import annotations

import ast
import builtins
import symtable
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..contracts import (CALLABLE, NEEDS_GLOBALS, NEEDS_INSTANCE, UNPARSEABLE,
                         Callability)

_BUILTINS = frozenset(dir(builtins))


@dataclass
class ModuleScope:
    """Module-level bindings, classified by whether they are safe to inline into a slice."""
    imports: Dict[str, str] = field(default_factory=dict)   # bound name -> import statement source
    defs: Dict[str, Any] = field(default_factory=dict)      # name -> FunctionDef/AsyncFunctionDef/ClassDef
    consts: Dict[str, str] = field(default_factory=dict)    # name -> assignment source (literal RHS)
    runtime: Dict[str, str] = field(default_factory=dict)   # name -> assignment source (computed RHS)
    side_effect_lines: List[int] = field(default_factory=list)
    order: List[str] = field(default_factory=list)


def _is_literal(node: ast.AST) -> bool:
    """True when an expression is a pure literal, so inlining it cannot execute anything."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_literal(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(_is_literal(k) for k in node.keys if k is not None) and \
               all(_is_literal(v) for v in node.values)
    if isinstance(node, ast.UnaryOp):
        return _is_literal(node.operand)
    if isinstance(node, ast.BinOp):
        return _is_literal(node.left) and _is_literal(node.right)
    if isinstance(node, ast.JoinedStr):      # f-string of literals only
        return all(_is_literal(v.value) if isinstance(v, ast.FormattedValue) else True
                   for v in node.values)
    return False


def _targets(node: ast.AST) -> List[str]:
    names: List[str] = []
    targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
    for t in targets:
        if isinstance(t, ast.Name):
            names.append(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            names.extend(e.id for e in t.elts if isinstance(e, ast.Name))
    return names


def module_scope(tree: ast.Module, source: str) -> ModuleScope:
    """Classify every module-level statement."""
    scope = ModuleScope()
    for stmt in tree.body:
        seg = ast.get_source_segment(source, stmt) or ""
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for alias in stmt.names:
                bound = alias.asname or alias.name.split(".")[0]
                scope.imports[bound] = seg
                scope.order.append(bound)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope.defs[stmt.name] = stmt
            scope.order.append(stmt.name)
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            value = getattr(stmt, "value", None)
            names = _targets(stmt)
            bucket = scope.consts if (value is not None and _is_literal(value)) else scope.runtime
            for name in names:
                bucket[name] = seg
                scope.order.append(name)
        else:
            scope.side_effect_lines.append(getattr(stmt, "lineno", 0))
    return scope


def _function_tables(table: symtable.SymbolTable) -> List[symtable.SymbolTable]:
    """All function scopes, including nested functions and methods inside classes."""
    out: List[symtable.SymbolTable] = []
    for child in table.get_children():
        if child.get_type() == "function":
            out.append(child)
            out.extend(_function_tables(child))
        elif child.get_type() == "class":
            out.extend(_function_tables(child))
    return out


def _globals_of(table: symtable.SymbolTable) -> Tuple[set, set]:
    """(read_globals, written_globals) for one function scope, including its nested scopes.

    Closure variables are excluded automatically: symtable reports those as *free*, not
    *global*, which is the main reason for using it over a hand-rolled walk.
    """
    reads, writes = set(), set()
    for sym in table.get_symbols():
        if not sym.is_global():
            continue
        name = sym.get_name()
        if name in _BUILTINS:
            continue
        if sym.is_assigned():
            writes.add(name)
        reads.add(name)
    for child in _function_tables(table):
        r, w = _globals_of(child)
        reads |= r
        writes |= w
    return reads, writes


def annotation_names(node: ast.AST) -> List[str]:
    """Root names a unit needs at DEF TIME: annotations, decorators, parameter defaults and
    class bases.

    symtable does not report any of these as globals of the function scope, and it is right
    not to — they are all evaluated in the *enclosing* scope when the ``def`` or ``class``
    statement executes, not when the function is called. But a slice carries them, so they
    must still be satisfied or the slice raises at IMPORT time rather than at call time.

    Each position was added after a real slice failed on it, at corpus scale:

      annotations   ``def filter_dataframe_by_value(df: pd.DataFrame) -> pd.DataFrame`` uses
                    ``pd`` nowhere in its body -> ``NameError: name 'pd' is not defined``.
      defaults      ``def evaluate(..., feats=FEATS)`` -> ``NameError: FEATS``. The module
                    constant simply was not carried into the slice.
      defaults      ``def plot_predictions(train_data=X_train, ...)`` -> ``NameError: X_train``.
                    This one is worse than a missing const: ``X_train`` is a RUNTIME binding,
                    so the unit was never independently callable and the analyzer said it was.
                    Counting defaults turns it into a blocker and the unit is correctly refused.
      class bases   ``class AgentState(TypedDict)`` -> ``NameError: TypedDict``. A base class is
                    evaluated when the class body executes.

    Stringized annotations are deliberately excluded: ``'gpd.GeoDataFrame'`` is never evaluated
    at def time and so cannot fail an import.
    """
    targets: List[ast.AST] = []
    args = getattr(node, "args", None)
    if args is not None:
        for group in ("posonlyargs", "args", "kwonlyargs"):
            for a in getattr(args, group, []) or []:
                if a.annotation is not None:
                    targets.append(a.annotation)
        for extra in (getattr(args, "vararg", None), getattr(args, "kwarg", None)):
            if extra is not None and extra.annotation is not None:
                targets.append(extra.annotation)
        # Default expressions run at def time in the enclosing scope, so a default naming a
        # module-level value is a real requirement — and one naming a RUNTIME value is a real
        # blocker. kw_defaults holds None for keyword-only params that have no default.
        targets.extend(d for d in (getattr(args, "defaults", []) or []) if d is not None)
        targets.extend(d for d in (getattr(args, "kw_defaults", []) or []) if d is not None)
    if getattr(node, "returns", None) is not None:
        targets.append(node.returns)          # type: ignore[attr-defined]
    targets.extend(getattr(node, "decorator_list", []) or [])
    # ClassDef: bases and metaclass keywords are evaluated when the class statement runs.
    targets.extend(getattr(node, "bases", []) or [])
    targets.extend(kw.value for kw in (getattr(node, "keywords", []) or [])
                   if getattr(kw, "value", None) is not None)

    names: List[str] = []
    for t in targets:
        for sub in ast.walk(t):
            # The ROOT of an attribute chain is what needs binding: pd.DataFrame -> pd
            if isinstance(sub, ast.Name) and sub.id not in _BUILTINS:
                names.append(sub.id)
            elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                # A stringized annotation ('gpd.GeoDataFrame') is NOT evaluated at def time,
                # so it cannot fail an import; deliberately ignored.
                continue
    return sorted(set(names))


def iter_units(tree: ast.Module) -> List[Tuple[str, ast.AST]]:
    """(qualified_name, node) for every top-level function and public method."""
    units: List[Tuple[str, ast.AST]] = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            units.append((stmt.name, stmt))
        elif isinstance(stmt, ast.ClassDef):
            for sub in stmt.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and not sub.name.startswith("_"):
                    units.append((f"{stmt.name}.{sub.name}", sub))
    return units


def analyze_module(source: str) -> Tuple[Dict[str, Callability], ModuleScope, Dict[str, Any]]:
    """Classify every unit in *source*.

    Returns ``(verdicts_by_qualified_name, module_scope, summary)``. Never raises: an
    unparseable module yields empty verdicts and ``summary["unparseable"] = True``, because a
    syntax error in one extracted module must not fail a whole ingest.
    """
    try:
        tree = ast.parse(source)
        table = symtable.symtable(source, "<extracted>", "exec")
    except SyntaxError as exc:
        return {}, ModuleScope(), {"unparseable": True, "error": f"{type(exc).__name__}: {exc}",
                                   "total": 0, "callable": 0}

    scope = module_scope(tree, source)
    by_name = {t.get_name(): t for t in _function_tables(table)}
    _class_tables = {c.get_name(): c for c in table.get_children() if c.get_type() == "class"}
    unit_nodes = dict(iter_units(tree))

    verdicts: Dict[str, Callability] = {}
    for qualname in unit_nodes:
        simple = qualname.split(".")[-1]
        ftable = by_name.get(simple)
        c = Callability(verdict=CALLABLE)
        if ftable is None:
            # A unit with no symtable entry cannot be reasoned about; do not claim it callable.
            c.verdict = UNPARSEABLE
            c.reason = "no symbol table for this function"
            verdicts[qualname] = c
            continue
        reads, writes = _globals_of(ftable)
        # Annotations and decorators are evaluated at def time in the enclosing scope, so a
        # slice must satisfy them too even though symtable does not count them as globals.
        reads |= set(annotation_names(unit_nodes[qualname]))
        c.global_writes = sorted(writes)
        for name in sorted(reads):
            if name in scope.imports:
                c.requires_imports.append(name)
            elif name in scope.defs:
                c.requires_units.append(name)
            elif name in scope.consts:
                c.requires_consts.append(name)
            elif name in scope.runtime:
                c.global_reads.append(name)
            elif name in unit_nodes or name in by_name:
                c.requires_units.append(name)
            else:
                c.free_names.append(name)
        if c.global_reads:
            c.verdict = NEEDS_GLOBALS
            first = c.global_reads[0]
            c.reason = (f"reads module-level value {first!r} produced by "
                        f"{scope.runtime.get(first, '')[:60]!r}")
        elif c.free_names:
            c.verdict = NEEDS_GLOBALS
            c.reason = f"unbound name(s): {', '.join(c.free_names)}"
        elif "." in qualname:
            # A bound method. It reads no globals — `self` is a PARAMETER — so every check
            # above passes and it was being verdicted `callable`. Measured on the 14-notebook
            # corpus that was 24 of 40 units, and it is a false-callable in the strict sense:
            # the emitted slice defines the CLASS, so `from <module> import build_api_url`
            # raises ImportError, and one such re-export poisoned its element's __init__ and
            # took every sibling unit down with it (26 of 40 import lines failed).
            #
            # Calling one really does require constructing the class, whose __init__ may open
            # files or hit the network, so this is not merely a naming problem to paper over.
            # Still emitted for discovery — the method body is useful evidence — but never
            # shipped as importable code.
            c.verdict = NEEDS_INSTANCE
            c.reason = (f"bound method of {qualname.split('.')[0]!r}; calling it requires an "
                        f"instance, so it is not importable as a standalone function")
        verdicts[qualname] = c

    # Module-level CLASSES are not units — `iter_units` never yields them and the promotion
    # loop skips any name it does not yield — but they ARE sliceable dependencies, and the
    # closure walk dead-ends on a name with no entry. Without this, slicing `make_workflow`
    # (which calls a helper returning `AgentState`) emitted `class AgentState(TypedDict)` and
    # not the `TypedDict` import, so the slice raised NameError at import.
    class_names: List[str] = []
    for stmt in tree.body:
        if not isinstance(stmt, ast.ClassDef) or stmt.name in verdicts:
            continue
        class_names.append(stmt.name)
        c = Callability(verdict=CALLABLE)
        reads = set(annotation_names(stmt))
        # NOT by_name: that is built from _function_tables, which recurses THROUGH class
        # tables to reach their methods without ever including the class scope itself. The
        # class block is where a body annotation lives, and symtable does report its names as
        # globals — `class AgentState(TypedDict): messages: Annotated[list, operator.add]`
        # yields globals {Annotated, list, operator}. Looking the class up in the function map
        # silently found nothing, and the slice raised `NameError: operator` on import.
        table = _class_tables.get(stmt.name)
        if table is not None:
            r, _w = _globals_of(table)
            reads |= r
        for name in sorted(reads):
            if name in scope.imports:
                c.requires_imports.append(name)
            elif name in scope.defs:
                c.requires_units.append(name)
            elif name in scope.consts:
                c.requires_consts.append(name)
            elif name in scope.runtime:
                c.global_reads.append(name)
                c.verdict = NEEDS_GLOBALS
                c.reason = f"class body reads module-level runtime value {name!r}"
            elif name not in unit_nodes:
                c.free_names.append(name)
        verdicts[stmt.name] = c

    _demote_transitively(verdicts)

    # Classes are dependencies, not units: counting them would inflate "N of M callable".
    unit_verdicts = {n: v for n, v in verdicts.items() if n not in set(class_names)}
    total = len(unit_verdicts)
    ok = sum(1 for v in unit_verdicts.values() if v.verdict == CALLABLE)
    summary = {
        "total": total,
        "callable": ok,
        "needs_globals": sorted(n for n, v in unit_verdicts.items() if v.verdict == NEEDS_GLOBALS),
        "needs_instance": sorted(n for n, v in unit_verdicts.items() if v.verdict == NEEDS_INSTANCE),
        "unparseable": sorted(n for n, v in unit_verdicts.items() if v.verdict == UNPARSEABLE),
        "blocked_by": _blocked_by_histogram(unit_verdicts),
        "module_side_effect_lines": scope.side_effect_lines,
    }
    return verdicts, scope, summary


def _demote_transitively(verdicts: Dict[str, Callability]) -> None:
    """A unit whose dependency closure touches a blocked unit is itself not callable.

    Copying ``good`` into a slice is only safe if everything ``good`` calls is also safe;
    otherwise the slice imports and then fails on the first call.
    """
    simple = {name.split(".")[-1]: name for name in verdicts}
    changed = True
    while changed:
        changed = False
        for name, c in verdicts.items():
            if c.verdict != CALLABLE:
                continue
            for dep in c.requires_units:
                target = verdicts.get(simple.get(dep, dep))
                if target is not None and target.verdict != CALLABLE:
                    c.verdict = NEEDS_GLOBALS
                    c.reason = f"depends on {dep!r}, which is not callable ({target.reason})"
                    changed = True
                    break


def _blocked_by_histogram(verdicts: Dict[str, Callability]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for c in verdicts.values():
        for name in c.blocked_by:
            hist[name] = hist.get(name, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: (-kv[1], kv[0])))


def analyze_unit(source: str, qualified_name: str) -> Callability:
    """Convenience wrapper for a single unit."""
    verdicts, _, summary = analyze_module(source)
    if qualified_name in verdicts:
        return verdicts[qualified_name]
    return Callability(verdict=UNPARSEABLE,
                       reason=summary.get("error") or f"{qualified_name} not found in module")


__all__ = ["ModuleScope", "module_scope", "iter_units", "annotation_names",
           "analyze_module", "analyze_unit"]
