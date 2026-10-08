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
    """(qualified_name, node) for every top-level function, public method, AND class.

    Classes are CANDIDATES here; the verdict decides whether one is promotable. Keeping the
    decision in the verdict rather than in a separate list is what makes the extractors work
    unchanged — they already iterate verdicts and skip whatever `iter_units` does not yield, so
    a class held back in a side list would have been analysed and then silently dropped.
    """
    units: List[Tuple[str, ast.AST]] = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            units.append((stmt.name, stmt))
        elif isinstance(stmt, ast.ClassDef):
            units.append((stmt.name, stmt))
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
    for qualname, node in unit_nodes.items():
        simple = qualname.split(".")[-1]
        is_class = isinstance(node, ast.ClassDef)
        # One loop for both kinds. The class scope is NOT in `by_name`: that map comes from
        # _function_tables, which recurses THROUGH class tables to reach their methods without
        # ever including the class block itself — and the class block is where a body
        # annotation lives. `class AgentState(TypedDict): messages: Annotated[list,
        # operator.add]` reports globals {Annotated, list, operator}; looking it up in the
        # function map silently found nothing and the slice raised `NameError: operator`.
        ftable = _class_tables.get(simple) if is_class else by_name.get(simple)
        c = Callability(verdict=CALLABLE)
        if ftable is None:
            # Cannot be reasoned about, so do not claim it callable.
            c.verdict = UNPARSEABLE
            c.reason = f"no symbol table for this {'class' if is_class else 'function'}"
            verdicts[qualname] = c
            continue
        reads, writes = _globals_of(ftable)
        # Annotations, decorators, parameter defaults and class bases are all evaluated at DEF
        # TIME in the enclosing scope, so a slice must satisfy them even though symtable does
        # not count them as globals of the body.
        reads |= set(annotation_names(node))
        c.global_writes = sorted(writes)
        for name in sorted(reads):
            if name == qualname or name == simple:
                continue                      # a class body referring to itself
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
        elif is_class:
            # A class is a legitimate unit of reuse — refusing all of them cost 50 units on the
            # corpus — but only when CONSTRUCTING it is safe. If `__init__` opens a file or hits
            # the network, the class is the object-shaped version of the hidden-global problem
            # and importing its slice into a sandbox with no network fails.
            unsafe = _construction_is_safe(node, scope, by_name, _class_tables)
            if unsafe:
                c.verdict = NEEDS_INSTANCE
                c.reason = (f"constructing {simple!r} is not isolated: {unsafe} — it cannot be "
                            f"instantiated in a sandbox with no network and no staged files")
        elif "." in qualname:
            # A bound method reads no globals — `self` is a PARAMETER — so every check above
            # passes and these were verdicted `callable`: 24 of 40 units on the first corpus.
            # The emitted slice defines the CLASS, so `from <module> import build_api_url`
            # raises ImportError, and one such re-export poisoned its element's __init__ and
            # took every sibling unit down with it. Reach a method through its class instead.
            c.verdict = NEEDS_INSTANCE
            c.reason = (f"bound method of {qualname.split('.')[0]!r}; call it via its class, "
                        f"which is promoted separately when construction is safe")
        verdicts[qualname] = c

    _demote_transitively(verdicts)

    # Classes ARE units now (a safely-constructible class is a legitimate unit of reuse), so
    # they count. Refusing all of them cost 50 units on the corpus.
    unit_verdicts = dict(verdicts)
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


_IO_CALLS = frozenset((
    # network / process
    "get", "post", "request", "urlopen", "connect", "download", "Session", "system", "run",
    # readers
    "open", "read_file", "read_csv", "read_parquet", "read_excel", "load", "loadtxt", "imread",
    "read_text", "read_bytes", "from_file", "load_state_dict",
    # writers
    "to_file", "to_csv", "to_parquet", "write_text", "write_bytes", "savefig", "save",
    # FILESYSTEM INSPECTION. Absent from the first version, and it is the case that actually
    # bit: `TIFDataset.__init__` calls os.listdir(images_dir) and then RAISES ValueError when
    # the directory holds no .tif files, so constructing it in a fresh sandbox fails every
    # time -- and it was verdicted `callable`.
    "makedirs", "mkdir", "listdir", "scandir", "iterdir", "glob", "rglob", "walk", "exists",
    "isfile", "isdir", "stat", "remove", "unlink", "rmtree", "copy", "copytree", "move",
    "chdir", "mkstemp", "mkdtemp", "NamedTemporaryFile", "TemporaryDirectory",
))

# Never treated as I/O even though the name matches a member of _IO_CALLS above. `os.path.join`
# and `os.path.dirname` are pure string manipulation, and `.get` is overwhelmingly a dict read.
_IO_SAFE_QUALIFIED = frozenset((
    "os.path.join", "os.path.dirname", "os.path.basename", "os.path.splitext",
    "os.path.abspath", "os.path.normpath", "os.path.relpath", "os.path.expanduser",
    "os.sep", "posixpath.join",
))


def _io_call_in(node: ast.AST, bodies: Dict[str, ast.AST], *,
                _seen: Optional[set] = None, _depth: int = 0) -> Optional[str]:
    """The first I/O call reachable from *node*, following calls into *bodies*.

    Intraprocedural before this, which missed the commonest shape:
    ``OptimizedWeatherDownloader.__init__`` calls ``self.setup_logging()``, and it is
    ``setup_logging`` that calls ``os.makedirs``. The construction is exactly as unsafe either
    way, and the indirection is one line of code.
    """
    seen = _seen if _seen is not None else set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        fn = sub.func
        name = getattr(fn, "attr", None) or getattr(fn, "id", None)
        if not name:
            continue
        try:
            qualified = ast.unparse(fn)
        except Exception:
            qualified = name
        if name in _IO_CALLS and qualified not in _IO_SAFE_QUALIFIED:
            return f"{qualified}() at line {getattr(sub, 'lineno', 0)}"
        if _depth < 4 and name in bodies and name not in seen:
            seen.add(name)
            found = _io_call_in(bodies[name], bodies, _seen=seen, _depth=_depth + 1)
            if found:
                return f"{found} via {name}()"
    return None


def _construction_is_safe(node: ast.ClassDef, scope: ModuleScope,
                          by_name: Dict[str, Any],
                          class_tables: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """``None`` when the class can be constructed safely, else the reason it cannot.

    Conservative and syntactic. A class whose construction loads data is not independently
    callable — it is the object-shaped version of the hidden-global problem, and promoting it
    ships a unit that fails in a sandbox with no network and no staged files.

    Returns the REASON rather than a bool so the verdict can say which call disqualified the
    class; "performs I/O" with no location is not actionable.

    Four things are checked, in decreasing order of severity:

    * the **class body**, whose statements run at IMPORT of the slice, not at construction —
      a failure there takes down every sibling unit in the module, not just this one;
    * ``__new__`` and any class decorator, both of which also run before ``__init__``;
    * ``__init__``, following calls into the class's own methods and module-level functions;
    * runtime-global reads in ``__init__``, looked up in the CLASS's own method table.
    """
    methods = {s.name: s for s in node.body
               if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))}
    # The class's own methods first, then module-level defs -- so `self.setup_logging()`
    # resolves to this class's method rather than a same-named function elsewhere.
    bodies: Dict[str, ast.AST] = {}
    for name, target in scope.defs.items() if isinstance(scope.defs, dict) else []:
        if isinstance(target, ast.AST):
            bodies[name] = target
    bodies.update(methods)

    # 1. Class-body statements execute at class-creation time, i.e. on IMPORT.
    for stmt in node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Pass)):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue                                  # the docstring
        found = _io_call_in(stmt, {})                  # no call-following: body scope only
        if found:
            return f"class body performs I/O at import time: {found}"

    # 2. Anything else that runs before or instead of __init__.
    for deco in (node.decorator_list or []):
        found = _io_call_in(deco, bodies)
        if found:
            return f"class decorator performs I/O: {found}"
    new = methods.get("__new__")
    if new is not None:
        found = _io_call_in(new, bodies)
        if found:
            return f"__new__ performs I/O: {found}"

    init = methods.get("__init__")
    if init is None:
        return None            # the default constructor does nothing

    # 3. Runtime-global reads, from THIS class's method table.
    #
    # `by_name.get("__init__")` was keyed by simple name across the whole module, so with more
    # than one class defining __init__ -- 4 of 26 promoted slices -- it read whichever table won
    # the dict, and the answer was about a different class entirely.
    table = None
    if class_tables is not None:
        ctable = class_tables.get(node.name)
        if ctable is not None:
            for child in _function_tables(ctable):
                if child.get_name() == "__init__":
                    table = child
                    break
    if table is None:
        table = by_name.get("__init__")
    if table is not None:
        reads, _writes = _globals_of(table)
        offender = next((n for n in sorted(reads) if n in scope.runtime), None)
        if offender:
            return f"__init__ reads module-level value {offender!r}"

    # 4. I/O in __init__, or in anything it calls.
    found = _io_call_in(init, bodies)
    if found:
        return f"__init__ performs I/O: {found}"
    return None


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
