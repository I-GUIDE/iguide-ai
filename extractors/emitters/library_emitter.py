"""Write callable units as an importable Python package: the method library.

    storage_root()/method_library/
      iguide_methods/
        __init__.py                     generated: MANIFEST, describe(), get()
        _registry.json                  symbol -> module, element, contract, requirements
        ke_cca9b545_chicago_crime/
          __init__.py                   re-exports the CURRENT version
          v_ab12cd34.py                 THE SLICE, content-addressed
          requirements.txt

Why a package rather than more tools
------------------------------------
The agent composes extracted methods **in Python**, not by chaining tool calls. Python
composition is strictly more expressive — loops, conditionals, intermediate values, error
recovery in-loop — and it keeps the tool schema O(1) in the number of ingested units. Adding
one tool per unit would put every extracted function into every prompt; measured earlier in
this project, a 24-tool peer already spends ~3,900 tokens on schemas alone.

Versioning
----------
The module filename carries ``slice_sha``, so re-ingesting an unchanged notebook rewrites a
byte-identical file (an idempotent no-op) while an edited function mints a NEW module and the
package ``__init__`` repoints to it. Older versions stay importable by sha, which is what
makes "which version produced this number" a resolvable question rather than a label.
"""

from __future__ import annotations

import json
import keyword
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import EMIT_LIBRARY
from ..manifest import UnifiedManifest

PACKAGE_NAME = "iguide_methods"
_IDENT_RE = re.compile(r"[^0-9a-zA-Z_]+")


def _storage_root() -> Path:
    from agent_runtime.file_store import storage_root
    return Path(storage_root())


def library_root() -> Path:
    root = _storage_root() / "method_library"
    root.mkdir(parents=True, exist_ok=True)
    return root


def package_dir() -> Path:
    p = library_root() / PACKAGE_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ident(text: str, *, fallback: str = "x") -> str:
    """A safe Python identifier fragment. Never let an element title become a path."""
    s = _IDENT_RE.sub("_", str(text or "")).strip("_").lower()
    if not s:
        s = fallback
    if s[0].isdigit():
        s = f"_{s}"
    if keyword.iskeyword(s):
        s = f"{s}_"
    return s[:48]


def _element_init(element_id: str, exports: List[tuple]) -> str:
    """Element package init that resolves symbols LAZILY (PEP 562 module ``__getattr__``).

    It used to eagerly re-export every unit::

        from .v_38f5f29289b4 import filter_dataframe_by_value as filter_dataframe_by_value
        from .v_45b703eec714 import load_chicago_crime_data as load_chicago_crime_data
        ...

    Python runs a parent package's ``__init__`` before any submodule, so
    ``from iguide_methods.ke_x.v_45b703eec714 import load_chicago_crime_data`` — the pinned
    line the registry advertises — executed *every* sibling module first. In the real sandbox
    that failed with ``ModuleNotFoundError: pandas``, raised by a DIFFERENT unit than the one
    being imported. It also made each unit's declared ``requirements`` wrong: the true install
    set was the union over every unit in the element.

    Lazily, importing one unit imports one module. ``from iguide_methods.ke_x import f`` still
    works, resolving only f's module.
    """
    table = ",\n".join(f"    {sym!r}: {mod!r}" for sym, mod in sorted(exports))
    return (
        f'"""Methods extracted from element {element_id}. Generated — do not edit.\n\n'
        f'Symbols resolve lazily, so importing one unit does not import its siblings\n'
        f'(and does not require their dependencies).\n"""\n\n'
        f"from importlib import import_module\n\n"
        f"_UNITS = {{\n{table},\n}}\n\n"
        f"__all__ = sorted(_UNITS)\n\n\n"
        f"def __getattr__(name):\n"
        f"    module = _UNITS.get(name)\n"
        f"    if module is None:\n"
        f"        raise AttributeError(name)\n"
        f"    return getattr(import_module(f'.{{module}}', __name__), name)\n\n\n"
        f"def __dir__():\n"
        f"    return sorted(set(globals()) | set(_UNITS))\n"
    )


def _defines_at_module_level(source: str, symbol: str) -> bool:
    """Does *source* bind *symbol* at module level, so ``from <mod> import <symbol>`` works?

    Parsed, not grepped: a method named in a class body, or the symbol appearing only inside
    another function, must not count. An unparseable slice returns False — refusing to
    advertise an import we cannot verify is the safe direction.
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if stmt.name == symbol:
                return True
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id == symbol:
                    return True
                if isinstance(t, (ast.Tuple, ast.List)) and any(
                        isinstance(e, ast.Name) and e.id == symbol for e in t.elts):
                    return True
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for alias in stmt.names:
                if (alias.asname or alias.name.split(".")[0]) == symbol:
                    return True
    return False


def element_package(element_id: str, title: str = "") -> str:
    """Per-element subpackage name: ``ke_<id8>_<slug>``.

    Namespaced by element id because two notebooks may both define ``load_data``; the id
    keeps them apart while the slug keeps the import readable.
    """
    short = _ident(str(element_id or "unknown")[:8], fallback="unknown")
    slug = _ident(title, fallback="element")
    return f"ke_{short}_{slug}" if slug != "element" else f"ke_{short}"


def _unit_assets(manifest: UnifiedManifest) -> List[Dict[str, Any]]:
    out = []
    for asset in manifest.assets or []:
        a = asset if isinstance(asset, dict) else getattr(asset, "__dict__", {})
        if EMIT_LIBRARY not in (a.get("emit_targets") or []):
            continue
        if not a.get("unit"):
            continue
        out.append(a)
    return out


def _requirements_for(units: List[Dict[str, Any]]) -> List[str]:
    reqs: set = set()
    for a in units:
        for spec in ((a.get("unit") or {}).get("requirements") or {}).get("pip", []) or []:
            if spec:
                reqs.add(str(spec))
    return sorted(reqs)


_PKG_INIT = '''"""Validated methods extracted from I-GUIDE knowledge elements.

Generated — do not edit. Regenerate by re-ingesting the source elements.

    from iguide_methods import describe, get
    describe("load_crime_points")     # contract: params, units, CRS expectation, provenance
    fn = get("load_crime_points")     # the callable itself

Every symbol here was verified to be *independently callable*: its slice carries the imports,
constants and helper functions it needs, and contains no module-level statement that executes.
Importing one has no side effects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

_REGISTRY_PATH = Path(__file__).with_name("_registry.json")


def _registry() -> Dict[str, Any]:
    try:
        return json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


MANIFEST = _registry()


def symbols() -> list:
    """Every callable symbol in the library."""
    return sorted(MANIFEST.keys())


def describe(symbol: str) -> Optional[Dict[str, Any]]:
    """Full contract for *symbol*: signature, params, declared units, CRS expectation,
    requirements and provenance back to the source element."""
    return MANIFEST.get(symbol)


def get(symbol: str):
    """Import and return the callable for *symbol*.

    Accepts a bare name when it is unambiguous, or a qualified ``<element_pkg>.<name>``.
    A bare name defined by more than one element raises rather than guessing.
    """
    entry = MANIFEST.get(symbol)
    if not entry:
        raise KeyError(f"unknown method {symbol!r}; available: {symbols()[:20]}")
    if entry.get("ambiguous"):
        raise KeyError(
            f"{symbol!r} is defined by more than one element. Use one of: "
            f"{entry.get('candidates')}"
        )
    import importlib

    module = importlib.import_module(entry["module"])
    return getattr(module, entry["library_symbol"])


__all__ = ["MANIFEST", "symbols", "describe", "get"]
'''


def emit(manifest: UnifiedManifest, *, root: Optional[Path] = None,
         dry_run: bool = False) -> Dict[str, Any]:
    """Write every EMIT_LIBRARY unit in *manifest* into the method library."""
    units = _unit_assets(manifest)
    summary: Dict[str, Any] = {"units": len(units), "written": [], "skipped": [],
                               "package": PACKAGE_NAME}
    if not units:
        return summary

    pkg = Path(root) / PACKAGE_NAME if root else package_dir()
    pkg.mkdir(parents=True, exist_ok=True)

    by_element: Dict[str, List[Dict[str, Any]]] = {}
    for a in units:
        prov = (a.get("unit") or {}).get("provenance") or {}
        by_element.setdefault(str(prov.get("element_id") or prov.get("parent_doc_id") or "unknown"),
                              []).append(a)

    registry: Dict[str, Any] = {}
    emitted: set = set()          # qualified keys written by THIS run
    touched: set = set()          # element packages this run re-extracted
    registry_path = pkg / "_registry.json"
    if registry_path.exists() and not dry_run:
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8")) or {}
        except ValueError:
            registry = {}

    for element_id, element_units in by_element.items():
        title = str(element_units[0].get("title") or "").split(" — ")[-1]
        subpkg_name = element_package(element_id, title)
        subpkg = pkg / subpkg_name
        exports: List[tuple] = []          # (symbol, module_name)

        claims: Dict[str, str] = {}          # registry key -> the source file that owns it
        collided: Dict[str, set] = {}        # short key -> the source stems fighting over it
        for a in element_units:
            unit = a.get("unit") or {}
            sha = str(unit.get("slice_sha") or "")
            source = a.get("slice_source") or ""
            symbol = str(unit.get("library_symbol") or unit.get("qualified_name") or "")
            if not (sha and source and symbol):
                summary["skipped"].append({"unit": unit.get("qualified_name"),
                                           "reason": "missing slice_sha, source or symbol"})
                continue
            if not _defines_at_module_level(source, symbol):
                # Never advertise an import that was not verified against the slice itself.
                # Bound methods reached here with a bare method name while the slice defined
                # only their CLASS, so `from .v_sha import build_api_url` raised ImportError —
                # and because that line lives in the element's __init__, it took every sibling
                # unit in the element down with it: 26 of 40 import lines failed on the real
                # corpus. The analyzer now verdicts methods needs_instance so they never get
                # here, and this check keeps any future promotion bug local to one unit.
                summary["skipped"].append({
                    "unit": unit.get("qualified_name"),
                    "reason": f"slice does not define {symbol!r} at module level"})
                continue
            module_name = f"v_{sha}"
            if not dry_run:
                subpkg.mkdir(parents=True, exist_ok=True)
                target = subpkg / f"{module_name}.py"
                # Content-addressed: identical source -> identical bytes -> a no-op rewrite.
                if not target.exists() or target.read_text(encoding="utf-8") != source:
                    target.write_text(source, encoding="utf-8")
            exports.append((symbol, module_name))
            dotted = f"{PACKAGE_NAME}.{subpkg_name}.{module_name}"
            unit["library_module"] = dotted
            entry = {
                "module": dotted,
                "library_symbol": symbol,
                "qualified_name": unit.get("qualified_name"),
                "element_package": subpkg_name,
                "signature": unit.get("signature"),
                # "class" vs "function" is an ACTION difference for the caller -- construct an
                # object or call a function -- and it was dropped here, so 0 of 657 registry
                # entries carried it while the extractor had it all along.
                "unit_kind": unit.get("unit_kind"),
                "doc_summary": unit.get("doc_summary"),
                "params": unit.get("params"),
                "returns": unit.get("returns"),
                "invariants": unit.get("invariants"),
                "requirements": unit.get("requirements"),
                "slice_sha": sha,
                "provenance": unit.get("provenance"),
            }
            # The registry is keyed by the FULLY QUALIFIED name, plus a bare alias only when
            # the short name is unambiguous. Keying by the bare symbol alone silently lost
            # units: measured on the real corpus, 40 modules produced only 37 registry
            # entries because `generate_random`, `generate_random_loc` and `get_url` are each
            # defined by two different notebooks, and the later ingest overwrote the earlier.
            # A resolver that returns "whichever element was ingested last" is worse than one
            # that admits the ambiguity.
            #
            # `{package}.{symbol}` fixed that ACROSS elements and left the same bug WITHIN one.
            # An element is a whole repository: 2641203f spans 35 source files, and two files
            # defining `load` collide on one key. Both modules get written -- they are
            # content-addressed, so the shas differ -- and the second registry write drops the
            # first, leaving a live module on disk that nothing points at.
            #
            # Measured on the corpus: 129 callable units in 2641203f, 92 registry entries; 85 and
            # 70 in 76485230; 7 and 3 in 187ec685. **58 of 362 callable code units, 16%, were
            # written and then made unreachable.** It also means part of the "617 modules for 551
            # units" gap I previously attributed entirely to superseded versions is this instead.
            #
            # Same remedy as the cross-element case, one level down: the source file disambiguates,
            # and the colliding short key becomes an explicit ambiguity rather than a winner.
            source_stem = _ident(Path(str((unit.get("provenance") or {}).get(
                "source_rel_path") or "")).stem, fallback="src")
            short = f"{subpkg_name}.{symbol}"
            qualified = short
            holder = claims.get(short)
            if holder is not None and holder != source_stem:
                # Re-registering the SAME source is an edit and must overwrite. A DIFFERENT source
                # is a genuine second unit, so both move to file-qualified keys -- including the
                # one that got here first, which is currently sitting on the short key and would
                # otherwise be destroyed when that key becomes the ambiguity stub.
                if short in registry:
                    moved = f"{subpkg_name}.{holder}.{symbol}"
                    registry[moved] = registry.pop(short)
                    emitted.discard(short)
                    emitted.add(moved)
                    claims[moved] = holder
                qualified = f"{subpkg_name}.{source_stem}.{symbol}"
                collided.setdefault(short, set()).update({holder, source_stem})
            claims[qualified] = source_stem
            claims.setdefault(short, source_stem)
            registry[qualified] = entry
            emitted.add(qualified)
            touched.add(subpkg_name)
            summary["written"].append(f"{subpkg_name}/{module_name}.py::{symbol}")

        for short_key, stems in collided.items():
            # The element package cannot re-export a name two of its files define. Saying so beats
            # picking one, which is what the old code did by accident.
            symbol_name = short_key.rsplit(".", 1)[-1]
            candidates = sorted(f"{subpkg_name}.{stem}.{symbol_name}" for stem in stems)
            registry[short_key] = {
                "ambiguous": True, "library_symbol": symbol_name,
                "candidates": candidates, "element_package": subpkg_name,
                "doc_summary": f"{symbol_name!r} is defined by {len(stems)} files of this "
                               f"element; import it by its qualified name."}
            emitted.add(short_key)
        ambiguous_symbols = {k.rsplit(".", 1)[-1] for k in collided}
        exports = [(sym, mod) for sym, mod in exports if sym not in ambiguous_symbols]
        if collided:
            summary.setdefault("ambiguous_within_element", []).extend(sorted(collided))

        if exports and not dry_run:
            (subpkg / "__init__.py").write_text(
                _element_init(element_id, exports), encoding="utf-8")
            reqs = _requirements_for(element_units)
            if reqs:
                (subpkg / "requirements.txt").write_text("\n".join(reqs) + "\n", encoding="utf-8")

    # Prune entries for elements this run RE-EXTRACTED but that it no longer defines, then
    # rebuild every bare alias from what survives.
    #
    # The registry was merge-only, and there is no delete anywhere in the ingest path. Measured
    # after one rebuild: 446 non-alias entries for 229 modules on disk -- so ~217 entries
    # described units from earlier analyzer versions. The same symbol appeared twice with
    # DIFFERENT contracts (`def DoubleConv()` from an old build alongside the correct
    # `DoubleConv(in_c, out_c)`), and `library_summary` reported 446 units where 229 exist. A
    # stale contract is worse than a missing one: the agent is handed a signature that no longer
    # matches the shipped code and has no way to tell.
    #
    # Scoped to `touched`: an element that failed to fetch this run keeps its entries, because
    # dropping working units on a transient network error would be its own bug.
    for key in [k for k, v in list(registry.items())
                if isinstance(v, dict) and not v.get("alias_for") and not v.get("ambiguous")
                and v.get("element_package") in touched and k not in emitted]:
        registry.pop(key, None)
        summary.setdefault("pruned", []).append(key)

    by_symbol: Dict[str, List[str]] = {}
    for key, value in registry.items():
        if isinstance(value, dict) and not value.get("alias_for") and not value.get("ambiguous"):
            by_symbol.setdefault(str(value.get("library_symbol") or key.split(".")[-1]),
                                 []).append(key)
    for symbol, keys in by_symbol.items():
        if len(keys) == 1:
            registry[symbol] = dict(registry[keys[0]], alias_for=keys[0])
        else:
            # A resolver that returns "whichever element was ingested last" is worse than one
            # that admits the ambiguity.
            registry[symbol] = {
                "ambiguous": True, "library_symbol": symbol, "candidates": sorted(keys),
                "doc_summary": f"{symbol!r} is defined by more than one element; "
                               f"import it by its qualified name."}
    # A bare alias whose qualified target was pruned would otherwise dangle.
    for key in [k for k, v in list(registry.items())
                if isinstance(v, dict) and v.get("alias_for")
                and v["alias_for"] not in registry]:
        registry.pop(key, None)

    if not dry_run:
        (pkg / "__init__.py").write_text(_PKG_INIT, encoding="utf-8")
        registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    summary["registry_size"] = len(registry)
    summary["root"] = str(pkg)
    return summary


__all__ = ["emit", "library_root", "package_dir", "element_package", "PACKAGE_NAME"]
