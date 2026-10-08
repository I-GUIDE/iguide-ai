"""The method library: extracted units written as an importable Python package.

The agent composes extracted methods in Python rather than by chaining tool calls, so the
units have to land as real importable modules, not as more tool schemas. One tool per unit
would put every ingested function into every prompt — a 24-tool peer already costs ~3,900
tokens of schema.

Two properties get the most attention here:

* ``get(symbol)`` really imports and returns a callable (a registry that only *describes*
  units is not a library), and
* symbol collisions are admitted rather than silently resolved. Measured on the real corpus,
  keying the registry by bare name turned 40 modules into 37 entries, because three function
  names are each defined by two different notebooks and the later ingest overwrote the
  earlier. A resolver returning "whichever element was ingested last" is worse than one that
  refuses.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from extractors.base import (EMIT_LIBRARY, EMIT_OPENSEARCH, KIND_METHOD_UNIT,
                             AssetRecord)
from extractors.emitters import library_emitter
from extractors.manifest import UnifiedManifest


def _unit_asset(symbol, element_id, source, *, sha, title="Demo", targets=(EMIT_OPENSEARCH, EMIT_LIBRARY)):
    return AssetRecord(
        asset_id=f"{element_id}::unit::{symbol}",
        kind=KIND_METHOD_UNIT,
        resource_type="MethodUnit",
        doc_id=f"{element_id}::unit::{symbol}",
        emit_targets=list(targets),
        title=f"{symbol} — {title}",
        contents=f"def {symbol}(...)",
        unit={"qualified_name": symbol, "library_symbol": symbol, "slice_sha": sha,
              "signature": f"def {symbol}(x)", "doc_summary": f"{symbol} summary",
              "params": [], "returns": "", "invariants": [],
              "requirements": {"pip": ["six"]},
              "provenance": {"element_id": element_id, "commit_sha": "abc"}},
        slice_source=source,
    )


def _manifest(*assets):
    m = UnifiedManifest(repo_id="r", source_url="u", cloned_at="now")
    from extractors.base import ExtractionResult
    m.add_result("notebook", ExtractionResult(assets=list(assets)))
    return m


@pytest.fixture()
def lib(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    return tmp_path / "root"


def _import_in_subprocess(pkg_parent: Path, code: str):
    return subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(pkg_parent)!r})\n{code}"],
        capture_output=True, text=True, timeout=120)


# --------------------------------------------------------------- it is a real package

def test_a_unit_becomes_an_importable_callable(lib):
    src = "def double(x):\n    return x * 2\n"
    out = library_emitter.emit(_manifest(_unit_asset("double", "elem1", src, sha="aaa111")),
                               root=lib)
    assert out["units"] == 1 and len(out["written"]) == 1

    r = _import_in_subprocess(lib, "import iguide_methods as M; print(M.get('double')(21))")
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.strip() == "42", f"get() did not return a working callable: {r.stdout!r}"


def test_describe_returns_the_contract(lib):
    src = "def f(x):\n    return x\n"
    library_emitter.emit(_manifest(_unit_asset("f", "elem1", src, sha="bbb222")), root=lib)
    r = _import_in_subprocess(lib, (
        "import iguide_methods as M, json;"
        "d = M.describe('f');"
        "print(json.dumps({'sig': d['signature'], 'sha': d['slice_sha'],"
        " 'elem': d['provenance']['element_id']}))"))
    assert r.returncode == 0, r.stderr[-400:]
    d = json.loads(r.stdout)
    assert d["sig"] == "def f(x)"
    assert d["sha"] == "bbb222"
    assert d["elem"] == "elem1"


def test_the_package_init_has_no_import_side_effects(lib):
    library_emitter.emit(_manifest(_unit_asset("f", "e", "def f():\n    return 1\n", sha="c1")),
                         root=lib)
    r = _import_in_subprocess(lib, "import iguide_methods; print('clean')")
    assert r.returncode == 0 and "clean" in r.stdout


# --------------------------------------------------------------- collisions

def test_colliding_bare_names_are_ambiguous_not_overwritten(lib):
    """The measured bug: 40 modules produced 37 registry entries."""
    a = _unit_asset("load", "elemA", "def load():\n    return 'A'\n", sha="a1")
    b = _unit_asset("load", "elemB", "def load():\n    return 'B'\n", sha="b1")
    out = library_emitter.emit(_manifest(a, b), root=lib)
    assert len(out["written"]) == 2, "a unit was dropped"

    r = _import_in_subprocess(lib, (
        "import iguide_methods as M;"
        "d = M.describe('load');"
        "print('AMBIGUOUS' if d.get('ambiguous') else 'RESOLVED', len(d.get('candidates') or []))"))
    assert r.returncode == 0, r.stderr[-400:]
    assert r.stdout.split()[0] == "AMBIGUOUS"
    assert r.stdout.split()[1] == "2"


def test_an_ambiguous_bare_name_raises_rather_than_guessing(lib):
    a = _unit_asset("load", "elemA", "def load():\n    return 'A'\n", sha="a1")
    b = _unit_asset("load", "elemB", "def load():\n    return 'B'\n", sha="b1")
    library_emitter.emit(_manifest(a, b), root=lib)
    r = _import_in_subprocess(lib, (
        "import iguide_methods as M\n"
        "try:\n    M.get('load'); print('RESOLVED')\n"
        "except KeyError as e:\n    print('RAISED')"))
    assert r.stdout.strip() == "RAISED", "an ambiguous name silently resolved"


def test_qualified_names_stay_reachable_when_ambiguous(lib):
    a = _unit_asset("load", "elemA", "def load():\n    return 'A'\n", sha="a1", title="Alpha")
    b = _unit_asset("load", "elemB", "def load():\n    return 'B'\n", sha="b1", title="Beta")
    library_emitter.emit(_manifest(a, b), root=lib)
    r = _import_in_subprocess(lib, (
        "import iguide_methods as M;"
        "qs = [s for s in M.symbols() if s.endswith('.load')];"
        "print(len(qs), sorted(M.get(q)() for q in qs))"))
    assert r.returncode == 0, r.stderr[-400:]
    n, vals = r.stdout.split(maxsplit=1)
    assert n == "2"
    assert "'A'" in vals and "'B'" in vals, "both colliding units must remain callable"


# --------------------------------------------------------------- versioning + targets

def test_reingesting_unchanged_source_is_a_no_op(lib):
    a = _unit_asset("f", "e", "def f():\n    return 1\n", sha="same")
    library_emitter.emit(_manifest(a), root=lib)
    mod = next((lib / "iguide_methods").rglob("v_same.py"))
    before = (mod.read_text(), mod.stat().st_mtime_ns)
    library_emitter.emit(_manifest(a), root=lib)
    assert mod.read_text() == before[0]
    assert mod.stat().st_mtime_ns == before[1], "an unchanged slice was rewritten"


def test_an_edit_mints_a_new_module_and_keeps_the_old(lib):
    library_emitter.emit(_manifest(_unit_asset("f", "e", "def f():\n    return 1\n", sha="v1")),
                         root=lib)
    library_emitter.emit(_manifest(_unit_asset("f", "e", "def f():\n    return 2\n", sha="v2")),
                         root=lib)
    versions = sorted(p.name for p in (lib / "iguide_methods").rglob("v_*.py"))
    assert versions == ["v_v1.py", "v_v2.py"], "an old version was destroyed"


def test_units_without_the_library_target_are_ignored(lib):
    a = _unit_asset("f", "e", "def f():\n    return 1\n", sha="x", targets=(EMIT_OPENSEARCH,))
    out = library_emitter.emit(_manifest(a), root=lib)
    assert out["units"] == 0 and out["written"] == []


def test_a_unit_without_source_is_skipped_not_written(lib):
    a = _unit_asset("f", "e", "", sha="x")
    out = library_emitter.emit(_manifest(a), root=lib)
    assert out["written"] == []
    assert out["skipped"] and "missing" in out["skipped"][0]["reason"]


def test_requirements_are_written_per_element(lib):
    library_emitter.emit(_manifest(_unit_asset("f", "e", "def f():\n    return 1\n", sha="x")),
                         root=lib)
    req = next((lib / "iguide_methods").rglob("requirements.txt"))
    assert "six" in req.read_text()


# --------------------------------------------------------------- naming safety

@pytest.mark.parametrize("title", ["../../escape", "Weird Title! **", "", "123"])
def test_element_package_names_are_safe_identifiers(title):
    name = library_emitter.element_package("abc12345", title)
    assert "/" not in name and ".." not in name
    assert name.replace("_", "a").isalnum()


# --------------------------------------------------------------- import-line verification

def test_a_slice_that_does_not_define_the_symbol_is_skipped_not_advertised(lib):
    """Defense in depth behind the analyzer's needs_instance verdict.

    The emitter used to write ``from .v_sha import <symbol>`` on the emitter's word alone. For
    bound methods the slice defined only the CLASS, so the import raised — and since that line
    lives in the element's __init__.py, ONE bad unit took down every sibling unit in the
    element. Measured on the real corpus: 26 of 40 advertised import lines failed, of which
    only 24 were themselves methods.
    """
    class_only = "class Downloader:\n    def build_url(self):\n        return 1\n"
    out = library_emitter.emit(
        _manifest(_unit_asset("build_url", "elemA", class_only, sha="m1")), root=lib)
    assert out["written"] == []
    assert out["skipped"] and "module level" in out["skipped"][0]["reason"]


def test_one_unimportable_unit_cannot_break_its_siblings(lib):
    good = _unit_asset("good_fn", "elemA", "def good_fn():\n    return 'ok'\n", sha="g1")
    bad = _unit_asset("method", "elemA", "class C:\n    def method(self):\n        return 1\n",
                      sha="b1")
    out = library_emitter.emit(_manifest(good, bad), root=lib)
    assert len(out["written"]) == 1
    r = _import_in_subprocess(lib, "import iguide_methods as M; print(M.get('good_fn')())")
    assert r.returncode == 0, r.stderr[-400:]
    assert r.stdout.strip() == "ok"


def test_every_registry_import_line_actually_imports(lib):
    """The property the registry exists to guarantee, asserted end to end."""
    a = _unit_asset("alpha", "elemA", "def alpha():\n    return 1\n", sha="a1")
    b = _unit_asset("beta", "elemB", "def beta():\n    return 2\n", sha="b1")
    library_emitter.emit(_manifest(a, b), root=lib)
    registry = json.loads((lib / "iguide_methods" / "_registry.json").read_text())
    lines = [f"from {e['module']} import {e['library_symbol']}"
             for e in registry.values()
             if isinstance(e, dict) and not e.get("ambiguous") and e.get("module")]
    assert lines
    for line in lines:
        r = _import_in_subprocess(lib, line)
        assert r.returncode == 0, f"advertised import failed: {line}\n{r.stderr[-300:]}"


def test_a_module_level_constant_unit_is_importable(lib):
    """Not every unit is a def; a symbol bound by assignment is a legitimate export."""
    out = library_emitter.emit(
        _manifest(_unit_asset("TABLE", "elemA", "TABLE = {'a': 1}\n", sha="c1")), root=lib)
    assert len(out["written"]) == 1


def test_an_unparseable_slice_is_refused_rather_than_advertised(lib):
    out = library_emitter.emit(
        _manifest(_unit_asset("broken", "elemA", "def broken(:\n", sha="x1")), root=lib)
    assert out["written"] == []
    assert out["skipped"]


# --------------------------------------------------------------- lazy element packages

def test_importing_one_unit_does_not_import_its_siblings(lib):
    """The sandbox failure that made the pinned import line unusable.

    Python runs a parent package's __init__ before any submodule, so the advertised
    ``from iguide_methods.ke_x.v_sha import good`` executed EVERY sibling module first. In the
    real container that raised ``ModuleNotFoundError: pandas`` from a different unit than the
    one being imported, and it made each unit's declared requirements wrong — the true install
    set was the union over the whole element.
    """
    good = _unit_asset("good", "elemA", "def good():\n    return 'ok'\n", sha="g1")
    heavy = _unit_asset("heavy", "elemA",
                        "import a_package_that_does_not_exist\n\ndef heavy():\n    return 1\n",
                        sha="h1")
    library_emitter.emit(_manifest(good, heavy), root=lib)

    pkg = next(p for p in (lib / "iguide_methods").iterdir() if p.is_dir())
    r = _import_in_subprocess(lib, f"from iguide_methods.{pkg.name}.v_g1 import good; print(good())")
    assert r.returncode == 0, f"a sibling's import broke this one:\n{r.stderr[-400:]}"
    assert r.stdout.strip() == "ok"


def test_the_element_alias_resolves_lazily(lib):
    good = _unit_asset("good", "elemA", "def good():\n    return 'ok'\n", sha="g1")
    heavy = _unit_asset("heavy", "elemA",
                        "import a_package_that_does_not_exist\n\ndef heavy():\n    return 1\n",
                        sha="h1")
    library_emitter.emit(_manifest(good, heavy), root=lib)
    pkg = next(p for p in (lib / "iguide_methods").iterdir() if p.is_dir())
    r = _import_in_subprocess(lib, f"from iguide_methods.{pkg.name} import good; print(good())")
    assert r.returncode == 0, r.stderr[-400:]
    assert r.stdout.strip() == "ok"


def test_a_missing_symbol_on_an_element_package_raises_attribute_error(lib):
    library_emitter.emit(_manifest(_unit_asset("good", "elemA", "def good():\n    return 1\n",
                                               sha="g1")), root=lib)
    pkg = next(p for p in (lib / "iguide_methods").iterdir() if p.is_dir())
    r = _import_in_subprocess(lib, (
        f"import iguide_methods.{pkg.name} as P\n"
        "try:\n    P.nope; print('RESOLVED')\n"
        "except AttributeError:\n    print('RAISED')"))
    assert r.stdout.strip() == "RAISED"


def test_dir_lists_the_units_without_importing_them(lib):
    good = _unit_asset("good", "elemA", "def good():\n    return 1\n", sha="g1")
    heavy = _unit_asset("heavy", "elemA",
                        "import a_package_that_does_not_exist\n\ndef heavy():\n    return 1\n",
                        sha="h1")
    library_emitter.emit(_manifest(good, heavy), root=lib)
    pkg = next(p for p in (lib / "iguide_methods").iterdir() if p.is_dir())
    r = _import_in_subprocess(lib, (
        f"import iguide_methods.{pkg.name} as P; "
        "print(sorted(n for n in dir(P) if not n.startswith('_')))"))
    assert r.returncode == 0, r.stderr[-300:]
    assert "'good'" in r.stdout and "'heavy'" in r.stdout


# --------------------------------------------------------------- re-ingest must not accumulate

def _reg(lib):
    return json.loads(
        (lib / library_emitter.PACKAGE_NAME / "_registry.json").read_text(encoding="utf-8"))


def _q(element_id, symbol):
    """The registry's qualified key. Computed, because the element package name embeds the
    element TITLE, so hardcoding it makes the test fixture-shaped rather than behaviour-shaped."""
    return f"{library_emitter.element_package(element_id, 'Demo')}.{symbol}"


def _units(reg):
    return {k: v for k, v in reg.items()
            if isinstance(v, dict) and not v.get("alias_for") and not v.get("ambiguous")}


def test_a_unit_the_element_no_longer_defines_is_dropped(lib):
    """The registry was merge-only and there is no delete anywhere in the ingest path. Measured
    after one rebuild of the real corpus: 446 non-alias entries for 229 modules on disk, so ~217
    described units from earlier analyzer versions — and the SAME symbol appeared twice with
    different contracts, ``def DoubleConv()`` from an old build alongside the corrected
    ``DoubleConv(in_c, out_c)``.

    A stale contract is worse than a missing one: the agent is handed a signature that no longer
    matches the shipped code and has no way to tell. After the fix the count matches the modules
    on disk exactly, 229/229."""
    library_emitter.emit(_manifest(_unit_asset("kept", "elem1", "def kept(x):\n    return x\n",
                                               sha="aaa"),
                                   _unit_asset("gone", "elem1", "def gone(x):\n    return x\n",
                                               sha="bbb")), root=lib)
    assert set(_units(_reg(lib))) == {_q("elem1", "kept"), _q("elem1", "gone")}

    # re-extract elem1; it no longer defines `gone`
    out = library_emitter.emit(
        _manifest(_unit_asset("kept", "elem1", "def kept(x):\n    return x\n", sha="aaa")),
        root=lib)
    assert set(_units(_reg(lib))) == {_q("elem1", "kept")}
    assert _q("elem1", "gone") in out.get("pruned", []), "a prune must be reported, never silent"


def test_pruning_never_touches_an_element_this_run_did_not_visit(lib):
    """6 of 180 elements fail to fetch on any given run. Dropping their working units because a
    network call failed would be a worse bug than the staleness this fixes."""
    library_emitter.emit(_manifest(_unit_asset("a", "elem1", "def a():\n    pass\n", sha="aaa")),
                         root=lib)
    library_emitter.emit(_manifest(_unit_asset("b", "elem2", "def b():\n    pass\n", sha="bbb")),
                         root=lib)
    assert set(_units(_reg(lib))) == {_q("elem1", "a"), _q("elem2", "b")}

    library_emitter.emit(_manifest(_unit_asset("a", "elem1", "def a():\n    pass\n", sha="aaa")),
                         root=lib)
    assert _q("elem2", "b") in _units(_reg(lib)), "elem2 was not re-extracted; its units must survive"


def test_a_bare_alias_does_not_dangle_after_its_target_is_pruned(lib):
    library_emitter.emit(_manifest(_unit_asset("gone", "elem1", "def gone():\n    pass\n",
                                               sha="bbb")), root=lib)
    assert _reg(lib)["gone"]["alias_for"] == _q("elem1", "gone")

    library_emitter.emit(_manifest(_unit_asset("other", "elem1", "def other():\n    pass\n",
                                               sha="ccc")), root=lib)
    reg = _reg(lib)
    assert "gone" not in reg, "an alias pointing at a pruned entry resolves to nothing"
    assert reg["other"]["alias_for"] == _q("elem1", "other")


def test_an_ambiguous_name_becomes_unambiguous_again_when_one_definer_drops_it(lib):
    """Ambiguity is recomputed from what survives, not accumulated. Left as a permanent stub, a
    name that is no longer contested would keep refusing to resolve forever."""
    library_emitter.emit(_manifest(_unit_asset("dup", "elem1", "def dup():\n    pass\n", sha="a"),
                                   ), root=lib)
    library_emitter.emit(_manifest(_unit_asset("dup", "elem2", "def dup():\n    pass\n", sha="b"),
                                   ), root=lib)
    assert _reg(lib)["dup"].get("ambiguous") is True

    library_emitter.emit(_manifest(_unit_asset("kept", "elem2", "def kept():\n    pass\n",
                                               sha="c")), root=lib)
    reg = _reg(lib)
    assert reg["dup"].get("ambiguous") is not True
    assert reg["dup"]["alias_for"] == _q("elem1", "dup")


def test_unit_kind_reaches_the_registry(lib):
    """0 of 657 entries carried it. "class" vs "function" is an ACTION difference for the
    caller — construct an object, or call a function — and the extractor had it all along."""
    asset = _unit_asset("Thing", "elem1", "class Thing:\n    pass\n", sha="aaa")
    asset.unit["unit_kind"] = "class"
    asset.unit["signature"] = "Thing(a, b)  # class"
    library_emitter.emit(_manifest(asset), root=lib)
    assert _reg(lib)[_q("elem1", "Thing")]["unit_kind"] == "class"


# ------------------------------------------------------- two files, one symbol name

def _two_file_element(tmp_path, name_a="readers.py", name_b="writers.py", symbol="load"):
    """One element whose repository defines the same symbol in two files."""
    from extractors.base import EMIT_LIBRARY, EMIT_OPENSEARCH, ExtractContext
    from extractors.code_extractor import CodeExtractor
    from extractors.manifest import UnifiedManifest

    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / name_a).write_text(
        f'def {symbol}(path):\n    """Read the gauge table."""\n    return open(path).read()\n')
    (repo / name_b).write_text(
        f'def {symbol}(path):\n    """Read the basin polygons."""\n    return open(path).read(2)\n')

    man = UnifiedManifest(element_id="elem0001", element_type="code")
    for fname in (name_a, name_b):
        ctx = ExtractContext(element_id="elem0001", element_type="code",
                             fields={"title": "Two files"},
                             targets=[EMIT_OPENSEARCH, EMIT_LIBRARY],
                             extra={"repo_dir": str(repo)})
        result = CodeExtractor().extract(str(repo / fname), ctx=ctx)
        man.assets.extend(result.assets)
        man.provenance_edges.extend(result.edges)
    return man


def _real_units(registry):
    return {k: v for k, v in registry.items()
            if isinstance(v, dict) and v.get("signature")
            and not v.get("alias_for") and not v.get("ambiguous")}


def test_two_files_defining_one_symbol_both_survive(tmp_path, monkeypatch):
    """The registry key was ``{element_package}.{symbol}``, which is not unique when one element
    spans several files — and an element IS a whole repository.

    Both modules were written (they are content-addressed, so their shas differ) and the second
    registry write dropped the first, leaving a live module on disk that nothing pointed at.
    Measured on the corpus before this fix: 2641203f had 129 callable units and 92 registry
    entries across 35 source files; 76485230 had 85 and 70; 187ec685 had 7 and 3. **58 of 362
    callable code units, 16%, were written and then made unreachable.**

    The cross-element form of this bug was fixed once before, by introducing the very key that
    leaves this case open.
    """
    root = tmp_path / "lib"
    man = _two_file_element(tmp_path)
    summary = library_emitter.emit(man, root=root)
    registry = json.loads(
        (root / "iguide_methods" / "_registry.json").read_text(encoding="utf-8"))

    units = _real_units(registry)
    assert len(units) == 2, f"a unit was overwritten: {sorted(units)}"
    assert len(summary["written"]) == 2

    by_source = {(v.get("provenance") or {}).get("source_rel_path"): k for k, v in units.items()}
    assert set(by_source) == {"readers.py", "writers.py"}
    # Each key must resolve to the module holding THAT file's code, not the other's.
    assert "Read the gauge table." == units[by_source["readers.py"]]["doc_summary"]
    assert "Read the basin polygons." == units[by_source["writers.py"]]["doc_summary"]
    assert (units[by_source["readers.py"]]["module"]
            != units[by_source["writers.py"]]["module"])


def test_the_colliding_short_name_becomes_an_ambiguity_not_a_winner(tmp_path):
    """Picking one silently is what the old code did by accident. An explicit ambiguity is the
    same answer the cross-element case already gives."""
    root = tmp_path / "lib"
    library_emitter.emit(_two_file_element(tmp_path), root=root)
    registry = json.loads(
        (root / "iguide_methods" / "_registry.json").read_text(encoding="utf-8"))

    short = "ke_elem0001_readers_py.load"
    assert registry[short]["ambiguous"] is True
    assert len(registry[short]["candidates"]) == 2
    for candidate in registry[short]["candidates"]:
        assert candidate in registry


def test_every_advertised_module_exists_on_disk(tmp_path):
    """The failure mode was a module written and never pointed at. Its mirror — a registry entry
    pointing at a module that is not there — must not be introduced by the fix."""
    root = tmp_path / "lib"
    library_emitter.emit(_two_file_element(tmp_path), root=root)
    registry = json.loads(
        (root / "iguide_methods" / "_registry.json").read_text(encoding="utf-8"))
    pkg = root / "iguide_methods"
    for key, value in _real_units(registry).items():
        relative = str(value["module"]).split(".", 1)[1].replace(".", "/") + ".py"
        assert (pkg / relative).is_file(), f"{key} advertises a missing module"


def test_re_emitting_the_same_file_overwrites_rather_than_duplicating(tmp_path):
    """Only a DIFFERENT source is a second unit. An edit to the same file is a new version and
    must replace, or every re-ingest would grow the registry."""
    root = tmp_path / "lib"
    library_emitter.emit(_two_file_element(tmp_path), root=root)
    library_emitter.emit(_two_file_element(tmp_path), root=root)
    registry = json.loads(
        (root / "iguide_methods" / "_registry.json").read_text(encoding="utf-8"))
    assert len(_real_units(registry)) == 2


def test_an_element_with_no_collision_keeps_the_plain_key(tmp_path):
    """The disambiguated key is uglier and appears in the import line, so it must only appear
    where it is actually needed."""
    from extractors.base import EMIT_LIBRARY, EMIT_OPENSEARCH, ExtractContext
    from extractors.code_extractor import CodeExtractor
    from extractors.manifest import UnifiedManifest

    repo = tmp_path / "solo"
    repo.mkdir()
    (repo / "only.py").write_text(
        'def load_gauges(path):\n    """Read the gauge table."""\n    return open(path).read()\n')
    man = UnifiedManifest(element_id="elem0002", element_type="code")
    ctx = ExtractContext(element_id="elem0002", element_type="code",
                         fields={"title": "One file"},
                         targets=[EMIT_OPENSEARCH, EMIT_LIBRARY],
                         extra={"repo_dir": str(repo)})
    result = CodeExtractor().extract(str(repo / "only.py"), ctx=ctx)
    man.assets.extend(result.assets)

    root = tmp_path / "lib"
    library_emitter.emit(man, root=root)
    registry = json.loads(
        (root / "iguide_methods" / "_registry.json").read_text(encoding="utf-8"))
    keys = sorted(_real_units(registry))
    assert keys == ["ke_elem0002_only_py.load_gauges"], keys
