"""Slice emission: a standalone, importable, side-effect-free module for one unit.

The central test here is `test_slice_imports_and_runs_in_a_subprocess`. Static checks are not
sufficient for this module, and that is not a hypothetical: the first implementation collected
requirements for the *target* unit only, so slicing `good` (which calls `helper`, which reads
the const `THRESH`) emitted `helper` but not `THRESH`. The slice parsed, contained no side
effects, passed every static assertion — and would have raised `NameError` on the first call.
Only importing and calling it catches that, so that is what these tests do.

The other property under test is what a slice must NOT contain: module-level runtime bindings
and side-effect statements. That omission is the whole safety argument for shipping extracted
code, and it is why a `needs_globals` unit is refused rather than patched.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

import pytest

from extractors.analysis import (analyze_module, build_module_source, build_unit_slice,
                                 has_module_side_effects, slice_sha)

DIRTY_MODULE = '''
import os
import geopandas as gpd
THRESH = 3
SCALE = 2
LOADED = open("/dev/null").read()
print("side effect at import")
for _i in range(2):
    pass

def helper(x):
    return x * THRESH

def deeper(x):
    return helper(x) + SCALE

def good(n):
    """Transitively needs deeper, helper, THRESH and SCALE."""
    return deeper(n)

def bad(n):
    return LOADED[:n]
'''


@pytest.fixture()
def sliced():
    verdicts, scope, summary = analyze_module(DIRTY_MODULE)
    return verdicts, scope, summary


def _slice(name, sliced):
    verdicts, scope, _ = sliced
    return build_unit_slice(DIRTY_MODULE, name, scope=scope, verdicts=verdicts)


# --------------------------------------------------------------- the real check

def test_slice_imports_and_runs_in_a_subprocess(sliced):
    """THE test. A slice that parses but cannot be called is worthless."""
    src = _slice("good", sliced)
    assert src, "good is callable and must produce a slice"
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "unit.py").write_text(src)
    r = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {str(d)!r});"
         " import unit; print(unit.good(5))"],
        capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"slice failed to import/run:\n{r.stderr[-600:]}"
    assert r.stdout.strip() == "17", f"wrong result: {r.stdout!r}"


def test_transitive_consts_are_included(sliced):
    """The bug this file was written for: helper reads THRESH, good does not."""
    src = _slice("good", sliced)
    assert "THRESH = 3" in src, "a const needed by a DEPENDENCY was dropped"
    assert "SCALE = 2" in src
    assert "def helper" in src and "def deeper" in src


def test_dependencies_are_emitted_before_their_callers(sliced):
    src = _slice("good", sliced)
    assert src.index("def helper") < src.index("def deeper") < src.index("def good")


# --------------------------------------------------------------- what must be absent

def test_runtime_bindings_are_never_emitted(sliced):
    """Importing a slice must not read a file, hit an API, or block on a credential."""
    src = _slice("good", sliced)
    assert "LOADED" not in src
    assert "open(" not in src


def test_module_side_effects_are_never_emitted(sliced):
    src = _slice("good", sliced)
    assert "print(" not in src
    assert "for _i in range" not in src
    assert has_module_side_effects(src) is False


def test_unused_imports_are_not_carried(sliced):
    """A slice declares only what it needs, so its requirements list means something."""
    src = _slice("good", sliced)
    assert "import os" not in src
    assert "import geopandas" not in src, "good never uses gpd"


def test_needed_imports_are_carried():
    src = "import geopandas as gpd\n\ndef load(p):\n    return gpd.read_file(p)\n"
    verdicts, scope, _ = analyze_module(src)
    out = build_unit_slice(src, "load", scope=scope, verdicts=verdicts)
    assert "import geopandas as gpd" in out


# --------------------------------------------------------------- refusal

def test_a_needs_globals_unit_is_refused(sliced):
    """Shipping it would require inlining the very statements slices exclude."""
    assert _slice("bad", sliced) == ""


def test_an_unknown_unit_is_refused(sliced):
    assert _slice("does_not_exist", sliced) == ""


def test_an_unparseable_module_yields_no_slice():
    verdicts, scope, _ = analyze_module("def broken(:\n pass")
    assert build_unit_slice("def broken(:\n pass", "broken",
                            scope=scope, verdicts=verdicts) == ""


# --------------------------------------------------------------- provenance + versioning

def test_provenance_header_is_present(sliced):
    verdicts, scope, _ = sliced
    src = build_unit_slice(DIRTY_MODULE, "good", scope=scope, verdicts=verdicts,
                           provenance={"element_id": "cca9b545", "cell_order": 4,
                                       "commit_sha": "abc123"})
    for token in ("cca9b545", "cell_order", "abc123", "unit          : good"):
        assert token in src


def test_slice_sha_is_stable_and_content_addressed(sliced):
    a = _slice("good", sliced)
    assert slice_sha(a) == slice_sha(a), "re-ingesting unchanged source must be a no-op"
    assert slice_sha(a) != slice_sha(a + "\n# edit\n"), "an edit must mint a new version"
    assert len(slice_sha(a)) == 12


# --------------------------------------------------------------- helpers

def test_decorators_that_would_not_resolve_are_stripped():
    """@tool comes from the agent runtime; a slice must be a plain callable."""
    src = "from langchain_core.tools import tool\n\n@tool\ndef f(x):\n    return x\n"
    verdicts, scope, _ = analyze_module(src)
    out = build_unit_slice(src, "f", scope=scope, verdicts=verdicts)
    assert out, "a decorated but self-contained function should still slice"
    assert "@tool" not in out


def test_annotation_only_imports_are_carried():
    """A name used ONLY in an annotation still has to be importable.

    symtable does not report annotation references as function globals — correctly, since
    annotations evaluate in the enclosing scope at def time. But the slice carries the
    annotation, so omitting the import makes the slice fail at IMPORT time.

    Measured on the real corpus: `def filter_dataframe_by_value(df: pd.DataFrame, ...) ->
    pd.DataFrame` uses `pd` nowhere in its body. It was marked callable and its slice died with
    `NameError: name 'pd' is not defined`. Corpus import rate 37/40 -> 39/40 after this fix.
    """
    src = ("import pandas as pd\n\n"
           "def f(df: pd.DataFrame, k: int) -> pd.DataFrame:\n"
           "    return df.head(k)\n")
    verdicts, scope, _ = analyze_module(src)
    assert verdicts["f"].verdict == "callable"
    assert "pd" in verdicts["f"].requires_imports, "annotation-only name was not required"
    out = build_unit_slice(src, "f", scope=scope, verdicts=verdicts)
    assert "import pandas as pd" in out

    d = pathlib.Path(tempfile.mkdtemp())
    (d / "unit.py").write_text(out)
    r = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {str(d)!r}); import unit; print('ok')"],
        capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"annotation-only import still broke the slice:\n{r.stderr[-400:]}"


def test_return_annotation_is_also_covered():
    src = "import geopandas as gpd\n\ndef g(x) -> gpd.GeoDataFrame:\n    return x\n"
    verdicts, scope, _ = analyze_module(src)
    assert "gpd" in verdicts["g"].requires_imports


def test_stringized_annotations_do_not_require_an_import():
    """A quoted annotation is not evaluated at def time, so it cannot fail an import."""
    src = "def h(x: 'gpd.GeoDataFrame'):\n    return x\n"
    verdicts, _, _ = analyze_module(src)
    assert verdicts["h"].verdict == "callable"
    assert verdicts["h"].requires_imports == []
    assert verdicts["h"].free_names == []


def test_build_module_source_marks_cells():
    out = build_module_source("My Notebook", [(1, "a = 1"), (3, "b = 2")])
    assert "My Notebook" in out
    assert "# Cell 1" in out and "# Cell 3" in out


def test_has_module_side_effects_detects_a_computed_assignment():
    assert has_module_side_effects("X = compute()\n") is True
    assert has_module_side_effects("X = 1\ndef f():\n    return X\n") is False
    assert has_module_side_effects('"""doc"""\nimport os\n') is False
