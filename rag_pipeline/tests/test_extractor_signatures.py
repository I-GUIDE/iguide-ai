"""Code-asset extraction: callable units from .py modules, and the entry-point gate.

Two things this file pins.

**Code produces units.** Until now the extractor emitted API-surface descriptions only, so
`code` was an element type the system claimed to extract methods from and did not — the entire
203-unit corpus library was `notebook`. A .py module is actually a *better* source than a
notebook: no cell state, so far fewer functions are blocked by a runtime global. The same
analyzer decides, so a module-level `df = pd.read_csv(...)` still blocks its readers.

**Importing argparse is not a CLI.** `_entry_point` promoted any module that merely imported
argparse, which ships the whole file as `module_source`. A library module with one optional
command-line helper was indistinguishable from an actual entry point.
"""

from __future__ import annotations



# --------------------------------------------------------------- code units (M7)

def _code_units(source, targets=None):
    """Extract a .py module and return its promoted MethodUnit assets."""
    import tempfile
    from pathlib import Path

    from extractors.base import EMIT_LIBRARY, EMIT_OPENSEARCH, ExtractContext
    from extractors.code_extractor import CodeExtractor

    path = Path(tempfile.mkdtemp()) / "mod.py"
    path.write_text(source, encoding="utf-8")
    ctx = ExtractContext(element_id="code01", element_type="code", fields={"title": "M"},
                         targets=targets if targets is not None
                         else [EMIT_OPENSEARCH, EMIT_LIBRARY])
    result = CodeExtractor().extract(str(path), ctx=ctx)
    return [a for a in result.assets if getattr(a, "unit", None)]


def test_a_code_module_promotes_callable_units():
    """`code` was an element type the system claimed to extract methods from and did not:
    the whole 203-unit corpus library was `notebook`."""
    units = _code_units("import geopandas as gpd\n\n"
                        "def load_points(path):\n"
                        '    """Read a point layer."""\n'
                        "    return gpd.read_file(path)\n")
    assert len(units) == 1
    unit = units[0].unit
    assert unit["qualified_name"] == "load_points"
    assert unit["callability"]["verdict"] == "callable"
    assert unit["requirements"]["pip"] == ["geopandas"]


def test_a_code_unit_reading_a_runtime_global_is_blocked():
    units = _code_units("import pandas as pd\n"
                        "CONFIG = pd.read_csv('c.csv')\n\n"
                        "def needs_config(x):\n    return CONFIG.loc[x]\n")
    assert units[0].unit["callability"]["verdict"] == "needs_globals"


def test_only_callable_code_units_reach_the_library():
    from extractors.base import EMIT_LIBRARY

    units = _code_units("import pandas as pd\n"
                        "CFG = pd.read_csv('c.csv')\n\n"
                        "def good():\n    return 1\n\n"
                        "def bad():\n    return CFG\n")
    by_name = {a.unit["qualified_name"]: a for a in units}
    assert EMIT_LIBRARY in by_name["good"].emit_targets
    assert EMIT_LIBRARY not in by_name["bad"].emit_targets


def test_a_code_unit_carries_its_source_element_provenance():
    unit = _code_units("def f():\n    return 1\n")[0].unit
    assert unit["provenance"]["element_id"] == "code01"
    assert unit["provenance"]["extractor"] == "code"
    assert unit["provenance"]["source_rel_path"].endswith("mod.py")


def test_a_code_units_default_referencing_a_constant_is_carried():
    """Same def-time rule as notebooks: a default is evaluated in the enclosing scope."""
    units = _code_units("THRESH = 0.5\n\ndef f(x, limit=THRESH):\n    return x > limit\n")
    assert units[0].unit["callability"]["verdict"] == "callable"
    assert "THRESH" in units[0].unit["callability"]["requires_consts"]


def test_units_are_not_given_the_library_target_when_it_was_not_requested():
    from extractors.base import EMIT_LIBRARY, EMIT_OPENSEARCH

    units = _code_units("def f():\n    return 1\n", targets=[EMIT_OPENSEARCH])
    assert EMIT_LIBRARY not in units[0].emit_targets


def test_an_unparseable_module_yields_no_units_and_no_crash():
    assert _code_units("def broken(:\n") == []


# --------------------------------------------------------------- argparse gate

def _entry(source):
    import ast

    from extractors.code_extractor import _entry_point

    return _entry_point(ast.parse(source))


def test_importing_argparse_is_not_an_entry_point():
    """Any module that merely imported argparse was promoted, shipping the whole file as
    module_source — a library module with one optional CLI helper looked like a CLI."""
    has, _ep, _p = _entry("import argparse\n\ndef helper():\n    return 1\n")
    assert has is False


def test_a_real_cli_is_still_an_entry_point():
    has, _ep, _p = _entry("import argparse\n\ndef main():\n"
                          "    p = argparse.ArgumentParser()\n    return p.parse_args()\n")
    assert has is True


def test_a_main_block_is_still_an_entry_point():
    has, _ep, _p = _entry("def go():\n    return 1\n\n"
                          "if __name__ == '__main__':\n    go()\n")
    assert has is True


def test_a_named_entry_function_wins():
    has, ep, _p = _entry("def run_workflow(a, b):\n    return a + b\n")
    assert has is True and ep == "run_workflow"
