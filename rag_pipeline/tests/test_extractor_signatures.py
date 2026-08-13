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

from extractors.analysis.signatures import params_of, signature_of  # noqa: E402


# ------------------------------------------------------------------ class units

CLASS_SRC = '''
class DoubleConv(nn.Module):
    """Two convolutions."""
    def __init__(self, in_c, out_c):
        super().__init__()

class NoInit(SomeBase):
    def forward(self): pass

class CatchmentBuilder:
    def __init__(self, catchments: "gpd.GeoDataFrame", radius_m: float = 5000):
        self.buf = catchments.buffer(radius_m)
'''


def _cls(name):
    import ast

    tree = ast.parse(CLASS_SRC)
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def test_a_class_advertises_its_constructor_call_not_def_x():
    """``def DoubleConv()`` was doubly wrong: a class is not a ``def``, and the constructor
    arguments were all dropped. Following that contract raises TypeError immediately, because
    the shipped code is ``def __init__(self, in_c, out_c)``. Measured on the live registry:
    35 units advertised a zero-argument ``def``; after the fix, 5 do — and all 5 are genuinely
    no-argument functions."""
    sig = signature_of(_cls("DoubleConv"))
    assert sig.startswith("DoubleConv(in_c, out_c)")
    assert "class" in sig and not sig.startswith("def ")


def test_a_class_with_no_constructor_of_its_own_says_so_rather_than_implying_no_args():
    """Whether it takes arguments depends on a base class this analyser deliberately does not
    resolve — the base may live in another module entirely. ``(...)`` is honest; ``()`` is a
    claim."""
    sig = signature_of(_cls("NoInit"))
    assert "(...)" in sig and "SomeBase" in sig


def test_class_params_come_from_init_without_self():
    params = params_of(_cls("DoubleConv"))
    assert [p.name for p in params] == ["in_c", "out_c"]


def test_a_class_constructor_gets_enforceable_invariants():
    """``params_of`` returned ``[]`` for every ClassDef, and ``contract_invariants`` iterates
    parameters — so a class whose constructor buffers a GeoDataFrame carried ZERO invariants and
    a geographic frame passed to it was never checked.

    Note for the record: on the current corpus this fix adds no invariants, because all 26
    promoted classes are neural-net layers and torch Datasets and none takes a frame. The chain
    was broken; this corpus just does not exercise it."""
    from extractors.analysis.signatures import contract_invariants, contract_params

    node = _cls("CatchmentBuilder")
    params = contract_params(node, "")
    invs = contract_invariants(params, node)
    assert ("projected_crs", "catchments") in [(i.check, i.target) for i in invs]


def test_a_class_guard_does_not_destroy_the_class():
    """``functools.wraps`` on a class returns a plain FUNCTION, so replacing the module
    attribute breaks ``isinstance(x, C)`` (arg 2 must be a type) and ``class Sub(C)`` (not an
    acceptable base type). Every class in this library is a torch ``nn.Module`` or ``Dataset`` —
    precisely the things that get subclassed and isinstance-checked. The guard patches
    ``__init__`` instead, which keeps identity and still sees every constructor argument."""
    import sys
    import types

    import geopandas as gpd
    from shapely.geometry import Point

    from agent_runtime.sandbox_verify import VIOLATIONS_GLOBAL, install_contract_guards

    mod = types.ModuleType("fake_class_unit_mod")

    class Builder:
        def __init__(self, catchments, radius_m=5000):
            self.buf = catchments.buffer(radius_m)

    mod.Builder = Builder
    sys.modules["fake_class_unit_mod"] = mod
    try:
        ns: dict = {}
        assert install_contract_guards(ns, {"Builder": {
            "module": "fake_class_unit_mod", "symbol": "Builder",
            "invariants": [{"check": "projected_crs", "target": "catchments"}]}}) == 1
        assert mod.Builder is Builder and isinstance(mod.Builder, type)

        def frame(crs):
            return gpd.GeoDataFrame({"a": [1]}, geometry=[Point(0, 0)], crs=crs)

        class Sub(mod.Builder):
            pass

        assert isinstance(Sub(frame("EPSG:32616")), mod.Builder)
        assert ns.get(VIOLATIONS_GLOBAL) == [], "a projected frame must not be flagged"

        mod.Builder(frame("EPSG:4326"))
        violations = ns[VIOLATIONS_GLOBAL]
        assert len(violations) == 1 and violations[0]["status"] == "fail"
        assert "catchments" in violations[0]["target"]
    finally:
        sys.modules.pop("fake_class_unit_mod", None)


# ------------------------------------------------------------------ inference follows calls

HELPER_SRC = '''
def _ratio(catchments, pop):
    return catchments.buffer(1000).area / pop.area

def entry_point(catchments: "gpd.GeoDataFrame", pop_data: "gpd.GeoDataFrame"):
    """The unit an agent actually calls."""
    return _ratio(catchments, pop_data)

def inline(catchments: "gpd.GeoDataFrame"):
    return catchments.buffer(1000).area

def no_metric_work(gdf: "gpd.GeoDataFrame"):
    return len(gdf)

def a(gdf: "gpd.GeoDataFrame"):
    return b(gdf)

def b(gdf):
    return a(gdf)
'''


def _bodies():
    import ast

    from extractors.analysis import iter_units

    return dict(iter_units(ast.parse(HELPER_SRC)))


def _crs_invariants(name, *, follow):
    import ast

    from extractors.analysis.signatures import contract_invariants, contract_params

    bodies = _bodies()
    node = bodies[name]
    params = contract_params(node, ast.get_docstring(node) or "",
                             bodies if follow else None)
    return [i.target for i in contract_invariants(params, node) if i.check == "projected_crs"]


def test_a_public_entry_point_whose_metric_work_is_in_a_helper_gets_the_invariant():
    """CRS inference was intraprocedural, so it saw only the unit's own body. Measured on the
    corpus: ``catchment_ratios_area`` computes inline and got ``projected_crs``, while
    ``catchment_ratios_centroid`` — same public interface, same requirement — delegated to
    ``calculate_centroid`` and got nothing. The entry points are precisely the units an agent
    calls. Real gain: ``e2sfca``, ``ae2sfca``, ``catchment_ratios_centroid`` and
    ``aggregate_ratios_centroid`` went from 3/3/3/2 invariants to 6/6/6/4, and units carrying a
    CRS invariant went 5 -> 9."""
    assert _crs_invariants("entry_point", follow=False) == []
    assert _crs_invariants("entry_point", follow=True) == ["catchments", "pop_data"]


def test_inline_metric_work_is_unchanged():
    assert _crs_invariants("inline", follow=True) == ["catchments"]


def test_a_unit_that_does_no_metric_work_gains_nothing():
    """Zero units lost an invariant and zero gained a spurious one across the 229-unit corpus;
    an over-eager CRS check would fail correct runs and get the gate switched off."""
    assert _crs_invariants("no_metric_work", follow=True) == []


def test_mutual_recursion_terminates():
    """``a`` calls ``b`` calls ``a``. A visited set, not a hope."""
    assert _crs_invariants("a", follow=True) == []


def test_the_evidence_names_the_helper_the_operation_was_found_in():
    """An inference reached through a call must be auditable, or a wrong one is unexplainable.
    The real corpus records two hops: ".centroid( at line 169 via calculate_centroid() via
    catchment_ratios_centroid()"."""
    import ast

    from extractors.analysis.signatures import contract_params

    bodies = _bodies()
    params = contract_params(bodies["entry_point"], "", bodies)
    evidence = next(p.evidence for p in params if p.name == "catchments")
    assert "via _ratio()" in evidence
