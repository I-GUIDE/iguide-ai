"""Per-function promotion in the notebook extractor.

Before this, one unparseable cell set ``all_parse_ok = False`` and the notebook produced
nothing reusable at all — the entire promotion was gated on every cell parsing. Real
notebooks routinely contain a cell that does not parse (shell escapes, partial edits,
notebook-only syntax), so the gate was costing whole notebooks for one bad cell.

Promotion is now per function, assembled from the cells that DID parse. Measured over the 14
real cached notebooks: 290 blocks and 41 units, 40 of them independently callable — including
units from three notebooks that have unparseable cells and would previously have yielded none.

A ``needs_globals`` unit is still indexed (discoverable, with its blocker visible) but never
given ``EMIT_LIBRARY``: shipping it would require inlining the module-level statements the
slice builder exists to exclude.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from extractors.base import (EMIT_LIBRARY, EMIT_OPENSEARCH, KIND_METHOD_UNIT,
                             VALID_TARGETS, ExtractContext)
from extractors.notebook_extractor import NotebookExtractor


def _nb(cells, *, language="python"):
    """Minimal nbformat-4 notebook."""
    return {
        "cells": [{"cell_type": t, "source": s, "metadata": {}, "outputs": [],
                   "execution_count": None} if t == "code" else
                  {"cell_type": t, "source": s, "metadata": {}}
                  for t, s in cells],
        "metadata": {"kernelspec": {"language": language, "name": language}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def _write(tmp_path, cells, *, language="python", name="nb.ipynb"):
    p = tmp_path / name
    p.write_text(json.dumps(_nb(cells, language=language)))
    return p


def _extract(path, *, targets=VALID_TARGETS, element_id="elem123"):
    ctx = ExtractContext(element_id=element_id, element_type="notebook",
                         targets=targets, commit_sha="deadbeef")
    return NotebookExtractor().extract(str(path), ctx=ctx)


def _units(result):
    return [a for a in result.assets if a.kind == KIND_METHOD_UNIT]


# --------------------------------------------------------------- the regression

def test_one_unparseable_cell_no_longer_costs_the_whole_notebook(tmp_path):
    """THE reason this exists. Previously: all_parse_ok=False -> nothing promoted."""
    nb = _write(tmp_path, [
        ("code", "import os\n"),
        ("code", "def clean(x):\n    return x.strip()\n"),
        ("code", "this is not (valid python\n"),          # unparseable
        ("code", "def total(xs):\n    return sum(xs)\n"),
    ])
    r = _extract(nb)
    names = {u.unit["qualified_name"] for u in _units(r)}
    assert names == {"clean", "total"}, "units from good cells were lost to one bad cell"
    assert any("not all code cells parsed" in w for w in r.warnings)


def test_blocks_are_still_emitted_alongside_units(tmp_path):
    nb = _write(tmp_path, [("code", "def f(x):\n    return x\n")])
    r = _extract(nb)
    assert _units(r), "no unit emitted"
    assert [a for a in r.assets if a.kind != KIND_METHOD_UNIT], "block assets disappeared"


# --------------------------------------------------------------- contract content

def test_the_unit_carries_a_usable_contract(tmp_path):
    nb = _write(tmp_path, [
        ("code", "import pandas as pd\n"),
        ("code", 'def head(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:\n'
                 '    """Return the first k rows."""\n    return df.head(k)\n'),
    ])
    u = _units(_extract(nb))[0].unit
    assert u["qualified_name"] == "head"
    assert "k: int=5" in u["signature"] and "pd.DataFrame" in u["signature"]
    assert u["doc_summary"] == "Return the first k rows."
    assert {p["name"] for p in u["params"]} == {"df", "k"}
    assert u["slice_sha"], "a callable unit must have a content-addressed version"
    assert u["provenance"]["element_id"] == "elem123"
    assert u["provenance"]["commit_sha"] == "deadbeef"


def test_a_callable_unit_targets_the_library(tmp_path):
    nb = _write(tmp_path, [("code", "def f(x):\n    return x * 2\n")])
    a = _units(_extract(nb))[0]
    assert a.unit["callability"]["verdict"] == "callable"
    assert EMIT_LIBRARY in a.emit_targets
    assert EMIT_OPENSEARCH in a.emit_targets


def test_a_needs_globals_unit_is_indexed_but_never_shipped(tmp_path):
    """Discoverable, with its blocker visible — but not importable code."""
    nb = _write(tmp_path, [
        ("code", "import geopandas as gpd\ngdf = gpd.read_file('x.shp')\n"),
        ("code", "def rows(k):\n    return gdf.head(k)\n"),
    ])
    a = next(u for u in _units(_extract(nb)) if u.unit["qualified_name"] == "rows")
    assert a.unit["callability"]["verdict"] == "needs_globals"
    assert "gdf" in a.unit["callability"]["global_reads"]
    assert EMIT_LIBRARY not in a.emit_targets, "an uncallable unit must not ship as code"
    assert EMIT_OPENSEARCH in a.emit_targets, "but it must stay discoverable"
    assert "not independently callable" in a.contents


def test_library_target_is_withheld_when_not_requested(tmp_path):
    nb = _write(tmp_path, [("code", "def f(x):\n    return x\n")])
    a = _units(_extract(nb, targets=(EMIT_OPENSEARCH,)))[0]
    assert EMIT_LIBRARY not in a.emit_targets


def test_contents_is_retrieval_text_not_the_raw_body(tmp_path):
    """Raw code retrieves poorly against natural-language questions."""
    nb = _write(tmp_path, [
        ("code", 'def compute_index(a, b):\n    """Compute the exposure index."""\n'
                 '    return a * 1.2345 + b\n'),
    ])
    a = _units(_extract(nb))[0]
    assert "Compute the exposure index." in a.contents
    assert "def compute_index" in a.contents          # the signature
    assert "1.2345" not in a.contents, "the body leaked into the retrieval text"


# --------------------------------------------------------------- guards

def test_an_r_notebook_promotes_nothing(tmp_path):
    """`nc <- st_read(...)` parses as valid Python (`<` then unary `-`)."""
    nb = _write(tmp_path, [("code", "nc <- st_read(system.file('shape.shp'))\n")],
                language="r")
    r = _extract(nb)
    assert _units(r) == []
    assert any("not python" in w for w in r.warnings)


def test_a_notebook_with_no_functions_yields_no_units(tmp_path):
    """Script-style notebooks are 4 of the 14 in the real corpus — the supply limit."""
    nb = _write(tmp_path, [("code", "x = 1\nprint(x)\n")])
    assert _units(_extract(nb)) == []


def test_doc_ids_are_name_keyed_so_reordering_is_safe(tmp_path):
    """`::block::{order}` renames on insertion; `::unit::{name}` does not."""
    first = _write(tmp_path, [("code", "def a(x):\n    return x\n"),
                              ("code", "def b(x):\n    return x\n")], name="one.ipynb")
    second = _write(tmp_path, [("code", "# a new leading cell\n"),
                               ("code", "def a(x):\n    return x\n"),
                               ("code", "def b(x):\n    return x\n")], name="two.ipynb")
    ids1 = {u.doc_id for u in _units(_extract(first))}
    ids2 = {u.doc_id for u in _units(_extract(second))}
    assert ids1 == ids2, "inserting a cell changed unit doc_ids"


def test_defines_edges_are_written(tmp_path):
    nb = _write(tmp_path, [("code", "def f(x):\n    return x\n")])
    r = _extract(nb)
    defines = [e for e in r.edges if e.rel == "DEFINES"]
    assert len(defines) == 1
    assert defines[0].detail["verdict"] == "callable"


def test_summary_warning_reports_the_headline_metric(tmp_path):
    nb = _write(tmp_path, [
        ("code", "import geopandas as gpd\ngdf = gpd.read_file('x')\n"),
        ("code", "def ok(x):\n    return x\n"),
        ("code", "def blocked(k):\n    return gdf.head(k)\n"),
    ])
    w = next(x for x in _extract(nb).warnings if "callable units" in x)
    assert "1 of 2" in w
    assert "gdf" in w, "the blocker should be named so it is actionable"


# --------------------------------------------------------------- real corpus

CORPUS = Path("/Users/yfkang/i-guide-platform-flask-servers/agent_chat_files/eval_notebooks")


@pytest.mark.skipif(not CORPUS.exists(), reason="cached notebook corpus not present")
def test_real_corpus_promotes_the_expected_number_of_units():
    """Pins the measured baseline so a regression in promotion is visible.

    The callable baseline was 39/41 and that number was **wrong**, not good: 24 of those
    "callable" units were bound methods, which read no globals (``self`` is a parameter) and
    so passed every blocker check. Their slices define the enclosing CLASS, so the advertised
    ``from <module> import <method>`` raised ImportError — and because that line lives in the
    element's ``__init__``, one bad unit took down every sibling. Only 14 of 40 advertised
    import lines actually worked.

    With methods verdicted ``needs_instance`` the honest count is 16, and 16/16 import. A
    smaller library that imports beats a larger one that lies; raising this number again
    should mean promoting real module-level functions, not re-admitting methods.
    """
    total = callable_ = methods = 0
    for f in sorted(CORPUS.glob("*.ipynb")):
        r = _extract(f, element_id=f.stem)
        units = _units(r)
        total += len(units)
        for u in units:
            verdict = u.unit["callability"]["verdict"]
            callable_ += verdict == "callable"
            if "." in (u.unit.get("qualified_name") or ""):
                methods += 1
                assert verdict != "callable", (
                    f"{u.unit['qualified_name']} is a bound method verdicted callable — "
                    f"its slice defines the class, so the import line will fail")
    assert total >= 40, f"unit count regressed to {total}"
    assert callable_ >= 16, f"callable count regressed to {callable_}"
    assert methods >= 20, "expected the corpus to still contain class-based notebooks"


# --------------------------------------------------------------- tags are the submitter's

def _tags_of(asset):
    return (asset.source_fields or {}).get("tags")


def test_import_names_never_become_platform_tags(tmp_path):
    """`tags` is the submitter's answer to "what is this element about". Import names used to be
    merged into it, so `os`, `time`, `copy`, `html` and `__future__` arrived as topics.

    Measured on the live index before the fix: 338 of 349 method-unit documents carried at least
    one stdlib module name as a tag, and 43 of 432 distinct tags were stdlib names — while the
    platform's own tags contain ZERO across all 180 notebook elements, so extraction invented
    every one of them. The import lists were already in `block.imports` and the unit's
    `requirements`, so the merge added no reachable information.
    """
    path = _write(tmp_path, [
        ("code", "import os\nimport time\nimport copy\nimport geopandas as gpd"),
        ("code", "def area_km2(gdf):\n    return gdf.to_crs(3435).area.sum() / 1e6\n")])
    ctx = ExtractContext(element_id="e1", element_type="notebook", targets=VALID_TARGETS,
                         fields={"title": "Illinois areas", "tags": ["Illinois", "hydrology"]})
    result = NotebookExtractor().extract(str(path), ctx=ctx)

    tagged = [a for a in result.assets if _tags_of(a) is not None]
    assert tagged, "no asset carried tags; the assertion below would be vacuous"
    for asset in tagged:
        extra = sorted(set(_tags_of(asset)) - {"Illinois", "hydrology"})
        assert not extra, f"{asset.doc_id} tags polluted with {extra}"


def test_the_imports_are_still_recorded_where_imports_belong(tmp_path):
    """Removing them from `tags` must not lose them. A block already records what its cell
    imports; that is the field a reader would look in."""
    path = _write(tmp_path, [("code", "import geopandas as gpd\nimport os\n")])
    blocks = [a for a in _extract(path).assets if a.block]
    assert blocks
    imports = set()
    for b in blocks:
        imports.update(b.block.get("imports") or [])
    assert {"geopandas", "os"} <= imports


def test_a_units_dependencies_are_its_own_not_the_notebooks(tmp_path):
    """The notebook imports matplotlib; this function does not. A unit's requirements come from
    its slice's transitive closure, which is what keeps the pip install inside the sandbox
    minimal rather than the notebook's whole environment."""
    path = _write(tmp_path, [
        ("code", "import geopandas as gpd\nimport matplotlib.pyplot as plt\nimport requests"),
        ("code", "def area_km2(gdf):\n    return gdf.to_crs(3435).area.sum() / 1e6\n")])
    units = _units(_extract(path))
    assert units
    pip = set((units[0].unit.get("requirements") or {}).get("pip") or [])
    assert "matplotlib" not in pip and "requests" not in pip


def test_the_skill_front_matter_carries_topics_not_module_names(tmp_path):
    """SKILL.md front matter is read by a human deciding whether the skill applies."""
    path = _write(tmp_path, [
        ("code", "import os\nimport geopandas as gpd"),
        ("code", "def run_workflow(gdf):\n    return gdf.to_crs(3435)\n")])
    ctx = ExtractContext(element_id="e2", element_type="notebook", targets=VALID_TARGETS,
                         fields={"title": "T", "tags": ["flooding"]})
    skill = NotebookExtractor().extract(str(path), ctx=ctx).skill
    assert skill is not None and skill.tags == ["flooding"]
