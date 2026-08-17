"""The emitters are not independent, and the order used to be wrong.

``library_emitter`` is the emitter that decides where a unit's code lives: it writes
``v_<slice_sha>.py`` and stamps the dotted path back onto the unit as ``library_module``
(``library_emitter.py:304``, mutating the live unit dict). The OpenSearch document then advertises
that path as the unit's import line — the one fact that makes a retrieved unit usable instead of
merely known.

``_fan_out`` ran OpenSearch first. So every indexed unit document was built before its module path
existed, and **0 of 130** corpus unit documents could carry an importable line. The defect was
invisible for as long as the document carried no contract at all: with no field for the module,
there was nothing to notice was empty. It became visible and fixable in the same change.

These tests pin the ordering by its observable consequence rather than by asserting call order, so
they keep working if the emitters are reorganised and stop working if the dependency is broken.
"""

from __future__ import annotations

import json

import pytest

from extractors.base import (AssetRecord, EMIT_LIBRARY, EMIT_OPENSEARCH, KIND_METHOD_UNIT,
                             ExtractContext)
from extractors.emitters.opensearch_emitter import build_docs
from extractors.ingest import _fan_out
from extractors.manifest import UnifiedManifest
from extractors.notebook_extractor import NotebookExtractor


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """A throwaway storage root and the file-backed KB, so nothing touches the real library.

    The variable is ``AGENT_FILE_STORAGE_ROOT`` (``file_store.storage_root``). Setting the wrong
    name silently writes into the developer's real method library — which is how I learned it.
    """
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENT_KB_BACKEND", "local")
    return tmp_path


def _notebook(tmp_path):
    cells = [
        {"cell_type": "code", "source": "import geopandas as gpd", "metadata": {},
         "outputs": [], "execution_count": None},
        {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
         "source": "def area_km2(gdf: gpd.GeoDataFrame) -> float:\n"
                   '    """Total projected area in square kilometres."""\n'
                   "    return gdf.to_crs(3435).area.sum() / 1e6\n"},
    ]
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps({
        "cells": cells, "metadata": {"kernelspec": {"language": "python", "name": "python"}},
        "nbformat": 4, "nbformat_minor": 5}))
    return path


def _manifest(tmp_path):
    ctx = ExtractContext(element_id="e1f2a3b4", element_type="notebook",
                         fields={"title": "Illinois areas", "tags": ["Illinois"]},
                         targets=[EMIT_LIBRARY, EMIT_OPENSEARCH])
    result = NotebookExtractor().extract(str(_notebook(tmp_path)), ctx=ctx)
    man = UnifiedManifest(element_id="e1f2a3b4", element_type="notebook")
    man.assets.extend(result.assets)
    man.provenance_edges.extend(result.edges)
    return man


def _unit_docs(manifest):
    return [src for _i, _d, src in build_docs(manifest)
            if (src.get("extracted") or {}).get("kind") == KIND_METHOD_UNIT]


def test_extraction_alone_cannot_know_the_import_line(isolated, tmp_path):
    """Not a defect — a statement of the dependency. The module path is assigned by whoever
    writes the module, and at extraction time nobody has."""
    docs = _unit_docs(_manifest(tmp_path))
    assert docs, "no unit documents; the assertions below would be vacuous"
    assert all(not (d["extracted"]["unit"].get("import_line")) for d in docs)


def test_after_the_fan_out_every_unit_document_carries_an_import_line(isolated, tmp_path):
    man = _manifest(tmp_path)
    _fan_out(man, ["library", "opensearch"])
    docs = _unit_docs(man)
    assert docs
    for doc in docs:
        unit = doc["extracted"]["unit"]
        assert unit["import_line"].startswith("from iguide_methods."), unit.get("import_line")
        assert unit["import_line"].endswith(f" import {unit['library_symbol']}")


def test_the_import_line_names_the_version_that_was_indexed(isolated, tmp_path):
    """Pinned to ``v_<slice_sha>``, not the element package's re-export: a re-ingest mints a new
    module and leaves the old importable, so an artifact that recorded the package alias would
    silently re-run against different code."""
    man = _manifest(tmp_path)
    _fan_out(man, ["library", "opensearch"])
    for doc in _unit_docs(man):
        unit = doc["extracted"]["unit"]
        assert f"v_{unit['slice_sha']}" in unit["import_line"]


def test_the_advertised_module_is_a_file_that_exists(isolated, tmp_path):
    """The strongest form of the check: resolve the advertised dotted path to disk. An import line
    that points at nothing fails inside the sandbox, where the agent cannot recover from it."""
    man = _manifest(tmp_path)
    _fan_out(man, ["library", "opensearch"])
    for doc in _unit_docs(man):
        module = doc["extracted"]["unit"]["library_module"]
        relative = module.split(".", 1)[1].replace(".", "/") + ".py"
        assert (tmp_path / "method_library" / "iguide_methods" / relative).is_file(), module


def test_indexing_without_the_library_target_advertises_nothing(isolated, tmp_path):
    """A unit can legitimately be indexed for discovery without being shipped. It must then say
    nothing about importing, rather than offer a line that cannot work."""
    man = _manifest(tmp_path)
    _fan_out(man, ["opensearch"])
    for doc in _unit_docs(man):
        assert not doc["extracted"]["unit"].get("import_line")


def test_a_library_failure_does_not_stop_the_indexing(isolated, tmp_path, monkeypatch):
    """Extraction already succeeded; one emitter's failure must not cost the others. Reordering
    put the library FIRST, so this is the case that reordering could have broken."""
    from extractors.emitters import library_emitter

    monkeypatch.setattr(library_emitter, "emit",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
    man = _manifest(tmp_path)
    _fan_out(man, ["library", "opensearch"])
    assert any("[library] emit failed" in w for w in man.warnings)
    assert any("[kb:" in w for w in man.warnings), "the index write was skipped too"


def test_the_manifest_records_which_emitters_ran(isolated, tmp_path):
    man = _manifest(tmp_path)
    _fan_out(man, ["library", "opensearch"])
    joined = " ".join(man.warnings)
    assert "[library] wrote" in joined and "[kb:local]" in joined
