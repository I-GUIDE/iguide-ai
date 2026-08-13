"""Re-ingest reconciliation: make the index match what the source produces NOW.

Before this, re-ingest meant "write the new docs" and nothing else. There is **no delete
anywhere in this repo** outside ``memory_module``, so a notebook that lost a cell kept its old
``::block::<n>`` documents forever — and kept having them retrieved as evidence for code that
no longer exists. With 3,830 blocks indexed from the corpus, that stops being theoretical.

Verified against the live cluster too (recorded in the DEVLOG): ingest 5 cells → 5 indexed;
re-ingest the same → 0 orphans; drop to 3 cells → **2 orphans found, 2 deleted, 3 remain**.
What is unit-tested here is the diff and the safety rail, with a fake client.
"""

from __future__ import annotations

import pytest

from extractors.base import (EMIT_OPENSEARCH, KIND_NOTEBOOK_BLOCK, AssetRecord,
                             ExtractionResult)
from extractors.doc_ids import resource_type_for
from extractors.emitters import opensearch_emitter as em
from extractors.indices import index_for
from extractors.manifest import UnifiedManifest

RT = resource_type_for(KIND_NOTEBOOK_BLOCK)
IDX = index_for(RT)


def manifest_with(n: int, parent: str = "nb1") -> UnifiedManifest:
    m = UnifiedManifest(repo_id="recon", source_url="t", cloned_at="now")
    assets = [AssetRecord(asset_id=f"{parent}::block::{i}", kind=KIND_NOTEBOOK_BLOCK,
                          resource_type=RT, doc_id=f"{parent}::block::{i}",
                          emit_targets=[EMIT_OPENSEARCH], title=f"cell {i}",
                          contents=f"print({i})",
                          extracted={"parent_doc_id": parent, "parent_type": "Notebook"})
              for i in range(n)]
    m.add_result(f"notebook:{parent}", ExtractionResult(assets=assets))
    return m


class FakeClient:
    """Just enough OpenSearch to exercise the diff."""

    def __init__(self, existing=()):
        # doc_id -> parent, derived from the "<parent>::block::<n>" convention. The fake MUST
        # honour the term filter: a search() that returns everything regardless of the query
        # cannot verify parent scoping at all, and would pass whether or not the code scoped.
        self.docs = {str(d) for d in existing}
        self.deleted: list = []
        self.indexed: list = []

        class _Indices:
            def __init__(self, outer):
                self._outer = outer

            def exists(self, index):
                return True

            def create(self, index, body=None):
                pass

            def refresh(self, index):
                pass

        self.indices = _Indices(self)

    def search(self, index, body):
        want = (((body or {}).get("query") or {}).get("term") or {}).get(
            "extracted.parent_doc_id")
        hits = [d for d in sorted(self.docs)
                if want is None or d.split("::")[0] == want]
        return {"hits": {"hits": [{"_source": {"doc_id": d}} for d in hits]}}

    def index(self, index, id=None, body=None):
        self.docs.add(str(id))

    def get(self, index, id):
        raise KeyError(id)


@pytest.fixture()
def bulk_spy(monkeypatch):
    """Capture bulk actions instead of talking to a cluster."""
    seen = {"index": [], "delete": [], "update": []}

    def fake_bulk(client, actions, **kw):
        actions = list(actions)
        for a in actions:
            seen[a["_op_type"]].append(a.get("_id"))
        return len(actions), []

    import opensearchpy.helpers as helpers
    monkeypatch.setattr(helpers, "bulk", fake_bulk)
    return seen


# ------------------------------------------------------------------ the diff

def test_a_lost_cell_becomes_an_orphan():
    client = FakeClient(existing=[f"nb1::block::{i}" for i in range(5)])
    plan = em.reconcile_plan(client, em.build_docs(manifest_with(3)))
    assert plan["orphan_count"] == 2
    assert plan["orphans"][IDX] == {"nb1::block::3", "nb1::block::4"}


def test_an_unchanged_element_has_no_orphans():
    client = FakeClient(existing=[f"nb1::block::{i}" for i in range(5)])
    plan = em.reconcile_plan(client, em.build_docs(manifest_with(5)))
    assert plan["orphan_count"] == 0


def test_a_growing_element_has_no_orphans():
    client = FakeClient(existing=[f"nb1::block::{i}" for i in range(2)])
    plan = em.reconcile_plan(client, em.build_docs(manifest_with(5)))
    assert plan["orphan_count"] == 0


def test_a_first_ingest_has_no_orphans():
    plan = em.reconcile_plan(FakeClient(), em.build_docs(manifest_with(3)))
    assert plan["orphan_count"] == 0


def test_the_diff_is_scoped_to_ONE_parent():
    """A shared index holds every element's blocks. Deleting by "not in this manifest" without
    scoping by parent would wipe every other notebook in the index."""
    client = FakeClient(existing=["nb1::block::0", "nb2::block::0", "nb2::block::1"])
    plan = em.reconcile_plan(client, em.build_docs(manifest_with(1, parent="nb1")))
    assert plan["orphan_count"] == 0, "another element's blocks were treated as orphans"


def test_a_search_failure_yields_no_orphans_rather_than_deleting_everything():
    """If the CURRENT state cannot be read, the safe diff is the empty one. Treating an
    unreadable index as "nothing is there" would delete the element's whole history."""
    class Broken(FakeClient):
        def search(self, index, body):
            raise RuntimeError("cluster unavailable")

    plan = em.reconcile_plan(Broken(existing=["nb1::block::9"]), em.build_docs(manifest_with(1)))
    assert plan["orphan_count"] == 0


# ------------------------------------------------------------------ the safety rail

def test_writing_outside_the_agent_prefix_is_refused():
    with pytest.raises(RuntimeError, match="non-agent index"):
        em._assert_agent_indices(["new-opensearch-index"])


def test_a_collision_with_the_platform_index_is_refused(monkeypatch):
    """This module DELETES documents. A misconfigured prefix resolving to OPENSEARCH_INDEX
    would let a re-ingest delete platform records, which is unrecoverable from here."""
    monkeypatch.setenv("OPENSEARCH_INDEX", "iguide_agent_notebook_blocks")
    with pytest.raises(RuntimeError, match="collides"):
        em._assert_agent_indices(["iguide_agent_notebook_blocks"])


def test_the_real_agent_indices_pass():
    from extractors.indices import all_agent_indices

    em._assert_agent_indices(all_agent_indices())


# ------------------------------------------------------------------ emit wiring

def test_emit_deletes_the_orphans_it_found(bulk_spy):
    client = FakeClient(existing=[f"nb1::block::{i}" for i in range(5)])
    out = em.emit(manifest_with(3), client=client, embed=False)
    assert out["orphans_found"] == 2
    assert set(bulk_spy["delete"]) == {"nb1::block::3", "nb1::block::4"}
    assert len(bulk_spy["index"]) == 3


def test_reconcile_can_be_turned_off(bulk_spy):
    client = FakeClient(existing=[f"nb1::block::{i}" for i in range(5)])
    out = em.emit(manifest_with(3), client=client, embed=False, reconcile=False)
    assert out["orphans_found"] == 0
    assert bulk_spy["delete"] == []


def test_writes_go_through_bulk_not_one_call_per_doc(bulk_spy):
    """The corpus backfill was 4,179 index calls plus 4,179 embed updates — ~8,300 round
    trips where a few bulk requests will do."""
    client = FakeClient()
    em.emit(manifest_with(50), client=client, embed=False)
    assert len(bulk_spy["index"]) == 50
    assert client.indexed == [], "the per-doc client.index path is no longer used"


# ------------------------------------------------------------------ skip-if-unchanged

def test_the_fingerprint_is_stable_for_identical_content():
    assert em.run_fingerprint(manifest_with(5)) == em.run_fingerprint(manifest_with(5))


def test_the_fingerprint_changes_when_content_changes():
    assert em.run_fingerprint(manifest_with(5)) != em.run_fingerprint(manifest_with(3))


def test_the_fingerprint_carries_the_schema_version():
    """A re-ingest of the same commit through a CHANGED extractor must not be skipped — that
    is precisely when skipping would hide a regression."""
    assert em.run_fingerprint(manifest_with(2)).startswith(f"v{em.SCHEMA_VERSION}-")


def test_no_previous_run_is_none_not_an_error():
    assert em.previous_run(FakeClient(), "nb1") is None
