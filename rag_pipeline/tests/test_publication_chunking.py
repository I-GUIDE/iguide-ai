"""Reading a whole paper instead of its first 12,000 characters.

``extract_method`` sent ``text[:12000]`` — a fixed offset, mid-word and mid-sentence. For any
paper longer than ~12 KB that discarded everything after it, and in a standard paper layout the
Methods section comes late. The model then answered honestly about the text it was shown, and the
result was recorded as *the method of the paper*. Measured on a 48,661-character document: the old
path saw **25%** of it and reported ``llm_extracted``.

Two properties are asserted here, and the second matters as much as the first:

* chunks fall on **paragraph** boundaries, so no chunk begins or ends mid-sentence;
* coverage is **recorded and surfaced**. A spec built from 2 of 5 sections and one built from the
  whole paper must not be indistinguishable — which is exactly what a silent truncation made
  them. ``llm_partial`` is a *qualified success*, deliberately not in ``DEGRADED_STATUSES``:
  real steps were extracted, so calling it degraded would throw them away.

No LLM is called. ``register_llm_callable`` injects a fake, so these are pure and cost nothing.
"""

from __future__ import annotations

import json

import pytest

from extractors.publication_extractor import (DEGRADED_STATUSES, STATUS_EXTRACTED,
                                              STATUS_PARTIAL, STATUS_UNAVAILABLE,
                                              STATUS_UNPARSEABLE, extract_method,
                                              paragraph_chunks)


def _paper(sections: int = 48, words: int = 200) -> str:
    return "\n\n".join(f"Section {i}. " + ("text " * words) for i in range(1, sections + 1))


@pytest.fixture()
def fake_llm(monkeypatch):
    """Injected via the module's own seam; returns a distinct spec per chunk."""
    from rag_pipeline import llm_utils

    calls = {"n": 0, "prompts": []}

    def fake(prompt):
        calls["n"] += 1
        calls["prompts"].append(prompt)
        n = calls["n"]
        return json.dumps({"summary": f"summary {n}", "steps": [f"step {n}", "shared step"],
                           "datasets_referenced": [f"ds{n}", "common"],
                           "tools_referenced": ["geopandas"], "params": {f"p{n}": n}})

    monkeypatch.setattr(llm_utils, "_LLM_CALLABLE", None, raising=False)
    llm_utils.register_llm_callable(fake)
    yield calls
    llm_utils.register_llm_callable(None)


# ------------------------------------------------------------------ chunk boundaries

def test_chunks_fall_on_paragraph_boundaries():
    chunks = paragraph_chunks(_paper(30, 60), max_chars=1200, max_chunks=10 ** 6)
    assert len(chunks) > 1
    assert all(c.startswith("Section ") for c in chunks), "a chunk began mid-paragraph"
    assert all(c == c.strip() for c in chunks)


def test_no_paragraph_is_lost_or_duplicated():
    """Splitting must partition the document, not sample it."""
    paper = _paper(30, 60)
    chunks = paragraph_chunks(paper, max_chars=1200, max_chunks=10 ** 6)
    assert sum(c.count("Section ") for c in chunks) == 30


def test_a_paragraph_longer_than_the_window_is_emitted_whole():
    """A hard cut inside a paragraph is the thing this exists to avoid, and the model tolerates
    an over-long chunk better than a truncated sentence."""
    giant = "Only paragraph. " + ("word " * 5000)
    chunks = paragraph_chunks(giant, max_chars=500, max_chunks=10 ** 6)
    assert len(chunks) == 1 and len(chunks[0]) > 500


def test_empty_and_whitespace_text_produce_no_chunks():
    assert paragraph_chunks("") == []
    assert paragraph_chunks("   \n\n  \t ") == []


def test_text_with_no_blank_lines_is_still_one_chunk():
    """A .txt export with single newlines has no paragraph breaks to find."""
    assert len(paragraph_chunks("line one\nline two\nline three", max_chars=12000)) == 1


# ------------------------------------------------------------------ coverage is recorded

def test_a_long_paper_is_read_in_chunks_not_truncated(fake_llm):
    paper = _paper()
    spec = extract_method(paper, max_chunks=20)
    assert fake_llm["n"] == 5, "one LLM call per chunk"
    assert spec["chunks_used"] == spec["chunks_total"] == 5
    assert spec["status"] == STATUS_EXTRACTED
    # The old path sent 12000 of 48661 characters and called that complete.
    assert spec["chars_seen"] > 40000 and spec["chars_total"] > 40000


def test_a_budgeted_read_is_partial_and_says_by_how_much(fake_llm):
    spec = extract_method(_paper(), max_chunks=2)
    assert spec["status"] == STATUS_PARTIAL
    assert spec["chunks_used"] == 2 and spec["chunks_total"] == 5
    assert spec["chars_seen"] < spec["chars_total"]
    assert fake_llm["n"] == 2, "the budget is the cost dial; it must be respected"


def test_partial_is_not_degraded():
    """It is a qualified success. Real steps were extracted, and folding it into `degraded` would
    discard them and report the paper as having no method."""
    assert STATUS_PARTIAL not in DEGRADED_STATUSES


def test_a_paper_that_fits_in_one_chunk_is_complete_not_partial(fake_llm):
    spec = extract_method("Intro.\n\nMethods: we buffered points by 25 km.")
    assert spec["status"] == STATUS_EXTRACTED
    assert spec["chunks_used"] == spec["chunks_total"] == 1


# ------------------------------------------------------------------ the reduce step

def test_steps_keep_document_order_and_are_deduplicated(fake_llm):
    """`steps` is documented as ordered — a method's third step read before its first is not a
    method. And the same step described in two sections is one step."""
    spec = extract_method(_paper(), max_chunks=3)
    assert spec["steps"] == ["step 1", "shared step", "step 2", "step 3"]


def test_datasets_and_tools_are_unioned(fake_llm):
    spec = extract_method(_paper(), max_chunks=3)
    assert spec["datasets_referenced"] == ["ds1", "common", "ds2", "ds3"]
    assert spec["tools_referenced"] == ["geopandas"], "repeated tools collapse to one"


def test_params_merge_with_the_first_mention_winning(fake_llm):
    spec = extract_method(_paper(), max_chunks=3)
    assert spec["params"] == {"p1": 1, "p2": 2, "p3": 3}


# ------------------------------------------------------------------ failure still discriminates

def test_a_raising_llm_is_unavailable_not_unparseable(monkeypatch):
    from rag_pipeline import llm_utils

    llm_utils.register_llm_callable(lambda _p: (_ for _ in ()).throw(RuntimeError("boom")))
    try:
        spec = extract_method(_paper())
    finally:
        llm_utils.register_llm_callable(None)
    assert spec["status"] == STATUS_UNAVAILABLE and spec["degraded"] is True
    assert spec["steps"] == [], "no steps may be fabricated when nothing came back"
    assert spec["chunks_total"] == 5, "coverage is recorded even for a total failure"


def test_unparseable_output_from_every_chunk_is_unparseable(monkeypatch):
    from rag_pipeline import llm_utils

    llm_utils.register_llm_callable(lambda _p: "I'm afraid I can't do that.")
    try:
        spec = extract_method(_paper())
    finally:
        llm_utils.register_llm_callable(None)
    assert spec["status"] == STATUS_UNPARSEABLE and spec["degraded"] is True


def test_one_bad_chunk_does_not_discard_the_good_ones(monkeypatch):
    """The whole point of map/reduce: a paper is not all-or-nothing. One unparseable section must
    cost that section, not the extraction."""
    from rag_pipeline import llm_utils

    calls = {"n": 0}

    def flaky(_prompt):
        calls["n"] += 1
        if calls["n"] == 2:
            return "not json at all"
        return json.dumps({"summary": "s", "steps": [f"step {calls['n']}"],
                           "datasets_referenced": [], "tools_referenced": [], "params": {}})

    llm_utils.register_llm_callable(flaky)
    try:
        spec = extract_method(_paper(), max_chunks=20)
    finally:
        llm_utils.register_llm_callable(None)

    assert spec["status"] == STATUS_PARTIAL, "not complete, and not a failure either"
    assert spec["degraded"] is False
    assert len(spec["steps"]) == 4, "4 of 5 chunks parsed"
    assert spec["chunks_parsed"] == 4 and spec["chunks_used"] == 5
    assert any("chunk 1" in f for f in spec["chunk_failures"]), (
        "which chunk failed must be recorded, not just that one did")


# ------------------------------------------------------------------ what the agent reads

def test_a_partial_spec_is_prefixed_so_the_caveat_survives_truncation(fake_llm, tmp_path,
                                                                     monkeypatch):
    """Prefixed for the same reason the UNAVAILABLE caveat is: the evidence view truncates, and a
    qualification that appears after 4,000 characters is one nobody reads."""
    from extractors.base import ExtractContext
    from extractors.publication_extractor import PublicationExtractor

    monkeypatch.setenv("PUB_MAX_CHUNKS", "2")
    (tmp_path / "paper.txt").write_text(_paper(), encoding="utf-8")
    ctx = ExtractContext(repo_id="r", element_id="e1", source_url="u", commit_sha="c")
    result = PublicationExtractor().extract(str(tmp_path / "paper.txt"), ctx=ctx)

    asset = result.assets[0]
    assert asset.contents.startswith("[PARTIAL METHOD SPEC]")
    assert "2 of 5 sections" in asset.contents
    assert asset.extracted["is_method_spec"] is True, "the steps are real"
    assert asset.extracted["chunks_used"] == 2 and asset.extracted["chunks_total"] == 5
    assert any("llm_partial" in w for w in result.warnings), (
        "an operator running a batch needs to know coverage was capped")


def test_a_complete_spec_carries_no_caveat(fake_llm, tmp_path, monkeypatch):
    """A caveat on every publication is a caveat nobody reads."""
    from extractors.base import ExtractContext
    from extractors.publication_extractor import PublicationExtractor

    monkeypatch.setenv("PUB_MAX_CHUNKS", "20")
    (tmp_path / "paper.txt").write_text(_paper(), encoding="utf-8")
    ctx = ExtractContext(repo_id="r", element_id="e1", source_url="u", commit_sha="c")
    result = PublicationExtractor().extract(str(tmp_path / "paper.txt"), ctx=ctx)

    assert "PARTIAL METHOD SPEC" not in result.assets[0].contents
    assert result.warnings == []


# ------------------------------------------------------------------ the budget dial

@pytest.mark.parametrize("raw,expected", [("1", 1), ("4", 4), ("20", 20),
                                          ("999", 20), ("0", 1), ("nope", 4), ("", 4)])
def test_the_chunk_budget_is_clamped_and_never_raises(raw, expected, monkeypatch):
    """Each chunk is one LLM call, so an unbounded value is an unbounded bill on a 180-paper
    batch — and a typo must not crash the extraction."""
    from extractors import publication_extractor

    monkeypatch.setenv("PUB_MAX_CHUNKS", raw)
    assert publication_extractor._max_chunks() == expected
