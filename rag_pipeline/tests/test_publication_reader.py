"""Why a publication yielded no text — three causes that used to be one status.

``_read_text`` swallowed every exception and returned ``""``, so the extractor emitted
``status: no_text`` for a file that is not a PDF, for a missing ``pypdf``, and for a scanned PDF
with no text layer. Those need three different fixes, and ``no_text`` points at the third.

What made it concrete: ``03bc2865__oa.pdf`` in the corpus cache begins with ``<head``. It is IOP
Publishing's bot-check page — "please can you confirm you are a human by ticking the box below" —
356 characters that the DOI fetcher counted as a downloaded open-access PDF. Indexing it would
file a CAPTCHA notice as a paper's methods section, and it would inflate the count of reachable
publications. A refusal gets recorded, never satisfied.
"""

from __future__ import annotations

import pytest

from extractors.publication_extractor import read_document, sniff_kind

BOT_CHECK = """<head><title>Just a moment</title></head><body>
<p>We apologize for the inconvenience...</p>
<p>To ensure we keep this website safe, please can you confirm you are a human by
ticking the box below.</p><p>Incident ID: 7cfb664c</p></body>"""

ARTICLE = """<!DOCTYPE html><html><head><title>A paper</title>
<style>body{color:red}</style><script>var x=1;</script></head><body>
<nav>Skip to content</nav>
<h2>Methods</h2>
<p>We reprojected the county boundaries to EPSG:32616 and buffered each gauge by 25 km.</p>
<p>Zonal statistics were computed with rasterstats over the resulting polygons.</p>
</body></html>"""


def _write(tmp_path, name, content, *, binary=False):
    path = tmp_path / name
    path.write_bytes(content) if binary else path.write_text(content)
    return str(path)


# ------------------------------------------------------------------ sniffing

@pytest.mark.parametrize("name,head,expected", [
    ("a.pdf", b"%PDF-1.7\n...", "pdf"),
    ("b.pdf", b"<head><title>x", "html"),               # the real corpus case
    ("c.pdf", b"<!DOCTYPE html><html>", "html"),
    ("d.docx", b"PK\x03\x04rest", "zip"),
    ("e.txt", b"plain words here", "text"),
    ("f.xhtml", b"<?xml version='1.0'?><html>", "html"),
])
def test_the_first_bytes_decide_not_the_extension(tmp_path, name, head, expected):
    assert sniff_kind(_write(tmp_path, name, head, binary=True)) == expected


def test_a_missing_file_is_named_as_unreadable(tmp_path):
    assert sniff_kind(str(tmp_path / "absent.pdf")) == "unreadable"


# ------------------------------------------------------------------ the walls

def test_a_bot_check_page_is_reported_and_never_treated_as_the_paper(tmp_path):
    text, reason = read_document(_write(tmp_path, "oa.pdf", BOT_CHECK))
    assert text == ""
    assert "bot-check page" in reason
    assert "not the document" in reason


@pytest.mark.parametrize("marker,label", [
    ("Please purchase access to continue reading", "paywall"),
    ("404 Not Found", "error page"),
    ("We use cookies to improve your experience", "cookie consent"),
])
def test_other_publisher_walls_are_named_too(tmp_path, marker, label):
    page = f"<html><body><p>{marker}</p></body></html>"
    _text, reason = read_document(_write(tmp_path, "oa.pdf", page))
    assert label in reason


def test_a_long_paper_mentioning_bots_is_still_read(tmp_path):
    """The interstitial check only applies to short documents. A real paper about web security can
    legitimately contain "are you a robot", and refusing it would be worse than the bug."""
    body = ("<p>We study how sites verify you are human by ticking the box below.</p>"
            + "<p>Methods paragraph with substantial content about CRS reprojection.</p>" * 90)
    text, reason = read_document(_write(tmp_path, "paper.html", f"<html><body>{body}</body></html>"))
    assert len(text) > 4000
    assert reason == ""


# ------------------------------------------------------------------ HTML is content

def test_a_full_text_html_article_is_read_rather_than_discarded(tmp_path):
    """Many open-access links resolve to a full-text HTML article carrying the same methods
    section the PDF would. Treating those as unreadable discards content over an extension."""
    text, reason = read_document(_write(tmp_path, "article.html", ARTICLE))
    assert "EPSG:32616" in text and "rasterstats" in text
    assert reason == ""


def test_script_style_and_navigation_are_not_part_of_the_text(tmp_path):
    text, _reason = read_document(_write(tmp_path, "article.html", ARTICLE))
    for noise in ("var x=1", "color:red", "Skip to content"):
        assert noise not in text


def test_html_served_under_a_pdf_name_is_read_and_the_mismatch_recorded(tmp_path):
    text, reason = read_document(_write(tmp_path, "oa.pdf", ARTICLE))
    assert "EPSG:32616" in text
    assert "content is HTML" in reason and ".pdf" in reason


# ------------------------------------------------------------------ the three PDF causes

def test_an_empty_file_says_the_fetch_produced_nothing(tmp_path):
    """Two documents in the corpus cache are 0 bytes — a failed download saved anyway. Without a
    reason this reads as "the document contains nothing"."""
    _text, reason = read_document(_write(tmp_path, "oa.pdf", b"", binary=True))
    assert "empty (0 bytes)" in reason and "fetch produced no content" in reason


def test_an_unparseable_pdf_names_the_parser_error(tmp_path):
    _text, reason = read_document(
        _write(tmp_path, "broken.pdf", b"%PDF-1.4\nnot really a pdf body", binary=True))
    assert "pypdf could not parse" in reason


def test_a_missing_pypdf_is_not_confused_with_an_empty_pdf(tmp_path, monkeypatch):
    """A dependency problem and a scanned page need opposite responses: install something, or run
    OCR. They produced the same empty string."""
    import builtins

    real_import = builtins.__import__

    def no_pypdf(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("no module named pypdf")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pypdf)
    _text, reason = read_document(_write(tmp_path, "x.pdf", b"%PDF-1.7\nbody", binary=True))
    assert "pypdf is not installed" in reason


def test_a_zip_under_a_pdf_name_is_not_fed_to_the_pdf_reader(tmp_path):
    _text, reason = read_document(_write(tmp_path, "x.pdf", b"PK\x03\x04zipbytes", binary=True))
    assert "zip archive" in reason


def test_a_plain_text_document_still_reads(tmp_path):
    text, reason = read_document(_write(tmp_path, "paper.md", "# Methods\nWe buffered by 25 km."))
    assert "25 km" in text and reason == ""


# ------------------------------------------------------------------ it reaches the record

def test_the_reason_reaches_the_asset_and_the_searchable_text(tmp_path):
    """`no_text` in a field nobody reads is the same as no explanation at all. The reason has to be
    in `extracted` for querying AND in `contents`, because the evidence view is what the agent
    sees."""
    from extractors.base import ExtractContext
    from extractors.publication_extractor import PublicationExtractor

    path = _write(tmp_path, "oa.pdf", BOT_CHECK)
    ctx = ExtractContext(element_id="p1", element_type="publication", fields={"title": "A paper"})
    asset = PublicationExtractor().extract(path, ctx=ctx).assets[0]
    assert asset.extracted["source_kind"] == "html"
    assert "bot-check page" in asset.extracted["read_note"]
    assert "bot-check page" in asset.contents
    assert asset.extracted["degraded"] is True
    assert asset.extracted["is_method_spec"] is False


def test_a_readable_document_records_no_reason(tmp_path):
    from extractors.base import ExtractContext
    from extractors.publication_extractor import PublicationExtractor

    path = _write(tmp_path, "paper.md", "# Methods\n" + "We reprojected to EPSG:32616. " * 40)
    ctx = ExtractContext(element_id="p2", element_type="publication", fields={"title": "T"})
    asset = PublicationExtractor().extract(path, ctx=ctx).assets[0]
    assert asset.extracted["read_note"] is None
    assert asset.extracted["source_kind"] == "text"


# ------------------------------------------------------------------ read fine, no method in it

def test_a_paper_with_no_method_is_not_a_method_spec(tmp_path, monkeypatch):
    """`is_method_spec` was defined as `not degraded`, conflating "the extractor succeeded" with
    "there is a method here".

    Measured on the 7 readable open-access documents in the corpus cache: THREE describe no
    computational method — a PDXScholar citation cover page, an editorial on pharmaceutical waste,
    and an argumentative paper on AI ethics. All three were emitted as `llm_extracted`,
    `degraded: False`, `is_method_spec: True` with an EMPTY steps list. Nearly half the type was
    indexed as a method spec describing no method.
    """
    import json as _json

    from extractors.base import ExtractContext
    from extractors.publication_extractor import STATUS_NO_METHOD, PublicationExtractor
    import rag_pipeline.llm_utils as llm

    monkeypatch.setattr(llm, "call_llm", lambda *a, **k: _json.dumps({
        "summary": "An editorial arguing that AI ethics cannot be isolated from governance.",
        "steps": [], "datasets_referenced": [], "tools_referenced": [], "params": {}}))

    path = _write(tmp_path, "oa.md", "# Opinion\n" + ("This is an argument, not a method. " * 60))
    ctx = ExtractContext(element_id="p9", element_type="publication", fields={"title": "T"})
    asset = PublicationExtractor().extract(path, ctx=ctx).assets[0]

    assert asset.extracted["status"] == STATUS_NO_METHOD
    assert asset.extracted["is_method_spec"] is False
    assert asset.extracted["degraded"] is False, "nothing failed — a fact about the paper"
    assert "NO COMPUTATIONAL METHOD" in asset.contents
    assert "not an extraction failure" in asset.contents
    assert "editorial" in asset.contents


def test_no_method_is_distinct_from_a_failure_to_read(tmp_path):
    """`no_text` means refetch or OCR; `no_method_described` means the paper is an essay. Merging
    them makes "publications with no extractable method" uncountable."""
    from extractors.base import ExtractContext
    from extractors.publication_extractor import (STATUS_NO_METHOD, STATUS_NO_TEXT,
                                                  PublicationExtractor)

    path = _write(tmp_path, "oa.pdf", b"", binary=True)
    ctx = ExtractContext(element_id="p10", element_type="publication", fields={"title": "T"})
    asset = PublicationExtractor().extract(path, ctx=ctx).assets[0]
    assert asset.extracted["status"] == STATUS_NO_TEXT != STATUS_NO_METHOD
    assert asset.extracted["degraded"] is True, "a failed read IS degraded"


def test_a_paper_with_steps_is_still_a_method_spec(tmp_path, monkeypatch):
    import json as _json

    from extractors.base import ExtractContext
    from extractors.publication_extractor import STATUS_EXTRACTED, PublicationExtractor
    import rag_pipeline.llm_utils as llm

    monkeypatch.setattr(llm, "call_llm", lambda *a, **k: _json.dumps({
        "summary": "Buffers gauges and computes zonal statistics.",
        "steps": ["Reproject to EPSG:32616", "Buffer each gauge by 25 km"],
        "datasets_referenced": ["NHDPlus"], "tools_referenced": ["geopandas"],
        "params": {"buffer_m": 25000}}))

    path = _write(tmp_path, "oa.md", "# Methods\n" + ("We buffered the gauges. " * 60))
    ctx = ExtractContext(element_id="p11", element_type="publication", fields={"title": "T"})
    asset = PublicationExtractor().extract(path, ctx=ctx).assets[0]
    assert asset.extracted["status"] == STATUS_EXTRACTED
    assert asset.extracted["is_method_spec"] is True
    assert len(asset.extracted["steps"]) == 2
    assert "NO COMPUTATIONAL METHOD" not in asset.contents
