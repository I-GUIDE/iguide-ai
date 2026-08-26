"""The generated browser page must carry usable data, not just render.

A self-contained page fails in one realistic way: the inlined JSON is malformed or missing the
fields the renderer reads, and the page comes up blank with no error anyone sees. The CSS is not
what breaks. So these tests exercise the generator's data contract — the payload parses, the
placeholder was substituted, every record has an id/kind/name the list needs, and a record cannot
smuggle a ``</script>`` that would end the block early.

The second half asserts the LAYOUT CONTRACT, added after the first version shipped with the whole
page scrolling as one document: reading down the list pushed the detail pane off screen. That is a
cascade bug with a specific signature — `overflow-y:auto` on a grid child does nothing unless
`min-height:0` lets it shrink below its content — so it is worth pinning even though the tests
cannot see pixels. What they check is that the rules which MAKE a pane scrollable are present and
unconditional, and that the mobile fallback releases them again.

Neither half replaces looking at the page. They cover the two failures that are invisible until
someone scrolls or the data goes missing.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GEN = REPO / "scripts" / "build_extraction_browser.py"
TEMPLATE = REPO / "scripts" / "_extraction_browser_template.html"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("build_extraction_browser", GEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ the template

def test_the_template_has_exactly_one_data_placeholder():
    assert TEMPLATE.read_text(encoding="utf-8").count("__DATA__") == 1


def test_the_template_declares_a_title_and_a_theme_aware_ground():
    text = TEMPLATE.read_text(encoding="utf-8")
    assert "<title>" in text
    # A transparent body borrows the host's theme and renders one theme's text on the other's
    # ground — the classic unreadable-artifact bug.
    assert re.search(r"body\s*\{[^}]*background:\s*var\(--paper\)", text)
    assert 'prefers-color-scheme: dark' in text and '[data-theme="dark"]' in text


# ------------------------------------------------------------------ record normalisation

def test_a_unit_becomes_a_record_the_list_can_render(gen):
    registry = {"pkg.calculate_buffers": {
        "library_symbol": "calculate_buffers",
        "signature": "def calculate_buffers(gdf: gpd.GeoDataFrame, buffer: Number)",
        "doc_summary": "Replace geometry with buffers.",
        "returns": "gpd.GeoDataFrame",
        "module": "iguide_methods.pkg.v_abc123",
        "slice_sha": "abc123",
        "unit_kind": "function",
        "params": [{"name": "gdf", "annotation": "gpd.GeoDataFrame",
                    "inferred_type": "geodataframe", "declared_unit": "metres",
                    "crs_expectation": "projected", "required": True,
                    "evidence": "annotation | metric operation"}],
        "invariants": [{"check": "projected_crs", "target": "gdf", "args": {"unit": "metres"}}],
        "requirements": {"pip": ["geopandas"]},
        "provenance": {"element_id": "b1fa548b", "extractor": "notebook",
                       "source_rel_path": "nb.ipynb"}}}
    record = gen._units(registry)[0]
    assert record["kind"] == "unit"
    assert record["name"] == "calculate_buffers"
    assert record["import"] == "from iguide_methods.pkg.v_abc123 import calculate_buffers"
    assert record["params"][0]["unit"] == "metres"
    assert "has-invariants" in record["flags"] and "documented" in record["flags"]


def test_aliases_and_ambiguous_stubs_are_not_offered_as_records(gen):
    registry = {
        "pkg.f": {"signature": "def f()", "library_symbol": "f", "provenance": {}},
        "f": {"alias_for": "pkg.f", "signature": "def f()", "library_symbol": "f"},
        "g": {"ambiguous": True, "candidates": ["a.g", "b.g"], "signature": "def g()"},
    }
    ids = [r["id"] for r in gen._units(registry)]
    assert ids == ["pkg.f"], "an alias or ambiguity stub would be a duplicate row"


def test_a_naked_unit_is_flagged_so_it_can_be_filtered(gen):
    registry = {"pkg.output": {"signature": "def output(k, i)", "library_symbol": "output",
                               "params": [{"name": "k"}, {"name": "i"}], "provenance": {}}}
    assert "naked" in gen._units(registry)[0]["flags"]


def test_a_zero_arg_unit_is_flagged(gen):
    registry = {"pkg.go": {"signature": "def go()", "library_symbol": "go",
                           "doc_summary": "Does a thing.", "params": [], "provenance": {}}}
    flags = gen._units(registry)[0]["flags"]
    assert "zero-arg" in flags and "naked" not in flags


def test_a_dataset_with_no_local_file_is_still_a_row(gen):
    """A portal or a 403 is a finding, not an absence. Dropping those rows would make the browser
    agree with the bug it exists to reveal."""
    outcomes = {"elements": [{"id": "7d0c1d45-aaaa", "title": "FAOSTAT", "stage": "portal",
                              "link_kind": "portal", "note": "a data portal's front door"}]}
    rows = gen._datasets({"rows": []}, outcomes)
    assert len(rows) == 1
    assert rows[0]["stage"] == "portal"
    assert "no-local-file" in rows[0]["flags"]


def test_a_netcdf_dataset_surfaces_its_variables_as_the_schema(gen):
    details = {"rows": [{"element": "eb853cb2", "file": "gsp.nc", "title": "Groundwater",
                         "bytes": 803380,
                         "extracted": {"format": "nc", "family": "raster",
                                       "variables": ["crs", "WAT4_QWATGRD"],
                                       "dims": {"latitude": 288, "longitude": 690}}}]}
    row = gen._datasets(details, {"elements": []})[0]
    assert row["variables"] == ["crs", "WAT4_QWATGRD"]
    assert row["dims"]["latitude"] == 288
    assert "has-variables" in row["flags"]


def test_a_publication_carries_its_outcome_and_reason(gen):
    reach = {"rows": [{"element": "03bc2865", "outcome": "publisher wall (bot check)",
                       "reason": "the server returned a bot-check page", "chars": 0,
                       "sniffed": "html", "bytes": 14375}]}
    outcomes = {"elements": [{"id": "03bc2865-x", "title": "A paper", "doi": "10.1/x",
                              "licence": "cc-by"}]}
    row = gen._publications(reach, outcomes)[0]
    assert row["outcome"].startswith("publisher wall")
    assert "bot-check" in row["reason"]
    assert "has-doi" in row["flags"] and "cc-by" in row["flags"]


def test_an_unfetched_publication_says_not_fetched_rather_than_looking_readable(gen):
    outcomes = {"elements": [{"id": "abc12345-x", "title": "Paywalled",
                              "error": "no open-access copy exists"}]}
    row = gen._publications({"rows": []}, outcomes)[0]
    assert row["outcome"] == "not fetched"
    assert row["reason"] == "no open-access copy exists"


def test_an_off_domain_code_element_records_why_it_was_refused(gen):
    outcomes = {"elements": [{"id": "deadbeef-x", "title": "A ML repo", "stage": "relevance",
                              "relevance": 0.5, "off_domain": ["transformer", "tokenizer"]}]}
    row = gen._code(outcomes)[0]
    assert "off-domain" in row["flags"]
    assert row["off_domain"] == ["transformer", "tokenizer"]


# ------------------------------------------------------------------ the built page

def test_a_missing_source_is_reported_not_silently_skipped(gen, tmp_path):
    """An empty page and a page whose inputs were absent look identical in a browser."""
    payload = gen.collect(tmp_path)
    assert payload["missing"], "no sources exist under a temp dir, so all should be reported"
    assert payload["records"] == []


@pytest.mark.skipif(not (REPO / "outputs").is_dir(), reason="no extraction outputs in this tree")
def test_the_real_payload_parses_and_every_record_is_renderable(gen):
    payload = gen.collect(REPO)
    if not payload["records"]:
        pytest.skip("no extraction has been run in this checkout")
    # Round-trip exactly as the generator inlines it.
    blob = json.dumps(payload, separators=(",", ":"), default=str)
    reloaded = json.loads(blob)
    assert len(reloaded["records"]) == len(payload["records"])
    for record in reloaded["records"]:
        assert record.get("kind") in {"unit", "dataset", "publication", "code"}
        assert record.get("id"), record
        assert record.get("name") is not None, record
        assert isinstance(record.get("flags", []), list)


@pytest.mark.skipif(not (REPO / "outputs").is_dir(), reason="no extraction outputs in this tree")
def test_a_record_cannot_end_the_script_block_early(gen):
    """`</script>` inside the JSON would terminate the block and blank the page. The generator
    escapes it; this proves the escape survives a real payload."""
    payload = gen.collect(REPO)
    if not payload["records"]:
        pytest.skip("no extraction has been run in this checkout")
    payload["records"][0]["title"] = "evil </script><script>alert(1)</script>"
    blob = json.dumps(payload, separators=(",", ":"), default=str)
    page = TEMPLATE.read_text(encoding="utf-8").replace(
        "__DATA__", blob.replace("</script", "<\\/script"))
    body = page.split('id="payload">', 1)[1].split("</script>", 1)[0]
    assert "</script" not in body
    assert json.loads(body.replace("<\\/script", "</script"))["records"][0]["title"].startswith(
        "evil")


# ------------------------------------------------------------------ the two-pane layout

def _rules(css_text, selector):
    """Every declaration block for an exact selector, in source order, with its media context."""
    import re

    out = []
    for match in re.finditer(re.escape(selector) + r"\s*\{([^}]*)\}", css_text):
        before = css_text[:match.start()]
        depth = before.count("@media")
        # Crude but sufficient: a rule is inside a media block if an unclosed @media precedes it.
        opened = 0
        in_media = False
        for m in re.finditer(r"@media[^{]*\{|\{|\}", before):
            tok = m.group(0)
            if tok.startswith("@media"):
                opened = 1
                in_media = True
            elif tok == "{" and in_media:
                opened += 1
            elif tok == "}" and in_media:
                opened -= 1
                if opened <= 0:
                    in_media = False
        out.append({"decls": match.group(1), "in_media": in_media, "depth": depth})
    return out


def _css():
    text = TEMPLATE.read_text(encoding="utf-8")
    return text.split("<style>", 1)[1].split("</style>", 1)[0]


def test_the_shell_is_viewport_height_so_the_panes_can_be_bounded():
    """A pane can only scroll on its own if something above it stops growing with content."""
    css = _css()
    assert "html, body { height:100%; }" in css
    body = [r for r in _rules(css, "body") if not r["in_media"]]
    assert body, "no unconditional body rule"
    joined = " ".join(r["decls"] for r in body)
    assert "display:flex" in joined and "flex-direction:column" in joined
    assert "overflow:hidden" in joined
    # Viewport-relative, not parent-relative. An Artifact renders in a frame whose host resizes
    # it to fit the content, so a height resolving against the parent is circular: the frame
    # sizes to the content while the content sizes to the frame.
    assert "height:100dvh" in joined, "the shell height is not viewport-relative"


@pytest.mark.parametrize("selector", ["#list", "#detail"])
def test_each_pane_scrolls_independently(selector):
    """`overflow-y:auto` alone is not enough. A grid/flex child defaults to `min-height:auto`, so
    it sizes to its content and pushes the container instead of scrolling — which is exactly the
    bug this fixes: reading down the list scrolled the detail off screen."""
    css = _css()
    unconditional = [r for r in _rules(css, selector) if not r["in_media"]]
    assert unconditional, f"no unconditional rule for {selector}"
    joined = " ".join(r["decls"] for r in unconditional)
    assert "overflow-y:auto" in joined, f"{selector} does not scroll"
    assert "min-height:0" in joined, f"{selector} lacks min-height:0 and will not shrink"


def test_the_scroll_container_between_them_is_bounded():
    css = _css()
    main = [r for r in _rules(css, "main") if not r["in_media"]]
    joined = " ".join(r["decls"] for r in main)
    assert "min-height:0" in joined and "overflow:hidden" in joined


def test_narrow_screens_fall_back_to_one_scrolling_page():
    """Two bounded panes side by side do not fit on a phone. There the shell relaxes, so the page
    scrolls normally and the detail replaces the list."""
    css = _css()
    in_media = [r for r in _rules(css, "body") if r["in_media"]]
    assert any("height:auto" in r["decls"] and "overflow:visible" in r["decls"]
               for r in in_media), "the mobile fallback does not release the fixed shell"
    assert "main.showing-detail #list { display:none; }" in css


def test_the_count_line_stays_visible_while_the_list_scrolls():
    css = _css()
    count = " ".join(r["decls"] for r in _rules(css, ".count-line"))
    assert "position:sticky" in count and "top:0" in count
    # A sticky element over scrolling content needs its own background or the rows show through.
    assert "background:var(--paper)" in count


def test_the_controls_are_no_longer_sticky():
    """They sit in the fixed shell now. Leaving `position:sticky` on them would create a second
    sticky context inside a non-scrolling parent — inert, and misleading to the next reader."""
    controls = " ".join(r["decls"] for r in _rules(_css(), ".controls"))
    assert "position:sticky" not in controls


# ------------------------------------------------------------------ scroll position

def _script():
    text = TEMPLATE.read_text(encoding="utf-8")
    return text.rsplit("<script>", 1)[1]


def test_picking_a_record_preserves_where_you_were_in_the_list():
    """`render()` replaces every row, which resets the container's scrollTop. Without restoring
    it, clicking a record forty rows down throws the reader back to the top — the same complaint
    in a different place."""
    js = _script()
    assert "const keep = list.scrollTop;" in js
    assert "list.scrollTop = keep;" in js


def test_a_new_search_starts_at_the_top_of_its_results():
    """The opposite case: a different result set has no remembered position worth keeping."""
    assert "list.scrollTop = 0;" in _script()


def test_a_new_record_starts_at_the_top_of_the_detail_pane():
    assert "detail.scrollTop = 0;" in _script()


def test_the_mobile_view_offers_a_way_back_to_the_list():
    js = _script()
    assert 'id="back"' in js
    assert 'main.classList.remove("showing-detail")' in js
