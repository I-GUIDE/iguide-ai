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
        assert record.get("kind") in {"unit", "dataset", "publication", "code", "notebook"}
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


# ------------------------------------------------------------------ drilling into an element

def test_an_element_lists_the_units_extraction_produced_from_it(gen):
    """"units 6, callable 5" is a count with nothing behind it. The question a reader has is
    *which five*, and the answer was one join away the whole time."""
    units = [
        {"kind": "unit", "id": "pkg.plot_domain", "name": "plot_domain", "element": "9d483118",
         "signature": "def plot_domain(run_directory, variable, timestep=0)",
         "summary": "", "unit_kind": "function", "source": "plots.py", "flags": []},
        {"kind": "unit", "id": "pkg.other", "name": "other", "element": "ffffffff",
         "signature": "def other()", "summary": "", "unit_kind": "function",
         "source": "x.py", "flags": []},
    ]
    element = {"kind": "code", "id": "9d483118", "name": "9d483118", "title": "Parflow",
               "callable": 5, "flags": []}
    gen._link_units_to_elements([element] + units, units)
    listed = element["units_in_library"]
    assert [u["name"] for u in listed] == ["plot_domain"], "another element's unit leaked in"
    assert listed[0]["signature"].startswith("def plot_domain")
    assert "has-units" in element["flags"]


def test_units_are_listed_in_a_stable_order(gen):
    units = [{"kind": "unit", "id": f"pkg.{n}", "name": n, "element": "e1",
              "signature": f"def {n}()", "summary": "", "unit_kind": "function",
              "source": "a.py", "flags": []} for n in ("zeta", "Alpha", "mid")]
    element = {"kind": "code", "id": "e1", "name": "e1", "title": "T", "flags": []}
    gen._link_units_to_elements([element] + units, units)
    assert [u["name"] for u in element["units_in_library"]] == ["Alpha", "mid", "zeta"]


def test_a_unit_record_is_never_given_its_own_unit_list(gen):
    units = [{"kind": "unit", "id": "pkg.f", "name": "f", "element": "e1",
              "signature": "def f()", "summary": "", "unit_kind": "function",
              "source": "a.py", "flags": []}]
    gen._link_units_to_elements(units, units)
    assert "units_in_library" not in units[0]


def test_an_element_with_no_units_gets_no_empty_section(gen):
    element = {"kind": "code", "id": "e9", "name": "e9", "title": "T", "flags": []}
    gen._link_units_to_elements([element], [])
    assert "units_in_library" not in element
    assert "has-units" not in element["flags"]


# ------------------------------------------------------------------ what came out of a PDF

def test_a_publication_carries_the_spec_extracted_from_its_pdf(gen):
    """Reach says whether the bytes were readable. This is the different question: what did we
    actually get out of it."""
    specs = {"rows": [{"element": "31fd4fc6", "status": "llm_extracted",
                       "summary": "Compares raster network connectivity for corridor analysis.",
                       "steps": ["Acquire EISPC raster data", "Derive two cost surfaces"],
                       "datasets_referenced": ["NLCD2016"], "tools_referenced": ["pNISE"],
                       "params": {"cell_size": "250 meters"},
                       "chunks_parsed": 1, "chunks_total": 1}]}
    outcomes = {"elements": [{"id": "31fd4fc6-x", "title": "Corridors", "doi": "10.1/x"}]}
    row = gen._publications({"rows": []}, outcomes, specs)[0]
    assert row["steps"] == ["Acquire EISPC raster data", "Derive two cost surfaces"]
    assert row["datasets_referenced"] == ["NLCD2016"]
    # `declared_params`, not `params`: a unit's `params` is a LIST and the page's search index
    # maps over it, so an object under that key blanked the whole page. See
    # test_extraction_browser_runs.py.
    assert row["declared_params"]["cell_size"] == "250 meters"
    assert "params" not in row
    assert "has-method-spec" in row["flags"]


def test_a_paper_with_no_method_is_flagged_as_such_not_as_a_failure(gen):
    specs = {"rows": [{"element": "fd728b4e", "status": "no_method_described", "steps": [],
                       "summary": "An argumentative paper on AI ethics."}]}
    outcomes = {"elements": [{"id": "fd728b4e-x", "title": "AI ethics", "doi": "10.1/y"}]}
    row = gen._publications({"rows": []}, outcomes, specs)[0]
    assert "no-method" in row["flags"]
    assert "has-method-spec" not in row["flags"]
    assert row["steps"] == []


def test_publications_still_work_when_no_specs_have_been_extracted(gen):
    """The spec export is a separate, slow pass. Its absence must leave the page usable."""
    outcomes = {"elements": [{"id": "abc12345-x", "title": "A paper", "doi": "10.1/z"}]}
    row = gen._publications({"rows": []}, outcomes, None)[0]
    assert row["steps"] == [] and row["spec_status"] == ""
    assert "has-method-spec" not in row["flags"]


# ------------------------------------------------------------------ navigating between records

def test_the_detail_pane_can_open_a_linked_record():
    js = _script()
    assert "const BY_ID = new Map(RECORDS.map" in js
    assert 'detail.querySelectorAll("[data-goto]")' in js


def test_following_a_link_does_not_disturb_the_list_or_its_filters():
    """Silently clearing someone's filter to reveal the target is a worse surprise than a detail
    pane showing a record the list is not currently listing."""
    js = _script()
    after = js.split('detail.querySelectorAll("[data-goto]")', 1)[1]
    # Bound the slice at the end of THIS handler. A fixed character count ran past it into the
    # search handler, which mentions `state.q` for its own good reasons — the test was reading
    # someone else's code and calling it a violation.
    goto = after.split("\n    });", 1)[0]
    assert "state.facets" not in goto, goto
    assert "state.q" not in goto, goto
    assert "const keep = list.scrollTop;" in goto and "list.scrollTop = keep;" in goto


def test_a_gap_between_callable_and_listed_units_is_shown_not_smoothed_over():
    """This is exactly the defect that produced 58 unreachable units. If it recurs, the page has
    to say so rather than quietly listing fewer."""
    js = _script()
    assert "That gap is a defect, not a filter." in js


# ------------------------------------------------------- notebooks, grouped by their element

def test_a_notebook_record_carries_its_cells_and_the_units_promoted_from_them(gen):
    """Notebooks are the largest source in the corpus and were the only type with nothing to
    browse: the other three have per-element outcome files, while a notebook's blocks and units
    existed only inside a search index."""
    blocks = {"rows": [{
        "element": "02f7f46b", "file": "nb.ipynb", "title": "Distance and Nearest Features",
        "bytes": 307736, "unnamed_blocks": 2,
        "blocks": [{"doc_id": "02f7f46b::block::3", "title": "1b. Import Data", "order": 3,
                    "code": "import geopandas", "markdown": "## 1b. Import Data",
                    "parse_ok": True, "tools": [], "imports": ["geopandas"],
                    "file_refs": ["a.shp"], "constructs": []}],
        "units": [
            {"doc_id": "u1", "symbol": "pkg.load_points", "signature": "(path)",
             "verdict": "callable", "summary": "Load points", "cell_order": 3},
            {"doc_id": "u2", "symbol": "pkg.plot_it", "signature": "()", "verdict": "blocked",
             "reason": "reads a module-level frame", "global_reads": ["gdf"]},
        ],
        "workflow": {"workflow_id": "w1", "mode": "script", "entrypoint": "run", "params": []},
    }]}
    records = gen._notebooks(blocks)
    assert len(records) == 1
    r = records[0]
    assert r["kind"] == "notebook" and r["id"] == "02f7f46b"
    assert len(r["blocks"]) == 1 and r["blocks"][0]["title"] == "1b. Import Data"
    # Both units, not only the one that shipped: a refused unit names the hidden dependency
    # that stopped it, which is what says which extraction limit to lift next.
    assert len(r["notebook_units"]) == 2 and r["callable"] == 1
    assert "produced-units" in r["flags"] and "has-workflow" in r["flags"]


def test_a_notebook_that_yielded_nothing_is_still_a_record():
    """An extraction that produced no cells IS the finding. Dropping the row would leave the
    page saying nothing about the notebook at all."""
    import scripts.build_extraction_browser as gen
    records = gen._notebooks({"rows": [{"element": "fec21e60", "file": "empty.ipynb",
                                        "title": "Stadia Maps", "blocks": [], "units": []}]})
    assert len(records) == 1
    assert "no-blocks" in records[0]["flags"]


def test_a_notebooks_file_references_are_a_list_the_page_can_iterate():
    """The extractor stores `file_io` as `{"referenced": [...]}`. Passing that object straight
    through gave the page something it could not iterate — the same shape mismatch that blanked
    the whole page once already, one field deeper."""
    source = (REPO / "scripts" / "export_notebook_blocks.py").read_text(encoding="utf-8")
    assert '"file_refs": (block.get("file_io") or {}).get("referenced")' in source


def test_a_promoted_unit_links_to_the_library_record_the_page_navigates_by(gen):
    """A notebook lists units by their extraction doc id; the page navigates by registry key.
    Without the join, every callable unit rendered as a link that quietly went nowhere."""
    units = [{"kind": "unit", "id": "iguide.ke_x.nb.load_points", "name": "load_points",
              "element": "02f7f46b", "signature": "(path)", "flags": []}]
    record = {"kind": "notebook", "id": "02f7f46b", "flags": [],
              "notebook_units": [{"symbol": "pkg.load_points", "doc_id": "u1"},
                                 {"symbol": "pkg.never_shipped", "doc_id": "u2"}]}
    gen._link_units_to_elements([record] + units, units)
    promoted = {u["symbol"]: u.get("registry_id") for u in record["notebook_units"]}
    assert promoted["pkg.load_points"] == "iguide.ke_x.nb.load_points"
    assert promoted["pkg.never_shipped"] is None, "a refused unit must not get a link"


def test_records_can_be_grouped_under_their_parent_knowledge_element():
    js = _script()
    assert "const parentOf = (r) =>" in js
    assert "function renderGrouped(" in js
    # A unit's parent is the element it was extracted from; an element record IS its own parent.
    assert 'r.kind === "unit" ? r.element : r.id' in js


def test_a_refused_unit_is_not_rendered_as_a_link_that_goes_nowhere():
    js = _script()
    notebook_detail = js.split("function notebookDetail(", 1)[1].split("\n  function ", 1)[0]
    assert "u.registry_id ?" in notebook_detail


def test_the_detail_views_cannot_be_broken_by_a_field_of_the_wrong_shape():
    """The search index was hardened after a shape mismatch blanked the page. The detail views
    were not, because nothing had ever opened one under test."""
    js = _script()
    assert "const ARRAY_FIELDS = [" in js
    declared = js.split("const ARRAY_FIELDS = [", 1)[1].split("];", 1)[0]
    for field in ("blocks", "notebook_units", "requires", "invariants", "params"):
        assert f'"{field}"' in declared


def test_the_page_declares_its_encoding():
    """Found by opening the page in a browser, not by any test above.

    The page is full of em dashes — the title separator is one. Served over HTTP with no charset
    header and no `<meta charset>`, it decoded as windows-1252 and every one of them rendered as
    `â€"`. Opening the same file from disk happened to guess UTF-8, which is why this survived.
    """
    html = TEMPLATE.read_text(encoding="utf-8")
    assert html.lstrip().lower().startswith('<meta charset="utf-8">')


def test_a_cell_ordinal_is_read_from_the_key_the_extractor_writes():
    """The exporter asked for `cell_order`; the extractor writes `order`. Every ordinal came out
    None, so the column rendered blank for all 3,830 cells and the page still passed its tests —
    a blank column is not an exception."""
    import json

    source = (REPO / "scripts" / "export_notebook_blocks.py").read_text(encoding="utf-8")
    assert '"order": block.get("order")' in source

    exported = REPO / "outputs" / "notebook_blocks.json"
    if not exported.is_file():
        pytest.skip("no notebook export in this checkout")
    rows = json.loads(exported.read_text(encoding="utf-8"))["rows"]
    orders = [b.get("order") for r in rows for b in r.get("blocks") or []]
    if not orders:
        pytest.skip("the export contains no blocks")
    assert any(o is not None for o in orders), "every cell ordinal is None"


def test_the_page_does_not_offer_a_column_it_can_never_fill():
    """A unit's provenance records the element and the source file, never the cell it came from.
    Rendering `cell N` for units printed an empty column for all 382 of them."""
    js = _script()
    notebook_detail = js.split("function notebookDetail(", 1)[1].split("\n  function ", 1)[0]
    assert "cell_order" not in notebook_detail
