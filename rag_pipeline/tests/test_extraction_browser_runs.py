"""Execute the generated page's JavaScript against the real payload.

Everything in ``test_extraction_browser.py`` checks the DATA CONTRACT — each record has an id, a
kind, a renderable name — and every one of those tests passed while the published page rendered
**nothing at all**.

The cause was a field-name collision. A unit's ``params`` is a LIST of parameter objects; I gave a
publication's declared parameters the same key, and they are an OBJECT. The search index is built
in one pass before anything renders, and it called ``.map()`` on every record's ``params``.
``{}.map is not a function`` threw on the first publication that had any, the IIFE died, and the
page came up blank. A per-record schema check cannot see that: every record was individually
valid, and the failure was in what the page DID with them.

So this file runs the actual script. It needs node, which is not a test dependency, so it skips
cleanly without one — a skip here means "unverified", not "fine".
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PAGE = REPO / "outputs" / "extraction_browser.html"

# A DOM stub with just enough surface for the page's render path. Deliberately not jsdom: the
# point is to run the real script cheaply, and a stub makes the assertions explicit.
HARNESS = r"""
const fs = require("fs");
const HTML = fs.readFileSync(process.argv[2], "utf8");
const payload = HTML.split('id="payload">')[1].split("</script>")[0];
const js = HTML.split("<script>").pop().split("</script>")[0];

function makeEl(id) {
  return {
    id, _children: [], textContent: "", innerHTML: "", hidden: false, scrollTop: 0, style: {},
    classList: { add() {}, remove() {}, contains: () => false },
    setAttribute() {}, getAttribute: () => null, addEventListener() {},
    appendChild(c) { this._children.push(c); },
    querySelectorAll: () => [], scrollIntoView() {}, onclick: null,
  };
}
const registry = new Map();
global.document = {
  getElementById(id) {
    if (id === "payload") return { textContent: payload };
    if (!registry.has(id)) registry.set(id, makeEl(id));
    return registry.get(id);
  },
  createElement(t) { return makeEl(t); },
  createDocumentFragment() { return makeEl("#fragment"); },
  querySelectorAll: () => [],
};
global.window = { matchMedia: () => ({ matches: false }), scrollTo() {},
                  document: global.document };
global.setTimeout = (fn) => { fn(); return 0; };
global.clearTimeout = () => {};

const out = { ok: false };
try {
  new Function(js)();
  const rows = registry.get("rows");
  // Rows go into a fragment that is appended once; count what actually got built.
  let built = 0;
  for (const child of (rows ? rows._children : [])) {
    built += child.id === "#fragment" ? child._children.length : 1;
  }
  out.ok = true;
  out.rows = built;
  out.countLine = (registry.get("countline") || {}).textContent || "";
  out.tally = (registry.get("tally") || {}).innerHTML || "";
  out.records = JSON.parse(payload.replace(/<\\\/script/g, "</script")).records.length;

  // --- drive the page, because rendering a list is not the whole page ---------------
  //
  // Everything above exercises the FIRST render only. Grouping and every detail view sit
  // behind a click, so a page that throws the moment someone picks a record would pass all
  // of it. `textContent = ""` does not clear the stub's children, so each render appends a
  // fresh fragment and the newest one is the current view.
  const newest = (el) => (el && el._children.length
    ? el._children[el._children.length - 1] : null);
  const detail = registry.get("detail");

  const flatten = (el) => {
    const acc = [];
    (function walk(n) {
      (n._children || []).forEach((c) => { acc.push(c); walk(c); });
    })(el);
    return acc;
  };

  // 1. Open a record of every kind and confirm each detail view builds.
  //
  // Reached through the kind chips rather than by scanning the first render: the list caps at
  // 800 rows and units alone exceed that, so four of the five kinds never appear in it.
  out.details = {};
  const kindChips = (registry.get("kinds") || { _children: [] })._children;
  for (const kind of ["notebook", "method unit", "dataset", "publication", "code element"]) {
    const chip = kindChips.find(
      (c) => (c.innerHTML || "").replace(/<span class="n">.*/, "") === kind);
    if (!chip) { out.details[kind] = null; continue; }
    chip.onclick();
    const rowsNow = flatten(newest(registry.get("rows"))).filter((c) => c.className === "row");
    // Match the kind SPAN, not the row text: a unit's meta column carries the name of the
    // extractor that produced it, so a loose match finds a unit for every kind.
    const hit = rowsNow.find((c) => (c.innerHTML || "").includes(
      '<span class="kind">' + kind + '</span>'));
    if (!hit) { out.details[kind] = null; continue; }
    hit.onclick();
    out.details[kind] = (detail.innerHTML || "").length;
    if (kind === "notebook") out.notebookDetail = detail.innerHTML || "";
  }
  const all = kindChips.find((c) => (c.innerHTML || "").startsWith("All"));
  if (all) all.onclick();

  // 2. Switch to the grouped view.
  const groupChips = (registry.get("grouping") || { _children: [] })._children;
  out.groupChipLabels = groupChips.map((c) => c.innerHTML);
  const byElement = groupChips.find((c) => (c.innerHTML || "").includes("By element"));
  if (byElement) {
    byElement.onclick();
    const frag = newest(registry.get("rows"));
    const groups = (frag ? frag._children : []).filter((c) => c.className === "grp");
    out.groups = groups.length;
    out.groupedCountLine = (registry.get("countline") || {}).textContent || "";
    out.groupHeads = groups.slice(0, 3).map((g) => {
      const head = (g._children || [])[0];
      return head ? head.innerHTML : "";
    });
    // 3. A collapsed group opens when its header is clicked.
    const first = groups[0];
    const head = first && first._children[0];
    out.rowsBeforeExpand = first ? first._children.length - 1 : -1;
    if (head) {
      head.onclick();
      const after = flatten(newest(registry.get("rows"))).filter((c) => c.className === "grp");
      out.rowsAfterExpand = after.length ? after[0]._children.length - 1 : -1;
    }
  }
} catch (e) {
  out.ok = false;
  out.error = String((e && e.message) || e);
  out.stack = String((e && e.stack) || "").split("\n").slice(0, 4).join(" | ");
}
process.stdout.write(JSON.stringify(out));
"""


def _run(page: Path) -> dict:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed, so the page script cannot be executed here")
    if not page.is_file():
        pytest.skip(f"{page} has not been generated in this checkout")
    harness = page.parent / "_run_page_harness.js"
    harness.write_text(HARNESS, encoding="utf-8")
    try:
        proc = subprocess.run([node, str(harness), str(page)], capture_output=True,
                              text=True, timeout=180)
    finally:
        harness.unlink(missing_ok=True)
    assert proc.returncode == 0, f"harness failed: {proc.stderr[:600]}"
    return json.loads(proc.stdout)


@pytest.fixture(scope="module")
def result():
    return _run(PAGE)


def test_the_page_script_runs_without_throwing(result):
    assert result["ok"], (
        f"the page throws before rendering anything: {result.get('error')}\n"
        f"{result.get('stack')}")


def test_the_page_actually_renders_rows(result):
    """A script that runs and renders nothing is the same blank page to a reader."""
    assert result["rows"] > 0, "no rows were built"


def test_the_count_line_reports_the_whole_payload(result):
    assert f"of {result['records']} record(s)" in result["countLine"], result["countLine"]


def test_the_header_tally_is_populated(result):
    assert "</b>" in result["tally"] and result["tally"].strip()


def test_every_kind_of_record_opens_without_throwing(result):
    """The first render is not the page. Every detail view sits behind a click, so a view that
    throws the moment someone picks that kind of record would pass every test above it."""
    built = {k: v for k, v in (result.get("details") or {}).items() if v is not None}
    assert built, "no record of any kind could be opened"
    empty = [k for k, v in built.items() if not v]
    assert not empty, f"opening these built an empty detail pane: {empty}"


def test_a_notebook_shows_its_extracted_cells(result):
    """The point of the notebook view: the artifacts extracted from THIS notebook, in order."""
    html = result.get("notebookDetail")
    if html is None:
        pytest.skip("no notebook record in this checkout's payload")
    assert 'class="blk"' in html, "no extracted cells rendered"
    assert "<pre>" in html, "cells rendered without their code"
    assert "cells extracted" in html


def test_the_grouped_view_organises_records_under_their_parent_element(result):
    assert "By element" in " ".join(result.get("groupChipLabels") or []), \
        "the grouping control is missing"
    assert (result.get("groups") or 0) > 1, "grouping produced no groups"
    assert "element(s)" in (result.get("groupedCountLine") or "")
    heads = result.get("groupHeads") or []
    assert heads and all(h.strip() for h in heads), "a group rendered without a header"


def test_a_group_header_toggles_its_members(result):
    """A group that cannot be opened or closed is just a heading."""
    before, after = result.get("rowsBeforeExpand"), result.get("rowsAfterExpand")
    if before is None or after is None:
        pytest.skip("grouping did not run in this checkout")
    assert before != after, f"clicking a group header changed nothing ({before} rows both times)"


def test_a_record_whose_field_has_an_unexpected_shape_cannot_blank_the_page(tmp_path):
    """The specific bug, pinned: a `params` that is an object rather than a list.

    Fixed at the source by renaming the publication field, and again here by making the search
    index defensive — because the next shape mismatch should cost a search term, not the page.
    """
    if not PAGE.is_file():
        pytest.skip("no generated page in this checkout")
    html = PAGE.read_text(encoding="utf-8")
    head, rest = html.split('id="payload">', 1)
    body, tail = rest.split("</script>", 1)
    payload = json.loads(body.replace("<\\/script", "</script"))

    hostile = dict(payload["records"][0])
    hostile.update({"id": "hostile-1", "name": "hostile", "title": "hostile",
                    "params": {"not": "a list"}, "schema": {"also": "not a list"},
                    "invariants": "a string", "requires": 42, "flags": []})
    payload["records"] = [hostile] + payload["records"]

    poisoned = tmp_path / "poisoned.html"
    poisoned.write_text(
        head + 'id="payload">'
        + json.dumps(payload, separators=(",", ":")).replace("</script", "<\\/script")
        + "</script>" + tail, encoding="utf-8")

    out = _run(poisoned)
    assert out["ok"], f"one malformed record still blanks the page: {out.get('error')}"
    assert out["rows"] > 0


def test_a_publications_declared_parameters_do_not_use_the_units_key():
    """Two shapes under one key is what caused this. Assert the separation at the source, so the
    generator cannot quietly reintroduce it."""
    source = (REPO / "scripts" / "build_extraction_browser.py").read_text(encoding="utf-8")
    publications = source.split("def _publications(", 1)[1].split("\ndef ", 1)[0]
    assert '"declared_params"' in publications
    assert '"params":' not in publications, (
        "a publication record is using the key a unit uses for its parameter LIST")
