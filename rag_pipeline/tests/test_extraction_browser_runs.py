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
