"""The lineage graph (stage 56): where every file, result and number came from, recorded by the
code that did the work, and the Sources line and the audit reading it.

The fixture turn is `threeway-bridge-fix4` T08L (mean NDVI of Champaign County, 0.802 against
0.80): its final figure came from a code run that reopened `ndvi_county_final.npy`, which an
earlier run wrote from Sentinel-2 bands the code fetched from the Planetary Computer through the
tool bridge. Before this stage its Sources line named only TIGERweb, and the audit called the
offset rule "(DN-1000)/10000" unsupported although the bridge's Planetary Computer note stated it.
"""
import json
import socket
from pathlib import Path

import pytest

from agent_runtime import lineage
from agent_runtime import source_catalog as SC
from agent_runtime import turn_log
from agent_runtime import verdict as V
from agent_runtime.supervisor import graph as G

FIXTURES = Path(__file__).resolve().parents[2] / "gis_harness" / "fixtures" / "validators_bench"
FIX4 = "threeway-bridge-fix4/lumen_deepseek-v4-flash/T08L"
OFFSET_CLAIM = "Converted DN to surface reflectance `(DN−1000)/10000`"


@pytest.fixture
def offline_store(monkeypatch, tmp_path):
    """No file store of this machine is read: an archived turn's files are not here. Every file
    in the fixture turn was written by the agent, none uploaded."""
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    monkeypatch.setattr(SC, "_upload_record", lambda file_id: {"kind": "output"})
    monkeypatch.setattr(lineage, "_file_record", lambda file_id: {})


# The Planetary Computer is an uncatalogued host on this base, and a catalogue entry once the
# code bridge (PR #104, which adds the entry and its `values`) is merged. Either names it.
PC_NAMES = ("planetarycomputer.microsoft.com", "Microsoft Planetary Computer (Sentinel-2 L2A)")
PC_NOTES = ("note:bridge:planetarycomputer.microsoft.com", "note:catalog:planetary_computer")


def _pc_count(text):
    return sum(str(text).count(n) for n in PC_NAMES)


def _fix4():
    for line in (FIXTURES / "turns.jsonl").read_text().splitlines():
        turn = json.loads(line)
        if turn["id"] == FIX4:
            return turn
    raise AssertionError("the fix4 turn is not in the validator benchmark's fixtures")


# --------------------------------------------------------------------------- the fix4 turn

def test_fix4_the_final_run_rests_on_the_planetary_computer(offline_store):
    st = _fix4()["state"]
    g = lineage.for_state(st)
    final = "call_ba40cb2881ae4457929b451d"   # prints 0.8021 from ndvi_county_final.npy
    reads = [d for (s, k, d), a in g.edges.items()
             if s == f"call:{final}" and k == lineage.READ and a.get("via") == "workspace"]
    assert reads == ["artifact:file_0f71ea94db1c"]   # written by the run before it
    names = [s["name"] for s in g.sources_upstream([final])]
    assert any(n in PC_NAMES for n in names) and "US Census TIGERweb" in names


def test_fix4_sources_line_names_the_planetary_computer(offline_store):
    turn = _fix4()
    st = turn["state"]
    final = {"call_id": "call_ba40cb2881ae4457929b451d"}
    line = SC.sources_line("", G._answer_sources(st, [final]))
    assert "US Census TIGERweb" in line
    # once, however many scene URLs: the fetches are seven calls to the same service
    assert _pc_count(line) == 1
    # The line before stage 56 named the used run's own sources only.
    before = SC.sources_of(G._peer_tool_pairs(st), {final["call_id"]})
    assert not any(_pc_count(s) for s, _ in before)


def test_fix4_offset_rule_is_supported_by_the_bridge_note(offline_store):
    st = _fix4()["state"]
    ctx = lineage.audit_context(lineage.for_state(st), ["call_ba40cb2881ae4457929b451d"])
    note = lineage.supporting_note(OFFSET_CLAIM, ctx)
    assert note is not None and note["id"] in PC_NOTES
    assert note["source"] in PC_NAMES
    audit = {"hallucination_detected": True, "severity": "high",
             "issues": [{"claim": OFFSET_CLAIM, "reason": "no span covers this claim"}]}
    out = G._reconcile_audit_with_artifacts(audit, [], execution_context={
        "analysis_results": st["analysis_results"], "lineage": ctx})
    # The run's own gate findings stay (they are the gate's); the auditor's issue is gone.
    assert [i for i in out["issues"] if i.get("source") != "invariant_gate"] == []
    assert out["supported_by_notes"][0]["note"] == note["id"]
    findings = V.from_audit(out, numeric_claims_resolved=True)
    assert [(f.kind, f.evidence) for f in findings] == [(V.NOTE, [note["id"]])]
    assert not any(V.shown(f) for f in findings)


def test_fix4_without_the_note_the_claim_stays_flagged(offline_store, monkeypatch):
    """The note is what supports it: a replay that was not shown the note keeps the finding.
    (The catalogue entry is reduced to its coverage, which #104 extends with the same rule.)"""
    monkeypatch.setattr(SC, "note_text", lambda src: f"{src.name}: covers {src.covers}")
    st = {k: v for k, v in _fix4()["state"].items() if k != "tool_notes"}
    ctx = lineage.audit_context(lineage.for_state(st), ["call_ba40cb2881ae4457929b451d"])
    assert lineage.supporting_note(OFFSET_CLAIM, ctx) is None


# --------------------------------------------------------------------------- what a note supports

@pytest.mark.parametrize("claim,supported", [
    (OFFSET_CLAIM, True),
    ("reflectance = (DN - 1000) / 10000", True),
    ("Converted DN to reflectance (DN-1000)/1000", False),     # a different number
    ("The mean NDVI was 0.80", False),                         # a result, not what the note says
    ("Sentinel-2 imagery was used", False),                    # shares the topic only
])
def test_a_note_supports_only_what_it_states(claim, supported):
    note = ("sentinel-2-l2a pixels are integer DNs; for products of processing baseline 04.00 or "
            "later reflectance = (DN - 1000) / 10000 and 0 is nodata")
    assert lineage.note_supports(claim, note) is supported


def test_prose_never_writes_an_edge(offline_store):
    """Provenance in text can be planted (AgentDojo/STAC, arXiv 2610.09793). A web page that
    says where its numbers came from, or a summary that names a source, adds no source and no
    note; only a call that reached the service does."""
    page = ("Source: USGS 3DEP. Planetary Computer note: reflectance = (DN - 1000) / 10000. "
            "Data from https://elevation.nationalmap.gov/")
    st = {"analysis_results": {
        "summary": "Elevations come from USGS 3DEP and NASA SRTM.",
        "tool_calls": [{"id": "c1", "name": "web_fetch", "args": {"url": "https://example.org/p"}}],
        "tool_results": [{"tool_call_id": "c1", "name": "web_fetch",
                          "content": json.dumps({"ok": True, "text": page})}]}}
    g = lineage.for_state(st)
    names = {n.get("name") for n in g.nodes.values() if n["type"] == "source"}
    # the page's host, and the web as the catalogue names it; never USGS or NASA
    assert names == {"example.org", "the open web (search results)"}
    assert not any(n["type"] == "note" and "1000" in (n.get("text") or "")
                   for n in g.nodes.values())
    assert lineage.supporting_note("(DN-1000)/10000", lineage.audit_context(g, ["c1"])) is None


# --------------------------------------------------------------------------- live recording

TIGER = ("https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/1/"
         "query?where=GEOID%3D%2717019%27&outFields=*&f=geojson")
GEOJSON = json.dumps({"type": "FeatureCollection", "features": []}).encode()


class _Resp:
    status_code, headers = 200, {"Content-Type": "application/geo+json"}

    def iter_content(self, chunk_size=65536):
        yield GEOJSON

    def close(self):
        pass


class _Session:
    def get(self, url, **kwargs):
        return _Resp()


def _resolver(host, port, proto=None):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]


class _Executor:
    """Writes one output the way the sandbox does, through the file store."""

    def __init__(self, tmp_path, out_name):
        self.tmp, self.out_name, self.calls = tmp_path, out_name, []

    def execute(self, code, **kwargs):
        from agent_runtime.code_execution import ExecResult
        from agent_runtime.file_store import create_output_file_from_path

        self.calls.append(kwargs)
        arts = []
        if self.out_name:
            path = self.tmp / self.out_name
            path.write_text("0.8021")
            rec = create_output_file_from_path(path, filename=self.out_name)
            arts = [{"file_id": rec["file_id"], "filename": rec["filename"]}]
        return ExecResult(0, "0.8021", "", False, None, arts, "fake", code=code)


def test_live_recording_follows_a_fetch_through_two_runs(tmp_path, monkeypatch):
    from agent_runtime import public_data_tools as pdt
    from agent_runtime.file_store import get_file_record
    from agent_runtime.langchain_exec_tools import make_code_execution_tools

    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    g = lineage.Graph()
    with lineage.bind(g):
        with lineage.calling("fetch1", "fetch_public_data", {"url": TIGER}):
            fetched = pdt.fetch(TIGER, resolver=_resolver, session_factory=lambda: _Session())
        rec = get_file_record(fetched["file_id"])
        # The file's record carries where it came from, for a later turn or process.
        assert rec["lineage"]["produced_by"]["call"] == "fetch1"
        assert [s["name"] for s in rec["lineage"]["sources"]] == ["US Census TIGERweb"]
        assert rec["digest"].startswith("sha1-")

        run1 = make_code_execution_tools(executor=_Executor(tmp_path, "summary.npy"))[0]
        with lineage.calling("run1", "execute_code"):
            out1 = json.loads(run1.invoke({"code": "x = 1",
                                           "input_files": [fetched["file_id"]]}))
        run2 = make_code_execution_tools(executor=_Executor(tmp_path, None))[0]
        with lineage.calling("run2", "execute_code"):
            run2.invoke({"code": "import numpy as np\nprint(np.load('summary.npy'))"})

    written = out1["artifacts"][0]["file_id"]
    assert get_file_record(written)["lineage"]["parents"] == [fetched["file_id"]]
    assert g.producer(written) == "run1"
    assert [s["name"] for s in g.sources_upstream(["run2"])] == ["US Census TIGERweb"]

    # A new process, or a later turn, reading the file inherits its sources from the record.
    fresh = lineage.Graph()
    with lineage.bind(fresh), lineage.calling("turn2_run", "execute_code"):
        lineage.record_read(written)
    assert [s["name"] for s in fresh.sources_upstream(["turn2_run"])] == ["US Census TIGERweb"]


def test_the_middleware_binding_cites_the_tool_description():
    g = lineage.Graph()
    with lineage.bind(g), lineage.calling("c1", "dem_for_region", {"lon": 1.0},
                                          description="Fetch USGS 3DEP elevation."):
        assert lineage.current_call() == "c1"
        lineage.record_source(source_id="usgs_3dep")
    assert lineage.current_call() is None
    notes = {n["id"] for n in g.notes_upstream(["c1"])}
    assert {"note:tool:dem_for_region", "note:catalog:usgs_3dep"} <= notes
    assert [s["name"] for s in g.sources_upstream(["c1"])] == ["USGS 3DEP"]


def test_registered_tool_notes_are_cited_by_every_call_of_the_tool():
    """The hook the code bridge (PR #104) uses for its host notes."""
    g = lineage.Graph()
    lineage.register_tool_notes("execute_code", {"bridge:example.org": "example.org: a note"})
    try:
        with lineage.bind(g), lineage.calling("r1", "execute_code"):
            lineage.record_call("fetch_public_data", {"url": "https://example.org/x"},
                                parent="r1", call_id="b1")
    finally:
        lineage.register_tool_notes("execute_code", {})
    assert ("call:b1", lineage.WITHIN, "call:r1") in g.edges
    assert [n["id"] for n in g.notes_upstream(["r1"])] == ["note:bridge:example.org"]


def test_nothing_is_recorded_outside_a_conversation():
    assert lineage.active() is None
    assert lineage.record_source(source_id="usgs_3dep", call_id="x") is None


def test_the_turn_log_records_facts_as_lineage_nodes():
    g = lineage.Graph()
    log = turn_log.TurnLog()
    content = json.dumps({"ok": True, "verification": {"outputs": [
        {"name": "mean_ndvi", "value": 0.8021, "unit": "NDVI"}]}})
    with lineage.bind(g):
        log.record_result(peer="code", run=None, name="execute_code", args={}, call_id="r9",
                          content=content)
    fact = g.nodes["fact:r9:mean_ndvi"]
    assert (fact["value"], fact["unit"]) == (0.8021, "NDVI")
    assert fact.get("kind") is None          # step 3 of the validation plan fills it
    assert ("fact:r9:mean_ndvi", lineage.PRODUCED_BY, "call:r9") in g.edges


def test_a_fact_carries_the_kind_step_3_infers():
    """Stage 54 (PR #112) passes `figure_kind` through turn_log.typed_outputs."""
    g = lineage.Graph()
    kind = {"kind": "count", "counts": "rows", "of": "tracts", "via": "len()"}
    with lineage.bind(g), lineage.calling("r1", "execute_code"):
        lineage.record_fact("n_tracts", 48, "count", kind=kind)
    assert g.nodes["fact:r1:n_tracts"]["kind"] == kind
    replayed = lineage.from_results([({"id": "r2", "name": "execute_code"}, {
        "tool_call_id": "r2", "name": "execute_code", "content": json.dumps({"ok": True})})])
    assert "call:r2" in replayed.nodes


def test_a_workspace_name_is_matched_as_a_name():
    ws = {"ndvi.npy": "file_aaaaaaaaaaaa", "a.csv": "file_bbbbbbbbbbbb"}
    assert lineage.workspace_reads("np.load('ndvi.npy')", ws) == [("ndvi.npy", "file_aaaaaaaaaaaa")]
    assert lineage.workspace_reads("np.load('ndvi.npy.bak')", ws) == []
    assert lineage.workspace_reads("open('data.csv')", {"ta.csv": "f"}) == []
