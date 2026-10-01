"""What the code agent is actually handed.

Three defects on the path between a tool result and the code peer's prompt: an output file_id
captured but never rendered, a re-grounding pass that never told this peer what went ungrounded,
and a sampled layer indistinguishable from a complete one.

Two more were attempted and withdrawn. Replacing the raw `json.dumps(...)[:N]` slice with a
hand-rolled compactor made things WORSE: on a real payload whose leaves are all short — a
GeoJSON of 120 features — it returned 60 characters of nothing where the slice returned 2,000
including the summary, and `evidence_quality._format_execution_context` already returns 8,048
with the summary AND the map layer. Making a bare filename stage under its clean name silently
overwrote same-named files in /work and truncated any real name containing a double underscore.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime.supervisor import graph as g


# --- 2. the output file_id reaches the prompt ----------------------------------------------

def test_a_produced_file_is_named_by_its_id_not_only_its_filename():
    row = {"tool": "add_map_layer", "args": {}, "outputs": "income.geojson",
           "file_id": "file_abc123"}
    line = g._ledger_lines([row])[0]
    assert "income.geojson" in line
    assert "file_abc123" in line, "the id was captured and never rendered"


def test_a_read_only_tool_is_never_said_to_have_produced_a_file():
    """read_text_file and inspect_file_for_analysis both return the file_id of the file the
    USER uploaded and create nothing. Keying the bracket on file_id alone claimed they had
    produced it — and the grounding auditor reads these same lines as evidence, so it would
    confirm the fabrication."""
    line = g._ledger_lines([{"tool": "read_text_file", "args": {"path": "income.csv"},
                             "file_id": "file_uploaded_by_the_user"}])[0]
    assert "produced" not in line
    assert "file_uploaded_by_the_user" not in line


def test_a_row_with_neither_says_nothing_about_files():
    line = g._ledger_lines([{"tool": "t", "args": {}}])[0]
    assert "produced" not in line


def test_a_failed_call_still_claims_no_output():
    row = {"tool": "t", "args": {}, "failed": True, "error": "boom",
           "outputs": "phantom.geojson", "file_id": "file_phantom"}
    line = g._ledger_lines([row])[0]
    assert "phantom" not in line and "file_phantom" not in line


# --- 5. a sampled layer says so ------------------------------------------------------------

def test_the_full_count_reaches_the_ledger_and_reads_clearly():
    assert "features_total" in g._LEDGER_FACTS
    phrase = g._fact_phrase("features_total", 801)
    assert "801" in phrase and "=" not in phrase.split(":")[0]
    assert "SAMPLE" in phrase


def test_the_sampled_boolean_is_deliberately_absent():
    """_pick keeps False, so `sampled` would stamp ': False' onto every complete layer."""
    assert "sampled" not in g._LEDGER_FACTS
    assert g._pick({"sampled": False}, ("sampled",)) == {"sampled": False}, \
        "_pick keeps False, which is why the bool must stay out of _LEDGER_FACTS"


def test_every_curated_fact_still_has_a_phrase():
    assert set(g._LEDGER_FACTS) - set(g._FACT_PHRASES) == set()


# --- 3. a re-grounding pass tells the CODE peer what went ungrounded ----------------------

def _captured_code_query(monkeypatch, state):
    """Drive default_code_fn far enough to capture the prompt the peer receives."""
    seen = {}

    class _Run:
        resp = {"messages": []}
        artifacts = []

    class _Session:
        def run(self, text):
            seen["query"] = text
            return _Run()

    # The node imports these from executor_factory INSIDE the function, so that is where the
    # patch has to land — patching the graph module would miss them entirely.
    import agent_runtime.executor_factory as ef

    monkeypatch.setattr(ef, "build_agent_executor", lambda **kw: object())
    monkeypatch.setattr(ef, "open_peer_session", lambda *a, **kw: _Session())
    monkeypatch.setattr(ef, "agent_config", lambda *a, **kw: {})

    fn = g.default_code_fn(llm=object(), code_exec=False)
    try:
        fn(state.get("query", ""), [], state)
    except Exception:
        pass                     # the tail of the node is not what this asserts on
    return seen.get("query", "")


def test_the_code_peer_is_told_what_went_ungrounded(monkeypatch):
    """_reground_note reached search and analyze but never code, so a code answer the auditor
    rejected was re-run blind and most likely repeated the same unsupported claim."""
    state = {"query": "compute the Gini for those tracts", "thread_id": "t1",
             "grounding_gaps": ["the Gini coefficient was 0.41"]}
    q = _captured_code_query(monkeypatch, state)
    assert "were NOT present in any tool result" in q
    assert "the Gini coefficient was 0.41" in q


def test_a_normal_turn_carries_no_regrounding_directive(monkeypatch):
    state = {"query": "compute the Gini for those tracts", "thread_id": "t1",
             "grounding_gaps": []}
    q = _captured_code_query(monkeypatch, state)
    assert "were NOT present in any tool result" not in q
    assert q.startswith("compute the Gini for those tracts")
