"""Which store the agent KB actually read, and whether anyone can tell.

The backend defaults to the LOCAL file-backed store so tests and offline runs never touch the
cluster. That default is right for them and wrong for a server: a deployment that simply does not
set ``AGENT_KB_BACKEND`` reads a token-overlap file store instead of the indexed corpus, and the
symptom is "fewer results" — which reads as a retrieval-quality problem, not a configuration one.

Observed live: a prototype turn logged ``agent_kb_search -> no results`` while the same query
against the cluster returned 8. Nothing anywhere reported which store had been consulted.

The default is deliberately NOT changed here — changing it would make an exported OPENSEARCH_NODE
pull the test suite onto the network. What changes is that the degradation stops being silent.
"""

from __future__ import annotations

from rag_pipeline.search import agent_kb


def test_the_payload_names_the_backend_it_used(monkeypatch):
    monkeypatch.delenv("AGENT_KB_BACKEND", raising=False)
    monkeypatch.delenv("OPENSEARCH_NODE", raising=False)
    assert agent_kb.agent_kb_search("anything")["backend"] == "local"


def test_reading_the_local_store_while_a_cluster_is_configured_says_so(monkeypatch):
    """The case that cost an hour: a cluster IS reachable, the KB is not reading it, and the
    only visible symptom is a short result list."""
    monkeypatch.delenv("AGENT_KB_BACKEND", raising=False)
    monkeypatch.setenv("OPENSEARCH_NODE", "https://cluster.example:9200")
    out = agent_kb.agent_kb_search("anything")
    assert out["backend"] == "local"
    assert "note" in out
    assert "NOT the cluster" in out["note"] and "AGENT_KB_BACKEND=opensearch" in out["note"]


def test_no_note_when_there_is_no_cluster_to_miss(monkeypatch):
    """Offline is not a misconfiguration. Warning about it would be noise on every test run."""
    monkeypatch.delenv("AGENT_KB_BACKEND", raising=False)
    monkeypatch.delenv("OPENSEARCH_NODE", raising=False)
    assert agent_kb.agent_kb_search("anything").get("note") is None


def test_the_warning_is_emitted_once_per_process(monkeypatch, caplog):
    """A per-call warning on every tool invocation would be scrolled past. Same reasoning as the
    resolved-embedding-URL log."""
    import logging

    monkeypatch.setattr(agent_kb, "_LOCAL_BACKEND_WARNED", False)
    monkeypatch.delenv("AGENT_KB_BACKEND", raising=False)
    monkeypatch.setenv("OPENSEARCH_NODE", "https://cluster.example:9200")
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            agent_kb.agent_kb_search("anything")
    assert sum(1 for r in caplog.records if "LOCAL file-backed store" in r.message) == 1


def test_the_dev_launcher_resolves_the_backend_rather_than_leaving_it_to_chance():
    """`run_agent_api_dev.sh` describes itself as the one supported way to start the server
    locally, so the env contract belongs in it."""
    from pathlib import Path

    src = Path("scripts/run_agent_api_dev.sh").read_text(encoding="utf-8")
    assert "AGENT_KB_BACKEND" in src and "OPENSEARCH_NODE" in src
    assert "AGENT_METHOD_LIBRARY_DIR" in src, (
        "kb_method_search and the sandbox mount both silently report an empty library without it")
    assert "agent KB backend:" in src, "the resolved value must be printed, not assumed"
