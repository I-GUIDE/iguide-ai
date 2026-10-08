"""The agent KB reads the cluster the platform's search reads, and staging asks the backend of the
knowledge base it staged from (S4).

Two halves of the extraction read raw environment names while everything beside them followed the
tier. ``agent_kb._os_client`` and the extraction's own emitter took bare ``OPENSEARCH_NODE`` /
``_USERNAME`` / ``_PASSWORD``, while keyword and semantic search resolve ``<NAME>_<SEARCH_TIER>``
first (``rag_pipeline.search.utils.getenv``). The developer ``.env`` this checkout runs against
defines all three forms, so under ``SEARCH_TIER`` the agent KB could address a different cluster
from the search its results are joined to. A refused query there reads as "no results". And
staging resolved element ids against prod's backend whatever tier the ids came from.

Latent today, not live: the deployed agent KB backend is ``local`` (AGENT_KB_BACKEND unset). It
becomes live the moment someone turns the indexed corpus on.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_NAMES = ("PLATFORM_TIER", "SEARCH_TIER", "OPENSEARCH_NODE", "OPENSEARCH_USERNAME",
          "OPENSEARCH_PASSWORD")
_TIERED = tuple(f"{n}_{t}" for n in _NAMES[2:] for t in ("DEV", "PROD"))

CONFIGS = {
    "bare names only": {"OPENSEARCH_NODE": "https://bare:9200", "OPENSEARCH_USERNAME": "bu",
                        "OPENSEARCH_PASSWORD": "bp"},
    "prod pair beside a bare node": {
        "SEARCH_TIER": "prod", "OPENSEARCH_NODE": "https://bare:9200",
        "OPENSEARCH_NODE_PROD": "https://prod:9200", "OPENSEARCH_USERNAME_PROD": "pu",
        "OPENSEARCH_PASSWORD_PROD": "pp", "OPENSEARCH_USERNAME": "bu", "OPENSEARCH_PASSWORD": "bp"},
    "search on dev, platform on prod": {
        "PLATFORM_TIER": "prod", "SEARCH_TIER": "dev", "OPENSEARCH_NODE": "https://bare:9200",
        "OPENSEARCH_NODE_DEV": "https://dev:9200", "OPENSEARCH_USERNAME_DEV": "du",
        "OPENSEARCH_PASSWORD_DEV": "dp"},
    # The deployed shape (platform_endpoints' own notes): a bare node naming prod's cluster, the
    # _PROD pair, and a prod table entry that names no cluster to conflict with.
    "bare node with the prod pair": {
        "PLATFORM_TIER": "prod", "SEARCH_TIER": "prod", "OPENSEARCH_NODE": "https://bare:9200",
        "OPENSEARCH_USERNAME_PROD": "pu", "OPENSEARCH_PASSWORD_PROD": "pp",
        "OPENSEARCH_USERNAME": "bu", "OPENSEARCH_PASSWORD": "bp"},
}


@pytest.fixture()
def clients(monkeypatch):
    """Record what each client would connect to. Patched where each module USES the name:
    keyword.py binds `OpenSearch` by from-import at import time, the other two import it at call
    time (CLAUDE.md: patch where a symbol is used, not where it is defined)."""
    import opensearchpy

    from extractors.emitters import opensearch_emitter
    from rag_pipeline.search import agent_kb, keyword

    seen = []

    class Recorder:
        def __init__(self, hosts=None, http_auth=None, **kwargs):
            seen.append(((hosts or [None])[0], http_auth))

    monkeypatch.setattr(opensearchpy, "OpenSearch", Recorder)
    monkeypatch.setattr(keyword, "OpenSearch", Recorder)
    cached = (keyword._os_client, agent_kb._os_client, opensearch_emitter._os_client)
    for fn in cached:
        fn.cache_clear()
    yield seen
    for fn in cached:
        fn.cache_clear()


def _configure(monkeypatch, env):
    for name in (*_NAMES, *_TIERED):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)


@pytest.mark.parametrize("label", sorted(CONFIGS))
def test_the_agent_kb_reads_the_cluster_platform_search_reads(label, monkeypatch, clients):
    from rag_pipeline.search import agent_kb, keyword

    _configure(monkeypatch, CONFIGS[label])
    keyword._os_client()
    agent_kb._os_client()
    platform, agent = clients
    assert agent == platform, label


@pytest.mark.parametrize("label", sorted(CONFIGS))
def test_an_ingest_lands_where_the_agent_kb_reads(label, monkeypatch, clients):
    from extractors.emitters import opensearch_emitter
    from rag_pipeline.search import agent_kb

    _configure(monkeypatch, CONFIGS[label])
    agent_kb._os_client()
    opensearch_emitter._os_client()
    reader, writer = clients
    assert writer == reader, label


def test_a_tiered_node_never_gets_the_untiered_credential(monkeypatch, clients):
    """Credential follows host. The untiered pair measured 401 against dev's cluster, so with no
    tiered pair the agent KB sends NONE rather than the wrong one. (prototype's keyword.py falls
    back per variable and does send it; that is the maintainer's to change, not asserted here.)"""
    from rag_pipeline.search import agent_kb

    _configure(monkeypatch, {"SEARCH_TIER": "dev", "OPENSEARCH_NODE_DEV": "https://dev:9200",
                             "OPENSEARCH_USERNAME": "bu", "OPENSEARCH_PASSWORD": "bp"})
    agent_kb._os_client()
    assert clients == [("https://dev:9200", None)], clients


def test_a_tiered_node_alone_counts_as_a_configured_cluster(monkeypatch):
    """The 'is a cluster configured?' checks read the bare name too, so a deployment that named
    its node only in the tiered form was told no cluster existed."""
    from rag_pipeline.search.agent_kb import _cluster

    _configure(monkeypatch, {"SEARCH_TIER": "prod", "OPENSEARCH_NODE_PROD": "https://prod:9200"})
    assert _cluster()[0] == "https://prod:9200"


def test_the_emitter_guards_the_platform_index_under_its_tiered_name(monkeypatch):
    """The delete guard compared agent indices with the BARE index name only, so on a tiered
    deployment it protected an index nothing was using and not the one search reads."""
    from extractors.emitters import opensearch_emitter
    from extractors.indices import all_agent_indices

    victim = all_agent_indices()[0]
    _configure(monkeypatch, {"SEARCH_TIER": "prod"})
    monkeypatch.setenv("OPENSEARCH_INDEX", "something-else")
    monkeypatch.setenv("OPENSEARCH_INDEX_PROD", victim)
    with pytest.raises(RuntimeError, match="collides"):
        opensearch_emitter._assert_agent_indices([victim])


def test_no_search_client_reads_the_bare_cluster_names():
    """The drift guard: the next client written against raw names fails here, by name."""
    import re

    offenders = []
    for path in [*(REPO / "rag_pipeline" / "search").glob("*.py"),
                 *(REPO / "extractors" / "emitters").glob("*.py")]:
        text = path.read_text(encoding="utf-8")
        if re.search(r'os\.getenv\(\s*"OPENSEARCH_(NODE|USERNAME|PASSWORD)"', text):
            offenders.append(path.relative_to(REPO).as_posix())
    assert not offenders, offenders


# ------------------------------------------------------------------ staging's backend

@pytest.fixture()
def backend_env(monkeypatch):
    for name in ("SEARCH_TIER", "PLATFORM_TIER", "IGUIDE_BACKEND_URL", "IGUIDE_BACKEND_URL_DEV",
                 "IGUIDE_BACKEND_URL_PROD"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_staging_asks_the_search_tiers_backend(backend_env):
    from agent_runtime.platform_endpoints import backend_url

    assert backend_url() == "https://backend.i-guide.io", "no tier: what staging always used"
    backend_env.setenv("SEARCH_TIER", "dev")
    assert backend_url() == "https://backend-dev.i-guide.io"
    backend_env.setenv("PLATFORM_TIER", "prod")
    assert backend_url() == "https://backend-dev.i-guide.io", "the SEARCH tier decides"
    backend_env.setenv("IGUIDE_BACKEND_URL_DEV", "https://staging.example/")
    assert backend_url() == "https://staging.example", "an explicit tiered URL still wins"


def test_a_bad_search_tier_raises_rather_than_staging_from_prod(backend_env):
    from agent_runtime.platform_endpoints import backend_url

    backend_env.setenv("SEARCH_TIER", "staging")
    with pytest.raises(ValueError):
        backend_url()


def test_an_element_is_fetched_from_that_backend(backend_env):
    import requests

    from agent_runtime import staging

    backend_env.setenv("SEARCH_TIER", "dev")
    asked = []

    class Response:
        status_code = 404

        def json(self):
            return {}

    def fake_get(url, **kwargs):
        asked.append(url)
        return Response()

    backend_env.setattr(requests, "get", fake_get)
    staging._element_metadata("b1fa548b-0000-4000-8000-000000000000")
    assert asked and asked[0].startswith("https://backend-dev.i-guide.io/api/elements/"), asked
