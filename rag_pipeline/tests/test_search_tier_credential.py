"""The agent search client takes the SEARCH tier's credential, and no search client reads the
bare cluster names.

``agents._os_client`` resolved ``OPENSEARCH_NODE`` through the tier rule
(``rag_pipeline.search.utils.getenv``: ``<NAME>_<SEARCH_TIER>`` first, then ``<NAME>``) and
``OPENSEARCH_USERNAME`` / ``OPENSEARCH_PASSWORD`` bare, while ``keyword.py``, ``semantic.py`` and
``spatial.py`` beside it resolved all three the same way. It did not fail in the deployment
because the bare pair IS prod's: under ``SEARCH_TIER=prod`` the pair authenticates against the
resolved node. Under ``SEARCH_TIER=dev`` the same client sent prod's password to dev's cluster,
the pair that measured 401 there on 2026-09-22, and a refused query reads as a network problem.
Stage S12.8 in docs/agent-architecture-changes.md.

Credential VALUES never appear in an assertion: a host or a pair is reported by which NAME it
came from (``dev``, ``prod``, ``bare``), never by what it was.
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

_NAMES = ("PLATFORM_TIER", "SEARCH_TIER", "OPENSEARCH_NODE", "OPENSEARCH_USERNAME",
          "OPENSEARCH_PASSWORD")
_TIERED = tuple(f"{n}_{t}" for n in _NAMES[2:] for t in ("DEV", "PROD"))

# One distinct host and pair per name, so an assertion can say WHICH one a client was given.
_HOSTS = {"dev": "https://dev:9200", "prod": "https://prod:9200", "bare": "https://bare:9200"}
_PAIRS = {"dev": ("dev-user", "dev-pw"), "prod": ("prod-user", "prod-pw"),
          "bare": ("bare-user", "bare-pw")}


def _triple(which: str, suffix: str = "") -> dict:
    user, pwd = _PAIRS[which]
    return {f"OPENSEARCH_NODE{suffix}": _HOSTS[which], f"OPENSEARCH_USERNAME{suffix}": user,
            f"OPENSEARCH_PASSWORD{suffix}": pwd}


# label -> (environment, (host the client got, pair the client got))
CONFIGS = {
    "bare names only": (
        {**_triple("bare")},
        ("bare", "bare")),
    # The defect's case: the node was already tiered, the pair was not, and the bare pair is
    # prod's. Before the fix this client carried ("dev", "bare").
    "search on dev beside the bare triple": (
        {"SEARCH_TIER": "dev", **_triple("dev", "_DEV"), **_triple("bare")},
        ("dev", "dev")),
    # SEARCH_TIER picks the pair, not PLATFORM_TIER: searching dev's knowledge base from the
    # prod platform is an ordinary thing to want (S11.1), and the credential must follow the
    # host it is sent to.
    "search on dev while the platform is on prod": (
        {"PLATFORM_TIER": "prod", "SEARCH_TIER": "dev", **_triple("dev", "_DEV"),
         **_triple("prod", "_PROD"), **_triple("bare")},
        ("dev", "dev")),
    "search on prod beside the bare triple": (
        {"SEARCH_TIER": "prod", **_triple("prod", "_PROD"), **_triple("bare")},
        ("prod", "prod")),
}


def _which(host, auth):
    """Name the host and the pair a client was built with. Never the values."""
    hosts = {v: k for k, v in _HOSTS.items()}
    pairs = {v: k for k, v in _PAIRS.items()}
    return hosts.get(host, "other"), (pairs.get(tuple(auth), "other") if auth else "none")


def _configure(monkeypatch, env):
    for name in (*_NAMES, *_TIERED):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)


@pytest.fixture()
def built(monkeypatch):
    """What ``agents._os_client`` would connect with. Patched where agents.py USES the name: it
    from-imports ``OpenSearch`` at import time (AGENTS.md: patch where a symbol is used, not
    where it is defined). The one-slot cache is cleared on both sides so no case sees another's
    client."""
    from rag_pipeline.search import agents

    seen = []

    class Recorder:
        def __init__(self, hosts=None, http_auth=None, **kwargs):
            seen.append(_which((hosts or [None])[0], http_auth))

    monkeypatch.setattr(agents, "OpenSearch", Recorder)
    agents._os_client.cache_clear()
    yield seen
    agents._os_client.cache_clear()


@pytest.mark.parametrize("label", sorted(CONFIGS))
def test_the_agent_client_takes_the_search_tiers_credential(label, monkeypatch, built):
    from rag_pipeline.search import agents

    env, expected = CONFIGS[label]
    _configure(monkeypatch, env)
    agents._os_client()
    assert built == [expected], built


def test_the_tiered_pair_is_what_reaches_the_wire(monkeypatch):
    """The defect's case through the real client: the ``Authorization`` header it would send
    decodes to the DEV pair. Guards the recorder above against a client that stops passing the
    pair as ``http_auth``. Building a client opens no connection."""
    from rag_pipeline.search import agents

    env, expected = CONFIGS["search on dev beside the bare triple"]
    _configure(monkeypatch, env)
    agents._os_client.cache_clear()
    try:
        conn = agents._os_client().transport.connection_pool.connections[0]
    finally:
        agents._os_client.cache_clear()
    headers = {k.lower(): v for k, v in conn.headers.items()}
    scheme, _, b64 = headers.get("authorization", "").partition(" ")
    sent = tuple(base64.b64decode(b64).decode().split(":", 1)) if scheme == "Basic" else None
    got = _which(conn.host, sent)   # names only, so a failure report cannot print the pair
    assert got == expected, got


def test_the_cache_sits_on_the_client_and_not_on_the_index_name():
    """``@lru_cache(maxsize=1)`` belongs on ``_os_client``. On the sibling branch an anchored edit
    here moved the decorator onto a settings helper, caching a setting: order-dependent, and
    invisible to a targeted run; the full suite caught it. ``_os_index`` is the settings helper
    beside it. Pinned so the next edit near it fails by name."""
    from rag_pipeline.search import agents

    assert hasattr(agents._os_client, "cache_clear")
    assert not hasattr(agents._os_index, "cache_clear")


# ------------------------------------------------------------------ the drift guard

_BARE_READ = re.compile(
    r"""os\.(?:getenv|environ\.get)\(\s*["']OPENSEARCH_(?:NODE|USERNAME|PASSWORD)["']"""
    r"""|os\.environ\[\s*["']OPENSEARCH_(?:NODE|USERNAME|PASSWORD)["']""")

# Known today, and deliberately not moved here. Both are moved onto
# ``platform_endpoints.search_cluster()`` by ``claude/extraction-integration``, whose credential
# rule (the pair follows the host; no bare fallback beside a tiered node) is not ``tiered_env``'s
# per-variable fallback. Pointing them at ``tiered_env`` now would be a second answer to the same
# question for that branch to unpick. Each entry expires itself: the moment the file stops
# reading bare names, the second assertion below names it for deletion.
_STILL_BARE = {
    "rag_pipeline/search/agent_kb.py",            # _os_client, and the "is a cluster set?" guard
    "extractors/emitters/opensearch_emitter.py",  # _os_client
}


def test_no_search_client_reads_the_bare_cluster_names():
    """The drift guard: the next client written against the raw names fails here, by name."""
    offenders = set()
    for root in (REPO / "rag_pipeline" / "search", REPO / "extractors" / "emitters"):
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            if _BARE_READ.search(path.read_text(encoding="utf-8")):
                offenders.add(path.relative_to(REPO).as_posix())
    new = sorted(offenders - _STILL_BARE)
    assert not new, ("read OPENSEARCH_NODE / USERNAME / PASSWORD bare; resolve them through the "
                     f"search tier (rag_pipeline.search.utils.getenv): {new}")
    expired = sorted(_STILL_BARE - offenders)
    assert not expired, f"no longer read the bare names; delete from _STILL_BARE: {expired}"
