"""Pin the deployment-shaped environment before anything imports it.

Four modules call a bare ``load_dotenv()`` (``memory_module``, ``search/semantic``,
``search/spatial``, ``api/server``). A bare call does not read "the repo's .env" — it walks
*upwards* from the calling module's directory until it finds one. From a git worktree under
``.claude/worktrees/<name>/`` that walk leaves the tree entirely and lands on the main
checkout's ``.env``: the developer's own file, whose contents track whatever was last deployed.

The failure that produced this file: those lines had grown to include
``AGENT_TOKEN_VERIFY=introspect``, ``AGENT_MODE=token`` and ``PLATFORM_TIER=dev`` while the
deployment was being switched over. Every test then ran with introspection ON, so identity
checks made real HTTPS calls to ``backend-dev.i-guide.io``, which answers 403 to an
unauthenticated caller. Thirty tests failed on assertions about their own locally-minted
tokens, and the suite went from two and a half minutes to **twenty-five**, all of it network.
Nothing in the repository had changed. The same checkout passed or failed depending on a file
outside it.

Two things happen here, at conftest import — before the test modules are imported and
therefore before any of those ``load_dotenv()`` calls run. The loader is replaced with a no-op,
so no ``.env`` is read from then on (the one ``rag_pipeline/__init__.py`` has already read is
taken back, below); and the deployment-shaped variables are given test values, so a process
that already inherited them from a shell does not carry them in either. A test that
wants different values still uses ``monkeypatch.setenv`` as usual.
"""
from __future__ import annotations

import os
from pathlib import Path

import dotenv
import pytest

# Neutralise the loader itself, not just its output.
#
# Pinning the variables was the first attempt and it is not enough. `test_demo_mode` calls
# `importlib.reload(api.server)`, which re-runs that module's `load_dotenv()` — and a test that
# had just `monkeypatch.delenv("AGENT_MODE")` leaves the name genuinely absent, so
# `override=False` no longer protects it and dotenv refills it with `AGENT_MODE=token` from the
# developer's file. Four demo-mode tests failed that way: they set `DEMO_MODE=true`, reloaded,
# and got `token` mode back.
#
# Any pinned value is defeated by delete-then-reload, so the loader goes instead. Tests declare
# their own environment; nothing outside the repository gets a say in what they assert.
#
# `RUN_LIVE_BACKEND_TESTS=1` opts back in. Three tests here are written to self-skip when the
# real services are unconfigured (`test_state_uniformity`, `test_spatial_routing_e2e`), and
# before this file they were running against OpenSearch and AnvilGPT for real — not because
# anyone chose that, but because the stray `.env` happened to configure them. That is worth
# keeping as a CHOICE: offline and deterministic by default, live when asked for, the same
# shape as `test_opengeodata_search`'s existing `RUN_REAL_OPEN_GEODATA_TEST=1`.
_LIVE = str(os.getenv("RUN_LIVE_BACKEND_TESTS") or "").strip().lower() in {"1", "true", "yes", "on"}

# One `.env` has been read before this line can run. `rag_pipeline/__init__.py` loads the repo
# root's `.env` by explicit path, and pytest imports that package in order to import this file,
# which lives inside it, so nothing done to the loader here comes early enough. A worktree has
# no file at that path; the main checkout does. Measured 2026-10-02 on a copy of the tree whose
# `.env` files had the main checkout's key names and sentinel values: 56 of those variables
# reached the tests, the three live tests above stopped skipping, six unit tests failed on
# assertions about values they had set themselves, and the run made 272 more network attempts
# than it does from a worktree, 206 of them to hosts the file names. So the load is taken back.
# A variable still holding the file's value came from the file, or from a shell that sourced
# it, and either way it is not a test's to rely on.
import rag_pipeline  # noqa: E402 - imported already, which is the problem described above

_REPO_DOTENV = Path(rag_pipeline.__file__).resolve().parent.parent / ".env"
if not _LIVE and _REPO_DOTENV.is_file():
    for _key, _value in dotenv.dotenv_values(_REPO_DOTENV).items():
        if _value is not None and os.environ.get(_key) == _value:
            del os.environ[_key]

# `dotenv_values` goes too: `search/opengeodata.py` and `search/opengeodata_new.py` copy the LLM
# endpoint and key variables out of `rag_pipeline/.env` with it at import. That file is untracked
# and only the main checkout has it, so leaving it would refill the keys taken back above.
if not _LIVE:
    dotenv.load_dotenv = lambda *a, **k: False   # noqa: E731 - a deliberate no-op
    dotenv.find_dotenv = lambda *a, **k: ""      # noqa: E731
    dotenv.dotenv_values = lambda *a, **k: {}    # noqa: E731

# Nor does any test reach a live service, unless it stubs one in on purpose. The code under test
# calls these three on its own, deep inside paths the tests drive for other reasons, and each
# degrades quietly when it fails: so every test passed with the network blocked, and nothing
# ever failed while they were reaching the real thing. Measured 2026-10-02 from a worktree,
# blocking and logging every attempt so that nothing was reached:
#
#   * The supervisor's web fallback runs a real DuckDuckGo search whenever a stubbed search
#     leaves it no platform evidence, then fetches the top hit: 36 requests from 4 tests, to the
#     8 engines ddgs's "auto" backend fans out to (Bing, Google, Brave, Yahoo, Yandex, Mojeek,
#     Mullvad Leta, Wikipedia). ddgs talks through primp, a Rust client, so a probe on Python's
#     `socket` module does not see those requests at all; only the page fetch after them shows.
#   * `_known_embedding_models` asks RS_EMBED_URL/api/models (localhost:8077 by default) for the
#     catalogue behind `_correct_artifact_claims`, on every synthesized answer, and caches only
#     a success: 268 connection attempts from 63 tests in 8 files.
#   * A peer built with MCP tools first asks the remote MCP server at 127.0.0.1:8000 for its tool
#     list, and keeps the answer for 60 s, so which test pays for it depends on timing. On the
#     machine this was measured on, that port was a local rs-embed webapp.
#
# Each is replaced with what an unreachable service produces, at the seam the module's own tests
# use: an offline provider in `web._PROVIDERS` (test_web_search.py), an empty catalogue in
# `_EMBED_MODELS_CACHE` (test_artifact_claims.py), and a failed remote listing, which
# `_make_remote_mcp_tools` turns into its local-import fallback. A test that wants one of them
# answering patches over it as usual. `RUN_LIVE_BACKEND_TESTS=1` leaves this in place: the live
# tests above use none of the three. The imports are inside the fixture so that the lines above
# and below still run before anything imports the code under test.
@pytest.fixture(autouse=True)
def _no_live_services(monkeypatch):
    from agent_runtime import langchain_mcp_tools
    from agent_runtime.supervisor import graph
    from rag_pipeline.search import web

    def _no_search_engine(query, **_kwargs):
        raise ConnectionError("rag_pipeline/tests reaches no search engine")

    async def _no_mcp_server(url):
        raise ConnectionError(f"rag_pipeline/tests reaches no MCP server ({url})")

    monkeypatch.setitem(web._PROVIDERS, "duckduckgo", _no_search_engine)
    monkeypatch.setattr(graph, "_EMBED_MODELS_CACHE", frozenset())
    monkeypatch.setattr(langchain_mcp_tools, "_remote_mcp_list_tools_async", _no_mcp_server)


# Identity and platform wiring. `local` verification in particular is what keeps the suite
# offline: `introspect` turns every identity check into a network round trip to whichever
# platform tier the developer last pointed at.
_TEST_ENV = {
    "AGENT_TOKEN_VERIFY": "local",
    "JWT_ACCESS_TOKEN_NAME": "jwt-access-token-dev",
}

# Variables with no safe default: a test that needs one sets it. Cleared rather than pinned,
# because the meaningful default for "what mode is this deployment in" is *unset*.
_CLEARED = (
    "AGENT_MODE",
    "DEMO_MODE",
    "AGENT_TOKEN_STRICT",
    "AGENT_MIN_ROLE",
    "AGENT_CHAT_API_KEY",
    "PLATFORM_TIER",
    "PLATFORM_SIGNIN_URL",
    "PLATFORM_REFRESH_URL",
    "PLATFORM_CHECK_TOKENS_URL",
)

for _name in ([] if _LIVE else _CLEARED):
    os.environ.pop(_name, None)
    # Pinned to empty rather than left absent: `load_dotenv(override=False)` fills anything
    # unset, so popping alone would let the outer file put it straight back. Every reader in
    # this repo treats "" as unset (`os.getenv(...) or default`), which is why this is safe.
    os.environ[_name] = ""

if not _LIVE:
    os.environ.update(_TEST_ENV)
