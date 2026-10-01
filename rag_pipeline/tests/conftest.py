"""Shared test setup. Two jobs, both about keeping the suite hermetic.

1. Pin the deployment-shaped environment before anything imports it (from prototype).

Four modules call a bare ``load_dotenv()`` (``memory_module``, ``search/semantic``,
``search/spatial``, ``api/server``). A bare call does not read "the repo's .env" — it walks
*upwards* from the working directory until it finds one. From a git worktree under
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
so no ``.env`` is read at all; and the deployment-shaped variables are given test values, so a
process that already inherited them from a shell does not carry them in either. A test that
wants different values still uses ``monkeypatch.setenv`` as usual.

2. Isolate generated state (from backend_swap's extraction work).

Extraction writes to two places under ``storage_root()`` — the generated method library and the
file-backed agent KB — and the agent's deterministic sweep reads BOTH. So any test that fakes the
other retrieval arms and asserts on the result set is really asserting "and nothing has been
ingested on this machine", which is not a property of the code.

This is not hypothetical. It has now happened twice, to five tests, from the same cause:

* ``test_sweep_adds_implied_methods`` and ``test_search_fn_unions_sweep_with_llm_harvest`` passed
  for months, then failed the moment a developer built the method library. That is what
  ``AGENT_METHOD_LIBRARY_DIR`` below is for.
* ``test_sweep_adds_implied_methods`` (again), ``test_direct_search_sweep_drops_unlisted`` and
  ``test_the_every_turn_sweep_still_never_touches_the_web`` failed the moment 45 documents landed
  in the local **KB store** — the half the first fix did not cover. Since indexing the corpus is
  the whole point of the extraction work, leaving it uncovered means the suite is scheduled to
  break on success.

Both halves are pointed at empty directories by default, so the suite depends only on the repo. A
test that WANTS either one opts in explicitly: monkeypatch
``agent_runtime.method_library.load_registry`` (see ``test_method_library_tools.py``), or set
``AGENT_METHOD_LIBRARY_DIR`` / ``AGENT_KB_STORE_DIR`` to a directory it populated itself (see
``test_fanout_order.py``).
"""
from __future__ import annotations

import os

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

if not _LIVE:
    dotenv.load_dotenv = lambda *a, **k: False   # noqa: E731 - a deliberate no-op
    dotenv.find_dotenv = lambda *a, **k: ""      # noqa: E731

# Identity and platform wiring. `local` verification in particular is what keeps the suite
# offline: `introspect` turns every identity check into a network round trip to whichever
# platform tier the developer last pointed at.
_TEST_ENV = {
    "AGENT_TOKEN_VERIFY": "local",
    "JWT_ACCESS_TOKEN_NAME": "jwt-access-token-dev",
    # The extraction bundle is OFF by default in every deployment (agent_runtime/extraction_flag.py)
    # and ON here, so the suite exercises it. What OFF means is pinned separately, with the flag
    # monkeypatched off, in test_extraction_flag.py.
    "AGENT_EXTRACTION": "1",
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




@pytest.fixture(autouse=True)
def _restore_warnings_warn(monkeypatch):
    """The invariant gate's prologue wraps ``warnings.warn`` to see geographic metric operations
    (sandbox_verify.install_operation_tracker). In the sandbox that lasts one process; a test
    that execs the prologue in-process must not hand the wrapper to every later test."""
    import warnings

    monkeypatch.setattr(warnings, "warn", warnings.warn)


@pytest.fixture(autouse=True)
def _isolate_generated_state(tmp_path_factory, monkeypatch):
    """Point the method library and the local agent KB at empty per-test directories."""
    monkeypatch.setenv("AGENT_METHOD_LIBRARY_DIR",
                       str(tmp_path_factory.mktemp("empty_method_library")))
    monkeypatch.setenv("AGENT_KB_STORE_DIR", str(tmp_path_factory.mktemp("empty_agent_kb")))
    yield
