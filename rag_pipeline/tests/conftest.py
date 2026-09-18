"""Pin the deployment-shaped environment before anything imports it.

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
"""
from __future__ import annotations

import os

import dotenv

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
