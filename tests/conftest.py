"""No ``.env`` reaches this suite unless ``RUN_LIVE_BACKEND_TESTS=1`` asks for one.

The same guard as ``rag_pipeline/tests/conftest.py``, for the same reason. Four modules call a
bare ``load_dotenv()`` (``memory_module``, ``search/semantic``, ``search/spatial``,
``api/server``), and python-dotenv walks upwards from the *calling module's* directory until it
finds a ``.env``. From a worktree under ``.claude/worktrees/<name>/`` that walk leaves the tree
and lands on the main checkout's file: the developer's own credentials, aimed at whatever was
last deployed.

Measured on 2026-10-02 from such a worktree, with every outbound connection blocked so that
nothing was actually reached: the seven skip-guarded tests in ``tests/live/`` ran, aimed at the
production OpenSearch node (that file's ``OPENSEARCH_NODE``), and four model-catalogue requests
in ``test_claude_peer.py`` were addressed to AnvilGPT with its real ``ANVILGPT_KEY``. The same
command, unblocked, had taken 6m41s; from a worktree with no ``.env`` above it, 14 s.

So the loader goes, before any test module imports it. ``dotenv_values`` goes too:
``search/opengeodata.py`` hydrates API credentials from ``rag_pipeline/.env`` with it at import,
and the non-live tests import that module. That file is untracked and exists in the main
checkout only, so leaving it would make the main checkout the one place the suite runs with
real credentials.
"""
from __future__ import annotations

import os

import dotenv

# The same opt-in as rag_pipeline/tests/conftest.py, so one variable decides for both suites.
_LIVE = str(os.getenv("RUN_LIVE_BACKEND_TESTS") or "").strip().lower() in {"1", "true", "yes", "on"}

if not _LIVE:
    dotenv.load_dotenv = lambda *a, **k: False   # noqa: E731 - a deliberate no-op
    dotenv.find_dotenv = lambda *a, **k: ""      # noqa: E731
    dotenv.dotenv_values = lambda *a, **k: {}    # noqa: E731
