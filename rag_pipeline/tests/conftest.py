"""Shared fixtures. Chiefly: keep the suite hermetic against developer-local state.

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

import pytest


@pytest.fixture(autouse=True)
def _isolate_generated_state(tmp_path_factory, monkeypatch):
    """Point the method library and the local agent KB at empty per-test directories."""
    monkeypatch.setenv("AGENT_METHOD_LIBRARY_DIR",
                       str(tmp_path_factory.mktemp("empty_method_library")))
    monkeypatch.setenv("AGENT_KB_STORE_DIR", str(tmp_path_factory.mktemp("empty_agent_kb")))
    yield
