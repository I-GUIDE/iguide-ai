"""Shared fixtures. Chiefly: keep the suite hermetic against developer-local state.

The method library is generated into ``storage_root()/method_library`` by
``scripts/build_method_library.py``. Once a developer runs that against the real corpus, 203
units appear on disk — and any test exercising ``_direct_search_sweep`` starts seeing them,
because the sweep now unions the library deterministically.

That is a genuine reproducibility hazard, not a nuisance: two pre-existing tests
(``test_sweep_adds_implied_methods``, ``test_search_fn_unions_sweep_with_llm_harvest``) faked
every other retrieval arm and passed for months, then failed the moment the corpus was built —
same code, same commit, different machine state. Pointing the library at an empty directory by
default makes the suite depend only on the repo.

A test that WANTS a library opts in explicitly, by monkeypatching
``agent_runtime.method_library.load_registry`` (see ``test_method_library_tools.py``) or by
setting ``AGENT_METHOD_LIBRARY_DIR`` itself.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_method_library(tmp_path_factory, monkeypatch):
    empty = tmp_path_factory.mktemp("empty_method_library")
    monkeypatch.setenv("AGENT_METHOD_LIBRARY_DIR", str(empty))
    yield
