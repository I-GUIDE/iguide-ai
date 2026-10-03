"""Everything in this directory talks to a live service, so none of it runs unless asked.

These files are scripts first (``python tests/live/test_keyword.py``, see ``README.md``), and
pytest collects them only because of their names. Seven of their thirty-two tests carried a skip
keyed on ``OPENSEARCH_NODE``, which the stray ``.env`` described in ``../conftest.py`` satisfied.
The other twenty-five had no guard and ran in every checkout. Without credentials they mostly
print instead of asserting, so they passed, and ``test_opengeodata_live.py`` called NASA CMR,
data.gov, Socrata and DataCite on every ``python3 -m pytest tests/``.

``RUN_LIVE_BACKEND_TESTS=1`` runs them, and also restores ``.env`` loading so they can find
their credentials. From a worktree, the file found that way is the main checkout's. Each test
also gets the ``integration`` marker declared in ``pytest.ini``, so ``-m "not integration"``
deselects them.
"""
from __future__ import annotations

import os

import pytest

_LIVE = str(os.getenv("RUN_LIVE_BACKEND_TESTS") or "").strip().lower() in {"1", "true", "yes", "on"}

# A skipif rather than a skip, and placed first: pytest evaluates every skipif before any skip,
# so a plain skip would report the files' own "OPENSEARCH_NODE unset (no repo-root .env)"
# instead, which is untrue in the main checkout, where the .env exists and is simply not loaded.
_NOT_ASKED = pytest.mark.skipif(True, reason="live service test: set RUN_LIVE_BACKEND_TESTS=1 to run it")


def pytest_itemcollected(item):
    # Only items under this directory reach this hook: pytest calls it through the item's own
    # conftest chain, so no path check is needed.
    item.add_marker(pytest.mark.integration)
    if not _LIVE:
        item.add_marker(_NOT_ASKED, append=False)
