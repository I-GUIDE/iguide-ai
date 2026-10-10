"""The sandbox's input-size budget applies to deployments, not to local experiments.

execute_code stages at most AGENT_CODE_EXEC_MAX_INPUT_MB of files into a run (200 by
default). On 2026-10-09 that refused a 306 MB population raster in a local-mode session, so
the analysis never ran. Unset, local mode now has no cap; an explicit value still wins
everywhere, and a deployment keeps the default.
"""

import pytest

from agent_runtime import langchain_exec_tools as lx

MB = 1024 * 1024


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("AGENT_CODE_EXEC_MAX_INPUT_MB", raising=False)
    monkeypatch.delenv("AGENT_MODE", raising=False)
    monkeypatch.delenv("DEMO_MODE", raising=False)


@pytest.mark.parametrize("mode", ["dev", "demo", "token"])
def test_a_deployment_keeps_the_default_budget(monkeypatch, mode):
    monkeypatch.setenv("AGENT_MODE", mode)
    assert lx._max_input_bytes() == lx.DEFAULT_MAX_INPUT_MB * MB


def test_local_mode_has_no_budget_unless_one_is_set(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "local")
    assert lx._max_input_bytes() is None
    monkeypatch.setenv("AGENT_CODE_EXEC_MAX_INPUT_MB", "50")
    assert lx._max_input_bytes() == 50 * MB


def test_an_explicit_budget_wins_on_a_deployment(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "token")
    monkeypatch.setenv("AGENT_CODE_EXEC_MAX_INPUT_MB", "500")
    assert lx._max_input_bytes() == 500 * MB


@pytest.mark.parametrize("raw,expected", [("inf", None), ("nan", 200 * MB),
                                          ("lots", 200 * MB), ("  ", 200 * MB)])
def test_odd_values(monkeypatch, raw, expected):
    monkeypatch.setenv("AGENT_MODE", "dev")
    monkeypatch.setenv("AGENT_CODE_EXEC_MAX_INPUT_MB", raw)
    assert lx._max_input_bytes() == expected


def test_an_unreadable_mode_keeps_the_budget(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "laptop")   # not a mode: current_mode() raises
    assert lx._max_input_bytes() == lx.DEFAULT_MAX_INPUT_MB * MB


def _staging_of(tmp_path, monkeypatch, size_mb):
    big = tmp_path / "population.tif"
    with open(big, "wb") as fh:
        fh.truncate(size_mb * MB)               # sparse: no real disk spent
    record = {"file_id": "file_big", "filename": big.name, "size_bytes": size_mb * MB}
    monkeypatch.setattr(lx, "_resolve_input_file", lambda ref: (big, record))
    return lx._build_staging(["file_big"])


def test_local_mode_stages_a_file_over_the_default_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "local")
    staging, staged, errors, skipped = _staging_of(tmp_path, monkeypatch, 306)
    assert not skipped and not errors
    assert staged and staged[0].get("file_id") == "file_big"


def test_a_deployment_still_skips_it_and_says_why(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "token")
    staging, staged, errors, skipped = _staging_of(tmp_path, monkeypatch, 306)
    assert not staged
    assert skipped[0]["reason"] == "max total input size exceeded"
    assert skipped[0]["limit_bytes"] == 200 * MB
