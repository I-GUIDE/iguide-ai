"""The library the agent lists is the library the sandbox can import (M8.63).

Two halves disagreed about where the library is. The reader honoured AGENT_METHOD_LIBRARY_DIR and
both writers did not, so pointing the library at a host-visible path, which Docker-out-of-Docker
requires, built it where nothing read it. And when the sandbox could not import it, the traceback
said only "No module named 'iguide_methods'", which three different causes produce.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
IMPORT_FAILS = ('Traceback (most recent call last):\n  File "/work/script.py", line 1, in <module>\n'
                "ModuleNotFoundError: No module named 'iguide_methods'\n")


# ------------------------------------------------------------------ one location

def test_the_writers_build_where_the_reader_reads(tmp_path, monkeypatch):
    from agent_runtime.code_execution import method_library_dir, method_library_root
    from extractors.emitters import library_emitter

    monkeypatch.setenv("AGENT_METHOD_LIBRARY_DIR", str(tmp_path / "lib"))
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "storage"))
    assert library_emitter.library_root() == method_library_root() == tmp_path / "lib"
    library_emitter.package_dir()
    assert method_library_dir() == tmp_path / "lib"
    assert not (tmp_path / "storage" / "method_library").exists(), "built where nothing reads"


def test_unset_both_default_to_the_storage_root(tmp_path, monkeypatch):
    from agent_runtime.code_execution import method_library_root
    from extractors.emitters import library_emitter

    monkeypatch.delenv("AGENT_METHOD_LIBRARY_DIR", raising=False)
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    assert library_emitter.library_root() == method_library_root() == tmp_path / "method_library"


def test_the_build_script_asks_the_same_resolver():
    """It hardcoded storage_root()/method_library, a second copy of the default."""
    src = (REPO / "scripts" / "build_method_library.py").read_text(encoding="utf-8")
    assert 'Path(storage_root()) / "method_library"' not in src
    assert "library_emitter.library_root()" in src


# ------------------------------------------------------------------ why it would not import

def _executor(stderr: str):
    from agent_runtime import code_execution as ce

    class ImportFails(ce.LocalSubprocessExecutor):
        def _run(self, work, timeout, dependencies=None, deps_cache=None, entrypoint=None):
            return 1, "", stderr, False, None

    return ImportFails()


@pytest.fixture()
def built(tmp_path, monkeypatch):
    from agent_runtime import code_execution as ce

    (tmp_path / "iguide_methods").mkdir()
    monkeypatch.setenv("AGENT_METHOD_LIBRARY_DIR", str(tmp_path))
    monkeypatch.setattr(ce, "_library_warned", set())
    return tmp_path


def test_a_library_built_but_invisible_to_the_sandbox_says_so(built):
    stderr = _executor(IMPORT_FAILS).execute("from iguide_methods import x").stderr
    # First, because _clip keeps the head and this is the explanation, not a footnote.
    assert stderr.startswith("[the method library is built"), stderr[:200]
    assert "write the function inline" in stderr
    assert "ModuleNotFoundError" in stderr, "the traceback itself is kept"


def test_the_operator_is_told_the_host_path_rule_once(built, caplog):
    from agent_runtime.code_execution import _diagnose_library_import

    with caplog.at_level(logging.WARNING, logger="agent_runtime.code_execution"):
        _diagnose_library_import(IMPORT_FAILS, "docker")
        _diagnose_library_import(IMPORT_FAILS, "docker")
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, warnings
    assert "same absolute path" in warnings[0] and str(built) in warnings[0]


def test_no_library_built_says_so():
    # conftest points AGENT_METHOD_LIBRARY_DIR at an empty directory
    stderr = _executor(IMPORT_FAILS).execute("from iguide_methods import x").stderr
    assert stderr.startswith("[no method library has been built"), stderr[:200]


def test_with_the_bundle_off_it_names_no_library(monkeypatch):
    monkeypatch.delenv("AGENT_EXTRACTION", raising=False)
    stderr = _executor(IMPORT_FAILS).execute("from iguide_methods import x").stderr
    assert stderr.startswith("[`iguide_methods` does not exist in this deployment"), stderr[:200]


def test_a_wrong_element_module_is_not_called_a_missing_library(built):
    stderr = _executor("ModuleNotFoundError: No module named 'iguide_methods.ke_nope'\n").execute(
        "from iguide_methods.ke_nope import x").stderr
    assert not stderr.startswith("["), stderr[:200]
