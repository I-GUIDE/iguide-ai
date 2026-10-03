"""agent_runtime.fork_safe: children started without forking the agent on macOS.

On the maintainer's Mac every child a long-lived server started after its first reprojection
died before exec (qgis_process -11, execute_code "failed during dependency installation"). The
reproduction is scripts/repro_macos_fork_crash.py, and the module docstring has the chain.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from agent_runtime import fork_safe

ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "scripts" / "repro_macos_fork_crash.py"

needs_posix_spawn = pytest.mark.skipif(
    os.name != "posix" or not getattr(subprocess, "_USE_POSIX_SPAWN", False),
    reason="CPython does not use posix_spawn on this platform")


@pytest.fixture
def macos_path(monkeypatch):
    """Take the macOS branch on any POSIX host: the mechanics are the same under glibc."""
    monkeypatch.setattr(fork_safe, "avoids_fork", lambda: True)


# --- the regression, in the state that killed the server's children -------------------------

@pytest.fixture(scope="module")
def after_a_reprojection():
    """One run of the reproduction, in a fresh interpreter: this one may already be in that state.

    It is launched through fork_safe for the same reason. A child that faults before exec exits
    with status 11 (the script points SIGSEGV at _exit) rather than leaving a crash report.
    """
    if importlib.util.find_spec("geopandas") is None:
        pytest.skip("the reproduction reprojects through geopandas")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        p for p in (str(ROOT), os.environ.get("PYTHONPATH", "")) if p)}
    proc = fork_safe.run([sys.executable, str(REPRO), "--json"], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=300)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    assert lines, f"no result from the reproduction (rc {proc.returncode}): {proc.stderr[-2000:]}"
    return json.loads(lines[-1])


def test_a_child_started_after_a_reprojection_runs(after_a_reprojection):
    """What qgis_process and execute_code need: the child runs, in the directory it was given."""
    got = after_a_reprojection["fork_safe"]
    assert got["returncode"] == 0, got
    assert got["stdout"] == after_a_reprojection["cwd"]


@pytest.mark.skipif(sys.platform != "darwin", reason="the fork crash is macOS-only")
def test_the_same_child_started_by_forking_dies_before_exec(after_a_reprojection):
    """The hazard itself, so the test above is known to have run in a state that kills a fork.

    Skipped rather than failed where it does not reproduce: that says this machine's PROJ or
    libsystem_trace differs, not that fork_safe is wrong.
    """
    got = after_a_reprojection["subprocess"]
    if got["returncode"] == 0:
        pytest.skip("this environment does not reproduce the crash (pyproj "
                    f"{after_a_reprojection['pyproj']}, PROJ {after_a_reprojection['proj']})")
    assert got["returncode"] in (10, 11), got        # SIGBUS / SIGSEGV, turned into _exit(sig)


# --- the mechanics that make it safe ------------------------------------------------------------

@needs_posix_spawn
def test_the_child_is_started_by_posix_spawn_with_and_without_a_cwd(macos_path, tmp_path, monkeypatch):
    """CPython falls back to fork_exec silently when any condition is missed; pin that it does not."""
    spawned = []
    real = subprocess.Popen._posix_spawn

    def spy(self, args, executable, *rest, **kwargs):
        spawned.append(os.fsdecode(executable))
        return real(self, args, executable, *rest, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "_posix_spawn", spy)
    assert fork_safe.run(["true"], capture_output=True).returncode == 0
    in_dir = fork_safe.run(["pwd"], cwd=tmp_path, capture_output=True, text=True)
    assert in_dir.stdout.strip() == os.path.realpath(tmp_path)
    assert len(spawned) == 2 and all(os.path.isabs(p) for p in spawned), spawned


def test_the_command_replaces_the_shell_that_changed_directory(macos_path, tmp_path):
    """The command, not /bin/sh, is our child: so its exit status is the command's, and a kill on
    timeout reaches the command instead of orphaning it behind a dead shell."""
    got = fork_safe.run([sys.executable, "-c", "import os; print(os.getppid())"], cwd=tmp_path,
                        capture_output=True, text=True)
    assert int(got.stdout) == os.getpid()
    status = fork_safe.run([sys.executable, "-c", "raise SystemExit(7)"], cwd=tmp_path)
    assert status.returncode == 7


def test_a_missing_program_or_directory_raises_what_subprocess_raises(macos_path, tmp_path):
    """Callers map FileNotFoundError to "qgis_process not found" / "docker executable not found"."""
    with pytest.raises(FileNotFoundError) as missing_program:
        fork_safe.run(["no-such-program-anywhere"], capture_output=True)
    assert missing_program.value.filename == "no-such-program-anywhere"
    with pytest.raises(FileNotFoundError) as missing_dir:
        fork_safe.run(["true"], cwd=tmp_path / "gone", capture_output=True)
    assert missing_dir.value.filename == str(tmp_path / "gone")


@pytest.mark.parametrize("option", [{"preexec_fn": lambda: None}, {"start_new_session": True},
                                    {"close_fds": True}, {"pass_fds": (3,)}, {"shell": True}])
def test_options_that_would_fork_are_refused_on_every_platform(option):
    """Refused everywhere, so a Linux-only test run still catches a caller that brings fork back."""
    with pytest.raises(TypeError, match="does not take"):
        fork_safe.run(["true"], **option)


def test_elsewhere_it_is_subprocess_run_unchanged(monkeypatch, tmp_path):
    seen = {}

    def fake_run(args, **kwargs):
        seen.update(args=args, **kwargs)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(fork_safe, "avoids_fork", lambda: False)
    monkeypatch.setattr(subprocess, "run", fake_run)
    fork_safe.run(["qgis_process", "--json", "run"], cwd=tmp_path, env={"A": "1"}, timeout=5)
    assert seen == {"args": ["qgis_process", "--json", "run"], "cwd": tmp_path, "env": {"A": "1"},
                    "timeout": 5}


# --- nothing in the agent process goes around it ------------------------------------------------

# multiprocessing too: its default 'spawn' start on macOS launches the child with fork_exec, and in
# the crashing state that child died exactly like qgis_process did (-11, measured). asyncio's
# subprocesses are subprocess.Popen with the default close_fds, so fork_exec as well.
_DIRECT_LAUNCH = re.compile(
    r"\bsubprocess\.(run|Popen|call|check_call|check_output|getoutput|getstatusoutput)\("
    r"|^\s*import subprocess as |^\s*from subprocess import .*\b(run|Popen|call|check_call|check_output)\b"
    r"|\bos\.(system|popen|fork|forkpty|spawn[lv]p?e?)\("
    r"|\bmultiprocessing\.(Process|Pool|get_context|set_start_method)\(|\bProcessPoolExecutor\("
    r"|\bcreate_subprocess_(exec|shell)\(", re.M)


def test_the_agent_process_starts_children_only_through_fork_safe():
    """A new tool calling subprocess.run would bring the crash back on a Mac, and only there,
    where no CI runs. The modules here are imported into the long-lived agent process."""
    offenders = []
    for package in ("agent_runtime", "rag_pipeline", "api"):
        for path in sorted((ROOT / package).rglob("*.py")):
            if "tests" in path.relative_to(ROOT).parts or path.name == "fork_safe.py":
                continue
            for match in _DIRECT_LAUNCH.finditer(path.read_text(encoding="utf-8", errors="replace")):
                offenders.append(f"{path.relative_to(ROOT)}: {match.group(0).strip()}")
    assert not offenders, "start children with agent_runtime.fork_safe.run:\n" + "\n".join(offenders)
