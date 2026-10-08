"""Tests for sandboxed code execution.

The local subprocess backend is exercised for real (host python). The Docker
backend is checked at the argv level (no Docker needed) + via the selector.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_runtime.code_execution import (
    DisabledExecutor,
    DockerCodeExecutor,
    ExecResult,
    LocalSubprocessExecutor,
    get_code_executor,
    is_code_exec_enabled,
)


# --- flag ------------------------------------------------------------------

def test_is_code_exec_enabled(monkeypatch):
    monkeypatch.delenv("AGENT_CODE_EXEC", raising=False)
    assert is_code_exec_enabled() is True            # ON by default
    monkeypatch.setenv("AGENT_CODE_EXEC", "0")
    assert is_code_exec_enabled() is False           # explicit falsy disables
    monkeypatch.setenv("AGENT_CODE_EXEC", "false")
    assert is_code_exec_enabled() is False
    monkeypatch.setenv("AGENT_CODE_EXEC", "1")
    assert is_code_exec_enabled() is True


# --- local backend (real execution) ---------------------------------------

def test_local_executor_runs_and_captures_stdout():
    r = LocalSubprocessExecutor().execute("print('hello-sandbox')")
    assert r.exit_code == 0 and r.ok
    assert "hello-sandbox" in r.stdout
    assert r.backend == "local-unsafe"


def test_local_executor_reports_errors():
    r = LocalSubprocessExecutor().execute("raise ValueError('boom')")
    assert r.exit_code not in (0, None) and not r.ok
    assert "boom" in r.stderr


def test_local_executor_timeout():
    r = LocalSubprocessExecutor().execute("import time; time.sleep(3)", timeout=1)
    assert r.timed_out is True and not r.ok


def test_local_executor_persists_artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    r = LocalSubprocessExecutor().execute("open('result.txt', 'w').write('data')")
    assert r.ok
    assert "result.txt" in [a["filename"] for a in r.artifacts]


def test_executed_code_is_returned_and_downloadable(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    src = "print('hello-with-code')"
    r = LocalSubprocessExecutor().execute(src)
    # inline
    assert r.code == src
    assert r.to_dict()["code"] == src
    # downloadable source artifact
    source_arts = [a for a in r.artifacts if a.get("kind") == "source"]
    assert source_arts and source_arts[0]["filename"] == "executed_code.py"
    assert source_arts[0].get("download_url")


def test_unsupported_language():
    r = LocalSubprocessExecutor().execute("echo hi", language="bash")
    assert r.error and "unsupported" in r.error.lower()


# --- docker backend (argv hardening) --------------------------------------

def test_docker_argv_is_hardened():
    argv = DockerCodeExecutor(image="img:test", memory="256m", cpus="0.5").build_argv(
        Path("/tmp/work"), "agentexec_x"
    )
    s = " ".join(argv)
    assert "docker run" in s and "--rm" in argv
    assert "--network none" in s          # no network during execution
    assert "--read-only" in argv          # read-only rootfs
    assert "--cap-drop ALL" in s          # dropped caps
    assert "no-new-privileges" in s       # no privilege escalation
    assert "--memory 256m" in s and "--cpus 0.5" in s and "--pids-limit" in s
    assert "-v" in argv                   # only the work-dir bind mount
    assert "PYTHONPATH=/work/.deps" in s  # installed deps importable
    assert argv[-3:] == ["img:test", "python", "/work/script.py"]


def test_docker_install_argv():
    argv = DockerCodeExecutor(image="img:test").build_install_argv(
        Path("/tmp/work"), ["numpy", "pandas==2.2"], "agentexec_pip_x"
    )
    s = " ".join(argv)
    assert "pip install --no-cache-dir --upgrade --target /work/.deps" in s
    # --upgrade is load-bearing since the target can be a warm cache: pip leaves an existing
    # copy in a --target dir alone and only warns, so without it a cached package would pin
    # whatever version arrived first and `pandas==2.2` would never take effect.
    assert "--upgrade" in argv
    assert "--network none" not in s             # install phase HAS network
    assert "TMPDIR=/work/.piptmp" in s           # pip scratch on real disk, not tmpfs
    assert argv[-2:] == ["numpy", "pandas==2.2"]


def test_install_phase_gets_larger_memory_than_exec():
    """P1-5: deps-install gets its own (larger) memory budget to avoid OOM (exit 137)."""
    ex = DockerCodeExecutor(image="img:test", memory="256m", install_memory="2g")
    install = " ".join(ex.build_install_argv(Path("/tmp/work"), ["numpy"], "n"))
    execute = " ".join(ex.build_argv(Path("/tmp/work"), "n"))
    assert "--memory 2g" in install     # install uses the larger budget
    assert "--memory 256m" in execute   # exec keeps the tighter limit
    assert "--memory 2g" not in execute


def test_sanitize_deps_accepts_valid_and_rejects_unsafe(monkeypatch):
    monkeypatch.delenv("AGENT_CODE_EXEC_PIP_ALLOW", raising=False)
    from agent_runtime.code_execution import _sanitize_deps

    allowed, rejected = _sanitize_deps(
        ["numpy", "pandas==2.2.0", "scikit-learn>=1.0", "fiona[s3]",
         "-rrequirements.txt", "evil; rm -rf /", "two words", "--index-url=x"]
    )
    assert allowed == ["numpy", "pandas==2.2.0", "scikit-learn>=1.0", "fiona[s3]"]
    assert "-rrequirements.txt" in rejected and "--index-url=x" in rejected
    assert "evil; rm -rf /" in rejected and "two words" in rejected


def test_sanitize_deps_allowlist(monkeypatch):
    from agent_runtime.code_execution import _sanitize_deps

    monkeypatch.setenv("AGENT_CODE_EXEC_PIP_ALLOW", "numpy,pandas")
    allowed, rejected = _sanitize_deps(["numpy", "pandas==2.2", "requests"])
    assert allowed == ["numpy", "pandas==2.2"]
    assert "requests" in rejected


# --- executor selection ----------------------------------------------------

def test_selector_local(monkeypatch):
    monkeypatch.setenv("AGENT_CODE_EXEC_BACKEND", "local")
    assert isinstance(get_code_executor(), LocalSubprocessExecutor)


def test_selector_docker_present(monkeypatch):
    import agent_runtime.code_execution as ce
    monkeypatch.setenv("AGENT_CODE_EXEC_BACKEND", "docker")
    monkeypatch.setattr(ce, "_docker_available", lambda: True)
    assert isinstance(ce.get_code_executor(), DockerCodeExecutor)


def test_selector_docker_missing_is_disabled(monkeypatch):
    import agent_runtime.code_execution as ce
    monkeypatch.setenv("AGENT_CODE_EXEC_BACKEND", "docker")
    monkeypatch.setattr(ce, "_docker_available", lambda: False)
    ex = ce.get_code_executor()
    assert isinstance(ex, DisabledExecutor)
    # never runs code — returns an error result
    assert ex.execute("print(1)").error


# --- the execute_code tool -------------------------------------------------

def test_execute_code_tool_with_stub_executor():
    from agent_runtime.langchain_exec_tools import make_code_execution_tools

    captured = {}

    class _Stub:
        def execute(self, code, language="python", timeout=None, dependencies=None,
                    input_files=None, **kwargs):  # **kwargs: session_id / tier (M1.1)
            captured["dependencies"] = dependencies
            captured["input_files"] = input_files
            return ExecResult(exit_code=0, stdout="captured-out", stderr="", backend="stub")

    tools = make_code_execution_tools(executor=_Stub())
    assert tools[0].name == "execute_code"
    out = json.loads(tools[0].invoke({"code": "import numpy", "dependencies": ["numpy"]}))
    assert out["ok"] is True
    assert out["stdout"] == "captured-out"
    assert out["exit_code"] == 0
    assert captured["dependencies"] == ["numpy"]  # dependencies threaded to the executor
    assert captured["input_files"] == []  # no uploads requested -> empty staging list


# --- uploaded-file staging into the sandbox --------------------------------

def test_stage_inputs_copies_and_rejects_traversal(tmp_path):
    from agent_runtime.code_execution import _stage_inputs

    src = tmp_path / "src.txt"
    src.write_text("payload")
    work = tmp_path / "work"
    work.mkdir()

    staged, errors, _shadowed = _stage_inputs(work, [
        {"source": str(src), "dest": "ok.txt"},
        {"source": str(src), "dest": "../escape.txt"},          # traversal
        {"source": str(src), "dest": "sub/nested.txt"},         # separator
        {"source": str(tmp_path / "missing"), "dest": "m.txt"},  # missing source
    ])
    assert staged == ["ok.txt"]
    assert (work / "ok.txt").read_text() == "payload"
    assert not (work / "escape.txt").exists()
    reasons = {e.get("error") for e in errors}
    assert "invalid destination name" in reasons
    assert "source file not found" in reasons


def test_execute_code_tool_resolves_and_stages_file_specs(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    from agent_runtime.file_store import create_output_file
    from agent_runtime.langchain_exec_tools import make_code_execution_tools

    rec = create_output_file("data.csv", "a,b\n1,2\n")
    fid = rec["file_id"]

    captured = {}

    class _Stub:
        def execute(self, code, language="python", timeout=None, dependencies=None,
                    input_files=None, **kwargs):  # **kwargs: session_id / tier (M1.1)
            captured["input_files"] = input_files
            return ExecResult(exit_code=0, stdout="", stderr="", backend="stub")

    tools = make_code_execution_tools(executor=_Stub())
    out = json.loads(tools[0].invoke({"code": "print(1)", "input_files": [fid]}))

    # staged under BOTH the file_id and the original filename
    dests = {s["dest"] for s in captured["input_files"]}
    assert fid in dests and "data.csv" in dests
    assert all(s["source"].endswith("data.csv") for s in captured["input_files"])
    info = out["input_files"][0]
    assert info["file_id"] == fid and "data.csv" in info["available_as"] and fid in info["available_as"]


def test_execute_code_reads_uploaded_file_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    from agent_runtime.file_store import create_output_file
    from agent_runtime.langchain_exec_tools import make_code_execution_tools

    rec = create_output_file("data.csv", "a,b\n1,2\n3,4\n")
    fid = rec["file_id"]
    tools = make_code_execution_tools(executor=LocalSubprocessExecutor())

    # read by original filename
    out = json.loads(tools[0].invoke(
        {"code": "print(open('data.csv').read())", "input_files": [fid]}))
    assert out["ok"] is True and "a,b" in out["stdout"]

    # read by file_id (the name the model used in the failing trace)
    out2 = json.loads(tools[0].invoke(
        {"code": f"print(open('{fid}').read())", "input_files": [fid]}))
    assert out2["ok"] is True and "a,b" in out2["stdout"]


def test_staged_input_not_re_persisted_but_new_outputs_are(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    from agent_runtime.file_store import create_output_file
    from agent_runtime.langchain_exec_tools import make_code_execution_tools

    rec = create_output_file("in.csv", "a\n1\n")
    fid = rec["file_id"]
    tools = make_code_execution_tools(executor=LocalSubprocessExecutor())

    out = json.loads(tools[0].invoke(
        {"code": "open('out.txt','w').write(open('in.csv').read())", "input_files": [fid]}))
    arts = [a["filename"] for a in out["artifacts"]]
    assert "out.txt" in arts          # genuine output is persisted
    assert "executed_code.py" in arts  # source is persisted
    assert "in.csv" not in arts        # staged input is NOT re-persisted as an output


def test_execute_code_unknown_input_file_reports_error(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    from agent_runtime.langchain_exec_tools import make_code_execution_tools

    tools = make_code_execution_tools(executor=LocalSubprocessExecutor())
    out = json.loads(tools[0].invoke(
        {"code": "print('ok')", "input_files": ["file_does_not_exist"]}))
    assert out["input_file_errors"][0]["ref"] == "file_does_not_exist"
    assert out["ok"] is True  # the run itself still succeeds


# --- auto-staging conversation files (default_input_file_ids) --------------

def test_default_input_file_ids_are_auto_staged(monkeypatch, tmp_path):
    """Conversation-attached files are staged without the model naming them."""
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    from agent_runtime.file_store import create_output_file
    from agent_runtime.langchain_exec_tools import make_code_execution_tools

    rec = create_output_file("conv.csv", "a\n1\n")
    fid = rec["file_id"]
    tools = make_code_execution_tools(
        executor=LocalSubprocessExecutor(), default_input_file_ids=[fid])

    # model did NOT pass input_files — file is available anyway
    out = json.loads(tools[0].invoke({"code": "print(open('conv.csv').read())"}))
    assert out["ok"] is True and "a" in out["stdout"]
    assert out["input_files"][0]["file_id"] == fid


def test_default_and_explicit_union_deduped(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    from agent_runtime.file_store import create_output_file
    from agent_runtime.langchain_exec_tools import make_code_execution_tools

    a = create_output_file("a.csv", "x\n")["file_id"]
    b = create_output_file("b.csv", "y\n")["file_id"]

    captured = {}

    class _Stub:
        def execute(self, code, language="python", timeout=None, dependencies=None,
                    input_files=None, **kwargs):  # **kwargs: session_id / tier (M1.1)
            captured["input_files"] = input_files
            return ExecResult(exit_code=0, stdout="", stderr="", backend="stub")

    tools = make_code_execution_tools(executor=_Stub(), default_input_file_ids=[a])
    out = json.loads(tools[0].invoke({"code": "print(1)", "input_files": [a, b]}))  # 'a' in both
    # each unique file staged once (deduped by source), under id + filename
    info_ids = [i["file_id"] for i in out["input_files"]]
    assert info_ids == [a, b]  # 'a' not duplicated
    dests = {s["dest"] for s in captured["input_files"]}
    assert {a, "a.csv", b, "b.csv"} <= dests


def test_input_file_count_cap_skips_extras(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENT_CODE_EXEC_MAX_INPUT_FILES", "1")
    from agent_runtime.file_store import create_output_file
    from agent_runtime.langchain_exec_tools import make_code_execution_tools

    a = create_output_file("a.csv", "x\n")["file_id"]
    b = create_output_file("b.csv", "y\n")["file_id"]

    class _Stub:
        def execute(self, code, language="python", timeout=None, dependencies=None,
                    input_files=None, **kwargs):  # **kwargs: session_id / tier (M1.1)
            return ExecResult(exit_code=0, stdout="", stderr="", backend="stub")

    tools = make_code_execution_tools(executor=_Stub(), default_input_file_ids=[a, b])
    out = json.loads(tools[0].invoke({"code": "print(1)"}))
    assert len(out["input_files"]) == 1
    assert out["input_files_skipped"][0]["reason"] == "max input files exceeded"


# --- per-request code_exec flag wires the tool into the code peer ----------

def test_code_exec_flag_controls_tool_wiring(monkeypatch):
    import agent_runtime.executor_factory as ef
    import agent_runtime.supervisor_graph as sg

    captured = {}

    def fake_build(**kwargs):
        captured["tools"] = [getattr(t, "name", "") for t in (kwargs.get("preloaded_tools") or [])]
        return object()

    monkeypatch.setattr(ef, "build_agent_executor", fake_build)
    monkeypatch.setattr(ef, "invoke_agent_with_payload_fallback", lambda *a, **k: {"messages": []})

    # Per-request ON -> execute_code wired even though env is unset.
    monkeypatch.delenv("AGENT_CODE_EXEC", raising=False)
    sg.default_code_fn(code_exec=True)("write code", [], {"thread_id": None})
    assert "execute_code" in captured["tools"]

    # Per-request OFF -> not wired.
    sg.default_code_fn(code_exec=False)("write code", [], {"thread_id": None})
    assert "execute_code" not in captured["tools"]


def test_unavailable_work_root_returns_tool_error_not_crash(monkeypatch, tmp_path):
    """A missing/uncreatable work root (e.g. the DooD bind mount absent) must yield an
    ExecResult error — the deployed failure mode was mkdtemp raising FileNotFoundError and
    killing the whole turn/stream."""
    import agent_runtime.code_execution as ce
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("file, not dir")
    # _work_root() catches makedirs failure -> force mkdtemp itself to fail instead.
    monkeypatch.setattr(ce, "_work_root", lambda: str(blocker))
    res = LocalSubprocessExecutor().execute("print('hi')")
    assert res.exit_code is None
    assert "work dir unavailable" in (res.error or "")
    assert "AGENT_CODE_EXEC_WORK_ROOT" in (res.error or "")


# --- artifacts are named for their purpose, not a fixed constant -----------------

def test_saved_source_is_named_for_what_the_run_does():
    """Several runs in one turn used to all arrive as `executed_code.py`."""
    from agent_runtime.code_execution import _describe_code, _persist_source

    assert _describe_code('"""Convert the uploaded CSV to GeoJSON."""\n') == "convert_the_uploaded_csv_to"
    assert _describe_code("# buffer the rivers by 2 km\n") == "buffer_the_rivers_by_2"
    assert _describe_code("def compute_flood_risk(x):\n    return x\n") == "compute_flood_risk"
    assert _describe_code("import json\nprint(1)\n") is None      # nothing to go on

    assert _persist_source("print(1)", label="CSV to GeoJSON")[0]["filename"] == "csv_to_geojson.py"
    assert _persist_source('"""Plot rivers."""\n')[0]["filename"] == "plot_rivers.py"
    assert _persist_source("print(1)")[0]["filename"] == "executed_code.py"   # last resort


def test_geo_artifact_name_prefers_caller_then_source():
    from agent_runtime.langchain_geo_tools import artifact_name

    assert artifact_name("Chicago Rivers", "geojson", source="upload.zip") == "chicago_rivers.geojson"
    assert artifact_name(None, "geojson", source="chicago_tracts.zip") == "chicago_tracts.geojson"
    assert artifact_name(None, "png", source=None, default="vector_plot") == "vector_plot.png"
    assert artifact_name(None, ".png", source="/vsizip//tmp/a/rivers.zip") == "rivers.png"


def test_signal_deaths_are_diagnosed_in_both_conventions():
    """A signalled run must name its cause: docker reports 128+N, subprocess reports -N."""
    from agent_runtime.code_execution import _diagnose_abnormal_exit as diagnose

    assert "SIGKILL" in diagnose(137, "", None) and "memory limit" in diagnose(137, "", None)
    assert "SIGSEGV" in diagnose(139, "", None)      # container segfault
    assert "SIGSEGV" in diagnose(-11, "", None)      # docker CLI killed
    assert "nothing was written" in diagnose(137, "", None)
    # An ordinary failure explains itself through stderr; don't editorialize over it.
    assert diagnose(1, "Traceback ...", None) is None
    assert diagnose(0, "", None) is None
    assert diagnose(137, "", "already diagnosed") is None


# --- a conversation's code keeps its workspace between runs -----------------------
#
# Both branches grew this independently: prototype copied a durable directory forward into a
# throwaway work dir each run, this one bind-mounts the durable directory as /work. The merge kept
# the mount — no per-run copy, and staged inputs and the gate's report live with the files instead
# of being copied around. These two tests are prototype's, retargeted: the properties it was
# protecting are real and the surviving design has to hold them too.

# The durable per-conversation workspace is prototype's design (copy-in / copy-out, read-only
# .deps cache). backend_swap retargeted these two tests onto its run-in-place design at the
# 2026-08-27 merge; with prototype's design surviving the 2026-10-01 integration, its originals
# are restored verbatim.

def test_session_workspace_is_per_conversation_and_opt_in(tmp_path, monkeypatch):
    from agent_runtime.code_execution import _session_workspace

    monkeypatch.setenv("AGENT_CODE_EXEC_WORK_ROOT", str(tmp_path))
    assert _session_workspace(None) is None          # no session -> throwaway run, as before
    assert _session_workspace("") is None
    a1 = _session_workspace("thread-a")
    a2 = _session_workspace("thread-a")
    b = _session_workspace("thread-b")
    assert a1 == a2 and a1.is_dir()                  # stable across runs
    assert b != a1                                   # conversations don't share a workspace
    assert _session_workspace("../../etc/passwd").name.startswith("agentws_")  # path-safe


def test_carried_files_are_not_re_persisted_unless_changed(tmp_path):
    """A file carried in from a previous run is not an output of THIS run."""
    import time
    from agent_runtime.code_execution import _copy_tree, _stat_map

    ws, work = tmp_path / "ws", tmp_path / "work"
    ws.mkdir(); work.mkdir()
    (ws / "project.txt").write_text("step 1")
    (ws / ".deps").mkdir(); (ws / ".deps" / "junk.py").write_text("x")

    _copy_tree(ws, work)
    carried = _stat_map(work)
    assert "project.txt" in carried
    assert not (work / ".deps").exists()             # dep/cache dirs are never carried

    (work / "new.geojson").write_text("{}")          # this run's real output
    time.sleep(0.01)
    (work / "project.txt").write_text("step 1 + 2")  # and a modified carry-in

    now = _stat_map(work)
    unchanged = {rel for rel, sig in now.items() if carried.get(rel) == sig}
    assert unchanged == set()                        # project.txt changed -> counts as output
    assert "new.geojson" in now


def test_large_outputs_are_called_out_in_the_result():
    """An 89MB intermediate was written every turn with nothing in the transcript saying so."""
    from agent_runtime.code_execution import _size_report

    assert _size_report([{"filename": "small.geojson", "size_bytes": 2_000_000}]) is None
    note = _size_report([{"filename": "incidents.geojson", "size_bytes": 89_184_842},
                         {"filename": "run.py", "size_bytes": 437}])
    assert "incidents.geojson" in note and "85.1 MB" in note
    assert "ORIGINAL upload" in note          # says how to avoid it, not just that it happened


# --- what the image already ships ------------------------------------------
# `pip install --target` sets ignore_installed=True: pip does NOT consult the image's
# site-packages, so a baked package is reinstalled every run unless the executor drops it
# first. These pin the dropping, which is the only thing that makes sandbox/Dockerfile pay off.

def test_bare_names_are_dropped_but_pinned_specs_are_not():
    from agent_runtime.code_execution import _drop_preinstalled

    kept, skipped = _drop_preinstalled(
        ["numpy", "geopandas==0.14", "scikit_learn", "rioxarray"],
        frozenset({"numpy", "geopandas", "scikit-learn"}),
    )
    assert kept == ["geopandas==0.14", "rioxarray"]
    # A pin must never be satisfied by whatever version the image happens to carry, and
    # scikit_learn/scikit-learn are the same distribution (PEP 503).
    assert skipped == ["numpy", "scikit_learn"]


def test_a_probe_that_fails_suppresses_nothing(monkeypatch):
    """Fail-safe: a probe that cannot run must not stop an install the code needs."""
    from agent_runtime import code_execution as ce

    monkeypatch.delenv(ce.PREINSTALLED_ENV, raising=False)
    monkeypatch.setattr(ce, "_probe_cache", {})
    ex = ce.DockerCodeExecutor(image="img:test")
    monkeypatch.setattr(ex, "_probe_versions", lambda: {})
    assert ex.preinstalled() == frozenset()
    assert ce._drop_preinstalled(["numpy"], ex.preinstalled()) == (["numpy"], [])


def test_probe_runs_once_per_image_then_is_cached(monkeypatch):
    from agent_runtime import code_execution as ce

    monkeypatch.delenv(ce.PREINSTALLED_ENV, raising=False)
    monkeypatch.setattr(ce, "_probe_cache", {})
    calls = []
    ex = ce.DockerCodeExecutor(image="img:test")
    monkeypatch.setattr(ex, "_probe_versions", lambda: (calls.append(1), {"numpy": "2.1.0"})[1])
    assert ex.preinstalled() == frozenset({"numpy"})
    assert ex.preinstalled() == frozenset({"numpy"})
    assert len(calls) == 1                       # a container start per run would defeat the point


def test_preinstalled_env_overrides_the_probe_and_empty_disables_it(monkeypatch):
    from agent_runtime import code_execution as ce

    monkeypatch.setattr(ce, "_probe_cache", {})
    ex = ce.DockerCodeExecutor(image="img:test")
    monkeypatch.setattr(ex, "_probe_versions", lambda: {"probed": "1.0"})
    monkeypatch.setenv(ce.PREINSTALLED_ENV, "numpy, pandas")
    assert ex.preinstalled() == frozenset({"numpy", "pandas"})
    monkeypatch.setenv(ce.PREINSTALLED_ENV, "")   # explicitly empty -> optimisation off
    assert ex.preinstalled() == frozenset()


def test_the_probe_is_no_less_confined_than_a_real_run(monkeypatch):
    """The probe must not be the thing that widens the sandbox."""
    from agent_runtime import code_execution as ce

    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        raise FileNotFoundError

    monkeypatch.setattr(ce.fork_safe, "run", fake_run)
    ce.DockerCodeExecutor(image="img:test")._probe_versions()
    s = " ".join(seen["argv"])
    assert "--network none" in s and "--read-only" in s and "--cap-drop ALL" in s


# --- the per-conversation dependency cache ---------------------------------

def test_deps_cache_is_mounted_not_copied(tmp_path, monkeypatch):
    """A site-packages tree is ~30k files; copying it twice a run would cost more than pip."""
    from agent_runtime.code_execution import DEPS_DIRNAME, _copy_tree

    ex = DockerCodeExecutor(image="img:test")
    cache = tmp_path / "cache"
    run = " ".join(ex.build_argv(Path("/w"), "n", cache))
    install = " ".join(ex.build_install_argv(Path("/w"), ["x"], "n", cache))
    assert f"{cache}:/work/{DEPS_DIRNAME}:ro" in run       # untrusted code cannot poison it
    assert f"{cache}:/work/{DEPS_DIRNAME}:rw" in install   # …but pip can fill it
    # And the copy path still skips it, so the bytes never move.
    ws, work = tmp_path / "ws", tmp_path / "work"
    (ws / DEPS_DIRNAME).mkdir(parents=True); (ws / DEPS_DIRNAME / "pkg.py").write_text("x")
    work.mkdir()
    _copy_tree(ws, work)
    assert not (work / DEPS_DIRNAME).exists()


def test_no_session_means_no_cache_mount():
    """graph_runtime wires execute_code with no session; it must behave exactly as before."""
    ex = DockerCodeExecutor(image="img:test")
    assert "/work/.deps:ro" not in " ".join(ex.build_argv(Path("/w"), "n"))


def test_cached_packages_are_read_from_dist_info(tmp_path):
    from agent_runtime.code_execution import cached_dep_names

    for name in ("numpy-2.1.0.dist-info", "scikit_learn-1.5.0.dist-info"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "RECORD").write_text("")     # pip writes this last
    (tmp_path / "loose_file.py").write_text("x")
    assert cached_dep_names(tmp_path) == frozenset({"numpy", "scikit-learn"})
    assert cached_dep_names(None) == frozenset()

    # An install killed mid-move leaves metadata with no package beside it. Trusting the
    # directory NAME alone would suppress the reinstall for the life of the conversation.
    (tmp_path / "torn-9.9.dist-info").mkdir()
    assert "torn" not in cached_dep_names(tmp_path)


def test_a_cached_package_is_importable_on_a_later_run(tmp_path, monkeypatch):
    """The whole point: run 2 imports what run 1 installed, having installed nothing itself."""
    monkeypatch.setenv("AGENT_CODE_EXEC_WORK_ROOT", str(tmp_path))
    from agent_runtime.code_execution import workspace_deps_dir

    cache = workspace_deps_dir("convo::codeexec")
    (cache / "tinylib.py").write_text("VALUE = 'cached'\n")
    result = LocalSubprocessExecutor().execute(
        "import tinylib; print(tinylib.VALUE)", session="convo::codeexec")
    assert result.exit_code == 0 and "cached" in result.stdout
    assert result.installed == []                 # nothing was sent to pip
    assert cache.is_dir()                         # and the cache outlived the run


def test_oversized_cache_is_reset(tmp_path, monkeypatch):
    from agent_runtime import code_execution as ce

    cache = tmp_path / ".deps"
    cache.mkdir()
    (cache / "big.bin").write_bytes(b"0" * 2_000_000)
    monkeypatch.setattr(ce, "DEPS_CACHE_MAX_MB", 100)
    assert ce.reset_deps_cache_if_oversized(cache) is False
    monkeypatch.setattr(ce, "DEPS_CACHE_MAX_MB", 1)
    assert ce.reset_deps_cache_if_oversized(cache) is True
    assert cache.is_dir() and not (cache / "big.bin").exists()   # reset, not removed


def test_stale_workspaces_are_swept_and_live_ones_are_not(tmp_path, monkeypatch):
    """Without this, caching .deps means one site-packages tree per conversation, forever."""
    import os
    import time

    from agent_runtime import code_execution as ce

    monkeypatch.setenv("AGENT_CODE_EXEC_WORK_ROOT", str(tmp_path))
    monkeypatch.setattr(ce, "WORKSPACE_TTL_HOURS", 24.0)
    monkeypatch.setattr(ce, "_last_sweep", 0.0)
    stale = tmp_path / "agentws_old"
    stale.mkdir()
    os.utime(stale, (time.time() - 90000, time.time() - 90000))   # 25h
    fresh = ce._session_workspace("live::codeexec")
    ce._sweep_workspaces(tmp_path, force=True)
    assert not stale.exists()
    assert fresh.is_dir()


def test_the_sweep_is_throttled_off_the_hot_path(tmp_path, monkeypatch):
    """It is called from every execute_code and every workspace file tool; globbing the work
    root each time is waste on a deployment with a thousand conversations in it."""
    import os
    import time

    from agent_runtime import code_execution as ce

    monkeypatch.setattr(ce, "WORKSPACE_TTL_HOURS", 24.0)
    stale = tmp_path / "agentws_old"
    stale.mkdir()
    os.utime(stale, (time.time() - 90000, time.time() - 90000))
    monkeypatch.setattr(ce, "_last_sweep", time.time())   # swept a moment ago
    ce._sweep_workspaces(tmp_path)
    assert stale.exists()                                  # throttled, not walked
    ce._sweep_workspaces(tmp_path, force=True)
    assert not stale.exists()


def test_using_a_workspace_marks_it_fresh(tmp_path, monkeypatch):
    """A dir's mtime only moves when an entry is added/removed, so a long conversation that
    rewrites the same files would look untouched and be swept out from under itself."""
    import os
    import time

    from agent_runtime import code_execution as ce

    monkeypatch.setenv("AGENT_CODE_EXEC_WORK_ROOT", str(tmp_path))
    ws = ce._session_workspace("s::codeexec")
    os.utime(ws, (time.time() - 90000, time.time() - 90000))
    assert ce._session_workspace("s::codeexec").stat().st_mtime > time.time() - 60


# --- patch-and-rerun --------------------------------------------------------

def test_workspace_paths_stay_inside_the_workspace(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setenv("AGENT_CODE_EXEC_WORK_ROOT", str(tmp_path))
    from agent_runtime.code_execution import resolve_workspace_file

    S = "s::codeexec"
    assert resolve_workspace_file(S, "sub/main.py").name == "main.py"
    for bad, why in [("../escape.py", "outside"), ("/etc/passwd", "absolute"),
                     (".deps/numpy.py", "installed packages"), ("", "no file named")]:
        with pytest.raises(ValueError) as err:
            resolve_workspace_file(S, bad)
        assert why in str(err.value)
    # No workspace at all: say what to do instead of failing opaquely.
    with pytest.raises(ValueError) as err:
        resolve_workspace_file(None, "main.py")
    assert "`code`" in str(err.value)


def test_entrypoint_runs_a_workspace_file_and_a_patch_changes_the_next_run(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CODE_EXEC_WORK_ROOT", str(tmp_path))
    from agent_runtime.langchain_exec_tools import make_code_execution_tools

    ex = LocalSubprocessExecutor()
    tools = {t.name: t for t in make_code_execution_tools(executor=ex, session_id="s::codeexec")}
    call = lambda n, **kw: json.loads(tools[n].func(**kw))

    assert call("write_workspace_file", path="main.py", content="V = 2\nprint(V * 3)\n")["ok"]
    first = call("execute_code", entrypoint="main.py")
    assert first["exit_code"] == 0 and first["stdout"].strip() == "6"
    # An entrypoint run has no inline source, so it must not persist an empty .py download.
    assert first["artifacts"] == []

    assert call("edit_workspace_file", path="main.py", old_text="V = 2", new_text="V = 5")["ok"]
    second = call("execute_code", entrypoint="main.py")
    assert second["stdout"].strip() == "15"       # re-run without re-sending the program


def test_an_edit_that_could_hit_the_wrong_line_is_refused(tmp_path, monkeypatch):
    """Replacing one of three identical lines yields code that runs and is wrong."""
    monkeypatch.setenv("AGENT_CODE_EXEC_WORK_ROOT", str(tmp_path))
    from agent_runtime.langchain_exec_tools import make_code_execution_tools

    tools = {t.name: t for t in make_code_execution_tools(session_id="s::codeexec")}
    call = lambda n, **kw: json.loads(tools[n].func(**kw))
    call("write_workspace_file", path="m.py", content="x = 1\nx = 1\n")

    assert "appears 2 times" in call("edit_workspace_file", path="m.py",
                                     old_text="x = 1", new_text="x = 2")["error"]
    assert "does not appear" in call("edit_workspace_file", path="m.py",
                                     old_text="y = 9", new_text="z")["error"]
    assert "write_workspace_file" in call("edit_workspace_file", path="m.py",
                                          old_text="", new_text="z")["error"]


def test_a_run_with_neither_code_nor_entrypoint_says_what_to_pass():
    result = LocalSubprocessExecutor().execute("")
    assert result.exit_code is None
    assert "`code`" in result.error and "`entrypoint`" in result.error


def test_a_missing_entrypoint_names_the_files_that_are_there(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CODE_EXEC_WORK_ROOT", str(tmp_path))
    from agent_runtime.code_execution import _session_workspace

    (_session_workspace("s::codeexec") / "actual.py").write_text("print(1)\n")
    result = LocalSubprocessExecutor().execute("", session="s::codeexec", entrypoint="ghost.py")
    assert "actual.py" in result.error and "write_workspace_file" in result.error


def test_workspace_tools_appear_only_with_a_durable_workspace():
    from agent_runtime.langchain_exec_tools import make_code_execution_tools

    assert [t.name for t in make_code_execution_tools()] == ["execute_code"]
    assert {t.name for t in make_code_execution_tools(session_id="s::codeexec")} == {
        "execute_code", "write_workspace_file", "read_workspace_file", "edit_workspace_file"}


# --- fixes for defects an adversarial review found in the above -------------

def test_entrypoint_cannot_escape_the_workspace(tmp_path, monkeypatch):
    """The path that RUNS must be validated, not just the one read for dependency inference."""
    monkeypatch.setenv("AGENT_CODE_EXEC_WORK_ROOT", str(tmp_path))
    from agent_runtime.code_execution import _session_workspace

    _session_workspace("s::codeexec")
    (tmp_path / "outside.py").write_text("print('ESCAPED')\n")
    ex = LocalSubprocessExecutor()
    for bad in ("../outside.py", str(tmp_path / "outside.py"), "sub/../../outside.py"):
        result = ex.execute("", session="s::codeexec", entrypoint=bad)
        assert result.exit_code is None, f"{bad} was executed"
        assert "outside the working directory" in result.error or "absolute path" in result.error


def test_reserved_dirs_are_matched_case_insensitively(tmp_path, monkeypatch):
    """Path.resolve() does not canonicalise case, so '.DEPS' slipped an exact-match check
    and wrote into the durable dependency cache on macOS."""
    import pytest

    monkeypatch.setenv("AGENT_CODE_EXEC_WORK_ROOT", str(tmp_path))
    from agent_runtime.code_execution import resolve_workspace_file

    for name in (".deps/x.py", ".DEPS/x.py", ".PipTmp/x.py", "__PYCACHE__/x.py"):
        with pytest.raises(ValueError):
            resolve_workspace_file("s::codeexec", name)


def test_code_and_entrypoint_together_are_refused(tmp_path, monkeypatch):
    """Silently running the file while saving the unrun code as the run's source would put a
    program that never ran next to output from a different one."""
    monkeypatch.setenv("AGENT_CODE_EXEC_WORK_ROOT", str(tmp_path))
    from agent_runtime.code_execution import _session_workspace

    (_session_workspace("s::codeexec") / "main.py").write_text("print('the file')\n")
    result = LocalSubprocessExecutor().execute(
        "print('the code')", session="s::codeexec", entrypoint="main.py")
    assert result.exit_code is None and "not both" in result.error


def test_eviction_happens_before_the_install_decision(tmp_path, monkeypatch):
    """Evicting after deciding would delete exactly the packages nothing will reinstall."""
    from agent_runtime import code_execution as ce

    monkeypatch.setenv("AGENT_CODE_EXEC_WORK_ROOT", str(tmp_path))
    cache = ce.workspace_deps_dir("s::codeexec")
    (cache / "tinylib.py").write_text("VALUE = 1\n")
    (cache / "tinylib-1.0.dist-info").mkdir()
    (cache / "bulk.bin").write_bytes(b"0" * 3_000_000)
    monkeypatch.setattr(ce, "DEPS_CACHE_MAX_MB", 1)

    installed = {}

    class Probe(ce.LocalSubprocessExecutor):
        def _run(self, work, timeout, dependencies=None, deps_cache=None, entrypoint=None):
            installed["deps"] = list(dependencies or [])
            return 0, "", "", False, None

    Probe().execute("import tinylib", session="s::codeexec", dependencies=["tinylib"])
    assert installed["deps"] == ["tinylib"]    # evicted, so it must be reinstalled


def test_a_failed_probe_is_not_cached(monkeypatch):
    """Caching a failure would serve the empty set forever — one slow first image pull would
    permanently disable the optimisation with nothing saying why."""
    from agent_runtime import code_execution as ce

    monkeypatch.delenv(ce.PREINSTALLED_ENV, raising=False)
    monkeypatch.setattr(ce, "_probe_cache", {})
    ex = ce.DockerCodeExecutor(image="img:test")
    results = [None, {"numpy": "2.1.0"}]
    monkeypatch.setattr(ex, "_probe_versions", lambda: results.pop(0))
    assert ex.preinstalled() == frozenset()          # failure -> nothing skipped
    assert ex.preinstalled() == frozenset({"numpy"})  # …and it retries, rather than giving up


def test_an_image_with_nothing_still_caches(monkeypatch):
    """An empty result is an ANSWER, not a failure: python:3.11-slim must not be re-probed."""
    from agent_runtime import code_execution as ce

    monkeypatch.delenv(ce.PREINSTALLED_ENV, raising=False)
    monkeypatch.setattr(ce, "_probe_cache", {})
    calls = []
    ex = ce.DockerCodeExecutor(image="slim:test")
    monkeypatch.setattr(ex, "_probe_versions", lambda: (calls.append(1), {})[1])
    assert ex.preinstalled() == frozenset()
    assert ex.preinstalled() == frozenset()
    assert len(calls) == 1


def test_the_probe_container_is_bounded_and_killable(monkeypatch):
    """It must not be the one unbounded, unnamed container on the host."""
    from agent_runtime import code_execution as ce

    seen = {}

    def fake_run(argv, **kw):
        seen.setdefault("argv", argv)
        seen.setdefault("calls", []).append(argv)
        raise ce.subprocess.TimeoutExpired(cmd="docker", timeout=1)

    monkeypatch.setattr(ce.fork_safe, "run", fake_run)
    # Returns None even though the cleanup kill ALSO fails — a probe that cannot clean up
    # must not raise into the caller's run.
    assert ce.DockerCodeExecutor(image="img:test")._probe_versions() is None
    s = " ".join(seen["argv"])
    assert "--name" in s and "--memory" in s and "--pids-limit" in s
    assert any(a[:2] == ["docker", "kill"] for a in seen["calls"])   # it tried to kill it


def test_editing_a_non_utf8_file_is_refused_not_corrupted(tmp_path, monkeypatch):
    """errors='replace' + write-the-whole-file-back rewrites every undecodable byte."""
    monkeypatch.setenv("AGENT_CODE_EXEC_WORK_ROOT", str(tmp_path))
    from agent_runtime.code_execution import _session_workspace
    from agent_runtime.langchain_exec_tools import make_code_execution_tools

    raw = b"name,pop\nMalm\xf6,344000\nThreshold,1\n"
    (_session_workspace("s::codeexec") / "cities.csv").write_bytes(raw)
    tools = {t.name: t for t in make_code_execution_tools(session_id="s::codeexec")}
    out = json.loads(tools["edit_workspace_file"].func(
        path="cities.csv", old_text="Threshold,1", new_text="Threshold,2"))
    assert out["ok"] is False and "not UTF-8" in out["error"]
    assert (_session_workspace("s::codeexec") / "cities.csv").read_bytes() == raw


def test_thread_ids_that_slug_alike_get_separate_workspaces(tmp_path, monkeypatch):
    """thread_id is raw client input; the slug maps every unsafe character to '_', so
    'sess:42' and 'sess_42' shared one directory — files, and now a package cache."""
    monkeypatch.setenv("AGENT_CODE_EXEC_WORK_ROOT", str(tmp_path))
    from agent_runtime.code_execution import _session_workspace

    assert _session_workspace("sess:42::codeexec") != _session_workspace("sess_42::codeexec")


def test_blank_env_values_do_not_break_the_import(monkeypatch):
    """A `.env` line `KEY=` and compose interpolation both produce an empty string, and
    int('') at module scope takes the process down before it serves a request."""
    from agent_runtime.code_execution import _num_env

    monkeypatch.setenv("SOME_KNOB", "")
    assert _num_env("SOME_KNOB", 72.0) == 72.0
    monkeypatch.setenv("SOME_KNOB", "not-a-number")
    assert _num_env("SOME_KNOB", 72.0) == 72.0
    monkeypatch.setenv("SOME_KNOB", "5")
    assert _num_env("SOME_KNOB", 72.0) == 5.0


def test_a_partial_walk_does_not_report_an_empty_cache(tmp_path, monkeypatch):
    """Returning 0.0 on an OSError reads as 'well under the cap' and disables it forever."""
    from agent_runtime.code_execution import _dir_size_mb

    (tmp_path / "readable").mkdir()
    (tmp_path / "readable" / "a.bin").write_bytes(b"0" * 1_500_000)
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "b.bin").write_bytes(b"0" * 10)
    blocked.chmod(0o000)
    try:
        assert _dir_size_mb(tmp_path) > 1.0      # counts what it can reach
    finally:
        blocked.chmod(0o755)


def test_eviction_renames_before_deleting(tmp_path, monkeypatch):
    """Another run of the same conversation may have this directory mounted into a live
    container; rmtree in place pulls site-packages out from under a running import."""
    from agent_runtime import code_execution as ce

    cache = tmp_path / ".deps"
    cache.mkdir()
    (cache / "pkg.bin").write_bytes(b"0" * 2_000_000)
    monkeypatch.setattr(ce, "DEPS_CACHE_MAX_MB", 1)
    holder = cache / "pkg.bin"
    opened = holder.open("rb")                   # stand-in for a live container's open file
    try:
        assert ce.reset_deps_cache_if_oversized(cache) is True
        assert opened.read(8) == b"0" * 8        # the old inode survived the swap
        assert cache.is_dir() and not (cache / "pkg.bin").exists()
    finally:
        opened.close()


def test_an_upload_shadowing_a_workspace_file_is_reported(tmp_path):
    """Uploads stage after the workspace is carried in, so the model can read its own edited
    file and silently get the original upload instead."""
    from agent_runtime.code_execution import _stage_inputs

    work = tmp_path / "work"
    work.mkdir()
    (work / "data.csv").write_text("CLEANED")          # carried in from the workspace
    src = tmp_path / "upload.csv"
    src.write_text("RAW UPLOAD")
    staged, errors, shadowed = _stage_inputs(work, [{"source": str(src), "dest": "data.csv"}])
    assert staged == ["data.csv"] and not errors
    assert shadowed == ["data.csv"]


def test_installs_are_pinned_to_the_images_own_versions(tmp_path, monkeypatch):
    """`--target` sets ignore_installed, so pip re-resolves the FULL closure and drops its own
    numpy into the cache — which precedes site-packages on PYTHONPATH. One small install would
    otherwise swap the numpy that the image's rasterio and geopandas were compiled against,
    for the rest of the conversation, while reporting `installed: []` for it."""
    from agent_runtime.code_execution import _constraints_text

    text = _constraints_text({"numpy": "2.1.3", "pandas": "2.2.3", "unknown": ""})
    assert "numpy==2.1.3" in text and "pandas==2.2.3" in text
    assert "unknown" not in text                 # no version to pin to -> no constraint

    ex = DockerCodeExecutor(image="img:test")
    argv = ex.build_install_argv(Path("/w"), ["xarray"], "n", tmp_path, "image-constraints.txt")
    s = " ".join(argv)
    assert "--constraint /work/.piptmp/image-constraints.txt" in s
    # Under .piptmp because that dir is already excluded from artifacts and from copy-out.
    assert argv[-1] == "xarray"
    # With nothing known about the image, no constraint is imposed at all.
    assert "--constraint" not in " ".join(ex.build_install_argv(Path("/w"), ["xarray"], "n", tmp_path))


def test_an_install_that_did_not_finish_drops_the_cache(tmp_path):
    """pip moves the package tree and its .dist-info into --target separately, so a kill on
    the timeout leaves the cache half-written. Rebuilding costs one install; keeping a torn
    cache costs every later run in the conversation, silently."""
    from agent_runtime.code_execution import _evict_torn_cache

    cache = tmp_path / ".deps"
    cache.mkdir()
    (cache / "half-1.0.dist-info").mkdir()
    _evict_torn_cache(cache)
    assert cache.is_dir() and not any(cache.iterdir())
    _evict_torn_cache(None)          # no session, no cache, no crash


def test_the_probe_reports_versions_not_just_names(monkeypatch):
    from agent_runtime import code_execution as ce

    monkeypatch.delenv(ce.PREINSTALLED_ENV, raising=False)
    monkeypatch.setattr(ce, "_probe_cache", {})
    ex = ce.DockerCodeExecutor(image="img:test")
    monkeypatch.setattr(ex, "_probe_versions", lambda: {"numpy": "2.1.3", "pandas": "2.2.3"})
    assert ex.preinstalled() == frozenset({"numpy", "pandas"})
    assert ex.preinstalled_versions()["numpy"] == "2.1.3"
    # An explicit override carries no versions, so nothing is pinned from it.
    monkeypatch.setenv(ce.PREINSTALLED_ENV, "numpy")
    assert ex.preinstalled_versions() == {}
