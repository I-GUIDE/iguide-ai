from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agent_runtime import opencode_peer as ocp


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Deterministic env for every test: no leaked LLM / opencode settings."""
    for var in (
        "AGENT_CODE_PEER", "AGENT_OPENCODE_MODEL", "AGENT_OPENCODE_BASE_URL",
        "AGENT_OPENCODE_API_KEY", "AGENT_OPENCODE_IMAGE", "AGENT_OPENCODE_NETWORK",
        "AGENT_OPENCODE_TIMEOUT", "AGENT_OPENCODE_MEMORY", "AGENT_OPENCODE_CPUS",
        "AGENT_OPENCODE_PIDS", "VLLM_API_KEY", "VLLM_MODEL", "VLLM_PROXY",
        "OPENAI_KEY", "OPENAI_CHAT_MODEL", "OPENAI_MODEL", "OPENAI_BASE_URL",
        "AGENT_CODE_EXEC_WORK_ROOT",
    ):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Flag gating
# ---------------------------------------------------------------------------

def test_disabled_by_default():
    assert not ocp.is_opencode_peer_enabled()


@pytest.mark.parametrize("value,expected", [
    ("opencode", True),
    ("OpenCode", True),
    ("  opencode  ", True),
    ("langchain", False),
    ("1", False),
    ("", False),
])
def test_flag_values(monkeypatch, value, expected):
    monkeypatch.setenv("AGENT_CODE_PEER", value)
    assert ocp.is_opencode_peer_enabled() is expected


# ---------------------------------------------------------------------------
# LLM settings resolution (same chain as build_default_llm + overrides)
# ---------------------------------------------------------------------------

def test_llm_settings_vllm_over_openai(monkeypatch):
    monkeypatch.setenv("VLLM_MODEL", "vllm-model")
    monkeypatch.setenv("VLLM_PROXY", "http://vllm:8000/v1")
    monkeypatch.setenv("VLLM_API_KEY", "vllm-key")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "openai-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_KEY", "openai-key")
    s = ocp.resolve_llm_settings()
    assert s == {"model": "vllm-model", "base_url": "http://vllm:8000/v1", "api_key": "vllm-key"}


def test_llm_settings_opencode_overrides_win(monkeypatch):
    monkeypatch.setenv("VLLM_MODEL", "vllm-model")
    monkeypatch.setenv("VLLM_API_KEY", "vllm-key")
    monkeypatch.setenv("AGENT_OPENCODE_MODEL", "big-model")
    monkeypatch.setenv("AGENT_OPENCODE_BASE_URL", "http://host.docker.internal:8000/v1")
    monkeypatch.setenv("AGENT_OPENCODE_API_KEY", "peer-key")
    s = ocp.resolve_llm_settings()
    assert s == {
        "model": "big-model",
        "base_url": "http://host.docker.internal:8000/v1",
        "api_key": "peer-key",
    }


def test_llm_settings_base_url_normalized(monkeypatch):
    monkeypatch.setenv("VLLM_PROXY", "http://vllm:8000/v1/chat/completions")
    assert ocp.resolve_llm_settings()["base_url"] == "http://vllm:8000/v1"


# ---------------------------------------------------------------------------
# Generated opencode.json
# ---------------------------------------------------------------------------

def test_opencode_config_shape():
    cfg = ocp.build_opencode_config("Qwen/Qwen3.5-9B", "http://vllm:8000/v1")
    provider = cfg["provider"]["vllm"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == "http://vllm:8000/v1"
    # The key must be an {env:...} reference — never a literal in the work dir.
    assert provider["options"]["apiKey"] == "{env:AGENT_OPENCODE_API_KEY}"
    assert "Qwen/Qwen3.5-9B" in provider["models"]
    # Explicit limits — without them opencode defaults max_tokens to 32000,
    # which endpoints like gpt-4o (16384 cap) reject.
    assert provider["models"]["Qwen/Qwen3.5-9B"]["limit"] == {"context": 128_000, "output": 8_192}
    assert cfg["model"] == "vllm/Qwen/Qwen3.5-9B"
    assert cfg["share"] == "disabled"
    assert cfg["autoupdate"] is False
    assert cfg["permission"] == {"edit": "allow", "bash": "allow"}
    json.dumps(cfg)  # must be JSON-serializable


def test_opencode_config_without_base_url():
    cfg = ocp.build_opencode_config("gpt-4o", None)
    assert "baseURL" not in cfg["provider"]["vllm"]["options"]


# ---------------------------------------------------------------------------
# docker run argv
# ---------------------------------------------------------------------------

def test_docker_argv_hardening_and_network(tmp_path, monkeypatch):
    argv = ocp.build_docker_argv(tmp_path, "agentoc_test", "m1", "do the thing")
    joined = " ".join(argv)
    assert argv[:3] == ["docker", "run", "--rm"]
    for flag in ("--cap-drop", "--read-only", "--pids-limit", "--memory", "--cpus"):
        assert flag in argv
    assert "no-new-privileges" in argv
    # Unlike execute_code, the opencode container KEEPS network access.
    assert "--network none" not in joined
    assert f"{tmp_path}:/work:rw" in argv
    assert "HOME=/work" in argv
    # last tokens: image, opencode, run, --model, ref, prompt
    assert argv[-5:] == ["opencode", "run", "--model", "vllm/m1", "do the thing"]
    assert argv[-6] == ocp.DEFAULT_OPENCODE_IMAGE


def test_docker_argv_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OPENCODE_IMAGE", "custom-oc:1")
    monkeypatch.setenv("AGENT_OPENCODE_NETWORK", "platform-net")
    monkeypatch.setenv("AGENT_OPENCODE_MEMORY", "4g")
    argv = ocp.build_docker_argv(tmp_path, "n", "m", "p")
    assert "custom-oc:1" in argv
    assert "platform-net" in argv[argv.index("--network") + 1]
    assert "4g" in argv[argv.index("--memory") + 1]


# ---------------------------------------------------------------------------
# run_opencode (subprocess mocked — no docker needed)
# ---------------------------------------------------------------------------

class _FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_opencode_requires_api_key():
    result = ocp.run_opencode("hi")
    assert result["ok"] is False
    assert "required" in result["error"]


def test_run_opencode_success(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_API_KEY", "k")
    monkeypatch.setenv("VLLM_MODEL", "m")
    monkeypatch.setenv("AGENT_CODE_EXEC_WORK_ROOT", str(tmp_path))
    captured = {}

    def fake_run(argv, **kwargs):
        if argv[0] == "docker":
            captured["argv"] = argv
            captured["env"] = kwargs.get("env")
            # The generated config must exist in the work dir at run time.
            mount = next(a for a in argv if a.endswith(":/work:rw"))
            work = Path(mount.split(":", 1)[0])
            captured["config"] = json.loads((work / "opencode.json").read_text())
            return _FakeProc(0, "\x1b[1mAll done:\x1b[0m wrote result.csv\n", "")
        return _FakeProc(0, "", "")

    monkeypatch.setattr(ocp.fork_safe, "run", fake_run)
    result = ocp.run_opencode("compute things")
    assert result["ok"] is True
    assert result["exit_code"] == 0
    assert result["answer"] == "All done: wrote result.csv"  # ANSI stripped
    assert result["backend"] == "opencode-docker"
    assert result["model"] == "vllm/m"
    assert captured["env"]["AGENT_OPENCODE_API_KEY"] == "k"
    assert captured["config"]["model"] == "vllm/m"
    # Work dirs are throwaway — cleaned up after the run.
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("agentoc_")]


def test_run_opencode_failure_and_timeout(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_API_KEY", "k")
    monkeypatch.setenv("AGENT_CODE_EXEC_WORK_ROOT", str(tmp_path))

    monkeypatch.setattr(ocp.fork_safe, "run", lambda *a, **k: _FakeProc(7, "", "boom"))
    result = ocp.run_opencode("x")
    assert result["ok"] is False and result["exit_code"] == 7 and result["stderr"] == "boom"

    calls = []

    def fake_timeout(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["docker", "kill"]:
            return _FakeProc(0)
        raise subprocess.TimeoutExpired(cmd=argv, timeout=1)

    monkeypatch.setattr(ocp.fork_safe, "run", fake_timeout)
    result = ocp.run_opencode("x", timeout=30)
    assert result["timed_out"] is True and result["ok"] is False
    assert any(c[:2] == ["docker", "kill"] for c in calls)  # container reaped


# ---------------------------------------------------------------------------
# Peer adapter: result shape + supervisor dispatch
# ---------------------------------------------------------------------------

def test_peer_result_shape_on_failure(monkeypatch):
    monkeypatch.setattr(
        ocp, "run_opencode",
        lambda prompt, **kw: {"ok": False, "exit_code": 1, "answer": "", "stderr": "trace",
                              "timed_out": False, "error": None, "artifacts": [],
                              "backend": "opencode-docker", "model": "vllm/m"},
    )
    result = ocp.run_opencode_code_peer("q", evidence=[], state={})
    assert "opencode code peer failed" in result["answer"]
    assert "trace" in result["answer"]
    assert result["tool_calls"][0]["name"] == "opencode_run"
    assert result["tool_results"][0]["content"]["exit_code"] == 1


def test_peer_prompt_includes_context(monkeypatch):
    seen = {}

    def fake_run(prompt, **kw):
        seen["prompt"] = prompt
        return {"ok": True, "exit_code": 0, "answer": "done", "stderr": "", "timed_out": False,
                "error": None, "artifacts": [], "backend": "opencode-docker", "model": "vllm/m"}

    monkeypatch.setattr(ocp, "run_opencode", fake_run)
    result = ocp.run_opencode_code_peer(
        "plot the data", evidence=None, state={"analysis_results": {"mean": 3}},
    )
    assert result["answer"] == "done"
    assert "plot the data" in seen["prompt"]
    assert '"mean": 3' in seen["prompt"]


def test_supervisor_code_fn_dispatches_to_opencode(monkeypatch):
    from agent_runtime.supervisor.graph import default_code_fn

    monkeypatch.setenv("AGENT_CODE_PEER", "opencode")
    sentinel = {"answer": "via opencode", "tool_calls": [], "tool_results": []}
    monkeypatch.setattr(ocp, "run_opencode_code_peer", lambda *a, **k: sentinel)
    fn = default_code_fn(input_file_ids=["f1"])
    assert fn("q", [], {"thread_id": "t"}) is sentinel
