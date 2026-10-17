"""A map layer whose attributes hold NaN or Infinity still reaches the map.

Observed 2026-10-17 (round r59 of the GIS assessment log): the Loop routing answer was right,
5.19 min, but the route never drew. execute_code wrote the route GeoJSON with Python's json,
and its `streets` list held five bare NaN (pandas NaN for unnamed ways). add_map_layer re-wrote
it through GDAL, which nulls a NaN scalar but copies a list through as raw JSON, so the NaN
survived. The download route served it with a 200, the browser's JSON.parse threw "Unexpected
token 'N'", and the answer told the user the route was on the map.

Every check here parses what a browser would receive with a STRICT parser, because Python's own
json.loads accepts NaN and would pass the very file that failed.
"""
from __future__ import annotations

import io
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from agent_runtime.strict_json import strict_dumps, strict_json_bytes


def _strict_loads(raw):
    """json.loads that refuses NaN/Infinity, as JSON.parse does."""
    def refuse(token):
        raise ValueError(f"non-finite token {token} is not JSON")
    return json.loads(raw, parse_constant=refuse)


def _route_fc() -> dict:
    """The r59 shape: a route line whose `streets` list names unnamed ways as NaN."""
    return {"type": "FeatureCollection", "features": [{
        "type": "Feature",
        "properties": {"time_min": 5.19, "length_km": 4.21, "grade": float("nan"),
                       "streets": ["W Van Buren St", float("nan"), "S Wells St",
                                   float("nan"), "E Wacker Dr"]},
        "geometry": {"type": "LineString",
                     "coordinates": [[-87.645, 41.866], [-87.633, 41.876], [-87.618, 41.889]]}}]}


def _python_json_text(obj) -> str:
    """What execute_code's json.dump writes: Python's default, allow_nan=True."""
    text = json.dumps(obj)
    assert "NaN" in text            # the fixture really is the broken kind
    return text


# --- the helper ---------------------------------------------------------------------------

def test_non_finite_values_become_null_wherever_they_sit():
    raw = (b'{"a": NaN, "b": [1, Infinity, -Infinity, "x"], "c": {"d": [NaN]}, '
           b'"e": 1e999, "f": 2.5}')
    fixed, count = strict_json_bytes(raw)
    assert count == 5
    assert _strict_loads(fixed) == {"a": None, "b": [1, None, None, "x"], "c": {"d": [None]},
                                    "e": None, "f": 2.5}


def test_strict_json_is_left_alone():
    # The word inside a string is not the token: nothing to fix, so the original is served.
    assert strict_json_bytes(b'{"name": "NaNcy Infinity Pool", "v": null}') == (None, 0)
    assert strict_json_bytes(b'{"v": 1.5}') == (None, 0)


def test_text_that_is_not_json_is_served_as_it_is():
    assert strict_json_bytes(b"NaN, but not JSON {") == (None, 0)


def test_strict_dumps_writes_null_for_non_finite_floats():
    payload = strict_dumps({"geojson": _route_fc()}, ensure_ascii=True, default=str)
    props = _strict_loads(payload)["geojson"]["features"][0]["properties"]
    assert props["grade"] is None
    assert props["streets"] == ["W Van Buren St", None, "S Wells St", None, "E Wacker Dr"]


# --- the two places a browser receives a layer ----------------------------------------------

@pytest.fixture()
def store(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    monkeypatch.delenv("AGENT_CHAT_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_PUBLIC_BASE_URL", raising=False)
    return tmp_path / "store"


@pytest.fixture()
def client(store):
    import api.server as srv
    return srv.app.test_client()


def _upload(name: str, data: bytes) -> str:
    from werkzeug.datastructures import FileStorage
    from agent_runtime.file_store import save_uploaded_file
    return save_uploaded_file(FileStorage(stream=io.BytesIO(data), filename=name))["file_id"]


def _node_parses(raw: bytes) -> None:
    """The browser's own parser, where node is installed: JSON.parse must not throw."""
    node = shutil.which("node")
    if not node:
        return
    proc = subprocess.run(
        [node, "-e", "let s='';process.stdin.on('data',d=>s+=d).on('end',()=>{"
                     "const j=JSON.parse(s);process.stdout.write(String(j.features.length))})"],
        input=raw, capture_output=True, timeout=30)
    assert proc.returncode == 0, proc.stderr.decode()
    assert proc.stdout.decode() == "1"


def test_a_raw_nan_geojson_is_served_strict_and_stored_unchanged(client, store):
    """A code peer's own file: layers_for_artifacts points the map straight at it, so no tool
    ever re-writes it. The download route is the one place every such file passes."""
    original = _python_json_text(_route_fc()).encode()
    fid = _upload("route.geojson", original)

    r = client.get(f"/agent/files/{fid}/download")
    assert r.status_code == 200
    served = _strict_loads(r.data)
    props = served["features"][0]["properties"]
    assert props["streets"] == ["W Van Buren St", None, "S Wells St", None, "E Wacker Dr"]
    assert props["grade"] is None and props["time_min"] == 5.19
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
    assert "attachment" in (r.headers.get("Content-Disposition") or "").lower()
    _node_parses(r.data)

    from agent_runtime.file_store import resolve_file_id
    assert Path(resolve_file_id(fid)).read_bytes() == original   # the record is not edited


def test_a_strict_geojson_is_served_byte_for_byte(client, store):
    original = json.dumps(_route_fc() | {"features": []}).encode()
    fid = _upload("empty.geojson", original)
    assert client.get(f"/agent/files/{fid}/download").data == original


def test_add_map_layer_writes_and_serves_strict_json(client, store, tmp_path):
    pytest.importorskip("geopandas")
    pytest.importorskip("pyogrio")
    from agent_runtime.file_store import resolve_file_id
    from agent_runtime.langchain_geo_tools import make_langchain_geo_tools

    src = tmp_path / "route.geojson"
    src.write_text(_python_json_text(_route_fc()))
    fid = _upload("route.geojson", src.read_bytes())
    tool = {t.name: t for t in make_langchain_geo_tools()}["add_map_layer"]
    res = json.loads(tool.func(file_id=fid, render="auto", name="fastest route"))
    assert res["ok"], res
    assert res["map_layer"]["render"] == "shapes"
    assert "null" in res["nonfinite_as_null"]       # the model is told what changed

    # The stored copy is strict: GDAL alone kept the NaN inside the list.
    stored = Path(resolve_file_id(res["file_id"])).read_bytes()
    props = _strict_loads(stored)["features"][0]["properties"]
    assert None in props["streets"] and "W Van Buren St" in props["streets"]

    # And so is what the map fetches from the layer's url.
    r = client.get(res["map_layer"]["url"])
    assert r.status_code == 200
    assert _strict_loads(r.data)["features"][0]["properties"]["time_min"] == 5.19
    _node_parses(r.data)


def test_an_inline_layer_event_is_strict_json(store):
    """A map_layer event can carry its GeoJSON inline; one NaN made the whole event unparseable."""
    import api.server as srv
    event = srv._sse_event("map_layer", {"kind": "map_layer", "id": "agent-route",
                                         "geojson": _route_fc()})
    data = event.split("data: ", 1)[1].strip()
    layer = _strict_loads(data)
    assert layer["geojson"]["features"][0]["properties"]["streets"][1] is None
