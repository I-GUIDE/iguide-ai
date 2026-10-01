"""Tools registered somewhere but missing from a set that needs them.

This class of bug has bitten this repo repeatedly and is always silent: the peer simply does
not have the tool, the model falls back to something worse, and no error is raised. Two rounds
of tool-description tuning once failed to fix a tool that was not registered at all.
"""
from __future__ import annotations

import pytest


# --- skill tools reach a peer built from preloaded_tools -----------------------------------
#
# build_agent_executor accepted skill_roots but forwarded it ONLY into _collect_tools, which is
# skipped when preloaded_tools is supplied. default_analyze_fn passes both, so the analyze peer
# silently had neither list_available_skills nor load_skill. The code peer worked only because
# it calls make_skill_tools itself.

def _bound_tools(monkeypatch, **kwargs):
    seen = {}
    import langchain.agents as la

    def _spy(model=None, tools=None, **kw):
        seen["names"] = [str(getattr(t, "name", "")) for t in (tools or [])]
        return object()

    monkeypatch.setattr(la, "create_agent", _spy)
    from agent_runtime.executor_factory import build_agent_executor

    build_agent_executor(llm=object(), **kwargs)
    return seen["names"]


@pytest.fixture()
def a_tool():
    from langchain_core.tools import tool

    @tool
    def preloaded(x: str) -> str:
        """A preloaded tool."""
        return x

    return preloaded


def test_a_preloaded_peer_still_gets_the_skill_tools(monkeypatch, a_tool):
    names = _bound_tools(monkeypatch, preloaded_tools=[a_tool], skill_roots=None)
    assert "preloaded" in names
    assert "list_available_skills" in names and "load_skill" in names


def test_the_skill_tools_are_not_added_twice(monkeypatch, a_tool):
    """The code peer already calls make_skill_tools itself; the dedup makes this a no-op."""
    from agent_runtime.skills import make_skill_tools

    names = _bound_tools(monkeypatch,
                         preloaded_tools=[a_tool, *make_skill_tools(skill_roots=None)],
                         skill_roots=None)
    assert names.count("load_skill") == 1


# --- add_map_layer exists with nothing attached --------------------------------------------

def test_add_map_layer_is_bound_on_a_no_upload_turn():
    """"show me Champaign County on the map" needs it, and the peer had no way to deliver.

    Asserted through the shared constant rather than the filter's source text: the filter used to
    name one tool inline, and pinning that string made the test fail when a second delivery route
    was added, which is the opposite of what it is guarding.
    """
    import inspect

    from agent_runtime.supervisor import graph as g

    assert "add_map_layer" in g._MAP_DELIVERY_TOOLS
    for fn in (g.default_analyze_fn, g.default_code_fn):
        src = inspect.getsource(fn)
        assert "_MAP_DELIVERY_TOOLS" in src, fn.__name__
        assert "if not input_file_ids:" in src, fn.__name__


def test_a_composed_raster_has_a_delivery_route_on_a_no_upload_turn():
    """A drawn region has no upload, and clustering its embedding in code yields PIXELS, not
    geometry. add_map_layer refuses images by name, so without this the work had nowhere to go."""
    from agent_runtime.supervisor import graph as g

    assert "add_raster_layer" in g._MAP_DELIVERY_TOOLS


def test_only_add_map_layer_is_hoisted_not_the_whole_factory():
    """The other five need a vector file, and render_map_image is the static-PNG route the
    map observation exists to discourage — the whole factory costs ~1,479 tokens vs ~299."""
    from agent_runtime.langchain_geo_tools import make_langchain_geo_tools

    all_names = {str(getattr(t, "name", "")) for t in
                 make_langchain_geo_tools(default_input_file_ids=None)}
    assert "add_map_layer" in all_names
    assert {"inspect_vector", "render_map_image", "vector_to_geojson"} <= all_names


def test_the_corrective_map_retry_is_still_gated_on_an_upload():
    """Deliberately NOT un-gated: three changes already make it fire more readily and
    _WANTS_MAP_RE matches a bare "on the map", so an ordinary follow-up would be told the map
    got nothing and redundantly re-add a layer already on screen."""
    import inspect

    from agent_runtime.supervisor import graph as g

    src = inspect.getsource(g.default_analyze_fn)
    assert "if input_file_ids and wants_map and not on_map" in src


# --- enabledSearchMethods reaches the analyze peer -----------------------------------------

def test_the_analyze_peer_accepts_an_allowlist():
    import inspect

    from agent_runtime.supervisor import graph as g

    assert "enabled_search_methods" in inspect.signature(g.default_analyze_fn).parameters
    src = inspect.getsource(g.default_analyze_fn)
    assert "enabled_search_methods=enabled_search_methods" in src


def test_orchestration_passes_the_allowlist_to_both_arms():
    import inspect

    from agent_runtime.supervisor import orchestration

    src = inspect.getsource(orchestration)
    assert src.count("enabled_search_methods=cfg.enabled_search_methods") >= 2


# --- the capability answer covers what the peers actually bind -----------------------------

def test_the_capability_inventory_covers_the_embedding_surface():
    """It omitted BOTH rs-embed factories plus admin_boundary, QGIS and code execution — five
    registries every peer binds — so "what can you do" never mentioned satellite embeddings."""
    from agent_runtime.capabilities import collect_capability_inventory

    inv = collect_capability_inventory(include_mcp_tools=False)
    entries = inv if isinstance(inv, list) else inv.get("tools", [])
    names = {e["name"] for e in entries}
    for want in ("embed_region", "embed_zones", "fit_zone_model", "predict_from_package",
                 "admin_boundary", "execute_code"):
        assert want in names, f"{want} is bound by every peer but invisible to the user"


def test_a_broken_registry_does_not_take_the_inventory_down(monkeypatch):
    """Behaviour, not source shape: one unconstructible registry must not lose the others.

    This used to count `except Exception:` blocks in collect_capability_inventory — one per
    hand-written registry. Those are gone: registries are discovered by convention, so a hand
    list can no longer be wrong (it was wrong twice, and an audit right after fixing it found
    two more factories still missing). Isolation now lives in the discovery loop.
    """
    import agent_runtime.capabilities as cap

    def _boom(**kwargs):
        raise RuntimeError("this registry cannot be built")

    real = cap._discover_registry_factories
    monkeypatch.setattr(cap, "_discover_registry_factories",
                        lambda: [("make_broken_tools", _boom), *real()])

    inv = cap.collect_capability_inventory(include_mcp_tools=False)
    names = {t["name"] for t in inv["tools"]}
    assert len(names) > 40, "a broken registry must not empty the inventory"
    assert "embed_zones" in names


def test_a_new_registry_needs_no_edit_to_this_module():
    """The point of discovery: adding a make_*tools factory is enough."""
    from agent_runtime.capabilities import _discover_registry_factories

    found = {n for n, _ in _discover_registry_factories()}
    # the ones a hand-written list had missed, in two separate rounds
    for want in ("make_rs_embed_tools", "make_rs_embed_zonal_tools",
                 "make_langchain_file_tools", "make_quality_tools"):
        assert want in found, f"{want} must be discovered, not listed"


def test_only_read_only_listers_are_handed_to_the_capability_agent():
    """Generic by prefix, so a new list_* tool is offered without an edit — and nothing that
    could act is included."""
    import agent_runtime.capabilities as cap
    import agent_runtime.executor_factory as ef

    captured = {}

    def _build(**kw):
        captured["tools"] = [str(getattr(t, "name", ""))
                             for t in (kw.get("preloaded_tools") or [])]
        return object()

    orig_build, orig_invoke, orig_llm = (ef.build_agent_executor,
                                         ef.invoke_agent_with_payload_fallback,
                                         ef.build_default_llm)
    ef.build_agent_executor = _build
    ef.invoke_agent_with_payload_fallback = lambda *a, **k: {"messages": []}
    ef.build_default_llm = lambda: object()
    try:
        cap.describe_capabilities(query="which models do you have?", include_mcp_tools=False)
    finally:
        (ef.build_agent_executor, ef.invoke_agent_with_payload_fallback,
         ef.build_default_llm) = orig_build, orig_invoke, orig_llm

    names = captured["tools"]
    assert "list_my_capabilities" in names
    assert "list_embedding_models" in names, "the real model names must be reachable"
    non_listers = [n for n in names if not n.startswith("list_")]
    assert non_listers == [], f"only read-only listers belong here, got {non_listers}"


# --- the upload gate -----------------------------------------------------------------------
#
# `if input_file_ids:` binds ~10,432 schema tokens on the presence of ANY file. Measured, only
# ~857 of those (8%) can be safely excluded from the file NAME: overlay, aggregate, temporal and
# spatial-stats tools cannot be ruled out by extension, because buffer_layer on a point CSV is
# legitimate, spatial weights can be built from coordinates, and a GeoJSON can carry a date
# column. Ruling them out would need the file READ, and misclassifying recreates the absent-tool
# bug behind half the selection failures in this repo — which a per-call filter cannot undo,
# since narrowing raises ValueError when it tries to widen.
#
# So this is scoped to tools that need a vector FILE, and it fails open everywhere else.

def _record(monkeypatch, mapping):
    import agent_runtime.file_store as fs
    monkeypatch.setattr(fs, "get_file_record",
                        lambda fid: ({"filename": mapping[fid]} if fid in mapping else None))


def test_a_csv_upload_does_not_offer_vector_file_tools(monkeypatch):
    from agent_runtime.supervisor.graph import _uploads_are_tabular_only

    _record(monkeypatch, {"f1": "incidents.csv"})
    assert _uploads_are_tabular_only(["f1"]) is True


def test_a_geojson_upload_still_offers_everything(monkeypatch):
    from agent_runtime.supervisor.graph import _uploads_are_tabular_only

    _record(monkeypatch, {"f1": "tracts.geojson"})
    assert _uploads_are_tabular_only(["f1"]) is False


def test_a_mixed_upload_offers_everything(monkeypatch):
    """One vector file among tabular ones means the vector tools are needed."""
    from agent_runtime.supervisor.graph import _uploads_are_tabular_only

    _record(monkeypatch, {"f1": "incidents.csv", "f2": "tracts.zip"})
    assert _uploads_are_tabular_only(["f1", "f2"]) is False


@pytest.mark.parametrize("name", ["data.parquet", "boundary.shp", "layer.gpkg", "odd.bin", ""])
def test_anything_unrecognised_fails_open(monkeypatch, name):
    """The expensive mistake is withholding a tool, so uncertainty binds everything."""
    from agent_runtime.supervisor.graph import _uploads_are_tabular_only

    _record(monkeypatch, {"f1": name})
    assert _uploads_are_tabular_only(["f1"]) is False


def test_it_never_decides_by_crashing(monkeypatch):
    import agent_runtime.file_store as fs

    from agent_runtime.supervisor.graph import _uploads_are_tabular_only

    def _boom(fid):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(fs, "get_file_record", _boom)
    assert _uploads_are_tabular_only(["f1"]) is False


def test_map_layer_tools_are_never_withheld():
    """add_map_layer and render_map_image work from geometry the peer already holds, so they
    must stay bound whatever was uploaded."""
    from agent_runtime.supervisor.graph import _VECTOR_FILE_TOOLS

    assert "add_map_layer" not in _VECTOR_FILE_TOOLS
    assert "render_map_image" not in _VECTOR_FILE_TOOLS


def test_both_peers_apply_the_refinement():
    import inspect

    from agent_runtime.supervisor import graph as g

    for fn in (g.default_analyze_fn, g.default_code_fn):
        assert "_uploads_are_tabular_only(input_file_ids)" in inspect.getsource(fn), fn.__name__


# --- MCP web tools are deliberately not bound ---------------------------------------------

def test_the_dominated_mcp_web_tools_are_not_bound(monkeypatch):
    """Both are DuckDuckGo searches like web_search, but with a canned query, NO fetch
    counterpart, no per-turn web budget, and they raise instead of returning an error envelope.
    Observed live: the agent chose mcp_search_external_resources over web_search/web_fetch for a
    "find datasets about X" phrasing — the weaker path, and one that structurally cannot read a
    page (SEARCH_AGENT_PROMPT rule 10).
    """
    monkeypatch.delenv("AGENT_MCP_UNBIND", raising=False)
    from agent_runtime import langchain_mcp_tools as m

    assert m._is_unbound_mcp_tool("search_external_resources")
    assert m._is_unbound_mcp_tool("web_search_geo_links")
    # everything else still binds
    for keep in ("describe_image", "run_notebook_workflow", "load_chicago_crime_data"):
        assert not m._is_unbound_mcp_tool(keep)


def test_the_unbind_list_is_overridable(monkeypatch):
    from agent_runtime import langchain_mcp_tools as m

    monkeypatch.setenv("AGENT_MCP_UNBIND", "none")
    assert not m._is_unbound_mcp_tool("search_external_resources"), "operators can bind them back"

    monkeypatch.setenv("AGENT_MCP_UNBIND", "describe_image")
    assert m._is_unbound_mcp_tool("describe_image")
    assert not m._is_unbound_mcp_tool("search_external_resources")

    # the mcp_ prefix is tolerated, since that is the name the agent sees
    monkeypatch.setenv("AGENT_MCP_UNBIND", "mcp_describe_image")
    assert m._is_unbound_mcp_tool("describe_image")


def test_both_build_paths_apply_the_filter():
    """The remote path and the local-import fallback each build tools; filtering only the
    remote one would quietly re-bind them whenever the MCP server is unreachable."""
    import inspect
    from agent_runtime import langchain_mcp_tools as m

    remote = inspect.getsource(m._make_remote_mcp_tools)
    factory = inspect.getsource(m.make_langchain_mcp_tools)
    assert "_is_unbound_mcp_tool(remote_name)" in remote
    assert "_is_unbound_mcp_tool(func.__name__)" in factory


def test_both_routes_to_the_rs_operations_exist():
    """The one-shot tools AND the package they can be composed from.

    #21 removed segment_region, embedding_change and predict_for_region on the argument that the
    embedding is the primitive. The argument is good and the composition works — verified live —
    but it costs resolution: the exported grid is decimated to a cell budget, stride 2 at the
    default footprint, where segment_region clustered the native grid server-side. So both are
    kept: the tool for the common case, the package for what the tool cannot express.

    Change and prediction never needed the removal to compose. Both were already multi-step
    before it, driven by how the question is phrased rather than by the router prompt — the
    pre-#21 router described one-shot tools and the agent composed anyway."""
    from agent_runtime.rs_embed_tools import make_rs_embed_tools

    names = {str(getattr(t, "name", "")) for t in make_rs_embed_tools()}
    assert {"segment_region", "embedding_change", "predict_for_region"} <= names, (
        "the one-shot route is the higher-resolution one and must stay")
    assert {"embed_region", "predict_from_package", "list_embedding_packages"} <= names, (
        "the package route must stay too — it is what expresses the uncommon cases")


def test_embed_region_says_how_to_compose_from_its_package():
    """Without this the plumbing works and the model has no reason to know it does: nothing else
    tells it that a TOOL's file_id can be staged into execute_code, or what is inside the .npz."""
    from agent_runtime.rs_embed_tools import make_rs_embed_tools

    doc = next(t for t in make_rs_embed_tools()
               if str(getattr(t, "name", "")) == "embed_region").description or ""
    for want in ("input_files", "grid__", "pooled__", "grid_stride", "add_raster_layer"):
        assert want in doc, f"embed_region's description never mentions {want}"


def test_a_lost_embedding_package_is_reported_not_omitted(monkeypatch):
    """The package is the only input to every composed operation, so losing it removes
    segmentation, change and prediction for the turn. It used to vanish from the result: the key
    was simply absent, which reads the same as "no package was made" and lets the answer go on to
    describe clustering that never happened."""
    import json as _json

    from agent_runtime import rs_embed_tools as rt

    monkeypatch.setattr(rt, "_svc", lambda path, payload=None, **kw: (
        {"models": [{"id": "gse"}]} if path == "/api/models" else {
            "results": [{"model": "gse", "ok": True, "type": "precomputed", "dim": 64,
                         "grid_hw": [8, 8], "norm": 1.0, "image": ""}],
            "package": {"models": ["gse"], "grids_saved": ["gse"]},
            "download_url": "/api/download/pkg.npz"}))
    monkeypatch.setattr(rt, "_save_png", lambda *a, **k: None)
    monkeypatch.setattr(rt, "_fetch_package", lambda *a, **k: None)   # the fetch fails

    tool = next(t for t in rt.make_rs_embed_tools()
                if str(getattr(t, "name", "")) == "embed_region")
    out = _json.loads(tool.invoke({"bbox": [-88.3, 40.05, -88.2, 40.12], "models": ["gse"]}))

    pkg = out.get("embedding_package") or {}
    assert "file_id" not in pkg, "nothing was fetched, so there is no file_id to offer"
    assert pkg.get("unavailable"), "a lost package must SAY so, not just omit the key"
    assert "compose" in pkg["unavailable"]
    assert "not available this turn" in (pkg.get("consequence") or "")


def test_embed_region_states_the_npz_contract_correctly(monkeypatch):
    """The contract is what composed code is written against, so a wrong claim in it costs a
    sandbox run or, worse, a silently mis-scaled result. `meta` is a 0-d ndarray and the stride
    fields live per-entry under meta["models"] — asserted because the first version of this
    docstring got both wrong."""
    from agent_runtime.rs_embed_tools import make_rs_embed_tools

    doc = next(t for t in make_rs_embed_tools()
               if str(getattr(t, "name", "")) == "embed_region").description or ""
    assert 'json.loads(str(z["meta"]))' in doc, "the naive json.loads(z['meta']) raises TypeError"
    assert 'meta["models"]' in doc, "grid_stride is per-model, not a top-level meta key"
    assert "grid_saved_hw" in doc
    # And the raster route has to say the PNG is the pixels, not a figure: a matplotlib figure's
    # axes and margins silently misregister the layer against its bounds.
    assert "fromarray" in doc or "imsave" in doc
    assert "matplotlib" in doc
