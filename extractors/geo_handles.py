"""Geodata file-handle adapter — pass (Geo)DataFrames between tool calls as file_ids.

Spatial functions can't exchange in-memory GeoDataFrames through JSON tool calls. This
adapter persists each (Geo)DataFrame to a file (GeoParquet, pickle fallback) registered
in the agent file store, and passes the **file_id** instead. ``make_file_handle_tool``
wraps an extracted function into a tool whose (Geo)DataFrame parameters become file_ids
(inferred from type hints), scalars pass through, a (Geo)DataFrame return is written to a
new file_id, and a plot (None return) is captured to a PNG file_id.

This makes the extracted spatial functions chainable as EXECUTED tool steps, and every
intermediate file_id is a persisted artifact → a data-lineage trail.
"""

from __future__ import annotations

import ast
import inspect
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from agent_runtime.file_store import create_output_file_from_path, resolve_file_id


def _is_frame_type(ann: Any) -> bool:
    # stringized annotations (PEP 563 / `from __future__ import annotations`)
    if isinstance(ann, str):
        return ann.split("[")[0].split(".")[-1] in {"DataFrame", "GeoDataFrame"}
    try:
        import pandas as pd
        return isinstance(ann, type) and issubclass(ann, pd.DataFrame)
    except Exception:
        return False


def write_geodata(obj: Any, name: str) -> str:
    """Persist a (Geo)DataFrame to a file and return its file_id (GeoParquet; pickle fallback)."""
    tmp = Path(tempfile.mkdtemp(prefix="geohandle_"))
    parquet = tmp / f"{name}.parquet"
    try:
        obj.to_parquet(parquet)          # GeoParquet for GeoDataFrame, parquet for DataFrame
        return create_output_file_from_path(parquet, filename=parquet.name)["file_id"]
    except Exception:
        pkl = tmp / f"{name}.pkl"
        obj.to_pickle(pkl)
        return create_output_file_from_path(pkl, filename=pkl.name)["file_id"]


def read_geodata(file_id: str) -> Any:
    """Load a (Geo)DataFrame from a file_id."""
    path = resolve_file_id(file_id)
    if str(path).endswith(".pkl"):
        import pandas as pd
        return pd.read_pickle(path)
    try:
        import geopandas as gpd
        return gpd.read_parquet(path)    # geometry-aware
    except Exception:
        import pandas as pd
        return pd.read_parquet(path)


def capture_current_fig(name: str) -> str | None:
    """Save the current matplotlib figure to a PNG file_id (for plot tools returning None)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not plt.get_fignums():
        return None
    tmp = Path(tempfile.mkdtemp(prefix="geohandle_")) / name
    plt.gcf().savefig(tmp, dpi=130, bbox_inches="tight")
    plt.close("all")
    return create_output_file_from_path(tmp, filename=tmp.name)["file_id"]


def make_file_handle_tool(func: Callable) -> Callable[..., Dict[str, Any]]:
    """Wrap ``func`` so (Geo)DataFrame params/returns travel as file_ids.

    Type-hint driven: a param annotated as a (Geo)DataFrame expects a file_id (loaded
    via read_geodata); other params pass through. A (Geo)DataFrame return is written to a
    new file_id; a ``None`` return with an open figure is captured to a PNG file_id.
    """
    sig = inspect.signature(func)
    fname = getattr(func, "__name__", "tool")

    def tool(**kwargs: Any) -> Dict[str, Any]:
        call_args: Dict[str, Any] = {}
        inputs: Dict[str, str] = {}
        for pname, p in sig.parameters.items():
            if pname not in kwargs:
                continue
            if _is_frame_type(p.annotation):
                inputs[pname] = kwargs[pname]
                call_args[pname] = read_geodata(kwargs[pname])   # file_id -> frame
            else:
                call_args[pname] = kwargs[pname]                 # scalar passthrough
        result = func(**call_args)
        out: Dict[str, Any] = {"tool": fname, "inputs": inputs}
        try:
            import pandas as pd
            if isinstance(result, pd.DataFrame):
                fid = write_geodata(result, f"{fname}_out")
                out.update(file_id=fid, rows=int(len(result)), columns=list(map(str, result.columns)))
                return out
        except Exception:
            pass
        if result is None:
            png = capture_current_fig(f"{fname}.png")
            out.update(png_file_id=png) if png else out.update(ok=True)
            return out
        out.update(result=result)
        return out

    tool.__name__ = fname
    tool.__doc__ = (func.__doc__ or "").strip()
    return tool


# --------------------------------------------------------------------------- #
# Analyze-peer tools: run extracted KB spatial functions + general GIS ops,
# all passing (Geo)DataFrames by file_id (so the GIS runs as executed tool steps).
# --------------------------------------------------------------------------- #
def _strip_tool_decorators(code: str) -> str:
    return "\n".join(l for l in code.splitlines() if l.strip() != "@tool")


_GEO_PRELUDE = ("import pandas as pd\nimport geopandas as gpd\n"
                "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n")


def _extract_function_source(code: str, fname: str) -> Optional[str]:
    """Return ONLY the source of top-level function ``fname`` (decorators stripped),
    isolated from surrounding notebook/agent code so we never exec agent-setup cells.
    A bare element_id resolves to all blocks concatenated; tolerate an unparseable
    bundle by scanning per-block segments (split on the '# --- doc_id ---' headers)."""
    def _from(segment: str) -> Optional[str]:
        try:
            tree = ast.parse(segment)
        except SyntaxError:
            return None
        for node in tree.body:  # top-level defs only
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fname:
                seg = ast.get_source_segment(segment, node)
                if seg:
                    return "\n".join(l for l in seg.splitlines() if l.strip() != "@tool")
        return None

    whole = _from(code)
    if whole:
        return whole
    for seg in re.split(r"^# --- .*? ---$", code, flags=re.M):
        found = _from(seg)
        if found:
            return found
    return None



def _nearest_blocks(*terms: str, limit: int = 5) -> dict:
    """Real KB blocks resembling a doc_id that does not exist.

    A bare "block not found" is a dead end: the model had guessed a plausible-sounding id
    (observed: load_chicago_crime_data, in 3 of 3 runs) and, told only that it was wrong,
    burned a turn before falling back to writing code. Naming the nearest REAL blocks turns
    the failure into a next step.
    """
    try:
        from rag_pipeline.search.agent_kb import agent_kb_search

        query = " ".join(t for t in terms if t)
        res = agent_kb_search(query, size=limit) or {}
        cands = [{"doc_id": d.get("doc_id"), "title": d.get("title")}
                 for d in (res.get("documents") or [])[:limit] if d.get("doc_id")]
        if not cands:
            return {"hint": "No similar KB block exists. Search first with agent_kb_search, "
                            "or do the work with execute_code instead of guessing an id."}
        return {"candidates": cands,
                "hint": "These real blocks are the closest matches — pass one of THEIR doc_ids, "
                        "or use execute_code if none fit."}
    except Exception:
        return {"hint": "Search agent_kb_search for a real block id; do not guess one."}


def kb_run_geofunction(doc_id: str, function_name: str, args_json: str = "{}") -> str:
    """Execute an extracted spatial function from a KB block via file handles.

    Loads the block's code (get_kb_block), isolates `function_name` via AST (ignoring
    surrounding notebook/agent code), defines it, and calls it with `args_json` (a JSON
    object). (Geo)DataFrame parameters must be passed as file_ids (from a prior step);
    the result is written to a new file_id (or a PNG file_id for a plot)."""
    import json
    from rag_pipeline.search.agent_kb import get_kb_block
    blk = get_kb_block(doc_id)
    if not blk.get("found"):
        return json.dumps({"error": f"block not found: {doc_id}",
                            **_nearest_blocks(doc_id, function_name)})
    code = ((blk.get("source") or {}).get("extracted") or {}).get("block", {}).get("code", "")
    func_src = _extract_function_source(code, function_name)
    if not func_src:
        # Say what the block DOES define, so the caller can pick instead of guessing again.
        import re as _re
        defined = _re.findall(r"^\s*def\s+([A-Za-z_]\w*)", code or "", _re.M)
        return json.dumps({"error": f"{function_name} not defined in {doc_id}",
                            "functions_in_block": defined[:20]})
    ns: dict = {}
    try:
        exec(_GEO_PRELUDE + func_src, ns)
        fn = ns.get(function_name)
        if not callable(fn):
            return json.dumps({"error": f"{function_name} not callable in {doc_id}"})
        out = make_file_handle_tool(fn)(**json.loads(args_json or "{}"))
        return json.dumps(out, default=str)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def kb_select_rows(df_file_id: str, column: str, values_csv: str) -> str:
    """Filter a (Geo)DataFrame file to rows whose `column` is in the comma-separated
    `values_csv` (case-insensitive set membership). Returns JSON with the new file_id."""
    import json
    import geopandas as gpd  # noqa: F401

    def _select(df, column, values_csv):  # local; annotations resolved at wrap time
        vals = {v.strip().upper() for v in values_csv.split(",")}
        return df[df[column].astype(str).str.upper().isin(vals)]
    _select.__annotations__ = {"df": "DataFrame", "column": str, "values_csv": str, "return": "DataFrame"}
    _select.__name__ = "select_rows"
    try:
        return json.dumps(make_file_handle_tool(_select)(df=df_file_id, column=column, values_csv=values_csv), default=str)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})



def _save_figure(fig, filename: str) -> dict:
    """Persist a matplotlib figure to the agent file store and return its record."""
    import tempfile, os
    from agent_runtime.file_store import create_output_file_from_path

    fd, path = tempfile.mkstemp(prefix="img_", suffix=".png")
    os.close(fd)
    try:
        fig.savefig(path, bbox_inches="tight", dpi=150)
        rec = create_output_file_from_path(path, filename=filename)
        return {"file_id": rec["file_id"], "filename": rec.get("filename"),
                "download_url": rec.get("download_url"), "size_bytes": rec.get("size_bytes")}
    finally:
        try: os.remove(path)
        except OSError: pass


def _read_any_vector(ref: str):
    """Load a points/polygons file for the image tools.

    These handles went through a parquet-only reader, so a .geojson the agent had just
    produced failed with "Parquet magic bytes not found" — observed when an on-the-map
    request tried the image tool first. Read whatever the file actually is.
    """
    from agent_runtime.langchain_geo_tools import _resolve, read_vector

    path, _rec = _resolve(str(ref))
    return read_vector(str(path))


def heatmap_image(points_file_id: str, title: str = "Density heat map",
                  name: str = "heatmap") -> str:
    """Draw a hexbin point-density heat map as a STATIC PNG PICTURE from a points file.

    An image to download or print — it cannot be panned, zoomed or clicked. To put a heat
    map on the user's interactive map instead, use add_map_layer(render="heatmap").
    Reads GeoJSON / shapefile / GeoPackage / GeoParquet / CSV-with-coordinates.
    Returns JSON with the PNG file_id.
    """
    import json

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gdf = _read_any_vector(points_file_id)
        pts = gdf[gdf.geometry.geom_type.isin(["Point", "MultiPoint"])] if hasattr(gdf, "geometry") else gdf
        if hasattr(pts, "geometry") and len(pts):
            xs, ys = pts.geometry.x, pts.geometry.y
        else:
            return json.dumps({"error": "no point geometry to build a density map from"})
        fig, ax = plt.subplots(figsize=(9, 9))
        hb = ax.hexbin(xs, ys, gridsize=40, cmap="inferno", mincnt=1)
        fig.colorbar(hb, ax=ax, label="incidents per cell")
        ax.set_title(f"{title} (n={len(pts):,})")
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        rec = _save_figure(fig, f"{re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_').lower() or 'heatmap'}.png")
        plt.close(fig)
        return json.dumps({"ok": True, "features": int(len(pts)), **rec}, default=str)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def choropleth_image(gdf_file_id: str, column: str, title: str = "Choropleth map",
                      scheme: str = "") -> str:
    """Draw a choropleth as a STATIC PNG PICTURE from a polygon file_id, colored by `column`
    (e.g. a count produced by spatial_join_and_count). Robust to library versions —
    use this instead of an extracted plot function that may target an old matplotlib. This is
    an image to download or print; for a choropleth the user can explore on their interactive
    map, use add_map_layer(render="choropleth", column=...) instead.
    Optional `scheme` (e.g. 'NaturalBreaks'=Jenks, 'Quantile') needs mapclassify; it
    silently falls back to a continuous ramp if unavailable. Returns a PNG file_id."""
    import json

    def _choro(gdf, column, title, scheme):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 9))
        kw = dict(column=column, legend=True, cmap="OrRd", edgecolor="0.8", linewidth=0.3)
        if scheme:
            try:
                import mapclassify  # noqa: F401
                kw["scheme"] = scheme
            except Exception:
                pass
        gdf.plot(ax=ax, **kw)
        ax.set_title(title)
        ax.set_axis_off()
    _choro.__annotations__ = {"gdf": "GeoDataFrame", "column": str, "title": str, "scheme": str, "return": type(None)}
    _choro.__name__ = "choropleth_map"
    try:
        return json.dumps(make_file_handle_tool(_choro)(
            gdf=gdf_file_id, column=column, title=title, scheme=scheme), default=str)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def make_geo_analysis_tools() -> list:
    """LangChain StructuredTools for the analyze peer: run extracted KB spatial functions
    and chain GIS ops by file_id (heat map, set filter)."""
    from langchain_core.tools import StructuredTool
    return [
        StructuredTool.from_function(
            func=kb_run_geofunction, name="kb_run_geofunction",
            description=("Execute an extracted spatial function from a knowledge-base block by "
                         "doc_id + function_name, passing (Geo)DataFrames as file_ids (args_json). "
                         "Returns the produced file_id. Chain steps by feeding one step's file_id "
                         "into the next. E.g. load_chicago_crime_data → file_id → spatial_join_and_count."),
            metadata={"category": "computation"}),
        StructuredTool.from_function(
            func=kb_select_rows, name="kb_select_rows",
            description=("Filter a (Geo)DataFrame file (df_file_id) to rows whose column is in a "
                         "comma-separated value set, e.g. violent crime types. Returns a new file_id."),
            metadata={"category": "computation"}),
        StructuredTool.from_function(
            func=heatmap_image, name="heatmap_image",
            description=("Render a hexbin point-density HEAT MAP from a points (Geo)DataFrame file_id. "
                         "Returns a PNG file_id. Use this for 'heat map' requests (not a choropleth)."),
            metadata={"category": "generation"}),
        StructuredTool.from_function(
            func=choropleth_image, name="choropleth_image",
            description=("Render a CHOROPLETH (shaded polygons) from a polygon (Geo)DataFrame file_id "
                         "colored by `column` (e.g. the count from spatial_join_and_count). Robust "
                         "renderer — prefer this over an extracted plot function that may fail on the "
                         "installed matplotlib. Optional `scheme` (NaturalBreaks=Jenks, Quantile). "
                         "Returns a PNG file_id. Read the count column from the prior step's `columns`."),
            metadata={"category": "generation"}),
    ]


__all__ = ["write_geodata", "read_geodata", "capture_current_fig", "make_file_handle_tool",
           "kb_run_geofunction", "kb_select_rows", "heatmap_image", "choropleth_image",
           "make_geo_analysis_tools"]
