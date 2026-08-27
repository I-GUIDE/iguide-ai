"""Extract a plottable map layer (GeoJSON) from a tool result.

Turns a geometry-bearing tool output -- notably ``overpass_search``, but any result
carrying a GeoJSON ``FeatureCollection`` or a list of ``{geometry, ...}`` features --
into a ``LayerArtifact``-shaped dict. The streaming layer emits it as a dedicated,
untruncated ``map_layer`` SSE event so a map client can plot the agent's spatial
findings live (the ``tool_result`` event carries only truncated text).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Cap features per emitted layer so a single event stays a sane size.
_MAX_FEATURES = 500

# tool_name -> (layer source category, display-label prefix)
_GEO_TOOLS: Dict[str, tuple] = {
    "overpass_search": ("overpass", "OSM"),
    "spatial_search": ("kb", "Spatial"),
}


def _coerce_obj(output: Any) -> Optional[Any]:
    """Best-effort parse of a tool output into a dict/list."""
    if output is None:
        return None
    if isinstance(output, (dict, list)):
        return output
    content = getattr(output, "content", None)  # LangChain ToolMessage
    text = content if isinstance(content, str) else (output if isinstance(output, str) else str(output))
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        return None


def _is_geometry(g: Any) -> bool:
    return isinstance(g, dict) and isinstance(g.get("type"), str) and g.get("coordinates") is not None


def _features_from(obj: Any) -> List[Dict[str, Any]]:
    """Collect GeoJSON Features from a FeatureCollection, an overpass-style
    ``{features:[{geometry,...}]}``, or a bare list of feature-like dicts."""
    out: List[Dict[str, Any]] = []

    def add(item: Any) -> None:
        if isinstance(item, dict) and _is_geometry(item.get("geometry")):
            props = item.get("properties")
            if not isinstance(props, dict):
                props = {k: v for k, v in item.items() if k != "geometry"}
            out.append({"type": "Feature", "geometry": item["geometry"], "properties": props})

    if isinstance(obj, dict):
        items = obj.get("features")
        if isinstance(items, list):
            for it in items:
                add(it)
    elif isinstance(obj, list):
        for it in obj:
            add(it)
    return out


def build_map_layers(tool_name: str, output: Any) -> List[Dict[str, Any]]:
    """Every layer a tool result delivers, in order.

    One tool call can legitimately produce more than one view — embed_zones returns the
    pixel-level embedding raster AND the zones grouped by it, and dropping either leaves the
    user looking at half the answer. Descriptors live under ``map_layers``; ``map_layer``
    stays the single-layer form.
    """
    obj = _coerce_obj(output)
    extra = obj.get("map_layers") if isinstance(obj, dict) else None
    if isinstance(extra, list) and extra:
        out: List[Dict[str, Any]] = []
        seen = set()
        for ml in extra:
            if not isinstance(ml, dict):
                continue
            built = build_map_layer(tool_name, {"map_layer": ml})
            if built and built["id"] not in seen:
                seen.add(built["id"])
                out.append(built)
        if out:
            return out
    one = build_map_layer(tool_name, output)
    return [one] if one else []


def build_map_layer(tool_name: str, output: Any) -> Optional[Dict[str, Any]]:
    """Return a ``LayerArtifact``-shaped dict for a geometry-bearing tool result, else None."""
    obj = _coerce_obj(output)
    if obj is None:
        return None

    # A tool that has ALREADY written a styled layer (add_map_layer) describes it explicitly:
    # the client fetches the GeoJSON by url rather than receiving it inline, which is what makes
    # a 50k-point heat map or an 800-polygon choropleth practical to stream.
    if isinstance(obj, dict) and isinstance(obj.get("map_layer"), dict):
        ml = dict(obj["map_layer"])
        url = str(ml.get("url") or "").strip()
        if url:
            render = str(ml.get("render") or "shapes")
            slug = re.sub(r"[^a-z0-9]+", "_", str(ml.get("label") or render).lower()).strip("_")[:40]
            out = {
                "kind": "map_layer",
                "id": ml.get("id") or f"agent-{slug or render}",
                "source": ml.get("source") or "analysis",
                "label": ml.get("label") or render,
                "url": url,
                "render": render,
                "style_by": ml.get("style_by"),
                "count": ml.get("count"),
                # Whether the layer is a SUBSET is part of the layer, not a remark in the
                # answer text: the client shows "shown/total" from these. Dropping them here
                # silently turned every sampled layer into one that looked complete.
                "sampled": bool(ml.get("sampled")),
                "total": ml.get("total"),
            }
            # A RASTER is an image draped over an extent: no features, no CRS to mis-declare,
            # no style column. The GeoJSON checks below have nothing to say about it, so it
            # returns here — leaving it to fall through would let a stricter QA downgrade it
            # to 'shapes' and the client would stop drawing it as a raster.
            if render == "raster":
                bounds = ml.get("bounds")
                if not (isinstance(bounds, (list, tuple)) and len(bounds) == 4):
                    logger.warning("raster map_layer %r has no usable bounds; dropping", out["id"])
                    return None
                out["bounds"] = [float(v) for v in bounds]
                out["opacity"] = float(ml.get("opacity") or 0.85)
                return out

            # A CATEGORICAL layer carries its own palette: the tool that assigned the classes
            # is the only thing that knows what they mean, so the legend travels WITH the layer
            # instead of being hardcoded per-tool in the client. Dropped when malformed rather
            # than passed through half-valid.
            legend = ml.get("legend")
            if isinstance(legend, list):
                entries = [
                    {"label": str(e.get("label")), "color": [int(v) for v in e.get("color")]}
                    for e in legend
                    if isinstance(e, dict) and e.get("label") is not None
                    and isinstance(e.get("color"), (list, tuple)) and len(e["color"]) == 4
                ]
                if entries:
                    out["legend"] = entries

            # INVARIANT: a 'categories' render is meaningless without its palette. The client
            # maps class NAMES through the legend; with no legend it falls back to the NUMERIC
            # ramp, where Number("High-High") is NaN, so all 801 features landed on one flat
            # fill while the answer text described five colours (observed, twice, from two
            # different tools). Enforce it here — the one boundary every tool's layer crosses —
            # rather than trusting each emitter. Downgrade instead of deriving: a URL-based
            # layer's features are not in hand at this point, so deriving a palette belongs in
            # the tool that assigned the classes. 'shapes' at least does not claim to encode
            # anything it isn't.
            # The written file is in the local store, so the descriptor can be checked against
            # the actual data rather than taken on trust. A choropleth over a constant column
            # or geometry in metres mislabelled EPSG:4326 is delivered-but-meaningless, and the
            # client has no way to tell.
            try:
                from agent_runtime.file_store import resolve_file_id
                from agent_runtime.layer_qa import inspect_geojson
                fid = str(url).rstrip("/").split("/")[-2] if "/files/" in str(url) else ""
                local = str(resolve_file_id(fid)) if fid else ""
                qa = inspect_geojson(local, render=render, style_by=out.get("style_by"),
                                    legend=out.get("legend")) if local else {"ok": True}
            except Exception:
                qa = {"ok": True}
            if not qa.get("ok"):
                logger.warning("map_layer %r would render meaninglessly (%s); shipping it as "
                               "plain shapes", out["id"], "; ".join(qa.get("problems") or []))
                out["render"] = "shapes"
                out.pop("style_by", None)
                out["degenerate"] = qa.get("problems") or True

            if render == "categories" and not out.get("legend"):
                logger.warning(
                    "map_layer %r declares render='categories' with no usable legend; "
                    "downgrading to 'shapes' so it does not render as one flat fill", out["id"])
                out["render"] = "shapes"
                out["legend_missing"] = True

            return out
    features = _features_from(obj)
    if not features:
        return None

    capped = features[:_MAX_FEATURES]
    source, prefix = _GEO_TOOLS.get(tool_name, ("analysis", tool_name))

    label = prefix
    suffix = ""
    if isinstance(obj, dict):
        query = obj.get("query") or {}
        feature = query.get("feature") or query.get("osm_filter")
        place = query.get("place")
        if feature:
            label = f"{prefix}: {feature}" + (f" in {place}" if place else "")
            suffix = re.sub(r"[^a-z0-9]+", "_", str(feature).lower()).strip("_")[:40]

    layer_id = f"agent-{tool_name}" + (f"-{suffix}" if suffix else "")
    return {
        "kind": "map_layer",
        "id": layer_id,
        "source": source,
        "label": label,
        "count": len(capped),
        "truncated": len(features) > len(capped),
        "geojson": {"type": "FeatureCollection", "features": capped},
    }
