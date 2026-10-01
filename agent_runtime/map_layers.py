"""Extract a plottable map layer (GeoJSON) from a tool result.

Turns a geometry-bearing tool output -- notably ``overpass_search``, but any result
carrying a GeoJSON ``FeatureCollection`` or a list of ``{geometry, ...}`` features --
into a ``LayerArtifact``-shaped dict. The streaming layer emits it as a dedicated,
untruncated ``map_layer`` SSE event so a map client can plot the agent's spatial
findings live (the ``tool_result`` event carries only truncated text).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Cap features per emitted layer so a single event stays a sane size.
_MAX_FEATURES = 500
# How much of a label's slug stays readable in a layer id. Beyond this a digest of the FULL
# slug is appended — see _slug_id, and never simply lengthen this instead.
_SLUG_ID_MAX = 40


def _slug_id(slug: str, fallback: str) -> str:
    """A readable, bounded and COLLISION-FREE id for a layer.

    This was `slug[:40]`. The id IS the layer's identity — the client replaces a layer whose id
    matches — so two labels agreeing in their first 40 characters silently collapsed into one
    layer on the map: no error, no log, and the tool result still reported the layer delivered.

    It fired the moment a region name pushed the discriminator past the cut. "Downtown
    Champaign 1 km box - gse embedding (PCA-RGB)" and "... (shared PCA)" both truncate to
    `downtown_champaign_1_km_box_gse_embeddin` (the stem is 41 characters), so the second
    overwrote the first; the three-character-shorter Urbana pair kept one distinguishing
    character and both survived. A one-character margin decided which region kept its raster.

    Short slugs are returned unchanged, so existing ids do not move. Only a slug that WOULD
    have lost information carries the digest, and it is taken over the whole slug so the id
    stays stable across runs — the same label still replaces its own layer.
    """
    slug = slug or fallback
    if len(slug) <= _SLUG_ID_MAX:
        return slug
    return f"{slug[:_SLUG_ID_MAX]}_{hashlib.sha1(slug.encode('utf-8')).hexdigest()[:8]}"

# tool_name -> (layer source category, display-label prefix)
_GEO_TOOLS: Dict[str, tuple] = {
    "overpass_search": ("overpass", "OSM"),
    "spatial_search": ("kb", "Spatial"),
}


def boundary_layer_id(file_id: str) -> str:
    """The id of the map layer that SHOWS a polygon file, keyed on the file itself.

    Two tools draw the same polygons: admin_boundary puts the outline up, and embed_zones then
    redraws it with what the embedding found inside. They arrived as separate layers — a city
    appeared twice in the layer list, once as "Urbana city" and once as "gse embedded zone
    1777005" — because admin_boundary set no id at all and fell back to a slug of its LABEL,
    which embed_zones has no way to reconstruct from the file_id it was handed.

    Keying on the file they both hold means the second can take the first's place instead of
    stacking on it, and means neither has to know the other's wording.
    """
    return f"boundary-{str(file_id or '').strip()}"


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


# Written geodata a peer with no tools can still get onto the map.
_LAYERABLE_SUFFIXES = (".geojson",)


def layers_for_artifacts(directory: Any, artifacts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map-layer descriptors for the geodata a TOOLLESS peer left behind.

    A sandboxed CLI code peer has no add_map_layer — it writes files and returns prose, so
    its geodata reached the user as a download link and nothing else. Every other producer
    in the system delivers a map by RETURNING a descriptor that the trace layer turns into a
    `map_layer` event, so the honest route is to build that same descriptor from the files
    the peer actually wrote, and let it cross the same validation as everyone else's.

    The render is chosen by the rule add_map_layer uses for ``render="auto"``, so a layer
    looks the same however it was produced. A file with no features is skipped rather than
    shipped: an empty layer draws nothing, and inspect_artifacts already tells the answer
    about it.
    """
    out: List[Dict[str, Any]] = []
    base = Path(str(directory))
    try:
        import pyogrio
    except Exception:  # pragma: no cover - optional dep
        return out
    for record in artifacts or []:
        name = str((record or {}).get("filename") or "")
        url = (record or {}).get("download_url")
        if not name.lower().endswith(_LAYERABLE_SUFFIXES) or not url:
            continue
        path = base / name
        if not path.is_file():
            continue
        try:
            info = pyogrio.read_info(str(path))
            features = int(info.get("features") or 0)
            geometry = str(info.get("geometry_type") or "")
        except Exception:  # noqa: BLE001 - an unreadable file is simply not a layer
            continue
        if features <= 0:
            continue
        is_point = "point" in geometry.lower()
        render = ("heatmap" if (is_point and features > 2000)
                  else ("points" if is_point else "shapes"))
        out.append({
            "url": url,
            "label": Path(name).stem.replace("_", " ").strip() or name,
            "render": render,
            "count": features,
            "source": "analysis",
        })
    return out


def build_map_layers(tool_name: str, output: Any, *, qa: bool = True) -> List[Dict[str, Any]]:
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
            built = build_map_layer(tool_name, {"map_layer": ml}, qa=qa)
            if built and built["id"] not in seen:
                seen.add(built["id"])
                out.append(built)
        if out:
            return out
    one = build_map_layer(tool_name, output, qa=qa)
    return [one] if one else []


def build_map_layer(tool_name: str, output: Any, *, qa: bool = True) -> Optional[Dict[str, Any]]:
    """Return a ``LayerArtifact``-shaped dict for a geometry-bearing tool result, else None.

    ``qa=False`` skips the data inspection below. That block only ever DOWNGRADES a
    descriptor's ``render`` (never returns None), so it cannot change whether a layer is
    delivered — which lets :func:`delivers_map_layer` ask that question without the
    file read.
    """
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
            slug = re.sub(r"[^a-z0-9]+", "_", str(ml.get("label") or render).lower()).strip("_")
            out = {
                "kind": "map_layer",
                "id": ml.get("id") or f"agent-{_slug_id(slug, render)}",
                "source": ml.get("source") or "analysis",
                "label": ml.get("label") or render,
                "url": url,
                "render": render,
                "style_by": ml.get("style_by"),
                "count": ml.get("count"),
                # Boundary-only. A layer drawn over a raster of its own polygon has to frame
                # it, not cover it — and this dict is a fixed shape, so a flag the tool sets
                # and the client honours still arrives as nothing unless it is named here.
                "outline": bool(ml.get("outline")),
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
                # The vectors the picture was made FROM. An embedding raster is a 3-colour
                # projection and answers nothing alone, so the pointer to its package is what
                # makes the layer a handle for a later question rather than just an image. Named
                # here for the same reason `outline` and `sampled` had to be: this dict is a
                # FIXED SHAPE, and a field the tool sets and the client reads still arrives as
                # nothing unless it is copied across.
                embedding = ml.get("embedding")
                if isinstance(embedding, dict) and embedding.get("file_id"):
                    out["embedding"] = {k: v for k, v in embedding.items()
                                        if v not in (None, "", [], {})}
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
                local = str(resolve_file_id(fid)) if fid and qa else ""
                qa_result = inspect_geojson(local, render=render, style_by=out.get("style_by"),
                                            legend=out.get("legend")) if local else {"ok": True}
            except Exception:
                qa_result = {"ok": True}
            if not qa_result.get("ok"):
                logger.warning("map_layer %r would render meaninglessly (%s); shipping it as "
                               "plain shapes", out["id"], "; ".join(qa_result.get("problems") or []))
                out["render"] = "shapes"
                out.pop("style_by", None)
                out["degenerate"] = qa_result.get("problems") or True

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


def delivers_map_layer(tool_name: str, output: Any) -> bool:
    """Whether this tool output would ACTUALLY put a layer on the user's map.

    The single authority on that question, because this module is the boundary the layer has to
    cross to reach the client: if :func:`build_map_layers` yields nothing, the user sees nothing,
    whatever the tool claimed.

    It replaces four looser signals in the supervisor that each answered by pattern rather than
    by construction — a tool NAME appearing in tool_calls, a bare ``"on_map": true`` anywhere in
    a nested payload, a regex over the JSON blob. All of them said "delivered" for a FAILED
    ``admin_boundary``, which returns ``{"ok": false}`` with no descriptor; the supervisor then
    suppressed its own corrective retry and told the user the layer was already on their map.

    ``qa=False``: the data inspection only downgrades ``render``, so skipping it cannot change
    the answer, and it keeps this cheap and free of file reads for a question asked per tool
    result.
    """
    try:
        return bool(build_map_layers(tool_name, output, qa=False))
    except Exception:      # a malformed payload delivers nothing; never fail the turn over it
        return False
