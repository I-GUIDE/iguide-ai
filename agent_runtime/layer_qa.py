"""Does this layer actually SHOW anything? — checks run before a visual is delivered.

Every "flat map that makes no sense" observed so far was detectable from the data, with no
need to look at pixels or ask the model to be careful:

- a choropleth whose ``style_by`` column has ONE distinct value paints every feature the same
  colour. The map looks broken; the data is simply constant (a join that matched nothing, or
  the wrong column).
- a choropleth whose ``style_by`` is non-numeric sends class names down a numeric ramp:
  ``float("High-High")`` fails for every feature, so they all fall to the default colour.
- a categorical layer with one class, or whose legend labels match none of the actual values,
  has a palette that cannot be applied.
- geometry in projected metres shipped as EPSG:4326 lands the layer thousands of degrees away —
  off the map entirely, with the basemap zoomed to nothing.
- an empty layer, or one whose bounding box is a single point, gives the client nothing to frame.
- a PNG that is one flat colour (an all-white axes, a figure drawn from an empty frame).

Reading is deliberately cheap: ``pyogrio.read_info`` gives the feature count, bounds and
geometry type without parsing geometry, and a single attribute column is read with
``read_geometry=False``. A 15 MB layer costs a few milliseconds, not a full parse.

Nothing here raises: a checker that breaks must never break the delivery it was inspecting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Beyond this a coordinate cannot be lon/lat — the layer is almost certainly projected metres
# mislabelled as EPSG:4326, which puts it off the map rather than merely in the wrong place.
_LON_LIMIT = 180.0
_LAT_LIMIT = 90.0
# A bounding box smaller than this in BOTH directions is a point as far as framing goes
# (~1 m at the equator), so fitBounds has nothing to zoom to.
_DEGENERATE_BBOX = 1e-5


def _numeric_variation(values: List[Any]) -> Optional[int]:
    """Distinct finite numeric values, or None if the column is not numeric at all."""
    seen = set()
    numeric = 0
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f != f:                      # NaN
            continue
        numeric += 1
        seen.add(f)
        if len(seen) > 2:               # more than "constant" is all we need to know
            return len(seen)
    return len(seen) if numeric else None


def inspect_geojson(path: str, *, render: str = "shapes", style_by: Optional[str] = None,
                    legend: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Report whether a written layer can render meaningfully.

    Returns ``{"ok": bool, "problems": [...], "notes": [...], "features": int, ...}``.
    ``ok=False`` means the layer would draw as nothing or as one flat colour; the caller
    decides whether to refuse, downgrade the render, or just pass the note along.
    """
    out: Dict[str, Any] = {"ok": True, "problems": [], "notes": []}
    try:
        import pyogrio
    except Exception:                                    # pragma: no cover - optional dep
        return out
    p = Path(path)
    if not p.exists():
        return out
    try:
        info = pyogrio.read_info(str(p))
    except Exception as exc:
        logger.debug("layer_qa: read_info failed for %s: %s", p.name, exc)
        return out

    features = int(info.get("features") or 0)
    out["features"] = features
    if features == 0:
        out["ok"] = False
        out["problems"].append("the layer has no features, so nothing will be drawn")
        return out

    bounds = info.get("total_bounds")
    if bounds is not None and len(list(bounds)) == 4:
        minx, miny, maxx, maxy = (float(v) for v in bounds)
        out["bounds"] = [minx, miny, maxx, maxy]
        if abs(minx) > _LON_LIMIT or abs(maxx) > _LON_LIMIT \
                or abs(miny) > _LAT_LIMIT or abs(maxy) > _LAT_LIMIT:
            out["ok"] = False
            out["problems"].append(
                f"coordinates are outside lon/lat range (bounds {out['bounds']}) — the geometry "
                "looks like projected metres labelled EPSG:4326, so it will land off the map. "
                "Reproject to EPSG:4326 before delivering it")
        elif (maxx - minx) < _DEGENERATE_BBOX and (maxy - miny) < _DEGENERATE_BBOX:
            out["notes"].append("every feature sits at effectively one location, so the map "
                                "cannot frame a meaningful extent")

    # The styling column is where flatness usually comes from.
    if style_by:
        try:
            frame = pyogrio.read_dataframe(str(p), columns=[style_by], read_geometry=False)
        except Exception:
            frame = None
        if frame is None or style_by not in getattr(frame, "columns", []):
            out["ok"] = False
            out["problems"].append(
                f"the styling column {style_by!r} is not in the written layer, so every feature "
                "falls back to one default colour")
        else:
            values = list(frame[style_by])
            distinct_numeric = _numeric_variation(values)
            distinct_all = len({str(v) for v in values if v is not None})
            out["distinct_values"] = distinct_all
            if render == "choropleth":
                if distinct_numeric is None:
                    out["ok"] = False
                    out["problems"].append(
                        f"{style_by!r} holds no numeric values, but a choropleth shades by "
                        "number — every feature will draw in the same colour. Use "
                        "render='categories' for class names, or pick a numeric column")
                elif distinct_numeric <= 1:
                    out["ok"] = False
                    out["problems"].append(
                        f"{style_by!r} has a single distinct value across all {features} "
                        "features, so the choropleth is one flat colour. Check the join or "
                        "choose a column that varies")
            elif render == "categories":
                if distinct_all <= 1:
                    out["ok"] = False
                    out["problems"].append(
                        f"{style_by!r} has one class across all {features} features, so every "
                        "feature draws identically")
                if legend:
                    labels = {str(e.get("label")) for e in legend if isinstance(e, dict)}
                    present = {str(v) for v in values if v is not None}
                    if labels and not (labels & present):
                        out["ok"] = False
                        out["problems"].append(
                            "no legend label matches any value in "
                            f"{style_by!r} (legend: {sorted(labels)[:4]}, data: "
                            f"{sorted(present)[:4]}) — the palette cannot be applied")
    return out


def inspect_image(path: str) -> Dict[str, Any]:
    """Report whether a rendered PNG has any visible content."""
    out: Dict[str, Any] = {"ok": True, "problems": []}
    try:
        from PIL import Image
    except Exception:                                    # pragma: no cover - optional dep
        return out
    p = Path(path)
    if not p.exists():
        return out
    try:
        with Image.open(p) as im:
            im = im.convert("RGBA")
            # Downscale first: a blank figure stays blank, and this bounds the cost for a
            # 300-dpi render to a few thousand pixels.
            im.thumbnail((160, 160))
            pixels = list(im.getdata())
    except Exception as exc:
        logger.debug("layer_qa: could not read image %s: %s", p.name, exc)
        return out
    if not pixels:
        return out
    if all(px[3] == 0 for px in pixels):
        out["ok"] = False
        out["problems"].append("the image is fully transparent — nothing was drawn")
        return out
    distinct = {px[:3] for px in pixels}
    out["distinct_colors"] = len(distinct)
    if len(distinct) <= 2:
        out["ok"] = False
        out["problems"].append(
            f"the image has only {len(distinct)} distinct colour(s) — it is effectively blank, "
            "which usually means the plotted frame was empty or the column had no values")
    return out


# Extensions worth checking, and which checker reads them.
_GEO_SUFFIXES = {".geojson", ".json"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def inspect_artifacts(directory: str, filenames: List[str]) -> List[Dict[str, Any]]:
    """Run the delivery checks over files a CLI code peer left behind.

    The tool path gets these checks inside ``add_map_layer``. A sandboxed CLI peer has
    NO tools — it writes files and returns prose — so nothing between it and the user
    ever looked at what it produced. That is precisely where a blank figure or an empty
    layer survives to be described as a result, because the peer's own summary is the
    only account of it and the peer is not the one that would notice.

    Returns one entry per file that fails, with the same wording the tool path uses, so
    the answer can say what is wrong instead of presenting it. Empty means nothing
    detectable is wrong — not that the output is good.
    """
    findings: List[Dict[str, Any]] = []
    base = Path(directory)
    for name in filenames or []:
        path = base / str(name)
        suffix = path.suffix.lower()
        try:
            if not path.is_file():
                continue
            if suffix in _GEO_SUFFIXES:
                report = inspect_geojson(str(path))
            elif suffix in _IMAGE_SUFFIXES:
                report = inspect_image(str(path))
            else:
                continue
        except Exception as exc:  # pragma: no cover - a checker must never break delivery
            logger.debug("layer_qa: could not inspect %s: %s", name, exc)
            continue
        if not report.get("ok") and report.get("problems"):
            findings.append({"file": str(name), "problems": list(report["problems"])})
    return findings


__all__ = ["inspect_geojson", "inspect_image", "inspect_artifacts"]
