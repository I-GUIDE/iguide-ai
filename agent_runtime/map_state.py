"""What is on the user's map, as the BROWSER reports it with each turn (Stage 81).

The agent cannot see the map. Until this stage it guessed from its own ledger of the layers it
had emitted (``_visible_state_lines`` in ``supervisor/graph.py``), and on 2026-10-18 that guess
was wrong live in two ways:

1. Layers that never drew. Both suitability layers failed in the browser with 401
   ``token_expired``; the next turn's answer still said the score was "available on your
   interactive map" beside an empty map. The ledger records what was SENT, not what arrived.
2. Things computed but never drawn. Schools demo, step 2 ("Show the site, its 1-mile buffer and
   the schools inside it on the map"): after step 1 had drawn a schools-only layer, the router's
   first decision was 'done' in 2 of 3 runs. It read "buffer computed" as "buffer on the map".

The map UI now sends ``map_state`` with every turn: the layers it holds (no geometry), the layers
that failed to draw and why, the viewport and the drawn region. This module validates it,
renders it for the router and the answer writer, reconciles the ledger against it, and holds the
one deterministic rule that uses it (:func:`unmapped_requested_items`).

Absent ``map_state`` (an older client, the harness, the JSON endpoint), nothing here runs and
every consumer behaves exactly as before.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

MAP_STATE_HEADING = "WHAT IS ON THE USER'S MAP NOW (reported by the browser)"

# Bounds on what one request may carry into a prompt. The client caps first and says so; these
# hold for any other caller. 40 layers is far past any real conversation (the longest demo ends
# with 6), and the rendered section stays under ~4 KB at the cap.
MAX_LAYERS = 40
MAX_UNDRAWN = 20
_LABEL_CHARS = 120
_REASON_CHARS = 200
_SHORT_CHARS = 40


def _text(value: Any, limit: int) -> str:
    text = " ".join(str(value if value is not None else "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _number(value: Any) -> Optional[float]:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    return n if math.isfinite(n) else None


def _bbox(value: Any) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    nums = [_number(v) for v in value]
    if any(n is None for n in nums):
        return None
    return [round(n, 4) for n in nums]  # type: ignore[arg-type]


def normalize(raw: Any) -> Optional[Dict[str, Any]]:
    """The client's report, validated and capped, or None when there is none.

    None (not an empty report) for anything that is not a mapping: an older client sends no
    field, and that must stay distinguishable from a client reporting an EMPTY map.
    """
    if not isinstance(raw, Mapping):
        return None
    layers: List[Dict[str, Any]] = []
    raw_layers = raw.get("layers") if isinstance(raw.get("layers"), list) else []
    for item in raw_layers:
        if not isinstance(item, Mapping):
            continue
        label = _text(item.get("label"), _LABEL_CHARS)
        if not label:
            continue
        layer: Dict[str, Any] = {"label": label}
        for key in ("id", "kind", "render", "source"):
            if item.get(key) not in (None, ""):
                layer[key] = _text(item.get(key), _SHORT_CHARS if key != "id" else 80)
        features = _number(item.get("features"))
        if features is not None and features >= 0:
            layer["features"] = int(features)
        if item.get("visible") is False:
            layer["visible"] = False
        opacity = _number(item.get("opacity"))
        if opacity is not None:
            layer["opacity"] = round(min(max(opacity, 0.0), 1.0), 2)
        layers.append(layer)
    undrawn: List[Dict[str, Any]] = []
    raw_undrawn = raw.get("undrawn") if isinstance(raw.get("undrawn"), list) else []
    for item in raw_undrawn:
        if not isinstance(item, Mapping):
            continue
        label = _text(item.get("label"), _LABEL_CHARS)
        if not label:
            continue
        entry = {"label": label, "reason": _text(item.get("reason"), _REASON_CHARS) or "unknown"}
        if item.get("id"):
            entry["id"] = _text(item.get("id"), 80)
        undrawn.append(entry)

    # What the client said it dropped, plus whatever this side drops on top of that.
    truncated = raw.get("truncated") if isinstance(raw.get("truncated"), Mapping) else {}
    layers_total = max(int(_number(truncated.get("layers")) or 0), len(layers))
    undrawn_total = max(int(_number(truncated.get("undrawn")) or 0), len(undrawn))
    out: Dict[str, Any] = {"layers": layers[:MAX_LAYERS], "undrawn": undrawn[:MAX_UNDRAWN]}
    if layers_total > len(out["layers"]) or undrawn_total > len(out["undrawn"]):
        out["truncated"] = {"layers": layers_total, "undrawn": undrawn_total}
    viewport = _bbox(raw.get("viewport"))
    if viewport:
        out["viewport"] = viewport
    region = _bbox(raw.get("region"))
    if region:
        out["region"] = region
    return out


# --- rendering ------------------------------------------------------------------------------

def _norm_label(label: Any) -> str:
    return " ".join(str(label or "").replace("…", "").lower().split())


def _same_layer(a: Any, b: Any) -> bool:
    """Ledger labels are cut at the ledger's value length, so a prefix counts as a match."""
    x, y = _norm_label(a), _norm_label(b)
    if not x or not y:
        return False
    return x == y or (len(x) >= 12 and y.startswith(x)) or (len(y) >= 12 and x.startswith(y))


def _layer_phrase(layer: Mapping[str, Any]) -> str:
    bits = [layer.get("render") or layer.get("kind")]
    if layer.get("features") is not None:
        n = layer["features"]
        bits.append(f"{n} feature{'' if n == 1 else 's'}")
    if layer.get("source"):
        bits.append(str(layer["source"]))
    if layer.get("visible") is False:
        bits.append("hidden by the user")
    elif layer.get("opacity") is not None and layer["opacity"] < 1:
        bits.append(f"opacity {layer['opacity']}")
    detail = ", ".join(str(b) for b in bits if b)
    return f"{layer['label']!r}" + (f" ({detail})" if detail else "")


def _absent_from_browser(map_state: Mapping[str, Any],
                         ledger_labels: Iterable[str]) -> List[str]:
    """Ledger layers the browser neither holds nor reports as failed."""
    held = [l.get("label") for l in map_state.get("layers") or []]
    failed = [u.get("label") for u in map_state.get("undrawn") or []]
    out: List[str] = []
    for label in ledger_labels:
        if str(label).startswith("+") and str(label).endswith("more"):
            continue                       # the ledger's own "+N more" marker, not a layer
        if any(_same_layer(label, h) for h in held) or any(_same_layer(label, f) for f in failed):
            continue
        if label not in out:
            out.append(str(label))
    return out


def render_lines(map_state: Mapping[str, Any], ledger_labels: Iterable[str] = ()) -> List[str]:
    """The section body, one fact per line. ``ledger_labels`` are the layers the ledger says
    earlier turns delivered; any the browser does not have are stated as NOT on the map."""
    lines: List[str] = []
    layers = list(map_state.get("layers") or [])
    if layers:
        lines.append("- on the map now: " + "; ".join(_layer_phrase(l) for l in layers))
    else:
        lines.append("- the map has no layers on it")
    for u in map_state.get("undrawn") or []:
        lines.append(f"- NOT on the map — {u['label']!r} failed to load in the browser "
                     f"({u.get('reason') or 'unknown'})")
    for label in _absent_from_browser(map_state, ledger_labels):
        lines.append(f"- NOT on the map — {label!r} was sent by an earlier turn but the browser "
                     "does not have it (it failed to load, or the user removed it)")
    trunc = map_state.get("truncated")
    if isinstance(trunc, Mapping):
        lines.append(f"- (this list is capped: {len(layers)} of {trunc.get('layers')} layers and "
                     f"{len(map_state.get('undrawn') or [])} of {trunc.get('undrawn')} failures "
                     "shown)")
    if map_state.get("viewport"):
        lines.append(f"- the user is looking at bbox {map_state['viewport']} "
                     "(minLon, minLat, maxLon, maxLat)")
    if map_state.get("region"):
        lines.append(f"- the user drew a region: bbox {map_state['region']}")
    return lines


def render_note(map_state: Mapping[str, Any], ledger_labels: Iterable[str] = ()) -> str:
    """The whole section, heading included."""
    return (f"{MAP_STATE_HEADING}:\n"
            "The browser's own list, sent with the user's message. It overrides any tool record "
            "of what earlier turns delivered: a layer not listed under 'on the map now' is NOT "
            "on the map — never tell the user it is, or that it is available on their map. "
            "Layers added during THIS turn are not in it yet.\n"
            + "\n".join(render_lines(map_state, ledger_labels)))


# --- the ledger, reconciled against the browser --------------------------------------------

def reconcile_rows(rows: Sequence[Any], map_state: Mapping[str, Any]) -> List[Any]:
    """Ledger rows whose ``map_layer`` keeps only the layers the browser really has.

    A label the browser does not hold moves to ``map_layer_absent`` as ``{label, why}``,
    so every consumer of the rows — the "[on the map as …]" annotation, the auditor's map
    amnesty (``_map_delivered_earlier``), the router's prior-turns payload — reads it as not on
    the map. Rows are copied; the session's ledger itself is never changed.
    """
    held = [l.get("label") for l in map_state.get("layers") or []]
    failed = {u.get("label"): u.get("reason") for u in map_state.get("undrawn") or []}
    out: List[Any] = []
    for row in rows or []:
        if not isinstance(row, dict) or not row.get("map_layer"):
            out.append(row)
            continue
        labels = row["map_layer"] if isinstance(row["map_layer"], (list, tuple)) \
            else [row["map_layer"]]
        keep, absent = [], []
        for label in labels:
            if str(label).startswith("+") and str(label).endswith("more"):
                keep.append(label)
            elif any(_same_layer(label, h) for h in held):
                keep.append(label)
            else:
                why = next((r for f, r in failed.items() if _same_layer(label, f)), None)
                absent.append({"label": label,
                               "why": f"failed to load: {why}" if why else "not in the browser"})
        new = dict(row)
        if keep:
            new["map_layer"] = keep
        else:
            new.pop("map_layer", None)
        if absent:
            new["map_layer_absent"] = absent
        out.append(new)
    return out


# --- the deterministic rule: a request to SHOW things that are not on the map -------------
#
# Evidence (prod e5e311f0, 2026-10-18, all 10 traces of the schools demo's step 2, "Show the
# site, its 1-mile buffer and the schools inside it on the map"): when step 1 had drawn NO layer,
# step 2 drew site, buffer and schools 6 of 6 times; when step 1 had drawn a schools-only layer,
# the router's first decision was 'done' 2 of 3 times — the buffer had been COMPUTED, and it took
# that for the buffer being on the map. With the browser's report there is no need to guess.
#
# Kept narrow on purpose. It fires only when (a) the request asks to put things ON THE MAP,
# (b) it names a drawable item from the small vocabulary below as the head of what is to be
# shown — "the site", "its 1-mile buffer", not "the schools within 1 mile OF THE SITE", where the
# site only locates the schools — and (c) no layer the browser reports has that item as the head
# of its label. Items outside the vocabulary ("the schools") are left to the router.

_ITEMS = (
    ("buffer", r"buffers?|buffer zones?|radius|radii|rings?"),
    ("site", r"sites?"),
    ("route", r"routes?|shortest paths?|paths?"),
    ("boundary", r"boundar(?:y|ies)|outlines?|borders?|perimeters?"),
    ("isochrone", r"isochrones?|service areas?|catchments?|drive[- ]?time areas?"),
    ("network", r"(?:road |street )?networks?"),
    ("centroid", r"centroids?"),
)
_ITEM_RES = [(name, re.compile(rf"\b(?:{pat})\b", re.I)) for name, pat in _ITEMS]

# What ends the noun phrase a conjunct starts with: a preposition or a relative clause.
_NP_END = re.compile(
    r"\b(?:of|around|near|from|to|within|inside|in|at|for|by|with|along|on|onto|that|which|who|"
    r"where|across|between|over|under|containing|covering|showing)\b|[(\[:—–]", re.I)
_CONJUNCTS = re.compile(r",|;|&|\band\b|\bplus\b|\bas well as\b|\bthen\b", re.I)
# "… on the map" / "… to the map" — the request is about what is drawn.
_ON_MAP = re.compile(r"\b(?:on|onto|to)\s+(?:the|a|my|your|this|our)\s+map\b", re.I)
_SHOW_VERB = re.compile(
    r"\b(?:show|draw|map|plot|display|visuali[sz]e|overlay|add|put|mark|highlight|render)\b",
    re.I)


def _heads(text: str) -> List[str]:
    """The vocabulary item each conjunct's leading noun phrase is about, in order."""
    found: List[str] = []
    for part in _CONJUNCTS.split(text.replace("_", " ")):
        np = _NP_END.split(part, maxsplit=1)[0]
        best, at = None, -1
        for name, rx in _ITEM_RES:
            for m in rx.finditer(np):
                if m.start() > at:      # the LAST item term is the head: "site 1mi buffer"
                    best, at = name, m.start()
        if best and best not in found:
            found.append(best)
    return found


def requested_items(query: str) -> List[str]:
    """Drawable items a request asks to see on the map, or [] when it asks for none."""
    items: List[str] = []
    for sentence in re.split(r"[.?!\n]+", str(query or "")):
        on_map = _ON_MAP.search(sentence)
        verb = _SHOW_VERB.search(sentence)
        if not on_map or not verb or verb.start() > on_map.start():
            continue
        # What is to be shown sits between the verb and "on the map", or after it
        # ("show on the map the site and its buffer").
        objects = sentence[verb.end():on_map.start()]
        if not objects.strip():
            objects = sentence[on_map.end():]
        for item in _heads(objects):
            if item not in items:
                items.append(item)
    return items


def layer_items(layer: Mapping[str, Any]) -> List[str]:
    """The items a reported layer shows: its label's head, plus what its render says."""
    items = _heads(str(layer.get("label") or ""))
    render = str(layer.get("render") or "")
    if render in ("route", "network") and render not in items:
        items.append(render)
    return items


def unmapped_requested_items(query: str, map_state: Optional[Mapping[str, Any]]) -> List[str]:
    """Items the request asks to show that no layer on the user's map shows. [] = nothing to
    enforce, including whenever there is no browser report."""
    if not isinstance(map_state, Mapping):
        return []
    wanted = requested_items(query)
    if not wanted:
        return []
    shown = {i for layer in map_state.get("layers") or [] for i in layer_items(layer)}
    return [i for i in wanted if i not in shown]


def draw_directive(missing: Sequence[str]) -> str:
    """Told to the peer the guard routes to, so it knows why it is running."""
    return ("THE USER ASKED TO SEE THESE ON THE MAP, AND THE BROWSER REPORTS THEY ARE NOT ON IT: "
            + ", ".join(missing) + ". Draw each one as its own map layer (add_map_layer, or the "
            "tool that makes it with its map layer). Reuse earlier turns' files and results where "
            "they hold it — do not recompute what is already known.")


__all__ = [
    "MAP_STATE_HEADING", "MAX_LAYERS", "MAX_UNDRAWN", "normalize", "render_lines",
    "render_note", "reconcile_rows", "requested_items", "layer_items",
    "unmapped_requested_items", "draw_directive",
]
