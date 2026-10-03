"""What a pretrained head was trained on, and whether a saved embedding is inside it.

The heads behind ``/api/predict_package`` are three RandomForestClassifiers fitted on corn
presence in Illinois from USDA CDL 2022. Nothing in the shipped artifacts records that:
``heads_meta.json`` carries task / kind / label / units / region / classes and a per-model
dim / score / n, and the pickles are bare estimators with no region and no dates at all. So a
vector from another year, another place or another footprint is scored in silence and the answer
looks exactly like an in-domain one.

This module keeps the training domain as constants and checks a package against them, so the
tool can say what it is extrapolating over instead of quoting a bare probability. The constants
come from the BUILDER source rather than from the service, because the service cannot tell you:
the demo builder samples a 2022-06-01..2022-09-01 window over USDA CDL 2022 inside its Illinois
bbox at a 1280 m point footprint. Change them here if the heads are ever refit.

Nothing in here imports scikit-learn or calls the service. The head stays behind the service
(that is the only copy of the trained weights); this is the part that can be tested offline.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- the training domain, transcribed from the head builder ---------------------
TRAIN_START = "2022-06-01"
TRAIN_END = "2022-09-01"
TRAIN_YEAR = 2022
# The builder's ILLINOIS_BBOX (minlon, minlat, maxlon, maxlat).
TRAIN_BBOX = (-91.6, 37.0, -87.5, 42.5)
TRAIN_REGION = "Illinois"
TRAIN_LABELS = "USDA CDL 2022"
# The builder sampled points with a 1280 m PointBuffer, so one training row describes roughly a
# 2.6 km square. A region much larger than that is being summarised more coarsely than anything
# the head ever saw.
TRAIN_FOOTPRINT_M = 1280
# How many times the training footprint a region may span before it is worth saying so. Set from
# what the store actually holds: the agent's own default buffer_m is 2048 (1.6x), so a tighter
# threshold would fire on almost every package and stop being read.
FOOTPRINT_WARN_RATIO = 4.0

# Pooled width is NOT a model identifier. Measured across the 99 packages in the store: five
# models share 768 and two share 1024. A head cannot notice it was handed the wrong model's
# vector of the right width — real olmoearth / prithvi / terrafm / thor vectors all scored
# cleanly through the dofa head, four of six labelled "corn", and the numbers sit inside the
# range correct-provenance vectors produce, so no plausibility check downstream can catch it.
# The model name must therefore come from the package's own ``pooled__<model>`` KEY and never
# from a caller-supplied string. This map exists only so a refusal can explain why.
MODELS_BY_WIDTH: Dict[int, Tuple[str, ...]] = {
    64: ("gse",),
    128: ("tessera",),
    384: ("terramind",),
    768: ("dofa", "olmoearth", "prithvi", "terrafm", "thor"),
    1024: ("clay", "satmae"),
}


def read_manifest(path: Any) -> Dict[str, Any]:
    """The package's own manifest, or ``{}`` when it has none or it will not parse.

    Loaded with ``allow_pickle=False`` on purpose. The premise of this whole path is "np.load
    whatever file the user hands me", and a package can come from an upload as easily as from
    the service, so deserialising pickles here would be handing arbitrary code execution to
    whoever produced the file. Nothing in a real package needs it: the arrays are float32 and
    ``meta`` is a unicode scalar.
    """
    try:
        with np.load(path, allow_pickle=False) as z:
            if "meta" not in z.files:
                return {}
            man = json.loads(str(z["meta"]))
    except Exception:  # noqa: BLE001 - a package with no readable manifest is still scorable
        return {}
    return man if isinstance(man, dict) else {}


def pooled_vectors(path: Any) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Every ``pooled__<model>`` vector in the package, keyed by the model in the key name."""
    out: Dict[str, np.ndarray] = {}
    problems: List[str] = []
    try:
        with np.load(path, allow_pickle=False) as z:
            for key in z.files:
                if not key.startswith("pooled__"):
                    continue
                model = key[len("pooled__"):]
                try:
                    out[model] = np.asarray(z[key], dtype=np.float32)
                except Exception as exc:  # noqa: BLE001
                    problems.append(f"{model}: unreadable ({type(exc).__name__})")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"the package could not be opened: {type(exc).__name__}: {exc}"[:200])
    return out, problems


def vector_refusal(model: str, vec: np.ndarray, expected_dim: Optional[int] = None) -> Optional[str]:
    """Why this vector must not be scored, or ``None`` when it may be.

    Three checks, each for a measured way to get a confident wrong number rather than an error:
    a stack of zone vectors flattens to one long row and is scored as if it were one region; a
    right-width wrong-model vector is accepted silently; and an all-nodata region's all-NaN
    vector is turned into zeros by the service's ``nan_to_num`` and comes back as a clean 0.45,
    where re-embedding the same region raises "the requested ROI contains no valid embedding
    pixels".
    """
    arr = np.asarray(vec)
    if arr.ndim != 1:
        return (f"the {model!r} entry is {arr.shape}, not a single pooled vector — a stack of "
                f"{arr.shape[0]} rows would be flattened into one row and scored as though it "
                "were one region")
    if expected_dim is not None and int(arr.size) != int(expected_dim):
        return (f"the {model!r} vector is {arr.size}-dimensional but the {model!r} head expects "
                f"{expected_dim}")
    if not np.isfinite(arr).any():
        return (f"the {model!r} vector has no finite value — the region it came from was all "
                "nodata, and scoring it would turn a blank region into a confident number")
    return None


def _bbox_of(manifest: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    geom = manifest.get("geometry")
    if not isinstance(geom, dict):
        return None
    try:
        box = (float(geom["minlon"]), float(geom["minlat"]),
               float(geom["maxlon"]), float(geom["maxlat"]))
    except (KeyError, TypeError, ValueError):
        return None
    return box


def _years_of(manifest: Dict[str, Any]) -> List[int]:
    """The years the package's imagery window spans, from the manifest's TOP-LEVEL start/end.

    Deliberately not the per-model ``meta.temporal``. That field is the more precise one when it
    is there, but it is absent far too often to guard with: across the 100 manifests in the store
    it is empty for 36 of 105 model entries, including EVERY satmae and dofa entry — two of the
    three models that have a head. Top-level ``start``/``end`` is present on 100 of 100 by
    construction, because the request carries defaults. Reading the precise field first and
    falling back would be worse than reading this one: the check would speak on gse and stay
    silent on satmae, which reads as "gse is the risky one" when the truth is the opposite.
    """
    years: List[int] = []
    for key in ("start", "end"):
        text = str(manifest.get(key) or "").strip()
        if len(text) >= 4 and text[:4].isdigit():
            years.append(int(text[:4]))
    return sorted(set(years))


def region_span_m(manifest: Dict[str, Any]) -> Optional[float]:
    """Roughly how far the embedded region reaches, as its longer side in metres.

    Approximate on purpose — this decides whether to print a sentence about extrapolation, not
    anything numeric in the answer. Taken from the bbox rather than from ``buffer_m``, which is
    echoed into every manifest whether or not the geometry was a point buffer.
    """
    box = _bbox_of(manifest)
    if box is None:
        buf = manifest.get("buffer_m")
        try:
            return 2.0 * float(buf) if buf else None
        except (TypeError, ValueError):
            return None
    minlon, minlat, maxlon, maxlat = box
    mid = math.radians((minlat + maxlat) / 2.0)
    ns = (maxlat - minlat) * 111_320.0
    ew = (maxlon - minlon) * 111_320.0 * max(math.cos(mid), 1e-6)
    return max(abs(ns), abs(ew))


def domain_warnings(manifest: Dict[str, Any]) -> List[str]:
    """Every way this package sits outside what the heads were fitted on.

    Warnings, not refusals: a prediction outside the training domain is still the model's honest
    output, and the user asked for it. What must not happen is reporting it as though it were
    inside. An empty list means every check that could run, passed — call
    :func:`unverifiable_domain` for the checks that could not run at all.
    """
    out: List[str] = []

    years = _years_of(manifest)
    if years and TRAIN_YEAR not in years:
        span = str(years[0]) if len(years) == 1 else f"{years[0]}-{years[-1]}"
        out.append(f"this embedding is from {span}; the heads were fitted on {TRAIN_LABELS}, so "
                   f"the labels they learned describe {TRAIN_YEAR} land cover")

    box = _bbox_of(manifest)
    if box is not None:
        tminlon, tminlat, tmaxlon, tmaxlat = TRAIN_BBOX
        disjoint = (box[2] < tminlon or box[0] > tmaxlon
                    or box[3] < tminlat or box[1] > tmaxlat)
        inside = (box[0] >= tminlon and box[1] >= tminlat
                  and box[2] <= tmaxlon and box[3] <= tmaxlat)
        if disjoint:
            out.append(f"this region is outside {TRAIN_REGION}, where the heads were trained — "
                       "the crop mix, field size and imagery season all differ")
        elif not inside:
            out.append(f"this region only partly overlaps {TRAIN_REGION}, where the heads were "
                       "trained")

    span_m = region_span_m(manifest)
    if span_m and span_m > FOOTPRINT_WARN_RATIO * 2.0 * TRAIN_FOOTPRINT_M:
        km = span_m / 1000.0
        out.append(f"this region spans about {km:.0f} km, against the ~{2 * TRAIN_FOOTPRINT_M / 1000:.1f} km "
                   "squares the heads were fitted on — pooling averages a region this size down "
                   "to one vector, so the prediction describes a mixture, not a field")
    return out


def unverifiable_domain(manifest: Dict[str, Any]) -> List[str]:
    """Which domain checks could not run, because the package does not say.

    Reported separately from :func:`domain_warnings` so that a silent skip cannot read as a pass.
    A package the agent or the user assembled by hand has no manifest at all, and that is exactly
    the state in which every check above is weakest.
    """
    if not manifest:
        return ["this package carries no manifest, so none of the region, date or footprint "
                "checks could run — treat the prediction as being of unknown provenance"]
    gaps: List[str] = []
    if not _years_of(manifest):
        gaps.append("the package records no date range, so it could not be checked against the "
                    f"{TRAIN_YEAR} labels the heads learned")
    if _bbox_of(manifest) is None:
        gaps.append(f"the package records no geometry, so it could not be checked against "
                    f"{TRAIN_REGION}")
    return gaps


def repack_pooled(path: Any, out_path: Any, models: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Write a pooled-only copy of the package, carrying the manifest along.

    The service's ``/api/predict_package`` reads the whole uploaded body into memory
    (``np.load(io.BytesIO(await file.read()))``) and then uses nothing but the ``pooled__`` keys.
    Packages in the store run to 216 MB because of their ``grid__`` arrays, so uploading the
    original would move a few hundred megabytes to read a few kilobytes of floats. Reading the
    pooled keys locally costs well under a millisecond; the grid is the expensive part and is not
    wanted. ``meta`` rides along so the uploaded file stays self-describing.
    """
    keep = {str(m) for m in models} if models else None
    vecs, problems = pooled_vectors(path)
    chosen = {m: v for m, v in vecs.items() if keep is None or m in keep}
    payload: Dict[str, Any] = {f"pooled__{m}": v for m, v in chosen.items()}
    manifest = read_manifest(path)
    if manifest:
        payload["meta"] = np.asarray(json.dumps(manifest))
    out = Path(str(out_path))
    np.savez_compressed(out, **payload)
    return {"path": str(out), "models": sorted(chosen),
            "size_bytes": out.stat().st_size if out.exists() else None,
            "problems": problems}


def width_note(dim: int) -> Optional[str]:
    """A sentence naming the other models that share this width, when any do."""
    names = MODELS_BY_WIDTH.get(int(dim)) or ()
    if len(names) < 2:
        return None
    return (f"{dim} dimensions is shared by {', '.join(names)}, so width alone cannot tell one "
            "model's vector from another's — the model here is the one named in the package")


def package_region(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Where and when the package was embedded, for the result and for the action ledger.

    Re-attached to the tool result on purpose. A tool that takes a bbox records its region and
    months as tool ARGUMENTS, and the ledger's whitelist picks them up from there; a tool that
    takes only a ``file_id`` has no such arguments, so without this the row would read
    ``predict_from_package (file_id=...)`` and a later turn could not say what was predicted.
    """
    out: Dict[str, Any] = {}
    box = _bbox_of(manifest)
    if box is not None:
        out["region_bbox"] = [round(float(v), 6) for v in box]
    start, end = manifest.get("start"), manifest.get("end")
    if start and end:
        out["months"] = f"{start}..{end}"
    return out
