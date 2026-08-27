"""Spatial statistics on AREAL data — the PySAL/GeoDa half of the toolkit.

The rest of the analysis modules answer questions about geometry and points: clip this,
buffer that, how many incidents per neighborhood, where are the points densest. None of
them can answer the questions a choropleth actually provokes — "is this pattern real or
could it be noise?", "which neighborhoods are the hot spots?", "does income explain this
once I account for the fact that neighbors resemble each other?", "group these tracts into
eight contiguous regions". Those are spatial-statistics questions, and they need a spatial
weights matrix.

Engines, kept to one per job so the model is never choosing between two ways to compute the
same number:

* ``libpysal`` builds the weights; ``esda`` does global (Moran's I, Geary's C, Getis-Ord G)
  and local (LISA, Gi*) autocorrelation; ``spreg`` does the regression.
* ``pygeoda`` (GeoDa's own libgeoda core) does regionalization — SKATER/REDCAP/AZP/max-p.
  GeoDa's desktop app has no scripting interface, so this binding is the scriptable path to
  its algorithms, and they are the ones PySAL has no direct equivalent of.

Conventions follow :mod:`agent_runtime.analysis_aggregate_tools` exactly and its helpers are
reused verbatim rather than reimplemented: read anything readable via ``file_id``, do the work,
write an EPSG:4326 GeoJSON artifact plus a CSV, and return a ``map_layer`` descriptor that
:func:`agent_runtime.map_layers.build_map_layer` forwards as the ``map_layer`` SSE event. Heavy
imports stay inside the tool bodies so importing this module can never fail the agent boot, and
no tool raises: failures come back as ``{"ok": false, "error": ..., "hint": ...}``.

Two things here are easy to get wrong and are handled once, centrally, because getting them
wrong produces confident nonsense rather than an error:

**Missing values.** Dropping rows with a missing variable AFTER building the weights leaves
the matrix indexed to rows that are gone, silently pairing each observation with the wrong
neighbors. :func:`_prepare` drops first and builds the weights on the surviving subset.

**Islands.** A polygon with no neighbors has no spatial lag, so its LISA/Gi* value is
undefined and it drags the global statistic toward zero. libpysal only warns. Every tool here
reports ``islands`` in its payload, and the local tools mark those rows rather than letting
them read as "not significant".
"""

from __future__ import annotations

import math
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from agent_runtime.analysis_aggregate_tools import (  # reuse, do not reinvent
    _asset,
    _attribute_columns,
    _bad,
    _dumps,
    _ensure_crs,
    _fail,
    _geom_name,
    _label_column,
    _map_layer,
    _label_for,
    _num,
    _numeric_columns,
    _numeric_series,
    _open_vector,
    _source_stem,
    _stem,
    _to_metric,
    _write_csv,
    _write_geojson,
)
from agent_runtime.langchain_geo_tools import _index_attached, artifact_name

# --- weights ------------------------------------------------------------------------

# GeoDa's own vocabulary, so a user who knows the desktop app asks for what they already know.
_WEIGHTS = ("queen", "rook", "knn", "distance_band", "kernel")
_CONTIGUITY = {"queen", "rook"}
# Row-standardisation ("r") is GeoDa's default and the only sane one for Moran's I: it makes the
# spatial lag a neighbour AVERAGE, so a tract with 3 neighbours and one with 12 are comparable.
_TRANSFORMS = ("r", "b", "v", "d", "o")

_SIB = ("For an uploaded shapefile, pass the .shp's file_id (or any single component) — the tool "
        "auto-finds the .shx/.dbf/.prj among the attached files. GeoJSON, GeoPackage, GeoParquet "
        "and a CSV of coordinates all work directly.")

# Areal statistics need enough observations for a permutation reference distribution to mean
# anything. Below this, report the number but say plainly that it is not inferable.
_MIN_OBS = 8


def _fmt(value: Any, digits: int = 4) -> Any:
    """Round for display without turning a real zero into None (``_num`` rounds to 6)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, digits)


def _stars(p: Optional[float]) -> Optional[str]:
    """The significance band, so the answer can say "p < 0.01" without the model doing mental
    arithmetic on a float it may well round wrong."""
    if p is None:
        return None
    for cut, label in ((0.001, "p < 0.001"), (0.01, "p < 0.01"), (0.05, "p < 0.05"),
                       (0.10, "p < 0.10")):
        if p < cut:
            return label
    return "not significant (p >= 0.10)"


def _interpret_global(stat: str, value: float, expected: float, p: Optional[float]) -> str:
    """One plain sentence for a global statistic. The sign convention differs between Moran's I
    and Geary's C (Geary is INVERTED: below its expectation of 1 means clustering), which is a
    classic misreading, so the direction is spelled out here rather than left to the model."""
    if p is None or p >= 0.05:
        return (f"no statistically significant spatial pattern ({stat} = {_fmt(value)}, "
                f"expected {_fmt(expected)} under spatial randomness) — the values are "
                "consistent with being randomly arranged in space")
    if stat == "Geary's C":
        clustered = value < expected
    else:
        clustered = value > expected
    if clustered:
        return (f"significant POSITIVE spatial autocorrelation ({stat} = {_fmt(value)} vs "
                f"{_fmt(expected)} expected): similar values sit next to each other — the map "
                "is clustered, not random")
    return (f"significant NEGATIVE spatial autocorrelation ({stat} = {_fmt(value)} vs "
            f"{_fmt(expected)} expected): neighbours tend to be DISSIMILAR — a checkerboard-"
            "like pattern, which is unusual and worth double-checking against the map")


@contextmanager
def _weights_for(gdf: Any, kind: str, k: int, threshold_km: Optional[float],
                 transform: str, notes: List[str]) -> Iterator[Any]:
    """Yield a libpysal ``W`` for *gdf*, built the way *kind* asks for.

    Contiguity is computed on the geometry as given (topology is CRS-independent), but every
    DISTANCE-based scheme is computed in a projected CRS: a threshold or k-nearest in degrees
    silently means 111 km per unit near the equator and far less near the poles.
    """
    from libpysal import weights

    kind = str(kind or "queen").strip().lower()
    if kind not in _WEIGHTS:
        raise ValueError(f"unsupported weights {kind!r}; use one of {list(_WEIGHTS)}")

    frame = gdf
    if kind not in _CONTIGUITY:
        frame, _ = _to_metric(gdf, notes, "layer")
        # Distance/knn need a single representative coordinate per feature; polygons get their
        # representative point (guaranteed inside the shape, unlike a concave polygon's centroid).
        types = {str(t) for t in frame.geom_type.dropna().unique()}
        if types - {"Point"}:
            frame = frame.copy()
            frame[_geom_name(frame)] = frame.representative_point()
    else:
        types = {str(t) for t in gdf.geom_type.dropna().unique()}
        if types and not (types & {"Polygon", "MultiPolygon"}):
            raise ValueError(
                f"{kind} contiguity needs POLYGONS, but this layer is {'/'.join(sorted(types))}"
                " — for points use weights='knn' (k nearest) or 'distance_band'")

    import warnings

    with warnings.catch_warnings():
        # The island warning is surfaced as structured `islands` output instead.
        warnings.simplefilter("ignore")
        if kind == "queen":
            w = weights.Queen.from_dataframe(frame, use_index=True)
        elif kind == "rook":
            w = weights.Rook.from_dataframe(frame, use_index=True)
        elif kind == "knn":
            kk = max(1, min(int(k), max(1, len(frame) - 1)))
            if kk != int(k):
                notes.append(f"k was clipped from {k} to {kk} (a layer of {len(frame)} features "
                             "cannot have more neighbours than that)")
            w = weights.KNN.from_dataframe(frame, k=kk, use_index=True)
        elif kind == "distance_band":
            if threshold_km is None:
                # min_threshold_distance guarantees every unit has >= 1 neighbour, which is the
                # only defensible automatic choice — a band that leaves islands is worse than
                # one that is slightly too generous.
                thresh = float(weights.min_threshold_distance(
                    [(geom.x, geom.y) for geom in frame.geometry]))
                notes.append(f"threshold_km not given; used the smallest distance that leaves no "
                             f"island ({thresh / 1000.0:.3f} km)")
            else:
                thresh = float(threshold_km) * 1000.0
                if not (thresh > 0) or not math.isfinite(thresh):
                    raise ValueError(f"threshold_km must be a positive number, got {threshold_km!r}")
            w = weights.DistanceBand.from_dataframe(frame, threshold=thresh, use_index=True,
                                                    silence_warnings=True)
        else:
            kk = max(2, min(int(k), max(2, len(frame) - 1)))
            w = weights.Kernel.from_dataframe(frame, fixed=False, k=kk, use_index=True)

    tr = str(transform or "r").strip().lower()
    if tr not in _TRANSFORMS:
        raise ValueError(f"unsupported transform {tr!r}; use one of {list(_TRANSFORMS)}")
    w.transform = tr
    yield w


def _weights_diagnostics(w: Any) -> Dict[str, Any]:
    """The connectivity facts that decide whether a result is trustworthy."""
    cards = list(w.cardinalities.values())
    return {
        "n": int(w.n),
        "islands": len(w.islands),
        "min_neighbors": int(min(cards)) if cards else 0,
        "max_neighbors": int(max(cards)) if cards else 0,
        "mean_neighbors": _fmt(w.mean_neighbors, 2),
        "median_neighbors": _fmt(sorted(cards)[len(cards) // 2], 1) if cards else None,
        "pct_nonzero": _fmt(w.pct_nonzero, 3),
    }


def _island_note(w: Any, notes: List[str]) -> None:
    if w.islands:
        n = len(w.islands)
        notes.append(
            f"{n} feature(s) have NO neighbours under these weights (islands). They have no "
            "spatial lag, so their local statistic is undefined and they pull the global "
            "statistic toward zero. Try weights='knn' (every feature then has exactly k "
            "neighbours) or a larger distance_band if this matters.")


def _prepare(gdf: Any, columns: List[str], notes: List[str]) -> Tuple[Any, Optional[str]]:
    """Drop rows unusable for the analysis, BEFORE any weights get built.

    Returns ``(subset, error)``. Rows are dropped for a missing/non-numeric value in any of
    *columns* or for missing geometry. Building weights first and subsetting after is the
    subtle bug this exists to prevent: the W would still be indexed to the dropped rows.
    """
    frame = gdf.copy()
    for col in columns:
        series = _numeric_series(frame, col, notes)
        if series is None:
            return frame, col
        frame[col] = series
    geom = _geom_name(frame)
    before = len(frame)
    frame = frame[frame[geom].notna() & ~frame[geom].is_empty]
    frame = frame.dropna(subset=columns)
    dropped = before - len(frame)
    if dropped:
        notes.append(f"dropped {dropped} of {before} feature(s) with a missing value in "
                     f"{', '.join(columns)} or no geometry; the weights were built on the "
                     f"remaining {len(frame)}, which is the only correct order")
    return frame.reset_index(drop=True), None


def _too_few(frame: Any, notes: List[str]) -> Optional[str]:
    if len(frame) < _MIN_OBS:
        return _bad(f"only {len(frame)} usable feature(s); spatial statistics need at least "
                    f"{_MIN_OBS} to say anything about a pattern",
                    hint="aggregate to fewer, larger areas, or use a layer with more features",
                    usable_features=int(len(frame)), notes=notes or None)
    return None


# --- categorical map layers ----------------------------------------------------------

# GeoDa's cluster-map colours, sent WITH the layer as a legend rather than hardcoded in the
# client: the map client shades a categorical layer straight from these, so a new categorical
# tool needs no client change, and the labels the user reads are the labels the tool assigned.
_LISA_COLORS: Dict[str, List[int]] = {
    "High-High": [215, 25, 28, 200],
    "Low-Low": [44, 123, 182, 200],
    "Low-High": [171, 217, 233, 200],
    "High-Low": [253, 174, 97, 200],
    "Not significant": [222, 222, 222, 160],
    "Island (no neighbours)": [150, 150, 150, 120],
}
_HOTSPOT_COLORS: Dict[str, List[int]] = {
    "Hot spot (p < 0.01)": [178, 24, 43, 210],
    "Hot spot (p < 0.05)": [239, 138, 98, 200],
    "Cold spot (p < 0.05)": [146, 197, 222, 200],
    "Cold spot (p < 0.01)": [33, 102, 172, 210],
    "Not significant": [222, 222, 222, 160],
    "Island (no neighbours)": [150, 150, 150, 120],
}


def _legend(colors: Dict[str, List[int]], present: Any) -> List[Dict[str, Any]]:
    """Legend entries for the classes that ACTUALLY occur, in the dictionary's order.

    Listing absent classes would invite the answer to describe hot spots that are not on the
    map, so the legend is filtered to what was really assigned.
    """
    seen = {str(v) for v in present}
    return [{"label": label, "color": rgba} for label, rgba in colors.items() if label in seen]


def _categorical_layer(rec: Dict[str, Any], label: str, style_by: str, count: int,
                       legend: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A ``map_layer`` whose classes are CATEGORIES, not a number to ramp.

    ``render="categories"`` matters: the client's choropleth ramp coerces the style column with
    Number(), so a categorical column like "High-High" becomes NaN for every feature and the
    layer renders in one flat colour — a map that looks fine and says nothing.
    """
    out = _map_layer(rec, label, "categories", style_by, count)
    out["legend"] = legend
    return out


# --- the tools -----------------------------------------------------------------------


def make_spatial_stats_tools(default_input_file_ids: Optional[List[str]] = None) -> List[Any]:
    """Build the spatial-statistics StructuredTools (libpysal / esda / spreg / pygeoda).

    ``default_input_file_ids`` is the conversation's attached file set, used only to
    auto-discover shapefile sidecars by basename exactly as the other analysis modules do.

    Weights are rebuilt from their PARAMETERS inside every tool rather than passed between
    calls as an object or a file. A saved weights matrix is indexed to the exact rows it was
    built from, so reusing one across calls that drop different missing values is a silent
    misalignment; rebuilding is cheap (queen contiguity on a few thousand polygons is
    milliseconds) and always consistent with the layer in hand.
    """
    from langchain_core.tools import StructuredTool

    attached = _index_attached(default_input_file_ids)

    def _open(ref, siblings=None, layer=None):
        return _open_vector(ref, siblings, attached, layer)

    def _load(file_id, siblings, columns, notes, layer=None):
        """Open, validate CRS, coerce the needed columns, drop unusable rows.

        Returns ``(frame, error_json)`` — ``error_json`` is already a finished tool response.
        """
        with _open(file_id, siblings, layer) as raw:
            gdf = _ensure_crs(raw, notes, "input layer")
        if len(gdf) == 0:
            return None, _bad("the input layer has no features")
        for col in columns:
            if col not in gdf.columns:
                cands = _numeric_columns(gdf)
                return None, _bad(
                    f"column {col!r} is not in the data",
                    hint="pick one of numeric_columns — spatial statistics need a number per area "
                         "(a rate, count, income, index...)",
                    numeric_columns=cands[:40], numeric_column_count=len(cands),
                    all_columns=_attribute_columns(gdf)[:60])
        frame, bad_col = _prepare(gdf, list(columns), notes)
        if bad_col:
            cands = _numeric_columns(gdf)
            return None, _bad(f"column {bad_col!r} is not numeric and could not be coerced",
                              hint="pick one of numeric_columns",
                              numeric_columns=cands[:40])
        err = _too_few(frame, notes)
        if err:
            return None, err
        return frame, None

    # --- 1. the weights matrix itself -------------------------------------------------

    def spatial_weights(file_id: str, weights: str = "queen", k: int = 6,
                        threshold_km: Optional[float] = None, transform: str = "r",
                        name: Optional[str] = None,
                        siblings: Optional[List[str]] = None) -> str:
        """Build a SPATIAL WEIGHTS matrix ("who is next to whom") and report its connectivity.

        The object every other spatial statistic depends on. Reports neighbour counts, islands
        and sparsity, and saves a GeoDa-compatible .gal/.gwt file. Run this first when a
        LISA/Moran result looks strange — the weights are usually why.
        """
        notes: List[str] = []
        try:
            with _open(file_id, siblings) as raw:
                gdf = _ensure_crs(raw, notes, "input layer")
            if len(gdf) == 0:
                return _bad("the input layer has no features")

            import numpy as np

            with _weights_for(gdf, weights, k, threshold_km, transform, notes) as w:
                diag = _weights_diagnostics(w)
                _island_note(w, notes)
                cards = np.asarray(list(w.cardinalities.values()))
                # A histogram beats min/max for spotting a broken matrix: "most features have 5-6
                # neighbours but 40 have 1" is the shape of a bad distance band.
                hist: Dict[str, int] = {}
                for value in sorted(set(cards.tolist())):
                    hist[str(int(value))] = int((cards == value).sum())
                if len(hist) > 20:
                    hist = dict(list(hist.items())[:20])
                    notes.append("neighbour histogram truncated to the 20 smallest counts")

                # .gal is the interchange format GeoDa itself reads, so the matrix built here can
                # be opened in the desktop app or fed to pygeoda later.
                from agent_runtime.file_store import create_output_file_from_path
                import libpysal

                suffix = "gwt" if weights in {"knn", "kernel"} else "gal"
                base = _stem(name, _source_stem(file_id), f"{weights}_weights")
                fname = artifact_name(base, suffix, source=_source_stem(file_id),
                                      default=f"{weights}_weights")
                tmpdir = Path(tempfile.mkdtemp(prefix="w_gal_"))
                try:
                    out_path = tmpdir / fname
                    handle = libpysal.io.open(str(out_path), "w")
                    handle.write(w)
                    handle.close()
                    rec = create_output_file_from_path(out_path, filename=fname)
                except Exception as write_exc:      # the diagnostics are the point, not the file
                    rec = None
                    notes.append(f"weights file not written ({type(write_exc).__name__}: "
                                 f"{write_exc}); the diagnostics below are complete")
                finally:
                    shutil.rmtree(tmpdir, ignore_errors=True)

            payload: Dict[str, Any] = {
                "ok": True, "weights": weights, "transform": transform,
                "k": int(k) if weights in {"knn", "kernel"} else None,
                "threshold_km": _fmt(threshold_km) if weights == "distance_band" else None,
                "connectivity": diag,
                "neighbor_count_histogram": hist,
                "interpretation": (
                    f"each feature has {diag['min_neighbors']}-{diag['max_neighbors']} neighbours "
                    f"(mean {diag['mean_neighbors']})"
                    + (f"; {diag['islands']} have NONE" if diag["islands"] else "; no islands")),
            }
            if rec:
                payload.update({"file_id": rec["file_id"], "filename": rec.get("filename"),
                                "download_url": rec.get("download_url"), "weights_file": _asset(rec)})
            if notes:
                payload["notes"] = notes
            return _dumps(payload)
        except ImportError as exc:
            return _fail(exc, "libpysal is not installed in this deployment")
        except Exception as exc:
            return _fail(exc, _SIB)

    # --- 2. global autocorrelation: "is this pattern real?" ---------------------------

    def global_spatial_autocorrelation(file_id: str, column: str, weights: str = "queen",
                                      k: int = 6, threshold_km: Optional[float] = None,
                                      permutations: int = 999,
                                      siblings: Optional[List[str]] = None) -> str:
        """Test whether a variable is spatially CLUSTERED at all: Moran's I, Geary's C, Getis-Ord G.

        The first question to ask of any choropleth — is the pattern real, or could a random
        arrangement look like this? Returns each statistic with its expected value under spatial
        randomness, a pseudo p-value from `permutations` random reshufflings, and a plain-language
        verdict. Run this BEFORE local_moran_lisa: if nothing is globally significant, hunting
        local clusters is usually chasing noise.
        """
        notes: List[str] = []
        try:
            import esda

            frame, err = _load(file_id, siblings, [column], notes)
            if err:
                return err
            perms = max(99, min(int(permutations or 999), 99999))
            y = frame[column].to_numpy(dtype="float64")

            import warnings

            with _weights_for(frame, weights, k, threshold_km, "r", notes) as w:
                diag = _weights_diagnostics(w)
                _island_note(w, notes)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    moran = esda.Moran(y, w, permutations=perms)
                    geary = esda.Geary(y, w, permutations=perms)
                    try:
                        # Getis-Ord G is only defined for a non-negative variable (it is a ratio
                        # of weighted to total sums), so a z-score column would make it nonsense.
                        getis = esda.G(y, w, permutations=perms) if (y >= 0).all() else None
                    except Exception:
                        getis = None

            results: Dict[str, Any] = {
                "morans_i": {
                    "statistic": _fmt(moran.I), "expected": _fmt(moran.EI),
                    "z_score": _fmt(moran.z_sim, 3), "p_value": _fmt(moran.p_sim, 5),
                    "significance": _stars(float(moran.p_sim)),
                    "interpretation": _interpret_global("Moran's I", float(moran.I),
                                                        float(moran.EI), float(moran.p_sim)),
                },
                "gearys_c": {
                    "statistic": _fmt(geary.C), "expected": _fmt(geary.EC),
                    "z_score": _fmt(geary.z_sim, 3), "p_value": _fmt(geary.p_sim, 5),
                    "significance": _stars(float(geary.p_sim)),
                    "interpretation": _interpret_global("Geary's C", float(geary.C),
                                                        float(geary.EC), float(geary.p_sim)),
                },
            }
            if getis is not None:
                results["getis_ord_g"] = {
                    "statistic": _fmt(getis.G), "expected": _fmt(getis.EG),
                    "z_score": _fmt(getis.z_sim, 3), "p_value": _fmt(getis.p_sim, 5),
                    "significance": _stars(float(getis.p_sim)),
                    "interpretation": ("high values cluster together"
                                       if float(getis.G) > float(getis.EG)
                                       else "low values cluster together")
                    + (" (significant)" if float(getis.p_sim) < 0.05 else " (NOT significant)"),
                }
            else:
                notes.append("Getis-Ord G was skipped: it is only defined for a non-negative "
                             "variable, and this column has negative values")

            return _dumps({
                "ok": True, "column": column, "features_analyzed": int(len(frame)),
                "weights": weights, "transform": "r", "permutations": perms,
                "connectivity": diag,
                "results": results,
                "verdict": results["morans_i"]["interpretation"],
                "next_step": ("run local_moran_lisa to see WHERE the clusters are"
                              if float(moran.p_sim) < 0.05
                              else "the global pattern is not significant; local clusters may "
                                   "still exist but treat them cautiously"),
                "notes": notes or None,
            })
        except ImportError as exc:
            return _fail(exc, "esda/libpysal are not installed in this deployment")
        except Exception as exc:
            return _fail(exc, _SIB)

    # --- 3. LISA: "WHERE are the clusters?" -------------------------------------------

    def local_moran_lisa(file_id: str, column: str, weights: str = "queen", k: int = 6,
                         threshold_km: Optional[float] = None, permutations: int = 999,
                         significance: float = 0.05, name: Optional[str] = None,
                         label_column: Optional[str] = None,
                         siblings: Optional[List[str]] = None) -> str:
        """LISA cluster map (Local Moran's I) — WHERE the hot and cold spots are, on the map.

        Classifies every area as High-High (a hot spot: high value among high neighbours),
        Low-Low (a cold spot), High-Low / Low-High (spatial outliers — a high area surrounded by
        low ones), or Not significant. Returns the classified layer as a CATEGORICAL map layer
        with a legend, plus a CSV of the per-area statistic and p-value. This is GeoDa's cluster
        map, and the answer to "which neighborhoods are the hot spots".
        """
        notes: List[str] = []
        try:
            import esda
            import numpy as np

            frame, err = _load(file_id, siblings, [column], notes)
            if err:
                return err
            alpha = float(significance or 0.05)
            if not (0 < alpha < 1):
                return _bad(f"significance must be between 0 and 1, got {significance!r}")
            perms = max(99, min(int(permutations or 999), 99999))
            y = frame[column].to_numpy(dtype="float64")

            import warnings

            with _weights_for(frame, weights, k, threshold_km, "r", notes) as w:
                diag = _weights_diagnostics(w)
                _island_note(w, notes)
                island_idx = set(int(i) for i in w.islands)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    lisa = esda.Moran_Local(y, w, permutations=perms, seed=42)

            # libpysal quadrant codes: 1=HH, 2=LH, 3=LL, 4=HL. Getting this mapping wrong swaps
            # the outliers for the clusters, which is why it is written out literally.
            quad = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}
            p_sim = np.asarray(lisa.p_sim, dtype="float64")
            classes: List[str] = []
            for i in range(len(frame)):
                if i in island_idx:
                    classes.append("Island (no neighbours)")
                elif p_sim[i] < alpha:
                    classes.append(quad.get(int(lisa.q[i]), "Not significant"))
                else:
                    classes.append("Not significant")

            out = frame.copy()
            out["lisa_class"] = classes
            out["lisa_i"] = [_num(v) for v in np.asarray(lisa.Is, dtype="float64")]
            out["lisa_z"] = [_num(v) for v in np.asarray(lisa.z_sim, dtype="float64")]
            out["lisa_p"] = [_num(v) for v in p_sim]
            out["lisa_quadrant"] = [int(q) for q in np.asarray(lisa.q)]

            counts: Dict[str, int] = {}
            for cls in classes:
                counts[cls] = counts.get(cls, 0) + 1

            base = _stem(name, _source_stem(file_id), f"{column}_lisa")
            rec = _write_geojson(out, base, _source_stem(file_id), "lisa")

            label_col = _label_column(out, label_column)
            csv_cols = [c for c in ([label_col] if label_col else []) +
                        [column, "lisa_class", "lisa_i", "lisa_z", "lisa_p"] if c in out.columns]
            csv_frame = out[csv_cols].copy()
            # Significant areas first: the answer needs to NAME the hot spots, and a CSV in
            # arbitrary feature order buries them among the not-significant majority.
            order = {"High-High": 0, "Low-Low": 1, "High-Low": 2, "Low-High": 3,
                     "Island (no neighbours)": 4, "Not significant": 5}
            csv_frame = csv_frame.assign(_o=[order.get(c, 9) for c in classes]) \
                .sort_values(["_o", "lisa_p"]).drop(columns=["_o"])
            csv_rec = _write_csv(csv_frame, base, _source_stem(file_id), "lisa")

            significant = sum(v for kls, v in counts.items()
                              if kls not in {"Not significant", "Island (no neighbours)"})
            top = csv_frame.head(10).to_dict(orient="records")
            return _dumps({
                "ok": True, "file_id": rec["file_id"], "filename": rec.get("filename"),
                "download_url": rec.get("download_url"),
                "column": column, "features_analyzed": int(len(frame)),
                "weights": weights, "permutations": perms, "significance_level": alpha,
                "connectivity": diag,
                "class_counts": counts,
                "significant_features": int(significant),
                "interpretation": (
                    f"{counts.get('High-High', 0)} hot-spot area(s) (High-High), "
                    f"{counts.get('Low-Low', 0)} cold-spot area(s) (Low-Low), "
                    f"{counts.get('High-Low', 0) + counts.get('Low-High', 0)} spatial outlier(s), "
                    f"out of {len(frame)} at p < {alpha}"),
                "top_features": top,
                "csv": _asset(csv_rec),
                "map_layer": _categorical_layer(rec, _label_for(name, rec) or f"{column} LISA",
                                                "lisa_class", int(len(out)),
                                                _legend(_LISA_COLORS, classes)),
                "notes": notes or None,
            })
        except ImportError as exc:
            return _fail(exc, "esda/libpysal are not installed in this deployment")
        except Exception as exc:
            return _fail(exc, _SIB)

    # --- 4. Getis-Ord Gi*: hot/cold spots by intensity --------------------------------

    def local_getis_ord(file_id: str, column: str, weights: str = "queen", k: int = 6,
                        threshold_km: Optional[float] = None, star: bool = True,
                        permutations: int = 999, name: Optional[str] = None,
                        label_column: Optional[str] = None,
                        siblings: Optional[List[str]] = None) -> str:
        """Getis-Ord Gi* HOT SPOT / COLD SPOT map — clusters of high or low values by intensity.

        The other standard hot-spot statistic (ArcGIS's "Hotspot Analysis"). Where LISA
        distinguishes clusters from outliers, Gi* answers "is this a cluster of HIGH values or
        of LOW values", banded by confidence. `star=True` includes each area itself in its own
        neighbourhood (Gi*, the usual choice); False gives classic Gi. Returns a categorical map
        layer with a legend plus a CSV of z-scores and p-values.
        """
        notes: List[str] = []
        try:
            import esda
            import numpy as np

            frame, err = _load(file_id, siblings, [column], notes)
            if err:
                return err
            perms = max(99, min(int(permutations or 999), 99999))
            y = frame[column].to_numpy(dtype="float64")

            import warnings

            with _weights_for(frame, weights, k, threshold_km, "r", notes) as w:
                diag = _weights_diagnostics(w)
                _island_note(w, notes)
                island_idx = set(int(i) for i in w.islands)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    g = esda.G_Local(y, w, permutations=perms, star=bool(star), seed=42)

            z = np.asarray(g.Zs, dtype="float64")
            p = np.asarray(g.p_sim, dtype="float64")
            classes: List[str] = []
            for i in range(len(frame)):
                if i in island_idx:
                    classes.append("Island (no neighbours)")
                    continue
                # The z-score's SIGN says hot or cold; the p-value says whether to believe it.
                if p[i] < 0.01:
                    classes.append("Hot spot (p < 0.01)" if z[i] > 0 else "Cold spot (p < 0.01)")
                elif p[i] < 0.05:
                    classes.append("Hot spot (p < 0.05)" if z[i] > 0 else "Cold spot (p < 0.05)")
                else:
                    classes.append("Not significant")

            out = frame.copy()
            out["hotspot_class"] = classes
            out["gi_z"] = [_num(v) for v in z]
            out["gi_p"] = [_num(v) for v in p]

            counts: Dict[str, int] = {}
            for cls in classes:
                counts[cls] = counts.get(cls, 0) + 1

            stat_name = "Gi*" if star else "Gi"
            base = _stem(name, _source_stem(file_id), f"{column}_hotspots")
            rec = _write_geojson(out, base, _source_stem(file_id), "hotspots")

            label_col = _label_column(out, label_column)
            csv_cols = [c for c in ([label_col] if label_col else []) +
                        [column, "hotspot_class", "gi_z", "gi_p"] if c in out.columns]
            csv_frame = out[csv_cols].copy().sort_values("gi_z", ascending=False)
            csv_rec = _write_csv(csv_frame, base, _source_stem(file_id), "hotspots")

            hot = sum(v for kls, v in counts.items() if kls.startswith("Hot"))
            cold = sum(v for kls, v in counts.items() if kls.startswith("Cold"))
            return _dumps({
                "ok": True, "file_id": rec["file_id"], "filename": rec.get("filename"),
                "download_url": rec.get("download_url"),
                "column": column, "statistic": stat_name,
                "features_analyzed": int(len(frame)), "weights": weights, "permutations": perms,
                "connectivity": diag, "class_counts": counts,
                "interpretation": (
                    f"{hot} hot-spot area(s) (clusters of HIGH {column}) and {cold} cold-spot "
                    f"area(s) (clusters of LOW {column}) out of {len(frame)}, by {stat_name}"),
                "hottest": csv_frame.head(10).to_dict(orient="records"),
                "coldest": csv_frame.tail(10).to_dict(orient="records")[::-1],
                "csv": _asset(csv_rec),
                "map_layer": _categorical_layer(rec,
                                                _label_for(name, rec) or f"{column} {stat_name}",
                                                "hotspot_class", int(len(out)),
                                                _legend(_HOTSPOT_COLORS, classes)),
                "notes": notes or None,
            })
        except ImportError as exc:
            return _fail(exc, "esda/libpysal are not installed in this deployment")
        except Exception as exc:
            return _fail(exc, _SIB)

    # --- 5. the Moran scatterplot -----------------------------------------------------

    def moran_scatterplot(file_id: str, column: str, weights: str = "queen", k: int = 6,
                          threshold_km: Optional[float] = None, permutations: int = 999,
                          annotate: bool = True, name: Optional[str] = None,
                          siblings: Optional[List[str]] = None) -> str:
        """The MORAN SCATTERPLOT as a PNG: each area's value against its neighbours' average.

        GeoDa's signature plot. The regression slope IS Moran's I, and the four quadrants are the
        LISA classes, so it shows the strength of the pattern and the individual outliers in one
        picture. Returns a downloadable PNG plus the statistic. Use it to SHOW a result that
        global_spatial_autocorrelation states numerically.
        """
        notes: List[str] = []
        try:
            import esda
            import numpy as np
            from libpysal.weights import lag_spatial

            frame, err = _load(file_id, siblings, [column], notes)
            if err:
                return err
            perms = max(99, min(int(permutations or 999), 99999))
            y = frame[column].to_numpy(dtype="float64")

            import warnings

            with _weights_for(frame, weights, k, threshold_km, "r", notes) as w:
                _island_note(w, notes)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    moran = esda.Moran(y, w, permutations=perms)
                    std = (y - y.mean()) / (y.std() or 1.0)
                    lag = np.asarray(lag_spatial(w, std), dtype="float64")

            import matplotlib
            matplotlib.use("Agg")            # headless: no display in the runtime
            import matplotlib.pyplot as plt
            from agent_runtime.file_store import create_output_file_from_path

            # Quadrant colours match the LISA cluster map's, so the plot and the map read as one
            # analysis rather than two unrelated pictures.
            colors = np.where((std > 0) & (lag > 0), "#d7191c",
                     np.where((std < 0) & (lag < 0), "#2c7bb6",
                     np.where((std > 0) & (lag < 0), "#fdae61", "#abd9e9")))
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.scatter(std, lag, c=colors, s=28, edgecolor="#33333355", linewidth=0.5, zorder=3)
            ax.axhline(0, color="#888888", linewidth=0.8, zorder=1)
            ax.axvline(0, color="#888888", linewidth=0.8, zorder=1)
            grid = np.linspace(float(std.min()), float(std.max()), 50)
            ax.plot(grid, float(moran.I) * grid, color="#111111", linewidth=1.6, zorder=4)
            if annotate:
                # Anchor the quadrant labels to the AXIS CORNERS, not to the data's own extent:
                # keyed off max() alone they landed inside the point cloud on a skewed
                # distribution (observed "Low-Low" printed across its own points).
                x_lo, x_hi = ax.get_xlim()
                y_lo, y_hi = ax.get_ylim()
                pad_x, pad_y = (x_hi - x_lo) * 0.04, (y_hi - y_lo) * 0.04
                for label, xx, yy, ha, va, col in (
                        ("High-High", x_hi - pad_x, y_hi - pad_y, "right", "top", "#d7191c"),
                        ("Low-High", x_lo + pad_x, y_hi - pad_y, "left", "top", "#4a9fc4"),
                        ("Low-Low", x_lo + pad_x, y_lo + pad_y, "left", "bottom", "#2c7bb6"),
                        ("High-Low", x_hi - pad_x, y_lo + pad_y, "right", "bottom", "#e08214")):
                    # Under strong autocorrelation the fit line runs corner to corner, straight
                    # through two of these labels, so each gets a backing box to stay legible.
                    ax.text(xx, yy, label, color=col, fontsize=9, fontweight="bold",
                            ha=ha, va=va, zorder=5,
                            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                      edgecolor="none", alpha=0.75))
            ax.set_xlabel(f"{column} (standardised)")
            ax.set_ylabel(f"spatial lag of {column} (neighbours' average)")
            ax.set_title(f"Moran scatterplot — {column}\nMoran's I = {_fmt(moran.I)} "
                         f"(the line's slope), p = {_fmt(moran.p_sim, 5)} "
                         f"from {perms} permutations")
            fig.tight_layout()

            base = _stem(name, _source_stem(file_id), f"{column}_moran_scatter")
            png_dir = Path(tempfile.mkdtemp(prefix="moran_png_"))
            try:
                png_name = artifact_name(base, "png", source=_source_stem(file_id),
                                         default="moran_scatterplot")
                png_path = png_dir / png_name
                fig.savefig(png_path, dpi=150)
                plt.close(fig)
                rec = create_output_file_from_path(png_path, filename=png_name)
            finally:
                shutil.rmtree(png_dir, ignore_errors=True)

            return _dumps({
                "ok": True, "file_id": rec["file_id"], "filename": rec.get("filename"),
                "download_url": rec.get("download_url"), "plot": _asset(rec),
                "column": column, "features_analyzed": int(len(frame)), "weights": weights,
                "morans_i": _fmt(moran.I), "p_value": _fmt(moran.p_sim, 5),
                "significance": _stars(float(moran.p_sim)),
                "interpretation": _interpret_global("Moran's I", float(moran.I), float(moran.EI),
                                                    float(moran.p_sim)),
                "notes": notes or None,
            })
        except ImportError as exc:
            return _fail(exc, "esda/libpysal/matplotlib are not installed in this deployment")
        except Exception as exc:
            return _fail(exc, _SIB)

    # --- 6. spatial regression --------------------------------------------------------

    def spatial_regression(file_id: str, y_column: str, x_columns: List[str],
                           model: str = "auto", weights: str = "queen", k: int = 6,
                           threshold_km: Optional[float] = None, name: Optional[str] = None,
                           siblings: Optional[List[str]] = None) -> str:
        """SPATIAL REGRESSION: does X explain Y once neighbouring areas' dependence is accounted for?

        Plain OLS on areal data is usually wrong — neighbours resemble each other, so residuals
        are correlated and the standard errors lie. This fits OLS WITH spatial diagnostics
        (Lagrange Multiplier tests) and, with `model="auto"`, follows the standard decision rule
        to refit as a spatial LAG model (spillover: the neighbours' Y affects mine) or a spatial
        ERROR model (a shared unobserved factor). `model` can also be "ols", "lag" or "error"
        explicitly. Returns the coefficient table, fit statistics, the diagnostics behind the
        choice, and a map layer of the residuals so remaining structure is visible.
        """
        notes: List[str] = []
        try:
            import numpy as np
            import spreg

            xs = [str(c) for c in (x_columns or []) if str(c).strip()]
            if not xs:
                return _bad("x_columns is empty — name at least one explanatory column",
                            hint="x_columns is a list, e.g. ['median_income', 'pct_renter']")
            if y_column in xs:
                return _bad(f"y_column {y_column!r} also appears in x_columns",
                            hint="the dependent variable cannot explain itself")
            want = str(model or "auto").strip().lower()
            if want not in {"auto", "ols", "lag", "error"}:
                return _bad(f"unsupported model {want!r}", hint="use auto|ols|lag|error")

            frame, err = _load(file_id, siblings, [y_column] + xs, notes)
            if err:
                return err
            if len(frame) <= len(xs) + 2:
                return _bad(f"{len(frame)} usable feature(s) is too few to fit {len(xs)} "
                            "explanatory variable(s)",
                            hint="use fewer x_columns or a layer with more features")

            Y = frame[[y_column]].to_numpy(dtype="float64")
            X = frame[xs].to_numpy(dtype="float64")
            # A constant column has no variance to explain anything with and makes X singular.
            flat = [c for c in xs if float(np.nanstd(frame[c].to_numpy(dtype="float64"))) == 0.0]
            if flat:
                return _bad(f"column(s) {flat} are constant across every feature",
                            hint="a variable with no variation cannot explain anything; drop it")

            import warnings

            with _weights_for(frame, weights, k, threshold_km, "r", notes) as w:
                diag = _weights_diagnostics(w)
                _island_note(w, notes)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ols = spreg.OLS(Y, X, w=w, spat_diag=True, moran=True,
                                    name_y=y_column, name_x=list(xs), name_w=weights,
                                    name_ds=str(Path(str(_source_stem(file_id) or "layer")).name))

                    lm_lag_p = float(ols.lm_lag[1])
                    lm_err_p = float(ols.lm_error[1])
                    rlm_lag_p = float(ols.rlm_lag[1])
                    rlm_err_p = float(ols.rlm_error[1])
                    # Anselin's specification-search rule: consult the ROBUST LM tests, which are
                    # each adjusted for the other form of dependence, and take the smaller p-value
                    # only when the plain test also flagged something.
                    if want == "auto":
                        if min(lm_lag_p, lm_err_p) >= 0.05:
                            chosen, why = "ols", ("neither LM test is significant, so OLS is "
                                                  "adequate — no spatial dependence to model")
                        elif rlm_lag_p < rlm_err_p:
                            chosen, why = "lag", (f"robust LM-lag (p = {_fmt(rlm_lag_p, 5)}) beats "
                                                  f"robust LM-error (p = {_fmt(rlm_err_p, 5)}), "
                                                  "pointing to a spillover process")
                        else:
                            chosen, why = "error", (f"robust LM-error (p = {_fmt(rlm_err_p, 5)}) "
                                                    f"beats robust LM-lag (p = {_fmt(rlm_lag_p, 5)}"
                                                    "), pointing to a shared unobserved factor")
                    else:
                        chosen, why = want, f"model={want} was requested explicitly"

                    fitted: Any = ols
                    if chosen == "lag":
                        fitted = spreg.ML_Lag(Y, X, w=w, name_y=y_column, name_x=list(xs),
                                              name_w=weights)
                    elif chosen == "error":
                        fitted = spreg.ML_Error(Y, X, w=w, name_y=y_column, name_x=list(xs),
                                                name_w=weights)

            def _coefficients(mod: Any) -> List[Dict[str, Any]]:
                """The coefficient table, aligned to spreg's own variable names.

                ML_Lag/ML_Error append the spatial parameter to `betas` but NOT to `name_x`, so
                zipping the two naively drops it. Read the length actually present instead.
                """
                betas = np.asarray(mod.betas, dtype="float64").flatten()
                stats = list(getattr(mod, "z_stat", None) or getattr(mod, "t_stat", None) or [])
                errs = np.asarray(getattr(mod, "std_err", []), dtype="float64").flatten()
                names = list(getattr(mod, "name_x", []) or [])
                if chosen == "lag":
                    names = names + ["W_" + y_column + " (rho, spatial lag)"]
                elif chosen == "error":
                    names = names + ["lambda (spatial error)"]
                rows: List[Dict[str, Any]] = []
                for i, beta in enumerate(betas):
                    stat = stats[i] if i < len(stats) else (None, None)
                    rows.append({
                        "variable": names[i] if i < len(names) else f"beta_{i}",
                        "coefficient": _fmt(beta, 5),
                        "std_error": _fmt(errs[i], 5) if i < len(errs) else None,
                        "statistic": _fmt(stat[0], 3) if stat and stat[0] is not None else None,
                        "p_value": _fmt(stat[1], 5) if stat and stat[1] is not None else None,
                        "significant_at_05": (bool(stat[1] < 0.05)
                                              if stat and stat[1] is not None else None),
                    })
                return rows

            coefs = _coefficients(fitted)
            fit: Dict[str, Any] = {
                "n": int(len(frame)), "k": len(xs),
                "r_squared": _fmt(getattr(fitted, "r2", None)),
                "pseudo_r_squared": _fmt(getattr(fitted, "pr2", None)),
                "log_likelihood": _fmt(getattr(fitted, "logll", None), 3),
                "aic": _fmt(getattr(fitted, "aic", None), 3),
                "schwarz": _fmt(getattr(fitted, "schwarz", None), 3),
            }
            if chosen == "lag":
                fit["rho_spatial_lag"] = _fmt(np.asarray(
                    getattr(fitted, "rho", np.nan), dtype="float64").flatten()[0], 5)
            if chosen == "error":
                fit["lambda_spatial_error"] = _fmt(np.asarray(
                    getattr(fitted, "lam", np.nan), dtype="float64").flatten()[0], 5)

            diagnostics = {
                "moran_residuals": {"statistic": _fmt(ols.moran_res[0]),
                                    "z_score": _fmt(ols.moran_res[1], 3),
                                    "p_value": _fmt(ols.moran_res[2], 5)}
                if getattr(ols, "moran_res", None) else None,
                "lm_lag": {"statistic": _fmt(ols.lm_lag[0], 3), "p_value": _fmt(lm_lag_p, 5)},
                "lm_error": {"statistic": _fmt(ols.lm_error[0], 3), "p_value": _fmt(lm_err_p, 5)},
                "robust_lm_lag": {"statistic": _fmt(ols.rlm_lag[0], 3),
                                  "p_value": _fmt(rlm_lag_p, 5)},
                "robust_lm_error": {"statistic": _fmt(ols.rlm_error[0], 3),
                                    "p_value": _fmt(rlm_err_p, 5)},
                "lm_sarma": {"statistic": _fmt(ols.lm_sarma[0], 3),
                             "p_value": _fmt(ols.lm_sarma[1], 5)},
                "jarque_bera_normality": ({"statistic": _fmt(ols.jarque_bera["jb"], 3),
                                           "p_value": _fmt(ols.jarque_bera["pvalue"], 5)}
                                          if getattr(ols, "jarque_bera", None) else None),
                "koenker_bassett_heteroskedasticity": (
                    {"statistic": _fmt(ols.koenker_bassett["kb"], 3),
                     "p_value": _fmt(ols.koenker_bassett["pvalue"], 5)}
                    if getattr(ols, "koenker_bassett", None) else None),
            }

            # The residual map is the honest check on the whole exercise: structure still visible
            # here means the model has not captured the geography, whatever the p-values say.
            out = frame.copy()
            resid = np.asarray(fitted.u, dtype="float64").flatten()
            out["residual"] = [_num(v) for v in resid]
            out["fitted"] = [_num(v) for v in
                             np.asarray(fitted.predy, dtype="float64").flatten()]
            base = _stem(name, _source_stem(file_id), f"{y_column}_{chosen}_residuals")
            rec = _write_geojson(out, base, _source_stem(file_id), "residuals")
            import pandas as pd

            csv_rec = _write_csv(pd.DataFrame(coefs), base, _source_stem(file_id), "coefficients")

            model_label = {"ols": "OLS (no spatial term)",
                           "lag": "spatial lag (ML_Lag)",
                           "error": "spatial error (ML_Error)"}[chosen]
            return _dumps({
                "ok": True, "file_id": rec["file_id"], "filename": rec.get("filename"),
                "download_url": rec.get("download_url"),
                "model": chosen, "model_label": model_label, "model_choice_reason": why,
                "requested_model": want,
                "y_column": y_column, "x_columns": xs, "weights": weights,
                "connectivity": diag,
                "coefficients": coefs, "fit": fit, "diagnostics": diagnostics,
                "interpretation": (
                    f"{model_label} on {len(frame)} areas. "
                    + ", ".join(f"{r['variable']} = {r['coefficient']}"
                                + (" (significant)" if r["significant_at_05"] else " (n.s.)")
                                for r in coefs if r["variable"] != "CONSTANT")),
                "coefficients_csv": _asset(csv_rec),
                "map_layer": _map_layer(rec, (_label_for(name, rec)
                                              or f"{y_column} {chosen} residuals"),
                                        "choropleth", "residual", int(len(out))),
                "notes": notes or None,
            })
        except ImportError as exc:
            return _fail(exc, "spreg/libpysal are not installed in this deployment")
        except Exception as exc:
            return _fail(exc, _SIB)

    # --- 7. regionalization (GeoDa / libgeoda) ----------------------------------------

    # GeoDa's own regionalization algorithms. Each takes a target region count except max-p,
    # which DERIVES the count from a minimum-size constraint instead.
    _METHODS = ("skater", "redcap", "azp", "schc", "maxp")
    # A qualitative palette: regions are nominal, so a sequential ramp would imply that region 7
    # is "more" than region 2. Cycled when there are more regions than colours.
    _REGION_COLORS = [
        [166, 206, 227, 200], [31, 120, 180, 200], [178, 223, 138, 200], [51, 160, 44, 200],
        [251, 154, 153, 200], [227, 26, 28, 200], [253, 191, 111, 200], [255, 127, 0, 200],
        [202, 178, 214, 200], [106, 61, 154, 200], [255, 255, 153, 200], [177, 89, 40, 200],
    ]

    def regionalize(file_id: str, columns: List[str], n_regions: int = 5,
                    method: str = "skater", bound_column: Optional[str] = None,
                    min_bound: Optional[float] = None, weights: str = "queen",
                    name: Optional[str] = None, label_column: Optional[str] = None,
                    siblings: Optional[List[str]] = None) -> str:
        """REGIONALIZATION — group areas into CONTIGUOUS regions that are similar on several
        variables (GeoDa's SKATER / REDCAP / AZP / max-p).

        Ordinary clustering (k-means, DBSCAN) ignores geography and returns scattered groups.
        These algorithms require every region to be a connected block on the map, which is what
        makes the output usable as districts, market areas, service zones or strata. `columns` is
        the list of variables to be similar on; `n_regions` is how many you want. `method="maxp"`
        instead finds as many regions as possible subject to each holding at least `min_bound` of
        `bound_column` (e.g. 50,000 people per region) — there `n_regions` is ignored. Returns the
        regions as a categorical map layer plus a per-region summary CSV.
        """
        notes: List[str] = []
        try:
            import numpy as np
            import pandas as pd
            import pygeoda

            cols = [str(c) for c in (columns or []) if str(c).strip()]
            if not cols:
                return _bad("columns is empty — name the variable(s) the regions should be "
                            "similar on",
                            hint="columns is a list, e.g. ['median_income', 'pct_college']")
            meth = str(method or "skater").strip().lower()
            if meth not in _METHODS:
                return _bad(f"unsupported method {meth!r}", hint=f"use one of {list(_METHODS)}")
            if meth == "maxp":
                if not bound_column or min_bound is None:
                    return _bad("method='maxp' needs bound_column and min_bound",
                                hint="e.g. bound_column='population', min_bound=50000 — max-p "
                                     "derives the region COUNT from that constraint, so pick "
                                     "another method if you want a specific n_regions")
                # Dedupe, order-preserving: the bound column is USUALLY also a clustering
                # variable ("regions similar in population, each with >=1,000,000 people"), and
                # a duplicated name makes frame[needed] a DataFrame-per-name, so pygeoda's
                # GetRealCol does DataFrame.to_list() and dies with an AttributeError that says
                # nothing about the real cause.
                needed = list(dict.fromkeys(cols + [str(bound_column)]))
            else:
                needed = list(dict.fromkeys(cols))

            frame, err = _load(file_id, siblings, needed, notes)
            if err:
                return err

            types = {str(t) for t in frame.geom_type.dropna().unique()}
            if types and not (types & {"Polygon", "MultiPolygon"}):
                return _bad(f"regionalization needs POLYGONS, but this layer is "
                            f"{'/'.join(sorted(types))}",
                            hint="contiguity is undefined for points — aggregate the points into "
                                 "areas first (count_points_in_areas / aggregate_to_grid), then "
                                 "regionalize the areas")

            target = int(n_regions or 5)
            if meth != "maxp":
                if target < 2:
                    return _bad(f"n_regions must be at least 2, got {n_regions!r}")
                if target >= len(frame):
                    return _bad(f"n_regions ({target}) must be fewer than the {len(frame)} areas "
                                "being grouped",
                                hint=f"try n_regions between 2 and {max(2, len(frame) // 2)}")

            # pygeoda reads the geometry from a GeoDataFrame directly, but only the columns it is
            # handed: pass a minimal frame so an odd dtype elsewhere in the layer cannot break it.
            geom = _geom_name(frame)
            slim = frame[needed + [geom]].copy()
            if geom != "geometry":
                slim = slim.rename(columns={geom: "geometry"}).set_geometry("geometry")

            gd = pygeoda.open(slim)
            w = getattr(pygeoda, f"{weights}_weights", pygeoda.queen_weights)(gd) \
                if weights in {"queen", "rook"} else pygeoda.queen_weights(gd)
            if weights not in {"queen", "rook"}:
                notes.append(f"weights={weights!r} is not a contiguity rule; regionalization "
                             "requires contiguity, so queen was used instead")
            # pygeoda exposes these as METHODS, not properties: `getattr(w, "has_isolates")`
            # returns the bound method, which is always truthy, so the un-called form put a false
            # island warning on every single result.
            try:
                isolated = bool(w.has_isolates())
            except Exception:
                isolated = False
            if isolated:
                notes.append("some areas have NO contiguous neighbour (islands). They cannot join "
                             "a connected region and the algorithm may place them alone or fail.")

            data = [gd.GetRealCol(c) for c in cols]
            if meth == "skater":
                result = pygeoda.skater(target, w, data)
            elif meth == "redcap":
                # GeoDa's default linkage for REDCAP, and the one its docs recommend first.
                result = pygeoda.redcap(target, w, data, "fullorder-completelinkage")
            elif meth == "azp":
                result = pygeoda.azp_greedy(target, w, data)
            elif meth == "schc":
                result = pygeoda.schc(target, w, data, "complete")
            else:
                result = pygeoda.maxp_greedy(w, data, gd.GetRealCol(str(bound_column)),
                                             float(min_bound))

            clusters = list(result["Clusters"] if isinstance(result, dict) else result)
            if not clusters or len(clusters) != len(frame):
                return _bad(f"{meth} returned {len(clusters)} labels for {len(frame)} areas",
                            hint="this usually means the layer's contiguity graph is "
                                 "disconnected; run spatial_weights to inspect it")
            found = sorted({int(c) for c in clusters})
            if len(found) < 2:
                return _bad(f"{meth} could not form more than one region",
                            hint=("relax min_bound — it may exceed what any single region can "
                                  "reach" if meth == "maxp" else
                                  "the contiguity graph may be too sparse; check spatial_weights"),
                            notes=notes or None)

            out = frame.copy()
            # A region id is a LABEL, not a quantity: keep it as text so nothing downstream
            # averages it, and so the categorical legend keys on it directly.
            out["region"] = [f"Region {int(c)}" for c in clusters]
            out["region_id"] = [int(c) for c in clusters]

            summary_rows: List[Dict[str, Any]] = []
            for region in found:
                mask = np.asarray([int(c) == region for c in clusters])
                row: Dict[str, Any] = {"region": f"Region {region}",
                                       "areas": int(mask.sum())}
                for col in cols:
                    values = frame.loc[mask, col].to_numpy(dtype="float64")
                    row[f"mean_{col}"] = _fmt(values.mean())
                    row[f"min_{col}"] = _fmt(values.min())
                    row[f"max_{col}"] = _fmt(values.max())
                if meth == "maxp" and bound_column:
                    row[f"total_{bound_column}"] = _fmt(
                        frame.loc[mask, str(bound_column)].to_numpy(dtype="float64").sum(), 2)
                summary_rows.append(row)

            base = _stem(name, _source_stem(file_id), f"{meth}_regions")
            rec = _write_geojson(out, base, _source_stem(file_id), "regions")
            csv_rec = _write_csv(pd.DataFrame(summary_rows), base, _source_stem(file_id),
                                 "region_summary")

            palette = {f"Region {r}": _REGION_COLORS[i % len(_REGION_COLORS)]
                       for i, r in enumerate(found)}
            ratio = None
            if isinstance(result, dict):
                for key in result:
                    if "ratio" in str(key).lower():
                        ratio = _fmt(result[key], 4)
                        break

            sizes = [int(r["areas"]) for r in summary_rows]
            return _dumps({
                "ok": True, "file_id": rec["file_id"], "filename": rec.get("filename"),
                "download_url": rec.get("download_url"),
                "method": meth, "engine": "pygeoda (GeoDa/libgeoda)",
                "columns": cols, "regions_found": len(found),
                "requested_regions": None if meth == "maxp" else target,
                "features_analyzed": int(len(frame)),
                "bound_column": bound_column if meth == "maxp" else None,
                "min_bound": _fmt(min_bound) if meth == "maxp" else None,
                "between_to_total_sum_of_squares": ratio,
                "region_sizes": {f"Region {r}": s for r, s in zip(found, sizes)},
                "interpretation": (
                    f"{len(found)} contiguous region(s) from {len(frame)} areas by {meth}, "
                    f"grouped on {', '.join(cols)}; region sizes range from {min(sizes)} to "
                    f"{max(sizes)} areas"
                    + (f". {ratio} of the total variance is between regions rather than within "
                       "them (higher is a cleaner grouping)" if ratio is not None else "")),
                "region_summary": summary_rows,
                "summary_csv": _asset(csv_rec),
                "map_layer": _categorical_layer(rec, _label_for(name, rec) or f"{meth} regions",
                                                "region", int(len(out)),
                                                _legend(palette, out["region"])),
                "notes": notes or None,
            })
        except ImportError as exc:
            return _fail(exc, "pygeoda is not installed in this deployment; the PySAL-backed "
                              "tools (LISA, Gi*, regression) are unaffected")
        except Exception as exc:
            return _fail(exc, _SIB)

    # --- registration -----------------------------------------------------------------

    meta = {"category": "geo"}
    return [
        StructuredTool.from_function(
            func=global_spatial_autocorrelation, name="global_spatial_autocorrelation",
            metadata=meta,
            description=(
                "Test whether a variable is spatially CLUSTERED — Moran's I, Geary's C and "
                "Getis-Ord G with permutation p-values. THE first question to ask about any "
                "choropleth or areal variable: 'is this pattern real or could it be random?', "
                "'is income spatially clustered?', 'is there spatial autocorrelation?'. Returns "
                "each statistic against its expected value under spatial randomness plus a "
                "plain-language verdict. Run before local_moran_lisa. `weights`: queen|rook|knn|"
                "distance_band|kernel. " + _SIB)),
        StructuredTool.from_function(
            func=local_moran_lisa, name="local_moran_lisa", metadata=meta,
            description=(
                "LISA CLUSTER MAP (Local Moran's I) — puts the HOT SPOTS and COLD SPOTS on the "
                "user's interactive map. Answers 'which neighborhoods/counties/tracts are the "
                "hot spots', 'where are the clusters', 'show me a LISA map'. Every area is "
                "classified High-High (hot spot), Low-Low (cold spot), High-Low or Low-High "
                "(spatial outliers), or Not significant, and the layer is delivered as a "
                "CATEGORICAL map layer with a legend plus a CSV of per-area statistics and "
                "p-values. This is GeoDa's cluster map. " + _SIB)),
        StructuredTool.from_function(
            func=local_getis_ord, name="local_getis_ord", metadata=meta,
            description=(
                "Getis-Ord Gi* HOT SPOT / COLD SPOT analysis — the ArcGIS-style hotspot map, "
                "banded by confidence (99%/95%). Use when the question is 'where are the "
                "clusters of HIGH values (or LOW values)' by intensity; use local_moran_lisa "
                "instead when spatial OUTLIERS matter too. `star=True` (default) includes each "
                "area in its own neighbourhood. Returns a categorical map layer with a legend "
                "plus a CSV of z-scores and p-values. " + _SIB)),
        StructuredTool.from_function(
            func=moran_scatterplot, name="moran_scatterplot", metadata=meta,
            description=(
                "The MORAN SCATTERPLOT as a downloadable PNG: each area's value against its "
                "neighbours' average, coloured by LISA quadrant, with the regression line whose "
                "slope IS Moran's I. GeoDa's signature plot — use it to SHOW spatial "
                "autocorrelation and its outliers, alongside "
                "global_spatial_autocorrelation which states it numerically. " + _SIB)),
        StructuredTool.from_function(
            func=spatial_regression, name="spatial_regression", metadata=meta,
            description=(
                "SPATIAL REGRESSION — 'does X explain Y?' done correctly for areal data, where "
                "plain OLS understates the standard errors because neighbours resemble each "
                "other. Fits OLS with Lagrange Multiplier spatial diagnostics and, by default "
                "(model='auto'), refits as a spatial LAG model (spillover between neighbours) or "
                "spatial ERROR model (shared unobserved factor) following the standard "
                "specification rule — it reports which it chose and why. model can also be "
                "ols|lag|error. `x_columns` is a LIST. Returns the coefficient table with "
                "p-values, fit statistics (R-squared/pseudo-R-squared, AIC, log-likelihood), the "
                "diagnostics, a coefficients CSV, and a residual map layer. " + _SIB)),
        StructuredTool.from_function(
            func=regionalize, name="regionalize", metadata=meta,
            description=(
                "REGIONALIZATION — group areas into CONTIGUOUS regions similar on several "
                "variables, using GeoDa's own algorithms (SKATER, REDCAP, AZP, SCHC, max-p). "
                "Answers 'group these tracts into 8 regions', 'build market areas / districts / "
                "sampling strata', 'cluster these counties but keep them connected'. Unlike "
                "k-means or cluster_points, every region is guaranteed to be a connected block "
                "on the map. `columns` is the LIST of variables to be similar on. method='maxp' "
                "maximises the region count subject to bound_column/min_bound (e.g. 50000 people "
                "each) and ignores n_regions. Returns a categorical region map with a legend and "
                "a per-region summary CSV. Needs polygons. " + _SIB)),
        StructuredTool.from_function(
            func=spatial_weights, name="spatial_weights", metadata=meta,
            description=(
                "Build and INSPECT the spatial weights matrix ('who is next to whom') that every "
                "other spatial statistic depends on: neighbour counts, islands (areas with NO "
                "neighbours), sparsity, and a GeoDa-compatible .gal/.gwt download. Use it to "
                "DIAGNOSE a surprising LISA/Moran result — the weights are usually the reason — "
                "or when asked about contiguity/neighbours directly. `weights`: queen|rook|knn|"
                "distance_band|kernel; distance schemes are computed in a projected CRS so "
                "threshold_km is real kilometres. " + _SIB)),
    ]


__all__ = ["make_spatial_stats_tools"]
