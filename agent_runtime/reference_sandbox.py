"""Reference recomputation of standard methods, INSIDE the sandbox (plan step 4b).

Every check before this one asked whether a run was consistent with itself: units, CRS, what it
printed against what it measured. A run that implements a standard method wrongly passes all of
them, because it is consistent with its own mistake. Two families of wrong answer in the scored
archive are exactly that:

* **T10, ordinary kriging.** 14 of 28 scored answers were wrong, and nearly all of them gave
  295.06 ppm where the reference is 309.53. The agent wrote the kriging system in numpy and put
  the nugget on the diagonal of the semivariance matrix (gamma(0) = nugget instead of 0).
* **T09, Getis-Ord Gi\\*.** 16 of 25 were wrong: "1 hot spot" where the reference is 13. esda's
  `G_Local` row-standardises its weights unless told `transform="B"`, so Gi\\* "with binary
  weights" silently ran on row-standardised ones (both in agent code and in the
  `local_getis_ord` tool, which fixed `"r"` until stage 55 made binary its default).

This module recomputes the method with a reference library, from the SAME inputs the run used,
and records both values. It never reads a variable's name to decide what a number is. The inputs
are recovered from what the run did:

* **Ordinary kriging, hand-written.** Every linear solve is recorded (`numpy.linalg.solve`,
  `numpy.linalg.lstsq`, `scipy.linalg.solve`). A system whose last row and column are ones with a
  zero corner is an ordinary-kriging system. From it: the variogram the run used is fitted to
  the matrix's off-diagonal against the distances between the samples. The samples' coordinates
  are the pair of the run's vectors whose distances reproduce it. The target point is
  trilaterated from the right-hand side. pykrige then kriges each of the run's length-n vectors
  at that point. The agent side picks the vector whose kriged value the run actually printed.
* **Gi\\* and Moran's I through esda.** The calls are recorded with their inputs and results.
  Gi\\* is recomputed with binary weights and the focal cell included (`transform="B"`,
  `star=True`, the statistic as Getis and Ord define it). Moran's I is recomputed with
  row-standardised weights.

Like sandbox_verify, the module is stdlib-only at import time and cannot fail the run: every step
is guarded and a failure is recorded, not raised. Its functions are inlined into the gate's
prologue (sandbox_verify._inlined_helpers).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

REFERENCE_GLOBAL = "_IGUIDE_REFERENCE_CALLS"


def install_reference_recorder(namespace: Dict[str, Any]) -> None:
    """Record ordinary-kriging systems and esda Gi*/Moran calls, from the moment numpy, scipy or
    esda is imported. Results are unchanged: each wrapper calls the original and records after.
    Process-wide, like the metric-op recorder, so a second run in one process re-points the
    record. Never raises."""
    import builtins
    import sys

    record = namespace.setdefault("_IGUIDE_REFERENCE_CALLS", [])
    state = sys.__dict__.setdefault("_iguide_reference_state", {"record": record})
    state["record"] = record
    max_n = 1500

    def keep(entry: Dict[str, Any]) -> None:
        rec = state["record"]
        if len(rec) < 12:
            rec.append(entry)

    def ok_system(a: Any, b: Any, x: Any, how: str) -> None:
        try:
            import numpy as np

            shape = getattr(a, "shape", None)
            if not shape or len(shape) != 2 or shape[0] != shape[1] or not 5 <= shape[0] <= max_n:
                return
            A = np.array(a, dtype=float)
            if (abs(A[-1, -1]) > 1e-12 or not np.allclose(A[-1, :-1], 1.0)
                    or not np.allclose(A[:-1, -1], 1.0)):
                return
            keep({"kind": "ok_system", "how": how, "A": A,
                  "b": np.array(b, dtype=float), "x": np.array(x, dtype=float)})
        except Exception:
            pass

    def patch_linalg() -> None:
        # Checked on every import, and marked per function rather than per module: a module seen
        # half-initialised (numpy.linalg while its `from ._linalg import *` is still running)
        # has no `solve` yet, or gets it rebound after the wrap.
        for modname, attr, how in (("numpy.linalg", "solve", "numpy.linalg.solve"),
                                   ("numpy.linalg", "lstsq", "numpy.linalg.lstsq"),
                                   ("scipy.linalg", "solve", "scipy.linalg.solve")):
            mod = sys.modules.get(modname)
            fn = getattr(mod, attr, None) if mod is not None else None
            if not callable(fn) or getattr(fn, "_iguide", False):
                continue

            def make(fn0, how0, attr0):
                def wrapper(a, b, *args, **kwargs):
                    out = fn0(a, b, *args, **kwargs)
                    ok_system(a, b, out[0] if attr0 == "lstsq" else out, how0)
                    return out
                wrapper._iguide = True
                wrapper.__name__, wrapper.__doc__ = attr0, getattr(fn0, "__doc__", None)
                return wrapper
            try:
                setattr(mod, attr, make(fn, how, attr))
            except Exception:
                pass

    def snapshot_w(w: Any) -> Optional[Dict[str, Any]]:
        try:
            return {"neighbors": {k: list(v) for k, v in w.neighbors.items()},
                    "weights": {k: [float(x) for x in v] for k, v in w.weights.items()},
                    "id_order": list(w.id_order)}
        except Exception:
            return None

    def patch_esda() -> None:
        for modname, clsname in (("esda.getisord", "G_Local"), ("esda.moran", "Moran")):
            mod = sys.modules.get(modname)
            orig = getattr(mod, clsname, None) if mod is not None else None
            if not isinstance(orig, type) or getattr(orig, "_iguide", False):
                continue
            if clsname == "G_Local":
                class G_Local(orig):                          # noqa: N801 - same public name
                    def __init__(self, y, w, *args, **kwargs):
                        super().__init__(y, w, *args, **kwargs)
                        try:
                            import numpy as np

                            keep({"kind": "gi", "y": np.asarray(y, dtype=float).ravel(),
                                  "w": snapshot_w(w), "star": bool(self.star),
                                  "transform": str(getattr(self, "w_transform", "R")),
                                  "Zs": np.asarray(self.Zs, dtype=float).ravel()})
                        except Exception:
                            pass
                new = G_Local
            else:
                class Moran(orig):                            # noqa: N801
                    def __init__(self, y, w, *args, **kwargs):
                        super().__init__(y, w, *args, **kwargs)
                        try:
                            import numpy as np

                            keep({"kind": "moran", "y": np.asarray(y, dtype=float).ravel(),
                                  "w": snapshot_w(w),
                                  "transform": str(getattr(w, "transform", "")),
                                  "I": float(self.I)})
                        except Exception:
                            pass
                new = Moran
            new.__module__, new.__qualname__ = orig.__module__, orig.__qualname__
            new.__doc__ = orig.__doc__
            new._iguide = True
            setattr(mod, clsname, new)
            top = sys.modules.get("esda")
            if top is not None and getattr(top, clsname, None) is orig:
                setattr(top, clsname, new)

    def patch() -> None:
        try:
            patch_linalg()
        except Exception:
            pass
        try:
            patch_esda()
        except Exception:
            pass

    patch()
    if state.get("hooked"):
        return
    original_import = builtins.__import__

    def hooked(name, *args, **kwargs):
        module = original_import(name, *args, **kwargs)
        if name.startswith(("numpy", "scipy", "esda")):
            patch()
        return module

    try:
        builtins.__import__ = hooked
        state["hooked"] = True
    except Exception:
        pass


# --------------------------------------------------------------------------- the references

def _variogram_models():
    """(name, gamma(h, *params), pykrige parameters from params). Fitted forms, not pykrige's own
    parameterisations: an exponential written exp(-h/a) is pykrige's range 3a."""
    import numpy as np

    def sph(h, c0, c, r):
        r = abs(r) + 1e-12
        hr = np.minimum(h / r, 1.0)
        return c0 + c * (1.5 * hr - 0.5 * hr ** 3)

    def expo(h, c0, c, a):
        return c0 + c * (1.0 - np.exp(-h / (abs(a) + 1e-12)))

    def gau(h, c0, c, a):
        return c0 + c * (1.0 - np.exp(-(h / (abs(a) + 1e-12)) ** 2))

    def lin(h, c0, s):
        return c0 + s * h

    return [
        ("spherical", sph, lambda p: {"psill": p[1], "range": abs(p[2]), "nugget": p[0]}),
        ("exponential", expo, lambda p: {"psill": p[1], "range": 3.0 * abs(p[2]),
                                         "nugget": p[0]}),
        ("gaussian", gau, lambda p: {"psill": p[1], "range": 7.0 * abs(p[2]) / 4.0,
                                     "nugget": p[0]}),
        ("linear", lin, lambda p: {"slope": p[1], "nugget": p[0]}),
    ]


def _length_n_vectors(namespace: Dict[str, Any], n: int) -> List[tuple]:
    """(name, float vector) for every length-n numeric vector the run left bound: arrays,
    columns of (n, k) arrays, Series, DataFrame columns. At most 40."""
    import numpy as np

    out: List[tuple] = []
    for name, obj in list(namespace.items()):
        if name.startswith("_") or len(out) >= 40:
            continue
        try:
            if hasattr(obj, "columns") and hasattr(obj, "dtypes") and len(obj) == n:
                for col in list(obj.columns)[:24]:
                    try:
                        v = np.asarray(obj[col], dtype=float).ravel()
                    except Exception:
                        continue
                    if v.size == n and np.isfinite(v).all():
                        out.append((f"{name}[{col!r}]", v))
                continue
            arr = np.asarray(obj) if hasattr(obj, "shape") or hasattr(obj, "__array__") else None
            if arr is None or arr.dtype.kind not in "fiu":
                continue
            if arr.ndim == 1 and arr.size == n:
                out.append((name, arr.astype(float)))
            elif arr.ndim == 2 and arr.shape[0] == n and arr.shape[1] <= 6:
                for j in range(arr.shape[1]):
                    out.append((f"{name}[:, {j}]", arr[:, j].astype(float)))
            elif arr.ndim == 2 and arr.shape[1] == n and arr.shape[0] <= 6:
                for j in range(arr.shape[0]):
                    out.append((f"{name}[{j}]", arr[j].astype(float)))
        except Exception:
            continue
    seen, uniq = set(), []
    for name, v in out:
        key = v.tobytes()
        if key not in seen and np.isfinite(v).all():
            seen.add(key)
            uniq.append((name, v))
    return uniq[:40]


def _fit_variogram(d: Any, g: Any):
    """The standard model that reproduces (d, g) to the precision a run computes in, or None."""
    import numpy as np
    from scipy.optimize import least_squares

    scale = max(float(np.max(np.abs(g))), 1e-12)
    best = None
    for name, fn, to_pk in _variogram_models():
        lo, hi = float(np.min(g)), float(np.max(g))
        if name == "linear":
            starts = [[lo, (hi - lo) / max(float(np.max(d)), 1e-12)]]
        else:
            starts = [[lo, hi - lo, float(np.percentile(d, q))] for q in (25, 50, 90)]
        for p0 in starts:
            try:
                res = least_squares(lambda p: (fn(d, *p) - g) / scale, p0, xtol=1e-15,
                                    ftol=1e-15, gtol=1e-15, max_nfev=4000)
            except Exception:
                continue
            err = float(np.max(np.abs(res.fun)))
            if best is None or err < best[0]:
                best = (err, name, fn, list(res.x), to_pk)
    if best is None or best[0] > 1e-6:
        return None
    return best


def _kriging_reference(rec: Dict[str, Any], namespace: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np

    A, b, x = rec["A"], rec["b"], rec["x"]
    n = A.shape[0] - 1
    G = A[:n, :n]
    rhs = (b[:, 0] if b.ndim == 2 else b)[:n]
    lam = (x[:, 0] if x.ndim == 2 else x)[:n]
    out: Dict[str, Any] = {"method": "ordinary_kriging", "n": int(n), "solver": rec.get("how"),
                           "diagonal_max": float(np.max(np.abs(np.diag(G))))}
    vectors = _length_n_vectors(namespace, n)
    iu = np.triu_indices(n, 1)
    rng = np.random.default_rng(0)
    pick = rng.choice(iu[0].size, size=min(400, iu[0].size), replace=False)
    pi, pj = iu[0][pick], iu[1][pick]
    found = None
    for a_i in range(len(vectors)):
        for b_i in range(a_i + 1, len(vectors)):
            xs, ys = vectors[a_i][1], vectors[b_i][1]
            d = np.hypot(xs[pi] - xs[pj], ys[pi] - ys[pj])
            g = G[pi, pj]
            if np.ptp(d) <= 0:
                continue
            order = np.argsort(d)
            if np.any(np.diff(g[order]) < -1e-9 * max(1.0, float(np.max(np.abs(g))))):
                continue                     # not a function of the distance: wrong pair
            fit = _fit_variogram(d, g)
            if fit is None:
                continue
            D = np.hypot(xs[:, None] - xs[None, :], ys[:, None] - ys[None, :])
            full = fit[2](D[iu], *fit[3])
            if np.max(np.abs(full - G[iu])) > 1e-6 * max(1.0, float(np.max(np.abs(G)))):
                continue
            found = (vectors[a_i][0], vectors[b_i][0], xs, ys, fit)
            break
        if found:
            break
    if not found:
        out["status"] = "unrecognised"
        out["reason"] = ("no pair of the run's vectors reproduces the matrix as a standard "
                         "variogram of distance")
        return out
    cx, cy, xs, ys, (err, model, fn, params, to_pk) = found
    out.update({"coordinates": [cx, cy], "model": model,
                "fitted": [float(p) for p in params], "pykrige_parameters": to_pk(params)})
    # The target: invert the variogram for each sample's distance, then trilaterate.
    from scipy.optimize import brentq, least_squares

    hmax = float(np.max(np.hypot(xs - xs.mean(), ys - ys.mean()))) * 4 + 1.0
    d0 = np.full(n, np.nan)
    sill_val = float(fn(np.array([hmax * 10]), *params)[0])
    for i in np.argsort(rhs)[:min(30, n)]:
        gi = float(rhs[i])
        if abs(gi) < 1e-12:
            d0[i] = 0.0
            continue
        if model != "linear" and gi >= sill_val - 1e-9 * max(1.0, abs(sill_val)):
            continue
        f0 = float(fn(np.array([0.0]), *params)[0])
        if gi < f0:
            continue
        try:
            d0[i] = brentq(lambda h: float(fn(np.array([h]), *params)[0]) - gi, 0.0, hmax)
        except Exception:
            continue
    ok = np.isfinite(d0)
    if ok.sum() < 3:
        out["status"] = "unrecognised"
        out["reason"] = "the target is out of range of too many samples to locate"
        return out
    idx = np.where(ok)[0]
    w0 = 1.0 / (d0[idx] + 1e-9)
    p0 = [float(np.sum(xs[idx] * w0) / w0.sum()), float(np.sum(ys[idx] * w0) / w0.sum())]
    sol = least_squares(lambda p: np.hypot(xs[idx] - p[0], ys[idx] - p[1]) - d0[idx], p0,
                        xtol=1e-14, ftol=1e-14, gtol=1e-14)
    tx, ty = float(sol.x[0]), float(sol.x[1])
    dt = np.hypot(xs - tx, ys - ty)
    rebuilt = np.where(dt < 1e-9, 0.0, fn(dt, *params))
    if np.max(np.abs(rebuilt - rhs) / max(1.0, float(np.max(np.abs(rhs))))) > 1e-5:
        # The right-hand side did not come from the matrix's variogram at any one point.
        out["status"] = "inconsistent"
        out["reason"] = "the right-hand side is not the matrix's variogram at one target point"
        return out
    out["target"] = [tx, ty]
    engine = "numpy"
    try:
        from pykrige.ok import OrdinaryKriging  # noqa: F401
        engine = "pykrige"
    except Exception:
        pass
    out["engine"] = engine
    cands = []
    for name, v in vectors:
        if name in (cx, cy):
            continue
        agent = float(np.dot(lam, v))
        try:
            if engine == "pykrige":
                from pykrige.ok import OrdinaryKriging

                pk = to_pk(params)
                ok_ = OrdinaryKriging(xs, ys, v, variogram_model=model, variogram_parameters=pk,
                                      exact_values=True)
                z, _ = ok_.execute("points", np.array([tx]), np.array([ty]))
                ref = float(np.asarray(z).ravel()[0])
            else:
                D = np.hypot(xs[:, None] - xs[None, :], ys[:, None] - ys[None, :])
                M = np.ones((n + 1, n + 1))
                M[:n, :n] = fn(D, *params)
                np.fill_diagonal(M[:n, :n], 0.0)
                M[n, n] = 0.0
                r = np.ones(n + 1)
                r[:n] = rebuilt
                lw = np.linalg.solve(M, r)[:n]
                ref = float(np.dot(lw, v))
        except Exception as exc:
            out.setdefault("errors", []).append(f"{type(exc).__name__}: {exc}"[:160])
            continue
        cands.append({"variable": name, "agent": agent, "reference": ref})
    out["candidates"] = cands[:12]
    out["status"] = "recomputed" if cands else "unrecognised"
    return out


def _uniform_rows(snap: Dict[str, Any]) -> bool:
    """Whether every row's weights are equal: contiguity or k-nearest, binary or
    row-standardised. Inverse-distance or kernel weights are not, and their reference is not
    the binary one, so they are not recomputed."""
    for v in (snap.get("weights") or {}).values():
        if v and max(v) - min(v) > 1e-9 * max(1.0, abs(max(v))):
            return False
    return True


def _w_from_snapshot(snap: Dict[str, Any], drop_self: bool):
    from libpysal.weights import W

    neigh = {}
    for k, v in snap["neighbors"].items():
        neigh[k] = [j for j in v if not (drop_self and j == k)]
    return W(neigh, id_order=snap.get("id_order"), silence_warnings=True)


def _gi_reference(rec: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np
    import esda

    out: Dict[str, Any] = {"method": "getis_ord_gi_star", "transform": rec.get("transform"),
                           "star": rec.get("star"), "n": int(rec["y"].size)}
    w = _w_from_snapshot(rec["w"], drop_self=True)
    ref = esda.G_Local(rec["y"], w, transform="B", star=True, permutations=0)
    z_ref = np.asarray(ref.Zs, dtype=float).ravel()
    z = rec["Zs"]
    out["agent_z_max"], out["reference_z_max"] = float(np.nanmax(z)), float(np.nanmax(z_ref))
    out["max_abs_z_diff"] = float(np.nanmax(np.abs(z - z_ref)))
    out["counts"] = {str(t): [int(np.sum(z > t)), int(np.sum(z_ref > t))]
                     for t in (1.645, 1.96, 2.576)}
    out["cold_counts"] = {str(t): [int(np.sum(z < -t)), int(np.sum(z_ref < -t))]
                          for t in (1.645, 1.96, 2.576)}
    out["status"] = "recomputed"
    return out


def _moran_reference(rec: Dict[str, Any]) -> Dict[str, Any]:
    import esda

    w = _w_from_snapshot(rec["w"], drop_self=True)
    w.transform = "r"                    # esda.Moran's keyword for it differs by version
    ref = esda.Moran(rec["y"], w, permutations=0)
    return {"method": "morans_i", "transform": rec.get("transform"), "n": int(rec["y"].size),
            "agent": float(rec["I"]), "reference": float(ref.I), "status": "recomputed"}


def reference_report(namespace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One entry per recorded call, recomputed with the reference library. Never raises."""
    import time

    out: List[Dict[str, Any]] = []
    started = time.time()
    for rec in list(namespace.get("_IGUIDE_REFERENCE_CALLS") or [])[:8]:
        if time.time() - started > 20:
            out.append({"method": rec.get("kind"), "status": "skipped",
                        "reason": "time budget spent"})
            continue
        try:
            if rec.get("kind") == "ok_system":
                out.append(_kriging_reference(rec, namespace))
            elif rec.get("kind") in ("gi", "moran") and rec.get("w"):
                if not _uniform_rows(rec["w"]):
                    out.append({"method": rec["kind"], "status": "unrecognised",
                                "reason": "weights are not contiguity or k-nearest"})
                elif rec["kind"] == "gi":
                    out.append(_gi_reference(rec))
                else:
                    out.append(_moran_reference(rec))
        except Exception as exc:
            out.append({"method": rec.get("kind"), "status": "error",
                        "reason": f"{type(exc).__name__}: {exc}"[:200]})
    return out
