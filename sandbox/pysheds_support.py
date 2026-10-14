"""What pysheds needs in the sandbox, done at the moment pysheds is imported and only then.

``sandbox/pysheds_support.pth`` imports this module at interpreter start. A ``.pth`` file is read
from site-packages whatever ``PYTHONPATH`` says, so this reaches every run. Importing it only
puts a finder on ``sys.meta_path``. The finder never loads anything itself. When it sees
``pysheds`` looked up it does three things and steps aside for the normal import:

1. **numpy.in1d.** pysheds 0.5, the newest release (2025-08), calls ``np.in1d`` in nine places
   in ``sgrid.py``: accumulation, flow distance, HAND, river network, stream order and four
   more. numpy deprecated ``in1d`` in 2.0 and removed it in 2.4, which the image ships, so each
   of those raised ``AttributeError: module 'numpy' has no attribute 'in1d'``. numpy cannot
   move to fix that: rasterio and geopandas were compiled against the numpy the image ships,
   and every per-run install is pinned to it for that reason (AGENTS.md, "Installs are pinned
   to the image's own versions"). So ``in1d`` is put back, as ``isin`` over the flattened
   input, which is what numpy's own ``in1d`` was. A numpy that still has ``in1d`` is left alone.
   This works for a pysheds loaded from anywhere, including ``/work/.deps``.

2. **numba, kept out of everyone else's way.** pysheds needs numba. esda does not, but
   uses it when it can import it, and its numba kernels are compiled on every run
   (``@njit`` without ``cache=True``). Measured natively, numba being importable turned a
   400-unit ``Moran_Local`` from 0.0 s into 8.2 s. So the image keeps numba and llvmlite in
   ``PRIVATE`` rather than site-packages, and that directory joins ``sys.path`` (at the end)
   only once pysheds is imported. Code that never imports pysheds sees the image as it was.

3. **A compiled cache.** pysheds compiles about eighty numba kernels at import, with
   ``cache=True``. A run's root filesystem is read-only and numba will not use a cache
   directory it cannot write, so every run compiled from scratch: 39.5 s of a 60 s run
   timeout, measured natively, against 5.7 s with a cache. The image compiles them once at
   build into ``BAKED_CACHE``, and the finder copies that into ``NUMBA_CACHE_DIR`` (the
   ``/tmp`` tmpfs) before pysheds loads. Both build and run set ``NUMBA_CPU_NAME=generic``,
   so the cache does not depend on the CPU of the machine that built the image.

Drop the in1d part once a pysheds release stops calling ``in1d``; ``check_hydrology.py``
reports whether the shim was what made it work.
"""

import sys

_TARGET = "pysheds"
PRIVATE = "/opt/pysheds-private"
BAKED_CACHE = "/opt/numba-cache"


def restore_in1d(np) -> bool:
    """Put ``in1d`` back on *np* if it is missing. True when it was added.

    numpy's ``in1d`` flattens its first argument and returns a 1-D result; ``isin`` keeps the
    shape. pysheds passes ``.ravel()`` already, but the shim keeps ``in1d``'s own contract.
    """
    try:
        np.in1d
        return False
    except AttributeError:
        pass

    def in1d(ar1, ar2, assume_unique=False, invert=False, *, kind=None):
        return np.isin(np.asarray(ar1).ravel(), ar2,
                       assume_unique=assume_unique, invert=invert, kind=kind)

    in1d.__doc__ = "numpy.in1d, removed in numpy 2.4, restored for pysheds (pysheds_support)."
    np.in1d = in1d
    return True


def expose_private(path=None) -> bool:
    """Append *path* to sys.path if it exists and is not there. True when it was added.

    Appended, not prepended: a numba the run installed itself (``/work/.deps``) still wins.
    """
    import os

    path = path or PRIVATE
    if not os.path.isdir(path) or path in sys.path:
        return False
    sys.path.append(path)
    return True


def seed_numba_cache(src=None, dst=None) -> bool:
    """Copy the build's compiled kernels to the writable cache dir. True when it copied."""
    import os
    import shutil

    src = src or BAKED_CACHE
    dst = dst if dst is not None else os.environ.get("NUMBA_CACHE_DIR", "")
    if not dst or not os.path.isdir(src) or os.path.exists(dst):
        return False                    # nothing baked, no cache dir, or already seeded/building
    shutil.copytree(src, dst)
    return True


class _PyshedsFinder:
    """A meta-path entry that never finds anything; it prepares the ground for pysheds."""

    def find_spec(self, name, path=None, target=None):
        if name == _TARGET:
            try:
                sys.meta_path.remove(self)      # once per process
            except ValueError:
                pass
            # Each step on its own: a failure here must leave pysheds' own error to be reported.
            for step in (expose_private, seed_numba_cache, _restore_in1d_on_numpy):
                try:
                    step()
                except Exception:
                    pass
        return None                             # the normal finders locate pysheds


def _restore_in1d_on_numpy() -> bool:
    import numpy

    return restore_in1d(numpy)


def install() -> None:
    if not any(isinstance(f, _PyshedsFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _PyshedsFinder())


install()
