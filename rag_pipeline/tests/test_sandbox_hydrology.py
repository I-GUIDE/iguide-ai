"""The sandbox image's hydrology libraries must delineate a watershed, not merely import.

pysheds 0.5 calls ``numpy.in1d``, which numpy 2.4 removed. It imports cleanly and then fails
inside ``grid.accumulation``, so an import check passes and every watershed fails. GIS harness
task T06 spent 29 of 30 steps and about 2M tokens on that error before this was fixed.

Three layers of guard:

* the import hook itself (``sandbox/pysheds_support.py``), tested here against a numpy without
  ``in1d`` and temporary directories, so this runs on any machine;
* the image build runs ``sandbox/check_hydrology.py``, which delineates a watershed of known
  area with each library; the Dockerfile is held to running it;
* ``test_the_sandbox_image_delineates_a_watershed`` runs the same check inside a built image,
  offline and read-only like a real run (``-m integration``; needs Docker and the image).
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

SANDBOX = Path(__file__).resolve().parents[2] / "sandbox"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_sandbox_{name}", SANDBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def compat(monkeypatch):
    before = list(sys.meta_path)
    module = _load("pysheds_support")              # loading it installs the finder
    yield module
    sys.meta_path[:] = before


def _numpy_without_in1d():
    """numpy as 2.4 ships it: isin, and no in1d."""
    fake = types.ModuleType("numpy")
    fake.isin = np.isin
    fake.asarray = np.asarray
    return fake


# ------------------------------------------------------------------ the shim


def test_in1d_is_restored_with_numpys_own_semantics(compat):
    fake = _numpy_without_in1d()
    assert compat.restore_in1d(fake) is True
    a = np.array([[1, 2], [3, 4]])
    # in1d flattens and returns 1-D; isin would keep the 2x2 shape.
    assert fake.in1d(a, [2, 4]).tolist() == [False, True, False, True]
    assert fake.in1d(a, [2, 4], invert=True).tolist() == [True, False, True, False]


def test_a_numpy_that_still_has_in1d_is_left_alone(compat):
    fake = _numpy_without_in1d()
    original = object()
    fake.in1d = original
    assert compat.restore_in1d(fake) is False
    assert fake.in1d is original


def test_the_hook_fires_on_pysheds_only_and_once(compat, monkeypatch, tmp_path):
    fake = _numpy_without_in1d()
    monkeypatch.setitem(sys.modules, "numpy", fake)
    private, baked, cache = tmp_path / "private", tmp_path / "baked", tmp_path / "run" / "numba"
    private.mkdir()
    (baked / "pysheds").mkdir(parents=True)
    (baked / "pysheds" / "kernel.nbi").write_text("compiled")
    monkeypatch.setattr(compat, "PRIVATE", str(private))
    monkeypatch.setattr(compat, "BAKED_CACHE", str(baked))
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(cache))
    monkeypatch.setattr(sys, "path", list(sys.path))
    finder = next(f for f in sys.meta_path if isinstance(f, compat._PyshedsFinder))

    assert finder.find_spec("esda") is None
    assert not hasattr(fake, "in1d"), "importing something else must not touch numpy"
    assert str(private) not in sys.path, "numba must stay hidden from code that skips pysheds"
    assert not cache.exists()
    assert finder in sys.meta_path

    assert finder.find_spec("pysheds") is None, "the hook must let the real finders load pysheds"
    assert hasattr(fake, "in1d")
    assert sys.path[-1] == str(private), "appended, so a run's own numba in .deps still wins"
    assert (cache / "pysheds" / "kernel.nbi").read_text() == "compiled"
    assert finder not in sys.meta_path, "the hook removes itself after it has fired"


def test_a_seeded_or_missing_cache_is_not_copied(compat, tmp_path):
    baked = tmp_path / "baked"
    assert compat.seed_numba_cache(str(baked), str(tmp_path / "dst")) is False   # nothing baked
    baked.mkdir()
    assert compat.seed_numba_cache(str(baked), "") is False                       # no cache dir
    assert compat.seed_numba_cache(str(baked), str(baked)) is False               # the build itself
    assert compat.seed_numba_cache(str(baked), str(tmp_path / "dst")) is True
    assert compat.seed_numba_cache(str(baked), str(tmp_path / "dst")) is False    # second process


def test_a_failing_step_leaves_the_import_to_pysheds(compat, monkeypatch):
    def boom(*a, **k):
        raise OSError("tmpfs full")

    monkeypatch.setattr(compat, "seed_numba_cache", boom)
    monkeypatch.setattr(compat, "expose_private", boom)
    finder = next(f for f in sys.meta_path if isinstance(f, compat._PyshedsFinder))
    assert finder.find_spec("pysheds") is None


def test_installing_twice_adds_one_hook(compat):
    compat.install()
    compat.install()
    assert sum(isinstance(f, compat._PyshedsFinder) for f in sys.meta_path) == 1


def test_the_pth_file_imports_the_shim():
    """A .pth line starting with `import` is executed by site at interpreter start, from
    site-packages, whatever PYTHONPATH says. That is how the shim reaches a pysheds installed
    into the run's /work/.deps."""
    lines = [ln.strip() for ln in (SANDBOX / "pysheds_support.pth").read_text().splitlines()
             if ln.strip()]
    assert lines == ["import pysheds_support"]


# ------------------------------------------------------------------ the image is built to check it


def _dockerfile() -> str:
    return (SANDBOX / "Dockerfile").read_text(encoding="utf-8")


def test_the_build_runs_the_hydrology_check_after_installing():
    src = _dockerfile()
    assert "RUN python /opt/check_hydrology.py" in src
    assert src.index("pysheds==") < src.index("RUN python /opt/check_hydrology.py")
    assert "pysheds_support.pth" in src and "pysheds_support.py" in src


def test_numba_is_kept_out_of_site_packages_and_the_cache_is_compiled_at_build():
    """esda compiles its numba kernels on every run when it can import numba (8.2 s on one
    Moran_Local), and pysheds compiles ~80 kernels at import unless a cache is there (~40 s
    of a 60 s run). The Dockerfile hides the first and bakes the second."""
    src = _dockerfile()
    assert "/opt/pysheds-private" in src and 'mv "$SITE"/numba' in src
    assert "find_spec('numba') is not None" in src, "the build must prove numba is hidden"
    assert "NUMBA_CPU_NAME=generic" in src
    baked = src.index("NUMBA_CACHE_DIR=/opt/numba-cache")
    check = src.index("RUN python /opt/check_hydrology.py")
    run_dir = src.index("ENV NUMBA_CACHE_DIR=/tmp/numba")
    assert baked < check < run_dir, "compile into the baked cache, THEN point runs at the tmpfs"


def test_pysheds_installs_against_the_images_own_versions():
    """Without the constraint pip may resolve numba to a release that wants a different numpy
    and move the numpy rasterio and geopandas were compiled against."""
    block = _dockerfile().split("pip freeze > /tmp/image-pins.txt", 1)
    assert len(block) == 2, "the pysheds install is not constrained by the image's freeze"
    assert "-c /tmp/image-pins.txt pysheds==" in block[1].split("\n\n", 1)[0]


def test_every_hydrology_check_names_a_baked_and_probed_library():
    check = _load("check_hydrology")
    verify = (SANDBOX / "verify_imports.py").read_text(encoding="utf-8")
    from agent_runtime.code_execution import _IMPORT_TO_PIP

    assert check.CHECKS, "no hydrology library is checked"
    for name in check.CHECKS:
        assert name in _dockerfile(), f"{name} is checked but not installed in the image"
        assert f'"{name}"' in verify, f"{name} is not in verify_imports.py"
        # So the executor's probe sees the baked copy and drops a declared dependency on it.
        assert name in _IMPORT_TO_PIP, f"{name} is not in _IMPORT_TO_PIP"


def test_verify_imports_covers_everything_the_probe_asks_about():
    """The probe skips installing what it can import; the build must prove each of those
    imports, or a present-but-broken package is skipped and the run dies."""
    from agent_runtime.code_execution import _IMPORT_TO_PIP

    verify = (SANDBOX / "verify_imports.py").read_text(encoding="utf-8")
    missing = [mod for mod in _IMPORT_TO_PIP if f'"{mod}"' not in verify]
    assert not missing, f"probed but never import-checked at build: {missing}"


def test_the_built_in_dem_has_the_area_it_claims(tmp_path):
    """The check's own answer key, computed independently of any hydrology library."""
    pytest.importorskip("rasterio")
    check = _load("check_hydrology")
    x, y, km2 = check.synthetic_dem(str(tmp_path / "dem.tif"))
    assert km2 == pytest.approx(16.2), "T06's own answer: 150 cols x 120 rows x 900 m^2"
    assert (x, y) == (392265.0, 4434015.0), "T06's own outlet"
    import rasterio

    with rasterio.open(tmp_path / "dem.tif") as src:
        z = src.read(1)
        row, col = src.index(x, y)
    assert (row, col) == (check.ROWS - 1, check.COLS // 2), "the outlet is the valley's last cell"
    assert z[row, col] == z.min(), "the outlet is the lowest cell"


# ------------------------------------------------------------------ inside a built image


@pytest.mark.integration
def test_the_sandbox_image_delineates_a_watershed():
    image = os.getenv("AGENT_CODE_EXEC_IMAGE") or "iguide-codeexec"
    if not shutil.which("docker"):
        pytest.skip("no docker")
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode:
        pytest.skip(f"image {image} is not built")
    proc = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "--read-only", "--cap-drop", "ALL",
         "--security-opt", "no-new-privileges", "--tmpfs", "/tmp:rw,size=256m",
         "--env", "HOME=/tmp", image, "python", "/opt/check_hydrology.py"],
        capture_output=True, text=True, timeout=600,
    )
    rows = [json.loads(ln) for ln in proc.stdout.splitlines() if ln.startswith("{")]
    assert proc.returncode == 0 and rows and all(r["ok"] for r in rows), proc.stdout + proc.stderr
    pysheds = next(r for r in rows if r["library"] == "pysheds")
    assert pysheds["numba_cache_seeded"], "the run compiled pysheds from scratch"
