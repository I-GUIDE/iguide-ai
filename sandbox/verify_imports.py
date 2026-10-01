"""Fail the image build if anything baked into it cannot actually be imported.

`pip install` succeeding is not evidence that a package works. fiona and rasterio install
cleanly on python:3.11-slim and then fail at import with
``ImportError: libexpat.so.1: cannot open shared object file`` — the bundled GDAL links
against a library the slim base does not ship. importlib.util.find_spec even reports them as
present, because it locates the module file without loading it.

That matters beyond the image: the executor probes this image to decide which packages it can
skip installing (`CodeExecutor.preinstalled` in agent_runtime/code_execution.py). A package that is
present-but-broken would be skipped and then break the run. So the build asserts what the
probe assumes.
"""

import importlib
import sys

MODULES = [
    # the import names behind _IMPORT_TO_PIP in agent_runtime/code_execution.py
    "numpy", "pandas", "scipy", "matplotlib", "seaborn", "statsmodels", "sklearn",
    "pyarrow", "networkx", "requests", "bs4", "PIL", "openpyxl",
    "geopandas", "fiona", "rasterio", "pyproj", "mapclassify", "folium",
    # plus the spatial-statistics stack
    "libpysal", "esda", "spreg", "pygeoda",
]

broken = []
for name in MODULES:
    try:
        importlib.import_module(name)
    except Exception as exc:
        broken.append(f"{name}: {type(exc).__name__}: {exc}")

if broken:
    print("BAKED PACKAGES THAT DO NOT IMPORT:", *broken, sep="\n  ")
    sys.exit(1)
print(f"all {len(MODULES)} baked packages import cleanly")
