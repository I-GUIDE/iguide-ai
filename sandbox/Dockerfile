# Pre-baked sandbox image for agent code execution (AGENT_CODE_EXEC_IMAGE).
#
#   docker build -t iguide-codeexec sandbox/
#   export AGENT_CODE_EXEC_IMAGE=iguide-codeexec
#
# With the common scientific stack already installed, queries like
# "plot y=x^2 and save plot.png" run WITHOUT a per-run `pip install`, which avoids
# the install-phase disk/time cost entirely. The agent only needs `dependencies`
# for packages not baked here.
#
# That saving is NOT automatic, and for a long time this file promised it without
# delivering: `pip install --target` sets ignore_installed=True, so pip does not look at the
# image's site-packages and reinstalls a baked package anyway. The executor closes the gap by
# probing the image once per process and dropping what it can already import
# (`CodeExecutor.preinstalled` / AGENT_CODE_EXEC_PREINSTALLED in agent_runtime/code_execution.py).
# The probe is why this list needs no matching declaration anywhere: add a package here and
# the executor notices; remove one and it silently goes back to installing it per run.
#
# The list below is exactly `_IMPORT_TO_PIP` in code_execution.py — the set the executor knows
# how to auto-install from an import statement — because those are the imports it will
# otherwise pay for. Keep the two in step.
FROM python:3.11-slim

ENV MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scipy \
    shapely \
    scikit-learn \
    statsmodels \
    pyarrow \
    networkx \
    requests \
    beautifulsoup4 \
    pillow \
    openpyxl

# Geospatial. No apt GDAL is needed: geopandas/fiona/rasterio/pyproj have shipped manylinux
# wheels with GEOS, PROJ and GDAL bundled for years, and installing the Debian gdal-dev stack
# beside them is how you end up with two PROJ data directories and silently wrong reprojection.
#
# libexpat1 IS needed, and its absence is nasty rather than obvious: fiona and rasterio import
# far enough for importlib to find them and then fail with
# `ImportError: libexpat.so.1: cannot open shared object file`. python:3.11-slim does not ship
# it; the bundled GDAL links against it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libexpat1 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    geopandas \
    pyogrio \
    fiona \
    rasterio \
    pyproj \
    mapclassify \
    folium

# Spatial statistics, so code the agent writes can `import esda` / `libpysal` / `spreg` /
# `pygeoda` without a per-run install. The curated tools in
# agent_runtime/analysis_spatial_stats_tools.py cover the common workflow (weights, Moran,
# LISA, Gi*, spatial regression, regionalization); these are here for everything past it —
# join counts, quantile LISA, GWR-style work, custom permutation schemes.
RUN pip install --no-cache-dir \
    libpysal \
    esda \
    spreg \
    pygeoda

# The run container is --read-only with only /work writable and a /tmp tmpfs, and runs as a
# uid that owns neither /root nor this image's HOME. Without these, matplotlib's font cache
# and pyproj's scratch writes hit a read-only filesystem and fail naming the library rather
# than the sandbox. HOME is also set on the docker run command line; setting it here keeps a
# direct `docker run` of this image behaving the same way.
ENV HOME=/tmp \
    XDG_CACHE_HOME=/tmp/cache

# MPLCONFIGDIR must be WRITABLE, and under --read-only the only writable paths are /work and
# the /tmp tmpfs. A font cache pre-built into the image cannot be used: matplotlib tests the
# directory for writability, and on failure abandons it entirely — cache and all — for a fresh
# temp dir, printing "is not a writable directory" plus a four-line recommendation to stderr on
# EVERY run. That noise lands in the stderr the agent reads back and reasons about, which is
# worth more than the ~0.5 s the cache would have saved.
ENV MPLCONFIGDIR=/tmp/mpl

# Warm the pyproj CRS database so the FIRST reprojection in a --network none container does not
# try to fetch a grid. The invariant gate reads `crs.axis_info` on every geometry frame, so this
# is on the hot path of every geospatial run, not an optimisation for a rare case.
RUN python -c "import pyproj; [pyproj.CRS.from_epsg(e) for e in (4326, 3857, 32616, 3435)]"

# Assert the image is actually usable, not merely installed — see verify_imports.py for why
# `pip install` succeeding is not evidence of that, and why the executor's probe depends on it.
COPY verify_imports.py /opt/verify_imports.py
RUN python /opt/verify_imports.py

WORKDIR /work
