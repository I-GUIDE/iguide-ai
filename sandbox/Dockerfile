# Pre-baked sandbox image for agent code execution (AGENT_CODE_EXEC_IMAGE).
#
#   docker build -t iguide-codeexec sandbox/
#   export AGENT_CODE_EXEC_IMAGE=iguide-codeexec
#
# With the common scientific stack already installed, queries like
# "plot y=x^2 and save plot.png" run WITHOUT a per-run `pip install`, which avoids
# the install-phase disk/time cost entirely. The agent only needs `dependencies`
# for packages not baked here.
FROM python:3.11-slim

ENV MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scipy \
    shapely

# Pre-build the matplotlib font cache so first-run plotting is fast and needs no
# writable HOME beyond /tmp.
RUN python -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; plt.plot([0,1]); plt.savefig('/tmp/_warm.png')" && rm -f /tmp/_warm.png

# Geospatial stack, ENABLED. This is the point of a pre-baked image for this system: without
# it every session that touches a GeoDataFrame pays a `pip install geopandas` before any work
# happens, and the method library's strongest clusters (2SFCA accessibility, SPASTC region
# building, remote sensing) are all geospatial.
#
# Installed from WHEELS rather than by building against system GDAL. pyogrio/rasterio/fiona all
# publish manylinux wheels that vendor their own GDAL, so the apt route below is neither needed
# nor faster — it adds a compiler toolchain and ~400 MB to the image for the same result. Kept
# as a comment because a future package without a wheel for the target platform would need it.
#
#   apt-get install -y gdal-bin libgdal-dev && CPLUS_INCLUDE_PATH=/usr/include/gdal pip install …
# Runtime shared libraries the geospatial WHEELS dlopen but do not vendor. Without libexpat1,
# `import rasterio` dies with "libexpat.so.1: cannot open shared object file" — the wheel
# installs cleanly and then fails at import, so a build that succeeds proves nothing. Caught by
# actually importing the stack in the built image rather than trusting `pip install`.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    geopandas \
    pyogrio \
    rasterio \
    pyproj \
    rtree \
    contextily \
    mapclassify

# Warm the pyproj CRS database so the FIRST reprojection in a --network none container does not
# try to fetch a grid. The invariant gate reads `crs.axis_info` on every geometry frame, so this
# is on the hot path of every geospatial run, not an optimisation for a rare case.
RUN python -c "import pyproj; [pyproj.CRS.from_epsg(e) for e in (4326, 3857, 32616, 3435)]"
