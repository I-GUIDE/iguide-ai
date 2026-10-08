"""Import name -> pip distribution, and a unit's real install set from its own slice.

Why this is derived from the SLICE and not from the notebook's dependency list: the slice is
what actually gets imported. A notebook may `pip install` ten packages while the one function
we extracted needs two, and — the failure that motivated this module — a unit may need a
package the notebook never declared because the declaration lived in a `!pip install` line or
in prose. Measured before this existed: all 16 units in the corpus library declared
``requirements: {}`` while their slices imported pandas, geopandas and smolagents, so the
contract handed to the agent was wrong and the import failed inside the sandbox.

For most packages the import name is the pip name. ``_ALIASES`` covers the ones where it is
not; anything unknown falls back to the import name, which is right far more often than it is
wrong, and the guess is reported in ``inferred`` so a wrong one is auditable rather than
silent.
"""

from __future__ import annotations

import ast
import sys
from typing import Dict, List, Set

# Import name -> pip distribution, for the cases where the two DIFFER.
_ALIASES: Dict[str, str] = {
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
    "osgeo": "GDAL",
    "mpl_toolkits": "matplotlib",
    "serial": "pyserial",
    "usb": "pyusb",
    "Bio": "biopython",
    "OpenSSL": "pyOpenSSL",
    "jwt": "PyJWT",
    "attr": "attrs",
    "google": "google-api-python-client",
    "pkg_resources": "setuptools",
    "IPython": "ipython",
    "ee": "earthengine-api",
}

# Packages whose import name IS the distribution name, confirmed rather than assumed. Kept
# separate from ``_ALIASES`` so ``inferred`` means what it says: a name nobody checked. Folding
# these in as identity entries made a verified name and a guess indistinguishable.
_VERIFIED_IDENTITY: Set[str] = {
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "requests", "geopandas", "shapely",
    "fiona", "rasterio", "pyproj", "netCDF4", "xarray", "rioxarray", "geemap", "folium",
    "contextily", "osmnx", "networkx", "pysal", "libpysal", "esda", "mgwr", "access",
    "nbformat", "torch", "tensorflow", "statsmodels", "plotly", "bokeh", "dask", "numba",
    "pyarrow", "openpyxl", "tqdm", "boto3", "smolagents",
}

# Local/relative and generated-package names that must never become a pip requirement.
_NEVER: Set[str] = {"iguide_methods", "__future__", "__main__"}

_STDLIB: Set[str] = set(getattr(sys, "stdlib_module_names", set())) | {
    "typing_extensions",  # not stdlib, but never worth pinning on its own
}


def distribution_for(import_name: str) -> str:
    """pip distribution for a top-level import name; ``""`` when nothing is needed."""
    root = (import_name or "").split(".")[0].strip()
    if not root or root in _NEVER or root in _STDLIB:
        return ""
    if root in _ALIASES:
        return _ALIASES[root]
    return root


def top_level_imports(source: str) -> List[str]:
    """Root import names bound anywhere in *source*, including inside functions.

    Walks the whole tree, not just module level: a unit that imports matplotlib inside its
    body still needs matplotlib installed to run, and reporting only module-level imports
    would understate the install set for exactly the lazy-import style that is common in
    notebook code.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    roots: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.extend(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:        # relative import: internal to the generated package
                continue
            if node.module:
                roots.append(node.module.split(".")[0])
    seen, out = set(), []
    for r in roots:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def requirements_from_source(source: str) -> Dict[str, List[str]]:
    """``{"pip": [...], "inferred": [...]}`` — the install set for one slice.

    ``inferred`` lists the distributions whose name was assumed to equal the import name
    rather than looked up. They are still installed; naming them keeps a wrong guess visible
    instead of turning into a confusing "no matching distribution" at run time.
    """
    pip: List[str] = []
    inferred: List[str] = []
    for name in top_level_imports(source):
        dist = distribution_for(name)
        if not dist:
            continue
        if dist not in pip:
            pip.append(dist)
            if name not in _ALIASES and name not in _VERIFIED_IDENTITY:
                inferred.append(dist)
    return {"pip": sorted(pip), "inferred": sorted(inferred)}


__all__ = ["distribution_for", "top_level_imports", "requirements_from_source"]
