"""A unit's install set, derived from its own slice.

Before this existed every unit shipped ``requirements: {}`` while its slice imported pandas
and geopandas — 16 of 16 on the real corpus. The contract handed to the agent said "no
dependencies", so nothing was installed and the advertised import died inside the sandbox
with ModuleNotFoundError. A contract that understates its dependencies is worse than one that
overstates them: the overstatement costs install time, the understatement fails the run.
"""

from __future__ import annotations

import pytest

from extractors.pkgmap import (distribution_for, requirements_from_source, top_level_imports)


# ------------------------------------------------------------------ name mapping

@pytest.mark.parametrize("import_name,expected", [
    ("sklearn", "scikit-learn"),
    ("skimage", "scikit-image"),
    ("cv2", "opencv-python"),
    ("PIL", "Pillow"),
    ("yaml", "PyYAML"),
    ("bs4", "beautifulsoup4"),
    ("osgeo", "GDAL"),
    ("dateutil", "python-dateutil"),
])
def test_known_mismatches_map_to_their_distribution(import_name, expected):
    assert distribution_for(import_name) == expected


@pytest.mark.parametrize("name", ["os", "json", "pathlib", "sys", "re", "typing", "dataclasses"])
def test_stdlib_needs_no_installation(name):
    assert distribution_for(name) == ""


def test_the_generated_package_is_never_a_requirement():
    """iguide_methods is the mount itself; pip-installing it would be nonsense."""
    assert distribution_for("iguide_methods") == ""


def test_an_unknown_package_falls_back_to_its_import_name():
    assert distribution_for("geopandas") == "geopandas"
    assert distribution_for("some_new_lib") == "some_new_lib"


def test_a_submodule_maps_via_its_root():
    assert distribution_for("sklearn.cluster") == "scikit-learn"
    assert distribution_for("os.path") == ""


# ------------------------------------------------------------------ import discovery

def test_imports_inside_a_function_body_still_count():
    """Lazy imports are common in notebook code and are still a hard runtime requirement."""
    src = "def f():\n    import matplotlib.pyplot as plt\n    return plt\n"
    assert "matplotlib" in top_level_imports(src)


def test_relative_imports_are_internal_and_never_requirements():
    src = "from .v_abc123 import helper\nfrom ..other import thing\n"
    assert requirements_from_source(src)["pip"] == []


def test_an_unparseable_slice_yields_no_requirements_rather_than_raising():
    assert requirements_from_source("def broken(:\n")["pip"] == []


def test_duplicate_imports_are_reported_once():
    src = "import pandas as pd\nimport pandas\nfrom pandas import DataFrame\n"
    assert requirements_from_source(src)["pip"] == ["pandas"]


# ------------------------------------------------------------------ the real shape

def test_a_realistic_slice_declares_what_it_imports():
    src = (
        "import pandas as pd\n"
        "import geopandas as gpd\n"
        "import os\n"
        "from sklearn.cluster import KMeans\n"
        "\n"
        "def load(path):\n"
        "    return gpd.read_file(path)\n"
    )
    reqs = requirements_from_source(src)
    assert reqs["pip"] == ["geopandas", "pandas", "scikit-learn"]
    assert "os" not in reqs["pip"]


def test_guessed_names_are_reported_so_a_wrong_guess_is_auditable():
    """A package nobody checked is a GUESS; 'geopandas' and 'scikit-learn' are both known."""
    reqs = requirements_from_source(
        "import geopandas\nfrom sklearn import cluster\nimport some_new_lib\n")
    assert reqs["inferred"] == ["some_new_lib"]
    assert set(reqs["pip"]) == {"geopandas", "scikit-learn", "some_new_lib"}


def test_a_stdlib_only_slice_legitimately_needs_nothing():
    src = "import json\nimport pathlib\n\ndef f(p):\n    return json.loads(pathlib.Path(p).read_text())\n"
    assert requirements_from_source(src) == {"pip": [], "inferred": []}
