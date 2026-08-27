"""Extracted cells are named for what they ARE.

Every block used to be titled ``<element> — cell 12``. That identifies a cell and describes
nothing, so a retrieved block told the agent (and the browser reader) only where it sat. The
author's own name for the step was one field away in ``markdown_context``.

These tests pin the FALLBACK ORDER, because that is the part that silently rots: adding a tier
in the wrong place, or letting a lower tier fire while a higher one had an answer, degrades
titles across the whole corpus without failing anything else.
"""

import pytest

from extractors.notebook_extractor import _block_title, _is_prose, _leading_comment


# --- tier 1: the author's markdown heading ------------------------------------------------

def test_a_markdown_heading_becomes_the_title():
    assert _block_title("## 1c. Project All Files to the Same CRS", "gdf.to_crs(3857)", 10, []) \
        == "1c. Project All Files to the Same CRS"


def test_a_heading_wins_over_the_prose_that_follows_it():
    md = "# Import Data\nWe start by reading the shapefiles from disk."
    assert _block_title(md, "gpd.read_file('a.shp')", 4, []) == "Import Data"


def test_a_heading_deeper_in_the_markdown_still_wins_over_prose_above_it():
    """`md_context` is every markdown cell since the last code cell, so a narrative paragraph
    can precede the heading that actually names this step."""
    md = "Some closing words about the previous step.\n\n### 2. Build the network graph"
    assert _block_title(md, "ox.graph_from_place(p)", 7, []) == "2. Build the network graph"


# --- tier 2: the author's prose ------------------------------------------------------------

def test_prose_is_used_when_there_is_no_heading():
    md = "Now we calculate the distance to the nearest airport. Then we plot it."
    assert _block_title(md, "x = 1", 3, []) == "Now we calculate the distance to the nearest airport"


def test_markup_is_flattened_because_titles_render_as_plain_text():
    md = "<b>Comparisons</b> evaluate **relative** values in `pandas`"
    title = _block_title(md, "x = 1", 3, [])
    assert title == "Comparisons evaluate relative values in pandas"
    assert "<" not in title and "*" not in title and "`" not in title


def test_a_long_sentence_is_clipped_rather_than_dropped():
    md = "We " + "very " * 40 + "long."
    title = _block_title(md, "x = 1", 3, [])
    assert len(title) <= 73 and title.endswith("…")


# --- tier 3: the comment block at the top of the code --------------------------------------

def test_a_leading_comment_names_a_cell_with_no_markdown():
    code = "# check joined data\ndouglas_comm[douglas_comm['douglas'] > 0]"
    assert _block_title("", code, 12, []) == "check joined data"


def test_banner_decoration_is_skipped_to_reach_the_real_comment():
    code = "# ==========================\n# Train the model here\n# ====\nmodel.fit(X, y)"
    assert _block_title("", code, 12, []) == "Train the model here"


@pytest.mark.parametrize("text, prose", [
    ("Define callbacks", True),
    ("check joined data (which communities have a value)", True),
    ("Mapping", True),                      # a lone word parses as an expression, still prose
    ("Results", True),
    ("model = model.to('cuda')", False),    # commented-out code, not a title
    ("from google.colab import drive", False),
    ("df.head()", False),
])
def test_commented_out_code_is_not_mistaken_for_a_sentence(text, prose):
    assert _is_prose(text) is prose


def test_a_cell_that_opens_with_commented_out_code_falls_past_that_comment():
    code = "# import geopandas as gpd\nx = 1"
    assert _leading_comment(code) == ""
    assert _block_title("", code, 9, []) == "cell 9"


# --- tier 4: what the cell defines ----------------------------------------------------------

def test_a_definition_cell_is_named_for_what_it_defines():
    code = "def reproject_tif(crs, path):\n    return path"
    assert _block_title("", code, 5, []) == "Define reproject_tif()"


def test_several_definitions_are_summarised_not_listed():
    code = "def a():\n    pass\ndef b():\n    pass\nclass C:\n    pass\ndef d():\n    pass"
    assert _block_title("", code, 5, []) == "Define a(), b() +more"


def test_a_leading_comment_outranks_the_definition_because_the_author_wrote_it():
    code = "# Helpers for the buffer step\ndef reproject_tif(crs):\n    pass"
    assert _block_title("", code, 5, []) == "Helpers for the buffer step"


# --- tier 5/6: inferred from the code, then the old positional name -------------------------

def test_an_unnarrated_cell_is_named_from_what_the_code_does():
    assert _block_title("", "gdf = gdf.to_crs(epsg=3857)", 8, []) == "Reproject"
    assert _block_title("", "!pip install geopandas", 1, []) == "Install dependencies"


def test_declared_packages_outrank_the_keyword_guess():
    constructs = [{"category": "DEPENDENCY", "detail": {"packages": ["rasterio", "pyproj"]}}]
    assert _block_title("", "%pip install rasterio pyproj", 1, constructs) \
        == "Install rasterio, pyproj"


def test_a_cell_with_nothing_to_go_on_keeps_its_positional_name():
    assert _block_title("", "df", 15, []) == "cell 15"
    assert _block_title("", "", 15, []) == "cell 15"


def test_the_tiers_are_ordered_heading_prose_comment_definition_code():
    """One cell carrying every signal at once resolves top-down, and removing each winner in
    turn exposes the next tier. This is the property the individual tests above assume."""
    code = "# A leading comment\ndef helper():\n    return gpd.read_file('x.shp')"
    assert _block_title("# Heading\nProse sentence here.", code, 2, []) == "Heading"
    assert _block_title("Prose sentence here.", code, 2, []) == "Prose sentence here"
    assert _block_title("", code, 2, []) == "A leading comment"
    assert _block_title("", code.split("\n", 1)[1], 2, []) == "Define helper()"
    assert _block_title("", "gpd.read_file('x.shp')", 2, []) == "Load data"
