"""Finding the URL an element records for its own source, across two incompatible vocabularies.

The platform names this field twice and the names do not overlap. Neo4j node properties are
snake_case and generic — ``external_link``. The REST API returns kebab-case and **type-suffixed**
names — ``external-link-publication``. Code that knew one vocabulary silently found nothing on the
other path, and "found nothing" was reported as a property of the element rather than of the
lookup.

Measured on 386 cached full records from ``/api/elements/{id}``:

* publications carry ``external-link-publication`` and **nothing** named ``external_link`` or
  ``external-link``;
* the library driver read ``meta.get("external_link")`` and reported ``unfetchable:no_doi`` for
  **195 of 203 publications**, every one of which has a DOI. The 8 that got through were the ones
  with an already-cached PDF, so the REST path had resolved exactly zero DOIs — for the type the
  whole open-access resolver was built for;
* the same field holds two shapes: 183 publications record an http URL and **19 record a bare
  DOI** (``10.1126/science.1144004``, ``DOI 10.1007/s13280-010-0098-0``). Requiring a URL scheme
  discards those 19, and there is no ``doi`` field to fall back on — the API's ``doi-status`` is a
  flag, not a value.
"""

from __future__ import annotations

import pytest

from extractors.open_access import extract_doi
from extractors.sources import SOURCE_LINK_FIELDS, source_link, source_link_or_doi


# ------------------------------------------------------------------ both vocabularies

@pytest.mark.parametrize("field", [
    "external-link-publication",     # what the REST API calls it for a publication
    "external_link_publication",
    "external-link",                 # generic kebab
    "external_link",                 # what the Neo4j node property is called
    "direct-download-link",
    "direct_download_link",
    "github-repo-link",
    "external-iframe-link",
    "notebook-url",
    "url",
])
def test_a_link_is_found_under_either_spelling(field):
    assert source_link({field: "https://example.org/paper.pdf"}) == \
        "https://example.org/paper.pdf"


def test_the_publication_field_the_driver_used_to_miss():
    """The exact record shape that produced `unfetchable:no_doi` for 195 of 203 elements."""
    record = {"id": "d1026b37", "resource-type": "publication", "doi-status": None,
              "external-link-publication": "https://doi.org/10.1088/1748-9326/ac1234"}
    assert source_link(record).endswith("10.1088/1748-9326/ac1234")
    assert extract_doi(source_link_or_doi(record)) == "10.1088/1748-9326/ac1234"


def test_a_type_suffixed_field_wins_over_a_generic_one():
    """Ordered most-specific first: an element carrying both should use the one that names its
    type, since the generic field is where a stale or secondary link tends to sit."""
    record = {"external-link": "https://generic.example/x",
              "direct-download-link": "https://specific.example/data.zip"}
    assert source_link(record) == "https://specific.example/data.zip"


def test_a_list_valued_field_yields_its_first_url():
    """OER elements record `oer_elink_urls` as a list."""
    assert source_link({"oer-elink-urls": ["not a url", "https://example.org/a",
                                           "https://example.org/b"]}) == "https://example.org/a"


# ------------------------------------------------------------------ bare DOIs

@pytest.mark.parametrize("value,expected_doi", [
    ("10.1126/science.1144004", "10.1126/science.1144004"),
    ("DOI 10.1007/s13280-010-0098-0", "10.1007/s13280-010-0098-0"),
    ("10.18653/v1/2025.emnlp-main.1095", "10.18653/v1/2025.emnlp-main.1095"),
    ("10.1109/ICDE53745.2022.00189", "10.1109/ICDE53745.2022.00189"),
])
def test_a_bare_doi_in_the_link_field_is_still_returned(value, expected_doi):
    record = {"external-link-publication": value}
    assert source_link(record) == "", "not a URL, so source_link declines it"
    assert extract_doi(source_link_or_doi(record)) == expected_doi


def test_a_url_is_preferred_over_a_bare_doi():
    record = {"external-link-publication": "https://doi.org/10.1/a", "doi": "10.2/b"}
    assert source_link_or_doi(record) == "https://doi.org/10.1/a"


@pytest.mark.parametrize("junk", ["", None, "false", "true", "None", "not a link", "n/a", 0])
def test_junk_values_are_not_mistaken_for_a_source(junk):
    assert source_link({"external-link-publication": junk}) == ""
    assert source_link_or_doi({"external-link-publication": junk}) == ""


def test_doi_status_is_a_flag_and_never_a_source():
    """`doi-status` is what the API returns alongside the link; treating it as a value would hand
    a caller the string "false" to resolve."""
    assert source_link_or_doi({"doi-status": False, "resource-type": "publication"}) == ""
    assert "doi-status" not in SOURCE_LINK_FIELDS


def test_an_element_with_no_source_returns_empty_rather_than_raising():
    assert source_link({}) == "" and source_link_or_doi({}) == ""


# ------------------------------------------------------------------ the real records

def test_the_lookup_finds_a_doi_for_the_corpus_publications():
    """Characterization against the cached API records, when they are present. 177 of 203
    publications carry an extractable DOI; the driver was finding 8."""
    import json
    from pathlib import Path

    meta_dir = Path(__file__).resolve().parents[2] / ".corpus_cache" / "_meta"
    if not meta_dir.is_dir():
        pytest.skip("no cached platform records in this checkout")
    records = []
    for path in meta_dir.glob("*.json"):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if str(record.get("resource-type")) == "publication":
            records.append(record)
    if len(records) < 100:
        pytest.skip(f"only {len(records)} publication records cached")

    with_doi = sum(1 for r in records if extract_doi(source_link_or_doi(r)))
    assert with_doi > len(records) * 0.8, (
        f"only {with_doi} of {len(records)} publications yielded a DOI — the link-field "
        f"vocabulary has probably drifted again")
