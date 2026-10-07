"""Place extraction for spatial_search: the step that decided the method returned nothing.

Observed on the deployed agent: `spatial_search` reported 0 results for both place-scoped questions
in a five-question geospatial run, while keyword and semantic search returned 8 each. Neither cause
was in the OpenSearch query — the index has the geo_shape field mapped and 181 of 619 documents
carry a bounding box. Both were in turning the question into a geocodable place:

* "land cover change in the Amazon basin" -> spaCy tagged "Amazon" (LOC) and the extractor stopped
  there, discarding "basin". Google geocodes "Amazon basin"; it returns nothing for "Amazon".
* "UTM zone Champaign Illinois" -> the retrieval peer rewrites questions into keyword form, which
  strips the determiners and prepositions NER leans on. spaCy returned NO entities for that string
  while extracting both places from the original sentence.

And because only ``locations[0]`` was ever geocoded, one unresolvable candidate ended the search
instead of falling through to one that would have resolved.

The extraction tests run whichever path this machine has: the capitalization fallback where
``en_core_web_sm`` is absent (the dev Mac, CI) and spaCy NER where it is installed (the agent image).
Their expectations are the fallback's output, and three of them failed in the image. The NER tests
further down feed the extractor the entities that model returns, so the path production runs is
tested on every machine.
"""

from __future__ import annotations

import pytest

import rag_pipeline.search.spatial as SP


@pytest.fixture(autouse=True)
def _clear_bbox_cache():
    SP._BBOX_CACHE.clear()
    yield
    SP._BBOX_CACHE.clear()


BOX = {"type": "polygon", "coordinates": [[[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]]}


# --- extraction -------------------------------------------------------------------


@pytest.mark.parametrize("query,expected_first", [
    ("land cover change in the Amazon basin", "Amazon basin"),
    ("datasets for the Mississippi River basin", "Mississippi River basin"),
    ("water quality in the Chesapeake Bay watershed", "Chesapeake Bay watershed"),
    ("soil moisture in the Great Plains", "Great Plains"),
])
def test_a_named_feature_keeps_its_feature_word(query, expected_first):
    """The geocodable form comes FIRST; the bare entity may follow as a fallback."""
    got = SP.extract_locations_from_query(query)
    assert got, f"nothing extracted from {query!r}"
    assert got[0] == expected_first


def test_the_bare_entity_survives_as_a_later_candidate():
    got = SP.extract_locations_from_query("land cover change in the Amazon basin")
    assert got[0] == "Amazon basin" and "Amazon" in got


def test_a_keyword_style_query_still_yields_a_place():
    """This is the form the search peer actually sends, and NER finds nothing in it."""
    got = SP.extract_locations_from_query("UTM zone Champaign Illinois")
    assert any("Champaign" in g for g in got)


@pytest.mark.parametrize("query", [
    "EPSG code for WGS 84",
    "convert a GeoJSON to a COG with GDAL",
    "what does the STAC API spec cover",
])
def test_technical_terms_are_not_offered_as_places(query):
    assert SP.extract_locations_from_query(query) == []


def test_plain_place_names_are_unaffected():
    assert SP.extract_locations_from_query("flood data for Illinois") == ["Illinois"]
    assert SP.extract_locations_from_query("urban heat in Chicago") == ["Chicago"]


# --- the NER path, with the entities production's model returns -------------------


@pytest.fixture(scope="module")
def _blank_english():
    spacy = pytest.importorskip("spacy")
    return spacy.blank("en")


@pytest.fixture
def ner_returns(monkeypatch, _blank_english):
    """Make the extractor's NER path see *ents*, given as ``(text, label)`` pairs.

    The pairs are what en_core_web_sm 3.8.0 returned for each query in a replica of the deployed
    agent-api image. A blank English pipeline tokenizes the same way, so the spans line up, and the
    test needs no model.
    """
    def install(ents):
        def nlp(text):
            doc = _blank_english(text)
            spans = []
            for phrase, label in ents:
                start = text.index(phrase)
                span = doc.char_span(start, start + len(phrase), label=label)
                assert span is not None, f"{phrase!r} does not fall on token boundaries"
                spans.append(span)
            doc.ents = spans
            return doc

        monkeypatch.setattr(SP, "nlp", nlp)

    return install


@pytest.mark.parametrize("query,ents,expected", [
    ("soil moisture in the Great Plains", [("the Great Plains", "FAC")], ["Great Plains"]),
    ("water quality in the Chesapeake Bay watershed", [("the Chesapeake Bay", "LOC")],
     ["Chesapeake Bay watershed", "Chesapeake Bay"]),
    ("land use in the United States", [("the United States", "GPE")], ["United States"]),
    # A capital "The" can be part of the name, so it stays.
    ("flooding in The Hague", [("The Hague", "GPE")], ["The Hague"]),
])
def test_ner_drops_a_leading_lowercase_article(ner_returns, query, ents, expected):
    ner_returns(ents)
    assert SP.extract_locations_from_query(query) == expected


@pytest.mark.parametrize("query,ents,expected", [
    ("convert a GeoJSON to a COG with GDAL", [("GeoJSON", "GPE")], []),
    ("download a DEM from USGS for Colorado", [("USGS", "GPE"), ("Colorado", "GPE")], ["Colorado"]),
    ("GeoJSON of Chicago neighborhoods", [("GeoJSON", "GPE"), ("Chicago", "GPE")], ["Chicago"]),
    ("GeoJSON Champaign Illinois", [("GeoJSON", "GPE"), ("Champaign Illinois", "LOC")],
     ["Champaign Illinois"]),
])
def test_ner_entities_that_are_technical_terms_are_not_places(ner_returns, query, ents, expected):
    ner_returns(ents)
    assert SP.extract_locations_from_query(query) == expected


def test_a_rejected_entity_does_not_open_the_capitalization_fallback(ner_returns):
    """NER found a "place" here and it was a file format, so the query names none. Falling back
    would offer the capitalized first word instead."""
    ner_returns([("Convert", "PERSON"), ("GeoJSON", "GPE")])
    assert SP.extract_locations_from_query("Convert a GeoJSON to a COG with GDAL") == []
    assert SP._capitalized_candidates("Convert a GeoJSON to a COG with GDAL") == ["Convert"]


def test_the_fallback_still_runs_when_ner_finds_no_place(ner_returns):
    ner_returns([])
    assert SP.extract_locations_from_query("UTM zone Champaign Illinois") == ["Champaign Illinois"]


# --- the fallback's words: any script, and never across a sentence end ------------


@pytest.mark.parametrize("query,expected", [
    ("Find the deforestation rate for Rondônia", ["Rondônia", "Find"]),
    ("urban growth in São Paulo", ["São Paulo"]),
    ("wetlands of the Île-de-France region", ["Île-de-France region", "Île-de-France"]),
    # U+2019, the typographic apostrophe that smart punctuation and language models write.
    ("tourism on Martha’s Vineyard", ["Martha’s Vineyard"]),
    ("air pollution in Xi’an", ["Xi’an"]),
])
def test_the_fallback_reads_places_in_any_script(query, expected):
    """With [A-Z] these gave ["Find"], ["Paulo"], ["France region", "France"], ["Vineyard", "Martha"]
    and []."""
    assert SP._capitalized_candidates(query) == expected


def test_decomposed_text_reads_as_composed():
    """"o" + U+0302 is two code points, and the combining mark is not a word character."""
    assert SP._capitalized_candidates("Rondo\u0302nia deforestation") == ["Rondônia"]


@pytest.mark.parametrize("query,ents,expected", [
    ("Find the deforestation rate for Rondônia", [("Rondônia", "ORG")], ["Rondônia", "Find"]),
    # Before: "France region" and "France", which name the whole country.
    ("wetlands of the Île-de-France region", [("the Île-de-France", "ORG")],
     ["Île-de-France region", "Île-de-France"]),
    ("tourism on Martha’s Vineyard", [("Martha’s Vineyard", "ORG")], ["Martha’s Vineyard"]),
])
def test_ner_misses_a_non_ascii_place_and_the_fallback_finds_it(ner_returns, query, ents, expected):
    ner_returns(ents)
    assert SP.extract_locations_from_query(query) == expected


@pytest.mark.parametrize("query,expected", [
    # GeoAnalystBench instructions, where the next sentence's first word joined the run before it.
    ("tin-tungsten deposits in Tasmania. The analysis should focus", ["Tasmania"]),
    ("campsite data quality in Wyoming. You'll initially address", ["Wyoming", "You'll"]),
    ("sponge distribution at Catalina Island. Update the database", ["Catalina Island", "Update"]),
    ("areas that need protection using the Arcpy. First, project", ["Arcpy", "First"]),
    # "Python. The" was one run, so "Python." kept its period and was not recognized as a non-place.
    ("interpolation techniques in Python. The analysis", []),
    # A closing quote after the period ends the sentence as well. The typographic one is a word
    # character now, for "Xi’an", so without the rule it would carry the run on.
    ("flood data for Ohio.' Then map it", ["Ohio", "Then"]),
    ("flood data for Ohio.’ Then map it", ["Ohio", "Then"]),
])
def test_a_run_ends_at_the_end_of_a_sentence(query, expected):
    assert SP._capitalized_candidates(query) == expected


def test_the_bare_place_is_offered_when_ner_misses_it_before_a_sentence(ner_returns):
    ner_returns([("flood depth Illinois", "PERSON")])
    got = SP.extract_locations_from_query("flood depth Illinois. Then county summary")
    assert "Illinois" in got
    assert not any("." in candidate for candidate in got)   # was "Illinois. Then county", "Illinois. Then"


@pytest.mark.parametrize("query,expected", [
    ("flood risk in St. Louis", ["St. Louis"]),
    ("flood risk in St. Louis. Then map it", ["St. Louis", "Then"]),
    ("flood risk in ST. LOUIS", ["ST. LOUIS"]),
    ("census tracts in the U.S. Virgin Islands", ["U.S. Virgin Islands"]),
    ("transit in Washington D.C. Metro area", ["Washington D.C. Metro"]),
    ("wind farms in N. Dakota", ["N. Dakota"]),
    ("trails near Ft. Collins", ["Ft. Collins"]),
    ("glaciers on Mt. Rainier", ["Mt. Rainier"]),
    ("shipping at Sault Ste. Marie", ["Sault Ste. Marie"]),
    ("UTM zone Champaign Illinois", ["Champaign Illinois"]),
])
def test_a_period_inside_a_name_does_not_end_the_run(query, expected):
    assert SP._capitalized_candidates(query) == expected


# --- resolution -------------------------------------------------------------------


def test_resolution_falls_through_to_a_candidate_that_geocodes(monkeypatch):
    """One unresolvable candidate must not end the search."""
    tried = []

    def fake_geocode(location):
        tried.append(location)
        return BOX if location == "Amazon basin" else None

    monkeypatch.setattr(SP, "get_bounding_box", fake_geocode)
    monkeypatch.setattr(SP, "extract_locations_from_query", lambda q: ["Amazonia", "Amazon basin"])

    assert SP.resolve_query_bbox("anything") == BOX
    assert tried == ["Amazonia", "Amazon basin"]      # stopped at the first that resolved


def test_resolution_returns_none_when_nothing_geocodes(monkeypatch):
    monkeypatch.setattr(SP, "get_bounding_box", lambda loc: None)
    monkeypatch.setattr(SP, "extract_locations_from_query", lambda q: ["Nowhere", "Nowhere else"])
    assert SP.resolve_query_bbox("q") is None


def test_no_place_means_no_geocode_call_at_all(monkeypatch):
    def boom(_loc):
        raise AssertionError("geocoding must not be attempted without a candidate place")

    monkeypatch.setattr(SP, "get_bounding_box", boom)
    monkeypatch.setattr(SP, "extract_locations_from_query", lambda q: [])
    assert SP.resolve_query_bbox("EPSG code for WGS 84") is None


def test_geocode_results_are_cached_including_failures(monkeypatch):
    """Geocoding is paid and rate-limited; an unresolvable candidate must not be retried on every
    query that mentions it."""
    calls = {"n": 0}

    def counting(location):
        calls["n"] += 1
        return None if location == "Nowhere" else BOX

    monkeypatch.setattr(SP, "get_bounding_box", counting)

    assert SP._cached_bounding_box("Illinois") == BOX
    assert SP._cached_bounding_box("Illinois") == BOX
    assert calls["n"] == 1                       # hit served from cache

    assert SP._cached_bounding_box("Nowhere") is None
    assert SP._cached_bounding_box("Nowhere") is None
    assert calls["n"] == 2                       # the FAILURE was cached too

    assert SP._cached_bounding_box("ILLINOIS") == BOX
    assert calls["n"] == 2                       # keyed case-insensitively


def test_spatial_search_returns_empty_without_a_resolvable_place(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("OpenSearch must not be queried without a bounding box")

    monkeypatch.setattr(SP, "resolve_query_bbox", lambda q: None)
    monkeypatch.setattr(SP, "_os_client", boom)
    assert SP.get_spatial_search_results("EPSG code for WGS 84") == []
