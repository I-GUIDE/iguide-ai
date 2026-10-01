"""The answer must describe the artifact that was actually produced.

Both checks come from one live query — "show me the clay embedding of urbana at
2025/03/01-2025/05/01" — which ran embed_region with no `model`, embedded with the default
(gse), and then described the result as "the Clay v1.5 embedding ... extracted from the global
LGND Clay Embeddings - Sentinel-2 collection ... 2.56 km MajorTOM grid cell" while the map
legend beside it read "gse embedding (PCA-RGB)". The same answer told the user to add the
layer from a URL, when it was already rendered. The LLM grounding audit flagged none of it,
which is why these are deterministic.
"""

from __future__ import annotations

import json

import pytest

from agent_runtime.supervisor import graph as g

CATALOG = frozenset({"clay", "gse", "tessera", "prithvi", "terrafm", "thor"})

# The answer as it actually shipped, trimmed.
CONFABULATED = (
    "Here's the Clay v1.5 embedding for Urbana, Illinois covering 2025-03-01 to 2025-05-01.\n"
    "The embedding was extracted from the global LGND Clay Embeddings - Sentinel-2 "
    "collection, which stores 1024-dimensional vectors for each 2.56 km MajorTOM grid cell.\n"
    'Add to the I-Guide map - in the map sidebar, choose "Add Layer -> From URL" and paste '
    "the download link above."
)
# `content`, not `output`: extract_search_artifacts emits {name, tool_call_id, content} and so
# do both CLI peers — nothing in the tree produces an `output` key. The invented shape passed
# only because the old delivery check walked every nested dict looking for a `map_layer`
# anywhere, which is the looseness this fixture now stops relying on.
GSE_RUN = {"tool_results": [{"name": "embed_region", "tool_call_id": "c1", "content": json.dumps({
    "ok": True, "model": "gse",
    "map_layer": {"url": "/f/1", "label": "gse embedding (PCA-RGB)"}})}]}


@pytest.fixture(autouse=True)
def catalog(monkeypatch):
    monkeypatch.setattr(g, "_EMBED_MODELS_CACHE", CATALOG)


def test_models_are_read_from_the_run_not_the_prose():
    assert g._models_used(GSE_RUN) == {"gse"}
    assert g._models_named_in(CONFABULATED) == {"clay"}
    # A word that is not a model id must not be mistaken for one.
    assert g._models_named_in("the embedding of urbana in illinois") == set()


def test_the_wrong_model_claim_is_corrected():
    out = g._correct_artifact_claims(CONFABULATED, GSE_RUN)
    assert out != CONFABULATED
    assert "produced by the **gse** model, not clay" in out
    # and it says why the fabricated provenance is wrong, not merely that the name differs
    assert "describes that other model" in out


def test_an_answer_that_denies_a_delivered_map_is_corrected():
    out = g._correct_artifact_claims(CONFABULATED, GSE_RUN)
    assert "already on your interactive map" in out


@pytest.mark.parametrize("phrase", [
    'choose "Add Layer -> From URL" and paste the download link',
    "add the layer to your map",
    "paste the download link into the viewer",
    "load it into the map yourself",
    "you can add this to the map",
])
def test_map_denial_phrasings(phrase):
    assert g._DENIES_MAP_RE.search(phrase), phrase


@pytest.mark.parametrize("phrase", [
    "the heat map shows density",
    "I added the layer to your map already",   # past tense, describing what happened
    "the map layer is ready",
])
def test_phrases_that_are_not_denials(phrase):
    # "I added ... already" is a description, not an instruction; the correction would be noise.
    if g._DENIES_MAP_RE.search(phrase):
        assert "already" in phrase, f"false positive: {phrase}"


def test_a_correct_answer_is_left_alone():
    good = "Here's the gse embedding for Urbana. It is on your map; click a cell to inspect it."
    assert g._correct_artifact_claims(good, GSE_RUN) == good
    clay_run = {"tool_results": [{"name": "embed_region",
                                  "output": {"model": "clay", "map_layer": {"url": "/f/2"}}}]}
    ok = "Here's the clay embedding for Urbana."
    assert g._correct_artifact_claims(ok, clay_run) == ok


def test_an_empty_answer_is_not_decorated():
    assert g._correct_artifact_claims("", GSE_RUN) == ""


def test_the_checks_disable_themselves_without_a_catalog(monkeypatch):
    """An unreachable embedding service must not make the check wrong — only absent."""
    monkeypatch.setattr(g, "_EMBED_MODELS_CACHE", frozenset())
    out = g._correct_artifact_claims(CONFABULATED, GSE_RUN)
    assert "not clay" not in out                 # model check silent
    assert "already on your interactive map" in out   # this one needs no catalog


def test_the_catalog_probe_fails_safe(monkeypatch):
    monkeypatch.setattr(g, "_EMBED_MODELS_CACHE", None)

    def boom(*a, **k):
        raise OSError("service down")

    import requests
    monkeypatch.setattr(requests, "get", boom)
    assert g._known_embedding_models() == frozenset()
    assert g._models_named_in("clay embedding") == set()


def test_the_mismatch_observation_names_both_models():
    msg = g._MODEL_MISMATCH_OBSERVATION.format(wanted="clay", used="gse")
    assert "model='clay'" in msg and "gse" in msg
