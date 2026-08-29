"""`stage_element` has to accept the id the rest of the system actually holds.

The 8-character short id is the canonical form everywhere in extraction — `doc_ids`, the
method-library registry, the Postgres record, every `provenance.element_id` — because that is
what the ingest pipeline stores. The platform's REST endpoint takes only the full UUID, so
`/api/elements/265e6957` 404s and staging answered "no platform record found".

That mismatch was not cosmetic, and an A/B of the knowledge base is what surfaced it. The
evidence view emits, at the point of use, `FIRST call stage_element("265e6957") to obtain
staged_path` — the id it holds is the short one — so the KB was steering the agent into a call
that could never succeed. On a two-dataset proximity problem the agent spent 2h07m and 21 KB
tool calls following that instruction and returned no answer; the same agent with the KB ablated
staged the files and answered correctly in 15 minutes.
"""

from __future__ import annotations

import pytest

from agent_runtime import staging


@pytest.fixture(autouse=True)
def _clear_index(monkeypatch):
    monkeypatch.setattr(staging, "_id_index", {}, raising=False)


def test_a_full_uuid_is_returned_untouched_and_costs_no_lookup(monkeypatch):
    """The common path must not pay for the rare one."""
    def explode(*a, **kw):  # pragma: no cover - fails the test if reached
        raise AssertionError("the index was fetched for an id that needed no resolution")

    monkeypatch.setattr(staging, "_element_index", explode)
    full = "265e6957-7ecf-48e0-a179-b498533eb627"
    assert staging.resolve_element_id(full) == full


def test_a_short_id_resolves_through_the_index(monkeypatch):
    monkeypatch.setattr(staging, "_element_index",
                        lambda: {"265e6957": "265e6957-7ecf-48e0-a179-b498533eb627"})
    assert (staging.resolve_element_id("265e6957")
            == "265e6957-7ecf-48e0-a179-b498533eb627")


def test_an_unknown_short_id_is_passed_through_rather_than_guessed(monkeypatch):
    """The caller then gets the platform's own "not found", which names the id it looked for.
    Substituting a near-match would stage the WRONG dataset and report success."""
    monkeypatch.setattr(staging, "_element_index", lambda: {"abcdef12": "abcdef12-0000-0000-0000-000000000000"})
    assert staging.resolve_element_id("99999999") == "99999999"


@pytest.mark.parametrize("value", ["", "   ", "not-an-id", "https://example.org/x.csv",
                                   "265e6957-7ecf"])
def test_anything_that_is_not_an_id_prefix_is_left_alone(monkeypatch, value):
    def explode():  # pragma: no cover
        raise AssertionError(f"looked up {value!r} as an element id")

    monkeypatch.setattr(staging, "_element_index", explode)
    assert staging.resolve_element_id(value) == value.strip()


def test_the_index_is_built_once_per_process(monkeypatch):
    calls = []

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"elements": [{"id": "265e6957-7ecf-48e0-a179-b498533eb627"},
                                 {"id": "b32bec3e-1d0c-42e6-a62b-222ec6373fa7"}]}

    class _Requests:
        @staticmethod
        def get(url, params=None, timeout=None):
            calls.append(params)
            return _Resp()

    monkeypatch.setitem(__import__("sys").modules, "requests", _Requests)
    assert staging._element_index()["265e6957"] == "265e6957-7ecf-48e0-a179-b498533eb627"
    staging._element_index()
    assert len(calls) == 1, "the listing was fetched more than once"


def test_the_listing_is_paged_with_the_parameter_the_api_honours(monkeypatch):
    """`limit` is ACCEPTED and ignored, returning the default page of ten — which would resolve
    only the first ten elements on the platform and fail silently for every other one."""
    seen = {}

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"elements": []}

    class _Requests:
        @staticmethod
        def get(url, params=None, timeout=None):
            seen.update(params or {})
            return _Resp()

    monkeypatch.setitem(__import__("sys").modules, "requests", _Requests)
    staging._element_index()
    assert "size" in seen and int(seen["size"]) > 750, seen
    assert "limit" not in seen


@pytest.mark.integration
def test_a_real_short_id_stages_against_the_live_platform():
    """The end-to-end property, against the public API. Read-only."""
    result = staging.stage_element("265e6957", session_id="pytest_short_id")
    assert result.get("staged_path"), result
    assert int(result.get("bytes") or 0) > 0
