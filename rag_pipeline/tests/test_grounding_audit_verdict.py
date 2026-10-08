"""The grounding audit's verdict is computed from the ledger, in code.

Why this exists. Driving the prototype produced a `severity: high` hallucination caveat on a
fully-grounded answer: it listed four notebooks, every one present in the retrieved evidence.
Measured over 5 runs against the live index, the false-positive rate was **5/5**, and the
offending row was always the same one — the answer's opening roll-up sentence, "The platform
offers several notebooks that compute spatial accessibility to hospitals."

That sentence cannot have a verbatim supporting span; no document says "the platform offers
several". Its truth is carried by the items listed beneath it. Under the prompt's
one-verbatim-span-per-row rule it is therefore unprovable and lands as "absent", which
promotes to high severity and reaches the user.

Adding an explicit prompt rule moved the rate only 5/5 -> 4/5: the instruction competes with
the model's own judgement and loses. Recomputing the verdict deterministically from the
ledger took it to **0/5** while leaving true positives at **5/5**.

These tests cover the pure functions, so they need no LLM and no network.
"""

from __future__ import annotations

import pytest

from agent_runtime.evidence_quality import _is_rollup_claim, _recompute_verdict


# --------------------------------------------------------------- roll-up detection

@pytest.mark.parametrize("claim", [
    "The platform offers several notebooks that compute spatial accessibility to hospitals",
    "There are a number of relevant datasets",
    "These resources provide a range of methods and tools",
    "Various notebooks address this topic",
    "Multiple approaches are available",
    "The catalog includes numerous publications on flooding",
])
def test_vague_roll_ups_are_detected(claim):
    assert _is_rollup_claim(claim) is True


@pytest.mark.parametrize("claim", [
    "Three notebooks compute spatial accessibility",
    "12 datasets cover Chicago crime",
    "This is the only notebook that uses A2SFCA",
    "All of the datasets are in EPSG:4326",
    "The A2SFCA notebook was published in 2019",
    "Accessibility improved by 41.2 points on GeoBench-Pro",
    "Pysal Access Compute Example introduces CyberGIS-Compute",
])
def test_hard_counts_and_specifics_are_never_roll_ups(claim):
    """A hard count IS checkable and must stay auditable — this is the line that keeps the
    fix from becoming a hole a fabricated number could walk through."""
    assert _is_rollup_claim(claim) is False


def test_empty_claim_is_not_a_rollup():
    assert _is_rollup_claim("") is False
    assert _is_rollup_claim(None) is False  # type: ignore[arg-type]


# --------------------------------------------------------------- verdict recomputation

def _ledger(*rows):
    return {"claim_ledger": [{"claim": c, "status": s, "evidence_quote": q}
                             for c, s, q in rows],
            "hallucination_detected": True, "severity": "high", "issues": [], "summary": ""}


def test_rollup_absent_row_is_reclassified_when_real_claims_are_supported():
    """The exact production false positive."""
    parsed = _ledger(
        ("The platform offers several notebooks that compute accessibility", "absent", "none"),
        ("Pysal Access Compute Example introduces CyberGIS-Compute", "supported", "[b41c] ..."),
        ("SPASTC presents a scalable travel-time algorithm", "supported", "[b1fa] ..."),
    )
    out = _recompute_verdict(parsed)
    assert out["severity"] == "none"
    assert out["hallucination_detected"] is False
    assert out["issues"] == []
    assert out["rollup_claims_reclassified"]


def test_a_real_unsupported_claim_still_flags():
    parsed = _ledger(
        ("The platform offers several notebooks", "absent", "none"),
        ("Pysal Access Compute Example introduces CyberGIS-Compute", "supported", "[b41c] ..."),
        ("Published in the Journal of Geospatial Analytics in 2019", "absent", "none"),
    )
    out = _recompute_verdict(parsed)
    assert out["hallucination_detected"] is True
    assert out["severity"] == "high"
    claims = " ".join(i["claim"] for i in out["issues"])
    assert "Journal of Geospatial Analytics" in claims
    assert "offers several notebooks" not in claims, "the roll-up should not be an issue"


def test_contradiction_is_preserved():
    parsed = _ledger(
        ("The dataset covers 2019", "contradicted", "[abc] data covers 2021"),
        ("Chicago crime data is available", "supported", "[def] ..."),
    )
    out = _recompute_verdict(parsed)
    assert out["hallucination_detected"] is True
    assert out["severity"] == "high"
    assert "contradicts" in out["issues"][0]["reason"]


def test_rollup_is_not_excused_when_nothing_real_is_supported():
    """If no genuine claim is supported the answer is not merely over-summarising — the
    roll-up must not be waved through on an otherwise-empty ledger."""
    parsed = _ledger(
        ("The platform offers several notebooks", "absent", "none"),
        ("Some other thing", "absent", "none"),
    )
    out = _recompute_verdict(parsed)
    assert out["hallucination_detected"] is True
    assert out["severity"] == "high"


def test_issues_are_re_derived_not_trusted():
    """The prompt says issues = the contradicted/absent rows; nothing enforced it, and a
    model can raise an issue for a row it marked supported."""
    parsed = {
        "claim_ledger": [{"claim": "A real supported thing", "status": "supported",
                          "evidence_quote": "[x] span"}],
        "hallucination_detected": True,
        "severity": "high",
        "issues": [{"claim": "A real supported thing", "reason": "model changed its mind"}],
        "summary": "",
    }
    out = _recompute_verdict(parsed)
    assert out["issues"] == [], "an issue survived for a row marked supported"
    assert out["severity"] == "none"
    assert out["hallucination_detected"] is False


def test_clean_label_over_unsupported_rows_is_corrected_upward():
    """The ledger wins in BOTH directions: a 'none' label with absent rows becomes high."""
    parsed = {
        "claim_ledger": [{"claim": "Published in Nature in 2020", "status": "absent",
                          "evidence_quote": "none"}],
        "hallucination_detected": False, "severity": "none", "issues": [], "summary": "",
    }
    out = _recompute_verdict(parsed)
    assert out["hallucination_detected"] is True
    assert out["severity"] == "high"


def test_missing_ledger_leaves_the_verdict_untouched():
    """Never invent a clean verdict for an audit that produced no ledger."""
    for parsed in ({"severity": "high", "hallucination_detected": True, "issues": [{"claim": "x"}]},
                   {"claim_ledger": [], "severity": "unknown", "hallucination_detected": False}):
        before = dict(parsed)
        out = _recompute_verdict(parsed)
        assert out["severity"] == before["severity"]
        assert out["hallucination_detected"] == before["hallucination_detected"]


def test_non_dict_rows_do_not_crash_the_audit():
    parsed = {"claim_ledger": ["not a dict", None], "severity": "high",
              "hallucination_detected": True, "issues": [], "summary": ""}
    out = _recompute_verdict(parsed)
    assert out["severity"] == "high"  # unusable ledger -> model's verdict stands
