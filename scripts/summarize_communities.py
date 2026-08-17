#!/usr/bin/env python3
"""Write one grounded report per community — the part of the GraphRAG pattern retrieval reads.

    LLM_PROVIDER=claude-cli CLAUDE_CLI_MODEL=sonnet \
      python scripts/summarize_communities.py --graph outputs/corpus_graph

The partition is the cheap half. A community is only usable by a person or a retrieval step once
something says what it *is*, and in GraphRAG the community report — not the clustering — is what
global search actually consumes.

One call per community (15 for this corpus), so this is minutes and cents rather than a batch job.

TWO GUARDS, both of which exist because of measured failures in this project:

**Grounding.** The model is given ONLY member titles, tags and type counts, and every claim in
the summary must rest on them. Any ``key_theme`` it returns that does not appear as a substring of
the input is dropped and counted. An earlier entity-extraction pass without this check was fine
(0.4% rejection) but the check is what makes that a measurement rather than a hope.

**The single-uploader caveat.** Five of this corpus's fifteen communities are one contributor's
bulk upload split five ways — 160 map elements at contributor concentration >= 0.97. A summary
that describes those as a research theme is actively misleading, so concentration is passed in
and the model is required to name it. It is not left to the model to notice.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PROMPT = """You are describing one cluster of research artifacts from the I-GUIDE geospatial \
knowledge platform, for a researcher browsing the corpus.

FACTS ABOUT THIS CLUSTER (this is all you know; do not use outside knowledge):
- {size} elements
- resource types: {types}
- distinctive tags (ranked by how over-represented they are here vs the whole corpus): {tags}
- most central members by connection strength:
{members}
- other member titles (sample): {others}
- {concentration_pct}% of members were uploaded by a single contributor{who}

RESPOND WITH STRICT JSON, no prose outside it:
{{
  "title": "<4-8 words naming what holds this cluster together>",
  "summary": "<2-3 sentences. What is this cluster, what would a researcher find here, and what \
would they use it for. Concrete, no filler.>",
  "key_themes": ["<3-5 short phrases, each drawn from the tags or titles above>"],
  "caveat": "<one sentence, or empty string. REQUIRED and non-empty if the single-contributor \
percentage is 90 or above: say plainly that this is largely one contributor's upload batch \
rather than an independently-formed research theme.>"
}}

Rules: every claim must be supported by the facts above. Do not invent method names, places, or \
findings that do not appear there. If the cluster looks incoherent, say so in the summary rather \
than manufacturing a theme."""


_STOP = {"and", "for", "the", "with", "of", "in", "to", "a", "an", "on", "at", "by", "or",
         "from", "using", "based", "data", "analysis", "level"}


def _stem(w: str) -> str:
    """Crudest useful stemmer: enough to see that ``mapping`` and ``map`` are the same word.

    Without this, ``"flood risk mapping"`` was rejected against a community whose tags are
    ``flood risk`` and ``flood map`` — two of three tokens matched, 0.67, just under threshold.
    The rejection was an inflection, not a fabrication.
    """
    for suf in ("ings", "ing", "edly", "ed", "ies", "ied", "es", "s", "tions", "tion",
                "ally", "ment"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[:-len(suf)]
            break
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiou":
        w = w[:-1]                      # mapp -> map
    return w


def grounded(theme: str, haystack: str, *, min_share: float = 0.7) -> bool:
    """Is ``theme`` built from words the model was actually shown?

    Token-level, deliberately NOT a verbatim substring test. The first version of this required
    the whole phrase to appear in the prompt and rejected **93%** of themes — including
    ``"flood risk mapping"`` for a community whose tags are ``flood risk`` and ``flood map``, and
    ``"county-level flood hazard"`` for members titled *County-Level FEMA Flood Risk Maps*. Those
    are paraphrases composed of grounded terms, which is what a good summary phrase is. The check
    was wrong and the model was right — the same mistake as comparing an import name against a
    distribution name (`ee` vs `earthengine-api`) in M7.2.

    What is still caught is the thing worth catching: a phrase naming a method, place or finding
    that has no basis in the member titles or tags at all.
    """
    words = [w for w in "".join(c if c.isalnum() or c.isspace() else " "
                               for c in theme.lower()).split()
             if len(w) > 2 and w not in _STOP]
    if not words:
        return False
    hay_stems = {_stem(w) for w in haystack.split()}
    hits = sum(1 for w in words if w in haystack or _stem(w) in hay_stems)
    return hits / len(words) >= min_share


def _parse(text: str) -> dict:
    """Pull the JSON object out of a model reply that may be fenced or padded."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("```")[1] if "```" in t[3:] else t[3:]
        t = t[4:] if t.lower().startswith("json") else t
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no JSON object in reply")
    return json.loads(t[a:b + 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=str(REPO / "outputs" / "corpus_graph"))
    ap.add_argument("--max-members", type=int, default=18,
                    help="member titles sent per community; the cost dial")
    ap.add_argument("--recheck", action="store_true",
                    help="re-apply the grounding check to saved reports, making NO LLM calls. "
                         "Exists because the check itself was wrong once, and re-scoring the "
                         "same outputs is the honest way to fix that without paying again or "
                         "silently swapping in different model text.")
    args = ap.parse_args()

    if not args.recheck and (os.getenv("LLM_PROVIDER") or "").strip() != "claude-cli":
        print("note: LLM_PROVIDER is not claude-cli; this will use whatever provider is "
              "configured and may cost API credit.", file=sys.stderr)

    gdir = Path(args.graph)
    payload = json.loads((gdir / "graph.json").read_text(encoding="utf-8"))
    profiles = json.loads((gdir / "communities.json").read_text(encoding="utf-8"))
    by_id = {n["id"]: n for n in payload["nodes"]}

    existing: dict = {}
    if args.recheck:
        existing = json.loads((gdir / "community_reports.json").read_text(encoding="utf-8"))
        print(f"rechecking {len(existing)} saved report(s), no LLM calls")

    reports, rejected_total, failures = {}, 0, []
    t0 = time.time()
    for prof in profiles:
        cid = prof["community"]
        members = [by_id[m] for m in prof["members"] if m in by_id]
        central_ids = {c["id"] for c in prof["central_members"]}
        others = [m["t"] for m in members if m["id"] not in central_ids][:args.max_members]
        conc = int(round(prof["contributor_concentration"] * 100))

        prompt = PROMPT.format(
            size=prof["size"],
            types=", ".join(f"{k} {v}" for k, v in prof["types"].items()),
            tags=", ".join(t["tag"] for t in prof["distinctive_tags"]) or "(none distinctive)",
            members="\n".join(f"  - [{c['resource_type']}] {c['title']}"
                              for c in prof["central_members"]),
            others="; ".join(others) or "(none)",
            concentration_pct=conc,
            who=f" ({prof['top_contributor']})" if prof.get("top_contributor") else "",
        )
        if args.recheck:
            rep = dict(existing.get(str(cid)) or {})
            if not rep:
                failures.append((cid, "no saved report"))
                continue
            # Re-score the FULL original theme list, not just what survived last time.
            rep["key_themes"] = list(rep.get("key_themes") or []) + \
                list(rep.get("themes_rejected") or [])
        else:
            from rag_pipeline.llm_utils import call_llm
            try:
                rep = _parse(call_llm(prompt))
            except Exception as exc:
                failures.append((cid, f"{type(exc).__name__}: {exc}"))
                print(f"  C{cid:<3} FAILED  {type(exc).__name__}: {str(exc)[:90]}")
                continue

        # Grounding: is the phrase built from words the model was shown? Token-level, not a
        # verbatim substring — see grounded() for why the substring version was wrong.
        haystack = prompt.lower()
        kept, dropped = [], []
        for theme in (rep.get("key_themes") or []):
            (kept if grounded(str(theme), haystack) else dropped).append(theme)
        rejected_total += len(dropped)
        rep["key_themes"] = kept
        rep["themes_rejected"] = dropped

        if conc >= 90 and not str(rep.get("caveat") or "").strip():
            # Required, so it is supplied rather than silently missing.
            rep["caveat"] = (f"{conc}% of these {prof['size']} elements come from one "
                             f"contributor — this is largely one upload batch, not an "
                             f"independently-formed research theme.")
            rep["caveat_supplied"] = True

        reports[str(cid)] = rep
        flag = " ⚠" if str(rep.get("caveat") or "").strip() else ""
        print(f"  C{cid:<3} {rep.get('title','?')[:56]:<58}{flag}")

    dt = time.time() - t0
    kept_total = sum(len(r.get("key_themes") or []) for r in reports.values())
    calls = 0 if args.recheck else len(profiles)
    print(f"\n{len(reports)}/{len(profiles)} reports in {dt:.0f}s · {calls} LLM calls · "
          f"{kept_total} themes kept, {rejected_total} dropped as ungrounded"
          + (f" ({rejected_total/(kept_total+rejected_total):.0%})"
             if kept_total + rejected_total else ""))
    if failures:
        print(f"FAILURES ({len(failures)}): " + "; ".join(f"C{c}" for c, _ in failures))

    (gdir / "community_reports.json").write_text(json.dumps(reports, indent=1), encoding="utf-8")
    payload["reports"] = reports
    (gdir / "graph.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {gdir/'community_reports.json'} and merged into graph.json")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
