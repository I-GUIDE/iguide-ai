#!/usr/bin/env python3
"""Count what is actually in the platform's Neo4j graph. READ ONLY.

    python scripts/graph_census.py                 # nodes and relationships by label/type
    python scripts/graph_census.py --compare-api   # reconcile against the public REST API

Written because three different node counts for this graph are in circulation and none of them
reconciles:

  * ``docs/DEVLOG.md:957-959`` records 799 elements / 819 graph nodes — a later measurement pass
    reported that this does not reproduce.
  * "3,205 nodes" appears in session notes with no query recorded next to it.
  * ``extractors/platform_graph.py``'s docstring records per-type denominators (notebook 200,
    publication 200, dataset 141, map 161, oer 44, code 38 = 784) that cannot all be label totals:
    publication 200 is *below* the 203 the public API returns, and the graph is the store the API
    is a view of, so it cannot hold fewer.

So this script exists to replace all three with one query and a printed number. Every statement it
makes comes from ``count()`` in the database, not from a sampled page.

SAFETY. Every query is ``MATCH … RETURN count(…)``. There is no write path here, deliberately —
this graph holds real user data (bookmarks, edit permissions, contributor links). Credentials are
read via python-dotenv, never by sourcing ``.env``: the password contains an unquoted shell
metacharacter and `set -a; source .env` corrupts it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def env_candidates(explicit: str = "") -> list[Path]:
    """Where to look for credentials, in order.

    A git worktree does not get the main checkout's untracked files, so `.env` is simply absent
    here — and copying a credentials file into every worktree duplicates secrets for no benefit.
    So the sibling checkout is a first-class fallback rather than something a developer has to
    discover.
    """
    if explicit:
        return [Path(explicit)]
    out = [REPO / ".env"]
    # /path/to/repo-branchname -> /path/to/repo  (the worktree naming convention here)
    name = REPO.name
    if "-" in name:
        out.append(REPO.parent / name.rsplit("-", 1)[0] / ".env")
    return out


def _load_env(explicit: str = "") -> Path | None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return None
    for path in env_candidates(explicit):
        if path.is_file():
            # override=False so a real exported variable still wins over the file.
            load_dotenv(path, override=False)
            return path
    return None


def _api_counts() -> dict:
    per: dict = {}
    total = None
    for start in range(0, 4000, 100):
        url = f"https://backend.i-guide.io/api/elements?from={start}&size=100"
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.load(r)
        if total is None:
            total = d.get("total-count")
        els = d.get("elements") or []
        if not els:
            break
        for e in els:
            per[e.get("resource-type")] = per.get(e.get("resource-type"), 0) + 1
    return {"total_count_field": total, "per_type": per, "sum": sum(per.values())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-api", action="store_true")
    ap.add_argument("--env-file", default="",
                    help="path to a .env holding NEO4J_*; defaults to this worktree's .env, "
                         "then the sibling main checkout's")
    args = ap.parse_args()
    loaded = _load_env(args.env_file)
    print(f"credentials from {loaded}" if loaded
          else "no .env found in " + " or ".join(str(p) for p in env_candidates(args.env_file)))

    from extractors import platform_graph

    if not platform_graph.is_enabled():
        print("NEO4J not configured in this environment — NOT ESTABLISHED, not estimated.\n"
              "Set these in .env (see .env.example:44-49), then re-run:\n"
              "  NEO4J_CONNECTION_STRING=bolt://<host>:7687   (or NEO4J_URI)\n"
              "  NEO4J_USER=neo4j                             (or NEO4J_USERNAME)\n"
              "  NEO4J_PASSWORD=<password>\n"
              "  NEO4J_DB=neo4j\n"
              "Load them with python-dotenv, never `set -a; source .env` — the password contains "
              "an unquoted shell metacharacter.")
        if args.compare_api:
            api = _api_counts()
            print(f"\nPublic REST API, measured now: total-count={api['total_count_field']}, "
                  f"enumerated={api['sum']}")
            for k, v in sorted(api["per_type"].items(), key=lambda kv: -kv[1]):
                print(f"  {k:<12} {v}")
        return 2

    driver = platform_graph._driver()
    db = os.getenv("NEO4J_DB") or None
    out: dict = {}
    try:
        with driver.session(database=db) as s:
            out["total_nodes"] = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            out["total_relationships"] = s.run(
                "MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

            print(f"TOTAL nodes         {out['total_nodes']}")
            print(f"TOTAL relationships {out['total_relationships']}\n")

            print("nodes by label (a node may carry several labels, so these can sum high):")
            labels = [r["label"] for r in s.run("CALL db.labels() YIELD label RETURN label")]
            out["nodes_by_label"] = {}
            for label in sorted(labels):
                # Label is from db.labels(), not user input, so interpolation is safe here.
                n = s.run(f"MATCH (n:`{label}`) RETURN count(n) AS c").single()["c"]
                out["nodes_by_label"][label] = n
                print(f"  {label:<22} {n}")

            # A relationship type's NAME is not its meaning. What it connects is — so report the
            # endpoint labels, which is what tells you whether a type carries knowledge structure
            # between elements or joins a person to a thing.
            print("\nrelationships by type, with what they actually connect:")
            out["rels_by_type"] = {}
            out["rel_schema"] = {}
            for r in s.run("CALL db.relationshipTypes() YIELD relationshipType AS t RETURN t"):
                t = r["t"]
                n = s.run(f"MATCH ()-[x:`{t}`]->() RETURN count(x) AS c").single()["c"]
                out["rels_by_type"][t] = n
                shapes = s.run(
                    f"MATCH (a)-[x:`{t}`]->(b) "
                    "RETURN labels(a) AS sa, labels(b) AS sb, count(*) AS c "
                    "ORDER BY c DESC LIMIT 12").data()
                out["rel_schema"][t] = [
                    {"from": sorted(x["sa"]), "to": sorted(x["sb"]), "count": x["c"]}
                    for x in shapes]
                print(f"  {t:<20} {n:>5}")
                for x in shapes:
                    fr = "|".join(sorted(x["sa"])) or "(no label)"
                    to = "|".join(sorted(x["sb"])) or "(no label)"
                    print(f"      {fr:>28} -> {to:<28} {x['c']:>5}")

            print("\nelement nodes by type, public vs all:")
            pub = platform_graph.counts(include_private=False)
            allc = platform_graph.counts(include_private=True)
            out["element_counts_public"], out["element_counts_all"] = pub, allc
            for t in sorted(allc, key=lambda k: -allc[k]):
                priv = allc[t] - pub.get(t, 0)
                print(f"  {t:<12} public {pub.get(t,0):<5} all {allc[t]:<5}"
                      + (f"  ({priv} private)" if priv else ""))
            print(f"  {'TOTAL':<12} public {sum(pub.values()):<5} all {sum(allc.values())}")
    finally:
        driver.close()

    if args.compare_api:
        api = _api_counts()
        print(f"\nreconcile against the public REST API "
              f"(total-count={api['total_count_field']}, enumerated={api['sum']}):")
        pub = out["element_counts_public"]
        for t in sorted(set(api["per_type"]) | set(pub)):
            a, g = api["per_type"].get(t, 0), pub.get(t, 0)
            flag = "" if a == g else "   <-- MISMATCH"
            print(f"  {t:<12} api {a:<5} graph(public) {g:<5}{flag}")
        out["api"] = api

    dest = REPO / "outputs" / "graph_census.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
