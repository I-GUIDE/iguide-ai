"""How many platform elements can we actually fetch the source of?

This is the number M3 (corpus-scale ingest) lives or dies on. Extraction quality is
irrelevant for an element whose file cannot be retrieved, and the failure modes are
distinguishable and worth counting separately: a renamed repo, a private repo, a path that
moved, and an element that never declared a source are four different problems with four
different fixes.

HEAD-only by default so a full sweep costs no bandwidth; ``--fetch`` downloads for real and
reports sha256 so the same run doubles as a corpus snapshot.

Usage
-----
    python scripts/measure_source_fetchability.py --type notebook
    python scripts/measure_source_fetchability.py --type notebook --fetch --out outputs/nb.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv  # noqa: E402

for candidate in (REPO / ".env", Path("/Users/yfkang/i-guide-platform-flask-servers/.env")):
    if candidate.exists():
        load_dotenv(candidate)
        break

import requests  # noqa: E402

from extractors.sources import (SourceError, raw_url_from_blob,  # noqa: E402
                                resolve_and_fetch, resolve_github_url)

BACKEND = "https://backend.i-guide.io"


def list_elements(element_type: str, limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    page = 0
    while len(out) < limit:
        r = requests.get(f"{BACKEND}/api/elements",
                         params={"element-type": element_type, "from": page * 50, "size": 50},
                         timeout=60)
        r.raise_for_status()
        batch = (r.json() or {}).get("elements") or []
        if not batch:
            break
        out.extend(batch)
        page += 1
    return out[:limit]


def detail(element_id: str) -> Dict[str, Any]:
    r = requests.get(f"{BACKEND}/api/elements/{element_id}", timeout=60)
    r.raise_for_status()
    return r.json()


def probe(url: str) -> int:
    """HEAD, falling back to a ranged GET — raw.githubusercontent answers HEAD, but some
    hosts do not, and treating a 405 as 'missing' would overstate the failure rate."""
    try:
        r = requests.head(url, timeout=30, allow_redirects=True)
        if r.status_code in (403, 405, 501):
            r = requests.get(url, timeout=30, stream=True, headers={"Range": "bytes=0-64"})
        return r.status_code
    except Exception:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", default="notebook")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--fetch", action="store_true", help="download instead of probing")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    elements = list_elements(args.type, args.limit)
    print(f"{args.type}: {len(elements)} elements listed\n")

    rows: List[Dict[str, Any]] = []
    reasons: collections.Counter = collections.Counter()
    strategies: collections.Counter = collections.Counter()
    dest = Path(".source_probe")

    for i, el in enumerate(elements, 1):
        eid = el.get("id") or ""
        row: Dict[str, Any] = {"id": eid, "title": (el.get("title") or "")[:60]}
        try:
            meta = detail(eid)
        except Exception as exc:
            row.update(ok=False, reason="metadata_error", detail=str(exc)[:120])
            reasons["metadata_error"] += 1
            rows.append(row)
            continue

        row["has_repo"] = bool(meta.get("notebook-repo"))
        row["has_blob_url"] = bool(raw_url_from_blob(str(meta.get("notebook-url") or "")))

        if args.fetch:
            try:
                src = resolve_and_fetch(meta, dest, element_id=eid)
                row.update(ok=True, url=src.origin_url, sha256=src.sha256, bytes=src.bytes)
                reasons["ok"] += 1
            except SourceError as exc:
                row.update(ok=False, reason=exc.kind, detail=str(exc)[:140])
                reasons[exc.kind] += 1
        else:
            url = resolve_github_url(meta)
            if not url:
                row.update(ok=False, reason="unresolvable")
                reasons["unresolvable"] += 1
            else:
                strategies["blob_url" if row["has_blob_url"] else "repo_plus_path"] += 1
                code = probe(url)
                row.update(url=url, status=code, ok=200 <= code < 300)
                reasons["ok" if row["ok"] else (f"http_{code}" if code else "network")] += 1
        rows.append(row)
        if i % 25 == 0:
            print(f"  … {i}/{len(elements)}  ok={reasons['ok']}")

    ok = reasons["ok"]
    total = len(rows)
    print(f"\n{'='*58}\nFETCHABLE: {ok}/{total}  ({100*ok/max(1,total):.1f}%)\n{'='*58}")
    for reason, n in reasons.most_common():
        if reason != "ok":
            print(f"   {n:>4}  {reason}")
    if strategies:
        print("\n  resolution strategy used:")
        for s, n in strategies.most_common():
            print(f"   {n:>4}  {s}")

    failures = [r for r in rows if not r.get("ok")]
    if failures:
        print(f"\n  first failures:")
        for r in failures[:8]:
            reason = r.get("reason") or f"http_{r.get('status')}"
            print(f"   {r['id'][:8]}  {reason:<14} {r['title'][:42]}")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"type": args.type, "total": total, "ok": ok,
                                 "reasons": dict(reasons), "rows": rows}, indent=2))
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
