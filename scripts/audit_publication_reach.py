#!/usr/bin/env python3
"""How many publications are actually READABLE — not how many fetched with a 200.

    python scripts/audit_publication_reach.py
    python scripts/audit_publication_reach.py --json outputs/publication_reach.json

WHY THIS IS SEPARATE FROM THE FETCH. ``build_method_library.py`` counts a publication as
``fetched`` when the HTTP request succeeds, and that number was quoted as "94 reachable
open-access PDFs". It is not the same quantity. A publisher can answer 200 with a bot-check page,
a paywall interstitial, a cookie wall, an HTML landing page named ``.pdf``, or zero bytes — all of
which the fetch counts as a success and none of which is a document.

So the reach number is derived from ``publication_extractor.read_document``, which routes on the
file's first bytes and returns a REASON when it yields nothing. Every outcome below is one of
those reasons, aggregated. No LLM runs here: whether a document is readable is a fact about the
bytes, and mixing in "did the model find a method" would conflate two different questions.

A bot check is reported and never satisfied.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Outcome buckets, ordered from best to worst so the report reads down.
READABLE = "readable document"
WALL = "publisher wall (bot check, paywall, cookie, error)"
WRONG_TYPE = "fetched, but not a document"
EMPTY = "empty file, the fetch produced no bytes"
NO_TEXT = "parsed, no text layer (needs OCR)"
BROKEN = "unparseable by the reader"

_BUCKETS = (READABLE, WALL, WRONG_TYPE, EMPTY, NO_TEXT, BROKEN)


def classify(reason: str, chars: int) -> str:
    """One outcome per document, from the reader's own reason string."""
    if not reason and chars > 0:
        return READABLE
    lowered = (reason or "").lower()
    if "bot-check" in lowered or "paywall" in lowered or "cookie consent" in lowered \
            or "error page" in lowered:
        return WALL
    if "empty (0 bytes)" in lowered:
        return EMPTY
    if "no text layer" in lowered:
        return NO_TEXT
    if "zip archive" in lowered or "no reader for extension" in lowered \
            or "carried no extractable text" in lowered:
        return WRONG_TYPE
    if "could not parse" in lowered or "not installed" in lowered \
            or "could not be opened" in lowered:
        return BROKEN
    # A reason that is only a note (content is HTML although named .pdf) with text is readable.
    return READABLE if chars > 0 else WRONG_TYPE


# A file written within this many seconds may still be downloading.
IN_FLIGHT_SECONDS = 30


def audit(cache: Path) -> dict:
    """Classify every fetched document, and flag any that may still be downloading.

    The flag exists because its absence fooled me: auditing mid-fetch reported one document as
    "unparseable by the reader" -- pypdf's "EOF marker not found" on a PDF that was half written.
    Two minutes later the same file read cleanly. A partially-written download is indistinguishable
    from a corrupt one at read time, so the only honest options are to wait, or to say which files
    were in flight when the count was taken.
    """
    import time

    from extractors.publication_extractor import read_document, sniff_kind

    now = time.time()
    rows, in_flight = [], []
    for path in sorted(cache.glob("*oa.pdf")) + sorted(cache.glob("*oa.html")):
        stat = path.stat()
        text, reason = read_document(str(path))
        chars = len(text)
        fresh = (now - stat.st_mtime) < IN_FLIGHT_SECONDS
        if fresh:
            in_flight.append(path.name)
        rows.append({
            "element": path.name.split("__", 1)[0],
            "file": path.name,
            "bytes": stat.st_size,
            "sniffed": sniff_kind(str(path)),
            "chars": chars,
            "reason": reason,
            "outcome": classify(reason, chars),
            "possibly_in_flight": fresh,
        })
    return {"cache": str(cache), "documents": len(rows), "rows": rows,
            "in_flight": in_flight}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(REPO / ".corpus_cache"))
    ap.add_argument("--fetch-report", default=str(REPO / "outputs" /
                                                  "extract_publication_rederive.json"),
                    help="the driver's run, for the elements that never produced a file")
    ap.add_argument("--json")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(REPO / ".env", override=False)

    result = audit(Path(args.cache))
    rows = result["rows"]
    if not rows:
        print(f"no fetched documents under {args.cache}")
        return 2

    counts = collections.Counter(r["outcome"] for r in rows)
    print(f"fetched documents on disk: {len(rows)}")
    if result.get("in_flight"):
        names = result["in_flight"]
        print(f"\n!! {len(names)} file(s) were written in the last {IN_FLIGHT_SECONDS}s and may "
              f"still be downloading: {', '.join(names[:4])}"
              f"{' ...' if len(names) > 4 else ''}\n"
              f"   A half-written PDF reads as truncated. Re-run once the fetch has finished.")
    print()
    print(f"{'outcome':<50}{'n':>5}   share of fetched")
    print("-" * 76)
    for bucket in _BUCKETS:
        n = counts.get(bucket, 0)
        if n:
            print(f"{bucket:<50}{n:>5}   {100 * n / len(rows):>5.1f}%")
    readable = counts.get(READABLE, 0)
    print("-" * 76)
    print(f"{'READABLE':<50}{readable:>5}   {100 * readable / len(rows):>5.1f}%")

    # The elements that never produced a file at all, from the driver's own record.
    report = Path(args.fetch_report)
    if report.is_file():
        try:
            data = json.loads(report.read_text())
        except ValueError:
            data = {}
        elements = data.get("elements") or []
        if elements:
            reasons = collections.Counter()
            for row in elements:
                err = str(row.get("error") or "")
                if not err:
                    continue
                key = re.sub(r"https?://\S+", "<url>", err)[:64]
                reasons[key] += 1
            print(f"\nelements attempted: {len(elements)}")
            print(f"  produced a file : {len(elements) - sum(reasons.values())}")
            print(f"  never fetched   : {sum(reasons.values())}")
            for key, n in reasons.most_common(10):
                print(f"      {n:>4}  {key}")
            total = len(elements)
            print(f"\nREACH over the whole type: {readable} of {total} "
                  f"({100 * readable / total:.1f}%) yield a readable document")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({**result, "counts": dict(counts)}, indent=2) + "\n")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
