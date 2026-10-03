"""What would break if token mode went strict: how many stored files nobody owns.

There is no honest BACKFILL for the legacy records. Ownership did not exist when they were
written, and the conversation stamp they do carry was never mapped to a user, so no rule can
attribute them after the fact. Inventing an owner would be worse than leaving them unowned.

What is possible is knowing the size of the problem before flipping AGENT_TOKEN_STRICT=1, which
is what this prints. The intended sequence:

    1. turn token mode on with AGENT_TOKEN_STRICT=0 — identity flows, new files get stamped
    2. run this until `unowned` stops growing and the recent files are all owned
    3. flip strict; unowned records stay readable to the AGENT (find_files, server-side reuse)
       and stop being downloadable over HTTP, which is the exposure being closed

Usage:  python scripts/file_ownership_report.py [--days 30] [--list-unowned 20]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent_runtime.file_store import _metadata_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=30.0,
                    help="treat files newer than this as 'recent' (default 30)")
    ap.add_argument("--list-unowned", type=int, default=0,
                    help="print this many of the most recent unowned filenames")
    args = ap.parse_args()

    cutoff = time.time() - args.days * 86400
    total = owned = unowned = recent_total = recent_unowned = 0
    owners: Counter = Counter()
    recent_orphans = []

    for meta in _metadata_dir().glob("*.json"):
        try:
            record = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        total += 1
        mtime = meta.stat().st_mtime
        is_recent = mtime >= cutoff
        recent_total += int(is_recent)
        owner = (record.get("owner_id") or "").strip() if record.get("owner_id") else ""
        if owner:
            owned += 1
            owners[owner] += 1
        else:
            unowned += 1
            recent_unowned += int(is_recent)
            if is_recent:
                recent_orphans.append((mtime, record.get("filename") or record.get("file_id")))

    print(f"stored files      {total}")
    print(f"  owned           {owned}  ({len(owners)} distinct users)")
    print(f"  unowned         {unowned}   <- these stop being downloadable once strict")
    print(f"last {args.days:g} days      {recent_total}")
    print(f"  unowned         {recent_unowned}   <- must reach 0 before flipping strict")
    if recent_unowned:
        print("\n  NOT READY: files are still being written without an owner. Either token mode "
              "is off, or a write path is bypassing the identity binding.")
    elif total:
        print("\n  Ready: nothing recent is unowned.")
    if args.list_unowned and recent_orphans:
        print(f"\nmost recent unowned ({min(args.list_unowned, len(recent_orphans))}):")
        for mtime, name in sorted(recent_orphans, reverse=True)[:args.list_unowned]:
            print(f"  {time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
