"""Create the agent-KB indices on the cluster, with the kNN dimension VERIFIED.

Idempotent: an index that already exists is left alone and reported, so this is safe to run
before every backfill.

Why the dimension is probed rather than trusted. ``AGENT_KB_EMBED_DIM`` defaults to 384 and
``ensure_index`` writes that into the ``knn_vector`` mapping. If the running embedder returns
a different width, nothing fails at index-creation time and nothing fails at write time —
OpenSearch rejects the *query* vector later, and the agent sees "no semantic results", which
is indistinguishable from "nothing matched". So the width is measured from the live service
first and the run refuses if it disagrees with the configured value.

Usage
-----
    python scripts/create_agent_indices.py                 # create what is missing
    python scripts/create_agent_indices.py --dry-run       # report only
    python scripts/create_agent_indices.py --recreate      # DELETE and rebuild (destructive)
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv  # noqa: E402

for candidate in (REPO / ".env", Path("/Users/yfkang/i-guide-platform-flask-servers/.env")):
    if candidate.exists():
        load_dotenv(candidate)
        break

from extractors.emitters.opensearch_emitter import _embed_dim, ensure_index  # noqa: E402
from extractors.indices import all_agent_indices, is_agent_index  # noqa: E402


def probe_embedding_dim(timeout: int = 30) -> Optional[int]:
    """Width of a real embedding from the configured service, or None if unreachable."""
    import requests

    url = (os.getenv("FLASK_EMBEDDING_URL") or "").rstrip("/")
    if not url:
        print("  FLASK_EMBEDDING_URL is not set")
        return None
    try:
        r = requests.post(f"{url}/get_embedding",
                          json={"text": "dimension probe"}, timeout=timeout)
        r.raise_for_status()
        vec = r.json().get("embedding")
        return len(vec) if isinstance(vec, list) else None
    except Exception as exc:
        print(f"  embedder unreachable at {url}: {type(exc).__name__}: {exc}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--recreate", action="store_true",
                    help="DELETE existing agent indices first (destructive)")
    ap.add_argument("--allow-dim-mismatch", action="store_true",
                    help="create anyway when the probe disagrees with AGENT_KB_EMBED_DIM")
    args = ap.parse_args()

    configured = _embed_dim()
    print(f"embedding endpoint : {os.getenv('FLASK_EMBEDDING_URL')}")
    print(f"AGENT_KB_EMBED_DIM : {configured}")
    probed = probe_embedding_dim()
    print(f"probed dimension   : {probed}")

    if probed is None:
        print("\nRefusing: the embedder must be reachable to verify the kNN dimension. "
              "An index built on a guessed width fails silently at QUERY time.")
        return 2
    if probed != configured and not args.allow_dim_mismatch:
        print(f"\nRefusing: the live embedder returns {probed}-d vectors but "
              f"AGENT_KB_EMBED_DIM is {configured}. Set AGENT_KB_EMBED_DIM={probed} "
              f"(or pass --allow-dim-mismatch if you know why they differ).")
        return 2

    from rag_pipeline.search.keyword import _os_client
    client = _os_client()

    indices: List[str] = all_agent_indices()
    general = os.getenv("OPENSEARCH_INDEX")
    for name in indices:
        # A misconfigured prefix must never let this touch the platform's own index.
        assert is_agent_index(name), f"{name} is not an agent index"
        assert name != general, f"{name} collides with OPENSEARCH_INDEX"

    print(f"\n{'index':<40}{'before':<12}{'action'}")
    print("-" * 70)
    created = kept = 0
    for name in indices:
        exists = client.indices.exists(index=name)
        count = client.count(index=name)["count"] if exists else 0
        before = f"{'exists' if exists else 'missing'}({count})"
        if exists and args.recreate:
            if args.dry_run:
                action = "would DELETE + recreate"
            else:
                client.indices.delete(index=name)
                ensure_index(client, name)
                action = "DELETED + recreated"
                created += 1
        elif exists:
            action = "kept (idempotent)"
            kept += 1
        elif args.dry_run:
            action = "would create"
        else:
            ensure_index(client, name)
            action = "created"
            created += 1
        print(f"{name:<40}{before:<12}{action}")

    print(f"\ncreated {created}, kept {kept}")
    if not args.dry_run:
        ok = 0
        for name in indices:
            mapping = client.indices.get_mapping(index=name)
            props = mapping[name]["mappings"]["properties"]
            dim = props.get("contents-embedding", {}).get("dimension")
            if dim == probed:
                ok += 1
            else:
                print(f"  WRONG DIMENSION {name}: mapping says {dim}, embedder gives {probed}")
        print(f"kNN dimension verified on {ok}/{len(indices)} indices at {probed}-d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
