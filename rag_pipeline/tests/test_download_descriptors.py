"""Every downloadable file a tool emits must carry its own name, and its own url.

The map UI builds its Downloads panel entirely client-side: `collectDownloads` in
`map-ui-prototype/src/agentClient.ts` walks the whole SSE payload and harvests ANY object that
has a `download_url` together with a `file_id` or a `filename`, treating the three as describing
one file. Nothing server-side assembles that list, so the shape a tool emits IS the contract.

`align_embedding_colors` broke it in both possible ways at once. Its per-region entry carried
the EMBEDDING PACKAGE's `file_id` next to the rendered IMAGE's `download_url`, and no filename
at all — so the panel showed `unnamed file (file_f2929f3dec7d)` three times, each link fetching
a picture while its id named a `.npz`. The files downloaded, which is why it read as cosmetic.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List


def harvest(value: Any) -> Iterator[Dict[str, Any]]:
    """The client's rule, in Python: anything with a url and an identifier is a download.

    Mirrors `collectDownloads` deliberately. A server-side test that checked some other rule
    would pass while the panel stayed wrong.
    """
    if isinstance(value, list):
        for item in value:
            yield from harvest(item)
        return
    if not isinstance(value, dict):
        return
    if value.get("download_url") and (value.get("file_id") or value.get("filename")):
        yield value
    for item in value.values():
        yield from harvest(item)


def assert_every_download_is_self_describing(payload: Any) -> List[Dict[str, Any]]:
    found = list(harvest(payload))
    for record in found:
        name = str(record.get("filename") or "").strip()
        assert name, f"download has no filename: {record}"
        # "download" is the placeholder collectDownloads substitutes for a missing name; the
        # panel renders it as "unnamed file (<id>)", which is the bug this file exists for.
        assert name.lower() != "download", f"download carries the placeholder name: {record}"
    return found


def _region_entry_as_shipped() -> Dict[str, Any]:
    """The shape align_embedding_colors emits per region, after the fix."""
    return {
        "file_id": "file_pkg2018",
        "filename": "drawn_region_2018_gse_2018-06_2018-09_vectors.npz",
        "download_url": "https://agent.i-guide.io/agent/files/file_pkg2018/download",
        "label": "2018 (Jun-Sep)",
        "bbox": [-87.33, 41.01, -87.30, 41.03],
        "grid": [243, 186],
        "image": {
            "file_id": "file_png2018",
            "filename": "2018_jun_sep_gse_shared_pca.png",
            "download_url": "https://agent.i-guide.io/agent/files/file_png2018/download",
        },
        "image_file_id": "file_png2018",
    }


def test_each_emitted_file_is_named():
    payload = {"ok": True, "regions": [_region_entry_as_shipped()]}
    found = assert_every_download_is_self_describing(payload)
    names = sorted(r["filename"] for r in found)
    assert names == ["2018_jun_sep_gse_shared_pca.png",
                     "drawn_region_2018_gse_2018-06_2018-09_vectors.npz"]


def test_an_id_is_never_paired_with_another_files_url():
    """The half of the bug that was not cosmetic: the entry downloaded something else.

    A file_id and a download_url in the same object have to name the same bytes, because the
    client presents them as one row: the label comes from one and the click from the other.
    """
    for record in harvest({"ok": True, "regions": [_region_entry_as_shipped()]}):
        assert str(record["file_id"]) in str(record["download_url"]), (
            f"file_id {record['file_id']} is not the file its download_url fetches: {record}")


def test_the_old_flat_shape_would_have_failed_this():
    """Guards the guard: the test must actually reject what shipped."""
    broken = {"ok": True, "regions": [{
        "file_id": "file_pkg2018",            # the .npz
        "label": "2018 (Jun-Sep)",
        "image_file_id": "file_png2018",
        "download_url": "https://agent.i-guide.io/agent/files/file_png2018/download",  # the .png
    }]}
    try:
        assert_every_download_is_self_describing(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("the unnamed-download check did not reject the shape that shipped")

    harvested = list(harvest(broken))
    assert harvested, "the client would still have shown this row"
    assert str(harvested[0]["file_id"]) not in str(harvested[0]["download_url"]), (
        "this fixture is meant to demonstrate the mismatched pair")
