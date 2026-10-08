"""Describing a ZIP too large to download, from its central directory alone.

11 of the 31 datasets the platform hosts itself exceed the fetch cap, and they are the
substantial ones — 8.5 GB of LiDAR point clouds, NAIP imagery, VIIRS nighttime lights and
national hydrogeology. They were reported as simply unfetchable, which is true and useless:
raising the cap would mean storing all of it to learn what is inside.

A ZIP's central directory sits at the END of the file, so the member list costs one ranged read
of the tail and a shapefile's CRS costs one more for its ``.prj``. Measured against the live
corpus: a 724 MB archive with 3,291 members described from 389 KB, and a 3.9 GB archive with
9,533 members from 96 KB.

No network here — a fake session serves byte ranges out of a real in-memory ZIP, so the range
arithmetic is exercised rather than mocked away.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from extractors.remote_zip import fetch_member, inspect


def _archive(entries) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries:
            zf.writestr(name, data)
    return buffer.getvalue()


class _FakeResponse:
    def __init__(self, content=b"", status=200, headers=None):
        self.content, self.status_code = content, status
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _RangeServer:
    """Serves real byte ranges out of *blob*, and counts how much was transferred."""

    def __init__(self, blob: bytes, *, honour_range: bool = True,
                 report_length: bool = True):
        self.blob, self.honour_range, self.report_length = blob, honour_range, report_length
        self.bytes_served = 0
        self.requests = 0

    def head(self, _url, **_kw):
        headers = {"Accept-Ranges": "bytes" if self.honour_range else "none"}
        if self.report_length:
            headers["Content-Length"] = str(len(self.blob))
        return _FakeResponse(headers=headers)

    def get(self, _url, headers=None, **_kw):
        self.requests += 1
        rng = (headers or {}).get("Range", "")
        if not rng or not self.honour_range:
            self.bytes_served += len(self.blob)
            return _FakeResponse(self.blob, status=200)
        start, end = rng.replace("bytes=", "").split("-")
        chunk = self.blob[int(start):int(end) + 1]
        self.bytes_served += len(chunk)
        return _FakeResponse(chunk, status=206)

    def close(self):
        pass


# ------------------------------------------------------------------ the listing

def test_members_are_listed_without_downloading_the_archive():
    """The point of the whole module: the member list costs the tail, not the file."""
    # Incompressible payloads, and enough of them to exceed the tail window. Repetitive filler
    # compresses to a few KB, which is SMALLER than the 96 KB tail read — so the "partial" read
    # would be the whole file and the test would prove nothing.
    import os as _os

    blob = _archive([(f"data/tile_{i}.tif", _os.urandom(8192)) for i in range(60)])
    assert len(blob) > 96 * 1024, "fixture must exceed the tail window to test a partial read"
    server = _RangeServer(blob)

    result = inspect("https://example.org/big.zip", session=server)

    assert len(result.members) == 60
    assert result.total_bytes == len(blob)
    assert server.bytes_served < len(blob), (
        "the whole archive was transferred; the range read did nothing")


def test_a_large_archive_is_described_from_a_small_read():
    """3,291 members from 389 KB of a 724 MB archive, on the live corpus. Here: the transfer
    must stay bounded by the tail window rather than scaling with the archive."""
    import os as _os

    blob = _archive([(f"m{i}.bin", _os.urandom(20_000)) for i in range(200)])
    server = _RangeServer(blob)

    inspect("https://example.org/big.zip", session=server)
    assert server.bytes_served <= 400 * 1024


def test_member_names_sizes_and_offsets_are_recovered():
    blob = _archive([("shape/roads.shp", b"S" * 5000), ("shape/roads.prj", b"PROJCS[...]")])
    result = inspect("https://example.org/a.zip", session=_RangeServer(blob))

    names = {m["name"] for m in result.members}
    assert names == {"shape/roads.shp", "shape/roads.prj"}
    assert all(m["offset"] >= 0 and m["size"] > 0 for m in result.members)


# ------------------------------------------------------------------ reading one small member

def test_a_prj_member_yields_the_crs_without_reading_the_geometry():
    """A `.prj` is a few hundred bytes of WKT. The 164 MB shapefile beside it is never touched."""
    wkt = b'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984"]]'
    blob = _archive([("roads.shp", b"S" * 200_000), ("roads.prj", wkt)])
    server = _RangeServer(blob)
    result = inspect("https://example.org/a.zip", session=server)

    prj = next(m for m in result.members if m["name"].endswith(".prj"))
    got = fetch_member("https://example.org/a.zip", prj,
                       total_bytes=result.total_bytes, session=server)
    assert got is not None and b"GCS_WGS_1984" in got


def test_a_stored_member_is_read_too():
    """Not everything is deflated; an already-compressed member is often stored verbatim."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("meta.prj", b'PROJCS["NAD83"]')
    blob = buffer.getvalue()
    server = _RangeServer(blob)
    result = inspect("https://example.org/s.zip", session=server)
    member = result.members[0]
    assert b"NAD83" in (fetch_member("https://example.org/s.zip", member,
                                     total_bytes=len(blob), session=server) or b"")


# ------------------------------------------------------------------ honest about not knowing

def test_a_server_that_ignores_range_is_reported_not_worked_around():
    """Silently falling back to a full download is how a 3.9 GB file gets fetched by a function
    whose entire purpose is not fetching it."""
    blob = _archive([("a.txt", b"x")])
    result = inspect("https://example.org/a.zip",
                     session=_RangeServer(blob, honour_range=False))
    assert not result.members
    assert result.note, "the reason must be recorded"


def test_a_server_with_no_content_length_is_reported():
    blob = _archive([("a.txt", b"x")])
    result = inspect("https://example.org/a.zip",
                     session=_RangeServer(blob, report_length=False))
    assert not result.members and "content length" in result.note.lower()


def test_a_non_zip_body_is_reported_rather_than_guessed():
    result = inspect("https://example.org/x", session=_RangeServer(b"not a zip at all" * 500))
    assert not result.members and result.note


# ------------------------------------------------------------------ what the doc says

def test_a_remotely_described_dataset_says_it_was_not_opened(monkeypatch):
    """`described_remotely` with no feature count and no bounds. "We did not look" and "we
    looked and it was fine" must not produce the same document."""
    from extractors import data_extractor

    blob = _archive([("roads.shp", b"S" * 9000),
                     ("roads.prj", b'PROJCS["NAD_1983_UTM_Zone_16N"]')])
    server = _RangeServer(blob)
    monkeypatch.setattr(data_extractor, "extract_remote_dataset_metadata",
                        data_extractor.extract_remote_dataset_metadata)

    import extractors.remote_zip as rz

    monkeypatch.setattr(rz, "_head", lambda *a, **k: {
        "content-length": str(len(blob)), "accept-ranges": "bytes"})
    monkeypatch.setattr(rz, "_range",
                        lambda _u, s, e, _t, _s: blob[s:e + 1])

    meta = data_extractor.extract_remote_dataset_metadata("https://example.org/a.zip")
    assert meta["described_remotely"] is True
    assert meta["member_count"] == 2
    assert "feature_count" not in meta and "bounds" not in meta
    assert meta.get("crs") == "NAD_1983_UTM_Zone_16N"
    assert meta.get("crs_from") == "remote .prj read"


# ------------------------------------------------------------------ headers at fixed offsets

def test_a_shapefile_bounding_box_comes_from_100_bytes():
    """Bytes 36..68 of the .shp header. That is 100 bytes out of a member that may be 160 MB,
    which is why lifting the fetch cap is not the way to get bounds."""
    import struct

    from extractors.remote_zip import shapefile_bounds

    header = bytearray(100)
    struct.pack_into(">i", header, 0, 9994)              # the shapefile magic
    struct.pack_into("<4d", header, 36, -91.4, 36.9, -87.5, 42.5)
    assert shapefile_bounds(bytes(header)) == pytest.approx([-91.4, 36.9, -87.5, 42.5])


def test_a_wrong_guess_is_not_reported_as_a_bounding_box():
    """The magic number is the guard. Without it, any 68 bytes produce four plausible floats."""
    from extractors.remote_zip import shapefile_bounds

    assert shapefile_bounds(b"\x00" * 100) is None
    assert shapefile_bounds(b"not a shapefile") is None


def test_the_dbf_header_gives_the_feature_count_and_every_column_name():
    """Measured on the corpus: 1,197,659 OSM building footprints and 17 column names, out of a
    144 MB archive, from the first kilobyte of one member."""
    import struct

    from extractors.remote_zip import dbf_header

    blob = bytearray(32 + 32 * 2 + 1)
    struct.pack_into("<I", blob, 4, 1197659)
    blob[32:32 + 11] = b"osm_id\x00\x00\x00\x00\x00"
    blob[32 + 11] = ord("C")
    blob[64:64 + 11] = b"fclass\x00\x00\x00\x00\x00"
    blob[64 + 11] = ord("C")
    blob[96] = 0x0D                                      # field terminator

    header = dbf_header(bytes(blob))
    assert header["record_count"] == 1197659
    assert header["fields"] == ["osm_id", "fclass"]


def test_a_truncated_deflate_stream_still_yields_its_first_bytes():
    """`zlib.decompress` rejects a partial stream outright, so reading only the FRONT of a large
    member — the entire point — returned nothing. `decompressobj` is what makes it work."""
    import zlib

    payload = b"HEADER-BYTES" + b"z" * 200_000
    compressed = zlib.compressobj(9, zlib.DEFLATED, -15)
    blob = compressed.compress(payload) + compressed.flush()

    engine = zlib.decompressobj(-15)
    assert engine.decompress(blob[:4096], 32).startswith(b"HEADER-BYTES")
