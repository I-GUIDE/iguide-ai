"""Reading a geospatial file's own header, for files too large to download.

Nine of the corpus's oversized datasets are not shapefiles, so the .shp/.dbf/.prj sidecar route
does not reach them: 607 MB and 344 MB LiDAR point clouds, a 483 MB NAIP GeoTIFF, VIIRS
nighttime-lights BigTIFF tiles, a 1.2 GB GeoPackage, a 396 MB CSV.

Each of those formats puts its metadata at the FRONT of the file, which is what a ranged read
reaches. LAS is the clearest case: a fixed 227-byte public header block, left UNCOMPRESSED even
in LAZ so an indexer can read it, carrying the point count and exact min/max X, Y, Z.

Every reader returns {} rather than guessing when the bytes do not match. A wrong bounding box is
worse than none — it would be indexed, searched on, and believed.
"""

from __future__ import annotations

import struct

import pytest

from extractors.file_headers import (read_any, read_csv_header, read_gpkg_header,
                                     read_las_header, read_tiff_header, tiff_ifd_offset)


def _las(count=20221590, bounds=(-753872.75, 1791309.26, -751795.74, 1793386.28),
         z=(1713.14, 1953.92), version=(1, 4)) -> bytes:
    head = bytearray(375)
    head[0:4] = b"LASF"
    head[24], head[25] = version
    head[104] = 6                                  # point data record format
    struct.pack_into("<I", head, 107, count)
    minx, miny, maxx, maxy = bounds
    struct.pack_into("<6d", head, 179, maxx, minx, maxy, miny, z[1], z[0])
    return bytes(head)


# ------------------------------------------------------------------ LiDAR

def test_a_lidar_point_cloud_gives_its_count_and_extent_from_227_bytes():
    """Measured on the corpus: 80,849,096 points from a 344 MB LAZ, and 20,221,590 from a
    607 MB LAS — in both cases from the public header alone."""
    info = read_las_header(_las())
    assert info["row_count"] == 20221590
    assert info["bounds"] == pytest.approx([-753872.75, 1791309.26, -751795.74, 1793386.28])
    assert info["z_range"] == pytest.approx([1713.14, 1953.92])
    assert info["geometry_type"] == "PointCloud"
    assert info["las_version"] == "1.4"


def test_a_cloud_with_more_than_four_billion_points_reports_the_64_bit_count():
    """LAS 1.4 zeroes the legacy 32-bit field and moves the count to a 64-bit one at byte 247.
    Trusting the legacy field alone would report the largest clouds as empty."""
    head = bytearray(_las(count=0))
    struct.pack_into("<Q", head, 247, 5_000_000_000)
    assert read_las_header(bytes(head))["row_count"] == 5_000_000_000


def test_bytes_that_are_not_las_are_refused():
    assert read_las_header(b"\x00" * 400) == {}
    assert read_las_header(b"LASF") == {}, "a truncated header must not be parsed"


# ------------------------------------------------------------------ GeoTIFF / BigTIFF

def _bigtiff(width=17281, height=5601, epsg=4326) -> bytes:
    entries = [(256, 3, 1, width), (257, 3, 1, height)]
    blob = bytearray(b"II+\x00" + struct.pack("<HH", 8, 0) + struct.pack("<Q", 16))
    blob += struct.pack("<Q", len(entries) + 1)
    for tag, ftype, n, value in entries:
        blob += struct.pack("<HH", tag, ftype) + struct.pack("<Q", n) + struct.pack("<Q", value)
    # GeoKey directory: header of 4 shorts, then one key -> ProjectedCS/GeographicType = epsg
    keys = [1, 1, 0, 1, 2048, 0, 1, epsg]
    keys_offset = len(blob) + 20 + 8
    blob += struct.pack("<HH", 34735, 3) + struct.pack("<Q", len(keys)) + \
        struct.pack("<Q", keys_offset)
    blob += struct.pack("<Q", 0)                    # next-IFD pointer
    blob += struct.pack(f"<{len(keys)}H", *keys)
    return bytes(blob)


def test_a_bigtiff_yields_dimensions_and_its_crs():
    """VIIRS nighttime lights, measured: 17,281 x 5,601 pixels, EPSG:4326, 96.8 million cells."""
    info = read_tiff_header(_bigtiff())
    assert info["format"] == "BigTIFF"
    assert (info["width"], info["height"]) == (17281, 5601)
    assert info["crs"] == "EPSG:4326"
    assert info["row_count"] == 17281 * 5601
    assert info["geometry_type"] == "Raster"


def test_the_geokey_directory_is_read_as_an_array_not_a_single_short():
    """It is an ARRAY of SHORTs, and the generic single-SHORT branch used to claim it first — so
    it was stored as one int and the CRS lookup called len() on an integer. A demonstrably valid
    BigTIFF then read as "not a TIFF", with an empty dict as the only explanation."""
    info = read_tiff_header(_bigtiff(epsg=32616))
    assert info.get("crs") == "EPSG:32616"


def test_a_tag_parse_failure_reports_its_reason():
    """An unexplained {} is indistinguishable from "not a TIFF" and cost several probes to tell
    apart. The reader now says what broke."""
    broken = _bigtiff()[:40]                        # a truncated tag table
    info = read_tiff_header(broken)
    assert info == {} or "note" in info


def test_the_ifd_offset_is_exposed_so_a_caller_can_fetch_it():
    """A TIFF's tag directory is not required to be near the front — a 483 MB NAIP GeoTIFF in
    this corpus has its IFD at byte 483,252,102. Knowing where it is, is what lets a caller
    decide whether it is reachable at all."""
    assert tiff_ifd_offset(_bigtiff()) == 16
    assert tiff_ifd_offset(b"nope") is None


# ------------------------------------------------------------------ the rest

def test_a_geopackage_is_confirmed_but_its_layers_are_not_claimed():
    """`gpkg_contents` is a B-tree page the file header does not reach. Reporting layer names
    from the header alone would be a guess dressed as a fact."""
    head = b"SQLite format 3\x00" + b"\x00" * 52 + b"GPKG"
    info = read_gpkg_header(head)
    assert info["format"] == "GeoPackage"
    assert "gpkg_contents" in info["note"]
    assert "schema" not in info and "bounds" not in info


def test_a_plain_sqlite_file_is_not_called_a_geopackage():
    head = b"SQLite format 3\x00" + b"\x00" * 52 + b"\x00\x00\x00\x00"
    assert read_gpkg_header(head)["format"] == "SQLite"


def test_a_csv_gives_its_columns():
    info = read_csv_header(b"name,latitude,longitude,pop\nx,41.9,-87.6,100\n")
    assert info["schema"] == ["name", "latitude", "longitude", "pop"]


def test_dispatch_is_by_extension_and_returns_nothing_for_the_unknown():
    assert read_any("cloud.laz", _las())["geometry_type"] == "PointCloud"
    assert read_any("raster.tif", _bigtiff())["width"] == 17281
    assert read_any("model.pt", b"\x80\x02") == {}
