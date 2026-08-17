"""Read a geospatial file's own header to learn what it holds, without reading the file.

Nine of the corpus's oversized datasets are not shapefiles, so the ``.shp``/``.dbf``/``.prj``
sidecar route does not reach them. Their primary members are:

    .tif   2   GeoTIFF     — 483 MB NAIP imagery, 62 MB VIIRS nighttime lights tiles
    .las   1   LiDAR       — 607 MB Colorado Springs point cloud
    .laz   1   LiDAR       — 344 MB UIUC campus point cloud
    .gpkg  1   GeoPackage  — 1.2 GB Brazil mobility analysis
    .csv   1               — 396 MB MapBiomas source table

Every one of those formats puts its metadata in a header at the FRONT of the file, which is why
a ranged read reaches it:

* **LAS/LAZ** — a fixed 227-byte public header block carrying the point count and exact
  min/max X, Y, Z. LAZ leaves it uncompressed precisely so a reader can index without
  decompressing, which is what makes a 607 MB point cloud describable in 227 bytes.
* **GeoTIFF** — the IFD holds width, height and the GeoKey directory; the CRS is a GeoKey.
* **GeoPackage** — a SQLite file, and ``gpkg_contents`` carries each layer's name, CRS and bounds.
* **CSV** — the first line is the schema.

Each reader returns ``{}`` rather than guessing when the bytes do not match the format. A wrong
bounding box is worse than none: it would be indexed, searched on, and believed.
"""

from __future__ import annotations

import struct
from typing import Any, Dict, List, Optional

# LAS public header block. Offsets are from the LAS 1.2-1.4 spec and are stable across versions
# for the fields used here.
_LAS_MAGIC = b"LASF"
_LAS_MIN_HEADER = 227


def read_las_header(head: bytes) -> Dict[str, Any]:
    """Point count and bounds from a LAS/LAZ public header block.

    LAZ keeps this block uncompressed, so a 607 MB compressed point cloud gives up its extent
    for 227 bytes. `point_count` falls back to the 1.4 64-bit field when the legacy 32-bit one is
    zero, which is how a cloud with more than 4.29 billion points reports itself.
    """
    if len(head) < _LAS_MIN_HEADER or not head.startswith(_LAS_MAGIC):
        return {}
    try:
        version = f"{head[24]}.{head[25]}"
        point_format = head[104] & 0b0111_1111
        legacy_count = struct.unpack("<I", head[107:111])[0]
        # 179..227 is max/min X, Y, Z as doubles, in that interleaved order.
        maxx, minx, maxy, miny, maxz, minz = struct.unpack("<6d", head[179:227])
        out: Dict[str, Any] = {
            "format": "LAS/LAZ",
            "las_version": version,
            "point_format": point_format,
            "row_count": int(legacy_count),
            "bounds": [float(minx), float(miny), float(maxx), float(maxy)],
            "z_range": [float(minz), float(maxz)],
            "geometry_type": "PointCloud",
        }
        if legacy_count == 0 and len(head) >= 255:
            # LAS 1.4 moved the count to a 64-bit field at 247 when the legacy field overflows.
            out["row_count"] = int(struct.unpack("<Q", head[247:255])[0])
        return out
    except Exception:
        return {}


# TIFF/GeoTIFF. Tag ids from the TIFF 6.0 spec plus the GeoTIFF extension.
_TIFF_TAGS = {256: "width", 257: "height", 258: "bits_per_sample", 277: "band_count",
              33550: "pixel_scale", 33922: "tie_point", 34735: "geo_key_directory",
              34737: "geo_ascii_params"}
_TIFF_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 11: 4, 12: 8, 16: 8}


def tiff_ifd_offset(head: bytes) -> Optional[int]:
    """Where a TIFF's first IFD starts, so a caller can fetch exactly that range.

    A TIFF's tag directory is NOT required to be near the front — writers commonly put it after
    the image data, so a 483 MB GeoTIFF's IFD can be 483 MB in. Reading the first 8 bytes tells
    you where to look; that is the whole reason this is separate from `read_tiff_header`.
    """
    if len(head) < 8 or head[:2] not in (b"II", b"MM"):
        return None
    endian = "<" if head[:2] == b"II" else ">"
    try:
        magic = struct.unpack(endian + "H", head[2:4])[0]
        if magic == 42:
            return int(struct.unpack(endian + "I", head[4:8])[0])
        if magic == 43 and len(head) >= 16:
            # BigTIFF: 8-byte offsets, and the first IFD offset is at byte 8.
            return int(struct.unpack(endian + "Q", head[8:16])[0])
    except Exception:
        return None
    return None


def read_tiff_header(head: bytes, *, ifd_bytes: Optional[bytes] = None,
                     ifd_at: int = 0) -> Dict[str, Any]:
    """Raster dimensions and, when present, the CRS from a GeoTIFF's first IFD.

    Only tags whose values fit inline (<= 4 bytes) or land inside the bytes provided are read —
    a tag pointing past the window is skipped rather than guessed at, because the alternative is
    reporting a pixel size read from unrelated bytes.
    """
    if len(head) < 8:
        return {}
    if head[:2] == b"II":
        endian = "<"
    elif head[:2] == b"MM":
        endian = ">"
    else:
        return {}
    try:
        magic = struct.unpack(endian + "H", head[2:4])[0]
        if magic not in (42, 43):
            return {}
        big = magic == 43
        # The IFD may live anywhere in the file. When the caller has fetched it separately, read
        # from that buffer with offsets rebased; otherwise fall back to the head buffer.
        if ifd_bytes is not None:
            body, base = ifd_bytes, ifd_at
            ifd_offset = 0
        else:
            body, base = head, 0
            offset = tiff_ifd_offset(head)
            if offset is None:
                return {}
            ifd_offset = offset
        if ifd_offset + (8 if big else 2) > len(body):
            # Honest partial answer: the format is known, the tags are not reachable from what
            # was read. A caller can use `tiff_ifd_offset` to fetch them.
            return {"format": "BigTIFF" if big else "GeoTIFF/TIFF",
                    "note": "tag directory outside the bytes read"}
        if big:
            count = int(struct.unpack(endian + "Q", body[ifd_offset:ifd_offset + 8])[0])
            entry_size, entry_start = 20, ifd_offset + 8
        else:
            count = struct.unpack(endian + "H", body[ifd_offset:ifd_offset + 2])[0]
            entry_size, entry_start = 12, ifd_offset + 2
        head = body                                  # tag values are read from the same buffer
        out: Dict[str, Any] = {"format": "BigTIFF" if big else "GeoTIFF/TIFF"}
        found: Dict[str, Any] = {}
        for i in range(min(count, 256)):
            entry = entry_start + i * entry_size
            if entry + entry_size > len(head):
                break
            if big:
                tag, ftype = struct.unpack(endian + "HH", head[entry:entry + 4])
                n = int(struct.unpack(endian + "Q", head[entry + 4:entry + 12])[0])
            else:
                tag, ftype, n = struct.unpack(endian + "HHI", head[entry:entry + 8])
            name = _TIFF_TAGS.get(tag)
            if not name:
                continue
            size = _TIFF_TYPE_SIZE.get(ftype, 0) * n
            inline = 8 if big else 4
            value_at = entry + (12 if big else 8)
            if size <= inline:
                raw = head[value_at:value_at + inline]
            else:
                if big:
                    offset = int(struct.unpack(endian + "Q",
                                               head[value_at:value_at + 8])[0]) - base
                else:
                    offset = int(struct.unpack(endian + "I",
                                               head[value_at:value_at + 4])[0]) - base
                if offset < 0 or offset + size > len(head):
                    continue                          # points past what we hold
                raw = head[offset:offset + size]
            if ftype in (3, 4) and len(raw) >= (2 if ftype == 3 else 4):
                fmt = "H" if ftype == 3 else "I"
                found[name] = struct.unpack(endian + fmt, raw[:2 if ftype == 3 else 4])[0]
            elif ftype == 12 and len(raw) >= 8:
                found[name] = list(struct.unpack(endian + f"{len(raw) // 8}d", raw))
            elif ftype == 2:
                found[name] = raw.split(b"\x00")[0].decode("latin-1", "replace")
            elif name == "geo_key_directory" and ftype == 3:
                found[name] = list(struct.unpack(endian + f"{len(raw) // 2}H", raw))
        if "width" in found:
            out["width"] = found["width"]
        if "height" in found:
            out["height"] = found["height"]
        if isinstance(found.get("pixel_scale"), list):
            out["pixel_size"] = found["pixel_scale"][:2]
        if isinstance(found.get("tie_point"), list) and len(found["tie_point"]) >= 6:
            out["origin"] = found["tie_point"][3:5]
        epsg = _epsg_from_geokeys(found.get("geo_key_directory"))
        if epsg:
            out["crs"] = f"EPSG:{epsg}"
        if found.get("geo_ascii_params"):
            out["crs_name"] = str(found["geo_ascii_params"])[:120].rstrip("|")
        if out.get("width") and out.get("height"):
            out["row_count"] = int(out["width"]) * int(out["height"])
            out["geometry_type"] = "Raster"
        return out
    except Exception:
        return {}


# GeoKey 3072 is ProjectedCSTypeGeoKey, 2048 is GeographicTypeGeoKey. Both carry an EPSG code
# directly in the value slot when the count is 1.
def _epsg_from_geokeys(keys: Optional[List[int]]) -> Optional[int]:
    if not keys or len(keys) < 4:
        return None
    number = keys[3]
    for i in range(number):
        base = 4 + i * 4
        if base + 4 > len(keys):
            break
        key_id, location, count, value = keys[base:base + 4]
        if key_id in (3072, 2048) and location == 0 and count == 1 and value not in (0, 32767):
            return int(value)
    return None


_SQLITE_MAGIC = b"SQLite format 3\x00"


def read_gpkg_header(head: bytes) -> Dict[str, Any]:
    """Confirm a GeoPackage from its SQLite header and application id.

    The layer table, CRS and bounds live in ``gpkg_contents``, which is a B-tree page that a
    ranged read of the file's front does not reliably reach — so this reports the FORMAT and
    stops. Claiming to know the layers from the header alone would be a guess dressed as a fact;
    reading them needs the whole file or a SQL-over-HTTP shim, which is a separate decision.
    """
    if not head.startswith(_SQLITE_MAGIC) or len(head) < 72:
        return {}
    app_id = head[68:72]
    out = {"format": "GeoPackage" if app_id in (b"GPKG", b"GP10", b"GP11") else "SQLite"}
    if out["format"] == "GeoPackage":
        out["note"] = ("GeoPackage confirmed from the SQLite application id; layer names, CRS "
                       "and bounds are in gpkg_contents and need more than the file header")
    return out


def read_csv_header(head: bytes, *, max_columns: int = 200) -> Dict[str, Any]:
    """Column names from a CSV's first line."""
    try:
        text = head.decode("utf-8", "replace")
    except Exception:
        return {}
    line = text.splitlines()[0] if text.splitlines() else ""
    if not line or line.count(",") < 1:
        return {}
    import csv
    import io

    try:
        row = next(csv.reader(io.StringIO(line)))
    except Exception:
        return {}
    fields = [c.strip() for c in row][:max_columns]
    return {"format": "CSV", "schema": fields} if fields else {}


def read_any(name: str, head: bytes) -> Dict[str, Any]:
    """Dispatch on the member's extension, returning {} when nothing is recognised."""
    lowered = (name or "").lower()
    if lowered.endswith((".las", ".laz")):
        return read_las_header(head)
    if lowered.endswith((".tif", ".tiff")):
        return read_tiff_header(head)
    if lowered.endswith(".gpkg"):
        return read_gpkg_header(head)
    if lowered.endswith((".csv", ".tsv", ".txt")):
        return read_csv_header(head)
    return {}


__all__ = ["read_las_header", "read_tiff_header", "read_gpkg_header", "read_csv_header",
           "read_any"]
