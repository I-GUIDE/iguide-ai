"""Describe a remote ZIP without downloading it, using HTTP range requests.

11 of the 31 datasets the platform hosts itself exceed the 64 MB fetch cap, and they are the
substantial ones — 3.9 GB of dam-failure vulnerability rasters, 2.3 GB of GLiM/GLHYMPS
hydrogeology, LiDAR point clouds, VIIRS nighttime lights, NAIP imagery. 8.5 GB in total, every
one `application/zip`. Raising the cap would mean storing all of it to learn what is inside.

That is unnecessary. A ZIP's central directory sits at the END of the file, so:

* one HEAD gives size and content type;
* one ranged read of the last ~64 KB gives every member name, size and offset;
* a second small ranged read gives an individual member — and the members that carry a
  shapefile's schema and CRS are the ``.dbf`` header and the ``.prj``, which are kilobytes.

So a 3.9 GB archive can be described from roughly 70 KB of transfer. The description is weaker
than a full read — no feature count, no bounds — and it says so, because "we did not open the
geometry" and "the geometry is fine" must not look the same.

Needs the server to honour ``Range``. MinIO does; a server that does not is reported as such
rather than silently falling back to a full download.
"""

from __future__ import annotations

import io
import struct
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

EOCD_SIGNATURE = b"PK\x05\x06"
TAIL_BYTES = 96 * 1024          # enough for the EOCD plus a large central directory
MEMBER_PREVIEW_BYTES = 64 * 1024


@dataclass
class RemoteZip:
    url: str
    total_bytes: int = 0
    members: List[Dict[str, Any]] = field(default_factory=list)
    supports_range: bool = False
    note: str = ""

    def member_names(self) -> List[str]:
        return [m["name"] for m in self.members]


def _head(url: str, timeout: int, session: Any) -> Dict[str, str]:
    resp = session.head(url, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    return {k.lower(): v for k, v in resp.headers.items()}


def _range(url: str, start: int, end: int, timeout: int, session: Any) -> bytes:
    """Bytes [start, end] inclusive. Raises if the server ignores the header."""
    resp = session.get(url, timeout=timeout, allow_redirects=True,
                       headers={"Range": f"bytes={start}-{end}"})
    if resp.status_code != 206:
        raise RuntimeError(f"server returned {resp.status_code}, not 206 Partial Content")
    return resp.content


def inspect(url: str, *, timeout: int = 60, session: Any = None) -> RemoteZip:
    """List a remote ZIP's members from its central directory alone."""
    import requests

    own = session is None
    session = session or requests.Session()
    out = RemoteZip(url=url)
    try:
        headers = _head(url, timeout, session)
        out.total_bytes = int(headers.get("content-length") or 0)
        out.supports_range = headers.get("accept-ranges", "").lower() == "bytes"
        if not out.total_bytes:
            out.note = "server did not report a content length; cannot range-read"
            return out

        tail_start = max(0, out.total_bytes - TAIL_BYTES)
        tail = _range(url, tail_start, out.total_bytes - 1, timeout, session)
        out.supports_range = True

        # zipfile can parse a truncated archive if the central directory is present and the
        # offsets are made relative to what we actually hold.
        eocd = tail.rfind(EOCD_SIGNATURE)
        if eocd < 0:
            out.note = ("no end-of-central-directory in the last "
                        f"{TAIL_BYTES // 1024} KB; archive may use ZIP64 with a large comment")
            return out

        cd_size, cd_offset = struct.unpack("<II", tail[eocd + 12:eocd + 20])
        if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
            out.note = "ZIP64 central directory; not parsed"
            return out
        if cd_offset < tail_start:
            # The directory starts before the window we fetched. Pull exactly it.
            tail = _range(url, cd_offset, out.total_bytes - 1, timeout, session)
            tail_start = cd_offset
            eocd = tail.rfind(EOCD_SIGNATURE)

        # Rebuild a minimal in-memory archive: the central directory plus the EOCD, with the
        # directory offset rewritten to where it now sits.
        shifted = bytearray(tail)
        new_offset = cd_offset - tail_start
        struct.pack_into("<I", shifted, eocd + 16, new_offset)
        try:
            with zipfile.ZipFile(io.BytesIO(bytes(shifted))) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    out.members.append({
                        "name": info.filename,
                        "size": int(info.file_size),
                        "compressed": int(info.compress_size),
                        "offset": int(info.header_offset),
                    })
        except Exception as exc:
            out.note = f"central directory unreadable: {type(exc).__name__}: {exc}"[:140]
            return out
        out.note = (f"listed {len(out.members)} member(s) from "
                    f"{len(tail) // 1024} KB of a {out.total_bytes / 1e6:.0f} MB archive")
    except Exception as exc:
        out.note = f"{type(exc).__name__}: {exc}"[:140]
    finally:
        if own:
            session.close()
    return out


def fetch_member(url: str, member: Dict[str, Any], *, total_bytes: int,
                 timeout: int = 60, session: Any = None,
                 max_bytes: int = MEMBER_PREVIEW_BYTES) -> Optional[bytes]:
    """The decompressed bytes of one small member, or None.

    Reads from the member's local header to a bounded distance past it, then lets zipfile do the
    decompression. Used for the members that carry metadata cheaply — a ``.prj`` is a few hundred
    bytes of WKT and a ``.dbf`` header holds the field names in its first kilobyte.
    """
    import requests

    own = session is None
    session = session or requests.Session()
    try:
        start = int(member["offset"])
        # Enough input to decompress `max_bytes` of output, plus the local header. Deflate rarely
        # expands, so a window a few times the target is ample and stays kilobyte-scale even for
        # a member inside a 3.9 GB archive.
        window = min(int(member["compressed"]) + 4096, max(max_bytes * 4, 32 * 1024))
        end = min(start + window, total_bytes - 1)
        blob = _range(url, start, end, timeout, session)

        name_len = struct.unpack("<H", blob[26:28])[0]
        extra_len = struct.unpack("<H", blob[28:30])[0]
        data_start = 30 + name_len + extra_len
        compressed = blob[data_start:]
        method = struct.unpack("<H", blob[8:10])[0]
        if method == 0:
            return compressed[:max_bytes]
        if method == 8:
            import zlib

            # `decompressobj`, not `decompress`: the stream is deliberately TRUNCATED, and
            # zlib.decompress rejects a partial stream outright. A .shp header is 100 bytes at
            # the front of a 160 MB member, so reading only the front is the entire point —
            # the bounding box sits at bytes 36..68 and the whole file is never needed.
            engine = zlib.decompressobj(-15)
            try:
                return engine.decompress(compressed, max_bytes)
            except zlib.error:
                return None
        return None
    except Exception:
        return None
    finally:
        if own:
            session.close()


def shapefile_bounds(shp_header: bytes) -> Optional[List[float]]:
    """[xmin, ymin, xmax, ymax] from a .shp header, or None.

    The bounding box is at a FIXED offset — bytes 36..68 of the 100-byte header — so it costs
    100 bytes out of a member that may be 160 MB. Byte 0 is a big-endian magic 9994; checking it
    is what stops a wrong guess being reported as a bounding box.
    """
    if len(shp_header) < 68:
        return None
    try:
        if struct.unpack(">i", shp_header[0:4])[0] != 9994:
            return None
        return [float(v) for v in struct.unpack("<4d", shp_header[36:68])]
    except Exception:
        return None


def dbf_header(dbf_bytes: bytes) -> Dict[str, Any]:
    """Record count and field names from a .dbf header.

    The count is a little-endian uint32 at bytes 4..8, then one 32-byte field descriptor per
    column until a 0x0D terminator. This is the schema and the feature count of a shapefile,
    both from its first kilobyte — measured on the corpus: 1,197,659 OSM building footprints
    and their 10 column names, out of a 144 MB archive.
    """
    out: Dict[str, Any] = {}
    if len(dbf_bytes) < 32:
        return out
    try:
        out["record_count"] = int(struct.unpack("<I", dbf_bytes[4:8])[0])
    except Exception:
        return out
    fields: List[str] = []
    pos = 32
    while pos + 32 <= len(dbf_bytes) and dbf_bytes[pos] != 0x0D:
        name = dbf_bytes[pos:pos + 11].split(b"\x00")[0].decode("latin-1").strip()
        if name:
            fields.append(name)
        pos += 32
    out["fields"] = fields
    return out


__all__ = ["inspect", "fetch_member", "shapefile_bounds", "dbf_header",
           "RemoteZip", "TAIL_BYTES"]
