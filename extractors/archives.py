"""Unpack a third-party archive without trusting a single byte of it.

18 of the 30 fetchable datasets in the corpus are ZIPs, and ``_handle_container`` listed their
members without extracting — deliberately, because writing to a path that came out of an archive
is how zip-slip works. Listing is safe and gives no schema, no geometry, no bounding box and no
loader, so a zipped shapefile was indistinguishable from an unreadable blob.

This unpacks, and treats every member name as hostile. Five things an archive can do that a
naive ``extractall`` permits:

* **escape** — a member named ``../../etc/passwd`` writes outside the destination;
* **absolute paths** — ``/etc/passwd`` ignores the destination entirely;
* **symlinks** — a member that is a link to ``/`` turns a later write into a write anywhere;
* **bombs** — 42.zip is 42 KB and decompresses to 4.5 PB;
* **flooding** — a million tiny members exhausts inodes rather than bytes.

Each is refused explicitly rather than by a general try/except, because "the archive was
malformed" and "the archive attacked us" should not look the same in a log.
"""

from __future__ import annotations

import os
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

MAX_TOTAL_BYTES = 2 * 1024 ** 3      # 2 GiB uncompressed across the whole archive
MAX_MEMBER_BYTES = 512 * 1024 ** 2   # 512 MiB for any single member
MAX_MEMBERS = 5000
MAX_RATIO = 200                      # uncompressed / compressed, per member


class UnsafeArchive(Exception):
    """The archive tried something that is never legitimate. Distinct from "malformed"."""


@dataclass
class Unpacked:
    root: Path
    members: List[Path] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)   # (name, why)
    total_bytes: int = 0
    truncated: bool = False

    def note(self) -> str:
        parts = [f"{len(self.members)} member(s) unpacked"]
        if self.skipped:
            reasons = {why for _n, why in self.skipped}
            parts.append(f"{len(self.skipped)} refused ({', '.join(sorted(reasons))})")
        if self.truncated:
            parts.append("member cap reached; archive not fully unpacked")
        return "; ".join(parts)


def is_archive(path: str | Path) -> bool:
    suffix = "".join(Path(path).suffixes[-2:]).lower()
    return (zipfile.is_zipfile(str(path)) or tarfile.is_tarfile(str(path))
            or suffix.endswith((".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2")))


def _safe_destination(root: Path, name: str) -> Optional[Path]:
    """Where *name* may be written, or None if it must not be written at all.

    ``Path.resolve()`` then a containment check, rather than string prefix matching: ``..`` is
    only one way out, and a symlinked parent directory is another that string comparison misses.
    """
    if not name or name.endswith("/"):
        return None
    candidate = Path(name)
    if candidate.is_absolute() or candidate.drive:
        return None
    if any(part == ".." for part in candidate.parts):
        return None
    target = (root / candidate).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return None
    return target


def _zip_members(archive: zipfile.ZipFile):
    for info in archive.infolist():
        if info.is_dir():
            continue
        # Bit 0x10 of the external attr is the MS-DOS directory flag; the high 16 bits carry
        # the unix mode, whose S_IFLNK marks a symlink. A symlink member is never data.
        mode = info.external_attr >> 16
        is_link = bool(mode & 0o120000 == 0o120000)
        yield info.filename, info.file_size, info.compress_size, is_link


def _tar_members(archive: tarfile.TarFile):
    for info in archive.getmembers():
        if info.isdir():
            continue
        yield info.name, info.size, info.size, (info.issym() or info.islnk())


def unpack(path: str | Path, dest: str | Path, *,
           max_total_bytes: int = MAX_TOTAL_BYTES,
           max_member_bytes: int = MAX_MEMBER_BYTES,
           max_members: int = MAX_MEMBERS) -> Unpacked:
    """Extract *path* into *dest*, refusing anything unsafe. Never raises on a hostile member.

    ``UnsafeArchive`` is raised only when the archive itself cannot be opened as one; individual
    bad members are skipped and recorded, because one malicious entry should not cost the
    dataset the other forty legitimate files.
    """
    source, root = Path(path), Path(dest)
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    out = Unpacked(root=root)

    if zipfile.is_zipfile(str(source)):
        opener, lister, reader = zipfile.ZipFile, _zip_members, "zip"
    elif tarfile.is_tarfile(str(source)):
        opener, lister, reader = tarfile.open, _tar_members, "tar"
    else:
        raise UnsafeArchive(f"not a readable zip or tar archive: {source.name}")

    with opener(str(source)) as archive:      # type: ignore[operator]
        for name, size, compressed, is_link in lister(archive):
            if len(out.members) >= max_members:
                out.truncated = True
                break
            if is_link:
                out.skipped.append((name, "symlink"))
                continue
            target = _safe_destination(root, name)
            if target is None:
                out.skipped.append((name, "path escapes destination"))
                continue
            if size > max_member_bytes:
                out.skipped.append((name, "member too large"))
                continue
            if compressed > 0 and size / max(compressed, 1) > MAX_RATIO:
                out.skipped.append((name, "compression ratio suggests a bomb"))
                continue
            if out.total_bytes + size > max_total_bytes:
                out.truncated = True
                break

            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                if reader == "zip":
                    with archive.open(name) as src, open(target, "wb") as dst:
                        dst.write(src.read(max_member_bytes + 1))
                else:
                    extracted = archive.extractfile(name)
                    if extracted is None:
                        out.skipped.append((name, "unreadable member"))
                        continue
                    with extracted as src, open(target, "wb") as dst:
                        dst.write(src.read(max_member_bytes + 1))
            except Exception as exc:
                out.skipped.append((name, f"{type(exc).__name__}"))
                continue
            out.total_bytes += size
            out.members.append(target)
    return out


# Extensions worth extracting FROM, most informative first. A zipped shapefile is the common
# case and its .shp is the member that carries schema, geometry and CRS — the sidecars are
# needed on disk but are not the file to describe.
_PRIMARY_ORDER = (".gpkg", ".shp", ".geojson", ".gdb", ".tif", ".tiff", ".nc", ".parquet",
                  ".csv", ".xlsx", ".json")


def primary_member(members: List[Path]) -> Optional[Path]:
    """The member that best represents the dataset, or None.

    Picks by extension priority then by size, because an archive routinely holds a 200 MB raster
    beside a 2 KB readme and the readme is not the dataset. Skips __MACOSX and dotfiles, which
    are packaging residue that would otherwise win on count.
    """
    usable = [m for m in members
              if not any(part.startswith((".", "__MACOSX")) for part in m.parts)]
    for ext in _PRIMARY_ORDER:
        matching = [m for m in usable if m.suffix.lower() == ext]
        if matching:
            return max(matching, key=lambda m: m.stat().st_size if m.exists() else 0)
    return max(usable, key=lambda m: m.stat().st_size if m.exists() else 0) if usable else None


__all__ = ["unpack", "primary_member", "is_archive", "Unpacked", "UnsafeArchive",
           "MAX_TOTAL_BYTES", "MAX_MEMBER_BYTES", "MAX_MEMBERS"]
