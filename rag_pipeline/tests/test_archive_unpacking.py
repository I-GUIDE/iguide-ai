"""Unpacking a third-party archive, assuming every member name is hostile.

18 of the 30 fetchable datasets in the corpus are ZIPs. They used to be LISTED and never
extracted — safe, and it left a zipped shapefile indistinguishable from an unreadable blob: no
schema, no geometry, no CRS, no bounding box, no loader.

Extracting means writing to a path that came out of an archive, which is exactly how zip-slip
works. Every attack below is constructed and run, not described, because a guard that has never
seen the attack it was written for is a guard nobody has tested.

One member behaving badly must not cost the archive: a shapefile bundle is a dozen files, and
refusing all of them because one is a symlink would trade a security hole for a data loss.
"""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import pytest

from extractors.archives import UnsafeArchive, primary_member, unpack


def _zip(path: Path, entries) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries:
            zf.writestr(name, data)
    return path


# ------------------------------------------------------------------ escaping the destination

def test_a_member_that_escapes_the_destination_is_refused(tmp_path):
    """Zip-slip. `../../evil.txt` writes outside the extraction directory, and on a real system
    that is a write to somewhere the process should never touch."""
    archive = _zip(tmp_path / "a.zip", [("good.csv", "a,b\n1,2\n"),
                                        ("../../evil.txt", "pwned")])
    out = unpack(archive, tmp_path / "dest")

    assert [m.name for m in out.members] == ["good.csv"]
    assert any("escapes" in why for _n, why in out.skipped)
    assert not (tmp_path.parent / "evil.txt").exists()
    assert not (tmp_path / "evil.txt").exists()


def test_an_absolute_path_member_is_refused(tmp_path):
    """`/etc/passwd` ignores the destination entirely rather than climbing out of it."""
    archive = _zip(tmp_path / "a.zip", [("/tmp/absolute.txt", "x"), ("fine.csv", "a\n1\n")])
    out = unpack(archive, tmp_path / "dest")
    assert [m.name for m in out.members] == ["fine.csv"]


def test_a_symlink_member_is_refused(tmp_path):
    """A member that is a link to `/` turns any LATER write into a write anywhere. Refused by
    the mode bits rather than by name, because the name looks ordinary."""
    archive = tmp_path / "link.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        info = zipfile.ZipInfo("sneaky")
        info.external_attr = (0o120777 << 16)          # S_IFLNK
        zf.writestr(info, "/etc")
        zf.writestr("real.csv", "a\n1\n")

    out = unpack(archive, tmp_path / "dest")
    assert [m.name for m in out.members] == ["real.csv"]
    assert any(why == "symlink" for _n, why in out.skipped)
    assert not (tmp_path / "dest" / "sneaky").is_symlink()


# ------------------------------------------------------------------ resource exhaustion

def test_a_compression_bomb_is_refused(tmp_path):
    """42.zip is 42 KB and decompresses to 4.5 PB. The ratio is the signal — a legitimate file
    does not compress 200:1."""
    archive = _zip(tmp_path / "bomb.zip", [("bomb.bin", "\0" * (5 * 1024 * 1024)),
                                           ("ok.csv", "a,b\n1,2\n")])
    out = unpack(archive, tmp_path / "dest")
    assert "ok.csv" in [m.name for m in out.members]
    assert any("bomb" in why for _n, why in out.skipped)


def test_the_member_count_is_capped(tmp_path):
    """A million tiny members exhausts inodes rather than bytes, so a byte cap alone misses it."""
    archive = _zip(tmp_path / "many.zip", [(f"f{i}.txt", "x") for i in range(50)])
    out = unpack(archive, tmp_path / "dest", max_members=10)
    assert len(out.members) == 10
    assert out.truncated, "silently stopping early would report a partial unpack as complete"


def test_the_total_size_is_capped(tmp_path):
    """Incompressible payloads on purpose: repetitive filler compresses past 200:1 and trips the
    BOMB guard first, which tests a different control and leaves this one unexercised."""
    import os as _os

    archive = _zip(tmp_path / "big.zip",
                   [(f"f{i}.bin", _os.urandom(14_000).hex()) for i in range(10)])
    out = unpack(archive, tmp_path / "dest", max_total_bytes=20_000)
    assert out.truncated, "the cap was not reached, so nothing was tested"
    assert out.total_bytes <= 20_000 + 28_000


# ------------------------------------------------------------------ it still does its job

def test_a_normal_archive_unpacks_completely(tmp_path):
    """The guards must not cost the ordinary case — that is how a security control gets removed."""
    archive = _zip(tmp_path / "ok.zip", [("data/sites.csv", "lat,lon\n41.9,-87.6\n"),
                                         ("data/readme.txt", "notes"),
                                         ("data/shape.shp", "\x00binary")])
    out = unpack(archive, tmp_path / "dest")
    assert len(out.members) == 3 and not out.skipped and not out.truncated
    assert (tmp_path / "dest" / "data" / "sites.csv").is_file()


def test_a_tar_archive_works_too(tmp_path):
    """`.tar`/`.tgz` were routed to a zip-only reader and always reported "could not read"."""
    payload = tmp_path / "x.csv"
    payload.write_text("a,b\n1,2\n", encoding="utf-8")
    archive = tmp_path / "a.tar"
    with tarfile.open(archive, "w") as tf:
        tf.add(payload, arcname="inner/x.csv")

    out = unpack(archive, tmp_path / "dest")
    assert [m.name for m in out.members] == ["x.csv"]


def test_a_non_archive_raises_rather_than_returning_empty(tmp_path):
    """"Not an archive" and "an archive containing nothing" are different facts."""
    plain = tmp_path / "notes.txt"
    plain.write_text("hello", encoding="utf-8")
    with pytest.raises(UnsafeArchive):
        unpack(plain, tmp_path / "dest")


# ------------------------------------------------------------------ choosing what to describe

def test_the_primary_member_is_the_dataset_not_the_readme(tmp_path):
    """An archive routinely holds a 200 MB raster beside a 2 KB readme, and the readme is not
    the dataset."""
    root = tmp_path / "dest"
    (root / "d").mkdir(parents=True)
    for name, size in (("readme.txt", 4000), ("sites.shp", 50), ("notes.csv", 10)):
        (root / "d" / name).write_bytes(b"x" * size)
    members = list((root / "d").iterdir())

    assert primary_member(members).name == "sites.shp", "extension priority beats size"


def test_packaging_residue_is_never_the_primary_member(tmp_path):
    """__MACOSX and dotfiles outnumber the real files in many archives."""
    root = tmp_path / "dest"
    (root / "__MACOSX").mkdir(parents=True)
    (root / "__MACOSX" / "._sites.csv").write_bytes(b"x" * 9000)
    (root / "sites.csv").write_bytes(b"lat,lon\n1,2\n")

    members = [root / "__MACOSX" / "._sites.csv", root / "sites.csv"]
    assert primary_member(members).name == "sites.csv"


# ------------------------------------------------------------------ end to end through the extractor

def test_a_zipped_csv_is_described_as_data_not_as_a_container(tmp_path):
    """The point of all of this: the dataset doc should describe the DATA, while still recording
    that it arrived in an archive."""
    from extractors.data_extractor import extract_dataset_metadata

    archive = _zip(tmp_path / "ds.zip",
                   [("sites.csv", "name,latitude,longitude\na,41.9,-87.6\nb,41.8,-87.7\n"),
                    ("readme.txt", "about this dataset")])
    described = extract_dataset_metadata(str(archive))

    assert described["format"] == "zip", "the archive's own format is still reported"
    assert described["member_count"] == 2
    assert described["primary_member"] == "sites.csv"
    assert "member(s) unpacked" in described["archive_note"]
    # And the member's own description came through, namespaced so it cannot overwrite the
    # archive's facts.
    assert described.get("member_format", "").lower() in ("csv", "tabular", "text/csv")


def test_an_archive_of_only_junk_says_so(tmp_path):
    from extractors.data_extractor import extract_dataset_metadata

    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "x")
    described = extract_dataset_metadata(str(archive))
    assert "archive_note" in described
