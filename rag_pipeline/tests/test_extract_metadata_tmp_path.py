"""The metadata extractor downloads an object by a caller-controlled KEY. It must not follow that
key out of /tmp.

The original wrote to f"/tmp/{key}". prototype replaced it with os.path.join("/tmp", key) plus
os.makedirs(...), which handled nested keys but not hostile ones: "../../etc/x" resolves above
/tmp, an absolute key makes join() discard "/tmp" entirely, and makedirs then created whatever
directories the key named. PR #12 (written against main) had the safe form; this pins it on
prototype, where the extractor actually runs.
"""
from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys
import types

import pytest

MOD = (pathlib.Path(__file__).resolve().parents[2]
       / "metadata-extraction-server" / "extract_metadata_code_notebooks.py")


def _load(monkeypatch):
    # The module imports fiona, rasterio and nbformat at the top, but none of them is touched on
    # the path this test exercises (choosing where to download; fiona is only for zipped
    # shapefiles). A stand-in for any that is not installed lets the test RUN everywhere: an
    # importorskip here skipped it on any machine without fiona, which protected nothing.
    for dep in ("nbformat", "fiona", "rasterio"):
        try:
            importlib.import_module(dep)
        except ImportError:
            monkeypatch.setitem(sys.modules, dep, types.ModuleType(dep))
    pytest.importorskip("boto3")       # the client is replaced below, but the module is real
    spec = importlib.util.spec_from_file_location("extract_metadata_code_notebooks", MOD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeS3:
    """Records where the extractor asked to download to; writes nothing anywhere."""

    def __init__(self):
        self.downloaded_to = None

    def download_file(self, bucket, key, path):
        self.downloaded_to = path

    def put_object_tagging(self, **kwargs):
        pass


@pytest.mark.parametrize("key", [
    "../../etc/escape.py",          # climbs out of /tmp
    "/etc/escape.py",               # absolute: os.path.join would discard /tmp
    "nested/dirs/notebook.ipynb",   # legitimate, but must not create directories
    "trailing/",                    # no basename at all
])
def test_a_key_cannot_place_the_download_outside_tmp(monkeypatch, key):
    m = _load(monkeypatch)
    s3 = _FakeS3()
    monkeypatch.setattr(m.boto3, "client", lambda *a, **k: s3)
    try:
        m.extract_metadata("bucket", key)
    except SystemExit:
        pass  # the extractor exits on failure; the download path was already chosen
    assert s3.downloaded_to is not None, "the extractor never reached the download"
    target = pathlib.Path(s3.downloaded_to)
    assert target.parent == pathlib.Path("/tmp"), f"{key!r} downloaded to {target}"
    assert ".." not in target.parts
