"""Tests for the /agent/files/<id>/download route's content-type, disposition, and
security headers — specifically that raster images are served inline, SVG is NEVER
served inline (stored-XSS guard), non-images download as attachments, and every
response carries X-Content-Type-Options: nosniff.
"""

from __future__ import annotations

import io

import pytest


def _upload(name: str, data: bytes) -> str:
    from werkzeug.datastructures import FileStorage
    from agent_runtime.file_store import save_uploaded_file
    return save_uploaded_file(FileStorage(stream=io.BytesIO(data), filename=name))["file_id"]


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    monkeypatch.delenv("AGENT_CHAT_API_KEY", raising=False)
    import api.server as srv
    return srv.app.test_client()


def _disp(resp) -> str:
    return (resp.headers.get("Content-Disposition") or "").lower()


def test_png_is_served_inline_with_nosniff(client):
    fid = _upload("plot.png", b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    r = client.get(f"/agent/files/{fid}/download")
    assert r.status_code == 200
    assert r.headers.get("Content-Type", "").startswith("image/png")
    assert "inline" in _disp(r) and "attachment" not in _disp(r)
    assert r.headers.get("X-Content-Type-Options") == "nosniff"


def test_jpeg_inline(client):
    fid = _upload("photo.JPG", b"\xff\xd8\xff" + b"0" * 32)
    r = client.get(f"/agent/files/{fid}/download")
    assert r.headers.get("Content-Type", "").startswith("image/jpeg")
    assert "inline" in _disp(r)


def test_svg_is_never_inline(client):
    """SVG can carry executable script -> must download, not render on tab-open."""
    fid = _upload("vec.svg", b"<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script></svg>")
    r = client.get(f"/agent/files/{fid}/download")
    assert "attachment" in _disp(r) and "inline" not in _disp(r)
    assert r.headers.get("X-Content-Type-Options") == "nosniff"


def test_force_download_param_overrides_inline(client):
    fid = _upload("plot.png", b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    r = client.get(f"/agent/files/{fid}/download?download=1")
    assert "attachment" in _disp(r)


def test_non_image_is_attachment(client):
    fid = _upload("data.parquet", b"PAR1" + b"0" * 32)
    r = client.get(f"/agent/files/{fid}/download")
    assert "attachment" in _disp(r)
    assert r.headers.get("X-Content-Type-Options") == "nosniff"


def test_unknown_file_id_404(client):
    r = client.get("/agent/files/does_not_exist/download")
    assert r.status_code == 404
