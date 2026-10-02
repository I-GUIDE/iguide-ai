"""Packages the code reaches for only at call time, exercised so that a missing one FAILS.

None of these is imported at module scope, and two are never imported by name at all: pandas
and geopandas load pyarrow inside to_parquet/read_parquet, and pandas loads openpyxl inside
read_excel, where an import grep cannot see either. The code around each one degrades instead
of raising. write_geodata falls back to a pickle no other tool can open, publication_extractor
reads a PDF as empty text, data_extractor returns a "reader unavailable" note, and
choropleth_image draws a continuous ramp in place of the scheme it was asked for. So the suite
passed on every machine that happened to have them, while the deployed image, which installs
requirements.txt and nothing else, had none of them (measured 2026-10-01 by running this suite
inside a replica of that image). The reproject_vector and spatial-join parquet paths are
covered in test_langchain_geo_tools.py.

There is deliberately no importorskip anywhere in this file. Every package exercised here is a
declared requirement, and a skip would turn its absence back into the silent pass this module
exists to prevent.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture()
def store(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    return tmp_path


# --- pyarrow -------------------------------------------------------------------------------

def test_geo_handles_pass_frames_as_geoparquet_not_the_pickle_fallback(store):
    import geopandas as gpd
    from shapely.geometry import Point

    from agent_runtime.file_store import resolve_file_id
    from extractors.geo_handles import read_geodata, write_geodata

    gdf = gpd.GeoDataFrame({"v": [1, 2]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326")
    fid = write_geodata(gdf, "pts")
    # Without pyarrow this silently becomes pts.pkl, which only read_geodata can open: not
    # inspect_vector, not add_map_layer, not the sandbox.
    assert resolve_file_id(fid).suffix == ".parquet"
    back = read_geodata(fid)
    assert list(back["v"]) == [1, 2] and back.crs.to_epsg() == 4326


# --- pypdf, python-docx --------------------------------------------------------------------

def _one_page_pdf(text: str) -> bytes:
    """A one-page PDF showing `text` in Helvetica, written by hand so that making it needs no
    library: the only PDF code that runs is the reader under test."""
    stream = f"BT /F1 18 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    trailer = b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
    out += trailer % (len(objects) + 1, xref)
    return bytes(out)


def test_publication_reader_extracts_text_from_a_pdf(tmp_path):
    from extractors.publication_extractor import _read_text

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_one_page_pdf("Flood exposure by census tract"))
    # Without pypdf this returns "", and the extractor files the paper with an empty method and
    # the note "no_text_extracted", which does not say why.
    assert "Flood exposure by census tract" in _read_text(str(pdf))


def test_publication_reader_extracts_text_from_a_docx(tmp_path):
    import docx

    from extractors.publication_extractor import _read_text

    path = tmp_path / "methods.docx"
    document = docx.Document()
    document.add_paragraph("Kernel density over geocoded 911 calls")
    document.save(path)
    assert "Kernel density over geocoded 911 calls" in _read_text(str(path))


# --- xarray --------------------------------------------------------------------------------

def test_data_extractor_opens_a_netcdf_file(tmp_path):
    import numpy as np
    from scipy.io import netcdf_file

    from extractors.data_extractor import _handle_raster

    path = tmp_path / "precip.nc"
    with netcdf_file(path, "w") as nc:          # NetCDF3, written without xarray
        nc.createDimension("time", 4)
        precip = nc.createVariable("precip", "f4", ("time",))
        precip[:] = np.arange(4, dtype="f4")
    meta = _handle_raster(str(path))
    # Without xarray: {"format": "nc", "note": "raster reader unavailable/failed: ..."}.
    # NetCDF3 on purpose: xarray reads it through scipy, which the stack already carries.
    # NetCDF4/HDF5 would need netCDF4 or h5netcdf, which nothing declares.
    assert "note" not in meta, meta.get("note")
    assert meta["variables"] == ["precip"] and meta["dims"] == {"time": 4}


# --- openpyxl ------------------------------------------------------------------------------

def test_time_series_reads_a_spreadsheet_with_no_coordinates(store):
    import pandas as pd
    from werkzeug.datastructures import FileStorage

    from agent_runtime.analysis_temporal_tools import make_temporal_tools
    from agent_runtime.file_store import save_uploaded_file

    xlsx = store / "daily_counts.xlsx"
    pd.DataFrame({"date": pd.date_range("2025-01-01", periods=90, freq="D"),
                  "count": range(90)}).to_excel(xlsx, index=False)
    with open(xlsx, "rb") as fh:
        fid = save_uploaded_file(FileStorage(stream=fh, filename=xlsx.name))["file_id"]
    tools = {t.name: t for t in make_temporal_tools([fid])}
    # GDAL opens a .xlsx, but read_vector refuses a table with no coordinates, so this falls
    # back to pandas.read_excel, which loads openpyxl. Without it the tool answers
    # "ImportError: `Import openpyxl` failed".
    out = json.loads(tools["time_series"].invoke({"file_id": fid, "freq": "month"}))
    assert out["ok"] is True, out.get("error")
    assert out["periods"] == 3


# --- mapclassify ---------------------------------------------------------------------------

def test_choropleth_image_applies_the_requested_scheme(store, monkeypatch):
    import geopandas as gpd
    import geopandas.plotting
    from shapely.geometry import box

    from extractors.geo_handles import choropleth_image, write_geodata

    seen = {}
    real_plot = geopandas.plotting.plot_dataframe

    def spy(df, *args, **kwargs):
        seen.update(kwargs)
        return real_plot(df, *args, **kwargs)

    monkeypatch.setattr(geopandas.plotting, "plot_dataframe", spy)
    zones = gpd.GeoDataFrame({"n": [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]},
                             geometry=[box(i, 0, i + 1, 1) for i in range(10)], crs="EPSG:4326")
    out = json.loads(choropleth_image(write_geodata(zones, "zones"), "n", scheme="Quantiles"))
    assert out.get("png_file_id"), out
    # Without mapclassify the tool drops the scheme and draws a continuous ramp, and nothing in
    # its result says so.
    assert seen.get("scheme") == "Quantiles"


# --- IPython -------------------------------------------------------------------------------

def test_notebook_cells_go_through_ipythons_own_transformer():
    from extractors.r1_ipython_frontend import transform_cell

    # Both parse only through IPython; the regex fallback leaves them as SyntaxErrors.
    for cell in ("np.mean?", "for f in ['a', 'b']:\n    !echo {f}"):
        _, parse_ok, note = transform_cell(cell)
        assert parse_ok, note
        assert note != "ipython_unavailable_fallback"
