"""Dataset extraction: bounding boxes that survive indexing, and formats read as themselves.

Two defects here were losing data silently on every ingest.

**The bbox.** ``_envelope`` wrote the file's NATIVE bounds straight into
``spatial-bounding-box-geojson``. That field is mapped ``{type: geo_shape, ignore_malformed:
true}``, so UTM metres did not fail the write — OpenSearch dropped the field and the document
indexed cleanly. The dataset was then absent from every spatial query with nothing recording
why. Measured on the live index: **181 of 619 docs carry a bbox**.

**The spreadsheet.** ``.xlsx`` routed to ``csv.reader``, which reads the binary container
without raising and returns a garbage single-column header. Worse than an error, because the
asset indexes with nonsense that looks like real metadata.

The asymmetry throughout: an ABSENT bbox is incomplete, a WRONG bbox is wrong — and since
``ignore_malformed`` makes them indistinguishable downstream, the only way to tell them apart
later is a note recorded at extraction time.
"""

from __future__ import annotations

import json
import zipfile

import pytest

from extractors.base import ExtractContext
from extractors.data_extractor import (DataExtractor, extract_dataset_metadata,
                                       family_for_ext)
from rag_pipeline.search.geo_shapes import (bbox_geo_shape, infer_geo_shape,
                                            plausible_wgs84, to_wgs84_bounds)

# Chicago, in UTM zone 16N metres — the shape of every bbox this used to mangle.
UTM16N_CHICAGO = [440000.0, 4630000.0, 460000.0, 4650000.0]
WGS84_CHICAGO = [-87.9, 41.6, -87.5, 42.0]


def _extract(path, **fields):
    ctx = ExtractContext(element_id="ds01", element_type="dataset", fields=fields)
    return DataExtractor().extract(str(path), ctx=ctx)


def _spatial(result):
    return result.assets[0].spatial or {}


# ------------------------------------------------------------------ reprojection

def test_utm_bounds_are_reprojected_to_lon_lat():
    pytest.importorskip("pyproj")
    out, note = to_wgs84_bounds(UTM16N_CHICAGO, "EPSG:32616")
    assert out is not None and plausible_wgs84(out)
    assert -88.5 < out[0] < -87.0 and 41.0 < out[1] < 42.5
    assert "reprojected" in note


def test_wgs84_bounds_pass_through_unchanged():
    out, note = to_wgs84_bounds(WGS84_CHICAGO, "EPSG:4326")
    assert out == WGS84_CHICAGO and note == ""


def test_projected_bounds_with_no_crs_yield_no_bbox():
    """Assuming EPSG:4326 when the CRS is missing IS the bug — most files that omit a CRS are
    not in degrees."""
    out, note = to_wgs84_bounds(UTM16N_CHICAGO, None)
    assert out is None
    assert "outside lon/lat range" in note


def test_degree_bounds_with_no_crs_are_accepted_with_a_note():
    out, note = to_wgs84_bounds(WGS84_CHICAGO, None)
    assert out == WGS84_CHICAGO
    assert "no CRS declared" in note


def test_a_crs_claiming_4326_with_metre_bounds_is_refused():
    """A mislabelled file must not produce a bbox that will be silently dropped."""
    out, note = to_wgs84_bounds(UTM16N_CHICAGO, "EPSG:4326")
    assert out is None and "outside lon/lat range" in note


def test_an_unknown_crs_yields_no_bbox_rather_than_raw_bounds():
    out, note = to_wgs84_bounds(UTM16N_CHICAGO, "EPSG:999999")
    assert out is None and "could not reproject" in note


@pytest.mark.parametrize("bounds", [
    [-181, 0, 10, 10],          # lon out of range
    [0, -91, 10, 10],           # lat out of range
    [10, 0, -10, 10],           # inverted
    UTM16N_CHICAGO,             # metres
])
def test_implausible_bounds_are_rejected(bounds):
    assert plausible_wgs84(bounds) is False


def test_bbox_geo_shape_builds_an_envelope():
    pytest.importorskip("pyproj")
    shape, _ = bbox_geo_shape(UTM16N_CHICAGO, "EPSG:32616")
    assert shape and shape["type"] == "envelope" and len(shape["coordinates"]) == 2


# ------------------------------------------------------------------ end to end

def test_a_geojson_dataset_gets_a_bbox(tmp_path):
    path = tmp_path / "points.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"n": 1},
         "geometry": {"type": "Point", "coordinates": [-87.6, 41.9]}},
        {"type": "Feature", "properties": {"n": 2},
         "geometry": {"type": "Point", "coordinates": [-87.5, 42.0]}}]}))
    spatial = _spatial(_extract(path))
    assert "spatial-bounding-box-geojson" in spatial


def test_a_dataset_whose_bbox_cannot_be_trusted_records_why(tmp_path, monkeypatch):
    """The loss must be surfaced. "No spatial extent" and "we had one and could not use it"
    are different facts, and only one of them is a bug to fix."""
    path = tmp_path / "mystery.tif"
    path.write_bytes(b"not really a tif")
    monkeypatch.setattr("extractors.data_extractor._handle_raster",
                        lambda p: {"format": "GeoTIFF", "bounds": UTM16N_CHICAGO, "crs": ""})
    result = _extract(path)
    spatial = _spatial(result)
    assert "spatial-bounding-box-geojson" not in spatial
    assert spatial["bounds"] == UTM16N_CHICAGO, "native bounds are kept for provenance"
    assert result.assets[0].extracted["bbox_note"]
    assert any("no spatial bbox emitted" in w for w in result.warnings)


def test_a_utm_raster_ends_up_with_a_lon_lat_bbox(tmp_path, monkeypatch):
    pytest.importorskip("pyproj")
    path = tmp_path / "dem.tif"
    path.write_bytes(b"stub")
    monkeypatch.setattr("extractors.data_extractor._handle_raster",
                        lambda p: {"format": "GeoTIFF", "bounds": UTM16N_CHICAGO,
                                   "crs": "EPSG:32616"})
    spatial = _spatial(_extract(path))
    shape = spatial["spatial-bounding-box-geojson"]
    for lon, lat in shape["coordinates"]:
        assert -180 <= lon <= 180 and -90 <= lat <= 90


# ------------------------------------------------------------------ formats

def test_xlsx_is_read_as_a_spreadsheet_not_as_csv(tmp_path):
    """csv.reader on a binary xlsx returns a garbage header with NO exception."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    path = tmp_path / "table.xlsx"
    pd.DataFrame({"latitude": [41.9, 42.0], "longitude": [-87.6, -87.5],
                  "value": [1, 2]}).to_excel(path, index=False)
    meta = extract_dataset_metadata(str(path))
    assert meta["format"] == "XLSX"
    assert set(meta["schema"]) == {"latitude", "longitude", "value"}
    assert meta["row_count"] == 2


def test_a_coordinate_table_yields_a_bbox(tmp_path):
    pytest.importorskip("pandas")
    path = tmp_path / "pts.csv"
    path.write_text("latitude,longitude,v\n41.9,-87.6,1\n42.0,-87.5,2\n", encoding="utf-8")
    meta = extract_dataset_metadata(str(path))
    assert meta.get("bounds") == [-87.6, 41.9, -87.5, 42.0]
    assert meta.get("crs") == "EPSG:4326"


def test_json_and_xml_route_to_a_handler_not_to_an_unhandled_sidecar():
    """Both are in SIDECAR_EXT, which has no handler, so a STAC item was indexed with no bbox."""
    from extractors.data_extractor import _HANDLERS

    for ext in (".json", ".xml"):
        assert family_for_ext(ext) == "metadata"
        assert "metadata" in _HANDLERS


def test_a_stac_items_declared_bbox_is_used(tmp_path):
    path = tmp_path / "item.json"
    path.write_text(json.dumps({"stac_version": "1.0.0", "type": "Feature_x", "id": "scene-1",
                                "bbox": [-87.9, 41.6, -87.5, 42.0],
                                "properties": {"datetime": "2024-01-01T00:00:00Z"}}))
    meta = extract_dataset_metadata(str(path))
    assert meta["format"] == "STAC"
    assert meta["bounds"] == [-87.9, 41.6, -87.5, 42.0]
    assert meta["bbox_from"] == "declared"


def test_a_geojson_named_dot_json_is_still_treated_as_data(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {},
         "geometry": {"type": "Point", "coordinates": [1.0, 2.0]}}]}))
    meta = extract_dataset_metadata(str(path))
    assert meta.get("bounds") == [1.0, 2.0, 1.0, 2.0]


def test_an_fgdc_xml_extent_is_read(tmp_path):
    path = tmp_path / "meta.xml"
    path.write_text("<metadata><idinfo><spdom><bounding>"
                    "<westbc>-87.9</westbc><eastbc>-87.5</eastbc>"
                    "<northbc>42.0</northbc><southbc>41.6</southbc>"
                    "</bounding></spdom></idinfo></metadata>", encoding="utf-8")
    meta = extract_dataset_metadata(str(path))
    assert meta["bounds"] == [-87.9, 41.6, -87.5, 42.0]


def test_a_tar_archive_lists_its_members(tmp_path):
    """.tar/.tgz/.gz routed to a zip-only reader and always said 'could not read container'."""
    import tarfile

    payload = tmp_path / "inner.csv"
    payload.write_text("a,b\n1,2\n", encoding="utf-8")
    archive = tmp_path / "bundle.tar"
    with tarfile.open(archive, "w") as tf:
        tf.add(payload, arcname="inner.csv")
    meta = extract_dataset_metadata(str(archive))
    assert meta["format"] == "tar"
    assert meta["member_count"] == 1
    assert meta["member_families"] == {"tabular": 1}


def test_a_zip_still_lists_its_members(tmp_path):
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("layer.shp", "x")
        zf.writestr("layer.dbf", "x")
    meta = extract_dataset_metadata(str(archive))
    assert meta["member_count"] == 2
    assert meta["member_families"].get("vector") == 1


def test_an_unreadable_archive_reports_rather_than_raises(tmp_path):
    path = tmp_path / "broken.tar"
    path.write_bytes(b"not an archive")
    meta = extract_dataset_metadata(str(path))
    assert "note" in meta


def test_a_binary_format_without_pandas_says_so_instead_of_guessing(tmp_path, monkeypatch):
    """The fallback must NOT be csv.reader for a spreadsheet — that is the garbage-header bug."""
    monkeypatch.setattr("extractors.data_extractor._read_tabular", lambda p: None)
    path = tmp_path / "t.xlsx"
    path.write_bytes(b"PK\x03\x04binary")
    meta = extract_dataset_metadata(str(path))
    assert "requires pandas" in meta.get("note", "")
    assert "schema" not in meta
