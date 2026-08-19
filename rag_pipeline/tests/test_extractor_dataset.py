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


# ------------------------------------------------------------------ generated loaders (M7)

from extractors.data_extractor import build_loader_unit  # noqa: E402


def _loader(family, fmt="GeoJSON", crs="EPSG:4326", title="Chicago Communities", **extra):
    meta = {"family": family, "format": fmt, "crs": crs, **extra}
    return build_loader_unit(meta, title=title, rel_path="f.dat",
                             provenance={"element_id": "ds01"})


@pytest.mark.parametrize("family,pkg,returns", [
    ("vector", "geopandas", "GeoDataFrame"),
    ("raster", "rasterio", "rasterio.DatasetReader"),
    ("tabular", "pandas", "DataFrame"),
])
def test_a_loader_is_generated_per_family(family, pkg, returns):
    unit = _loader(family)
    assert unit["requirements"]["pip"] == [pkg]
    assert unit["returns"] == returns


def test_the_generated_source_actually_executes():
    """Generated code has to be compiled by whatever generates it — there is no reviewer in
    this path. The first version assembled the docstring with a conditional indent that skipped
    lines starting with a triple quote, un-indenting the docstring and making every loader a
    SyntaxError."""
    unit = _loader("vector")
    ns = {}
    exec(compile(unit["source"], "<loader>", "exec"), ns)
    assert unit["symbol"] in ns and callable(ns[unit["symbol"]])


def test_a_family_with_no_sensible_reader_yields_none():
    """A container or a metadata sidecar gets no loader, rather than one that cannot work."""
    assert _loader("container", fmt="zip") is None
    assert _loader("metadata", fmt="STAC") is None


def test_the_loader_takes_a_staged_path_not_a_url():
    """Staging happens agent-side; no URL, no bucket, no credential reaches the sandbox. That
    is why --network none can stay closed."""
    unit = _loader("vector")
    assert unit["signature"] == f"def {unit['symbol']}(staged_path)"
    assert "http" not in unit["source"]
    assert "bucket" not in unit["source"] and "boto3" not in unit["source"]


def test_a_declared_crs_is_set_only_when_missing():
    """The file is authoritative about its own projection; silently reassigning it would be
    the same class of error the invariant gate exists to catch."""
    source = _loader("vector", crs="EPSG:4326")["source"]
    assert 'if getattr(frame, "crs", None) is None' in source
    assert 'set_crs("EPSG:4326")' in source


def test_no_crs_means_no_set_crs_call():
    assert "set_crs" not in _loader("vector", crs="")["source"]


def test_xlsx_and_parquet_get_the_right_reader():
    assert "read_excel" in _loader("tabular", fmt="XLSX")["source"]
    assert "read_parquet" in _loader("tabular", fmt="GeoParquet")["source"]
    assert "read_csv" in _loader("tabular", fmt="CSV")["source"]


def test_the_loader_docstring_carries_provenance():
    source = _loader("vector", schema=["id", "name"])["source"]
    assert "Source element : ds01" in source
    assert "Declared CRS   : EPSG:4326" in source
    assert "id, name" in source


@pytest.mark.parametrize("title,expected_prefix", [
    ("Chicago Communities", "load_chicago_communities"),
    ("2024 Survey Points", "load_ds_2024"),
    ("weird!! name??", "load_weird_name"),
    # An empty title falls back to the FILENAME, which is more informative than a generic
    # "load_dataset" — several platform datasets carry no title at all.
    ("", "load_f_dat"),
])
def test_symbol_names_are_safe_identifiers(title, expected_prefix):
    unit = _loader("vector", title=title)
    assert unit["symbol"].startswith(expected_prefix)
    assert unit["symbol"].isidentifier()


# ------------------------------------------------------------------ the loader as an asset

def test_a_dataset_emits_a_loader_method_unit(tmp_path):
    from extractors.base import EMIT_LIBRARY

    path = tmp_path / "communities.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": "Loop"},
         "geometry": {"type": "Point", "coordinates": [-87.63, 41.88]}}]}))
    result = _extract(path, title="Chicago Communities")
    units = [a for a in result.assets if getattr(a, "unit", None)]
    assert len(units) == 1
    unit = units[0]
    assert unit.unit["library_symbol"] == "load_chicago_communities"
    assert unit.unit["callability"]["verdict"] == "callable"
    assert EMIT_LIBRARY in unit.emit_targets
    ns = {}
    exec(compile(unit.slice_source, "<loader>", "exec"), ns)
    assert "load_chicago_communities" in ns


def test_the_loader_unit_declares_its_crs_invariant(tmp_path):
    """So the gate has something real to check on the frame this returns."""
    path = tmp_path / "x.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {},
         "geometry": {"type": "Point", "coordinates": [1.0, 2.0]}}]}))
    unit = [a for a in _extract(path, title="X").assets if getattr(a, "unit", None)][0]
    assert unit.unit["invariants"] == [
        {"check": "crs_equals", "target": "return", "args": {"crs": "EPSG:4326"}}]


def test_a_container_dataset_now_emits_a_loader_for_its_primary_member(tmp_path):
    """This asserted the opposite until archives could be unpacked safely.

    Refusing to describe an archive was the right call while extraction meant writing to a path
    that came out of one — but 18 of the corpus's 30 fetchable datasets are ZIPs, so it left the
    majority of the type as an unreadable blob. With the guards in extractors/archives.py the
    primary member can be described, and the loader points at it.
    """
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("sites.csv", "name,latitude,longitude\na,41.9,-87.6\n")
        zf.writestr("readme.txt", "notes")
    result = _extract(archive, title="Bundle")

    units = [a for a in result.assets if getattr(a, "unit", None)]
    assert units, "a zipped dataset should now yield a loader"
    doc = result.assets[0].extracted
    assert doc["format"] == "zip", "the archive's own format is still recorded"
    assert doc["primary_member"] == "sites.csv"


def test_an_archive_whose_members_are_all_refused_emits_no_loader(tmp_path):
    """Nothing safe to describe means nothing to load — and the reason is recorded."""
    archive = tmp_path / "hostile.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "x")
    result = _extract(archive, title="Hostile")
    assert [a for a in result.assets if getattr(a, "unit", None)] == []
    assert "archive_note" in result.assets[0].extracted


def test_the_loader_is_content_addressed(tmp_path):
    """Same scheme as an extracted slice, so re-ingesting an unchanged dataset is a no-op."""
    path = tmp_path / "x.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {},
         "geometry": {"type": "Point", "coordinates": [1.0, 2.0]}}]}))
    a = [x for x in _extract(path, title="X").assets if getattr(x, "unit", None)][0]
    b = [x for x in _extract(path, title="X").assets if getattr(x, "unit", None)][0]
    assert a.unit["slice_sha"] == b.unit["slice_sha"] and len(a.unit["slice_sha"]) == 12


# ------------------------------------------------------------- everything a handler measured

def test_every_field_a_handler_measures_reaches_the_document():
    """The asset's `extracted` payload used a WHITELIST, and it lost the same information twice.

    Version one carried eight hardcoded fields, so a dataset document said only
    "GeoJSON, vector, 566 bytes". That was replaced by a nineteen-field list — which still dropped
    every field the list's author had not personally needed. Measured:

      * a GeoTIFF's ``resolution``, ``bands`` and ``dtypes`` were computed by ``_handle_raster``
        and discarded, so "what resolution is this raster" was unanswerable from the index;
      * a NetCDF's ``variables`` and ``dims`` likewise. The corpus's groundwater-policy dataset
        computed ``variables: ['crs', 'WAT4_QWATGRD'], dims: {latitude: 288, longitude: 690}`` and
        indexed ``{format: nc, family: raster, size_bytes: 803380}``. For a NetCDF the variable
        list IS the schema, so that type indexed no schema at all.

    A whitelist fails silently and in the same direction every time a handler learns something new.
    """
    from extractors.data_extractor import _describable

    measured = {"format": "GeoTIFF", "crs": "EPSG:32616", "bounds": [0, 0, 1, 1],
                "resolution": [30.0, 30.0], "bands": 3, "dtypes": ["uint16"],
                "variables": ["WAT4_QWATGRD"], "dims": {"latitude": 288},
                "some_future_field_nobody_has_written_yet": 42}
    out = _describable(measured)
    for key, value in measured.items():
        assert out[key] == value, f"{key} was dropped"


def test_a_handler_cannot_redefine_the_documents_identity():
    """The denylist keeps the property the whitelist was protecting: a handler that produced a key
    like `doc_id` or `contents` would silently overwrite the record."""
    from extractors.data_extractor import _describable

    out = _describable({"format": "CSV", "doc_id": "HIJACKED", "contents": "HIJACKED",
                        "parent_doc_id": "HIJACKED", "unit": {"x": 1}})
    assert out["format"] == "CSV"
    for reserved in ("doc_id", "contents", "parent_doc_id", "unit"):
        assert reserved not in out


def test_a_refused_key_is_named_rather_than_silently_dropped():
    """The failure mode being replaced was silence. A denylist that also went quiet would only
    move the problem."""
    from extractors.data_extractor import _describable

    out = _describable({"format": "CSV", "doc_id": "x", "spatial": {}})
    assert out["handler_keys_refused"] == ["doc_id", "spatial"]


def test_a_clean_handler_result_records_nothing_refused():
    from extractors.data_extractor import _describable

    assert "handler_keys_refused" not in _describable({"format": "CSV", "row_count": 3})


def test_a_real_geotiff_carries_its_resolution_bands_and_dtypes(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    numpy = pytest.importorskip("numpy")
    from rasterio.transform import from_origin

    from extractors.base import EMIT_OPENSEARCH, ExtractContext
    from extractors.data_extractor import DataExtractor

    path = tmp_path / "probe.tif"
    with rasterio.open(path, "w", driver="GTiff", height=20, width=30, count=3, dtype="uint16",
                       crs="EPSG:32616", transform=from_origin(400000, 4600000, 30, 30)) as dst:
        for band in range(1, 4):
            dst.write(numpy.full((20, 30), band, dtype="uint16"), band)

    ctx = ExtractContext(element_id="tif1", element_type="dataset",
                         fields={"title": "probe raster"}, targets=[EMIT_OPENSEARCH])
    asset = [a for a in DataExtractor().extract(str(path), ctx=ctx).assets
             if a.kind == "dataset"][0]
    assert asset.extracted["resolution"] == [30.0, 30.0]
    assert asset.extracted["bands"] == 3
    assert asset.extracted["dtypes"] == ["uint16", "uint16", "uint16"]
    assert asset.extracted["crs"] == "EPSG:32616"
