"""CSV/TSV coordinate tables must be usable as point layers — including DMS coordinates.

Live failure: uploading a CSV of DMS coordinates made vector_to_geojson raise
"'DataFrame' object has no attribute 'to_file'" and inspect_vector report no geometry/CRS, so the
agent concluded the file was unprocessable (blaming missing pyarrow/fastparquet).
"""

from __future__ import annotations

import json

import pytest

from agent_runtime.langchain_geo_tools import (
    dataframe_to_points,
    make_langchain_geo_tools,
    parse_coordinate,
    read_vector,
)

DMS_ROWS = [
    ("24°07'12.0\"N", "121°14'22.0\"E"),   # Taiwan
    ("21°13'34.2\"S", "47°22'11.4\"E"),    # Madagascar
    ("19°59'00.0\"N", "97°11'00.0\"W"),    # Mexico
]


def test_parse_coordinate_decimal_and_dms():
    assert parse_coordinate(40.68) == 40.68
    assert parse_coordinate("-97.1833") == pytest.approx(-97.1833)
    assert parse_coordinate("24°07'12.0\"N") == pytest.approx(24.12)
    assert parse_coordinate("121°14'22.0\"E") == pytest.approx(121.23944, abs=1e-5)
    assert parse_coordinate("21°13'34.2\"S") == pytest.approx(-21.22617, abs=1e-5)   # S negates
    assert parse_coordinate("97°11'00.0\"W") == pytest.approx(-97.18333, abs=1e-5)   # W negates
    assert parse_coordinate("46°46.5'N") == pytest.approx(46.775)                    # degrees+minutes
    assert parse_coordinate("bogus") is None and parse_coordinate("") is None
    assert parse_coordinate(None) is None


def test_dataframe_to_points_builds_wgs84_points():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"Latitude": [r[0] for r in DMS_ROWS],
                       "Longitude": [r[1] for r in DMS_ROWS],
                       "qPCR": ["pos"] * 3})
    gdf = dataframe_to_points(df)
    assert str(gdf.crs) == "EPSG:4326"
    assert list(gdf.geom_type.unique()) == ["Point"]
    assert len(gdf) == 3
    assert gdf["qPCR"].tolist() == ["pos"] * 3          # attributes preserved
    assert gdf.geometry.iloc[0].y == pytest.approx(24.12)
    assert gdf.geometry.iloc[2].x == pytest.approx(-97.18333, abs=1e-5)


def test_dataframe_to_points_reports_clear_errors():
    pd = pytest.importorskip("pandas")
    with pytest.raises(ValueError, match="no recognizable coordinate columns"):
        dataframe_to_points(pd.DataFrame({"site": ["a"], "qPCR": ["pos"]}))
    with pytest.raises(ValueError, match="could not parse any coordinates"):
        dataframe_to_points(pd.DataFrame({"lat": ["x"], "lon": ["y"]}))


def test_alternate_column_names_and_tsv(tmp_path):
    pytest.importorskip("pandas")
    csv = tmp_path / "sites.csv"
    csv.write_text("site_lat,site_long,label\n40.68,-105.77,a\n45.99,-81.94,b\n")
    gdf = read_vector(str(csv))
    assert len(gdf) == 2 and str(gdf.crs) == "EPSG:4326"
    tsv = tmp_path / "sites.tsv"
    tsv.write_text("Latitude\tLongitude\n24°07'12.0\"N\t121°14'22.0\"E\n")
    assert len(read_vector(str(tsv))) == 1


def test_geo_tools_end_to_end_on_a_dms_csv(tmp_path, monkeypatch):
    """inspect -> geojson -> plot all succeed on the reported CSV shape."""
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    monkeypatch.delenv("AGENT_PUBLIC_BASE_URL", raising=False)
    pytest.importorskip("geopandas")
    from werkzeug.datastructures import FileStorage
    from agent_runtime.file_store import save_uploaded_file

    src = tmp_path / "qpcr.csv"
    src.write_text("Latitude,Longitude,qPCR\n" + "".join(f"{a},{b},pos\n" for a, b in DMS_ROWS))
    with src.open("rb") as fh:
        fid = save_uploaded_file(FileStorage(stream=fh, filename="qpcr.csv"))["file_id"]

    tools = {t.name: t for t in make_langchain_geo_tools(default_input_file_ids=[fid])}
    info = json.loads(tools["inspect_vector"].invoke({"file_id": fid}))
    assert info["ok"] and info["geometry_type"] == "Point" and info["crs"] == "EPSG:4326"
    assert info["feature_count"] == 3 and "derived from coordinate columns" in info["geometry_source"]

    gj = json.loads(tools["vector_to_geojson"].invoke({"file_id": fid, "output_filename": "o.geojson"}))
    assert gj["ok"] and gj.get("feature_count") == 3 and gj.get("download_url")

    png = json.loads(tools["render_map_image"].invoke({"file_id": fid, "title": "qPCR"}))
    assert png["ok"] and str(png.get("filename", "")).endswith(".png")
