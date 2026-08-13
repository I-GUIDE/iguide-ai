"""Dataset / metadata extractor (#2) — webhook (upload) path.

Dispatches an uploaded data file by family to a handler and emits ONE Dataset
AssetRecord: `crs`, `spatial-bounding-box-geojson`, and an `extracted` block
(format, size, + family fields: resolution/bands/variables for raster;
schema/geometry/feature_count/layers for vector; columns/row_count for tabular).
Index-only; never executable.

Heavy GIS libs (rasterio/xarray/fiona/geopandas) are OPTIONAL: stdlib handlers cover
GeoJSON/CSV/zip; raster/vector handlers try the lib and degrade with a note if it's
absent or the file can't be read. See EXTRACTOR_DESIGN.md §7.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    EMIT_OPENSEARCH,
    KIND_DATASET,
    AssetRecord,
    ExtractContext,
    Extractor,
    ExtractionResult,
)
from .doc_ids import dataset_doc_id, resource_type_for
from .fileclass import CONTAINER_EXT, RASTER_EXT, TABULAR_EXT, VECTOR_EXT

logger = logging.getLogger(__name__)

_LATS = {"lat", "latitude", "y"}
_LONS = {"lon", "lng", "long", "longitude", "x"}


# _envelope used to live here and wrote the file's NATIVE bounds straight into the
# `spatial-bounding-box-geojson` geo_shape field. Because the index maps that field with
# `ignore_malformed: true`, writing UTM metres did not fail the write — OpenSearch silently
# dropped the field and the document indexed cleanly, so the dataset was simply absent from
# every spatial query with nothing recording why. Measured on the live index: 181 of 619 docs
# carry a bbox. Reprojection now goes through rag_pipeline.search.geo_shapes.bbox_geo_shape,
# which returns None (plus a note) rather than guessing.


def _iter_coords(obj: Any):
    """Yield [x, y] pairs from arbitrary GeoJSON coordinate nesting."""
    if isinstance(obj, (list, tuple)):
        if len(obj) >= 2 and all(isinstance(v, (int, float)) for v in obj[:2]):
            yield obj[0], obj[1]
        else:
            for item in obj:
                yield from _iter_coords(item)


def _handle_geojson(path: str) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    feats = data.get("features") if isinstance(data, dict) else None
    feats = feats if isinstance(feats, list) else ([data] if isinstance(data, dict) else [])
    xs: List[float] = []
    ys: List[float] = []
    geom_types: set = set()
    prop_keys: set = set()
    for f in feats:
        geom = (f or {}).get("geometry") or {}
        if geom.get("type"):
            geom_types.add(geom["type"])
        for x, y in _iter_coords(geom.get("coordinates")):
            xs.append(x); ys.append(y)
        for k in ((f or {}).get("properties") or {}):
            prop_keys.add(k)
    meta: Dict[str, Any] = {"format": "GeoJSON", "feature_count": len(feats),
                            "geometry_types": sorted(geom_types), "schema": sorted(prop_keys),
                            "crs": "EPSG:4326"}
    if xs and ys:
        meta["bounds"] = [min(xs), min(ys), max(xs), max(ys)]
    return meta


def _read_tabular(path: str) -> Optional[Any]:
    """A DataFrame for a tabular file, or None when pandas is unavailable.

    `.xlsx`/`.xls` MUST come through pandas. They were routed to `csv.reader`, which happily
    reads the binary container and returns a garbage single-column header with **no exception**
    — the worst possible outcome, because the asset indexes with a nonsense schema that looks
    like real metadata.
    """
    ext = Path(path).suffix.lower()
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return None
    try:
        if ext in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if ext in {".parquet", ".geoparquet"}:
            return pd.read_parquet(path)
        return pd.read_csv(path, sep=None, engine="python", nrows=20000)
    except Exception:
        return None


def _bounds_from_frame(df: Any) -> Dict[str, Any]:
    """Coordinate columns via the TESTED helpers, not a second hand-rolled scan.

    ``_pick_coord_column`` / ``parse_coordinate`` already handle DMS, hemisphere suffixes and
    the many column spellings, and have coverage in ``test_tabular_points.py``. The local
    ``_LATS``/``_LONS`` sets recognised eight names and float() only, so a table with
    ``Latitude (N)`` or ``41°52'N`` produced no bbox at all.
    """
    out: Dict[str, Any] = {}
    try:
        from agent_runtime.langchain_geo_tools import (_LAT_KEYS, _LON_KEYS,
                                                      _pick_coord_column, parse_coordinate)
    except Exception:
        return out
    columns = [str(c) for c in getattr(df, "columns", [])]
    lat_col = _pick_coord_column(columns, _LAT_KEYS)
    lon_col = _pick_coord_column(columns, _LON_KEYS)
    if not (lat_col and lon_col):
        return out
    xs: List[float] = []
    ys: List[float] = []
    for lat_raw, lon_raw in zip(df[lat_col].tolist()[:5000], df[lon_col].tolist()[:5000]):
        lat = parse_coordinate(lat_raw)
        lon = parse_coordinate(lon_raw)
        if lat is not None and lon is not None:
            ys.append(lat)
            xs.append(lon)
    if xs and ys:
        out["bounds"] = [min(xs), min(ys), max(xs), max(ys)]
        out["crs"] = "EPSG:4326"          # parse_coordinate yields degrees by construction
        out["geometry_from"] = {"lon": lon_col, "lat": lat_col}
        out["coordinate_rows"] = len(xs)
    return out


def _handle_tabular(path: str) -> Dict[str, Any]:
    ext = Path(path).suffix.lower()
    fmt = {".csv": "CSV", ".tsv": "TSV", ".xlsx": "XLSX", ".xls": "XLS",
           ".parquet": "Parquet", ".geoparquet": "GeoParquet"}.get(ext, ext.lstrip("."))
    df = _read_tabular(path)
    if df is not None:
        meta: Dict[str, Any] = {"format": fmt,
                                "schema": [str(c) for c in df.columns],
                                "row_count": int(len(df))}
        meta.update(_bounds_from_frame(df))
        return meta

    # stdlib fallback, TEXT formats only. Falling back to csv.reader for a spreadsheet is what
    # produced the garbage header, so a binary format without pandas reports that plainly.
    if ext in {".xlsx", ".xls", ".parquet", ".geoparquet"}:
        return {"format": fmt,
                "note": f"{fmt} requires pandas (and pyarrow for parquet); not installed, "
                        f"so no schema was read"}
    delim = "\t" if ext == ".tsv" else ","
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.reader(fh, delimiter=delim))
    if not rows:
        return {"format": fmt, "row_count": 0, "schema": []}
    return {"format": fmt, "schema": rows[0], "row_count": len(rows) - 1,
            "note": "pandas unavailable; header parsed with the stdlib csv reader"}


def _handle_raster(path: str) -> Dict[str, Any]:
    ext = Path(path).suffix.lower()
    try:
        if ext in {".nc", ".hdf", ".h5", ".he5", ".grib", ".grb", ".grib2", ".zarr"}:
            import xarray as xr  # type: ignore
            ds = xr.open_dataset(path)
            return {"format": ext.lstrip("."), "variables": list(map(str, ds.data_vars)),
                    "dims": {k: int(v) for k, v in ds.dims.items()}}
        import rasterio  # type: ignore
        with rasterio.open(path) as src:
            b = src.bounds
            return {"format": "GeoTIFF" if ext in {".tif", ".tiff"} else ext.lstrip("."),
                    "crs": str(src.crs), "bounds": [b.left, b.bottom, b.right, b.top],
                    "resolution": list(src.res), "bands": src.count,
                    "dtypes": [str(d) for d in src.dtypes]}
    except Exception as exc:
        return {"format": ext.lstrip("."), "note": f"raster reader unavailable/failed: {type(exc).__name__}: {exc}"}


def _handle_vector(path: str) -> Dict[str, Any]:
    ext = Path(path).suffix.lower()
    if ext in {".geojson", ".json"}:
        return _handle_geojson(path)
    if ext in {".parquet", ".geoparquet"}:
        # fiona cannot read parquet. Try geopandas (which carries the CRS and geometry), then
        # fall back to the plain tabular reader so at least the schema is recorded.
        try:
            import geopandas as gpd  # type: ignore
            gdf = gpd.read_parquet(path)
            b = list(gdf.total_bounds)
            return {"format": "GeoParquet", "crs": str(gdf.crs), "bounds": b,
                    "geometry_type": (str(gdf.geom_type.iloc[0]) if len(gdf) else None),
                    "schema": [str(c) for c in gdf.columns if c != "geometry"],
                    "feature_count": int(len(gdf))}
        except Exception:
            return _handle_tabular(path)
    try:
        import fiona  # type: ignore
        layers = fiona.listlayers(path)
        with fiona.open(path) as src:
            return {"format": src.driver, "crs": str(src.crs), "bounds": list(src.bounds),
                    "geometry_type": src.schema.get("geometry"),
                    "schema": list((src.schema.get("properties") or {}).keys()),
                    "feature_count": len(src), "layers": layers}
    except Exception as exc:
        return {"format": ext.lstrip("."), "note": f"vector reader unavailable/failed: {type(exc).__name__}: {exc}"}



def _handle_metadata(path: str) -> Dict[str, Any]:
    """STAC / ISO / FGDC sidecars. Prefer the bbox the document DECLARES.

    A declared bbox is authoritative and already in degrees; deriving one from the data would
    be both less trustworthy and more work. `.json`/`.xml` previously routed to the "sidecar"
    family, which has no handler at all, so a STAC item carrying a perfectly good bbox was
    indexed with none.
    """
    ext = Path(path).suffix.lower()
    if ext == ".json":
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:
            return {"format": "json", "note": f"unreadable json: {type(exc).__name__}"}
        if not isinstance(doc, dict):
            return {"format": "json"}
        if doc.get("type") in {"Feature", "FeatureCollection"} or "features" in doc:
            return _handle_geojson(path)          # it is data, not metadata
        meta: Dict[str, Any] = {"format": "STAC" if doc.get("stac_version") else "json"}
        bbox = doc.get("bbox") or ((doc.get("extent") or {}).get("spatial") or {}).get("bbox")
        if isinstance(bbox, list) and bbox and isinstance(bbox[0], list):
            bbox = bbox[0]                        # STAC collections nest one level
        if isinstance(bbox, list) and len(bbox) >= 4:
            try:
                meta["bounds"] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                meta["crs"] = "EPSG:4326"         # STAC bbox is WGS84 by specification
                meta["bbox_from"] = "declared"
            except (TypeError, ValueError):
                pass
        for key in ("id", "title", "description", "license"):
            if isinstance(doc.get(key), str):
                meta[key] = doc[key]
        if isinstance(doc.get("properties"), dict):
            meta["schema"] = sorted(str(k) for k in doc["properties"])[:60]
        return meta

    try:
        import xml.etree.ElementTree as ET
        root = ET.parse(path).getroot()
    except Exception as exc:
        return {"format": "xml", "note": f"unreadable xml: {type(exc).__name__}"}
    meta = {"format": "xml", "root": str(root.tag).rsplit("}", 1)[-1]}
    # ISO 19115 and FGDC spell the same four numbers differently; read whichever appears.
    found: Dict[str, float] = {}
    for node in root.iter():
        tag = str(node.tag).rsplit("}", 1)[-1].lower()
        if tag in {"westbc", "eastbc", "northbc", "southbc", "westboundlongitude",
                   "eastboundlongitude", "northboundlatitude", "southboundlatitude"} \
                and tag not in found and (node.text or "").strip():
            try:
                found[tag] = float(node.text.strip())
            except ValueError:
                pass
    west = found.get("westbc", found.get("westboundlongitude"))
    east = found.get("eastbc", found.get("eastboundlongitude"))
    north = found.get("northbc", found.get("northboundlatitude"))
    south = found.get("southbc", found.get("southboundlatitude"))
    if None not in (west, east, north, south):
        meta["bounds"] = [west, south, east, north]
        meta["crs"] = "EPSG:4326"
        meta["bbox_from"] = "declared"
    return meta


def _handle_container(path: str) -> Dict[str, Any]:
    """zip / tar / tgz / gz. Members are listed, never extracted.

    `.tar`, `.tgz` and `.gz` were routed to a zip-only reader and so always reported
    "could not read container". Listing without extracting also sidesteps zip-slip entirely:
    nothing is ever written to a path derived from an archive entry.
    """
    ext = Path(path).suffix.lower()
    members: List[str] = []
    fmt = "zip"
    try:
        if ext == ".zip":
            with zipfile.ZipFile(path) as zf:
                members = [n for n in zf.namelist() if not n.endswith("/")]
        else:
            import tarfile
            fmt = "tar" if ext == ".tar" else ext.lstrip(".")
            if tarfile.is_tarfile(path):
                with tarfile.open(path) as tf:
                    members = [m.name for m in tf.getmembers() if m.isfile()]
            elif ext == ".gz":
                import gzip
                with gzip.open(path, "rb") as fh:
                    fh.read(1)                    # confirm it really is gzip
                inner = Path(path).stem
                return {"format": "gzip", "member_count": 1, "members": [inner],
                        "member_families": {family_for_ext(Path(inner).suffix.lower()): 1}}
            else:
                return {"format": fmt, "note": "not a recognised archive"}
    except Exception as exc:
        return {"format": fmt, "note": f"could not read container: {type(exc).__name__}: {exc}"}
    fams: Dict[str, int] = {}
    for m in members:
        fam = family_for_ext(Path(m).suffix.lower())
        fams[fam] = fams.get(fam, 0) + 1
    return {"format": fmt, "member_count": len(members), "member_families": fams,
            "members": members[:50]}


def family_for_ext(ext: str) -> str:
    ext = ext.lower()
    # .json/.xml are checked FIRST: both appear in SIDECAR_EXT (which had no handler) and .json
    # is also a vector extension, so a STAC item was classified as an unhandled sidecar.
    if ext in {".json", ".xml"}:
        return "metadata"
    if ext in RASTER_EXT:
        return "raster"
    if ext in VECTOR_EXT:
        return "vector"
    if ext in TABULAR_EXT:
        return "tabular"
    if ext in CONTAINER_EXT:
        return "container"
    return "sidecar"


# Handler names, resolved at CALL time. Binding the function objects here froze them at import,
# so a test (or a caller) replacing `_handle_raster` on the module had no effect at all — the
# dict still held the original. Dispatching by name keeps one source of truth.
_HANDLERS = {"raster": "_handle_raster", "vector": "_handle_vector",
             "tabular": "_handle_tabular", "container": "_handle_container",
             "metadata": "_handle_metadata"}


def _handler_for(family: str):
    name = _HANDLERS.get(family)
    return globals().get(name) if name else None


def extract_dataset_metadata(path: str) -> Dict[str, Any]:
    ext = Path(path).suffix.lower()
    family = family_for_ext(ext)
    handler = _handler_for(family)
    meta: Dict[str, Any] = {"family": family, "ext": ext}
    if handler:
        try:
            meta.update(handler(path))
        except Exception as exc:
            meta["note"] = f"{family} handler failed: {type(exc).__name__}: {exc}"
    else:
        meta["format"] = ext.lstrip(".") or "unknown"
    try:
        meta["size_bytes"] = os.path.getsize(path)
    except OSError:
        pass
    return meta



def _sha_of(source: str) -> str:
    """Content address for a generated slice — same scheme as an extracted one."""
    import hashlib
    return hashlib.sha1((source or "").encode("utf-8")).hexdigest()[:12]


def _slug(text: str) -> str:
    """A safe python identifier fragment from a title/filename."""
    import re as _re
    out = _re.sub(r"[^0-9a-zA-Z]+", "_", str(text or "")).strip("_").lower()
    out = _re.sub(r"_+", "_", out)[:48]
    if not out or out[0].isdigit():
        out = f"ds_{out}" if out else "dataset"
    return out


def build_loader_unit(meta: Dict[str, Any], *, title: str, rel_path: str,
                      provenance: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A generated ``load_<slug>(staged_path)`` unit for a dataset, or None.

    Not synthetic filler: this is exactly the staging step several eval tasks declined, and it
    is what lets the agent open a platform dataset without inventing a reader or guessing a
    CRS. Three properties make it worth generating rather than leaving to the model:

    * **the reader is chosen from what extraction actually observed** — raster / vector /
      tabular — instead of the model guessing from a file extension;
    * **the declared CRS travels with the loader**, so a frame that arrives without one is set
      explicitly and the invariant gate has something real to check;
    * **it takes a STAGED LOCAL PATH.** No URL, no bucket, no credential. Staging happens
      agent-side, which is why the sandbox can keep ``--network none``.

    Returns None for a family with no sensible reader (a container, a metadata sidecar) rather
    than emitting a loader that cannot work.
    """
    family = meta.get("family")
    readers = {
        "raster": ("rasterio", "import rasterio",
                   "return rasterio.open(staged_path)"),
        "vector": ("geopandas", "import geopandas as gpd",
                   "return gpd.read_file(staged_path)"),
        "tabular": ("pandas", "import pandas as pd",
                    "return pd.read_csv(staged_path)"),
    }
    if family not in readers:
        return None
    pkg, import_line, read_line = readers[family]
    if family == "tabular" and str(meta.get("format", "")).upper() in {"XLSX", "XLS"}:
        read_line = "return pd.read_excel(staged_path)"
    elif family == "tabular" and "arquet" in str(meta.get("format", "")):
        read_line = "return pd.read_parquet(staged_path)"

    symbol = f"load_{_slug(title or rel_path)}"
    crs = str(meta.get("crs") or "").strip()
    schema = meta.get("schema") or meta.get("variables") or []
    docline = f"Load {title!r} ({meta.get('format') or family})."

    # Built line by line at a fixed indent and COMPILED before returning. The first version
    # assembled the docstring with a conditional indent that skipped lines starting with a
    # triple quote — which un-indented the docstring itself and made every generated loader a
    # SyntaxError. Generated code has to be compiled by whatever generates it; there is no
    # reviewer in this path.
    doc: List[str] = [f'    """{docline}', ""]
    doc.append(f"    Source element : {provenance.get('element_id')}")
    doc.append(f"    Source file    : {rel_path}")
    if crs:
        doc.append(f"    Declared CRS   : {crs}")
    if schema:
        doc.append(f"    Fields         : {', '.join(map(str, schema[:12]))}")
    doc.append('    """')

    read_expr = read_line[len("return "):]
    body: List[str] = [f"def {symbol}(staged_path):", *doc]
    if crs and family == "vector":
        # Sets a MISSING crs, never overrides a present one: the file is authoritative about
        # its own projection, and silently reassigning it would be the same class of error the
        # invariant gate exists to catch.
        body += [f"    frame = {read_expr}",
                 '    if getattr(frame, "crs", None) is None:',
                 f'        frame = frame.set_crs("{crs}")',
                 "    return frame"]
    else:
        body += [f"    return {read_expr}"]

    source = f"{import_line}\n\n\n" + "\n".join(body) + "\n"
    try:
        compile(source, f"<loader:{symbol}>", "exec")
    except SyntaxError as exc:
        logger.warning("generated loader for %s did not compile: %s", title, exc)
        return None

    return {"symbol": symbol, "source": source, "requirements": {"pip": [pkg]},
            "signature": f"def {symbol}(staged_path)",
            "doc_summary": docline,
            "returns": {"raster": "rasterio.DatasetReader", "vector": "GeoDataFrame",
                        "tabular": "DataFrame"}[family],
            "crs": crs, "family": family}


class DataExtractor:
    name = "dataset"

    def extract(self, path: str, *, ctx: ExtractContext) -> ExtractionResult:
        fname = os.path.basename(path)
        meta = extract_dataset_metadata(path)
        anchor = ctx.anchor() or fname
        doc_id = dataset_doc_id(anchor)
        f = ctx.fields or {}
        title = str(f.get("title") or fname)

        spatial: Dict[str, Any] = {}
        bbox_note = ""
        if meta.get("crs"):
            spatial["crs"] = meta["crs"]
        if meta.get("bounds"):
            from rag_pipeline.search.geo_shapes import bbox_geo_shape

            b = meta["bounds"]
            spatial["bounds"] = b                 # native bounds, kept for provenance
            shape, bbox_note = bbox_geo_shape(b, meta.get("crs"),
                                              is_raster=(meta.get("family") == "raster"))
            if shape is not None:
                spatial["spatial-bounding-box-geojson"] = shape
            # else: NO bbox field. Explicit absence beats a silently-dropped wrong value —
            # the field is mapped `ignore_malformed`, so a wrong bbox and no bbox are
            # indistinguishable at query time, and only the note records which happened.
        for k in ("resolution", "schema", "geometry_type", "feature_count", "variables"):
            if meta.get(k) is not None:
                spatial[k] = meta[k]

        fields_desc = ", ".join(map(str, meta.get("schema") or meta.get("variables") or []))
        contents = f"{title}\nformat={meta.get('format')} family={meta.get('family')}" + \
                   (f"\nfields: {fields_desc}" if fields_desc else "") + \
                   (f"\n{f.get('abstract')}" if f.get("abstract") else "")
        source_fields = {k: f[k] for k in ("authors", "contributor", "abstract", "tags", "license", "doi")
                         if f.get(k)}

        asset = AssetRecord(
            asset_id=doc_id, kind=KIND_DATASET, resource_type=resource_type_for(KIND_DATASET),
            doc_id=doc_id, emit_targets=[EMIT_OPENSEARCH], source_rel_path=fname,
            title=title, contents=contents.strip(),
            spatial=(spatial or None), source_fields=source_fields,
            extracted={"format": meta.get("format"), "family": meta.get("family"),
                       "size_bytes": meta.get("size_bytes"), "note": meta.get("note"),
                       "member_families": meta.get("member_families"),
                       "bbox_note": bbox_note or None,
                       "bbox_from": meta.get("bbox_from"),
                       "parent_type": "Dataset", "parent_title": title},
        )
        # A generated loader, emitted as a MethodUnit so it reaches iguide_methods through the
        # same path as an extracted notebook function. This is what makes `dataset` a type the
        # agent can OPEN rather than only read about.
        unit_asset = self._loader_asset(meta, doc_id=doc_id, title=title, rel_path=fname, ctx=ctx)

        warnings = [f"dataset: {meta['note']}"] if meta.get("note") else []
        # A dataset that HAS bounds but got no bbox is a coverage loss worth surfacing, not a
        # silent omission: it is the difference between "no spatial extent" and "we had one and
        # could not use it".
        if meta.get("bounds") and "spatial-bounding-box-geojson" not in spatial:
            warnings.append(f"dataset: no spatial bbox emitted — {bbox_note}")
        assets = [asset] + ([unit_asset] if unit_asset else [])
        return ExtractionResult(assets=assets, warnings=warnings)

    def _loader_asset(self, meta: Dict[str, Any], *, doc_id: str, title: str,
                      rel_path: str, ctx: ExtractContext) -> Optional[AssetRecord]:
        """Wrap a generated loader as a MethodUnit asset, or None when none applies."""
        import dataclasses

        from .base import EMIT_LIBRARY, KIND_METHOD_UNIT
        from .contracts import CALLABLE, ANALYZER_VERSION, Callability, UnitContract
        from .doc_ids import method_unit_doc_id

        provenance = {"element_id": ctx.anchor(), "parent_doc_id": doc_id,
                      "source_rel_path": rel_path, "commit_sha": ctx.commit_sha,
                      "extractor": self.name, "analyzer_version": ANALYZER_VERSION,
                      "generated": True}
        unit = build_loader_unit(meta, title=title, rel_path=rel_path, provenance=provenance)
        if not unit:
            return None
        contract = UnitContract(
            qualified_name=unit["symbol"], unit_kind="function",
            signature=unit["signature"], params=[],
            returns=unit["returns"], docstring=unit["doc_summary"],
            doc_summary=unit["doc_summary"],
            # GENERATED code, so callability is asserted by construction rather than analysed:
            # it has one parameter, imports exactly one library, and was compiled above. Running
            # the analyser over our own template would only re-derive that.
            callability=Callability(verdict=CALLABLE,
                                    reason="generated loader: single staged-path argument",
                                    requires_imports=list(unit["requirements"]["pip"])),
            slice_sha=_sha_of(unit["source"]),
            library_symbol=unit["symbol"],
            requirements=unit["requirements"],
            invariants=([{"check": "crs_equals", "target": "return",
                          "args": {"crs": unit["crs"]}}] if unit.get("crs") else []),
            provenance=provenance,
        )
        unit_doc_id = method_unit_doc_id(doc_id, unit["symbol"])
        targets = [EMIT_OPENSEARCH]
        if EMIT_LIBRARY in (ctx.targets or ()):
            targets.append(EMIT_LIBRARY)
        return AssetRecord(
            asset_id=unit_doc_id, kind=KIND_METHOD_UNIT,
            resource_type=resource_type_for(KIND_METHOD_UNIT), doc_id=unit_doc_id,
            emit_targets=targets, source_rel_path=rel_path,
            title=f"{unit['symbol']} — {title}",
            contents=f"{unit['signature']}\n\n{unit['doc_summary']}",
            unit=dataclasses.asdict(contract), slice_source=unit["source"],
            extracted={"parent_doc_id": doc_id, "parent_type": "Dataset",
                       "callable": True, "unit_name": unit["symbol"], "generated": True},
        )


_: Extractor = DataExtractor()  # type: ignore[assignment]

__all__ = ["DataExtractor", "extract_dataset_metadata", "family_for_ext"]
