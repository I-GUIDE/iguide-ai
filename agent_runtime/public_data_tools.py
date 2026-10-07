"""Download public data from approved hosts into the conversation's files, for code to read.

The code sandbox runs with ``--network none`` and keeps it. A container with network reaches
whatever its host can: the services beside it and the cloud provider's internal endpoints, not
just the public internet. And the code that would run there is written by a model that reads
web pages, documents and uploads, any of which can carry instructions. So data comes in through
the agent instead, one request at a time, and the request has to pass a gate:

- HTTPS GET only, to a host on the allowlist (``AGENT_PUBLIC_FETCH_HOSTS`` replaces the
  defaults). No credentials in the URL, no other port, no IP literal.
- Every address the host resolves to must be public (``ipaddress.is_global``). An allowlisted
  name that resolves to a private, loopback or link-local address is refused.
- Redirects are followed by hand, at most five, and each hop is checked like the first.
- At most ``AGENT_PUBLIC_FETCH_MAX_MB`` (default 50) per file, read in a stream and abandoned at
  the cap, and at most ``AGENT_PUBLIC_FETCH_PER_TURN`` (default 10) fetches per peer run.
- No proxy settings, ``.netrc`` or cookies from the environment: the agent's own credentials
  never ride along.

The download is saved as a file in the conversation, like any tool output, and ``execute_code``
reads it through ``input_files``. Off unless ``AGENT_PUBLIC_FETCH=1``.

What the gate does not do: it resolves the name before connecting and does not pin the
connection to the checked address. The window is a DNS answer changing between the check and
the connect, and only a host on the list could exploit it. The defaults are government and
open-data services, so the risk is accepted. Add a host only if you would trust its DNS.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import os
import socket
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlsplit

logger = logging.getLogger(__name__)

ENABLE_ENV = "AGENT_PUBLIC_FETCH"

#: Public data services the agent's analyses draw on. Exact host names. An entry starting with
#: "." also matches that domain's subdomains.
DEFAULT_HOSTS: Tuple[Tuple[str, str], ...] = (
    ("tigerweb.geo.census.gov", "Census TIGERweb: boundaries, places, tracts (ArcGIS REST)"),
    ("www2.census.gov", "Census files, including TIGER/Line shapefiles"),
    ("api.census.gov", "Census data API: ACS and decennial tables"),
    ("elevation.nationalmap.gov", "USGS 3DEP elevation (ImageServer exportImage)"),
    ("tnmaccess.nationalmap.gov", "USGS National Map product search"),
    ("earthquake.usgs.gov", "USGS earthquake catalog and feeds"),
    ("waterservices.usgs.gov", "USGS water data"),
    ("overpass-api.de", "OpenStreetMap features (Overpass API)"),
    ("data.cityofchicago.org", "City of Chicago open data (Socrata)"),
)

_MAX_REDIRECTS = 5
_CONNECT_TIMEOUT_S = 10
_READ_TIMEOUT_S = 60
_DEADLINE_S = 120
_SUMMARY_MAX_BYTES = 20 * 1024 * 1024  # parse JSON for a shape summary only below this
_USER_AGENT = "I-GUIDE-agent/1.0 (public data fetch; https://i-guide.io)"


class Refused(ValueError):
    """The request did not pass the gate. The message says why, in words a model can act on."""


def is_enabled() -> bool:
    # The registry reads the same switch to decide whether the decider is told about this tool.
    from agent_runtime.capability_registry import flag_on

    return flag_on(ENABLE_ENV)


def allowed_hosts() -> List[str]:
    raw = os.getenv("AGENT_PUBLIC_FETCH_HOSTS")
    if raw is None:
        return [h for h, _ in DEFAULT_HOSTS]
    return [h.strip().lower() for h in raw.split(",") if h.strip()]


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name) or default))
    except ValueError:
        return default


def _host_allowed(host: str, hosts: Sequence[str]) -> bool:
    for entry in hosts:
        if entry.startswith("."):
            if host.endswith(entry) or host == entry[1:]:
                return True
        elif host == entry:
            return True
    return False


def check_url(url: str, hosts: Optional[Sequence[str]] = None) -> Tuple[str, str]:
    """``(url, host)`` when the URL may be fetched; ``Refused`` otherwise."""
    hosts = list(hosts if hosts is not None else allowed_hosts())
    try:
        parts = urlsplit(str(url or "").strip())
        port = parts.port
    except ValueError as exc:
        raise Refused(f"not a valid URL: {exc}") from None
    if parts.scheme != "https":
        raise Refused("only https:// URLs are fetched")
    if parts.username or parts.password:
        raise Refused("a URL carrying credentials is not fetched")
    host = (parts.hostname or "").lower()
    if not host:
        raise Refused("the URL has no host")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise Refused("an IP address is not fetched; use the service's host name")
    if port not in (None, 443):
        raise Refused("only the standard https port is fetched")
    if not _host_allowed(host, hosts):
        raise Refused(f"{host} is not an approved source. Approved: {', '.join(hosts)}")
    return parts.geturl(), host


def resolve_public(host: str,
                   resolver: Callable[..., Any] = socket.getaddrinfo) -> List[str]:
    """Every address ``host`` resolves to, if all of them are public; ``Refused`` otherwise."""
    try:
        infos = resolver(host, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise Refused(f"{host} does not resolve: {exc}") from None
    addresses = sorted({info[4][0] for info in infos})
    if not addresses:
        raise Refused(f"{host} does not resolve")
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
        if not ip.is_global or ip.is_multicast:
            raise Refused(f"{host} resolves to a non-public address, so it is not fetched")
    return addresses


def _session():
    import requests

    session = requests.Session()
    session.trust_env = False      # no proxy variables, no .netrc: nothing of the agent's rides along
    session.headers.update({"User-Agent": _USER_AGENT, "Accept": "*/*"})
    return session


def _api_error(payload: Any) -> Optional[str]:
    """ArcGIS services (TIGERweb) answer HTTP 200 with an error object for a bad query."""
    if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
        err = payload["error"]
        details = "; ".join(str(d) for d in (err.get("details") or []) if d)
        return " ".join(x for x in (str(err.get("message") or "error"), details) if x)[:300]
    return None


def _shape(payload: Any) -> Dict[str, Any]:
    """What the data is shaped like, without its contents: keys, counts and geometry types."""
    if isinstance(payload, dict):
        out: Dict[str, Any] = {"json_keys": sorted(payload)[:20]}
        features = payload.get("features")
        if isinstance(features, list):
            out["features"] = len(features)
            types = set()
            for feature in features[:500]:
                geom = feature.get("geometry") if isinstance(feature, dict) else None
                if isinstance(geom, dict):
                    # GeoJSON names its type; ArcGIS JSON implies it by its keys.
                    kind = geom.get("type") or ("Polygon" if "rings" in geom else
                                                "LineString" if "paths" in geom else
                                                "Point" if "x" in geom else None)
                    if kind:
                        types.add(str(kind))
            if types:
                out["geometry_types"] = sorted(types)
            first = features[0] if features and isinstance(features[0], dict) else {}
            props = first.get("properties") or first.get("attributes") or {}
            if isinstance(props, dict):
                out["property_names"] = sorted(props)[:25]
        if "count" in payload:                                 # ArcGIS returnCountOnly
            out["count"] = payload["count"]
        if isinstance(payload.get("elements"), list):          # Overpass
            out["elements"] = len(payload["elements"])
        return out
    if isinstance(payload, list):
        return {"json_list_length": len(payload)}
    return {}


_PATH_NOISE = {"arcgis", "rest", "services", "mapserver", "imageserver", "featureserver",
               "query", "exportimage", "api", "interpreter", "resource"}

_DATA_SUFFIXES = {".geojson", ".json", ".csv", ".zip", ".tif", ".tiff", ".xml", ".txt", ".kml",
                  ".kmz", ".gpkg", ".parquet", ".nc", ".shp"}

_EXT_BY_TYPE = {"application/geo+json": ".geojson", "application/json": ".json",
                "text/csv": ".csv", "image/tiff": ".tif", "application/zip": ".zip",
                "application/x-zip-compressed": ".zip", "text/plain": ".txt",
                "application/xml": ".xml", "text/xml": ".xml"}


def _filename(url: str, content_type: str, requested: Optional[str]) -> str:
    from werkzeug.utils import secure_filename

    if requested:
        name = secure_filename(str(requested))
        if name:
            return name
    # Name it after what the path says it is. ArcGIS REST paths end in a verb ("query",
    # "exportImage") after the service and layer, so those, not the verb, make the name:
    # .../TIGERweb/State_County/MapServer/1/query -> State_County_1.
    parts = urlsplit(url)
    meaningful = [seg for seg in parts.path.split("/") if seg and seg.lower() not in _PATH_NOISE]
    if meaningful and "." in meaningful[-1]:
        name = secure_filename(meaningful[-1])          # already a file name
    else:
        name = secure_filename("_".join(meaningful[-2:])) if meaningful else ""
    name = name or secure_filename((parts.hostname or "download").split(".")[0]) or "download"
    ext = ".geojson" if "f=geojson" in url.lower() else _EXT_BY_TYPE.get(content_type, "")
    if ext and Path(name).suffix.lower() not in _DATA_SUFFIXES:
        name += ext
    return name


def fetch(url: str, *, filename: Optional[str] = None, max_bytes: Optional[int] = None,
          hosts: Optional[Sequence[str]] = None,
          resolver: Callable[..., Any] = socket.getaddrinfo,
          session_factory: Callable[[], Any] = _session) -> Dict[str, Any]:
    """Fetch ``url`` through the gate and save it as a conversation file. Raises ``Refused``."""
    from agent_runtime.file_store import create_output_file_from_path

    cap = max_bytes or _int_env("AGENT_PUBLIC_FETCH_MAX_MB", 50) * 1024 * 1024
    deadline = time.monotonic() + _DEADLINE_S
    session = session_factory()
    current = url
    for _hop in range(_MAX_REDIRECTS + 1):
        current, host = check_url(current, hosts)
        resolve_public(host, resolver)
        resp = session.get(current, allow_redirects=False, stream=True,
                           timeout=(_CONNECT_TIMEOUT_S, _READ_TIMEOUT_S))
        if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("Location"):
            current = urljoin(current, resp.headers["Location"])
            resp.close()
            continue
        break
    else:
        raise Refused(f"more than {_MAX_REDIRECTS} redirects")
    content_type = (resp.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
    size = 0
    tmp = tempfile.NamedTemporaryFile(prefix="public_fetch_", delete=False)
    try:
        if resp.status_code != 200:
            raise Refused(f"{host} answered HTTP {resp.status_code}")
        declared = resp.headers.get("Content-Length")
        if declared and declared.isdigit() and int(declared) > cap:
            raise Refused(f"the file is {int(declared) / 1e6:.1f} MB, over the {cap / 1e6:.0f} MB "
                          "limit; ask the service for less (a smaller area, fewer fields, a page)")
        with tmp:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                size += len(chunk)
                if size > cap:
                    raise Refused(f"the file is over the {cap / 1e6:.0f} MB limit; ask the "
                                  "service for less (a smaller area, fewer fields, a page)")
                if time.monotonic() > deadline:
                    raise Refused(f"the download took longer than {_DEADLINE_S} s")
                tmp.write(chunk)
        resp.close()
        shape: Dict[str, Any] = {}
        if size <= _SUMMARY_MAX_BYTES and ("json" in content_type or current.lower().endswith(
                (".json", ".geojson")) or "f=geojson" in current.lower() or "f=json" in current.lower()):
            try:
                payload = json.loads(Path(tmp.name).read_bytes())
            except (ValueError, UnicodeDecodeError):
                payload = None
            message = _api_error(payload)
            if message:
                raise Refused(f"{host} returned an error instead of data: {message}")
            shape = _shape(payload)
        record = create_output_file_from_path(tmp.name, filename=_filename(current, content_type, filename))
    finally:
        resp.close()
        tmp.close()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    logger.info("public fetch: %s%s -> %s (%d bytes)", host, urlsplit(current).path,
                record.get("file_id"), size)
    return {"ok": True, "file_id": record.get("file_id"), "filename": record.get("filename"),
            "size_bytes": size, "content_type": content_type or None, "source": current,
            "download_url": record.get("download_url"), "shape": shape,
            "next": "Pass this file_id in execute_code's input_files; the file is then in the "
                    "working directory under its filename."}


def make_public_data_tools(*, per_turn: Optional[int] = None) -> List[Any]:
    """``fetch_public_data``, or nothing when the gate is off (``AGENT_PUBLIC_FETCH``)."""
    if not is_enabled():
        return []
    from langchain_core.tools import StructuredTool

    from agent_runtime.tool_args import accept_null_defaults

    limit = per_turn or _int_env("AGENT_PUBLIC_FETCH_PER_TURN", 10)
    used = {"n": 0}
    hosts = allowed_hosts()
    listed = "; ".join(f"{h} ({why})" for h, why in DEFAULT_HOSTS if h in hosts)
    extra = [h for h in hosts if h not in {d for d, _ in DEFAULT_HOSTS}]
    if extra:
        listed = "; ".join(x for x in (listed, ", ".join(extra)) if x)

    def fetch_public_data(url: str, filename: Optional[str] = None) -> str:
        if used["n"] >= limit:
            return json.dumps({"ok": False, "error": f"at most {limit} fetches per turn; work "
                               "with the files already fetched"})
        used["n"] += 1
        try:
            return json.dumps(fetch(url, filename=filename, hosts=hosts), default=str)
        except Refused as exc:
            return json.dumps({"ok": False, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - a network failure is an answer, not a crash
            return json.dumps({"ok": False, "error": f"the request failed: {type(exc).__name__}"})

    description = (
        "Download a file from an approved public data source into this conversation's files, "
        "so code can read it: pass the returned file_id in execute_code's input_files. The "
        "code sandbox has no network; this is how public data reaches it. HTTPS GET only, "
        f"approved hosts only: {listed}. Ask the service for what the analysis needs, not "
        "everything. TIGERweb (https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/): "
        "counties are State_County/MapServer/1 (GEOID, NAME, STATE, COUNTY), incorporated places "
        "Places_CouSub_ConCity_SubMCD/MapServer/4 (BASENAME, STATE), census tracts "
        "Tracts_Blocks/MapServer/0 (STATE, COUNTY); a layer answers "
        ".../MapServer/<id>/query?where=...&outFields=*&f=geojson, with returnCountOnly=true for "
        "a count, and any service lists its layers at .../MapServer?f=json. 3DEP answers "
        "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/"
        "exportImage?bbox=...&bboxSR=4326&size=...&format=tiff&f=image. Returns file_id, "
        "filename, size and the data's "
        f"shape (keys, feature count). Limits: {limit} fetches per turn, "
        f"{_int_env('AGENT_PUBLIC_FETCH_MAX_MB', 50)} MB per file."
    )
    return [StructuredTool.from_function(func=accept_null_defaults(fetch_public_data),
                                         name="fetch_public_data", description=description)]


__all__ = ["DEFAULT_HOSTS", "ENABLE_ENV", "Refused", "allowed_hosts", "check_url", "fetch",
           "is_enabled", "make_public_data_tools", "resolve_public"]
