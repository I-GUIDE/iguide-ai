"""fetch_public_data: public data reaches the offline sandbox through one gate, and only through it.

No test here reaches the network. Name resolution and the HTTP session are parameters, so every
refusal below is the gate's own decision, made before any connection would be opened.
"""
import json
import socket

import pytest

from agent_runtime import public_data_tools as pdt

TIGER = ("https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/1/"
         "query?where=GEOID%3D%2717019%27&outFields=*&f=geojson")
GEOJSON = json.dumps({"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {"GEOID": "17019", "NAME": "Champaign County"},
     "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}}]}).encode()
ARCGIS_ERROR = json.dumps({"error": {"code": 400, "message": "Unable to complete operation.",
                                     "details": ["Invalid query parameters."]}}).encode()


def public(*addresses):
    def resolver(host, port, proto=None):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port)) for a in addresses]
    return resolver


class Resp:
    def __init__(self, status=200, body=b"", headers=None):
        self.status_code, self._body, self.headers = status, body, dict(headers or {})
        self.closed = False

    def iter_content(self, chunk_size=65536):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def close(self):
        self.closed = True


class Session:
    """Answers each URL from a table, and records what was asked."""

    def __init__(self, table):
        self.table, self.asked = table, []

    def get(self, url, **kwargs):
        assert kwargs.get("allow_redirects") is False, "redirects must be followed by hand"
        self.asked.append(url)
        return self.table[url]


@pytest.fixture(autouse=True)
def _store(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    monkeypatch.delenv("AGENT_PUBLIC_FETCH_HOSTS", raising=False)
    monkeypatch.delenv("AGENT_PUBLIC_FETCH_MAX_MB", raising=False)
    monkeypatch.delenv("AGENT_PUBLIC_FETCH_PER_TURN", raising=False)


def outputs():
    from agent_runtime.file_store import storage_root
    d = storage_root() / "outputs"
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


# --- the URL -----------------------------------------------------------------------------

@pytest.mark.parametrize("url,why", [
    ("http://tigerweb.geo.census.gov/x", "https"),
    ("https://user:pw@tigerweb.geo.census.gov/x", "credentials"),
    ("https://169.254.169.254/latest/meta-data/", "IP address"),
    ("https://[::1]/x", "IP address"),
    ("https://tigerweb.geo.census.gov:8443/x", "port"),
    ("https://evil.example.com/x", "not an approved source"),
    ("https://tigerweb.geo.census.gov.evil.example.com/x", "not an approved source"),
    ("ftp://www2.census.gov/x", "https"),
])
def test_a_url_outside_the_gate_is_refused_and_says_why(url, why):
    with pytest.raises(pdt.Refused, match=why):
        pdt.check_url(url)


def test_a_refused_host_is_told_what_is_approved():
    with pytest.raises(pdt.Refused) as exc:
        pdt.check_url("https://evil.example.com/x")
    assert "tigerweb.geo.census.gov" in str(exc.value), "errors name the alternatives"


def test_the_allowlist_can_be_replaced_and_a_leading_dot_takes_subdomains(monkeypatch):
    monkeypatch.setenv("AGENT_PUBLIC_FETCH_HOSTS", ".example.org, data.example.com")
    assert pdt.check_url("https://maps.example.org/a")[1] == "maps.example.org"
    assert pdt.check_url("https://data.example.com/a")[1] == "data.example.com"
    with pytest.raises(pdt.Refused):
        pdt.check_url(TIGER)           # replaced, not extended


# --- where the name points ---------------------------------------------------------------

@pytest.mark.parametrize("address", ["169.254.169.254", "10.0.0.5", "127.0.0.1", "172.17.0.1",
                                     "192.168.1.10", "100.64.0.1", "::1", "fd00::1"])
def test_an_approved_name_that_resolves_inward_is_refused(address):
    with pytest.raises(pdt.Refused, match="non-public"):
        pdt.resolve_public("tigerweb.geo.census.gov", resolver=public(address))


def test_one_inward_address_among_public_ones_is_enough_to_refuse():
    with pytest.raises(pdt.Refused):
        pdt.resolve_public("tigerweb.geo.census.gov", resolver=public("152.199.4.33", "10.1.2.3"))


def test_a_public_name_passes():
    assert pdt.resolve_public("tigerweb.geo.census.gov",
                              resolver=public("152.199.4.33")) == ["152.199.4.33"]


# --- the request -------------------------------------------------------------------------

def run(table, url=TIGER, resolver=None, **kw):
    session = Session(table)
    out = pdt.fetch(url, resolver=resolver or public("152.199.4.33"),
                    session_factory=lambda: session, **kw)
    return out, session


def test_a_fetch_is_saved_as_a_conversation_file_with_its_shape():
    from agent_runtime.file_store import resolve_file_id

    out, _ = run({TIGER: Resp(200, GEOJSON, {"Content-Type": "application/json"})})
    assert out["ok"] and out["file_id"].startswith("file_")
    assert out["filename"].endswith(".geojson")
    assert resolve_file_id(out["file_id"]).read_bytes() == GEOJSON
    assert out["shape"]["features"] == 1 and out["shape"]["geometry_types"] == ["Polygon"]
    assert out["shape"]["property_names"] == ["GEOID", "NAME"]
    assert "input_files" in out["next"]


def test_a_redirect_is_checked_like_the_first_request():
    hop = "https://evil.example.com/steal"
    with pytest.raises(pdt.Refused, match="not an approved source"):
        run({TIGER: Resp(302, b"", {"Location": hop})})
    assert outputs() == []


def test_a_redirect_into_an_internal_address_is_refused():
    def resolver(host, port, proto=None):
        address = "169.254.169.254" if host == "www2.census.gov" else "152.199.4.33"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]
    with pytest.raises(pdt.Refused, match="non-public"):
        run({TIGER: Resp(302, b"", {"Location": "https://www2.census.gov/geo/x.zip"})},
            resolver=resolver)


def test_redirects_within_the_allowlist_are_followed_and_bounded():
    target = "https://www2.census.gov/geo/tiger/x.zip"
    out, session = run({TIGER: Resp(302, b"", {"Location": target}),
                        target: Resp(200, b"PK\x03\x04", {"Content-Type": "application/zip"})})
    assert out["ok"] and session.asked == [TIGER, target]
    loop = {TIGER: Resp(302, b"", {"Location": TIGER})}
    with pytest.raises(pdt.Refused, match="redirects"):
        run(loop)


def test_a_declared_size_over_the_cap_is_refused_before_reading():
    big = Resp(200, b"x" * 10, {"Content-Length": str(200 * 1024 * 1024)})
    with pytest.raises(pdt.Refused, match="over the"):
        run({TIGER: big})
    assert outputs() == []


def test_an_undeclared_body_is_cut_at_the_cap_and_nothing_is_kept():
    with pytest.raises(pdt.Refused, match="over the"):
        run({TIGER: Resp(200, b"x" * 5000)}, max_bytes=1000)
    assert outputs() == []


def test_a_failing_status_is_reported():
    with pytest.raises(pdt.Refused, match="HTTP 503"):
        run({TIGER: Resp(503, b"busy")})


def test_an_arcgis_error_with_status_200_is_reported_not_saved():
    """TIGERweb answers a bad query with HTTP 200 and an error object. Saved, it would reach
    the code as a 'data' file holding no data."""
    with pytest.raises(pdt.Refused, match="Invalid query parameters"):
        run({TIGER: Resp(200, ARCGIS_ERROR, {"Content-Type": "application/json"})})
    assert outputs() == []


def test_the_session_carries_nothing_from_the_environment():
    assert pdt._session().trust_env is False


# --- the tool ----------------------------------------------------------------------------

def test_the_tool_does_not_exist_unless_switched_on(monkeypatch):
    monkeypatch.delenv("AGENT_PUBLIC_FETCH", raising=False)
    assert pdt.make_public_data_tools() == []
    monkeypatch.setenv("AGENT_PUBLIC_FETCH", "1")
    tools = pdt.make_public_data_tools()
    assert [t.name for t in tools] == ["fetch_public_data"]
    assert "tigerweb.geo.census.gov" in tools[0].description


def test_the_tool_answers_a_refusal_in_words_and_caps_the_turn(monkeypatch):
    monkeypatch.setenv("AGENT_PUBLIC_FETCH", "1")
    calls = []

    def fake_fetch(url, filename=None, hosts=None):
        calls.append(url)
        if "evil" in url:
            raise pdt.Refused("evil.example.com is not an approved source")
        return {"ok": True, "file_id": "file_000000000001"}

    monkeypatch.setattr(pdt, "fetch", fake_fetch)
    tool = pdt.make_public_data_tools(per_turn=2)[0]
    refused = json.loads(tool.invoke({"url": "https://evil.example.com/x"}))
    assert refused == {"ok": False, "error": "evil.example.com is not an approved source"}
    assert json.loads(tool.invoke({"url": TIGER, "filename": None}))["ok"] is True
    capped = json.loads(tool.invoke({"url": TIGER}))
    assert capped["ok"] is False and "per turn" in capped["error"]
    assert len(calls) == 2, "the capped call never reached fetch"


@pytest.mark.parametrize("url,ctype,want", [
    ("https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/1/"
     "query?where=x&f=geojson", "application/json", "State_County_1.geojson"),
    ("https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/"
     "exportImage?bbox=1&format=tiff", "image/tiff", "3DEPElevation.tif"),
    ("https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_week.geojson",
     "application/json", "4.5_week.geojson"),
    ("https://overpass-api.de/api/interpreter?data=x", "application/json", "overpass-api.json"),
])
def test_a_download_is_named_for_what_it_is(url, ctype, want):
    """An ArcGIS path ends in a verb ("query"); the service and layer before it make the name."""
    assert pdt._filename(url, ctype, None) == want
