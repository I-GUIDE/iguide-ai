"""Staging a platform dataset into the sandbox workspace.

Extraction generates 44 loaders shaped ``def load_x(staged_path)`` and nothing could produce a
``staged_path``: the only staging that existed resolved a user-uploaded ``file_id``. An entire
element type was readable and not runnable.

Two properties carry this module, and the second is the one with teeth:

* the fetch happens AGENT-SIDE, so credentials never enter a container that runs with
  ``--network none``, and every staged file records its origin, size and sha256 in
  ``inputs.jsonl``;
* the URL comes from a language model, and this process holds MinIO keys, OpenSearch credentials
  and a route into the platform's private subnet. ``169.254.169.254`` hands out instance
  credentials; ``10.0.147.52:7687`` is the platform's Neo4j; ``127.0.0.1`` is the agent's own API.
  A staging tool without an SSRF guard is a credential-exfiltration path with extra steps.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_runtime import staging
from agent_runtime.staging import (StagingError, inputs_dir, safe_filename, stage_element,
                                   stage_object, stage_url, staged_inputs)


@pytest.fixture()
def session(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENT_CODE_EXEC_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.delenv("AGENT_STAGING_ALLOW_PRIVATE", raising=False)
    return "sess-1"


class _Source:
    def __init__(self, path, origin="", bucket="", key=""):
        self.local_path, self.origin_url, self.bucket, self.key = str(path), origin, bucket, key
        self.sha256, self.bytes, self.content_type = "deadbeef", 42, "text/csv"
        self.fetched_at, self.backend = "2026-08-26T00:00:00+00:00", "http"


# ------------------------------------------------------------------ the SSRF guard

@pytest.mark.parametrize("url,why", [
    ("file:///etc/passwd", "reads the agent's own disk"),
    ("ftp://example.org/x.csv", "not http(s)"),
    ("/etc/passwd", "a bare path, not a URL"),
    ("", "empty"),
    ("https://", "no host"),
])
def test_only_public_http_urls_are_accepted(url, why, session):
    with pytest.raises(StagingError) as err:
        stage_url(url, session)
    assert err.value.kind in {"scheme", "unresolvable"}, why


@pytest.mark.parametrize("host,address,what", [
    ("metadata.example", "169.254.169.254", "the cloud metadata service, which hands out creds"),
    ("localhost.example", "127.0.0.1", "the agent's own API"),
    ("neo4j.example", "10.0.147.52", "the platform's Neo4j on the private subnet"),
    ("internal.example", "192.168.1.10", "a private-range host"),
    ("cgnat.example", "172.16.5.4", "another private range"),
    ("v6.example", "::1", "IPv6 loopback"),
])
def test_a_private_or_metadata_address_is_refused(host, address, what, session, monkeypatch):
    monkeypatch.setattr(staging, "_addresses_for", lambda h: [address])
    called = []
    monkeypatch.setattr("extractors.sources.fetch_url",
                        lambda *a, **k: called.append(a) or _Source("x"))
    with pytest.raises(StagingError) as err:
        stage_url(f"https://{host}/data.csv", session)
    assert err.value.kind == "blocked", what
    assert address in str(err.value)
    assert not called, "the fetch ran despite the address being refused"


def test_the_refusal_explains_itself_and_names_the_escape_hatch(session, monkeypatch):
    """A rejection a reader cannot act on gets worked around by disabling the check wholesale."""
    monkeypatch.setattr(staging, "_addresses_for", lambda h: ["10.0.0.5"])
    with pytest.raises(StagingError) as err:
        stage_url("https://internal.example/x.csv", session)
    message = str(err.value)
    assert "credential-exfiltration" in message
    assert "AGENT_STAGING_ALLOW_PRIVATE" in message


def test_every_resolved_address_must_pass_not_just_the_first():
    """A host that resolves to one public and one private address is the DNS-rebinding shape."""
    import agent_runtime.staging as mod

    original = mod._addresses_for
    try:
        mod._addresses_for = lambda h: ["93.184.216.34", "127.0.0.1"]
        with pytest.raises(StagingError) as err:
            mod._assert_fetchable("https://mixed.example/x.csv")
        assert err.value.kind == "blocked"
    finally:
        mod._addresses_for = original


def test_the_guard_can_be_disabled_deliberately(session, monkeypatch, tmp_path):
    """Some deployments have no credentials to steal. The opt-out is explicit and named."""
    monkeypatch.setenv("AGENT_STAGING_ALLOW_PRIVATE", "1")
    monkeypatch.setattr(staging, "_addresses_for", lambda h: ["127.0.0.1"])

    def fake_fetch(url, dest, **kwargs):
        Path(dest).write_text("a,b\n1,2\n")
        return _Source(dest, origin=url)

    monkeypatch.setattr("extractors.sources.fetch_url", fake_fetch)
    result = stage_url("https://localhost.example/x.csv", session)
    assert result["staged_path"].endswith("/x.csv")


def test_an_unresolvable_host_is_reported_not_fetched(session, monkeypatch):
    monkeypatch.setattr(staging, "_addresses_for",
                        lambda h: (_ for _ in ()).throw(
                            StagingError("nope", kind="unresolvable")))
    with pytest.raises(StagingError) as err:
        stage_url("https://nowhere.invalid/x.csv", session)
    assert err.value.kind == "unresolvable"


# ------------------------------------------------------------------ filenames

@pytest.mark.parametrize("raw,expected", [
    ("crime.csv", "crime.csv"),
    ("../../etc/passwd", "passwd"),
    ("a/b/c/data.zip", "data.zip"),
    ("weird name (1).tif", "weird_name_1_.tif"),
    ("export.csv?token=secret", "export.csv"),
    ("", "input.bin"),
    ("...", "input.bin"),
])
def test_a_filename_cannot_escape_the_inputs_directory(raw, expected):
    assert safe_filename(raw) == expected


def test_a_query_string_does_not_become_part_of_the_name(session, monkeypatch, tmp_path):
    monkeypatch.setattr(staging, "_addresses_for", lambda h: ["93.184.216.34"])

    def fake_fetch(url, dest, **kwargs):
        Path(dest).write_text("x")
        return _Source(dest, origin=url)

    monkeypatch.setattr("extractors.sources.fetch_url", fake_fetch)
    result = stage_url("https://example.org/export.csv?signature=abc123", session)
    assert result["filename"] == "export.csv"
    assert "abc123" not in result["staged_path"], "a signed URL's token leaked into the workspace"


# ------------------------------------------------------------------ where it lands

@pytest.fixture()
def fetched(session, monkeypatch):
    monkeypatch.setattr(staging, "_addresses_for", lambda h: ["93.184.216.34"])

    def fake_fetch(url, dest, **kwargs):
        Path(dest).write_text("ID,Latitude,Longitude\n1,41.9,-87.6\n")
        return _Source(dest, origin=url)

    monkeypatch.setattr("extractors.sources.fetch_url", fake_fetch)
    return stage_url("https://example.org/crime.csv", session)


def test_the_returned_path_is_the_one_the_sandbox_sees(fetched):
    """A generated loader takes `staged_path` and runs INSIDE the container. Handing it the host
    path would fail there, and the failure would surface as a missing file inside a sandbox with
    no network to investigate with."""
    assert fetched["staged_path"] == "/work/inputs/crime.csv"
    assert fetched["host_path"].endswith("/inputs/crime.csv")
    assert fetched["host_path"] != fetched["staged_path"]


def test_the_bytes_actually_land_in_the_workspace(fetched, session):
    host = Path(fetched["host_path"])
    assert host.is_file()
    assert host.parent == inputs_dir(session)
    assert "Latitude" in host.read_text()


def test_provenance_is_recorded_for_the_re_run(fetched, session):
    entries = staged_inputs(session)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["origin"] == "https://example.org/crime.csv"
    assert entry["sha256"] == "deadbeef"
    assert entry["bytes"] == 42
    assert entry["fetched_at"]


def test_the_manifest_is_append_only(session, monkeypatch):
    """A partially-written run must still yield every input that completed — a re-run needs to
    know what it HAD, not only what a tidy final state says."""
    monkeypatch.setattr(staging, "_addresses_for", lambda h: ["93.184.216.34"])

    def fake_fetch(url, dest, **kwargs):
        Path(dest).write_text("x")
        return _Source(dest, origin=url)

    monkeypatch.setattr("extractors.sources.fetch_url", fake_fetch)
    stage_url("https://example.org/a.csv", session)
    stage_url("https://example.org/b.csv", session)
    origins = [e["origin"] for e in staged_inputs(session)]
    assert origins == ["https://example.org/a.csv", "https://example.org/b.csv"]


def test_two_sessions_do_not_share_a_workspace(session, monkeypatch):
    monkeypatch.setattr(staging, "_addresses_for", lambda h: ["93.184.216.34"])

    def fake_fetch(url, dest, **kwargs):
        Path(dest).write_text("x")
        return _Source(dest, origin=url)

    monkeypatch.setattr("extractors.sources.fetch_url", fake_fetch)
    stage_url("https://example.org/a.csv", session)
    assert staged_inputs("sess-2") == []


# ------------------------------------------------------------------ objects and elements

def test_an_object_is_fetched_with_credentials_that_stay_here(session, monkeypatch):
    """The whole point of staging agent-side: MinIO keys never cross into a container."""
    seen = {}

    def fake_object(bucket, key, dest, **kwargs):
        seen.update(bucket=bucket, key=key)
        Path(dest).write_text("bytes")
        return _Source(dest, bucket=bucket, key=key)

    monkeypatch.setattr("extractors.sources.fetch_object", fake_object)
    result = stage_object("iguide", "datasets/crime.csv", session)
    assert seen == {"bucket": "iguide", "key": "datasets/crime.csv"}
    assert result["origin"] == "iguide/datasets/crime.csv"
    assert result["staged_path"] == "/work/inputs/crime.csv"


@pytest.mark.parametrize("bucket,key", [("", "k"), ("b", ""), ("", "")])
def test_an_object_request_missing_its_coordinates_is_refused(bucket, key, session):
    with pytest.raises(StagingError) as err:
        stage_object(bucket, key, session)
    assert err.value.kind == "bad_request"


def test_an_element_resolves_through_the_platforms_own_link_field(session, monkeypatch):
    """`source_link` knows both naming vocabularies — the graph's `external_link` and the REST
    API's type-suffixed `external-link-publication`. Reading one of them found a link on 8 of 203
    publications."""
    monkeypatch.setattr(staging, "_addresses_for", lambda h: ["93.184.216.34"])

    def fake_fetch(url, dest, **kwargs):
        Path(dest).write_text("x")
        return _Source(dest, origin=url)

    monkeypatch.setattr("extractors.sources.fetch_url", fake_fetch)
    record = {"id": "265e6957", "direct-download-link": "https://example.org/export.csv"}
    result = stage_element("265e6957", session, metadata=record)
    assert result["origin"] == "https://example.org/export.csv"


def test_an_element_with_a_bucket_and_key_prefers_the_object_path(session, monkeypatch):
    def fake_object(bucket, key, dest, **kwargs):
        Path(dest).write_text("x")
        return _Source(dest, bucket=bucket, key=key)

    monkeypatch.setattr("extractors.sources.fetch_object", fake_object)
    record = {"id": "e1", "bucket": "iguide", "key": "a/b.csv",
              "external-link": "https://example.org/should-not-be-used.csv"}
    assert stage_element("e1", session, metadata=record)["origin"] == "iguide/a/b.csv"


def test_a_portal_pointer_says_there_is_no_file_rather_than_failing_obscurely(session):
    """20 of the corpus's 130 datasets are a portal front door. There is no file and there never
    was, so the refusal has to say that rather than read as a broken download."""
    record = {"id": "7d0c1d45", "title": "FAOSTAT"}
    with pytest.raises(StagingError) as err:
        stage_element("7d0c1d45", session, metadata=record)
    assert err.value.kind == "no_source"
    assert "portal pointer" in str(err.value)


def test_a_fetch_failure_keeps_its_kind(session, monkeypatch):
    """`too_large`, `forbidden` and `not_found` need different responses. Collapsing them to one
    error is how "this is paywalled" became indistinguishable from "the network blipped"."""
    from extractors.sources import SourceError

    monkeypatch.setattr(staging, "_addresses_for", lambda h: ["93.184.216.34"])
    monkeypatch.setattr("extractors.sources.fetch_url",
                        lambda *a, **k: (_ for _ in ()).throw(
                            SourceError("too big", kind="too_large")))
    with pytest.raises(StagingError) as err:
        stage_url("https://example.org/huge.tif", session)
    assert err.value.kind == "too_large"


# ------------------------------------------------------------------ the wiring

def test_the_staging_tools_are_registered_with_the_names_the_filter_allows():
    """`RAG_COMPONENT_TOOL_NAMES` is a hard filter — a name absent from it is stripped for every
    intent, so a tool registered under a name the set does not carry is silently unreachable."""
    from agent_runtime.graph_state import RAG_COMPONENT_TOOL_NAMES
    from agent_runtime.langchain_granular_tools import make_langchain_staging_tools

    names = {getattr(t, "name", "") for t in make_langchain_staging_tools(session_id="s")}
    assert names == {"stage_element", "stage_url", "list_staged_inputs"}
    assert names <= set(RAG_COMPONENT_TOOL_NAMES), names - set(RAG_COMPONENT_TOOL_NAMES)


def test_staging_shares_the_workspace_the_sandbox_mounts():
    """Staging and execute_code must key the workspace identically. Staging into a different key
    would put the bytes in a directory the sandbox never carries into /work, and the failure would
    surface inside a container with no network as "file not found".

    The key is prototype's "codeexec" (what deployed conversations' workspaces already use);
    backend_swap used "code_exec" until the 2026-10-01 integration."""
    import re

    source = Path(__file__).resolve().parents[2] / "agent_runtime" / "supervisor" / "graph.py"
    text = source.read_text(encoding="utf-8")
    key = re.compile(r'child_thread_id\(state\.get\("thread_id"\), "([a-z_]+)"\)')
    staging_call = text.split("make_langchain_staging_tools(", 1)[1][:200]
    staging_key = key.search(staging_call).group(1)
    exec_keys = {m.group(1) for m in key.finditer(text)
                 if "make_code_execution_tools(" in text[max(0, m.start() - 300):m.start()]}
    assert staging_key == "codeexec", staging_call
    assert exec_keys == {staging_key}, (staging_key, exec_keys)


def test_a_refusal_reaches_the_model_as_an_answer_not_a_traceback():
    """A stack trace is something the model cannot act on. A refusal with a kind and a sentence is
    something it can respond to — by staging a different element, or by telling the user the
    dataset is a portal pointer with no file."""
    import json as _json

    from agent_runtime.langchain_granular_tools import stage_url_tool

    payload = _json.loads(stage_url_tool("file:///etc/passwd", "s"))
    assert payload["ok"] is False
    assert payload["kind"] == "scheme"
    assert "http" in payload["error"]


def test_a_successful_stage_returns_the_container_path_to_the_model(tmp_path, monkeypatch):
    import json as _json

    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENT_CODE_EXEC_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setattr(staging, "_addresses_for", lambda h: ["93.184.216.34"])

    def fake_fetch(url, dest, **kwargs):
        Path(dest).write_text("a,b\n1,2\n")
        return _Source(dest, origin=url)

    monkeypatch.setattr("extractors.sources.fetch_url", fake_fetch)
    from agent_runtime.langchain_granular_tools import stage_url_tool

    payload = _json.loads(stage_url_tool("https://example.org/x.csv", "s2"))
    assert payload["ok"] is True
    assert payload["staged_path"] == "/work/inputs/x.csv"
    assert payload["sha256"]


def test_listing_staged_inputs_is_readable_by_the_model(tmp_path, monkeypatch):
    import json as _json

    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENT_CODE_EXEC_WORK_ROOT", str(tmp_path / "work"))
    from agent_runtime.langchain_granular_tools import list_staged_inputs_tool

    payload = _json.loads(list_staged_inputs_tool("empty-session"))
    assert payload == {"count": 0, "inputs": []}
