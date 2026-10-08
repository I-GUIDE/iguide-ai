"""The agent stack publishes its ports on 127.0.0.1, never on every interface.

A compose ``ports:`` entry without a host address (``"8000:8000"``) binds ``0.0.0.0`` and ``[::]``,
which makes the service exactly as reachable as the host. Until 2026-10-03 the compose file
published all three services of the stack that way. One is mcp-server, whose REST routes
(``POST /api/tool/<name>`` in ``MCP_server/server.py``) run any registered tool with no
authentication.

None of the stack's own traffic needs a published port. agent-api reaches embedding-server and
mcp-server by service name over the compose network, and a health check always runs inside its own
container. The deployment's mcp-server access log over 2026-09-22 → 2026-10-03 shows it. Of the
stack's own traffic there, 231 requests came from agent-api, all to ``/mcp/`` and all from its
compose-network address, and 31,576 were health checks from inside the container. None came from
the host side, and no one at all called ``/api/tool/*``. What does use a published port runs on the
host itself: a reverse proxy in front of agent-api, a developer's curl, or an SSH tunnel.
``127.0.0.1`` keeps those and nothing else.

This reads the compose file as YAML, so it does not replace ``docker compose config``. What it adds
is that a new service, or an edit back to ``"8000:8000"``, fails on a checkout.
"""

from __future__ import annotations

import ipaddress
import re
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
COMPOSE = REPO / "docker-compose.yml"

# Services that still publish on every interface, each with the reason. Anything else that
# publishes a port publishes it on loopback.
PUBLISHED_EVERYWHERE = {
    # The ingestion profile's entrance. The platform's form posts submissions to /ingest, and MinIO
    # its legacy bucket events to /webhook, from wherever they run, which need not be this host.
    # Not part of the agent stack, and left undecided on 2026-10-03.
    "metadata-extraction-server",
}


def _services() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]


_DEFAULTED = re.compile(r"\$\{\w+:?-([^}]*)\}")
_BARE_VARIABLE = re.compile(r"\$\{\w+\}")


def _host_address(entry) -> str:
    """The host address a ``ports:`` entry binds, or ``""`` for every interface.

    Read the way ``docker compose config`` reads it. The short syntax is
    ``[HOST:]HOST_PORT:CONTAINER_PORT[/PROTOCOL]``. Everything before the last two fields is the
    host, so ``"[::1]:8000:8000"`` and ``"::1:8000:8000"`` both bind ``::1``. A bare
    ``CONTAINER_PORT`` binds a random host port on every interface. In the long syntax the
    address is ``host_ip``.
    """
    if isinstance(entry, dict):
        return str(entry.get("host_ip") or "")
    # Compose interpolates before it parses, and ``${AGENT_DB_PORT:-5544}`` has a colon of its own.
    # A default stands in for its variable; a variable with none (``${BIND}``) is read as empty, so
    # an interpolated ADDRESS can never pass for loopback.
    text = _DEFAULTED.sub(r"\1", str(entry))
    text = _BARE_VARIABLE.sub("", text)
    fields = text.split("/", 1)[0].rsplit(":", 2)
    return fields[0].strip("[]") if len(fields) == 3 else ""


def _is_loopback(address: str) -> bool:
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:
        return False


def _environment(service: dict) -> dict[str, str]:
    env = service.get("environment") or {}
    if isinstance(env, list):
        env = dict(item.split("=", 1) for item in env if "=" in item)
    return {key: str(value) for key, value in env.items()}


def test_every_published_port_of_the_agent_stack_is_on_loopback():
    """The guard. ``"8000:8000"``, ``"0.0.0.0:8000:8000"`` or a new service published without an
    address fails here."""
    services = _services()
    assert PUBLISHED_EVERYWHERE <= set(services), "an exception names a service that is gone"

    exposed = [
        f"{name}: {entry!r}"
        for name, service in services.items()
        if name not in PUBLISHED_EVERYWHERE
        for entry in service.get("ports") or []
        if not _is_loopback(_host_address(entry))
    ]
    assert not exposed, "published beyond 127.0.0.1: " + ", ".join(exposed)


def test_no_service_shares_the_host_network():
    """``network_mode: host`` skips publishing altogether: whatever the container listens on is a
    listener of the host, on every interface it binds."""
    assert [name for name, service in _services().items() if service.get("network_mode") == "host"] == []


@pytest.mark.parametrize(
    ("variable", "service"),
    [("MCP_SERVER_URL", "mcp-server"), ("FLASK_EMBEDDING_URL", "embedding-server")],
)
def test_the_agent_reaches_each_service_by_name_on_a_shared_network(variable, service):
    """Why loopback costs the agent nothing: it never goes through the host. A URL naming the host
    (``host.docker.internal``, an address) would need the published port, and loopback would cut
    it off. Compose's ``environment`` wins over ``env_file``, so ``.env`` cannot redirect these."""
    services = _services()
    agent = services["agent-api"]
    url = _environment(agent)[variable]

    assert urlsplit(url).hostname == service, url
    assert set(agent.get("networks") or []) & set(services[service].get("networks") or [])


def test_each_health_check_calls_its_own_container():
    """A health check runs inside its container, where ``localhost`` is the service itself. One
    that called the host instead would go through the published port and fail on loopback. This
    reads the ``http(s)://`` URLs in each check; a check without one is not inspected."""
    outside = []
    for name, service in _services().items():
        test = (service.get("healthcheck") or {}).get("test") or []
        command = test if isinstance(test, str) else " ".join(map(str, test))
        for url in re.findall(r"https?://[^\s'\")]+", command):
            if urlsplit(url).hostname not in {"localhost", "127.0.0.1"}:
                outside.append(f"{name}: {url}")
    assert not outside


# Each entry below was read by `docker compose config` (Compose 5.5.1) on 2026-10-03, and its
# host_ip agreed with _host_address.
@pytest.mark.parametrize(
    "entry",
    [
        pytest.param("127.0.0.1:8000:8000", id="this-change"),
        pytest.param("127.0.0.1:3500:5002", id="different-ports"),
        pytest.param("127.0.0.1::8000", id="random-host-port"),
        pytest.param("127.0.0.1:5000-5002:5000-5002", id="a-range"),
        pytest.param("127.0.0.1:8000:8000/tcp", id="with-a-protocol"),
        pytest.param("127.0.0.2:8000:8000", id="anywhere-in-127/8"),
        pytest.param("[::1]:8000:8000", id="ipv6-in-brackets"),
        pytest.param("::1:8000:8000", id="ipv6-without-brackets"),
        pytest.param({"target": 8000, "published": "8000", "host_ip": "127.0.0.1"}, id="long-syntax"),
        pytest.param("127.0.0.1:${AGENT_DB_PORT:-5544}:5432", id="interpolated-port"),
    ],
)
def test_the_reader_accepts_each_loopback_form(entry):
    assert _is_loopback(_host_address(entry))


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param("8000:8000", id="as-before-this-change"),
        pytest.param("8000:8000/tcp", id="with-a-protocol"),
        pytest.param("8000", id="random-host-port"),
        pytest.param(8000, id="an-unquoted-number"),
        pytest.param(":8000:8000", id="an-empty-address"),
        pytest.param("0.0.0.0:8000:8000", id="every-ipv4-interface"),
        pytest.param("[::]:8000:8000", id="every-ipv6-interface"),
        pytest.param("10.0.0.5:8000:8000", id="one-real-interface"),
        pytest.param({"target": 8000, "published": "8000"}, id="long-syntax-without-host_ip"),
        pytest.param({"target": 8000, "published": "8000", "host_ip": "0.0.0.0"}, id="long-syntax-everywhere"),
        pytest.param("${AGENT_DB_PORT:-5544}:5432", id="interpolated-port-no-address"),
        pytest.param("${BIND}:5544:5432", id="interpolated-address"),
    ],
)
def test_the_reader_rejects_each_form_that_leaves_loopback(entry):
    """A guard is only as good as the wrong answers it rejects."""
    assert not _is_loopback(_host_address(entry))
