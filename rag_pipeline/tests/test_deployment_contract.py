"""Deployment packaging: capabilities that exist on a checkout must exist in the container.

The failure mode this file guards is the worst kind of gap — a feature that works in
development and silently disappears in deployment, where nobody is watching a test suite.
``skills.py`` looks for SKILL bundles under ``REPO_ROOT/.agents/skills`` and
``REPO_ROOT/skills``; the image copied neither, so ``list_available_skills`` returned an empty
list in every deployed container.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
DOCKERFILE = REPO / "rag_pipeline" / "Dockerfile"
COMPOSE = REPO / "docker-compose.yml"


def _dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def test_the_image_copies_every_package_the_agent_imports():
    """A missing COPY surfaces only as an ImportError inside a running container."""
    text = _dockerfile()
    for package in ("rag_pipeline/", "agent_runtime/", "api/", "MCP_server/", "extractors/"):
        assert f"COPY {package}" in text, f"{package} is not copied into the image"


def test_the_image_copies_the_skill_bundles():
    """EVERY skill root that exists in the checkout must be copied. This used `any(...)`, so it
    passed with .agents/ copied while skills/ — the FIRST root, holding two of the three curated
    skills — was not; the deployed registry discovered 0 (2026-10-01)."""
    from agent_runtime.skills import DEFAULT_SKILL_ROOTS, REPO_ROOT

    text = _dockerfile()
    missing = []
    for root in DEFAULT_SKILL_ROOTS:
        if not root.is_dir():
            continue
        top = root.relative_to(REPO_ROOT).parts[0]          # "skills", ".agents"
        if f"COPY {top}/" not in text:
            missing.append(top)
    assert not missing, f"skill roots not copied into the image: {missing}"


def test_generated_skills_are_written_to_the_persistent_volume():
    """Written under /app/.agents/skills they would vanish on every restart — and the image
    runs as non-root, so that directory is not writable anyway."""
    text = COMPOSE.read_text(encoding="utf-8")
    assert "AGENT_GENERATED_SKILLS_ROOT=/app/agent_chat_files/skills" in text
    assert "AGENT_SKILL_PATHS=/app/agent_chat_files/skills" in text


def test_the_embedding_url_is_pinned_for_every_service_that_needs_it():
    """env_file supplies whatever .env holds, which was a decommissioned remote host. Only
    agent-api pinned it; the others were silently embedding-less."""
    text = COMPOSE.read_text(encoding="utf-8")
    assert text.count("FLASK_EMBEDDING_URL=http://embedding-server:5000") >= 3


def test_the_dev_only_llm_backend_is_not_configured_in_compose():
    """LLM_PROVIDER=claude-cli must never back a deployed server: Anthropic's consumer terms
    restrict automated access to API-key access and forbid serving other users through a
    personal subscription."""
    text = COMPOSE.read_text(encoding="utf-8")
    assert "LLM_PROVIDER=claude-cli" not in text


def test_workflow_execution_stays_disabled_in_compose():
    """generic_executor_tools runs ingested notebook source through a bare exec() in a process
    holding cluster credentials. Deliberately off; asserted so it cannot drift on."""
    text = COMPOSE.read_text(encoding="utf-8")
    assert "AGENT_ALLOW_WORKFLOW_EXEC=1" not in text


# ------------------------------------------------------------------ the sandbox image

def _sandbox_dockerfile() -> str:
    """Named distinctly from the module-level `_dockerfile` (the agent-api image). Defining a
    second `_dockerfile` shadowed it and silently redirected two existing tests at the wrong
    file — they still ran, still asserted, and were checking something else entirely."""
    from pathlib import Path

    return Path("sandbox/Dockerfile").read_text(encoding="utf-8")


def test_the_geospatial_stack_is_baked_into_the_image():
    """Without it, every session touching a GeoDataFrame pays a `pip install geopandas` before
    any work happens — and the library's strongest clusters (2SFCA accessibility, SPASTC region
    building, remote sensing) are all geospatial."""
    src = _sandbox_dockerfile()
    for pkg in ("geopandas", "rasterio", "pyproj", "pyogrio"):
        assert pkg in src, f"{pkg} is not baked into the sandbox image"
    assert "# RUN pip install --no-cache-dir geopandas" not in src, (
        "the geospatial block is still commented out")


def test_the_runtime_library_the_wheels_dlopen_is_installed():
    """`import rasterio` dies with "libexpat.so.1: cannot open shared object file" on
    python:3.11-slim. The wheel installs cleanly and fails at import, so a green build proves
    nothing — this shipped past one already."""
    assert "libexpat1" in _sandbox_dockerfile()


def test_the_crs_database_is_warmed_at_build_time():
    """The gate reads `crs.axis_info` on every geometry frame, inside a container with
    --network none. A cold pyproj database on that path is a failure, not a slow start."""
    src = _sandbox_dockerfile()
    assert "pyproj.CRS.from_epsg" in src


def test_a_tag_is_resolved_to_a_digest():
    """`python:3.11-slim` resolves to different bytes next month, so an artifact recording a tag
    records nothing about the environment that produced its number."""
    from agent_runtime.artifacts import resolve_image_digest

    already = "python@sha256:" + "a" * 64
    assert resolve_image_digest(already) == already, "a digest must pass through unchanged"
    assert resolve_image_digest("") is None


# ------------------------------------------------------------------ CI actually checks the claim

def _workflow() -> str:
    from pathlib import Path

    path = Path(".github/workflows/verify.yml")
    assert path.is_file(), "no CI workflow"
    return path.read_text(encoding="utf-8")


def test_ci_runs_the_deployment_contract_separately():
    """These assertions are not reachable from unit tests — nothing imports compose."""
    assert "test_deployment_contract.py" in _workflow()


def test_ci_proves_the_gate_still_catches_a_degrees_buffer():
    """The gate's whole purpose. If it passes silently, every downstream "verified" claim is
    worthless and nothing else in CI would notice."""
    src = _workflow()
    assert "25000" in src and "EPSG:4326" in src
    assert "fail" in src and "pass" in src


def test_ci_imports_the_stack_with_no_network():
    """The sandbox runs --network none, so an import that needs the network is a real failure
    that a normal build would not surface."""
    assert "--network none" in _workflow()


def test_ci_installs_with_the_pinned_constraints():
    """Without -c constraints.txt the host resolves different versions than the image, and the
    suite proves something about an environment nobody deploys."""
    assert "-c constraints.txt" in _workflow()


# ------------------------------------------------------------------ declared vs inherited deps

def _requirements() -> str:
    return open("requirements.txt", encoding="utf-8").read()


def _constraints() -> str:
    return open("constraints.txt", encoding="utf-8").read()


def test_every_directly_imported_third_party_graph_dep_is_declared():
    """A dependency that arrives transitively is a dependency you do not control.

    networkx was importable here only because torch requires it (`pip show networkx` ->
    Required-by: intake, mapclassify, osmnx, scikit-image, torch). That is the pyarrow failure
    again: present in dev via anaconda, absent from a clean build the moment the package that
    dragged it in changes. Community detection over the extracted corpus imports it directly,
    so it must be declared directly.
    """
    assert "\nnetworkx" in _requirements(), "networkx is imported directly; declare it"


def test_networkx_is_pinned_because_the_resolve_drifts():
    """Measured: dev runs 3.4.2 while a clean `pip install -r requirements.txt -c
    constraints.txt` resolves 3.6.1 — so a partition measured here would have run on a
    different implementation in CI. That is exactly the drift constraints.txt exists to stop,
    caught this time before anything depended on it rather than after."""
    assert "networkx==" in _constraints(), "an unpinned graph library makes results unreplayable"


def test_the_installed_networkx_matches_the_pin():
    """The pin is only worth something if this environment honours it. A pin that disagrees with
    what is imported means every number recorded here describes an environment nobody runs."""
    import re

    import networkx

    m = re.search(r"^networkx==([\d.]+)", _constraints(), re.M)
    assert m, "no networkx pin to check against"
    assert networkx.__version__ == m.group(1), (
        f"constraints.txt pins networkx=={m.group(1)} but this environment has "
        f"{networkx.__version__}; measurements here do not describe the deployed stack")


def test_the_partition_algorithm_this_work_depends_on_exists_and_is_seedable():
    """Louvain without a seed is nondeterministic, and a knowledge graph whose communities
    change between runs cannot carry stable ids or cached summaries."""
    import inspect

    from networkx.algorithms import community

    assert hasattr(community, "louvain_communities")
    params = inspect.signature(community.louvain_communities).parameters
    assert "seed" in params and "weight" in params and "resolution" in params


def test_a_seed_alone_does_NOT_make_the_partition_reproducible():
    """The correction to the test above, which certified a property the system did not have.

    `seed=` pins Louvain's own randomness but NOT the order it visits nodes — that follows the
    graph's insertion order. Build the same graph from a `set` and CPython's string hashing
    (PYTHONHASHSEED) reorders insertion, so an identical graph partitions differently. Measured
    across three hash seeds on one fixed graph: set-insertion gave Q 0.235526 / 0.244236 /
    0.234232 and 7 / 7 / 8 communities, while sorted-insertion gave 0.243509 and an identical
    partition every time.

    This is not academic. Community *ids* moving between two runs over an unchanged corpus
    invalidates every cached community summary and every community_id written onto an element
    document — the drift would look like the corpus changed when nothing had.

    The requirement this pins: build the graph from a SORTED node and edge sequence. The
    assertion is on the fix, not on the bug, so it stays true if networkx ever hardens this.
    """
    import random

    import networkx as nx
    from networkx.algorithms import community

    # A seeded RANDOM graph, not a structured one. First attempt built edges by modular
    # arithmetic, which is regular enough that Louvain resolves it identically from any order —
    # so the test passed for the wrong reason and proved nothing. Measured over random graphs,
    # 9 orderings give 9 distinct partitions at every size from 60 nodes up to 750 nodes /
    # 9,112 edges, which are the real fused graph's dimensions.
    _rng = random.Random(4)
    nodes = [f"n{i:03d}" for i in range(120)]
    _e: set = set()
    while len(_e) < 700:
        _a, _b = _rng.sample(nodes, 2)
        _e.add(tuple(sorted((_a, _b))))
    edges = sorted(_e)

    def partition(node_order, edge_order, *, canonicalise):
        """`canonicalise=True` is the discipline under test: sort before inserting."""
        g = nx.Graph()
        g.add_nodes_from(sorted(node_order) if canonicalise else node_order)
        g.add_edges_from((a, b, {"weight": 1.0})
                         for a, b in (sorted(edge_order) if canonicalise else edge_order))
        return sorted(tuple(sorted(p))
                      for p in community.louvain_communities(g, weight="weight", seed=17))

    rng = random.Random(11)
    orders = []
    for _ in range(6):
        n, e = list(nodes), list(edges)
        rng.shuffle(n)
        rng.shuffle(e)
        orders.append((n, e))

    # THE PROPERTY WE DEPEND ON: however a caller hands us the corpus, canonicalising the
    # insertion order yields one partition. Without this, the seed is decoration.
    canonical = {partition(n, e, canonicalise=True) == partition(nodes, edges, canonicalise=True)
                 for n, e in orders}
    assert canonical == {True}, (
        "sorted insertion must make the partition independent of caller order")

    # And the hazard is real at this version rather than hypothetical: at least one raw
    # (unsorted) ordering of the SAME graph disagrees with the canonical partition. If networkx
    # ever hardens this the assertion fails loudly and the comment above gets revisited —
    # which is the correct outcome, not a silent pass.
    reference = partition(nodes, edges, canonicalise=True)
    raw = [partition(n, e, canonicalise=False) for n, e in orders]
    assert any(p != reference for p in raw), (
        "expected insertion order to change the partition at networkx "
        f"{nx.__version__}; if it no longer does, the canonicalisation may be unnecessary")


# ------------------------------------------------------------------ the extraction service

def _extraction_dockerfile() -> str:
    from pathlib import Path

    return Path("metadata-extraction-server/Dockerfile").read_text(encoding="utf-8")


def test_the_extraction_image_copies_every_package_its_entrypoint_imports():
    """The service's only real work is `from extractors.ingest import ingest_submission`, and
    the image copied only `metadata-extraction-server/`. Every request would have died with
    ModuleNotFoundError — while the container reported HEALTHY, because /health imports none of
    it. A liveness probe that cannot fail for the reason the service exists is not a probe.
    """
    text = _extraction_dockerfile()
    for package in ("extractors/", "rag_pipeline/", "agent_runtime/"):
        assert f"COPY {package}" in text, f"{package} is not in the extraction image"


def test_the_vector_reader_geopandas_actually_uses_is_pinned():
    """GeoPandas 1.x reads vectors through pyogrio; pinning only fiona left every shapefile in
    the corpus reporting "vector reader unavailable" in an environment that had a working
    reader installed."""
    from pathlib import Path

    requirements = Path("requirements.txt").read_text(encoding="utf-8").lower()
    assert "pyogrio" in requirements


def test_extraction_runs_without_importing_the_agent_graph():
    """Extraction is triggered by the PLATFORM, not by the agent, so it has to work when the
    agent is down. Importing the ingest entrypoint must not drag in langgraph or the supervisor.

    This is the property that decides where the tools live: if it holds, extraction can be its
    own image; if it breaks, the two are one deployment whether or not that was intended.
    """
    import subprocess
    import sys

    probe = (
        "import sys; import extractors.ingest;"
        "bad = sorted(m for m in sys.modules"
        "             if m.startswith(('langgraph', 'langchain_openai'))"
        "             or m.startswith('agent_runtime.supervisor'));"
        "print(','.join(bad))"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                            timeout=180)
    assert result.returncode == 0, result.stderr[-400:]
    leaked = [m for m in result.stdout.strip().split(",") if m]
    assert not leaked, (
        f"importing extractors.ingest pulled in the agent stack: {leaked[:5]} — extraction "
        f"cannot then be deployed or scaled separately from the agent")


# ------------------------------------------------------- compose must load on a bare .env

def test_compose_has_no_required_variable_that_fails_every_service():
    """`${VAR:?msg}` is interpolated over the WHOLE file before profiles apply (checked against
    docker compose v5.5.1), so one required variable for an optional service made `docker compose
    up` fail for every service on a host whose .env lacked it. Enforce such values where they are
    used instead — the postgres image already refuses an empty superuser password."""
    import re

    text = (Path(__file__).resolve().parents[2] / "docker-compose.yml").read_text(encoding="utf-8")
    config = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    required = re.findall(r"\$\{[A-Za-z_][A-Za-z0-9_]*:\?[^}]*\}", config)
    assert not required, required


def test_the_agent_db_is_behind_a_profile():
    """The extraction bundle is off by default, so its database must not start with `up -d`."""
    yaml = pytest.importorskip("yaml")
    compose = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "docker-compose.yml").read_text(encoding="utf-8"))
    assert compose["services"]["agent-db"].get("profiles"), "agent-db starts by default"
    assert "agent-db" not in (compose["services"]["agent-api"].get("depends_on") or {}), (
        "agent-api must not depend on a service that is off by default")


# ------------------------------------------------------------------ turning the extraction on
#
# agent-api starts sandbox runs through the HOST's Docker daemon, which resolves every `-v` source
# on the host. The library's default location is on a named volume that the host does not have at
# that path (verified on the VM: /app does not exist there), so the daemon mounted an EMPTY
# directory and every library import failed while kb_method_search kept listing the methods.

EXTRACTION_OVERRIDE = REPO / "docker-compose.extraction.yml"
_LIBRARY_VAR = r"\$\{AGENT_METHOD_LIBRARY_DIR:\?[^}]*\}"


def _override_agent_api() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(EXTRACTION_OVERRIDE.read_text(encoding="utf-8"))["services"]["agent-api"]


def test_the_override_mounts_the_library_at_the_identical_path():
    """The same rule the main file follows for /tmp/iguide_codeexec."""
    import re

    svc = _override_agent_api()
    env = dict(entry.split("=", 1) for entry in svc["environment"])
    assert env["AGENT_EXTRACTION"] == "1", "including the override is the switch"
    path = "/srv/iguide/method_library"
    assert re.sub(_LIBRARY_VAR, path, env["AGENT_METHOD_LIBRARY_DIR"]) == path
    assert f"{path}:{path}" in [re.sub(_LIBRARY_VAR, path, v) for v in svc["volumes"]]


def test_the_override_requires_the_library_path():
    """`:?` is right HERE and only here: this file is read only when the bundle is being turned
    on, so a missing path stops `up` instead of surfacing as the first failed import. The main
    file's rule is test_compose_has_no_required_variable_that_fails_every_service."""
    import re

    text = EXTRACTION_OVERRIDE.read_text(encoding="utf-8")
    config = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    uses = re.findall(r"\$\{AGENT_METHOD_LIBRARY_DIR[^}]*\}", config)
    assert len(uses) == 3 and all(":?" in u for u in uses), uses


def test_the_image_carries_the_preflight_scripts():
    """The pre-flight has to run where turns run. From the host, the smoke test mounted the
    library from a host path and passed while every turn in the container got an empty one."""
    copies = [l for l in _dockerfile().splitlines() if l.startswith("COPY ")]
    for script in ("scripts/build_method_library.py", "scripts/smoke_end_to_end.py"):
        assert any(script in line for line in copies), f"{script} is not copied into the image"


def _compose(tmp_path, *args, env=None):
    import os
    import shutil
    import subprocess

    if shutil.which("docker") is None:
        pytest.skip("docker CLI not installed")
    for name in ("docker-compose.yml", "docker-compose.extraction.yml"):
        shutil.copy(REPO / name, tmp_path / name)
    (tmp_path / ".env").write_text("")          # env_file: .env must exist to render
    full_env = {k: v for k, v in os.environ.items() if k != "AGENT_METHOD_LIBRARY_DIR"}
    full_env.update(env or {})
    proc = subprocess.run(["docker", "compose", "-f", "docker-compose.yml", "-f",
                           "docker-compose.extraction.yml", *args],
                          cwd=tmp_path, env=full_env, capture_output=True, text=True, timeout=60)
    if "is not a docker command" in proc.stderr:
        pytest.skip("docker compose plugin not installed")
    return proc


def test_compose_merges_the_override_the_way_this_file_says(tmp_path):
    """Compose's real merge, not a reading of the YAML: the override's volume and environment
    entries are ADDED to the main file's, and the work-root mount survives beside the library."""
    import json

    path = "/srv/iguide/method_library"
    proc = _compose(tmp_path, "config", "--format", "json", env={"AGENT_METHOD_LIBRARY_DIR": path})
    assert proc.returncode == 0, proc.stderr
    svc = json.loads(proc.stdout)["services"]["agent-api"]
    binds = {(v.get("source"), v.get("target")) for v in svc["volumes"] if v.get("type") == "bind"}
    assert (path, path) in binds
    assert ("/tmp/iguide_codeexec", "/tmp/iguide_codeexec") in binds
    assert svc["environment"]["AGENT_EXTRACTION"] == "1"
    assert svc["environment"]["AGENT_METHOD_LIBRARY_DIR"] == path


def test_compose_refuses_the_override_without_a_library_path(tmp_path):
    proc = _compose(tmp_path, "config", "--quiet")
    assert proc.returncode != 0 and "AGENT_METHOD_LIBRARY_DIR" in proc.stderr, proc.stderr
