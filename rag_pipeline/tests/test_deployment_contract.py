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
    from agent_runtime.skills import DEFAULT_SKILL_ROOTS

    text = _dockerfile()
    copied = any(f"COPY {root.name}/" in text or f"COPY .{root.name}/" in text
                 or "COPY .agents/" in text for root in DEFAULT_SKILL_ROOTS)
    assert copied, "no skills root is copied into the image; list_available_skills will be empty"


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
