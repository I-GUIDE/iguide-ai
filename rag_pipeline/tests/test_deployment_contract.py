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
