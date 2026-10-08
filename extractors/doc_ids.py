"""Stable, deterministic id derivation + the workflow_id <-> MCP tool-name bridge.

All ids are content-independent (position/path/name based) so re-ingesting a
repo upserts the same OpenSearch doc / manifest instead of duplicating. The
``mcp_tool_name_for`` function is the single source of truth for the find<->run
linkage: the extractor stores it in ``runnable.runnable_tool`` and the MCP
executor must register the workflow under the same name. See EXTRACTOR_DESIGN.md §4.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict

from .base import (
    KIND_CODE_BLOCK,
    KIND_DATASET,
    KIND_NOTEBOOK_BLOCK,
    KIND_METHOD_UNIT,
    KIND_PUBLICATION,
)

_SLUG_RE = re.compile(r"[^a-z0-9]+")

# kind -> OpenSearch resource-type (drives element_type synthesis + spatial filter)
_RESOURCE_TYPE = {
    KIND_NOTEBOOK_BLOCK: "NotebookBlock",
    KIND_CODE_BLOCK: "CodeAsset",
    KIND_DATASET: "Dataset",
    KIND_PUBLICATION: "PublicationMethodSpec",
    KIND_METHOD_UNIT: "MethodUnit",
}


def _sha1(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def normalize_repo_url(url: str) -> str:
    u = (url or "").strip().rstrip("/")
    u = re.sub(r"\.git$", "", u)
    return u.lower()


def repo_id(url: str) -> str:
    return f"repo_{_sha1(normalize_repo_url(url))}"


def resource_type_for(kind: str) -> str:
    return _RESOURCE_TYPE.get(kind, "Resource")


# ---- deterministic doc ids (see EXTRACTOR_DESIGN.md §2) -------------------
def notebook_block_doc_id(notebook_doc_id: str, order: int) -> str:
    return f"{notebook_doc_id}::block::{order}"


def code_asset_doc_id(repo_doc_id: str, rel_path: str, qualified_name: str) -> str:
    return f"{repo_doc_id}::code::{rel_path}::{qualified_name}"


def dataset_doc_id(element_id_or_filename: str) -> str:
    # Datasets are first-class platform elements; prefer the platform id when known.
    return str(element_id_or_filename)


def publication_methodspec_doc_id(publication_doc_id: str, n: int = 0) -> str:
    return f"{publication_doc_id}::methodspec" + (f"::{n}" if n else "")


def parent_doc_id(doc_id: str) -> str:
    """Recover the parent element id from a derived block/code/methodspec id."""
    return doc_id.split("::", 1)[0] if "::" in doc_id else doc_id


# ---- find<->run bridge ---------------------------------------------------
def workflow_id_for(asset_id: str, *, code: bool = False) -> str:
    prefix = "cwf" if code else "nbwf"
    return f"{prefix}_{_sha1(asset_id, 16)}"


def mcp_tool_name_for(workflow_id: str) -> str:
    """Agent-side name of the executor that can run *workflow_id*.

    The previous implementation returned ``mcp_run_<workflow_id>``, on the belief
    that the executor registers one tool per workflow. It does not:
    ``MCP_server/tools/generic_executor_tools.py`` registers exactly two fixed
    tools, ``run_notebook_workflow`` and ``run_code_element``, each taking the id
    as an *argument*. So every ``mcp_run_nbwf_*`` name ever emitted named a tool
    that could not exist, and one of them shipped into a SKILL.md ``allowed-tools``
    list where the model was told to invoke it.

    NOTE these executors are deliberately NOT reachable in beta: they are gated by
    ``AGENT_ALLOW_WORKFLOW_EXEC`` (default 0) and ``generic_executor_tools`` is not
    in ``langchain_mcp_tools.DEFAULT_MCP_MODULES``, because the execution body is a
    bare ``exec()`` of ingested notebook source inside the MCP server process
    (``generated_notebook_tools.py:43-47``). Reuse goes through the method library
    instead. This function therefore describes the *correct* name for a manifest
    record; it must not be used to tell a model what to call.
    """
    return "mcp_run_code_element" if workflow_id.startswith("cwf_") else "mcp_run_notebook_workflow"


def run_invocation_for(workflow_id: str) -> Dict[str, Any]:
    """The full, truthful invocation for *workflow_id* — tool name plus argument.

    A tool name alone is not actionable for these executors, since the workflow id
    travels as a parameter. Kept so manifests record a complete call if workflow
    execution is ever re-enabled behind a real sandbox.
    """
    tool = mcp_tool_name_for(workflow_id)
    key = "element_id" if tool.endswith("code_element") else "workflow_id"
    return {"tool": tool, "args": {key: workflow_id}}


def method_unit_doc_id(parent_doc_id: str, qualified_name: str, rel_path: str = "") -> str:
    """Doc id for one callable unit.

    Keyed on the unit's NAME, not on a cell index, so it survives cell reordering — unlike
    ``{element}::block::{order}``, where inserting a cell renames every doc after it.
    """
    tail = f"{rel_path}::{qualified_name}" if rel_path else qualified_name
    return f"{parent_doc_id}::unit::{tail}"


def slugify(text: str, *, max_len: int = 64) -> str:
    """Lowercase-hyphen slug matching skills.py name regex ^[a-z0-9][a-z0-9-]{0,63}$."""
    s = _SLUG_RE.sub("-", (text or "").strip().lower()).strip("-")
    if not s:
        s = "skill"
    if not s[0].isalnum():
        s = f"s{s}"
    return s[:max_len].rstrip("-")


__all__ = [
    "normalize_repo_url", "repo_id", "resource_type_for",
    "notebook_block_doc_id", "code_asset_doc_id", "dataset_doc_id",
    "publication_methodspec_doc_id", "parent_doc_id", "method_unit_doc_id",
    "workflow_id_for", "mcp_tool_name_for", "run_invocation_for", "slugify",
]
