"""Notebook extractor (#1) — GitHub path.

Per .ipynb: split cells, IPython-transform code cells (NON-lossy via the R1 front
end), classify constructs, and emit:
  - one NotebookBlock AssetRecord per code cell -> OpenSearch (code + adjacent
    markdown, resolved_tools, file_io, imports);
  - one whole-notebook runnable descriptor (deterministic workflow_id + runnable_tool,
    function|script mode, entrypoint/params, module_source) when all code cells parse;
  - INCLUDES (notebook->block) + HAS_WORKFLOW (workflow->blocks) provenance edges;
  - the ordered pipeline -> a SkillSpec.

Reuse: extractors.r1_ipython_frontend (transform_cell / classify_line / _ast_extra /
_resolve_tool) replaces the lossy notebook_workflow_builder._sanitize_line. Entry-point
+ module-source logic is inlined (small) to keep the extractor self-contained.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nbformat

from .base import (
    EMIT_LIBRARY,
    EMIT_MCP,
    EMIT_OPENSEARCH,
    EMIT_SKILL,
    KIND_METHOD_UNIT,
    KIND_NOTEBOOK_BLOCK,
    AssetRecord,
    ExtractContext,
    Extractor,
    ExtractionResult,
    ProvenanceEdge,
    SkillSpec,
)
from .doc_ids import (
    mcp_tool_name_for,
    notebook_block_doc_id,
    resource_type_for,
    slugify,
    workflow_id_for,
)
from .fileclass import RASTER_EXT, TABULAR_EXT, VECTOR_EXT
from .r1_ipython_frontend import _ast_extra, classify_line, transform_cell

_ENTRYPOINT_PRIORITY = ("run_workflow", "main", "run")
_DATA_EXT = RASTER_EXT | VECTOR_EXT | TABULAR_EXT
_FILE_TOKEN_RE = re.compile(r"[\w./\-]+\.(?:" + "|".join(e.lstrip(".") for e in _DATA_EXT) + r")\b")


def _notebook_doc_id(ctx: ExtractContext, rel_path: str) -> str:
    # The notebook IS the knowledge element -> anchor on the platform element_id.
    if ctx.element_id:
        return ctx.element_id
    base = ctx.repo_id or "nb"
    stem = slugify(Path(rel_path).stem)
    return f"{base}::notebook::{stem}"


def _inherited(ctx: ExtractContext) -> tuple[str, list, dict]:
    """Return (title, form_tags, source_fields) inherited from the submission form."""
    f = ctx.fields or {}
    title = str(f.get("title") or "")
    form_tags = f.get("tags") or []
    if isinstance(form_tags, str):
        form_tags = [t.strip() for t in form_tags.split(",") if t.strip()]
    source_fields = {k: f[k] for k in ("authors", "contributor", "abstract", "description", "license", "doi")
                     if f.get(k)}
    return title, list(form_tags), source_fields


def _file_refs(text: str) -> List[str]:
    """Best-effort data-file references in a command or code string (referenced, not
    yet split into reads/writes — that is a follow-on refinement)."""
    return sorted({m.group(0) for m in _FILE_TOKEN_RE.finditer(text or "")})


def _classify_cell(source: str) -> Tuple[str, bool, List[Dict[str, Any]], List[str], List[str], List[str]]:
    """Return (transformed, parse_ok, constructs, resolved_tools, imports, file_refs)."""
    transformed, parse_ok, _note = transform_cell(source)
    constructs: List[Dict[str, Any]] = []
    tools: List[str] = []
    file_refs: List[str] = []
    for line in source.splitlines():
        res = classify_line(line)
        if not res:
            continue
        cat, detail = res
        constructs.append({"category": cat, "detail": detail})
        if detail.get("tool"):
            tools.append(detail["tool"])
        if detail.get("wrapper"):
            tools.append(detail["wrapper"])
        if detail.get("command"):
            file_refs.extend(_file_refs(detail["command"]))
    imports: List[str] = []
    for c in _ast_extra(transformed, 0):
        if c.category == "IMPORT" and c.detail.get("module"):
            imports.append(c.detail["module"])
        if c.category == "CLI_STEP" and c.detail.get("tool"):
            tools.append(c.detail["tool"])
    file_refs.extend(_file_refs(source))
    return (transformed, parse_ok,
            constructs, sorted(set(tools)), sorted(set(imports)), sorted(set(file_refs)))


def _top_level_functions(module_source: str) -> List[str]:
    try:
        tree = ast.parse(module_source)
    except SyntaxError:
        return []
    return [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _find_entrypoint(fns: List[str]) -> Optional[str]:
    for name in _ENTRYPOINT_PRIORITY:
        if name in fns:
            return name
    for name in fns:
        if not name.startswith("_"):
            return name
    return None


def _entrypoint_params(module_source: str, entrypoint: Optional[str]) -> List[str]:
    if not entrypoint:
        return []
    try:
        tree = ast.parse(module_source)
    except SyntaxError:
        return []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == entrypoint:
            return [a.arg for a in n.args.args]
    return []


class NotebookExtractor:
    name = "notebook"

    def extract(self, path: str, *, ctx: ExtractContext) -> ExtractionResult:
        nb = nbformat.read(path, as_version=4)
        rel_path = os.path.relpath(path, ctx.extra.get("repo_dir", os.path.dirname(path) or ".")) \
            if ctx.extra.get("repo_dir") else os.path.basename(path)
        nb_doc_id = _notebook_doc_id(ctx, rel_path)
        title_base = Path(rel_path).stem
        form_title, form_tags, source_fields = _inherited(ctx)
        title = form_title or title_base

        assets: List[AssetRecord] = []
        edges: List[ProvenanceEdge] = []
        ordered_steps: List[Dict[str, Any]] = []
        transformed_code: List[Tuple[int, str]] = []
        parsed_cells: List[Tuple[int, str]] = []
        all_parse_ok = True
        # R notebooks are a live false-positive risk: `nc <- st_read(...)` parses as valid
        # Python (`<` then unary `-`), and r1 rewrites `%%R ...` to `_cellmagic('R ...')`,
        # which also parses cleanly. Without this guard an R notebook would be promoted as
        # runnable Python.
        nb_language = str(((nb.get("metadata") or {}).get("kernelspec") or {}).get("language")
                          or ((nb.get("metadata") or {}).get("language_info") or {}).get("name")
                          or "").strip().lower()
        is_python_nb = nb_language in ("", "python")
        md_buffer: List[str] = []
        tags: set[str] = set()

        for order, cell in enumerate(nb.cells):
            if cell.cell_type == "markdown":
                md_buffer.append(str(cell.source or ""))
                continue
            if cell.cell_type != "code" or not str(cell.source or "").strip():
                continue
            source = str(cell.source)
            md_context = "\n".join(md_buffer).strip()
            md_buffer = []

            transformed, parse_ok, constructs, tools, imports, file_refs = _classify_cell(source)
            all_parse_ok = all_parse_ok and parse_ok
            transformed_code.append((order, transformed))
            if parse_ok:
                # Per-function promotion assembles the module from cells that PARSE, so one
                # bad cell costs that cell -- not every unit in the notebook.
                parsed_cells.append((order, transformed))
            tags.update(imports)

            doc_id = notebook_block_doc_id(nb_doc_id, order)
            contents = (f"{md_context}\n\n{source}" if md_context else source).strip()
            assets.append(AssetRecord(
                asset_id=doc_id,
                kind=KIND_NOTEBOOK_BLOCK,
                resource_type=resource_type_for(KIND_NOTEBOOK_BLOCK),
                doc_id=doc_id,
                emit_targets=[EMIT_OPENSEARCH],
                source_rel_path=rel_path,
                title=f"{title} — cell {order}",
                contents=contents,
                source_fields={**source_fields, "tags": sorted(set(form_tags) | set(imports))},
                block={
                    "code": source,
                    "transformed": transformed,
                    "markdown_context": md_context,
                    "constructs": constructs,
                    "resolved_tools": tools,
                    "imports": imports,
                    "file_io": {"referenced": file_refs},
                    "parse_ok": parse_ok,
                    "order": order,
                },
                extracted={"parent_doc_id": nb_doc_id, "parent_type": "Notebook",
                           "parent_title": title, "order": order},
            ))
            edges.append(ProvenanceEdge(src=nb_doc_id, rel="INCLUDES", dst=doc_id, detail={"order": order}))
            ordered_steps.append({"order": order, "tools": tools, "summary": (md_context or source.splitlines()[0])[:120]})

        # ---- per-function promotion --------------------------------------------------
        # Independent of the whole-notebook gate below. Previously a single unparseable cell
        # set all_parse_ok=False and the notebook yielded NOTHING reusable; now it costs that
        # cell only. Measured on the 14-notebook corpus: 40 of 41 functions are independently
        # callable, and every one of their slices imports in a clean subprocess.
        unit_summary: Dict[str, Any] = {}
        if parsed_cells and is_python_nb:
            unit_summary = self._promote_units(
                assets, edges, parsed_cells, nb_doc_id=nb_doc_id, rel_path=rel_path,
                title_base=title_base, title=title, ctx=ctx,
                source_fields=source_fields, form_tags=form_tags, tags=tags,
            )

        # whole-notebook runnable descriptor (promotion gate: every code cell parsed)
        if assets and all_parse_ok and is_python_nb:
            module_source = self._build_module_source(title_base, transformed_code)
            fns = _top_level_functions(module_source)
            entrypoint = _find_entrypoint(fns)
            mode = "function" if entrypoint else "script"
            wid = workflow_id_for(nb_doc_id)
            runnable_tool = mcp_tool_name_for(wid)
            wf_doc_id = f"{nb_doc_id}::workflow"
            runnable = {
                "workflow_id": wid,
                "runnable_tool": runnable_tool,
                "mode": mode,
                "entrypoint": entrypoint,
                "entrypoint_parameters": _entrypoint_params(module_source, entrypoint),
                "module_source": module_source,
            }
            assets.append(AssetRecord(
                asset_id=wf_doc_id,
                kind=KIND_NOTEBOOK_BLOCK,
                resource_type=resource_type_for(KIND_NOTEBOOK_BLOCK),
                doc_id=wf_doc_id,
                emit_targets=([EMIT_OPENSEARCH, EMIT_MCP, EMIT_SKILL] if EMIT_MCP in ctx.targets else [EMIT_OPENSEARCH]),
                source_rel_path=rel_path,
                title=f"{title} — workflow",
                # Identify the workflow WITHOUT naming a callable tool. The old marker
                # was "[runnable: mcp_run_<wid>]", a tool that cannot exist (see
                # doc_ids.mcp_tool_name_for) and whose executors are gated off anyway.
                # Telling the model to invoke it wasted a turn and taught it a fiction.
                contents=(f"[workflow {wid}] Whole-notebook workflow extracted from "
                          f"{rel_path} ({mode} mode). Not directly callable; reuse the "
                          f"extracted functions."),
                runnable=runnable,
                source_fields={**source_fields, "tags": sorted(set(form_tags) | tags)},
                extracted={"parent_doc_id": nb_doc_id, "parent_type": "Notebook", "runnable_tool": runnable_tool},
            ))
            edges.append(ProvenanceEdge(src=wf_doc_id, rel="HAS_WORKFLOW", dst=wid,
                                        detail={"mcp_tool": runnable_tool, "mode": mode}))
            skill = SkillSpec(
                name=slugify(title),
                description=str(ctx.fields.get("abstract") or ctx.fields.get("description")
                               or f"Run the {title} workflow extracted from {rel_path}."),
                # Deliberately empty: the only name we could put here is an executor
                # that is gated off and not registered. A SKILL.md shipped in this repo
                # advertised "mcp_run_nbwf_d01e717421c1b0ff" in allowed-tools.
                allowed_tools=[],
                tags=sorted(set(form_tags) | tags),
                ordered_steps=ordered_steps,
            )
        else:
            skill = None

        warnings: List[str] = []
        if assets and not all_parse_ok:
            warnings.append("not all code cells parsed; whole-notebook workflow not promoted "
                            "(blocks and callable units are unaffected).")
        if not is_python_nb:
            warnings.append(f"notebook kernel language is {nb_language!r}, not python; "
                            "no units or workflow promoted.")
        if unit_summary:
            warnings.append(
                f"callable units: {unit_summary.get('callable', 0)} of "
                f"{unit_summary.get('total', 0)} functions independently callable"
                + (f"; blocked_by={unit_summary.get('blocked_by')}"
                   if unit_summary.get("blocked_by") else ""))
        return ExtractionResult(assets=assets, edges=edges, skill=skill, warnings=warnings)

    def _promote_units(self, assets: List[AssetRecord], edges: List[ProvenanceEdge],
                       parsed_cells: List[Tuple[int, str]], *, nb_doc_id: str, rel_path: str,
                       title_base: str, title: str, ctx: ExtractContext,
                       source_fields: Dict[str, Any], form_tags: Any, tags: set) -> Dict[str, Any]:
        """Emit one MethodUnit asset per top-level function, with its contract.

        Only ``callable`` units are given EMIT_LIBRARY: a ``needs_globals`` unit is still
        indexed (so it is discoverable and its blocker is visible) but is never shipped as
        importable code, because making it run would mean inlining the module-level statements
        the slice builder exists to exclude.
        """
        from .analysis import analyze_module, build_unit_slice, iter_units, slice_sha
        from .analysis.signatures import (contract_invariants, contract_params,
                                  signature_of)
        from .pkgmap import requirements_from_source
        from .contracts import ANALYZER_VERSION, CALLABLE, UnitContract
        from .doc_ids import method_unit_doc_id
        import ast
        import dataclasses

        module_source = self._build_module_source(title_base, parsed_cells)
        try:
            verdicts, scope, summary = analyze_module(module_source)
        except Exception:
            return {}
        if not verdicts:
            return summary or {}

        try:
            tree = ast.parse(module_source)
        except SyntaxError:
            return summary or {}
        nodes = dict(iter_units(tree))

        for qualname, callability in verdicts.items():
            node = nodes.get(qualname)
            if node is None:
                continue
            doc = ast.get_docstring(node) or ""
            _params = contract_params(node, doc)
            slice_src = build_unit_slice(module_source, qualname, scope=scope,
                                         verdicts=verdicts,
                                         provenance={"element_id": ctx.anchor(),
                                                     "parent_doc_id": nb_doc_id,
                                                     "source_rel_path": rel_path,
                                                     "commit_sha": ctx.commit_sha,
                                                     "extractor": self.name,
                                                     "analyzer_version": ANALYZER_VERSION})
            contract = UnitContract(
                qualified_name=qualname,
                # A ClassDef was labelled "function" here, so the contract said `def X()`
                # for 35 units in the live registry. The caller CONSTRUCTS a class; being
                # told to call it as a function is a different action.
                unit_kind=("class" if isinstance(node, ast.ClassDef)
                           else "method" if "." in qualname else "function"),
                signature=signature_of(node),
                params=_params,
                # Enforceable form of what the params declare. Without this the expectation was
                # shown to the model and checked by nothing.
                invariants=[dataclasses.asdict(i) for i in contract_invariants(_params, node)],
                returns=(ast.unparse(node.returns) if getattr(node, "returns", None) else ""),
                docstring=doc,
                doc_summary=(doc.strip().splitlines() or [""])[0][:200],
                callability=callability,
                slice_sha=slice_sha(slice_src) if slice_src else "",
                library_symbol=qualname.split(".")[-1],
                # Derived from the SLICE, which is what actually gets imported. Left
                # unpopulated, every unit shipped `requirements: {}` while its slice imported
                # pandas and geopandas — so the contract the agent reads said "no
                # dependencies" and the import then failed inside the sandbox.
                requirements=requirements_from_source(slice_src) if slice_src else {},
                provenance={"element_id": ctx.anchor(), "parent_doc_id": nb_doc_id,
                            "source_rel_path": rel_path, "commit_sha": ctx.commit_sha,
                            "extractor": self.name, "analyzer_version": ANALYZER_VERSION},
            )
            unit_doc_id = method_unit_doc_id(nb_doc_id, qualname)
            is_callable = callability.verdict == CALLABLE
            targets = [EMIT_OPENSEARCH]
            if is_callable and EMIT_LIBRARY in (ctx.targets or ()):
                targets.append(EMIT_LIBRARY)
            # contents is the RETRIEVAL text: signature + summary, never the raw body — raw
            # code retrieves poorly against natural-language questions.
            contents = (f"{contract.signature}\n\n{contract.doc_summary}").strip()
            if not is_callable:
                contents += f"\n\n[not independently callable: {callability.reason}]"
            assets.append(AssetRecord(
                asset_id=unit_doc_id,
                kind=KIND_METHOD_UNIT,
                resource_type=resource_type_for(KIND_METHOD_UNIT),
                doc_id=unit_doc_id,
                emit_targets=targets,
                source_rel_path=rel_path,
                title=f"{qualname} — {title}",
                contents=contents,
                unit=dataclasses.asdict(contract),
                slice_source=slice_src if is_callable else "",
                source_fields={**source_fields, "tags": sorted(set(form_tags) | tags)},
                extracted={"parent_doc_id": nb_doc_id, "parent_type": "Notebook",
                           "callable": is_callable, "unit_name": qualname},
            ))
            edges.append(ProvenanceEdge(src=nb_doc_id, rel="DEFINES", dst=unit_doc_id,
                                        detail={"verdict": callability.verdict}))
        return summary or {}

    @staticmethod
    def _build_module_source(title: str, cells: List[Tuple[int, str]]) -> str:
        lines = [f'"""Generated workflow source from notebook: {title}."""', ""]
        for order, code in cells:
            lines.append(f"# Cell {order}")
            lines.append(code)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


_: Extractor = NotebookExtractor()  # type: ignore[assignment]

__all__ = ["NotebookExtractor"]
