"""Code-asset extractor (#4) — GitHub path.

Per .py file: AST-extract the API surface (top-level functions, classes, and their
methods) as CodeAsset blocks (signature + docstring + file path + module imports),
element_id-anchored. Detect entry points (a top-level ``main``/``run``, an
``if __name__ == '__main__'`` block, or argparse usage) and promote them to runnable
code manifests (consumed by the generic executor). Emits ``DEFINES`` provenance edges
(element → asset). Index-only for non-entry-point assets.

Reuse: doc_ids (element-anchored ids, workflow_id_for(code=True), mcp_tool_name_for),
the form ``fields`` inherited into source_fields (mirrors NotebookExtractor).
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    EMIT_LIBRARY,
    EMIT_MCP,
    EMIT_OPENSEARCH,
    KIND_CODE_BLOCK,
    KIND_METHOD_UNIT,
    AssetRecord,
    ExtractContext,
    Extractor,
    ExtractionResult,
    ProvenanceEdge,
)
from .doc_ids import (
    code_asset_doc_id,
    mcp_tool_name_for,
    resource_type_for,
    workflow_id_for,
)


def _rel_path(ctx: ExtractContext, path: str) -> str:
    repo_dir = ctx.extra.get("repo_dir")
    if repo_dir:
        try:
            return os.path.relpath(path, repo_dir)
        except ValueError:
            pass
    return os.path.basename(path)


def _inherited(ctx: ExtractContext) -> Tuple[List[str], Dict[str, Any]]:
    f = ctx.fields or {}
    tags = f.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    source_fields = {k: f[k] for k in ("authors", "contributor", "abstract", "description", "license", "doi")
                     if f.get(k)}
    return list(tags), source_fields


def _signature(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = [a.arg for a in node.args.args]
        if node.args.vararg:
            args.append("*" + node.args.vararg.arg)
        if node.args.kwarg:
            args.append("**" + node.args.kwarg.arg)
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {node.name}({', '.join(args)})"
    if isinstance(node, ast.ClassDef):
        bases = [getattr(b, "id", getattr(b, "attr", "")) for b in node.bases]
        return f"class {node.name}({', '.join(b for b in bases if b)})" if bases else f"class {node.name}"
    return ""


def _module_imports(tree: ast.AST) -> List[str]:
    mods: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module.split(".")[0])
    return sorted(set(mods))


def _func_params(node: ast.AST) -> List[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return [a.arg for a in node.args.args]
    return []


def _api_surface(tree: ast.AST) -> List[Tuple[str, str, ast.AST]]:
    """Return (qualified_name, kind, node) for top-level funcs/classes + methods."""
    out: List[Tuple[str, str, ast.AST]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append((node.name, "function", node))
        elif isinstance(node, ast.ClassDef):
            out.append((node.name, "class", node))
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and not sub.name.startswith("_"):
                    out.append((f"{node.name}.{sub.name}", "method", sub))
    return out


def _entry_point(tree: ast.AST) -> Tuple[bool, Optional[str], List[str]]:
    """(has_entry_point, entrypoint_func_or_None, params). Function mode if a
    top-level main/run exists, else script mode if a __main__ block / argparse."""
    top_funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for name in ("run_workflow", "main", "run"):
        if name in top_funcs:
            return True, name, _func_params(top_funcs[name])
    has_main_block = any(
        isinstance(n, ast.If) and "__main__" in ast.dump(n.test) for n in tree.body
    )
    # Merely IMPORTING argparse is not an entry point. Any module that imports it was being
    # promoted, which ships the whole file as `module_source` — a library module with an
    # argparse import for one optional helper looked exactly like a CLI. Require the full
    # pattern: the import PLUS a parser PLUS a parse_args() call.
    dumped = ast.dump(tree)
    uses_argparse = ("argparse" in dumped
                     and "ArgumentParser" in dumped
                     and "parse_args" in dumped)
    if has_main_block or uses_argparse:
        return True, None, []   # script mode
    return False, None, []


class CodeExtractor:
    name = "code"

    def _promote_units(self, assets: List[AssetRecord], source: str, *, rel_path: str,
                       ctx: ExtractContext, tags: Any, source_fields: Dict[str, Any]) -> Dict[str, Any]:
        """One MethodUnit per independently-callable top-level function in this module.

        A .py module is a *better* source of units than a notebook: it has no cell state, so
        far fewer functions are blocked by a runtime global. The verdicts still decide — only
        `callable` units get EMIT_LIBRARY — so a module-level `df = pd.read_csv(...)` blocks its
        readers here exactly as it does in a notebook.
        """
        from .analysis import analyze_module, build_unit_slice, iter_units, slice_sha
        from .analysis.signatures import (contract_invariants, contract_params,
                                  signature_of)
        from .contracts import ANALYZER_VERSION, CALLABLE, UnitContract
        from .doc_ids import method_unit_doc_id
        from .pkgmap import requirements_from_source

        import dataclasses

        verdicts, scope, summary = analyze_module(source)
        if summary.get("unparseable"):
            return summary
        try:
            nodes = dict(iter_units(ast.parse(source)))
        except SyntaxError:
            return summary

        parent_doc_id = code_asset_doc_id(ctx.anchor() or "repo", rel_path, "__module__")
        for qualname, callability in verdicts.items():
            node = nodes.get(qualname)
            if node is None:
                continue
            doc = ast.get_docstring(node) or ""
            _params = contract_params(node, doc)
            provenance = {"element_id": ctx.anchor(), "parent_doc_id": parent_doc_id,
                          "source_rel_path": rel_path, "commit_sha": ctx.commit_sha,
                          "extractor": self.name, "analyzer_version": ANALYZER_VERSION}
            slice_src = build_unit_slice(source, qualname, scope=scope, verdicts=verdicts,
                                         provenance=provenance)
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
                requirements=requirements_from_source(slice_src) if slice_src else {},
                provenance=provenance,
            )
            is_callable = callability.verdict == CALLABLE
            targets = [EMIT_OPENSEARCH]
            if is_callable and EMIT_LIBRARY in (ctx.targets or ()):
                targets.append(EMIT_LIBRARY)
            unit_doc_id = method_unit_doc_id(parent_doc_id, qualname)
            contents = f"{contract.signature}\n\n{contract.doc_summary}".strip()
            if not is_callable:
                contents += f"\n\n[not independently callable: {callability.reason}]"
            assets.append(AssetRecord(
                asset_id=unit_doc_id, kind=KIND_METHOD_UNIT,
                resource_type=resource_type_for(KIND_METHOD_UNIT), doc_id=unit_doc_id,
                emit_targets=targets, source_rel_path=rel_path,
                title=f"{qualname} — {rel_path}",
                contents=contents,
                unit=dataclasses.asdict(contract),
                slice_source=slice_src if is_callable else "",
                source_fields={**source_fields, "tags": sorted(set(tags or ()))},
                extracted={"parent_doc_id": parent_doc_id, "parent_type": "Code",
                           "callable": is_callable, "unit_name": qualname},
            ))
        return summary

    def extract(self, path: str, *, ctx: ExtractContext) -> ExtractionResult:
        rel_path = _rel_path(ctx, path)
        anchor = ctx.anchor() or "repo"
        tags, source_fields = _inherited(ctx)
        try:
            source = Path(path).read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, SyntaxError) as exc:
            return ExtractionResult(warnings=[f"code: skipped {rel_path}: {type(exc).__name__}: {exc}"])

        imports = _module_imports(tree)
        assets: List[AssetRecord] = []
        edges: List[ProvenanceEdge] = []

        # Callable units, through the SAME analyzer/slicer/library path the notebook extractor
        # uses. Code assets previously produced API-surface descriptions only, so `code` was an
        # element type the system claimed to extract methods from and did not: the corpus
        # library was 203 units, all of them `notebook`.
        unit_summary = self._promote_units(assets, source, rel_path=rel_path, ctx=ctx,
                                           tags=tags, source_fields=source_fields)

        for qualname, kind, node in _api_surface(tree):
            doc_id = code_asset_doc_id(anchor, rel_path, qualname)
            sig = _signature(node)
            doc = ast.get_docstring(node) or ""
            contents = f"{sig}\n{doc}\n# file: {rel_path}".strip()
            assets.append(AssetRecord(
                asset_id=doc_id,
                kind=KIND_CODE_BLOCK,
                resource_type=resource_type_for(KIND_CODE_BLOCK),
                doc_id=doc_id,
                emit_targets=[EMIT_OPENSEARCH],
                source_rel_path=rel_path,
                title=f"{qualname}  ({rel_path})",
                contents=contents,
                source_fields={**source_fields, "tags": tags},
                block={
                    "qualified_name": qualname, "kind": kind, "signature": sig,
                    "docstring": doc, "file_path": rel_path, "imports": imports,
                    "params": _func_params(node),
                },
                extracted={"parent_doc_id": anchor, "parent_type": "Code",
                           "parent_title": str((ctx.fields or {}).get("title") or "")},
            ))
            edges.append(ProvenanceEdge(src=anchor, rel="DEFINES", dst=doc_id, detail={"kind": kind}))

        # promote a per-file runnable entry point
        has_ep, entrypoint, params = _entry_point(tree)
        if has_ep:
            asset_id = f"{anchor}::codewf::{rel_path}"
            wid = workflow_id_for(asset_id, code=True)
            runnable_tool = mcp_tool_name_for(wid)
            assets.append(AssetRecord(
                asset_id=asset_id,
                kind=KIND_CODE_BLOCK,
                resource_type=resource_type_for(KIND_CODE_BLOCK),
                doc_id=asset_id,
                emit_targets=([EMIT_OPENSEARCH, EMIT_MCP] if EMIT_MCP in ctx.targets else [EMIT_OPENSEARCH]),
                source_rel_path=rel_path,
                title=f"{rel_path}  (runnable)",
                # Identify, do not advertise a callable tool — see
                # doc_ids.mcp_tool_name_for and notebook_extractor's matching marker.
                contents=f"[workflow {wid}] Entry point in {rel_path} "
                         f"({'function:' + entrypoint if entrypoint else 'script'} mode). "
                         f"Not directly callable; reuse the extracted functions.",
                source_fields={**source_fields, "tags": tags},
                runnable={
                    "workflow_id": wid, "runnable_tool": runnable_tool,
                    "mode": "function" if entrypoint else "script",
                    "entrypoint": entrypoint, "entrypoint_parameters": params,
                    "module_source": source,
                },
                extracted={"parent_doc_id": anchor, "parent_type": "Code", "runnable_tool": runnable_tool},
            ))
            edges.append(ProvenanceEdge(src=asset_id, rel="HAS_WORKFLOW", dst=wid,
                                        detail={"mcp_tool": runnable_tool}))

        return ExtractionResult(assets=assets, edges=edges, skill=None)


_: Extractor = CodeExtractor()  # type: ignore[assignment]

__all__ = ["CodeExtractor"]
