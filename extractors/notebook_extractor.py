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
from .analysis.citations import platform_citations
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



_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.M)
_DEF_RE = re.compile(r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_]\w*)", re.M)
_TAG_RE = re.compile(r"<[^>]{1,40}>")
_MDLINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
# Cells that are neither narrated nor named still fall into a few recognisable jobs.
# Ordered: the first match wins, so the more specific verbs come first.
_CODE_HINTS = (
    ("pip install", "Install dependencies"),
    ("conda install", "Install dependencies"),
    (".to_crs(", "Reproject"),
    ("sjoin", "Spatial join"),
    (".buffer(", "Buffer"),
    (".dissolve(", "Dissolve"),
    (".to_file(", "Write output"),
    (".to_csv(", "Write output"),
    (".fit(", "Fit a model"),
    ("read_file(", "Load data"),
    ("read_csv(", "Load data"),
    ("read_parquet(", "Load data"),
    ("rasterio.open(", "Load raster"),
    (".plot(", "Plot"),
    ("plt.", "Plot"),
    ("explore(", "Map"),
    ("folium", "Map"),
    ("import ", "Import libraries"),
)


def _is_prose(text: str) -> bool:
    """Is this comment a sentence, or is it commented-out code?

    Both live behind a `#` and only one is a title. Decided by parsing rather than by looking
    for `=` or `import`: `model = model.to("cuda")` is valid Python and "Define callbacks" is
    not, which is exactly the distinction. A lone word like `Mapping` parses as an expression
    and is still prose, so bare names are excluded from the code verdict.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return True
    body = tree.body
    if len(body) == 1 and isinstance(body[0], ast.Expr):
        return isinstance(body[0].value, (ast.Name, ast.Constant))
    return False


def _leading_comment(code: str) -> str:
    """The comment block at the top of a cell, which is the author naming the cell in code."""
    for line in (code or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            break
        text = stripped.lstrip("#").strip()
        # Banner rules (`# ====`, `# ----`) are decoration, not a name; skip past them.
        if text and re.sub(r"[^\w]", "", text) and _is_prose(text):
            return text
    return ""


def _block_title(md_context: str, code: str, order: int, constructs) -> str:
    """A name for one extracted cell that says what the cell IS.

    Blocks were titled ``<element> — cell 12``, which identifies the cell and describes nothing.
    The notebook's own markdown heading was sitting unused in ``markdown_context`` one field away:
    ``1c. Project All Files to the Same CRS`` instead of ``cell 10``. Authors write these names for
    human readers, which is exactly the audience a retrieved block has — so every tier here reads a
    name the author already wrote, and only the last two invent one.

    Order of preference, each falling through when it yields nothing usable:

    1. the cell's own markdown heading — the author's name for this step;
    2. its first sentence of prose, when the author narrated but wrote no heading;
    3. the comment block at the top of the code — the same act, one layer down;
    4. what the cell defines, for a cell whose job is defining ``reproject_tif``;
    5. what the code evidently does, by construct or call;
    6. ``cell N``, which is where we started and still beats an empty title.

    Tiers 3-5 were rebuilt after measuring: of the 871 cells that reached this point with no
    markdown at all, classified constructs fired on 15. Leading comments and ``def`` lines are
    what those cells actually carry.
    """
    for match in _HEADING_RE.finditer(md_context or ""):
        heading = match.group(1).strip().strip("*_`")
        if heading:
            return _clip(heading)

    prose = " ".join((md_context or "").split())
    if prose:
        sentence = re.split(r"(?<=[.!?])\s", prose, maxsplit=1)[0]
        return _clip(sentence.rstrip(".") if len(sentence) >= 12 else prose)

    comment = _leading_comment(code)
    if len(comment) >= 8:
        return _clip(comment.rstrip(":."))

    defined = _DEF_RE.findall(code or "")
    if defined:
        shown = ", ".join(f"{name}()" for name in defined[:2])
        return _clip(f"Define {shown}" + (" +more" if len(defined) > 2 else ""))

    for construct in (constructs or []):
        packages = (construct.get("detail") or {}).get("packages")
        if packages:
            return _clip(f"Install {', '.join(map(str, packages[:3]))}")
    # Comments are excluded: this tier reports what the cell DOES, and a commented-out
    # `# import geopandas` is a line the cell deliberately does not run.
    lowered = "\n".join(ln for ln in (code or "").lower().splitlines()
                        if not ln.lstrip().startswith("#"))
    for needle, name in _CODE_HINTS:
        if needle in lowered:
            return name
    return f"cell {order}"


def _clip(text: str, limit: int = 72) -> str:
    """Flatten one line of authored markdown into a plain-text title.

    Titles are rendered as text everywhere they are used, so `<b>` and `**` arrive as literal
    characters rather than as emphasis — markup that meant something in the notebook becomes
    noise in a search result.
    """
    text = _TAG_RE.sub("", text or "")
    text = _MDLINK_RE.sub(r"\1", text)
    text = " ".join(text.replace("**", "").replace("*", "").replace("`", "").split())
    return text if len(text) <= limit else text[:limit].rstrip(" ,;:-") + "…"


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
        # Element references the author wrote into the notebook. Collected across EVERY cell,
        # including markdown, which is why this sits above the two `continue`s below: a markdown
        # cell that is not followed by a code cell never becomes `md_context` and would
        # otherwise be dropped entirely — and prose is exactly where a citation lives.
        cited: set = set()

        for order, cell in enumerate(nb.cells):
            cited.update(platform_citations(str(cell.source or "")))
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

            doc_id = notebook_block_doc_id(nb_doc_id, order)
            contents = (f"{md_context}\n\n{source}" if md_context else source).strip()
            assets.append(AssetRecord(
                asset_id=doc_id,
                kind=KIND_NOTEBOOK_BLOCK,
                resource_type=resource_type_for(KIND_NOTEBOOK_BLOCK),
                doc_id=doc_id,
                emit_targets=[EMIT_OPENSEARCH],
                source_rel_path=rel_path,
                # Named for what the cell IS, not where it sits. See `_block_title`.
                title=f"{_block_title(md_context, source, order, constructs)} — {title}",
                contents=contents,
                # `tags` is the SUBMITTER's vocabulary and stays that way. Import names used to be
                # merged in here, which put `os`, `time`, `copy`, `html` and `__future__` in a
                # field whose only claim is "what this element is about". Measured: 338 of 349
                # indexed unit documents carried at least one stdlib module name as a tag, 43 of
                # 432 distinct tags were stdlib names, and the platform's own tags contain zero
                # across all 180 notebook elements — so every one of them was invented here.
                # The imports were already in `block.imports`; nothing was gained by the merge.
                source_fields={**source_fields, "tags": list(form_tags)},
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

        # ---- cross-element citations ---------------------------------------------------
        # The only edges this extractor emits whose dst is NOT derivable from src. Everything
        # else here is `{nb_doc_id}::…`, i.e. a restatement of the id-building rule in doc_ids.
        # Measured over the 174 cached corpus notebooks: 99 edges (CITES 92, USES 7) from 30
        # notebooks to 40 targets, 41 of 41 of which resolve to a live platform element.
        #
        # A self-citation is dropped rather than emitted: a notebook whose prose links to its own
        # element page is a real sentence but a self-loop, and one of the 174 does exactly that.
        for cite in sorted(cited):
            if cite.element_id == nb_doc_id:
                continue
            edges.append(ProvenanceEdge(
                src=nb_doc_id, rel=cite.rel, dst=cite.element_id,
                # `path_hint` is the URL's own segment ('notebooks'), NOT an assertion about the
                # target's type — resolving that needs the platform API and this runs offline.
                detail={"by": "platform_url", "host": cite.host, "path_hint": cite.path_hint,
                        "confidence": "high"}))

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
                source_fields=source_fields, form_tags=form_tags,
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
                source_fields={**source_fields, "tags": list(form_tags)},
                extracted={"parent_doc_id": nb_doc_id, "parent_type": "Notebook",
                           "runnable_tool": runnable_tool},
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
                # A SKILL.md's front matter is read by humans deciding whether the skill applies.
                # "os, time, copy, __future__" answers no question anyone asks of it.
                tags=list(form_tags),
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
                       source_fields: Dict[str, Any], form_tags: Any) -> Dict[str, Any]:
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
            _params = contract_params(node, doc, nodes)
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
                source_fields={**source_fields, "tags": list(form_tags)},
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
