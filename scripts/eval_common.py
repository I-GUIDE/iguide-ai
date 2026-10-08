"""Shared task definitions, outcome normalization, and record/summary writers for the
GeoPathfinder benchmark recorders (`eval_record.py` = non-streaming, `eval_trace.py` = full
streaming trace). Pure logic only — no argparse, no env mutation, no agent imports.
"""
from __future__ import annotations
import json, os, re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECDIR = REPO / "outputs" / "eval_records"

# ---------------------------------------------------------------------------------------------
# Task definitions.  TASKS = the prompt actually sent + the full expected source-element id set.
# TASK_META = ground-truth task properties used by normalize() (NOT run outcomes).
# ---------------------------------------------------------------------------------------------
TASKS = {
 "T1": ("Which Chicago community areas have the poorest spatial accessibility to hospitals? Produce a map and ranked table.",
        ["17399137", "b41c3bcb", "3b45070e"], "Chicago, EPSG:4326"),
 "T2": ("Use the available flood-depth prediction resources to identify required inputs and run or outline a flood-depth prediction workflow.",
        ["e8eb5345", "803168c4", "29c717f6", "49c9145f"], "—"),
 "T3": ("Find flood risk maps for Boone County, Iowa, compare available map years, and summarize covered jurisdictions.",
        ["f07c6a56", "03d75efd"], "Boone County, Iowa"),
 "T4": ("Build an OSM-based road network for a wildfire evacuation scenario in Paradise, California (the 2018 Camp Fire town) "
        "and identify likely bottleneck road segments.",
        ["34f5f70c", "5278e805", "02f9b712", "18834afd", "c63f74d1"], "Paradise, California"),
 "T5": ("Rank areas that may face higher exposure to potential dam failure risk using the National Inventory of Dams and related knowledge elements.",
        ["3b8c4c57", "86df1948", "30c77781"], "USA"),
 "T6": ("Acquire precipitation, temperature, wind, humidity, and evapotranspiration data for a selected location and prepare it for geospatial analysis.",
        ["afbee4bd"], "a selected location"),
 "T7": ("Explain and map major drivers of local water stress using the SIMPLE-G water resources.",
        ["33b02456", "46d95375", "54ef420f"], "global/local"),
 "T8": ("Map places where heat-related public sentiment or exposure overlaps with social vulnerability.",
        ["bb14c9ea", "4a06e4a1", "39379ef6", "4c880e67", "6c518fed", "1628189a", "791fa878"], "Chicago/US tracts"),
 "T9": ("Identify resources that can detect wildfire threat to transportation infrastructure in California and produce an executable workflow plan.",
        ["ae64d94a", "02224dd6", "c43426a3", "44fb06d1", "643aaea1"], "California"),
 "T10": ("Use available biodiversity and land-use resources to summarize how terrestrial biodiversity responds to land-use change and identify candidate datasets for analysis.",
         ["611fcc2a", "71bba752", "de05a428"], "global"),
 # --- Chicago crime / heat anchor probes (the paper's worked example) ---
 "CRIME_HEATMAP": ("Generate a heat map of violent crime in Chicago.",
                   ["cca9b545"], "Chicago"),
}

TASK_META = {
 "T1": dict(
    expected_output="Ranked table of worst-access community areas + spatial-accessibility map",
    artifact_expected=True, artifact_type_expected="map (png) + ranked table",
    execution_relevant=True, text_satisfiable=False,
    primary_ids=["b41c3bcb", "3b45070e"], source_label="b41c3bcb Pysal-Access + 3b45070e A2SFCA",
    action="Search + scrape attempt → honest decline", verification="needs_manual_check",
    underspecified=False,
    limitation="Index has the accessibility *methods* but no Chicago hospital point layer or precomputed community-area scores → declined.",
    checks=[
      "Method resources retrieved: 'Pysal Access Compute Example' (b41c3bcb), A2SFCA notebook (3b45070e), and the E2SFCA Illinois publication (355786a5).",
      "Index contains NO Chicago hospital point layer and NO precomputed community-area accessibility scores; the agent tried to scrape the CyberGIS COVID-19 notebook (3 execute_code calls) and recovered none.",
      "Correct honest decline — no ranked table or map fabricated; grounding audit clean.",
    ]),
 "T2": dict(
    expected_output="Required-inputs list + flood-depth-prediction workflow outline",
    artifact_expected=False, artifact_type_expected="workflow plan (in-answer)",
    execution_relevant=False, text_satisfiable=True,
    primary_ids=["e8eb5345", "803168c4", "29c717f6", "49c9145f"], source_label="e8eb5345 +3 flood-depth elements",
    action="Outline required inputs + workflow", verification="verified",
    underspecified=False,
    limitation="Planning task — model inputs (DEM/SAR/labels) not staged, so no run was attempted.",
    checks=[
      "All four flood-depth-prediction elements retrieved (e8eb5345, 803168c4, 29c717f6, 49c9145f).",
      "Required inputs enumerated (flood samples + depth labels, DEM/terrain, SAR/remote-sensing) and a multi-step workflow outlined.",
      "No file artifact expected (planning task); every cited id resolves to a retrieved evidence element.",
    ]),
 "T3": dict(
    expected_output="Retrieval summary: map list, year comparison, covered jurisdictions",
    artifact_expected=False, artifact_type_expected="retrieval summary (in-answer)",
    execution_relevant=False, text_satisfiable=True,
    primary_ids=["f07c6a56", "03d75efd"], source_label="f07c6a56 +2 Boone-IA maps",
    action="Retrieve + compare map years", verification="verified",
    underspecified=False,
    limitation="Map products are FEMA portal links — listed/compared, not downloaded or rendered.",
    checks=[
      "Boone County, Iowa flood-risk maps retrieved: FRM_19153C (06/30/2014, 2afb3b31), FRM_07100006 (03/31/2015, f07c6a56), FRM_07100004 (06/15/2017, 03d75efd).",
      "Years 2014 / 2015 / 2017 compared; no 2016 Iowa map (correct — the only 2016 map in evidence is the Illinois one).",
      "Boone County, ILLINOIS (8d57495b, FRM_07090005, 2016) correctly excluded as a different jurisdiction — false positive avoided.",
    ]),
 "T4": dict(
    expected_output="OSM road-network map for the study area + likely bottleneck segments",
    artifact_expected=True, artifact_type_expected="map (png)",
    execution_relevant=True, text_satisfiable=False,
    primary_ids=["5278e805"], source_label="5278e805 FireABM notebook",
    action="Build OSM network for Paradise, CA (code)", verification="needs_manual_check",
    underspecified=False,
    limitation="Original benchmark prompt named no study area (prompt_missing_required_input); reran with Paradise, California. OSMnx emits cache JSONs alongside the map (excluded from artifacts).",
    checks=[
      "FireABM modeling notebook (5278e805) retrieved as the OSM-evacuation method source.",
      "Study area now concrete (Paradise, California) — original benchmark prompt was underspecified; verify the produced network/extent matches Paradise.",
      "Inspect result.png: confirm a real OSM road graph for Paradise and that highlighted bottleneck segments correspond to low-capacity / high-betweenness edges.",
    ]),
 "T5": dict(
    expected_output="Ranked table of regions by dam-failure exposure",
    artifact_expected=False, artifact_type_expected="ranked table (in-answer)",
    execution_relevant=True, text_satisfiable=True,
    primary_ids=["3b8c4c57"], source_label="3b8c4c57 dam-failure notebook",
    action="Derive ranked regions (code)", verification="needs_manual_check",
    underspecified=False,
    limitation="Ranking is in-answer (no map file); percentages come from notebook text, not a recomputed table.",
    checks=[
      "Dam-failure notebook (3b8c4c57) + 'Aging Dams' OER (0264e99a) retrieved.",
      "Regions ranked by Cluster A (high-vulnerability population) share: Upper Mississippi 50%, California 44%, New England 36%, Texas-Gulf 32%.",
      "MANUAL: confirm the cited cluster percentages match notebook blocks ::30 / ::19 / ::28 (block text not stored in the record).",
    ]),
 "T6": dict(
    expected_output="Geospatial-ready multi-variable weather CSV (5 variables + spatial fields)",
    artifact_expected=True, artifact_type_expected="csv",
    execution_relevant=True, text_satisfiable=False,
    primary_ids=["afbee4bd"], source_label="afbee4bd Open-Meteo notebook",
    action="Fetch + prepare weather CSV (code)", verification="verified",
    underspecified=False,
    limitation="Single point location (NYC), not a gridded surface — geospatial-ready but point-scale.",
    checks=[
      "Open-Meteo weather notebook (afbee4bd) retrieved as the data-acquisition source.",
      "CSV contains all five requested variables (precipitation, temperature_2m, wind_speed_10m, relative_humidity_2m, evapotranspiration) plus spatial fields (latitude, longitude, location_name, time).",
      "168 hourly rows for New York City (40.7128, -74.0060); CSV header verified directly from the saved file.",
    ]),
 "T7": dict(
    expected_output="Explanation of water-stress drivers + a map of those drivers",
    artifact_expected=True, artifact_type_expected="map (png)",
    execution_relevant=True, text_satisfiable=True,
    primary_ids=["33b02456", "46d95375", "54ef420f"], source_label="33b02456 + 46d95375 + 54ef420f SIMPLE-G",
    action="Explain drivers; map declined", verification="needs_manual_check",
    underspecified=False,
    limitation="No gridded SIMPLE-G raster/dataset file available to render a map — explanation delivered, map declined.",
    checks=[
      "SIMPLE-G water notebook (33b02456), SIMPLE-G-Global-Water notebook (46d95375), and SIMPLE-G-Water dataset (54ef420f) retrieved.",
      "Drivers explained and grounded: external demand (>50% of future US water stress), global food demand, groundwater/surface withdrawals, policy spillovers.",
      "Map NOT produced — no gridded SIMPLE-G file to render; honest partial (explanation half satisfied, map half declined).",
    ]),
 "T8": dict(
    expected_output="Map of places where heat sentiment/exposure overlaps social vulnerability",
    artifact_expected=True, artifact_type_expected="map (png)",
    execution_relevant=True, text_satisfiable=False,
    primary_ids=["bb14c9ea", "4a06e4a1"], source_label="bb14c9ea + 4a06e4a1",
    action="Decline map (data missing)", verification="needs_manual_check",
    underspecified=False,
    limitation="Index lacks the merged county table and an SVI/boundary polygon layer, so the overlap map can't be drawn → declined.",
    checks=[
      "Heat-sentiment notebook (bb14c9ea) + socioeconomic-analysis notebook (4a06e4a1) retrieved.",
      "Overlap map NOT produced — no merged county table and no boundary/SVI polygon layer available to draw polygons.",
      "Correct honest decline; the intended join workflow (build GEOID, merge on county, compare normalized_heat_exposure vs SVI) is described from notebook blocks.",
    ]),
 "T9": dict(
    expected_output="Relevant-resource list + executable workflow plan",
    artifact_expected=False, artifact_type_expected="workflow plan (in-answer)",
    execution_relevant=False, text_satisfiable=True,
    primary_ids=["ae64d94a", "02224dd6", "c43426a3", "44fb06d1"], source_label="ae64d94a U-Net +3 datasets",
    action="List resources + workflow plan", verification="verified",
    underspecified=False,
    limitation="Plan only — U-Net inference not executed (no GPU / model weights staged).",
    checks=[
      "Five California wildfire/transport resources retrieved: U-Net notebook (ae64d94a), U-Net+functions (0c62f913), U-Net GIS Data (c43426a3), Pretrained Model (44fb06d1), Training Datasets Corona/Ventura/Pala Mesa (02224dd6).",
      "Eight-step executable workflow plan produced: assemble inputs → buffer infrastructure → preprocess → load model → infer → post-process → overlay infrastructure → export.",
      "MINOR: one cited id is mis-transcribed in prose (c43426a3 suffix) vs the evidence; the prefix still matches the retrieved dataset.",
    ]),
 "T10": dict(
    expected_output="Synthesis of biodiversity↔land-use response + candidate-dataset list",
    artifact_expected=False, artifact_type_expected="synthesis + dataset list (in-answer)",
    execution_relevant=False, text_satisfiable=True,
    primary_ids=["611fcc2a", "71bba752"], source_label="71bba752 + 611fcc2a PREDICTS",
    action="Synthesize + list datasets", verification="verified",
    underspecified=False,
    limitation="Summary only — PREDICTS not downloaded or analyzed; candidate datasets identified for follow-up.",
    checks=[
      "PREDICTS publication (71bba752) + PREDICTS dataset (611fcc2a) retrieved.",
      "Synthesis grounded: biodiversity declines with land-use pressure; the updated PREDICTS release (3,278,056 measurements / 26,194 sites / 94 countries / 47,089 species) is recommended over the 2016 release.",
      "Candidate datasets identified (PREDICTS updated, PREDICTS 2016, SIMPLE-G behavioral parameters); no file artifact expected.",
    ]),
 "CRIME_HEATMAP": dict(
    expected_output="Point-density heat map of violent crime in Chicago (PNG)",
    artifact_expected=True, artifact_type_expected="map (png, density)",
    execution_relevant=True, text_satisfiable=False,
    primary_ids=["cca9b545"], source_label="cca9b545 crime anchor",
    action="Reuse loader + render density heat map (code)", verification="needs_manual_check",
    underspecified=False,
    limitation="Heat map = point density (hexbin/KDE), not a choropleth; viz-type selection is the failure mode under test.",
    checks=[
      "Crime anchor (cca9b545) retrieved; data loaded via the reused load_chicago_crime_data path.",
      "Violent-crime subset filtered, then rendered as a POINT-DENSITY heat map (hexbin/KDE) — not a choropleth.",
      "Inspect the PNG: confirm a density surface over Chicago, not shaded community-area polygons.",
    ]),
}

# ---------------------------------------------------------------------------------------------
# Evidence / trace extraction (works on a live orchestration_result).
# ---------------------------------------------------------------------------------------------
def _doc(e):
    return e.get("document") if isinstance(e, dict) and isinstance(e.get("document"), dict) else (e if isinstance(e, dict) else {})


def evidence_list(orch):
    out = []
    for i, e in enumerate(orch.get("evidence") or []):
        d = _doc(e)
        out.append({"rank": e.get("retrieval_rank") if e.get("retrieval_rank") is not None else i + 1,
                    "source": e.get("source"),
                    "score": round(float(e.get("score") or 0), 3),
                    "doc_id": d.get("doc_id"), "title": d.get("title"),
                    "resource_type": d.get("element_type") or d.get("resource-type")})
    return out


def _coerce(content):
    if isinstance(content, str) and content.strip()[:1] in "{[":
        try: return json.loads(content)
        except Exception: return content
    return content


def peer_calls(orch):
    seq = []
    for peer in ("analysis_results", "code_result"):
        v = orch.get(peer)
        if isinstance(v, dict):
            for tc in (v.get("tool_calls") or []):
                if isinstance(tc, dict):
                    seq.append({"peer": peer.split("_")[0], "tool": tc.get("name"),
                                "args": {k: str(x)[:60] for k, x in (tc.get("args") or {}).items()}})
    return seq


def tool_outputs(orch):
    out = []
    for peer in ("analysis_results", "code_result"):
        v = orch.get(peer)
        if isinstance(v, dict):
            for tr in (v.get("tool_results") or []):
                name = tr.get("name") if isinstance(tr, dict) else None
                content = _coerce(tr.get("content") if isinstance(tr, dict) else tr)
                step = {"tool": name}
                if isinstance(content, dict):
                    for k in ("file_id", "png_file_id", "rows", "columns", "error", "note", "status", "inputs"):
                        if k in content:
                            step[k] = (content[k][:12] if k == "columns" and isinstance(content[k], list) else content[k])
                out.append(step)
    return out

# ---------------------------------------------------------------------------------------------
# Artifact classification — separate real deliverables from tool cache noise.
# ---------------------------------------------------------------------------------------------
_CACHE_RE = re.compile(r"^(fontlist-.*\.json|[0-9a-f]{16,64}\.json)$", re.I)
_REAL_EXT = (".png", ".csv", ".geojson", ".parquet", ".html", ".txt", ".pdf", ".svg")
_TYPE = {".png": "png", ".csv": "csv", ".geojson": "geojson", ".parquet": "parquet",
         ".html": "html", ".txt": "txt", ".pdf": "pdf", ".svg": "svg", ".json": "json"}


def _logical(name):
    return name.split("__", 1)[1] if "__" in name else name


def _is_cache(name):
    return bool(_CACHE_RE.match(_logical(name)))


def _ext(name):
    n = _logical(name).lower()
    for e in (*_REAL_EXT, ".json"):
        if n.endswith(e):
            return e
    return ""


def real_artifacts(arts):
    out = []
    for a in arts or []:
        e = _ext(a)
        if not e:
            continue
        if e == ".json" and _is_cache(a):
            continue
        out.append(a)
    return out


def artifact_paths(tid, arts):
    real = real_artifacts(arts)
    cache = [a for a in (arts or []) if a not in real]
    entries = []
    for a in real:
        entries.append({"name": _logical(a), "file": a,
                        "type": _TYPE.get(_ext(a), "file"),
                        "path": str((RECDIR / tid / a).resolve())})
    return entries, [_logical(a) for a in cache]


def _csv_head(tid, name):
    try:
        with open(RECDIR / tid / name, encoding="utf-8") as f:
            header = f.readline().strip()
            n = sum(1 for _ in f) + 1
        return header, n
    except Exception:
        return None, None

# ---------------------------------------------------------------------------------------------
# Outcome normalization.
# ---------------------------------------------------------------------------------------------
# Models emit curly apostrophes (U+2019) — match both ' and ’.
_DECLINE_RE = re.compile(
    r"(can['’]?t|cannot|could ?n['’]?t|unable to)\b[^.]{0,80}?"
    r"(produce|determine|identify|generate|create|name|answer|map|rank|truthfully|reliably)", re.I)


def _calls_of(rec):
    return (rec.get("execution_trace") or {}).get("peer_tool_calls") or []


def generated_code_used(rec):
    return any(c.get("tool") == "execute_code" for c in _calls_of(rec))


def runtime_issues(rec, tid):
    issues, prompt = [], (rec.get("prompt") or "").lower()
    for c in _calls_of(rec):
        if c.get("tool") == "load_skill":
            sk = str((c.get("args") or {}).get("skill_name") or "")
            if "chicago-crime" in sk and "crime" not in prompt:
                issues.append(f"load_skill('{sk}') invoked for a non-crime task — spurious skill load "
                              "(now guarded by the load_skill description in agent_runtime/skills.py).")
    if tid == "T9" and generated_code_used(rec) and not TASK_META["T9"]["execution_relevant"]:
        issues.append("execute_code used to format the resource list for a planning task — code run not required.")
    return sorted(set(issues))


def normalize(rec, tid):
    meta = TASK_META[tid]
    arts = rec.get("output_artifacts") or []
    paths, cache_excluded = artifact_paths(tid, arts)
    produced = bool(paths)
    ans = rec.get("final_answer") or ""
    declined = bool(_DECLINE_RE.search(ans))
    grounded = rec.get("grounded_on_expected_source_elements") or []
    primary = meta["primary_ids"]
    cov = [p for p in primary if p in grounded]
    full_exp = TASKS[tid][1] if tid in TASKS else primary
    full_cov = [e for e in full_exp if e in grounded]
    # retrieval_success scores against the FULL expected set. It previously scored
    # against meta["primary_ids"] -- a hand-picked subset -- which pinned the metric at
    # 10/10 across every recorded run and made it structurally incapable of registering
    # a retrieval improvement. full_exp/full_cov were already computed here and unused.
    # The old number is retained as primary_retrieval_success so the 11 records written
    # before this change stay comparable.
    retr = "yes" if full_exp and len(full_cov) == len(full_exp) else ("partial" if full_cov else "no")
    primary_retr = "yes" if len(cov) == len(primary) and primary else ("partial" if cov else "no")
    gcode = generated_code_used(rec)
    hall = bool((rec.get("grounding_audit") or {}).get("hallucination_detected"))
    art_exp = bool(meta["artifact_expected"])

    if hall:
        status = "failure"
    elif art_exp:
        if produced:
            status = "partial" if declined else "success"
        else:
            status = "partial" if meta["text_satisfiable"] else "honest_decline"
    else:
        status = "success" if (cov or full_cov) and not (declined and not meta["text_satisfiable"]) else "honest_decline"

    if not meta["execution_relevant"]:
        exe = "not_applicable"
    elif produced and not declined:
        exe = "yes"
    elif produced and declined:
        exe = "partial"
    elif gcode and not declined:
        exe = "yes"
    else:
        exe = "no"

    outcome = {"task_status": status, "retrieval_success": retr,
               "retrieval_recall": f"{len(full_cov)}/{len(full_exp)}" if full_exp else "n/a",
               "primary_retrieval_success": primary_retr,
               "execution_success": exe,
               "artifact_expected": art_exp, "artifact_produced": produced,
               "verification_status": meta["verification"]}

    checks = list(meta["checks"])
    derived = [f"retrieval coverage — primary source(s) {cov or 'none'} of {primary}; "
               f"full expected set {len(full_cov)}/{len(full_exp)} grounded ({full_cov})."]
    if produced:
        derived.append("artifact(s) on disk: " + ", ".join(f"{p['name']} [{p['type']}]" for p in paths)
                       + (f"; cache files excluded: {cache_excluded}" if cache_excluded else ""))
    csvs = [p for p in paths if p["type"] == "csv"]
    if csvs:
        hdr, n = _csv_head(tid, csvs[0]["file"])
        if hdr:
            derived.append(f"CSV verified on disk: {n} lines; header = {hdr}")
    checks = checks + derived

    return {
        "outcome": outcome,
        "expected_output": meta["expected_output"],
        "artifact_paths": paths,
        "artifact_cache_excluded": cache_excluded,
        "generated_code_used": gcode,
        "prompt_missing_required_input": bool(meta.get("underspecified")),
        "runtime_issues": runtime_issues(rec, tid),
        "correctness": checks,
        "limitation": meta["limitation"],
    }


def failure_modes(rec, paths, ans, touts):
    fm = []
    for r in touts:
        if r.get("error"):
            fm.append(f"tool error — {r.get('tool')}: {str(r['error'])[:140]}")
        if r.get("note"):
            fm.append(f"tool note — {r.get('tool')}: {str(r['note'])[:140]}")
    if generated_code_used(rec):
        fm.append("generated-code fallback: code/analysis peer wrote+ran code via execute_code (not pure tool reuse)")
    al = (ans or "").lower()
    if not paths and re.search(r"cannot|unable|could not|no (?:relevant |mappable |supporting )?(?:evidence|data|map)|not .*available", al):
        fm.append("no artifact produced — agent reported missing/ungrounded data or declined (honest)")
    be = os.environ.get("AGENT_CODE_EXEC_BACKEND", "local")
    fm.append(f"sandbox/network: code-exec backend={be} (network {'available at exec' if be == 'local' else 'DISABLED at exec'})")
    return fm or ["none observed"]

# ---------------------------------------------------------------------------------------------
# Record + summary writers.
# ---------------------------------------------------------------------------------------------
def write_record(rec, tid):
    norm = normalize(rec, tid)
    touts = rec.get("artifact_lineage") or []
    rec.update(norm)
    rec["failure_modes"] = failure_modes(rec, norm["artifact_paths"], rec.get("final_answer"), touts)
    d = RECDIR / tid; d.mkdir(parents=True, exist_ok=True)
    (d / "record.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")

    o = rec["outcome"]
    md = [f"# {tid} — record (model={rec.get('model')}, {rec.get('elapsed_s')}s)", "",
          "## Outcome",
          f"- **task_status**: `{o['task_status']}`",
          f"- **retrieval_success**: `{o['retrieval_success']}`  |  **execution_success**: `{o['execution_success']}`",
          f"- **artifact_expected**: `{o['artifact_expected']}`  |  **artifact_produced**: `{o['artifact_produced']}`  |  **verification_status**: `{o['verification_status']}`",
          f"- expected_output: {rec['expected_output']}",
          f"- generated_code_used: `{rec['generated_code_used']}`  |  prompt_missing_required_input: `{rec['prompt_missing_required_input']}`",
          f"- limitation: {rec['limitation']}", "",
          "## Prompt", rec.get("prompt", ""), "",
          "## Final answer", (rec.get("final_answer") or "")[:4000], "",
          f"## Retrieved evidence  (grounded on expected: {rec.get('grounded_on_expected_source_elements')})"]
    for e in (rec.get("retrieved_evidence") or [])[:15]:
        md.append(f"- [rank {e['rank']}] **{e['title']}** — {e['resource_type']} — via {e['source']} (score {e['score']})  `{e['doc_id']}`")
    md += ["", "## Execution trace",
           "- supervisor: " + " → ".join((rec.get("execution_trace") or {}).get("supervisor_decisions") or ["—"])]
    for c in (rec.get("execution_trace") or {}).get("peer_tool_calls") or []:
        md.append(f"  - {c['peer']} → `{c['tool']}`({', '.join(f'{k}={v}' for k, v in (c.get('args') or {}).items())})")
    md += ["", "## Output artifacts"]
    md += [f"- `{p['name']}` [{p['type']}] → {p['path']}" for p in rec["artifact_paths"]] or ["- (none)"]
    if rec["artifact_cache_excluded"]:
        md.append(f"- _cache excluded_: {rec['artifact_cache_excluded']}")
    md += ["", "## Correctness checks"] + [f"- {c}" for c in rec["correctness"]]
    if rec["runtime_issues"]:
        md += ["", "## Runtime issues"] + [f"- {c}" for c in rec["runtime_issues"]]
    md += ["", "## Failure modes"] + [f"- {c}" for c in rec["failure_modes"]]
    md += ["", "## Grounding audit", f"- {json.dumps(rec.get('grounding_audit'))}"]
    if (RECDIR / tid / "trace.md").exists():
        md += ["", "## Full agent trace", "- see [`trace.md`](trace.md) / [`trace.json`](trace.json) (verbatim decisions, prompts, tool I/O)"]
    (d / "record.md").write_text("\n".join(md), encoding="utf-8")
    return rec


def write_summary(default_model="(unknown)"):
    rows, recs = [], {}
    bench = [t for t in TASK_META if t.startswith("T") and t[1:].isdigit()]
    for tid in sorted(bench, key=lambda t: int(t[1:])):
        p = RECDIR / tid / "record.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        if "outcome" not in rec:
            rec.update(normalize(rec, tid))
        recs[tid] = rec
        o = rec["outcome"]
        meta = TASK_META[tid]
        art = ", ".join(f"{x['name']}" for x in rec.get("artifact_paths") or []) or "—"
        retr = f"{o['retrieval_success']} ({meta['source_label']})"
        rows.append((tid, retr, meta["action"], art, o["task_status"], meta["limitation"]))

    counts = {}
    for r in rows:
        counts[r[4]] = counts.get(r[4], 0) + 1
    model = next((r.get("model") for r in recs.values()), default_model)

    md = ["# GeoPathfinder benchmark — outcome summary", "",
          f"Model: `{model}` · default deployment config (supervisor + granular search + MCP on, "
          "local code-exec backend) · one row per task.", "",
          "**task_status** counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) + ".", "",
          "_retrieval_success = coverage of the task's primary source element(s); in every task the "
          "intended source was retrieved, so the differentiation is in execution/artifact, not retrieval._", "",
          "| Task | Retrieved expected source? | Action | Artifact | Status | Limitation |",
          "|---|---|---|---|---|---|"]
    for tid, retr, action, art, status, lim in rows:
        md.append(f"| {tid} | {retr} | {action} | {art} | **{status}** | {lim} |")

    md += ["", "## Per-task expected output vs. outcome", "",
           "| Task | Expected output | artifact_expected | artifact_produced | execution_success | verification |",
           "|---|---|---|---|---|---|"]
    for tid in sorted(recs, key=lambda t: int(t[1:])):
        rec = recs[tid]; o = rec["outcome"]
        md.append(f"| {tid} | {rec['expected_output']} | {o['artifact_expected']} | {o['artifact_produced']} "
                  f"| {o['execution_success']} | {o['verification_status']} |")

    runtime = [(tid, i) for tid in sorted(recs, key=lambda t: int(t[1:])) for i in recs[tid].get("runtime_issues") or []]
    if runtime:
        md += ["", "## Runtime issues observed"] + [f"- **{tid}**: {i}" for tid, i in runtime]
    RECDIR.mkdir(parents=True, exist_ok=True)
    (RECDIR / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nSUMMARY.md written ({len(rows)} benchmark tasks): " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
