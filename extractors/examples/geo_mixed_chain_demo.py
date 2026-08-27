"""Mixed chain: executed tools + code-gen as an INTERMEDIATE step, via file_id handles.

Division of labor:
  - reusable / expensive GIS ops  → executed tools (kb_run_geofunction, heatmap_image)
  - small bespoke transforms      → code-gen (execute_code) — e.g. the violent-crime filter

Everything is passed by file_id (GeoParquet), so code-gen output flows into the next tool
and vice-versa. This is the "code gen as an intermediate step" design.

    load_chicago_crime_data [TOOL] → file_id
        → filter violent [CODE-GEN: execute_code, input_files=[file_id]] → new file_id
        → point_hotspot_map [TOOL] → PNG file_id

Run:  AGENT_CODE_EXEC=1 AGENT_CODE_EXEC_BACKEND=local python -m extractors.examples.geo_mixed_chain_demo
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("AGENT_FILE_STORAGE_ROOT", "/tmp/iguide_kb_demo")
os.environ.setdefault("AGENT_CODE_EXEC", "1")
os.environ.setdefault("AGENT_CODE_EXEC_BACKEND", "local")  # host subprocess (dev)
os.environ.pop("AGENT_KB_BACKEND", None)
os.environ.pop("OPENSEARCH_NODE", None)

from pathlib import Path

NB = str(Path.home() / "Downloads" / "code_agent_analysis.ipynb")

FILTER_CODE = '''
import glob, pandas as pd                                # pandas-only: filtering rows needs no geopandas
src = sorted(glob.glob("*.parquet"))[0]                  # the staged crime file (from the load TOOL)
df = pd.read_parquet(src)
VIOLENT = {"HOMICIDE","BATTERY","ASSAULT","ROBBERY","CRIMINAL SEXUAL ASSAULT"}
v = df[df["primary_type"].astype(str).str.upper().isin(VIOLENT)]
v.to_parquet("violent.parquet")                          # → persisted as a new file_id artifact
print("violent rows:", len(v))
'''


def main() -> int:
    from extractors.submission import Submission
    from extractors.ingest import ingest_submission
    from rag_pipeline.search.agent_kb import agent_kb_search, get_kb_block
    from extractors.geo_handles import kb_run_geofunction, heatmap_image
    from agent_runtime.langchain_exec_tools import make_code_execution_tools

    if Path(NB).exists():
        ingest_submission(Submission.from_payload({
            "element_id": "ke_crimeagent", "element_type": "notebook",
            "source": {"github_url": str(Path(NB).parent), "notebook_file": Path(NB).name},
            "fields": {"title": "Chicago Crime Code Agent", "tags": ["chicago", "crime"]},
            "targets": ["opensearch"]}))
    doc_id = None
    for d in agent_kb_search("load chicago crime spatial join", size=8)["documents"]:
        code = ((get_kb_block(d["doc_id"]).get("source") or {}).get("extracted") or {}).get("block", {}).get("code", "")
        if "def load_chicago_crime_data" in code:
            doc_id = d["doc_id"]; break

    execute_code = make_code_execution_tools()[0]

    print("=" * 74)
    print("MIXED chain — executed tools + code-gen intermediate (file_id handles)")
    print("=" * 74)

    # 1) TOOL: load (extracted spatial function executed)
    r1 = json.loads(kb_run_geofunction(doc_id, "load_chicago_crime_data", "{}"))
    crime_fid = r1["file_id"]
    print(f"1. [TOOL]    kb_run_geofunction(load_chicago_crime_data) → {crime_fid}  ({r1['rows']} rows)")

    # 2) CODE-GEN: the small bespoke transform (violent filter) runs in execute_code,
    #    reading the tool's output file and writing a new file.
    res = json.loads(execute_code.func(code=FILTER_CODE, input_files=[crime_fid]))
    arts = res.get("artifacts") or []
    violent_fid = next((a["file_id"] for a in arts if str(a.get("filename", "")).endswith("violent.parquet")), None)
    print(f"2. [CODE-GEN] execute_code(filter violent) input_files=[{crime_fid}] → {violent_fid}")
    print(f"             exit={res.get('exit_code')}  stdout={ (res.get('stdout') or '').strip() }")

    # 3) TOOL: render the heat map from the code-gen output file
    r3 = json.loads(heatmap_image(violent_fid, "Violent crime density — mixed tool+code chain"))
    png_fid = r3.get("png_file_id")
    print(f"3. [TOOL]    heatmap_image({violent_fid}) → PNG {png_fid}")

    from agent_runtime.file_store import resolve_file_id
    print("\nlineage:  load[TOOL]=%s → filter[CODE]=%s → heatmap[TOOL]=%s" % (crime_fid, violent_fid, png_fid))
    print("✅ heat map:", resolve_file_id(png_fid) if png_fid else "(none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
