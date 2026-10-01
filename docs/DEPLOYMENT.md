# Deploying the I-GUIDE agent

Everything here is either enforced by a test in `rag_pipeline/tests/test_deployment_contract.py`
or was measured on this branch. Where a number appears, it came from a run — not an estimate.

---

## 1. What must be true before deploying

These are the settings whose absence produces a system that *looks* like it works. Each one has
a symptom that reads as something else, which is why they are listed with the symptom rather
than just the value.

| Variable | Value | If unset, the symptom is… |
|---|---|---|
| `AGENT_CHAT_API_KEY` | required | The service credential for callers without a browser (scripts, eval). A verified platform user (token mode) or demo mode is an alternative to it. **Unset disables the key check** outside token mode, where identity still refuses anonymous callers (prototype's model, kept by the 2026-10-01 integration); `/query` and `/query/batch` use the same check as the agent routes. |
| `AGENT_CORS_ORIGINS` | explicit allowlist | Browser blocks every request from the prototype; looks like a dead backend. |
| `AGENT_KB_BACKEND` | `opensearch` | **Silently reads a local file store.** The symptom is "fewer results", which reads as a retrieval-quality problem, not a config one. The payload now names the backend and warns once, but set it. |
| `AGENT_METHOD_LIBRARY_DIR` | path to `method_library` | `kb_method_search` reports an empty library and the sandbox mount is missing — the agent concludes no such method exists. |
| `OPENSEARCH_NODE` + credentials | cluster | Agent-KB arm returns nothing. |
| `FLASK_EMBEDDING_URL` | reachable embedder | A *wrong* URL is completely silent; the resolved value is logged once at first use. |
| `AGENT_SEARCH_TOP_K` | `20` | Retrieval window drops to 8; recall@20 was measured at **34/37**, recall@8 at 22/37. |
| `AGENT_INVARIANT_GATE` | `1` | No `checks.json`, so no numeric claim is ever verified. |
| `AGENT_ARTIFACT_EMIT` | `1` | Runs are not reproducible after the fact. |
| `AGENT_CODE_EXEC_IMAGE` | `iguide-codeexec@sha256:…` | See §2 — a tag is not a pin. |
| `LLM_PROVIDER` | `vllm` or `openai` | **Never `claude-cli`.** See §4. |
| `AGENT_ALLOW_WORKFLOW_EXEC` | unset / `0` | See §4. |
| `WEB_CONCURRENCY` | `1` | See §5. |

MinIO / platform credentials are **agent-side only**. They must not appear in the sandbox
container's environment or in the MCP service env. `--network none` in the sandbox is what makes
that a real boundary rather than a convention.

---

## 2. The sandbox image

Build and pin by **digest**, not by tag. `python:3.11-slim` resolves to different bytes next
month, so an artifact recording a tag records nothing about the environment that produced the
number in it.

```bash
docker build -t iguide-codeexec:latest sandbox/
docker image inspect iguide-codeexec:latest --format '{{index .RepoDigests 0}}'
```

Use the digest as `AGENT_CODE_EXEC_IMAGE`. `agent_runtime/artifacts.py::resolve_image_digest`
records it in every manifest; a reference that already contains `@sha256:` is passed through
unchanged, and one that cannot be resolved is recorded as `null` rather than omitted — "we could
not pin this" is information a replay needs.

The image bakes the geospatial stack (`geopandas`, `pyogrio`, `rasterio`, `pyproj`, `rtree`,
`contextily`, `mapclassify`) from wheels. Without it every session that touches a GeoDataFrame
pays a `pip install geopandas` before any work happens, and the library's strongest clusters are
all geospatial. The pyproj CRS database is warmed at build time because the invariant gate reads
`crs.axis_info` on every geometry frame, inside a container with no network.

---

## 3. Order of operations for a fresh environment

Each step has a check that tells you whether it worked, because most of these fail quietly.

1. **Create the agent indices** — `python scripts/create_agent_indices.py`.
   *Check:* 5 indices exist and the kNN dimension matches the embedder's real output. A
   mismatch fails at query time, not at creation.
2. **Build the method library** — `python scripts/build_method_library.py`.
   *Check:* `modules written` equals the non-alias unit count in `_registry.json`. They matched
   at 227/227 on this branch; a divergence means the registry kept entries for units that no
   longer exist, and a stale contract is worse than a missing one.
3. **Backfill the agent KB** — the extraction with `--targets opensearch,library`.
   *Check:* `agent_kb_search` returns a non-zero `count` and `backend: "opensearch"`.
4. **Smoke-test the sandbox mount**:
   ```bash
   docker run --rm -v "$PWD/agent_chat_files/method_library":/opt/iguide_methods:ro \
     -e PYTHONPATH=/opt/iguide_methods iguide-codeexec:latest \
     python -c "from iguide_methods import describe; print(describe('calculate_buffers'))"
   ```
   *Check:* it prints a contract, **and** the container environment contains no MinIO or
   platform credentials.
5. **Run the suite inside the built image**, not on the host. Host and image resolve different
   dependency versions; the drift was 3 packages when last measured.

---

## 4. Two settings that are security boundaries, not preferences

**`AGENT_ALLOW_WORKFLOW_EXEC` stays off.** `MCP_server/tools/generated_notebook_tools.py` is a
bare `exec()` of ingested third-party notebook source, in a process that has network access,
mounts the shared `agent_chat_files` volume, and carries cluster credentials. The method-library
path delivers the same reuse safely. Re-open this only if `_run_generated_manifest` is
re-pointed at `DockerCodeExecutor`.

**`LLM_PROVIDER=claude-cli` is development-only.** It shells out to a personal Claude Code
subscription. Anthropic's consumer terms restrict automated access to API-key access and
separately forbid making the account available to others, so it must not back a server handling
other people's requests. `test_deployment_contract.py` asserts it is absent from compose.

It is also *operationally* unsuitable regardless of terms: every supervisor step is a separate
subprocess, and a full code-executing turn took **several minutes** end-to-end when measured
through the prototype.

---

## 5. Single worker, deliberately

`WEB_CONCURRENCY=1`. The supervisor graph now compiles **with** a checkpointer, but under a
**per-run** namespace — deliberately not the conversation thread, because binding the
conversation would make a second turn resume the first, inheriting its `step`, `actions` and
`evidence` so a follow-up would start at step 8 and route straight to `done`.

Raising concurrency therefore needs more than flipping the number: it needs a *shared* (not
in-memory) checkpointer and a decision about turn identity. Until then use sticky routing and
publish the concurrent-session ceiling. Recoverability and cross-turn resumption look like the
same feature and are not.

---

## 6. Verifying a deployment

```bash
pytest rag_pipeline/tests/test_deployment_contract.py rag_pipeline/tests/test_api_auth.py -q
```

Then, against the running service:

| Check | Expected |
|---|---|
| `POST /agent/chat` with no key | 401 |
| `POST /agent/chat/stream` with a key | SSE stream ending in a `result` event |
| A geospatial question | answer, plus `checks.json` in the run's artifact directory |
| A deliberately geographic-CRS buffer | `verdict: fail`, and the answer carries the ⛔ caveat *with the remedy text* |
| A correct reproject-then-measure run | `verdict: pass` and **no** caveat |

That last row matters as much as the one above it. A gate that flags correct runs trains its
reader to ignore it, and then it is worse than absent — it cost real work and buys nothing.

---

## 7. What is knowingly not covered

- **Retrieval ranking for the method library** is token overlap with no IDF. It is adequate at
  227 units and is the first thing to index properly as the corpus grows.
- **Non-notebook element types** contribute few callable units; extraction coverage is the
  binding constraint on the library's breadth, not the extractor's correctness.
- **The `analyze` peer was not retired** (planned as M5). Across 7 live runs, analyze↔code
  ping-pong was **0**, so the exit criterion was already met and the change would have bought
  nothing for 44 sites of orchestration-core risk. Tagged `pre-m5-analyze-retire`.
