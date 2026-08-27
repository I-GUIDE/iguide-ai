# I-GUIDE Platform Flask Servers

A multi-agent RAG (Retrieval-Augmented Generation) system for geospatial research on the [I-GUIDE Platform](https://platform.i-guide.io). Searches across keyword, semantic, spatial, knowledge graph, and external data catalogs to answer research questions with cited, grounded answers.

## Architecture

The codebase has four main packages:

```
api/                   Flask REST API layer (thin routing over the services)
rag_pipeline/          Search, retrieval, generation (the RAG engine)
agent_runtime/         Multi-agent orchestration (decides how to answer)
MCP_server/            Geospatial analysis tools exposed via Model Context Protocol
```

Plus supporting services:

```
embedding-server/              Dense embedding generation (port 5000)
metadata-extraction-server/    Spatial metadata extraction (port 5001)
```

### How a query flows

```
User query
  -> agent_runtime: OrchestratorAgent decides which sub-agents to invoke
       -> SearchAgent: retrieves evidence using search tools
       -> AnalysisAgent: synthesizes answer from evidence
       -> CodeAgent: generates executable code when needed
       -> DirectAnswerAgent: answers from chat history alone
  -> rag_pipeline: search backends + generation
       -> search/keyword.py:          BM25 via OpenSearch
       -> search/semantic.py:         Vector kNN via embedding service
       -> search/neo4j.py:            Knowledge graph (3-tier: pattern tools -> Text2Cypher -> keyword)
       -> search/spatial.py:          Geographic filtering via NLP + Google Maps geocoding
       -> search/opengeodata.py:      Federated STAC/OGC/CKAN/NASA CMR catalog search
       -> generation.py:              LLM answer synthesis with citations
  -> Response with answer, citations, and evidence
```

## Project Structure

```
agent_runtime/                   Multi-agent orchestration (canonical home for all agent code)
  graph_runtime.py               Public API: run_agent_query(), stream_agent_query_events(), run_code_agent_query()
  graph_nodes.py                 Tool factories that expose sub-agents (search/analysis/code/direct) as tools
  graph_state.py                 Type definitions, tool-name constants
  executor_factory.py            Agent prompts, LLM config, executor builders (full_pipeline + granular strategies)
  intent_classifier.py           Query intent classification (analysis/code/discovery/hybrid) + tool routing
  tool_policy.py                 Tool filtering by intent
  skills.py                      Filesystem-backed SKILL.md skill discovery/loading
  agent_chat_service.py          Chat session management
  file_store.py                  File upload/download storage
  langchain_tool.py              RAG pipeline as a LangChain StructuredTool
  langchain_granular_tools.py    Individual search backends as LangChain tools
  langchain_mcp_tools.py         MCP server bridge into LangChain
  langchain_file_tools.py        File I/O tools for agents
  langchain_agent_executor.py    AgentExecutor wiring helpers
  runtime_utils.py               Response parsing, trace building
  streaming_trace.py             SSE event streaming for /agent/chat/stream
  trace_analyzer.py              Trace post-processing / analysis
  trace_store.py                 Trace persistence

api/                             Flask REST API layer
  server.py                      All HTTP routes (/query, /agent/chat, /agent/files/*) (port 5002)

rag_pipeline/                    RAG engine
  pipeline.py                    Pipeline orchestration (memory -> search -> rerank -> generate)
  routing.py                     Retrieval -> reranking -> generation glue
  state.py                       Shared AgentState schema
  generation.py                  LLM answer synthesis with citations
  reranker_llm.py                LLM-as-judge reranking
  reranker.py                    ColBERT reranking (optional)
  hallucination_check.py         Post-generation audit (optional)
  router_llm.py                  LLM-based search method selection
  memory_module.py               Chat history, follow-up detection, query augmentation
  llm_utils.py                   LLM client wrapper (OpenAI-compatible)
  qgis_headless_tools.py         Headless QGIS Processing / metric-buffer tools
  qgis_pyqgis_worker.py          Standalone PyQGIS subprocess worker (layer summary, map render)
  agent_chat_service.py          Compat shim -> agent_runtime.agent_chat_service
  agent_file_store.py            Compat shim -> agent_runtime.file_store
  langchain_tool.py              Compat shim -> agent_runtime.langchain_tool
  langchain_granular_tools.py    Compat shim -> agent_runtime.langchain_granular_tools
  langchain_mcp_tools.py         Compat shim -> agent_runtime.langchain_mcp_tools
  langchain_file_tools.py        Compat shim -> agent_runtime.langchain_file_tools
  langchain_agent_executor.py    Compat shim -> agent_runtime.langchain_agent_executor
  search/                        Search backends
    core.py                      Search orchestrator (runs all backends)
    keyword.py                   OpenSearch BM25 keyword search
    semantic.py                  Vector embedding kNN search
    neo4j.py                     Neo4j knowledge graph search
    neo4j_graph_tools.py         Prewritten Cypher templates + pattern detection
    agents.py                    3-tier Neo4j dispatcher + LLM-generated OpenSearch DSL
    spatial.py                   Geographic search (NLP + geocoding + geo_shape)
    opengeodata.py               Federated search (STAC, OGC, CKAN, NASA CMR)
    utils.py                     Logging, env parsing, normalization

MCP_server/                      Model Context Protocol tool server
  server.py                      FastMCP server with auto-discovery (port 8000)
  tools/
    data_tools.py                Chicago community areas + crime data loading
    spatial_analysis_tools.py    Spatial joins, choropleth maps, resource search
    search_tools.py              External resource search (DuckDuckGo)
    image_tools.py               Vision model image/map analysis
    notebook_workflow_tools.py   Jupyter notebook -> callable tool conversion
    generated_notebook_tools.py  Shared executor body for generated workflow manifests
    generic_executor_tools.py    run_notebook_workflow / run_code_element (one generic executor)
    ingest_tools.py              ingest_github_repo (agent-callable ingestion)
  notebook_workflow_builder.py   Notebook AST parsing and artifact generation

embedding-server/                Dense embedding service
  dense_embedding_server.py      Flask server for all-MiniLM-L6-v2 embeddings (port 5000)
  dense_embedding.py             Embedding model wrapper
  create_embedding_for_existing.py  Batch embedding generation
  reindex_wkt_spatial.py         WKT to GeoJSON conversion

metadata-extraction-server/      Metadata extraction
  minio_webhook.py               MinIO webhook listener (port 5001)
  extract_metadata.py            Spatial metadata extraction
  extract_metadata_code_notebooks.py  Code/notebook metadata extraction
```

> **Note:** All agent code now lives in `agent_runtime/`. The agent-related
> modules under `rag_pipeline/` (`agent_chat_service.py`, `agent_file_store.py`,
> `langchain_*.py`) are thin backward-compatibility shims that re-export from
> `agent_runtime/` so existing `rag_pipeline.*` import paths keep working. Import
> from `agent_runtime` directly in new code.

## Quick Start

### Prerequisites

- Python 3.11+
- OpenSearch instance
- LLM endpoint (OpenAI-compatible, e.g., AnvilGPT or vLLM)

### Setup

```sh
cp .env.example .env
# Edit .env with your OpenSearch, LLM, and Neo4j credentials

pip install -r requirements.txt
```

### Run the API server

```sh
python -m api.server
# Runs on port 5002
```

### Run the agent from CLI

```sh
python -m agent_runtime.graph_runtime \
  "Generate the code for a RAG grader" \
  --tool-strategy granular \
  --include-mcp-tools
```

```sh
python -m agent_runtime.graph_runtime \
  "What's the risk of aging dams" \
  --tool-strategy full_pipeline
```

```sh
python -m agent_runtime.graph_runtime \
  "Analyze: What's the community in Chicago with most theft" \
  --tool-strategy granular \
  --include-mcp-tools
```

### Run with Docker Compose

```sh
docker-compose up
```

Starts three services: embedding-server (5000), metadata-extraction-server (5001), rag-pipeline (5002).

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/query` | POST | RAG pipeline (retrieval + generation) |
| `/agent/chat` | POST | Agent-backed conversation |
| `/agent/chat/stream` | POST | Streaming agent responses (SSE) |
| `/agent/files/upload` | POST | Upload files for analysis |
| `/agent/files/<id>/download` | GET | Download files |
| `/query/batch` | POST | Batch query processing |

## Configuration

See [.env.example](.env.example) for all configuration options. Key variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENSEARCH_NODE` | Yes | OpenSearch endpoint |
| `OPENSEARCH_INDEX` | Yes | Document index name |
| `VLLM_API_KEY` or `OPENAI_KEY` | Yes | LLM API key |
| `VLLM_PROXY` or `OPENAI_BASE_URL` | Yes | LLM endpoint URL |
| `FLASK_EMBEDDING_URL` | Yes | Embedding service URL |
| `NEO4J_URI` | No | Neo4j connection (enables graph search) |
| `GOOGLE_MAPS_API_KEY` | No | Enables NLP-based spatial search |
| `MCP_SERVER_URL` | No | MCP tool server URL (default: http://127.0.0.1:8000/mcp) |
| `QGIS_PROCESS_BIN` | No | `qgis_process` executable for headless QGIS Processing tools |
| `QGIS_PYTHON_BIN` | No | Python executable that can import `qgis.*` for standalone PyQGIS tools |
| `QGIS_PREFIX_PATH` | No | Optional QGIS install prefix for standalone PyQGIS initialization |
| `QGIS_JOB_ROOT` | No | Optional root for per-session QGIS job artifacts |
| `AGENT_SKILLS_ENABLED` | No | Enables native filesystem skill discovery (default: `1`) |
| `AGENT_SKILL_PATHS` | No | Comma-separated extra skill roots; defaults also scan `skills/` and `.agents/skills/` |
| `AGENT_SKILL_MAX_RESOURCE_BYTES` | No | Max bytes loaded from a skill resource (default: `65536`) |

### Headless QGIS Tools

Granular LangChain agent mode includes optional headless QGIS tools:

- `qgis_processing_help` inspects QGIS Processing algorithm parameters.
- `qgis_processing_run` runs one Processing algorithm through `qgis_process`.
- `qgis_metric_buffer` reprojects, buffers by meters, and reprojects output for safer metric buffers.
- `pyqgis_layer_summary` inspects one layer in a standalone PyQGIS subprocess.
- `qgis_map_image` renders layers to a PNG in a standalone PyQGIS subprocess.

Each call writes artifacts under `AGENT_FILE_STORAGE_ROOT/qgis_jobs/<session>/<job_id>` by default. The session id is derived from the agent thread id when the tool is called through the orchestrated LangChain runtime, so different conversations do not share QGIS project state. These tools require QGIS to be installed on the host or in the container.

## Agent Skills

Native skills are filesystem bundles that add reusable workflow guidance without
loading the full instructions into every prompt. Put skills under `skills/` or
`.agents/skills/`, or pass explicit roots via `AGENT_SKILL_PATHS`, CLI
`--skill-paths`, or API `skillPaths`.

Each skill lives in its own directory:

```text
skills/
  geospatial-report/
    SKILL.md
    references/
      style-guide.md
    scripts/
      validate.py
```

`SKILL.md` requires YAML-style front matter:

```md
---
name: geospatial-report
description: Use when producing a cited geospatial analysis report.
allowed-tools: keyword_search, semantic_search, spatial_search
tags: [geospatial, reporting]
---

# Geospatial Report

Follow this workflow when the user asks for a report...
```

At runtime, agents see only skill names and descriptions through the
`list_available_skills` and `load_skill` tools. When a skill is relevant, the
agent calls `load_skill(skill_name=...)` to load the full instructions, and can
call it again with `resource_path` for listed reference files.

Treat skills as trusted developer-controlled assets. The native loader reads
instructions and text resources only; executable actions should still be exposed
as MCP or LangChain tools with schemas, permissions, and audit logging.

## Testing

```sh
# Run unit tests (no live services needed)
python -m pytest MCP_server/ rag_pipeline/tests/test_opengeodata_search.py rag_pipeline/tests/test_mcp_cache.py -v

# Run integration tests (requires OpenSearch, LLM, etc.)
python -m pytest rag_pipeline/tests/ -v

# Dev verification scripts — runnable E2E checks (see scripts/ dir)
python scripts/check_neo4j_search.py            # Neo4j 3-tier search
python scripts/run_local_rag_test.py            # Full pipeline with connectivity checks
python scripts/run_full_pipeline.py             # Full pipeline with step-by-step output
python scripts/check_hallucination.py           # Hallucination audit on a sample query
python scripts/demo_reranker.py                 # ColBERT reranker demo
python scripts/check_mcp_install.py             # MCP dependency sanity check
python scripts/check_memory_thread.py           # Multi-turn memory thread verification
```

## LangSmith Tracing

```sh
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=your_langsmith_api_key
export LANGSMITH_PROJECT=i-guide-agent
```

Agent execution from `agent_runtime.graph_runtime` will appear in LangSmith traces.

## Inspect Agent Flow

```sh
python tools/inspect_agent_query_graph.py
python tools/inspect_agent_query_graph.py --format mermaid
python tools/inspect_agent_query_graph.py --format json
```
