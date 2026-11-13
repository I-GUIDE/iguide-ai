# I-GUIDE AI agent

The agent behind [agent.i-guide.io](https://agent.i-guide.io). A user asks a geospatial question,
in chat or on a map. The agent searches the I-GUIDE catalog and public data, runs analysis code in
a sandbox, puts results on the map, and writes an answer that cites where every number came from.

Start with [AGENTS.md](AGENTS.md) for anything you plan to change. It lists the settings that fail
silently and the conventions that look wrong until you know why. This README is the map.

## How a question is answered

```
browser (map-ui-prototype/)
  -> api/server.py                       POST /agent/chat or /agent/chat/stream
  -> agent_runtime/agent_chat_service.py picks the model, loads chat history
  -> agent_runtime/graph_runtime.py      runs the graphs, streams trace events
  -> agent_runtime/orchestrator_graph.py triage:
       fast_answer    small talk, one model call
       capabilities   "what can you do?"
       orchestrate -> agent_runtime/supervisor/graph.py
           clarify     ask the user when the request is ambiguous
           supervisor  picks the next step, one at a time
           search      catalog, web and open-data search  (rag_pipeline/search/)
           analyze     GIS tools: boundaries, overlays, statistics, map layers
           code        writes and runs Python in the sandbox (agent_runtime/code_execution.py)
           synthesize  writes the answer, checks its figures against what was computed
```

The peers (search, analyze, code) run one after another, never at the same time. AGENTS.md
explains why the design is still a supervisor over peers, and what that costs.

Code runs are checked before and during execution (`agent_runtime/code_watch.py`). Distance
and area results are checked for CRS and unit mistakes (`agent_runtime/sandbox_verify.py`).
Each turn's tool calls and results are recorded in `agent_runtime/turn_log.py`.

## Where things are

| Folder | What it holds |
|---|---|
| `api/` | Flask API: chat, streaming, file upload/download, examples, admin |
| `agent_runtime/` | The agent: graphs, peers, prompts, tools, sandbox, checks |
| `rag_pipeline/` | Search backends (`search/`), the older RAG `/query` pipeline, and most tests (`tests/`) |
| `map-ui-prototype/` | The web app at agent.i-guide.io (React, MapLibre, deck.gl) |
| `MCP_server/` | Geospatial tools served over the Model Context Protocol |
| `gis_harness/` | Whole-task regression runs with exact expected answers ([README](gis_harness/README.md)) |
| `extractors/`, `metadata-extraction-server/` | Building the knowledge base from datasets, notebooks and code |
| `embedding-server/` | Text embeddings for semantic search |
| `sandbox/` | The Docker image code runs in |
| `docker-compose*.yml`, `deploy/` | Containers, and the watchdog service that restarts the agent ([docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)) |
| `docs/` | Design records, below |
| `scripts/`, `tools/` | One-off checks and maintenance scripts |
| `skills/` | Skill bundles the agent can load on request |
| `opengeodata_prototype/` | Old prototype; `rag_pipeline/search/opengeodata.py` still reads its `.env` |

## Design records

- [agent_runtime/agent_architecture.md](agent_runtime/agent_architecture.md): the architecture,
  anchored to `file:line`.
- [docs/agent-architecture-changes.md](docs/agent-architecture-changes.md): every architectural
  change, why it was made and what was measured, numbered by stage.
- [docs/DEVLOG.md](docs/DEVLOG.md): the working log.
- [docs/system-components.md](docs/system-components.md): the services and how they connect.

## Run it

```bash
cp .env.example .env          # OpenSearch, model provider and service URLs
pip install -r requirements.txt -c constraints.txt

# Agent API on a laptop. AGENT_MODE=local keeps your test turns out of production storage.
PYTHONPATH="$PWD" PORT=5055 AGENT_MODE=local AGENT_CHAT_API_KEY= python3 api/server.py

# Map UI against it (Node 18+)
cd map-ui-prototype && AGENT_TARGET=http://localhost:5055 npm run dev
```

`docker-compose up` starts the embedding server, the MCP server and the agent API, on
127.0.0.1 only. See AGENTS.md "Run it" before running locally: without `AGENT_MODE=local`, the
`.env` points at shared infrastructure.

Which model answers is chosen per request (`/agent/models` lists them). The providers are OpenAI,
NCSA Lumen and Purdue AnvilGPT; AGENTS.md "Which model answers" has the details.

## Test it

```bash
python3 -m pytest rag_pipeline/tests/ -q     # offline by default
python3 -m pytest tests/ -q

# Whole tasks against a live model, scored mechanically
python -m gis_harness.run --start-server 5301 --model lumen:deepseek-v4-flash --tasks sample
```

## API

| Endpoint | Method | What it does |
|---|---|---|
| `/agent/chat` | POST | One agent turn |
| `/agent/chat/stream` | POST | One agent turn, streamed as server-sent events |
| `/agent/chat/cancel` | POST | Stop a running turn |
| `/agent/files/upload`, `/agent/files/<id>/download` | POST, GET | Files for a conversation |
| `/agent/conversations[/<id>]` | GET, PUT, DELETE | Saved conversations and their traces |
| `/agent/examples[...]` | various | The Spatial Analysis examples |
| `/agent/models`, `/agent/ui-config`, `/agent/whoami` | GET | What the UI needs to start |
| `/query`, `/query/batch` | POST | The older RAG pipeline, without the agent |
| `/health` | GET | Health check |

The full spec is served at `/apidocs` when the API is running.
