# The system's components, and a plan to split them

Two halves, deliberately kept in one file because the second only makes sense given the first.

**Part 1 is what exists**, verified against the tree rather than remembered.
**Part 2 is a proposal** for a middleware between the frontend and the agent. It has not been
built. Nothing below Part 1 describes current behaviour.

---

## Part 1 — What the system is made of

| Layer | What it is |
| --- | --- |
| **Frontend** | `map-ui-prototype/` — React, MapLibre + deck.gl. SSE consumer, IndexedDB session cache, auth client. Analyses arrive as map layers, not as text to interpret. |
| **HTTP surface** | `api/server.py` — **2,846 lines, 14 routes**. Identity, API-key gate, deployment modes, request normalisation, upload/download, conversation CRUD, trace retrieval, model catalogue, UI config, SSE framing. |
| **Agent runtime** | `agent_runtime/` — the supervisor graph (a single 238 KB `supervisor/graph.py`), search/analyze/code peers, `executor_factory` (LLM clients), `streaming_trace`, `skills`, `session_memory`. |
| **Tool families** | geo, terrain, rs-embed, spatial-stats / aggregate / temporal / overlay, admin-boundary, file, code execution, MCP and QGIS-MCP. |
| **Sandbox** | Docker-out-of-Docker. Per-conversation `/work` bind mount, 72-hour TTL, per-conversation dependency cache. |
| **Storage** | OpenSearch (`chat_memory`, `chat_traces`, the KB index), the file-store volume, code workspaces, browser IndexedDB. See [persistent-state.md](persistent-state.md). |
| **External services** | LLM providers (AnvilGPT / OpenAI), rs-embed, the platform backend (token introspection), the platform Neo4j, OSM/Overpass, 3DEP. |
| **Configuration** | `platform_endpoints` (`PLATFORM_TIER`, `SEARCH_TIER`), `deployment_mode` (`dev`/`demo`/`token`), `identity`. |

### The observation this plan rests on

**`api/server.py` is already a middleware — it is simply fused into the agent's process.** Adding
a layer is mostly *extracting* one that exists, which is why the plan below is more subtraction
than construction.

---

## Part 2 — Proposed split *(not built)*

### What moves, in order of value

**1. Identity and authorization.** The agent has no business knowing what a CILogon token is,
what `role <= 4` means, or how to introspect against a platform backend. Middleware terminates
auth and passes a resolved principal.

The strongest argument is not tidiness. The agent process today holds OpenSearch
`admin`/`all_access` **because it does its own persistence** — on a host that runs
LLM-generated code and mounts the Docker socket. Move persistence out and the agent needs no
datastore credential at all.

**2. Conversation and file persistence.** The agent needs chat *history* for context; it does not
need to own the record. `session_snapshot` — layers, `fileIds`, the drawn region — is a UI
artifact that lives in the agent's memory module for historical reasons only. Ownership,
retention, quotas and download serving are all policy.

**3. Trace capture.** The middleware already relays the SSE stream, so it can tee and persist it.
That removes `chat_traces` from the agent's concerns too.

**4. Deadlines, budgets, rate limits.** None of this exists today, which is why a stalled model
once cost 2 h 23 m in a benchmark run: the socket stays healthy and keepalives keep flowing, so
nothing can tell a hang from progress. A middleware with a clock is the right home for a turn
deadline — better placed than the LLM client, because it bounds the whole turn rather than one
call.

**5. The request contract.** camelCase/snake_case coalescing, defaults, validation, and the model
*allow-list* — which models a caller may select, as distinct from which one the agent picks.

### What stays in the agent

Routing, tool selection, the supervisor loop, the grounding audit, prompt construction. If the
middleware begins deciding *which tool to call*, the agent has been split in half rather than
decoupled.

### Three kinds of state, and only two move

| | moves? | why |
| --- | --- | --- |
| **Conversation memory** — `chat_history` | **yes** | what was said; the agent consumes it, needn't own it |
| **Client view** — `session_snapshot` | **yes** | a UI artifact the agent never needed |
| **Execution state** — LangGraph checkpoints, `/work`, the dependency cache | **no** | what is on disk *now*; different lifetime, different owner |

That third row is the trap. `agentws_<slug>_<sha256(thread_id::codeexec)>` holds what the last
turn's code wrote. Move it out and *"now add a heatmap of that file you just made"* stops
working, and every turn pays a cold dependency install.

### The interaction, once memory is out

The supervisor is **already history-agnostic**: `run_supervisor` and `stream_agent_query_events`
take `chat_history` as a plain list. The fetching happens one layer up, in
`agent_chat_service` (`get_or_create_memory` → `_build_chat_history` → pass it in). Taking memory
out is therefore mostly *deleting* that fetch and accepting the list from the caller — the
internal contract already exists.

**Inbound**, a self-contained turn:

```
principal        resolved; the agent never sees a JWT
thread_id        opaque but STABLE — it hashes to the sandbox directory
chat_history[]   the recent window; the agent still trims to its own token budget
file_handles[]   already authorized; the full accumulated set, not a delta
capabilities     which tools and models this caller may use
deadline         the turn budget
```

**Outbound**, nothing new is required. The agent already emits a terminal `result` event carrying
the answer, plus `map_layer` and artifact events, on the stream the middleware is relaying
anyway. **Persistence reads the stream it is already carrying** — no callback, no write API, no
circular dependency.

### Push, not pull

History should be pushed in rather than fetched back by the agent. Pull would give the agent a
client and a credential *for the middleware*, recreating the coupling being removed — and that
coupling is exactly why the agent holds cluster-admin credentials today. Push keeps a turn a pure
function of its request: testable, replayable from a stored trace, and holding no datastore
credential.

The cost is payload size on long conversations. `recent_k` already truncates today, so the
middleware applies the same policy; an opt-in retrieval token can be added later if that is ever
not enough.

### Traps specific to this system

- **Do not buffer the stream.** Map-native delivery depends on layers arriving *during* the turn.
  A hop that buffers SSE turns a live map into a slideshow.
- **`thread_id` must be stable and unique per conversation.** It is not a label: it keys the
  sandbox workspace. Minting a fresh one per request silently gives every turn an empty `/work`.
- **`get_session_files(thread_id)` accumulates uploads across turns.** If the middleware owns
  files it must send the resolved *full* set each turn; leaving the accumulation in the agent as
  well creates two sources of truth that drift.
