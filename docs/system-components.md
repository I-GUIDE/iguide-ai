# The system's components, and a plan to split them

Two halves, deliberately kept in one file because the second only makes sense given the first.

**Part 1 is what exists**, measured against the tree rather than remembered — last against
`prototype` at `3f84220` (2026-10-03). That commit includes the thirty-five-branch integration
recorded in Stage 29 of [agent-architecture-changes.md](agent-architecture-changes.md#stage-29),
and Stages 30 and 31 after it. The tree is ahead of the deployment: on that date the VM still ran
`5ae6d92` (2026-09-22), so nothing landed since then is live until the next deploy.
**Part 2 is a proposal** for a middleware between the frontend and the agent. It has not been
built. Nothing below Part 1 describes current behaviour, though every fact Part 2 cites about
the code was re-checked on the same date.

---

## Part 1 — What the system is made of

| Layer | What it is |
| --- | --- |
| **Frontend** | `map-ui-prototype/` — React 18; MapLibre 6.11 with deck.gl 9.4 drawn *interleaved* through `@deck.gl/maplibre`; react-map-gl 8. SSE consumer, IndexedDB session cache, auth client. Analyses arrive as map layers, not as text to interpret. The Downloads panel is assembled **client-side** from whatever the stream carries (`collectDownloads`), so a tool's result shape is its contract with that panel. |
| **HTTP surface** | `api/server.py` — **2,949 lines, 14 routes**. Identity, API-key gate, deployment modes, request normalisation, upload/download (an upload is stamped with the conversation that made it), conversation CRUD, trace retrieval, model catalogue, UI config, SSE framing. |
| **Agent runtime** | `agent_runtime/` — the supervisor graph (a single 242 KB `supervisor/graph.py`), search/analyze/code peers, `capability_registry` (what each peer can do; the supervisor's description of its peers is generated from it, and a test holds it against the peer builders), `executor_factory` (LLM clients), `streaming_trace`, `skills`, `session_memory`, `fork_safe` (starts child processes without forking the agent). |
| **Tool families** | Fifteen toolsets both the analyze and code peers bind — geocode, admin-boundary, terrain, overlay, aggregate, temporal, spatial-stats, geo, QGIS, rs-embed, rs-embed zonal, granular, conversation files, code execution, skills — and three only analyze binds: geo-analysis (knowledge-base renders), file, and MCP. `capability_registry.py` is the list; read it rather than trusting a count here. |
| **Sandbox** | Docker-out-of-Docker. Per-conversation `/work` bind mount, 72-hour TTL (`AGENT_CODE_EXEC_WS_TTL_HOURS`), per-conversation dependency cache. Children start through `fork_safe`: on macOS a long-lived agent cannot `fork()` once PROJ has read its database, and every child it started died of SIGSEGV before `exec`. |
| **Storage** | OpenSearch (`chat_memory`, `chat_traces`, the KB index), the file-store volume, code workspaces, browser IndexedDB — and, for operations, the host journal and the watchdog's incident bundles. See [persistent-state.md](persistent-state.md). |
| **External services** | LLM providers (OpenAI, AnvilGPT, Anthropic), rs-embed (Earth Engine through a service account), the platform backend (token introspection), the platform Neo4j, Census TIGERweb (named boundaries), Nominatim (geocoding), OSM/Overpass, USGS 3DEP, and a web-search fallback. |
| **Configuration** | `deployment_mode` — `dev`, `demo`, `token`, `local`. `local` adds one guarantee to `dev`: the process writes **nothing** to a shared store. It is unrelated to the map UI's *mock* mode, which also calls itself "local". `platform_endpoints` — `PLATFORM_TIER` owns every per-tier value in `_TIERS` (hosts, cluster, redirect id); `SEARCH_TIER` picks the knowledge base separately. `identity` — the JWT, verified by the platform backend. |
| **Build and CI** | Every image installs through `constraints.txt`, the deployed image's own `pip freeze` (177 pins), so a rebuild reproduces what runs rather than that day's newest. `.github/workflows/verify.yml` runs the suite on Linux / Python 3.11 under those versions on every push and pull request. |
| **Operations** | Containers log through compose's `journald` driver to the host journal, so a recreate no longer deletes the logs. `deploy/agent-watchdog.sh`, a systemd timer, restarts a container that has been unhealthy for ten minutes — after capturing an incident bundle, because the restart destroys the state that explains the failure. Health checks assert the status code. Every port the stack publishes binds `127.0.0.1` (Stage 31): nginx on the host reaches agent-api, and agent-api reaches its services by name inside the compose network. |

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
datastore credential at all. Part of that is already demonstrated: `AGENT_MODE=local` runs a
whole turn with the memory store refusing to open a client, so an agent without a datastore
credential is a configuration the code already supports, not a hypothesis.

**2. Conversation and file persistence.** The agent needs chat *history* for context; it does not
need to own the record. `session_snapshot` — layers, `fileIds`, the drawn region — is a UI
artifact that lives in the agent's memory module for historical reasons only. Ownership,
retention, quotas and download serving are all policy.

**3. Trace capture.** The middleware already relays the SSE stream, so it can tee and persist it.
That removes `chat_traces` from the agent's concerns too.

**4. Deadlines, budgets, rate limits.** None of this exists today (re-checked 2026-10-03: the LLM
client is built with no request timeout, no turn deadline exists, and the only timeouts in
`executor_factory` bound the model-catalogue fetch), which is why a stalled model
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
