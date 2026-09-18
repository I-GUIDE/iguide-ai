# Where state lives

A reference for the four places this agent keeps data, what each one holds, who owns it and what
reclaims it. Companion to [`agent-architecture-changes.md`](agent-architecture-changes.md), which
records *why things changed*; this one records *what is true now*.

Figures are from the deployment on 2026-09-18 and are there to give a sense of scale, not to be
kept current. The layout is the durable part.

---

## At a glance

| Store | Where | Holds | Owner check | Reclaimed by |
| --- | --- | --- | --- | --- |
| Conversations | OpenSearch index `chat_memory` | agent memory + the client's view | `owner_id` on the document | nothing |
| Files | Docker volume at `/app/agent_chat_files` | uploads, outputs, QGIS jobs | `owner_id` in the record | `AGENT_FILE_RETENTION_DAYS` (**disabled**) |
| Code workspaces | `/tmp/iguide_codeexec`, bind-mounted | one working dir per conversation | keyed by conversation | 72-hour TTL |
| Browser cache | IndexedDB `iguide-map-ui` | the client's own copy | `ownerId` on the record | never (per browser) |

Two of these grow without bound by design; see [Nothing reclaims](#nothing-reclaims).

---

## 1. Conversations — OpenSearch

**Index** `chat_memory` (`OPENSEARCH_MEMORY_INDEX`), on `OPENSEARCH_NODE`. One document per
conversation, document id = the `memoryId` the client and the agent share.

```
chat_history        the AGENT's memory: what was asked and answered, context for the next turn
session_snapshot    the CLIENT's view: messages, layer descriptors, fileIds, region, model
conversationName    title, mirrored from the snapshot so the list can sort and label without
                    opening a transcript
owner_id            the platform user id (a CILogon URL)
threadId            the agent thread this conversation continues
messageCount        counts live on the DOCUMENT so a sidebar of fifty titles does not have to
layerCount          fetch fifty transcripts to count them
fileCount
createdAt  updatedAt
```

### The two halves are written by different things

`chat_history` comes from `create_memory` / `update_memory`, during a turn. `session_snapshot`
comes from `save_session_snapshot`, when the client PUTs its record at the end of a turn. **A
document can have either without the other**, and that asymmetry is load-bearing:

- `GET /agent/conversations` lists documents that have **both** an `owner_id` match and an
  `exists: session_snapshot`. Listed therefore means *restorable*.
- `GET /agent/conversations/<id>` serves the snapshot, and 404s without one.

The snapshot is **stored, not rebuilt**. `sessionStore.ts` was written server-shaped on purpose,
and reconstructing the client's view here would duplicate its layer-descriptor rules in a second
place where they would drift.

### Writes wait to be searchable

`save_session_snapshot` writes with `refresh="wait_for"`. OpenSearch is near-real-time — an
indexed document is not searchable until the next refresh, about a second — and the client's very
next action is to re-list. Without this the save lands and the immediate list does not see it.

### Size limit

A snapshot over `AGENT_SESSION_SNAPSHOT_MAX_BYTES` (default 5,000,000) is refused with
`SnapshotTooLarge` rather than silently truncated. The limit is read at call time, not frozen at
import, so it can be raised without a restart.

### What the numbers looked like

**1,264 documents, 5.27 MB — of which 6 have an `owner_id` and 3 have a client snapshot.**

The other ~1,258 predate identity: agent memories written before ownership existed, with no
client snapshot. They are invisible to the history list twice over, and correctly so — there is
no transcript in them for the UI to restore. If they are ever wanted, they hold `chat_history`
and would need both an owner backfilled and a snapshot synthesised.

### What a stored snapshot actually contains

Worth knowing precisely, because it decides what a past session can be used for:

```
title  id  memoryId  threadId  ownerId
messages[]     role, text, html — and per agent message:
                 trace[]      {kind: node|llm|tool|result, text}  the RENDERED trace
                 artifacts[]  {filename, file_id, download_url}
layers[]       full descriptors: kind, url/sourceUrl, bounds, style, fitBounds
fileIds[]      everything attached across the conversation
region         the drawn AOI, if any
model provider the model that answered
```

The trace survives, at **display fidelity**: tool names with their arguments *truncated at the
render cap*, and results as headlines — `1 feature · 0.4s`, `1 layer on the map · 2.67s` — rather
than payloads. LLM steps record only `Asking <model>`. So a past session shows what happened and
what it produced; it is not a replay log. See
[Reproducing a session](#reproducing-a-session).

---

## 2. Files — a Docker volume

**`/app/agent_chat_files`** (`AGENT_FILE_STORAGE_ROOT`), backed by
`/media/volume/i-guide-agent-volume/docker/volumes/…_agent_chat_files/_data`, so it survives an
image rebuild.

```
outputs/     what tools produced: rasters, GeoJSON, PNGs — by far the largest
uploads/     what people attached
qgis_jobs/   QGIS job working directories
metadata/    one record per file, including owner_id
element_cache/  platform element lookups
agent_kb/
generated_notebook_workflows/
```

Every record carries `owner_id`, stamped at creation. `may_read(record, allow_unowned=True)`
enforces it on `GET /agent/files/<id>/download`; a mismatch answers **404, not 403**, because a
403 would confirm the id exists and make the endpoint an enumeration oracle.

`AGENT_FILE_RETENTION_DAYS` drives the sweep. **It is `0` on the deployment, which disables it
entirely** — nothing here is ever deleted. ~7.3 GB, of which `outputs/` is 6.6 GB.

---

## 3. Code workspaces — a bind mount

**`/tmp/iguide_codeexec`** (`AGENT_CODE_EXEC_WORK_ROOT`), bind-mounted into the container at the
same absolute path. That sameness is required, not cosmetic: the agent shells out to the *host's*
Docker daemon, so `docker run -v <work>:/work` has to name a path the host can actually resolve.

One directory per conversation, `agentws_<slug>_<sha256-prefix>`. The slug alone would be
many-to-one — it maps every unsafe character to `_` and truncates — so `sess:42` and `sess_42`,
both raw client input, would share a directory and its files. The digest is what keeps them apart.

The container is ephemeral (`--rm`, that is the sandbox); the **workspace is not**, so a follow-up
turn can build on what the last one wrote instead of re-uploading it. `.deps/` inside holds the
conversation's dependency cache, bind-mounted read-only into runs.

Swept on `AGENT_CODE_EXEC_WS_TTL_HOURS` (default **72**), throttled to one walk per
`SWEEP_INTERVAL_S` = 600 s because every `execute_code` resolves a workspace and globbing the root
each time is pure waste. ~24 workspaces, 5.3 GB.

---

## 4. Browser — IndexedDB

**Database `iguide-map-ui`, object store `sessions`**, keyed by the client's own session id (not
the `memoryId`), each record stamped with `ownerId`.

In dev and demo this is the only copy. **In token mode it is a cache**: the server owns history so
conversations follow the account to another browser. IndexedDB is per-*origin*, not per-user, and
signing out of the platform does not touch it — hence `visibleTo(record, viewer)`: an owned record
is its owner's alone, an unowned one is shared. Another person's records are hidden, never
deleted; it is their data and this is their browser too.

---

## Nothing reclaims

| | size | reclaimed |
| --- | ---: | --- |
| `agent_chat_files` | 7.3 GB | no — `AGENT_FILE_RETENTION_DAYS=0` |
| `iguide_codeexec` | 5.3 GB | yes — 72-hour TTL |
| `chat_memory` | 5.27 MB | no |

About **12.6 GB on one volume**, with only the workspaces expiring. The file store's retention is
off by configuration rather than by omission, so turning it on is one variable — but note that
every answer publishes file ids as download links, and `outputs/` is where those point.

---

## Reproducing a session

A stored conversation is close to self-contained, because the pieces cross-reference each other:
`session_snapshot.fileIds` and each `artifacts[].file_id` point into the file store, which never
deletes; `layers[].sourceUrl` are those same files; `threadId` identifies the agent thread and,
through `sha256(f"{threadId}::codeexec")`, the code workspace directory.

**Available for any past session:** the question, the final answer, every file it produced, the
layer descriptors, the tool sequence with truncated arguments, and the model that answered.

**Not available, and each for its own reason:**

| Missing | Why |
| --- | --- |
| Full tool arguments | the trace stores the *rendered* line, truncated at the display cap |
| Tool result payloads | results are stored as headlines; the artifacts often stand in for them |
| Prompts and LLM responses | never persisted; `chat_history` holds turn text, not system prompts |
| Token usage / cost | emitted through `turn_instrumentation` to logs, not into the snapshot |
| Reasoning effort, code peer, orchestration | `model` and `provider` are stored; the rest of `AgentCfg` is not |
| The code the sandbox ran | lives in the workspace, gone after 72 hours |

That makes past sessions a good basis for **building benchmark cases** — a real question, the
files it produced and an expected answer is exactly the shape an EarthVerse-style task takes —
and a weaker basis for **reproducing a specific failure**, where the truncated arguments and the
expired workspace are what you most need.

Closing that gap means persisting the raw trace events rather than the rendered ones (the
`tool_call` / `tool_result` events already carry full arguments and outcomes server-side; they are
simply not stored), and either lengthening the workspace TTL or copying the workspace into the
file store at the end of a turn. Both push against the 5 MB snapshot cap, so raw traces probably
belong beside the snapshot rather than inside it.

Note the corpus is young: **3 of 1,264 documents carry a snapshot**, because client-side
persistence only began working on 2026-09-17.

---

## Not used

**No Postgres and no Neo4j for the agent's own state.** `POSTGRES_HOST`, `POSTGRES_DB` and
`DATABASE_URL` are unset. `NEO4J_CONNECTION_STRING` points at the *platform's* graph and is read
for related-elements; the agent writes nothing to it. OpenSearch's other indices belong to the
website — `chat_memory` is the agent's own.

---

## Keeping this current

Change what a store holds, where it lives, or what reclaims it, and change this file in the same
commit — the standing rule in `AGENTS.md`. The figures are illustrative and do not need chasing;
the structure does.
