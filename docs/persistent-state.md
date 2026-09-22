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
| Turn traces | OpenSearch index `chat_traces` | raw events, one document per turn | `owner_id`, via the conversation | nothing |
| Files | Docker volume at `/app/agent_chat_files` | uploads, outputs, QGIS jobs | `owner_id` in the record | `AGENT_FILE_RETENTION_DAYS` (**disabled**) |
| Code workspaces | `/tmp/iguide_codeexec`, bind-mounted | one working dir per conversation | keyed by conversation | 72-hour TTL |
| Browser cache | IndexedDB `iguide-map-ui` | the client's own copy | `ownerId` on the record | never (per browser) |

Two of these grow without bound by design; see [Nothing reclaims](#nothing-reclaims).

---

## 1. Conversations — OpenSearch

**Index** `chat_memory` (`OPENSEARCH_MEMORY_INDEX`), on `149.165.155.135` since 2026-09-22. The
host comes from `OPENSEARCH_NODE` when set, otherwise from the tier — `PLATFORM_TIER=dev` names dev's cluster, and prod names none, so a
production deployment cannot silently inherit dev's. One document per conversation, document id =
the `memoryId` the client and the agent share.

> **Host and credential both come from the tier**, so flipping `PLATFORM_TIER` moves them
> together. Set `OPENSEARCH_USERNAME_DEV` / `OPENSEARCH_PASSWORD_DEV` (and the `_PROD` pair);
> the bare `OPENSEARCH_USERNAME` / `OPENSEARCH_PASSWORD` remain as an un-tiered fallback and are
> flagged at boot when a tier is set, because on a host that switches tiers they are the trap:
> the cluster moves and the password does not, and the 401 reads as a network problem.
>
> Note the precedence is the **reverse** of the URL rule. For a URL the tier supplies a value and
> `PLATFORM_*_URL` overrides it; for a credential the tier supplies no value at all — secrets are
> never in this repository — so the tiered name is simply the more specific one. A tiered pair is
> selected whole or not at all, so a username for the tier and a password from the bare variable
> can never combine into two different accounts.
>
> **Which cluster, and why it is a tier fact.** Dev's OpenSearch moved from `149.165.155.195` to
> `149.165.155.135` on 2026-09-18, and `OPENSEARCH_NODE` stayed pinned to the old host. That host
> kept answering and kept accepting writes, so nothing failed — the agent simply went on reading
> and writing a cluster nobody maintained. The tier now names the cluster, and an explicit
> `OPENSEARCH_NODE` that disagrees with it is logged at boot rather than obeyed in silence.
>
> **Then it stopped accepting writes.** On 2026-09-22 the old cluster crossed the 95% flood-stage
> watermark and OpenSearch set `read_only_allow_delete` on `chat_memory`. Reads kept working, so
> nothing looked broken: turns streamed complete answers, the history list rendered, and every new
> conversation was silently lost. The only trace was one `429 cluster_block_exception` per turn,
> caught on purpose so a storage failure never costs someone their answer. **Four days of
> conversations were lost this way before anyone asked.**
>
> The 1,274 existing documents were copied to `149.165.155.135` (ids preserved — `memoryId` *is*
> the `_id`), verified for the `owner_id.keyword` subfield the list query needs, and
> `OPENSEARCH_NODE` was removed so `PLATFORM_TIER=dev` supplies the host. Copy first, switch
> second: dev's `chat_memory` was empty, so flipping first would have emptied the history UI.
>
> The lesson worth keeping is not about disk. A write path that fails silently while the read path
> succeeds is indistinguishable from a working system from the outside — no error reached a user,
> a log anyone watched, or the health check. If storage failure is survivable by design, it has to
> be *visible* by design too.

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

The trace here survives at **display fidelity**: tool names with their arguments *truncated at
the render cap*, and results as headlines — `1 feature · 0.4s`, `1 layer on the map · 2.67s` —
rather than payloads. LLM steps record only `Asking <model>`. It is what a person reads, not what
a turn is reproduced from; the raw events live in their own index, below.

---

## 1b. Raw turn traces — OpenSearch index `chat_traces`

**Index** `chat_traces` (`OPENSEARCH_TRACE_INDEX`). One document per **turn**, not per
conversation — a conversation has many, and they are large.

```
memory_id  thread_id  owner_id      the conversation this turn belongs to
query  answer  model  provider      what was asked, what came back, by what
events[]                            the events AS EMITTED: full tool arguments, full outcomes
event_count  dropped_count          how many were kept, and how many the cap removed
createdAt
```

### Recorded unfiltered, on purpose

`agent_dev` decides what the *client* sees: without it, detail-tier events — tool I/O, LLM steps,
routing detail — are never streamed. The recorder sits **before** that filter, so the record is
the same whether or not anyone was watching closely. The alternative inverts the value: the turns
most worth studying are the ones nobody had dev mode on for.

The two sinks are independent in both directions. A client that disconnects cannot cost the
record, and a recorder that throws cannot break the stream.

### It cannot fail a turn

`save_turn_trace` swallows its own errors and reports them in its return value. Diagnostics are
the least important thing happening during a turn, and losing an answer because the trace could
not be written would invert that.

### Two ceilings

`AGENT_TRACE_MAX_EVENTS` (default 4,000) bounds the list *in memory* during a turn — a runaway
loop is a memory problem long before it is a storage one. `AGENT_TRACE_MAX_BYTES` (default 2 MB)
bounds the stored document, and trims **from the middle**: the head holds the question and the
tail holds the outcome, while a turn that blew the limit did so in between, usually a retry loop
repeating itself. `dropped_count` records how much went, so a trimmed trace never reads as
complete.

### Reading them back

`GET /agent/conversations/<memory_id>/traces` lists a summary per turn **without** the events —
choosing which turn to open should not mean downloading all of them. `?traceId=<id>` returns one
turn with its full event list. Ownership is asserted on the *conversation*, so a trace can never
be reachable by an id its conversation would refuse.

---

## 1c. Which tier, and the two switches

`PLATFORM_TIER=dev|prod` selects the things that must agree about **identity and this agent's own
state**: the backend that mints tokens, the frontend a visitor signs in at, the check-tokens
endpoint, the OpenSearch cluster holding `chat_memory`, and — via `OPENSEARCH_USERNAME_<TIER>` /
`OPENSEARCH_PASSWORD_<TIER>` — the credential for it.

`SEARCH_TIER=dev|prod` selects **which knowledge base to search**, and falls back to
`PLATFORM_TIER` when unset. They are separate because they answer different questions: running
the dev platform against the prod knowledge base is an ordinary thing to want, and before this it
meant editing index names by hand and remembering to put them back. A split is logged at boot, so
it is never something to deduce from surprising results.

Any setting can be tiered by adding `_DEV` / `_PROD` to its name — `tiered_env()` is the general
rule, and every search read goes through it. In practice that is the index names:

```
OPENSEARCH_INDEX          new-opensearch-index              (untiered fallback)
OPENSEARCH_INDEX_DEV      iguide-platform-embeddings-dev    (dev's knowledge base)
```

An **empty** tiered value counts as unset, so a half-written `FOO_PROD=` cannot blank out a
working `FOO`. An **unrecognised** tier raises rather than falling back, in both switches: a
wrong cluster fails loudly, but a wrong *index* fails silently — the query succeeds and simply
returns nothing, and the agent answers confidently from the wrong corpus.

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
| Prompts and LLM responses | never persisted; `chat_history` holds turn text, not system prompts |
| Token usage / cost | emitted through `turn_instrumentation` to logs, not into either store |
| Reasoning effort, code peer, orchestration | `model` and `provider` are stored; the rest of `AgentCfg` is not |
| The code the sandbox ran | lives in the workspace, gone after 72 hours |

Full tool arguments and outcomes *were* on this list. They are now in `chat_traces`, which is
what that index exists for.

That makes past sessions a good basis for **building benchmark cases** — a real question, the
files it produced and an expected answer is exactly the shape an EarthVerse-style task takes —
and, with the raw events, a workable basis for **reproducing a failure** for as long as the
workspace survives. Past 72 hours the arguments are still there but the code that ran is not, so
the remaining gap is the workspace: either a longer TTL or copying it into the file store at the
end of a turn.

Note the corpus is young: **3 of 1,264 documents carry a snapshot** and trace recording began on
2026-09-18, so the traces start from zero. That is an argument for deciding what to capture
before a few hundred sessions accumulate without it.

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
