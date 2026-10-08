# Where state lives

A reference for the places this agent keeps data, what each one holds, who owns it and what
reclaims it. Companion to [`agent-architecture-changes.md`](agent-architecture-changes.md), which
records *why things changed*; this one records *what is true now*.

Figures are from the deployment on 2026-09-18 and are there to give a sense of scale, not to be
kept current. The layout is the durable part.

**Re-checked on 2026-10-03.** Locations come from the running deployment. It is on
`PLATFORM_TIER=prod`, so conversations and traces live on the prod cluster, `149.165.155.195`.
Behaviour comes from `prototype`, which is ahead of the deployment: the VM still runs `5ae6d92`
(2026-09-22). Wherever a change below landed after that, the text gives its date, and it is
not live until the next deploy.

---

## At a glance

| Store | Where | Holds | Owner check | Reclaimed by |
| --- | --- | --- | --- | --- |
| Conversations | OpenSearch index `chat_memory` | agent memory + the client's view | `owner_id` on the document | nothing |
| Turn traces | OpenSearch index `chat_traces` | raw events, one document per turn | `owner_id`, via the conversation | nothing |
| Files | Docker volume at `/app/agent_chat_files` | uploads, outputs, QGIS jobs | `owner_id` in the record | `AGENT_FILE_RETENTION_DAYS` (**disabled**) |
| Code workspaces | `/tmp/iguide_codeexec`, bind-mounted | one working dir per conversation | keyed by conversation | 72-hour TTL |
| Browser cache | IndexedDB `iguide-map-ui` | the client's own copy | `ownerId` on the record | never (per browser) |
| Operational records | host journal; `/var/log/iguide-agent/incidents/` | container logs; one evidence bundle per watchdog incident | host access only | journal capped at 3 GB; newest 60 bundles |

Two of these grow without bound by design; see [Nothing reclaims](#nothing-reclaims). Under
`AGENT_MODE=local` the first two are never written at all — see [Local mode](#local-mode).

---

## 1. Conversations — OpenSearch

**Index** `chat_memory` (`OPENSEARCH_MEMORY_INDEX`), on the prod cluster `149.165.155.195` since
the deployment moved to `PLATFORM_TIER=prod` on 2026-09-22. The host comes from `OPENSEARCH_NODE`
when set, otherwise from the tier — `PLATFORM_TIER=dev` names dev's cluster, and prod names none,
so a production deployment cannot silently inherit dev's. That is why the deployment names
`149.165.155.195` in `OPENSEARCH_NODE` explicitly. One document per conversation, document id =
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
>
> **Then the deployment moved to prod** (2026-09-22). `PLATFORM_TIER=prod` with `OPENSEARCH_NODE`
> naming `149.165.155.195`, the cluster that had gone read-only. Its space was freed first, and
> the switch waited for proof rather than a status: red to yellow, no write-blocked index, a real
> write answering 201. The two clusters do **not** hold the same conversations. At the switch,
> `.135` had 1,281 and `.195` had 1,274, so the seven written to dev after the copy are visible
> only from a dev-tier deployment. The tier owns the conversation store, and moving tiers is not
> a migration.

```
chat_history        the AGENT's memory: what was asked and answered, context for the next turn
session_snapshot_json
                    the CLIENT's view, as ONE JSON string: messages, layer descriptors,
                    fileIds, region, model (since M8.77; see below)
session_snapshot    the same record as an OBJECT, where it was stored before M8.77. Read,
                    never written; null once the conversation is saved again
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

`chat_history` comes from `create_memory` / `update_memory`, during a turn. The client record
comes from `save_session_snapshot`, when the client PUTs its record at the end of a turn. **A
document can have either without the other**, and that asymmetry is load-bearing:

- `GET /agent/conversations` lists documents that have **both** an `owner_id` match and a client
  record (`exists` on `session_snapshot_json` or `session_snapshot`). Listed therefore means
  *restorable*.
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
import, so it can be raised without a restart. Since M8.77 it counts the UTF-8 bytes of the
string actually stored.

### The record is stored as text, never as fields the index maps

**The rule: a payload whose shape a tool or a model decides never goes into a dynamically mapped
field.** `chat_memory` has no explicit mapping, so OpenSearch types each field from the first
document that carries it, and a document that disagrees is refused *whole*.

Until M8.77 the client's record went in as an object, and the record carries each answer's
`agent_result`: tool arguments and tool results, whose shape is decided per turn by the tool and
the model. On **2026-10-08 21:36:39 UTC** `overpass_search` sent `bbox` as the string
`"-87.93,41.87,…"`, where an earlier conversation had sent a float array, and the save came back

```
PUT /agent/conversations/sess-07bc717f-… 500
mapper_parsing_exception: failed to parse field [session_snapshot.messages.response.agent_result.
orchestration_result.analysis_results.tool_calls.args.bbox] of type [float] … value: '-87.93…'
```

The conversation document existed with **0** snapshot messages, so the turn never reached
History, and the client was told nothing it showed the user. Any turn whose tool payload took a
different JSON type from one already seen would have been lost the same way. The live mapping,
read that evening (`GET chat_memory/_mapping`, 1,722 documents):

- **300** leaf fields, **266** of them under `session_snapshot`, **181** under `agent_result`
  alone;
- **559** fields counted toward the index's **1,000**-field limit (`.keyword` sub-fields
  included). Every new argument name or result key added more, and past the limit *every* save
  fails, not only the odd one;
- typed traps already in place beside `bbox`: `layers.style.fill` is `long` (a hex colour is
  refused), `opengeodata_results.datetime` is `date`, `region.coordinates` is `float`, and the
  properties of inline GeoJSON features are mapped by name.

Since M8.77 `save_session_snapshot` stores the record as **one JSON string in
`session_snapshot_json`**. A string has one type whatever it holds, so no record can conflict
with another and none adds a field. Nothing searches inside the record: the list filters and
sorts on the document's own fields (`owner_id`, `updatedAt`, the counts), which keep their fixed
types. The same write sets the old `session_snapshot` object to null, so a conversation holds one
copy of its record. Reading prefers the string and falls back to the object, so a conversation
not saved since still opens and lists, and `GET /agent/conversations/<id>` returns exactly the
shape it did before. The client (`agentClient.ts`, `restoreSession`) is unchanged.

Measured on a disposable OpenSearch 2.14.0 (prod's version) created with the exact live mapping:
the old object shape of the 21:36 record is refused with the same `…tool_calls.args.bbox` error;
the new writer stores the float-array record, the string record, and the string record over the
float-array one, and each reads back identical when compared as JSON. A legacy object document opens and
lists, re-saves to `session_snapshot: null` plus the string, and an untouched one still lists. A
4.8 MB record stores. Thirty saves with thirty different argument shapes added no field beyond
`session_snapshot_json` (`text` + `.keyword`). `test_snapshot_mapping.py` runs the same checks
against a fake that keeps each field's first type, from the live mapping and from an empty index.

**Not fixed by this, and why.** The 266 old fields stay in the mapping until the index is
rebuilt, because a field cannot be removed in place; the 1,000-field limit now has about 440 to spare
and no writer that grows it. `session_snapshot_json` is analysed as text, which indexes words
nobody searches. The better long-term shape is a new index with an explicit mapping
(`dynamic: strict` at the top level; the record under a field with `"index": false`, or an object
with `"enabled": false`), copied with `_reindex` and swapped in behind an alias. That changes the
prod cluster, so it is the user's call; the proposal is in the M8.77 PR.

### The other writers to this cluster

Audited 2026-10-08 for the same pattern:

| writer | field | shape decided by | state |
| --- | --- | --- | --- |
| `save_session_snapshot` | the client record | tools and models | fixed here (M8.77) |
| `save_turn_trace` | `chat_traces.events` | tools and models | fitted to the mapping since M8.74 |
| `update_memory` | `chat_history[].elements` | search results | latent: every live caller passes `[]`; the 14 mapped `elements.*` fields come from older writers |
| `update_rating` | `chat_history[].ratings` | the caller | latent: no caller left in the server; `ratings.*` are `long`, so a word would be refused |
| `create_memory`, `get_or_create_memory` | document fields | this code | fixed shape |
| `extractors/emitters/opensearch_emitter.py` | KB index `extracted.*` | the extraction analyzers | explicit mapping above `extracted`, dynamic below it by design; a separate writer, not changed here |

The file metadata store is not on this cluster: it is one JSON file per record on the Docker
volume (§2), and does not have a mapping to conflict with.

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

### It must fit the index's mapping, or the whole turn is refused

`chat_traces` has **no explicit mapping**. OpenSearch typed each field from the first document
that carried it, and a document that disagrees with a field's type is rejected *whole*: one bad
field costs every event of the turn. `save_turn_trace` can only log the 400, so the loss is
silent. Until M8.74 this lost **30 turns between 10-01 and 10-08**, every one on
`events.data.args`:

- LangChain hands `on_tool_start` a Python repr of the arguments, `"{'query': 'x'}"`. That is not
  JSON, so `args` stayed a string, and the field was mapped `text`.
- A tool called with **no** arguments arrives as `"{}"`. That *is* JSON, so it parsed to a dict,
  and a dict cannot go in a `text` field. `list_conversation_files` and `list_available_skills`
  take no arguments, so turns that listed files or skills were the ones not recorded.
- Emitters that send real dicts (`{query=…, limit=6}` in the error) were refused the same way.

The fix is on the writer (`rag_pipeline/trace_mapping.py`). The index is on the shared prod
cluster, and a field's type cannot be changed in place. Every document is fitted before it is
sent:

| value | stored as |
|---|---|
| a tool's arguments (`args`, `arguments`, `tool_args`), anywhere | JSON text, always |
| a structure in a mapped `text` field | JSON text, same field |
| anything a mapped numeric, boolean or object field cannot hold | JSON text in a sibling `<field>_text` |
| an unmapped field: strings | as they are |
| an unmapped field: anything else, or a date-like string | JSON text |

Arguments are text rather than an object because no single object mapping could hold them: each
tool has its own argument names and types (`limit: 6` here, `limit: "all"` there), and as an
object they would collide, and grow the mapping by one field for every argument name ever used.

New fields go in as text because dynamic typing is the bug itself, not just the dict case.
Measured on a disposable OpenSearch 2.14.0 (the prod version) loaded with the live mapping:

- a new field first seen as `6` maps `long`, then refuses `"all"`;
- a new field seen as text and as a number in the *same* document is refused at once
  (`cannot be changed from type [text] to [long]`);
- a new string field first seen as `"2026-10-08"` maps `date`, then refuses `"yesterday"`.

The cost is that a new numeric or date field is searchable only as text until someone maps it
deliberately. Doing that needs a new index, which is a change to the prod cluster and the user's
call. `TRACE_MAPPING` in that module is a snapshot of the live mapping, read 2026-10-08. It is a
Python module and not a `.json` file because `.dockerignore` excludes `*.json`, so a JSON snapshot
would not reach the image. When the live mapping gains a field, add it there.

Already-stored traces are untouched. Turns refused before M8.74 are gone; the journal holds only
the warning.

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

The deployment runs `PLATFORM_TIER=prod` and `SEARCH_TIER=prod` (checked 2026-10-03). Beyond
hosts and credential, the tier also owns this agent's redirect id (`redirect_domain_id`: dev `006`,
prod `003`). The access-cookie name is **not** in `_TIERS`. `JWT_ACCESS_TOKEN_NAME` is set per
deployment, and `consistency_warning()` requires it to end in the tier's own suffix, `-dev` or
`-prod`. Since 2026-10-03 the knowledge-base search client takes the tier's credential as well;
before that it took the untiered one.

Any setting can be tiered by adding `_DEV` / `_PROD` to its name — `tiered_env()` is the general
rule, and every search read goes through it. In practice that is the index names:

```
OPENSEARCH_INDEX          iguide-feb12                      (untiered fallback)
OPENSEARCH_INDEX_PROD     iguide-feb12                      (prod's knowledge base)
OPENSEARCH_INDEX_DEV      iguide-platform-embeddings-dev    (dev's knowledge base)
```

**The prod knowledge base is `iguide-feb12` since 2026-10-07**, the index the platform's own
search reads. Before that the deployment searched `new-opensearch-index`, an index created
2025-10-27 that has fallen behind. It had 619 elements against feb12's 763, so the 148
elements the platform added since could not be found. A user saw it as a dataset the platform's search
listed by its exact title while the agent said no such data existed. The two indices have the
same mapping, and the 615 elements in both carry identical vectors, so the agent's embedding
service still matches. Two gaps remain in feb12 itself:

- 108 of its 763 elements have no `contents-embedding`, mostly ones added after the copy.
  Semantic search cannot return them. Keyword and spatial search can, and so can Neo4j
  search, since the graph is current.
- Keyword search matches `contents` only. An element whose description does not repeat its
  title's words, such as "Chicago Crime data 2026", is found only by words from its
  description.

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

Every record carries `owner_id`, stamped at creation, and `may_read(record, allow_unowned=True)`
enforces it. Until 2026-10-03 only the download endpoint, `GET /agent/files/<id>/download`, made
that check. Since [Stage 30](agent-architecture-changes.md#stage-30), `get_file_record` makes it,
so every lookup by id does, from any tool. Another user's file answers like an id that was never
minted: **404, not 403**, because a 403 would confirm the id exists and make the endpoint an
enumeration oracle. A path into the store's `uploads/` or `outputs/` is checked as the record it
names, and `metadata/` cannot be reached by path at all.

A record also carries `session`, the conversation that made it. Outputs have been stamped since
2026-09-09. Uploads were not until 2026-10-03: the upload route bound the caller but never the
thread, so every upload landed in the unstamped pool. Two rules read the stamp:

- **A bare filename** resolves among this conversation's files plus the unstamped legacy pool
  (`find_files`, `resolve_file_ref`), never among another conversation's. Records written before
  stamping began carry no `session` and stay visible to every conversation, still subject to the
  owner check. That is deliberate: they are the reuse pool, saved embedding packages included.
- **An `overwrite=True` write** reuses only an output this conversation wrote for this owner
  (`_output_to_replace`, 2026-10-03). Before that it took any output with the same name.
  `qgis_metric_buffer` overwrote under the default name `buffer.geojson`, so one person's buffer
  took over another's file id, and the first person's link served the second person's bytes.
  The QGIS buffer and map render no longer overwrite at all. Each run is a new file, because an
  earlier answer still links to the old one. A write by path follows the same rule
  (`may_replace`, Stage 30).

An output may also carry `content_key`, a digest of the inputs that produced it (2026-10-03).
Map layers are keyed on that rather than on the file id, so a repeated step replaces its layer
instead of stacking a byte-identical copy.

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

## 5. Operational records — the host

Not agent data, but persistent state that outlives a container, kept so a failure leaves evidence.

**Container logs** go through compose's `journald` driver to the host journal (`/var/log/journal`,
persistent). A recreate, which every deploy performs, no longer deletes them, as the old
`json-file` driver did. Read them with `journalctl CONTAINER_NAME=agent-api --since "2 days ago"`.
Capped by size (`SystemMaxUse=3G`, `SystemKeepFree=5G`), with the per-unit rate limit off for
docker, since a failing service is exactly when logging bursts.

**Incident bundles.** `deploy/agent-watchdog.sh` runs every minute from a systemd timer and
watches `agent-api`, `mcp-server` and `embedding-server`. It restarts a container that has been
unhealthy for ten minutes, but first writes
`/var/log/iguide-agent/incidents/<utc>-<container>-<reason>/`. Each bundle holds `inspect`, the
health log, the last 20,000 lines of output, the container's process table, `docker stats`, the
host's disk, memory and recent kernel messages, six hours of journal, and `py-spy` stacks for
every process in the container. The stacks are the part that says *where* it was stuck. The
newest 60 bundles are kept; the restart budget (three an hour) lives in
`/var/lib/iguide-agent-watchdog/`.

On 2026-10-03 there were three. The first records the container found exited on 2026-09-22,
from an image that predated `py-spy`, so it has no stacks. The other two are real hangs, on
2026-09-27 and 2026-09-29. Each was restarted after ten minutes unhealthy and was healthy again
within 30 seconds, and both have stacks.

---

## Local mode

`AGENT_MODE=local` ([Stage 24](agent-architecture-changes.md#stage-24), 2026-10-03) is for one
developer's machine, and it changes which of the stores above are written.
`_get_opensearch_client()` in `rag_pipeline/memory_module.py` raises `PersistentMemoryDisabled`
**before** returning even a cached client, and every read and write of `chat_memory` and
`chat_traces` goes through it. So a local run writes neither, whatever its requests ask for.
Download links stay host-relative, so they resolve on the machine serving them. The file store
and code workspaces still work, on that machine's own disk. Knowledge-base search is unaffected:
it uses its own clients, and it only reads.

It exists because the override recipe it replaced failed twice. The main checkout's `.env` points
at shared infrastructure, and on 2026-10-01 a local verification run wrote seven conversations into
prod with nothing erroring.

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
the record's `fileIds` and each `artifacts[].file_id` point into the file store, which never
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
what that index exists for. Turns the index refused before M8.74 are not there (see 1b).

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
`DATABASE_URL` are unset. An optional Postgres record, `AGENT_KB_DB`, exists only on unmerged
branches (`backend_swap`, `claude/extraction-integration`); nothing on `prototype` reads it.
`NEO4J_CONNECTION_STRING` points at the *platform's* graph and is read for related-elements; the
agent writes nothing to it. OpenSearch's other indices belong to the
website — `chat_memory` is the agent's own.

---

## Keeping this current

Change what a store holds, where it lives, or what reclaims it, and change this file in the same
commit — the standing rule in `AGENTS.md`. The figures are illustrative and do not need chasing;
the structure does.
