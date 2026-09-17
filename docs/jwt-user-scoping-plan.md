# Per-user identity for the agent: JWT, owned files, owned chat history

**Status:** plan, not yet implemented. Step 0 (below) is DONE and deployed.
**Written:** 2026-09-17

## Goal

Load persistent chat history according to the caller's platform credentials, and make generated
and uploaded files reachable only by the user who owns them — reusing the I-GUIDE platform's
existing JWT rather than inventing a second identity system.

## What the platform already gives us

From `iguide-ue-backend/utils/jwtUtils.js` and `server.js`:

| | |
|---|---|
| transport | httpOnly cookie, name from `JWT_ACCESS_TOKEN_NAME` (dev: `jwt-access-token-dev`) |
| algorithm | HS256 symmetric, `JWT_ACCESS_TOKEN_SECRET`, `expiresIn: '1h'` |
| payload | `{ id, role }` only. `role` is numeric, **lower = more privileged** (`req.user.role <= requiredRole`) |
| cookie scope | `domain: JWT_TARGET_DOMAIN` = **`.i-guide.io`**, `sameSite: 'Strict'`, `httpOnly`, `secure` in prod |
| refresh | refresh tokens in OpenSearch `refresh_tokens`; `POST /api/refresh-token`; `GET /api/check-tokens` → `{id, role}` |
| dev bypass | header `JWT-API-KEY` == `JWT_API_KEY_VALUE`, only when `NODE_ENV === "development"` |

The agent VM's `.env` already carries this whole block, including both 128-char signing secrets.

### Why this works without a token exchange

The agent is served at `https://agent.i-guide.io` (DNS → 149.165.147.219, the same VM as the
jetstream name; the TLS cert's lineage is the jetstream name with `agent.i-guide.io` as a SAN).
The map UI is served from that same origin, so:

* the `.i-guide.io` cookie **is** sent to the agent — same-site, and same-origin from the UI;
* `sameSite: 'Strict'` is satisfied for navigations from `https://platform.i-guide.io`;
* `<img src="/agent/files/.../download">` is a same-origin request, so **signed download URLs
  are unnecessary** — the cookie rides along on the image request.

Because the signing secret is already present on the agent host, the agent verifies the token
**locally** (no round-trip to the backend, no dependency on backend uptime). Introspection via
`GET /api/check-tokens` was considered and rejected: its only real advantage is not holding the
secret, which is already moot.

## Step 0 — canonical origin (DONE, deployed 2026-09-17)

Prerequisite, because it silently breaks everything downstream if left wrong.

1. `AGENT_PUBLIC_BASE_URL` on the VM: `https://iguide-agent-dev...jetstream-cloud.org`
   → `https://agent.i-guide.io`. `file_store._with_public_url()` uses it to absolutize every
   emitted `download_url`; pointing at the jetstream name made every download link and every
   inline `![](...)` image **cross-site**, so the cookie would never be sent to them.
   Backed up as `.env.bak-basurl-20260917-073307`; container recreated; verified by uploading a
   probe file and reading back `https://agent.i-guide.io/agent/files/<id>/download`.
2. nginx: the main 443 vhost now answers only for `agent.i-guide.io`; a second vhost answers for
   the jetstream name and issues **307** to `https://agent.i-guide.io$request_uri`.
   * 307, not 301 — a 301/302 rewrites POST to GET and would silently break
     `POST /agent/chat/stream` for anything still on the old host. 307 also stays revertible.
     Promote to 308 once nothing uses the old name.
   * `/.well-known/acme-challenge/` is excluded from the redirect: the cert lineage is issued
     under the jetstream name and HTTP-01 renewal must still reach it.
   * `server_names_hash_bucket_size 128` in `nginx.conf` (the long jetstream name overflows the
     default 64 once it appears in a third server block).
   * Config backups live in `/etc/nginx/backups/`, **not** in `sites-enabled/` — nginx includes
     `sites-enabled/*` and parses a backup left there as live config.

Verified: jetstream GET → 307 → canonical 200; POST → 307 with method preserved; acme path 404
(not redirected); real Chrome lands on `agent.i-guide.io` with the stored API key intact.

## Step 1 — identity middleware

`api/server.py`. Today `_require_agent_chat_api_key()` is a *shared-secret gate*: one key for
everyone, carrying no identity. Identity is a second, orthogonal axis — keep the API key for
service callers.

* Read the cookie named by `JWT_ACCESS_TOKEN_NAME`; fall back to `Authorization: Bearer` for
  non-browser callers.
* Verify locally with PyJWT: **pin `algorithms=["HS256"]`** (never trust the token's own `alg`,
  never accept `none`), small `leeway` for clock skew.
* Put `{user_id, role}` into a contextvar.
* ⚠️ The SSE worker thread needs `contextvars.copy_context()` + `ctx.run()` — contextvars do not
  cross threads. This is exactly the bug that made the file-store session stamp come out `None`
  (`agent_runtime/graph_runtime.py`).
* Expiry, refresh and the 401-vs-403 contract: see **Step 1b** below.
* If we mirror the backend's dev bypass, gate it on its own explicit agent-side env var and make
  it fail closed — not on a shared `NODE_ENV`.

### Step 1b — expiry and refresh

The access token lives **one hour**. An idle tab that sends a turn 90 minutes later must not be
told to log in again — the browser still holds a valid refresh token.

**Who refreshes: the client, not the agent.** The agent stays a pure verifier. On expiry it
answers 401; the UI calls the backend's `POST /api/refresh-token` directly (credentialed,
same-site), which validates the refresh cookie against the OpenSearch `refresh_tokens` index,
mints a new access token, sets the `.i-guide.io` cookie, and returns `{id, role}`. The UI then
retries the agent request **once**.

The agent never reads, stores, relays or mints a refresh token, and never needs
`JWT_REFRESH_TOKEN_SECRET`. Minting stays in exactly one place.

**The status contract that makes this work** — mirror `authenticateJWT` exactly:

| condition | agent answers | client does |
|---|---|---|
| valid token, role ≤ 4 | 200 | — |
| **expired** (`ExpiredSignatureError`) | **401** | refresh once, then retry once |
| invalid signature / malformed / missing | 403 | show "please log in" |
| valid but role > 4 | 403 | show "insufficient role"; do **not** refresh |

Collapsing expired into 403 breaks this: the client can no longer tell "refresh me" from "give
up", so it either never refreshes or refreshes forever against a token that will never validate.
The backend already makes this distinction (`TokenExpiredError` → 401, everything else → 403);
we match it rather than invent our own.

**Bound the retry.** One refresh attempt per request, one retry after it. A 401 from
`/api/refresh-token` means the refresh token is gone or expired — stop and ask the user to sign
in. Never loop. (This repo has already burned a day on unbounded retry loops; see the
polling-loops note in `docs/agent-notes/`.)

**Long turns.** Validate at connect, never mid-stream: a turn may run for many minutes and the
token can expire while it streams. Killing a running turn at minute 40 because a clock ticked is
worse than finishing it. The next request refreshes.

**Backend change required — CORS.** `ALLOWED_DOMAIN_LIST` is currently
`["https://dev.i-guide.io", "http://localhost", "http://localhost:80", "http://localhost:5500",
"http://localhost/"]`. **`https://agent.i-guide.io` is not in it**, so a credentialed refresh XHR
from the agent's UI will be blocked by the browser — `jwtCorsMiddleware` will echo
`FRONTEND_DOMAIN` instead of the caller's origin. Adding the agent origin to that list is a
prerequisite, and it lives on the backend deploy, not ours.

**Security note to verify.** Both cookies are scoped to `.i-guide.io`, so if the refresh cookie's
path is `/`, the **agent receives the refresh token on every request** whether or not it uses it.
The refresh cookie is not set anywhere in `iguide-ue-backend` (the auth service sets it), so its
path needs checking. If it is `/`, that is one more reason the agent host — which runs
LLM-generated code in a Docker-socket sandbox — is a higher-value target than it looks, alongside
the HS256 secret already sitting there.

## Step 2 — file ownership

`agent_runtime/file_store.py`. The record today is
`{file_id, filename, kind, path, relative_path, size_bytes, download_url, session}` — there is a
session stamp but **no user**.

* Add `owner_id`, stamped in `save_uploaded_file`, `create_output_file`, and
  `create_output_file_from_path`.
* Enforce in `download_agent_file` (`api/server.py`): 404 (not 403) on a mismatch, so the
  endpoint does not confirm that a file id exists.
* **Legacy records have no owner.** Same shape as the `include_unowned` decision already made for
  sessions. For a security control, deny unowned rather than defaulting them public, and migrate
  deliberately.
* `role` can later allow an admin (low role number) to read across owners.

## Step 3 — chat history keyed by user

`rag_pipeline/memory_module.py`. `create_memory()` mints a bare UUID and
`get_or_create_memory(memory_id)` fetches by id with **no owner check** — anyone holding a
memory_id can read that conversation today.

* Add `owner_id` to the memory document; filter `get_memory` / `update_memory` on it.
* Add a `term: {owner_id}` query — "list my conversations" is the feature this whole plan is for.
* `agent_runtime/session_memory.py` needs the same key. It is a process-local cache, not the
  source of truth, so it must not be the thing that enforces the boundary.

### Step 3b — what a conversation actually contains

The server's persistent memory stores `{userQuery, messageId, answer, elements, ratings?}` per
turn. The map UI holds considerably more, and **the client already defines the right shape**:
`map-ui-prototype/src/sessionStore.ts` was written server-shaped on purpose — "moving it behind a
per-user endpoint later means swapping the transport, not the record". So the target is not a new
design; it is making the server store `StoredSession`.

| `StoredSession` field | in server memory today |
|---|---|
| `messages` (rendered turn + trace) | only `userQuery` / `answer` strings |
| `layers` | **missing** |
| `fileIds` | **missing** |
| `threadId` | missing (`memoryId` is the doc id) |
| `region`, `model`, `provider` | missing |
| `title`, `createdAt`, `updatedAt` | only `conversationName` |
| `elements`, `ratings` | present server-side, not modelled client-side |

Both directions need closing: the server gains layers/files/thread/region/model/timestamps, and
the client record keeps `elements` and `ratings` so citations and feedback survive a reload.

**Losing `fileIds` is a functional break, not a cosmetic one.** The full file toolset attaches
only `if input_file_ids:` — a conversation restored without them silently loses the ability to
analyse its own uploads, and the agent will say it cannot see files the transcript plainly shows.
This is the same registered-vs-reachable trap that `list_conversation_files` already hit.

**Layers restore by pointer, so ownership and retention become load-bearing.** A stored layer
keeps a `sourceUrl` and re-fetches through the same path the live `map_layer` event uses
(geometry is inlined only under ~2 MB, for layers delivered with no url). Two consequences once
Step 2 lands:

* the re-fetch is an ordinary `GET /agent/files/<id>/download` **as the restoring user**, so
  owner enforcement must admit them — and any conversation whose files predate `owner_id` will
  fail to restore its layers if unowned records deny. Backfill before enforcing.
* `sessionStore.ts` already relies on "retention is disabled server-side, so the url stays good".
  That assumption is now structural: **file retention must be ≥ conversation retention**. If
  `AGENT_FILE_RETENTION_DAYS` is ever enabled, restore must degrade *visibly* — "this layer's
  data has expired" — not silently draw an empty map. Silent is the worst outcome here, because
  the stored answer text says the features are on the map (synthesis rule 7), so a silently empty
  restore makes the transcript lie.

**Store markdown, not rendered HTML.** The client currently keeps rendered answer HTML. Persisting
that server-side freezes today's renderer into the record and stores active markup as if it were
data; keep the markdown source and re-render on load, so renderer fixes apply retroactively.

**One authority.** Once the server owns conversations, the client's IndexedDB is a cache, not a
second source of truth — otherwise a rename on one device silently loses to a stale write from
another. `dev` and `demo` modes keep using IndexedDB alone; `token` mode reads through to the
server.

## Three deployment modes

One mode is active at a time, named, explicit, and logged at boot. Three named profiles beat
three independent booleans: the booleans have eight combinations and five of them are nonsense.

```
AGENT_MODE=dev | demo | token        # default: dev
```

| | **dev** | **demo** | **token** |
|---|---|---|---|
| who it is for | the team | a public showcase | the integrated platform |
| settings panel | shown | hidden | hidden (profile instead) |
| credential | `X-API-KEY` required | none required | JWT cookie |
| model | per-request, user-selectable | forced `DEMO_MODEL` | per-request or server default |
| identity | none | none | `{id, role}` from the JWT |
| chat history | session-scoped, process-local | session-scoped, process-local | **per user, durable** |
| files | session-scoped | session-scoped | `owner_id`-scoped |
| extra UI | — | curated suggestions | user profile, "my conversations" |

Two of these already exist in all but name: `demo` is today's `DEMO_MODE=true` (hidden settings,
no API key, forced model) and `dev` is today's default. The new work is `token` plus collapsing
the existing flags behind one selector.

### Where the migration state went

An earlier draft of this plan proposed a tri-state `AGENT_USER_AUTH=off|optional|required`. The
mode selector replaces it as the *user-facing* knob, but the middle state still has to exist
somewhere: you cannot go from "nothing has an owner" to "ownership enforced" without a window in
which identity flows and records get stamped while nothing is rejected yet. So it survives as a
**temporary override inside token mode only**:

```
AGENT_TOKEN_STRICT=0      # token mode, but a missing/invalid JWT is not rejected
```

Used during rollout and backfill, then deleted. It is not a fourth mode and should not outlive
the migration.

### Service callers still need a way in

`token` mode cannot simply drop the API key. The eval harness (`scripts/run_eval_cases.py`,
`scripts/eval_baseline.py`), the benchmark runner, and any scripted client have no browser and no
JWT. In token mode the server therefore accepts **either**:

* a valid JWT → user-scoped: `owner_id` set, durable history, profile; or
* the service API key → no user identity, so it falls back to **session** scoping.

The rule that must not be broken: "no identity" resolves to *session* scoping, never to a single
shared owner. Otherwise every anonymous or service caller lands in one bucket and sees each
other's files.

### User profile

The JWT payload is only `{id, role}` — no name, email, or avatar. Rendering a profile needs a
backend call to `GET /api/users/:id` (cookie-authenticated, `routes/users_v2.js:83`), which the
agent can proxy or the UI can call directly from `platform.i-guide.io`. Cache it for the token's
lifetime; it is not needed to authorize anything, only to display.

### Durable storage is token mode's real prerequisite

`dev` and `demo` can keep the process-local `session_memory`. `token` cannot: "load my
conversations" has to survive a restart and a second worker. That makes the durable per-user
store the gating dependency for this mode — and it is the same work as the durable checkpointer
already queued as step 2 of the autonomous-agent roadmap. Worth doing once, for both.


## Decisions taken

* **Anonymous in token mode → "please log in".** No throwaway session, no degraded mode. A visitor
  without a JWT gets an explicit sign-in message, not a 403 with no explanation and not a silent
  empty history. `/agent/ui-config` reports the mode so the client can render this state rather
  than discovering it from a failed POST.
* **Role-gated at `role <= 4`** (`UNRESTRICTED_CONTRIBUTOR`), using the backend's own convention
  that lower is more privileged. Admits SUPER_ADMIN (1), ADMIN (2), CONTENT_MODERATOR (3) and
  UNRESTRICTED_CONTRIBUTOR (4). **Excludes TRUSTED_USER_PLUS (5), TRUSTED_USER (8) and
  UNTRUSTED_USER (10)** — so an ordinary logged-in .edu account is refused. This is deliberate:
  the agent spends LLM budget and runs generated code in a sandbox, so access starts narrow and
  can be widened by raising one constant. Two consequences follow:

  1. **"Not logged in" and "logged in but not permitted" are different answers** and must not
     share a message. The first is fixable by the user (sign in); the second is not (request
     access). Since the threshold is restrictive, the *second* is what most platform users will
     hit, so that message carries the weight — it should name the required role and say how to
     ask for it, not just say "forbidden".
  2. **Role is baked into the token at mint time and the token lives an hour.** A promotion does
     not take effect until the user's next token, and neither does a demotion. Acceptable for
     granting access; worth knowing if a role is ever revoked urgently, where the only fast
     remedy is invalidating the refresh token server-side.
* **Mode does NOT decide the API key.** `AGENT_MODE` governs UI and identity; `AGENT_CHAT_API_KEY`
  keeps governing service gating exactly as it does today (set → required and fails closed, unset
  → open with a loud boot warning). Otherwise `AGENT_MODE=dev` would mean one thing on a laptop
  and something dangerous on the deployed dev tier, which is public. A local developer simply does
  not set the key; the deployed dev server does. This makes `dev` mode a pure rename of what is
  running today, with no security change and no migration risk.

## Test plan

* Unit: token accepted / expired / wrong-signature / `alg: none` rejected / missing cookie.
* Unit: expired token → **401**, invalid signature → 403, role > 4 → 403. The three are distinct
  and a client can act on them without parsing a message body.
* Unit: the retry is bounded — one refresh, one replay, then a terminal sign-in state; a refresh
  that itself 401s does not trigger another.
* Unit: role gate — 1, 2, 3, 4 admitted; 5, 8, 10 refused; a missing or non-numeric `role` claim
  refused rather than defaulted. The two refusal messages ("sign in" vs "insufficient role") are
  distinct and neither leaks the other's condition.
* Unit: file record stamped with owner; download by owner 200, by another user 404, unowned 404.
* Unit: memory read and write filtered by owner; cross-owner read returns empty, not another
  user's turns.
* Round-trip: save a session with layers (one by-url, one inline), uploads and citations; reload
  it and assert the map draws the same layers, the downloads section lists the same files, and
  `fileIds` is intact so the file toolset still attaches.
* Degradation: a stored layer whose file has been swept restores as a visible "expired" state,
  never as a silently empty map.
* Threading: assert the owner is visible *inside* the SSE worker thread — the regression that
  already bit once.
* Live, in real Chrome on `agent.i-guide.io` (never the in-app pane, which has no key):
  a full turn that produces a file, with the inline image rendering from the cookie alone.

## Open questions

1. **Production cookie name and secret** — everything above is read from the dev `.env`
   (`jwt-access-token-dev`, `NODE_ENV=development`). Confirm the prod values before shipping;
   read the name from env, never hardcode it.
2. **The dev JWT bypass is currently active** (`NODE_ENV="development"` + a populated
   `JWT_API_KEY_VALUE`), so a `JWT-API-KEY` header skips authentication entirely on that tier.
   Intended for dev, but it means the dev tier's auth is header-bypassable.
3. **HS256 on a host that runs LLM-generated code.** The 128-char secret sits on the agent VM,
   which runs generated code under a Docker-socket sandbox; anything holding it can mint a valid
   token for any platform user. Pre-existing, not introduced here, but it is the argument for
   moving the platform to RS256 and giving the agent only a public key. Separate ticket.
4. *(resolved — see Decisions taken: `role <= 4`.)*

## Depends on someone else

* **`https://agent.i-guide.io` must be added to the backend's `ALLOWED_DOMAIN_LIST`** before
  client-driven refresh can work (see Step 1b). Backend deploy, not ours.
* **Production `JWT_ACCESS_TOKEN_NAME` / `JWT_ACCESS_TOKEN_SECRET`** — every value read for this
  plan came from the dev tier (`jwt-access-token-dev`, `NODE_ENV=development`). Needed at prod
  cutover, not before Step 1.

## Rollback

* Step 0: restore `.env.bak-basurl-20260917-073307` and the nginx backup in `/etc/nginx/backups/`,
  then `docker compose up -d agent-api` and `sudo systemctl reload nginx`.
* Steps 1–3: set `AGENT_MODE=dev` — identity stops being parsed and the store reverts to session
  scoping, with no redeploy and no data migration. `AGENT_TOKEN_STRICT=0` is the softer step:
  stay in token mode but stop rejecting.
