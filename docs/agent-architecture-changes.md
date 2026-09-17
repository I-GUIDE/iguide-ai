# Agent architecture changes: what the supervisor is told, what tools accept, and who the caller is

**Covers:** 16 commits on two branches, both cut from `origin/prototype` at `9e35950`.
**Written:** 2026-09-17

| branch | commits | subject |
|---|---|---|
| `claude/evidence-summary` | 6 (`1044afb` … `25c3e1e`) | what the supervisor's decider reads; what the tool surface accepts |
| `claude/jwt-identity` | 10 (`152a537` … `4daf2e1`) | deployment modes, platform identity, record ownership |

The branches are **unmerged** and independent — neither touches a file the other touches, and
neither is a prerequisite for the other. Deployment state is at the end; do not infer it from the
commits.

This is organised as six **stages**. A stage is one coherent architectural shift, not one commit:
the evidence-summary branch contains two unrelated shifts interleaved in time (the supervisor work
is commits 1, 2 and 6; the tool work is 3, 4 and 5), and the jwt branch's ten commits collapse into
four. Every change carries the measurement that motivated it, because without the number the reason
is just an opinion.

---

## Stage 1 — The decider is told what happened, not just how much of it there is

Commits `1044afb`, `99d71ad`, `25c3e1e`. All in `agent_runtime/supervisor/graph.py` plus one new
module.

The supervisor picks one action per step — `search`, `analyze`, `code`, `done` — from a JSON payload
built by `_distill(state, for_decision=True)` and a hand-written prompt built by
`default_decide_fn`. Three separate things it was choosing from turned out to be wrong in the same
way: each was *cheap and available* rather than *what the decision actually needed*.

| | before | after |
|---|---|---|
| what the evidence is | counts, titles, `topical_coverage`, `top_score` — all lexical | plus `evidence_summary`, four sentences written by the model that read the documents |
| what peers can do | a hand-written paragraph, derived from nothing | generated from `agent_runtime/capability_registry`, with a test holding the registry against the peer builders |
| what this turn already did | `has_analysis` / `has_code` booleans and counts; the ledger of *previous* turns only | the same `this_turn` ledger lines the answering model and the grounding auditor already read |

### 1.1 Evidence gains a description (`1044afb`)

**The measurement.** Live on a self-hosted model: two full search rounds where the second added
nothing the first had not. "Is this enough?" was being answered from counts. The signals available
were all lexical — they can report "8 documents, 0.75 of them mention your subject terms" and still
leave the decider unable to separate a set of PySAL accessibility notebooks from a set of DEM
sources, because both mention "elevation".

**Structural.** `SupervisorState` gains `evidence_summary: Optional[str]`. The search node's state
update now computes it:

```python
"evidence_summary": _summarize_evidence(llm, q, merged) or state.get("evidence_summary"),
```

The `or` matters: a failed summarisation on round two keeps round one's summary rather than blanking
it. `_distill` emits the field beside the lexical signals — `evidence_titles`, `evidence_sources`,
`topical_coverage`, `top_score` — deliberately *beside* and not *instead of*, so a wrong summary is
something the decider can disagree with rather than something it must obey.

Failure is always `None`, from every direction: no LLM, no documents, a thrown exception, an empty
reply, or `AGENT_EVIDENCE_SUMMARY=0` — a summary is an aid to a decision, never a precondition for
one. Three caps, because it rides in every later decision prompt for the rest of the turn:
`_EVIDENCE_SUMMARY_DOCS = 8` documents in, `_EVIDENCE_SUMMARY_SNIPPET = 400` characters of each,
`_EVIDENCE_SUMMARY_MAX_CHARS = 700` out.

**Prompt (new).** `_EVIDENCE_SUMMARY_PROMPT` asks for at most four sentences covering what the
results contain, which parts of the request they address, and which they do not. The constraint is
the interesting half:

> Describe only. Do NOT recommend an action, do not say whether to search again, and do not say
> whether the evidence is sufficient — another step decides that and needs your description, not
> your verdict.

This is a separation-of-powers rule, not politeness. The decider also sees the deterministic signals
and the action history; a summary that announced "this is enough" would collapse two independent
checks into one and give a weak model's opinion the final word.

**Prompt (decider, revised).** The decider prompt gains a paragraph telling it how to weigh the new
field — because a new field with no guidance is a field a model will over-read:

> `evidence_summary` in Progress is a DESCRIPTION of what was retrieved […] It deliberately does not
> say whether the evidence is sufficient — that is your call. Search again only when it names a
> specific gap a DIFFERENT query could fill; repeating a search because the count looks small
> returns the same documents and wastes the step. And when the request is work for a tool rather
> than a question about the literature — computing a DEM, buffering, embedding a region — retrieval
> cannot help at all, however thin the evidence looks.

Two behaviours targeted: the observed repeat-search (fixed by "a DIFFERENT query"), and searching at
all for something no document contains (fixed by the tool-work clause, which is the same failure
Stage 1.2 attacks from the other side).

Guarded by `rag_pipeline/tests/test_evidence_summary.py`.

### 1.2 The supervisor's view of its peers becomes generated (`99d71ad`)

**The measurement.** The capability paragraph had drifted behind the peers three times. Terrain,
administrative boundaries and geocoding were all bound to a peer while the supervisor's description
of that peer never mentioned them. The visible cost: asked for a DEM, the supervisor searched the
knowledge base for "digital elevation model", because as far as it had been told, `analyze` did
overlays and embeddings. That was a *correct decision from a stale description* — the distinction
matters, because a model that gets this right is guessing past the prompt rather than following it,
and that is not a property you can rely on.

**Structural.** New module `agent_runtime/capability_registry.py`. A frozen `Toolset(factory,
summary)` dataclass, three tuples — `_SHARED` (14 toolsets bound by both peers), `_ANALYZE_ONLY` (3),
`_CODE_ONLY` (1) — and `CAPABILITIES: Dict[str, Tuple[Toolset, ...]]` mapping `analyze` and `code` to
their unions. The three-way split, rather than one flat list, is what stops a capability only one
peer binds from being described as if both had it. Two functions: `factories()` for the guard,
`describe(capability)` for the prompt.

`graph.py` gains `_capability_inventory(capability)`, which calls `describe()` and falls back to the
generic phrase `"geospatial analysis over the evidence or uploaded files"` if the import fails —
a prompt must still be produced.

**What deliberately stayed hand-written.** The REASONING guidance: when to stop, how to read the
evidence summary, that model names are arguments rather than datasets. That is judgement, not
inventory, and generating it would lose the nuance that makes it useful. The split is the point of
the design — the inventory is a fact that drifts, the guidance is a decision that does not.

**Prompt (decider, revised).** The `analyze` clause was:

> analyze: run a GIS/data analysis workflow with EXISTING purpose-built tools (QGIS/PyQGIS,
> overlay/buffer/clip/dissolve, aggregation, temporal analysis, statistics, vector
> inspect/plot/reproject) over the evidence or uploaded files.

It is now the generated list, plus a hand-written sentence that converts the inventory into a routing
rule:

> Anything in that list is analyze work, not a retrieval question — a DEM, a boundary and a geocode
> all come from live services, not from the knowledge base, so searching for them finds writing
> ABOUT them and never the thing itself.

The `code` clause also changed, from `"produce and run NEW code for work no existing tool covers"` to
the same plus `"It binds the same toolkit as analyze, plus packaged skills and saved workflows"` —
previously the prompt implied `code` was a bare interpreter, which understates it and encourages
writing from scratch what a bound tool already does.

**The guard came first and earned its place immediately.**
`rag_pipeline/tests/test_supervisor_knows_its_peers.py` was written against terrain alone and failed
on its first run, naming two more undescribed toolsets nobody had noticed. It holds the registry
against reality in **both** directions:

- `test_the_registry_describes_everything_the_peers_bind` — `_bound_toolsets()` regexes
  `make_*_tools(` call sites out of `graph.py` and diffs them against `factories()`.
- `test_the_registry_does_not_claim_tools_the_peers_do_not_bind` — the reverse, which matters
  independently: promising a capability that is not bound routes work to a peer that cannot deliver,
  and the turn then fails much further downstream where the cause is hard to see.
- `test_the_inventory_actually_reaches_the_prompt` asserts `_capability_inventory("analyze") ==
  describe("analyze")`, because a silent fallback would leave every keyword check passing against
  leftover prose. The keyword checks (`REQUIRED_TERMS`) now read the prompt *the model receives*,
  composed at runtime, rather than the source file — which no longer contains the inventory.

### 1.3 This turn's ledger reaches the decider (`25c3e1e`)

**The measurement.** Asked for a DEM, the supervisor ran `analyze` — which fetched it and drew it —
then routed to `code`, which fetched the same DEM again and drew a second copy. In the sweep that
second pass cost **266 seconds and 16 `execute_code` iterations** to redo work one tool call had
already done.

**Structural — this is a readership change, not a new computation.** Nothing was missing. The rows
accumulate on `state["action_rows"]`; `_ledger_line` already renders exactly what is needed,
including `[produced area_dem.tif, file_id ...]` and `FAILED ... -> DID NOT RUN`; the trace UI
already shows them as `dem_for_region(...) -> 1 layer on the map`. `_map_delivered_this_turn`
already existed (added on `prototype` in `8f9f24a`) and already required the tool to have succeeded.

The gap was **who got to read them**. `_ledger_lines` had exactly two consumers — the answering model
(`_prior_actions_note`) and the grounding auditor — and the decider was not one of them. `_distill`
handed it counts and flags about the current turn plus the ledger of *previous* turns. So it knew
what earlier turns did and what this turn's state totals were, but not what this turn's tools had
actually done. Routing to a peer to redo finished work was a reasonable decision from that view.

`_distill(..., for_decision=True)` now emits `this_turn` in the same rendering, with a note:

> What THIS turn has already done, oldest first — the same record the answering model and the auditor
> see. A line here is work that is DONE: routing to a peer to redo it produces a second copy, not a
> better answer. A line marked FAILED means the tool did not run and its result does not exist.

The last sentence exists because a ledger without it is ambiguous in the dangerous direction: a
decider that reads `FAILED` as "was attempted, so it exists" stops re-running something that never
produced anything.

This is the same fix already made for the grounding auditor on `prototype`, which had the identical
blind spot and audited a turn against evidence its own ledger contained.

**Revised during the work: the boolean became a ledger.** The first attempt was the narrow fix —
surface `map_layer_delivered`, the one question the prompt asks directly. The test file is still
named `test_delivered_signal.py` and its docstring is still about the boolean. That fix was too
small: it answers "is a layer on the map?" and nothing else, so the same class of duplicated work
recurs for any deliverable that is not a layer (a written file, a computed statistic, a failed tool
whose failure the decider cannot see). The boolean is **kept alongside** as an unmissable signal for
the question the prompt asks in one line, and the ledger carries the general case.

**Prompt (decider, revised).** A rule for the boolean, framed as a signal rather than a veto:

> `map_layer_delivered` in Progress means a layer is ALREADY on the user's map. When the request was
> to see something and it is there, choose `done` […] Choose `code` after a successful analyze only
> when the request asks for something the delivered result does not contain.

Deliberately not a hard veto on `code`: "map it, then compute the statistics" legitimately needs
`code` after a successful `analyze`, and vetoing `code` whenever a layer exists would break it.

### Decider prompt revisions in Stage 1, in one place

| revision | commit | behaviour it targets |
|---|---|---|
| `analyze` capability clause replaced by generated inventory + "not a retrieval question" | `99d71ad` | a DEM request becoming a knowledge-base search |
| `code` clause gains "binds the same toolkit as analyze, plus skills" | `99d71ad` | `code` read as a bare interpreter |
| `evidence_summary` guidance paragraph | `1044afb` | a second search round that returns the first round's documents |
| `map_layer_delivered` rule | `25c3e1e` | routing to `code` to redo a delivered layer |
| `this_turn` + `this_turn_note` in the decision payload | `25c3e1e` | the general case of the above |

---

## Stage 2 — The tool surface absorbs how models actually call tools

Commits `8fe0dcb`, `ae862b5`, `5a10a58`.

Four separate live failures with one shape: the model's call was reasonable, the tool refused it, and
the error message was accurate but useless. The position taken across all four is that **the fix
belongs to the tool, not the model** — a parameter whose name invites the wrong value is a tool
defect, and no amount of prompt engineering fixes a schema that rejects a call before any of our code
runs. A tool's docstring and its error strings are LLM-facing prompts too; the rewrites below are
listed with the tool they belong to rather than with Stage 1's decider prompt.

### 2.1 An explicit `null` means "use the default" (`8fe0dcb`)

**The measurement.** A sweep of the deployed agent over ten spatial prompts with `gpt-oss:120b`:

```
embed_region({..., 'buffer_m': None, 'lon': None, 'lat': None,
              'bbox': None, 'start': None, 'end': None})
ValidationError: 3 validation errors for embed_region
```

Nothing was wrong with that call. A model writing tool arguments fills every slot the schema offers
and puts `null` in the ones it does not need; `start=None` means "no opinion about the date window",
which is precisely what the default expresses. The tool had a good answer and refused to use it over
a type annotation.

**Why this could not be a few patches.** A scan of the surface found **141 such parameters across 62
of the 80 exposed tools**. Nor can it be fixed inside the function bodies: pydantic validates against
the *schema*, which LangChain infers from the signature, so the call dies before any of our code
runs. The signature itself has to say nullable.

**Structural.** New module `agent_runtime/tool_args.py`, one function
`accept_null_defaults(func)`. It rewrites `func.__signature__` and `func.__annotations__` so every
defaulted parameter becomes `Optional[...]`, and wraps the call so a `None` arriving for one of those
is replaced by its original default. Setting those two attributes *is* the fix rather than a cosmetic
touch-up, because they are what LangChain reads.

Applied at every `StructuredTool.from_function(func=...)` call site across 15 modules in
`agent_runtime/` — the analysis, geo, terrain, rs-embed, file, exec, MCP, quality, skill and
granular toolsets.

**What it deliberately does not do.** This widens what is *accepted* and changes nothing about what a
tool then does:

- a real value still wins;
- an omitted parameter behaves exactly as before;
- a parameter with **no** default stays required, because there a `null` genuinely is an error;
- a parameter already defaulting to `None` is untouched, and an unannotated one has nothing to widen;
- `0` and `False` are **not** treated as null. They are answers, and substituting the default for
  them would silently ignore the caller — a worse bug than the one being fixed.

**Guards.** `rag_pipeline/tests/test_null_tool_args.py` asserts that no exposed tool refuses a null
optional, so the 141 cannot quietly come back; a second test guards the *premise* by checking the
unwrapped function still raises. If that ever stops being true, the wrapper is solving nothing and
should be removed rather than carried.

### 2.2 A GeoTIFF reaches the map on the first call (`ae862b5`)

**The measurement.** From a live trace, a DEM took **four calls** to draw:

| call | result |
|---|---|
| `add_map_layer(.tif)` | "unreadable vector/tabular source", hinted at shapefile sidecars |
| `add_raster_layer(.tif)` | "not an image" |
| `add_map_layer(.png)` | "an image has no geometry" |
| `add_raster_layer(.png)` | worked |

Every message was accurate and none was useful, because a GeoTIFF is neither of the two things these
tools knew about. This is **the same capability, callable now** — nothing new can be drawn that could
not be drawn before; it now takes one call instead of four.

**Structural.** In `agent_runtime/langchain_geo_tools.py`:

- a `_GEOTIFF_EXTS = {".tif", ".tiff"}` class alongside the existing `_IMAGE_EXTS` /
  `_MAPPABLE_EXTS`, so the two tools stop having to force a GeoTIFF into one of the other two;
- `_geotiff_to_drapable(path, name_on_disk)` renders the raster to a PNG and reads its true extent
  out of the file, reprojecting to lon/lat with `rasterio.warp.transform_bounds` when the source is
  projected. It reuses `terrain_tools._render` rather than a matplotlib figure on purpose: axes,
  margins and a colorbar would become part of the image, and a draped layer is positioned solely by
  its bounds, so the pixels would stop lining up with the ground. Missing `rasterio` returns a tool
  error naming the alternative, not an exception.

**Signature change.** `add_raster_layer(file_id, bounds, name, opacity)` →
`add_raster_layer(file_id, bounds=None, name=None, opacity=0.85)`. `bounds` becomes optional.

**The precedence rule and why it is not "caller wins".** For a georeferenced file the FILE wins even
when `bounds` were passed. A caller restating that box can only agree or be wrong, and a wrong box
draws a plausible layer in the wrong place that nothing downstream can detect — that misregistration
already cost three rounds of debugging once, when a DEM was draped over the requested bbox instead of
the one actually served. The single exception is a GeoTIFF carrying no CRS: there the file knows
nothing and the caller's box is all there is, so `bounds or drawn["bounds"]` applies.

**Error message rewrite.** `add_map_layer` on a `.tif` no longer sends the model looking for a `.shx`
that does not exist. It now names the other tool and pre-empts the next failure:

> Drape it with add_raster_layer, passing this same file_id — it reads the bounds out of the GeoTIFF,
> so you do not need to supply them.

### 2.3 `session_context_json` takes the object as readily as the string (`ae862b5`)

**The measurement.** The parameter says "json", and a model that took that at face value sent the
**object**, which pydantic rejected before the tool ran. Observed live: the identical search issued
twice, once as a dict and once stringified, the first call wasted entirely.

**Signature change.** In `agent_runtime/langchain_granular_tools.py`:
`opengeodata_search_tool(query, limit=8, session_context_json: Optional[str])` →
`Optional[Union[str, Dict[str, Any]]]`. A `Mapping` is copied directly; a string still goes through
`json.loads`, now also catching `TypeError`. Nothing is gained by demanding the caller serialise
something parsed on the next line.

### 2.4 `admin_boundary` reads inverted arguments instead of failing four times (`5a10a58`)

**The measurement.** Live with `gpt-oss:120b`, asked for the DEM of Urbana:

```
admin_boundary({'state':'Illinois','level':'city','name':'Urbana','area':'city'})  failed
admin_boundary({'name':'Urbana','area':'city','state':'Illinois','level':'city'})  failed
admin_boundary({'area':'city','state':'Illinois'})                                 failed
admin_boundary({'area':'Urbana','state':'Illinois','level':'city'})                worked
```

`area` holds the place NAME but reads like the KIND of place, so the model put `"city"` in it — and
had the right answer sitting in `name` from the very first call, because `name` is the *output
filename stem*. It had the two slots exactly inverted, three times.

**Structural.** `agent_runtime/admin_boundary_tools.py` gains `_LEVEL_WORDS` (city, county, state,
place, town, municipality, cdp, tract, block_group and plurals). When `area` holds one of them:

- if `name` holds something that is *not* a level word, that is the place. The tool swaps them,
  keeps the level word as `level`, and **says so** in a `note` on the successful result — a silent
  correction teaches the caller nothing, and the next call repeats the mistake. The recovery is safe
  precisely because `level` already carries the kind of place, so an `area` holding a level word has
  exactly one sensible reading;
- if nothing can be recovered (`area='city'` with no usable `name`, which is what three of those four
  calls looked like), it refuses — but the error now describes the **fix** rather than the symptom.

**Error message rewrite.** `"no incorporated place named 'city'"` was accurate and told the model
nothing about which argument was wrong. It is now:

> `area` is the place NAME, not the kind of place — 'city' is a level.
> hint: Call it as admin_boundary(area='Urbana', level='city', state='Illinois'). `level` takes
> city/county/state/cdp; `name` is only the output filename.

**Docstring revision** (the model's primary source for this): the summary now opens with `` `area` is
the PLACE NAME — "Urbana", "Champaign County", "Illinois". It is NOT the kind of place `` and
disambiguates `name` in the same breath.

### New capability vs. same capability, callable now

| change | which |
|---|---|
| `accept_null_defaults` across 62 tools | same capability, callable now |
| `add_raster_layer` accepting a GeoTIFF and deriving bounds | **new capability** — it renders and reprojects, which no tool did before |
| `add_raster_layer(bounds=None)` | same capability, callable now |
| `session_context_json` accepting a dict | same capability, callable now |
| `admin_boundary` argument recovery | same capability, callable now |
| the three error/docstring rewrites | same capability, findable now |

---

## Stage 3 — Deployment configuration becomes named, not inferred

Commits `152a537`, `f9b7081`. Two applications of one rule: when N independent knobs have to agree,
replace them with one name that sets them all, and make an unrecognised name *raise* rather than fall
back.

### 3.1 `AGENT_MODE` (`152a537`)

**The reason.** Three independent flags have eight combinations and **five of them are nonsense** —
"settings hidden AND a key required" is a page that demands a credential it gives you no way to
enter. `AGENT_MODE=dev|demo|token` has three states and each sets every axis coherently.

New module `agent_runtime/deployment_mode.py`: `current_mode()`, `is_dev()`, `is_demo()`,
`is_token()`, `boot_warning()`. `api/server.py`'s `_demo_mode()` becomes a one-line delegate, kept as
a named helper only because a dozen call sites read it.

**Behaviour-preserving by construction.** `DEMO_MODE=true` with no `AGENT_MODE` still selects demo,
which is how the deployed server is configured, so its behaviour is unchanged — a refactor that
quietly changes what a live server does is not a refactor. `/agent/ui-config` gains `mode` and
**keeps** `demo_mode`, because dropping it would blank the settings panel on every page still holding
a pre-mode bundle.

**What the mode deliberately does not decide: the API key.** Letting it would make `AGENT_MODE=dev`
mean one thing on a laptop (harmless) and something else on the deployed dev tier, which is public.
`AGENT_CHAT_API_KEY` keeps governing service access on its own, in every mode.

**An unknown `AGENT_MODE` raises.** This selects security behaviour, and a typo that silently
resolves to a working mode is the failure nobody notices on a public host. A container that refuses to
boot is noticed immediately.

Boot logging moves from "warn only in demo" to "state the mode on every boot, plus a warning when the
mode is open" — the mode an operator *thinks* is set is the one thing worth saying out loud.

`docs/jwt-user-scoping-plan.md` (360 lines) lands in this commit as the design for the rest of the
branch.

### 3.2 `PLATFORM_TIER` (`f9b7081`)

**The reason.** The same shape one level down. Three URLs — where the browser refreshes, where an
unsigned-in visitor goes, where the agent asks who a caller is — were being set one at a time, which
invites exactly the state this prevents: two pointing at dev, one left on prod, and a verification
that fails for a reason nobody can see from the outside.

**Revised during the work.** `84c60b2` had introduced `refresh_url` / `signin_url` on
`/agent/ui-config` read straight from `PLATFORM_REFRESH_URL` / `PLATFORM_SIGNIN_URL`; `95a48de` then
added a third, `PLATFORM_CHECK_TOKENS_URL`, read independently inside `identity.py`. Three
independently-set URLs that must agree is the problem `AGENT_MODE` had already solved once, so
`f9b7081` replaces the reads with a resolver.

New module `agent_runtime/platform_endpoints.py`: a `_TIERS` table (`dev` →
`backend-dev.i-guide.io` / `dev.i-guide.io`; `prod` → `backend.i-guide.io` /
`platform.i-guide.io`) and `refresh_url()`, `signin_url()`, `check_tokens_url()`. Each explicit
`PLATFORM_*_URL` still wins, because a tier table cannot anticipate a staging host somebody stands up
next month. An unrecognised tier raises: picking the wrong platform silently verifies tokens against
a backend that never minted them, and the resulting "invalid token" explains nothing.

The pairings are **verified, not assumed** — each backend answers `/api/refresh-token` with its 401
"no refresh cookie" reply, and each frontend answers `/auth/login` with a 302 to CILogon.

**The one half-switched state that remains is named rather than guessed at.** The cookie NAME is the
platform's own setting (`JWT_ACCESS_TOKEN_NAME`), differs between tiers, and does not move with
`PLATFORM_TIER`. Mismatched, every request fails and looks like a rejected token rather than a
misconfiguration. `consistency_warning()` says so at boot — and does **not** invent the right name,
because prod's is not known.

---

## Stage 4 — A request gains a caller

Commits `69863b8`, `95a48de`, `c59a69d`. New module `agent_runtime/identity.py` plus enforcement in
`api/server.py`.

Before: every agent endpoint was gated, at most, by a shared secret — `_require_agent_chat_api_key()`
answers "does this caller hold the key", which carries no identity. After: identity is a second,
orthogonal axis, and in token mode it is the *stronger* credential.

### 4.1 Verifying the platform's JWT (`69863b8`)

**Why verify rather than exchange.** The agent is served from `agent.i-guide.io`: same origin as the
map UI, same registrable domain as the platform, so the `.i-guide.io` access cookie arrives here on
its own — including on the `<img>` request for an inline map artifact. That is why downloads need no
signed URL.

**Why locally.** Calling the backend's `/api/check-tokens` per request would buy only "the agent does
not hold the signing secret", and the secret is already in this deployment's `.env`.

**Four choices, each of which is a way this fails open if reversed.**

| choice | what reversing it does |
|---|---|
| `algorithms=["HS256"]` pinned | a decoder that trusts the token's own `alg` accepts the `none` forgery the caller wrote |
| `options={"require": ["exp"]}` | a token issued once is valid forever |
| missing / non-numeric `role` refused, never defaulted | `0` would make a broken token the most privileged caller on the system |
| `TokenExpired` raised separately from `TokenInvalid`; 401 vs 403 | the client cannot tell "refresh and retry" from "give up", so the UI either never refreshes or refreshes forever against a token that will never validate |

`_coerce_role` also rejects a boolean explicitly, since `bool` is an `int` in Python and a boolean
role is nonsense. `_LEEWAY_SECONDS = 30` absorbs clock drift without meaningfully extending a
one-hour token.

**The role gate** is `role <= 4` (`UNRESTRICTED_CONTRIBUTOR`), matching the platform's own backwards
scale where lower is more privileged. This **excludes** an ordinary `TRUSTED_USER` (8): the agent
spends LLM budget and runs generated code, so access starts narrow and widens by raising one
constant, `DEFAULT_MIN_ROLE`. `AGENT_MIN_ROLE` can override it, and a non-numeric value raises rather
than guessing in either direction.

**Identity is carried in a `ContextVar`**, for the same reason the file store uses one: threading a
`User` through every call site would touch every tool. The module carries an explicit warning that a
`ContextVar` does not cross threads — the bug that already made the file-store session stamp come out
`None` once. The streaming path is safe because `graph_runtime` already does
`contextvars.copy_context()` + `ctx.run()` around its worker; identity rides along on the mechanism
that fix installed.

**Service callers keep working.** The eval harness has no browser, presents the API key, gets no user
identity, and falls back to session scoping. The rule that must not be broken: "no identity" resolves
to **session** scoping, never to one shared owner every anonymous caller lands in and can read each
other's files through. `_extract_user_token()` accepts `Authorization: Bearer` only when the value is
structurally a JWT (`value.count(".") == 2`), because `Bearer` is also how a service caller presents
the API key, and treating that key as a token turns a valid service request into a confusing 403
about signatures.

`PyJWT>=2.8` is pinned in `requirements.txt` although it already arrives transitively: identity must
not break because something else drops its dependency.

### 4.2 Verifying without holding the signing secret (`95a48de`)

**This reverses a decision the plan document records as settled**, and the reason it does is worth
keeping. `docs/jwt-user-scoping-plan.md` states that introspection "was considered and rejected: its
only real advantage is not holding the secret, which is already moot." That reasoning is correct for
the dev tier and **inverts against production**: this host runs LLM-generated code in a sandbox with
a Docker socket, and a sandbox escape that found the *production* HS256 secret would be equivalent to
minting tokens for every account on the platform.

**Structural.** `identity.py` gains a second verification path selected by
`AGENT_TOKEN_VERIFY=local|introspect`, behind one new entry point `identify(token)` that all callers
use. `introspect_token()` forwards the received cookie to the backend's own `/api/check-tokens`,
which reads it with its own secret and its own cookie name and answers `{id, role}`. Nothing secret
lives here — and the production cookie name stops mattering too. `_require_user` was changed from
`decode_token` to `identify` in the same commit. Default stays `local`, so nothing changes for the
dev tier.

**Fails CLOSED in every direction that matters.** An unreachable backend, an unexpected status, a
non-JSON reply and an unset URL all raise `IdentityNotConfigured` rather than returning `None` —
"cannot verify" must never resolve to "nobody is signed in", which would silently degrade a token
deployment to anonymous access. A reply with no `id`, or no usable `role`, is refused rather than
defaulted.

**The cache collapses a burst and nothing more.** `_INTROSPECT_TTL_SECONDS = 60`,
`_INTROSPECT_MAX_ENTRIES = 512`, `_INTROSPECT_TIMEOUT = 8`. Failures are never cached — a cached
rejection keeps refusing someone who has since signed in again. Eviction drops the soonest-to-expire
rather than an arbitrary entry, so a burst of new tokens cannot evict the ones still in use. Keys are
`sha256` of the token, never the token: this dict is exactly the thing that ends up in a heap dump or
a debug print.

### 4.3 A signed-in visitor is not also required to hold the API key (`c59a69d`)

**The measurement.** Found live: sign in at the platform, then get "You are not signed in".

**The bug, and why it was structural.** Token mode ran the API-key gate **before** identity, so a
visitor with a valid JWT and no key was refused by the key check before anything asked who they
were — and token mode hides the settings panel precisely because there is nothing to paste, so they
could not supply one. In an incognito window, where `localStorage` starts empty, that is every user.

**Revised ordering.** `_require_user()` now runs first on `/agent/chat` and `/agent/chat/stream`, and
`_require_agent_chat_api_key(user)` returns immediately when a user was identified. A verified user
*is* a credential, and the stronger one: the key says only "someone who has the key", the JWT says
who. The two are **alternatives**, never a pair; the key remains how a caller with no browser gets
in.

**New endpoint: `GET /agent/whoami`.** Reports mode, verify path, the caller, their role, the
required role, the platform tier, the resolved check-tokens URL, and — when it thinks nobody is
signed in — why. It reports the cookie **names** it received and the one it expects, never values, so
a tier using an unexpected cookie name is something the server states rather than something we guess
at. **Always 200**, including for anonymous and refused callers: an endpoint that exists to explain a
refusal cannot answer with one.

The client half of this fix is in Stage 6.2.

---

## Stage 5 — Records gain an owner

Commits `d934cbe`, `e067055`, `1f880d3`.

`owner_id` is a **second axis, not a replacement** for the conversation stamp. A user has many
conversations, and in dev/demo there is no user at all — outside token mode `owner_id` is `None` on
every record and scoping stays exactly the per-conversation behaviour it is today.

### 5.1 Files (`d934cbe`)

**The hole.** `GET /agent/files/<id>/download` had never checked anything. Every answer publishes
file ids as download links, so each one was effectively a permanent public URL: any id, no
credential, any file.

**Structural.** `agent_runtime/file_store.py` gains `current_owner()` (reading the identity
`ContextVar` rather than a second one of its own, so there is one place a caller is established and
one place it can be wrong), `record_owner()`, and `may_read(record, *, allow_unowned=True)`.
`owner_id` is stamped in `save_uploaded_file`, `create_output_file` and
`create_output_file_from_path`. `find_files` filters through `may_read`.

**A mismatch answers 404, not 403.** A 403 confirms the id exists, which turns the endpoint into an
oracle for enumerating other people's files. The same rule is used for conversations.

**`allow_unowned` is a parameter rather than a decision**, because the two callers genuinely want
different answers for the 1,325 records that predate ownership:

| caller | `allow_unowned` | why |
|---|---|---|
| `find_files` (server-side reuse) | `True` | those files were written by a deployment that identified nobody; hiding them breaks the reuse the lookup exists for |
| `download_agent_file` (browser) | `not _token_strict()` | once strict, a file nobody owns is a file nobody downloads — that is the exposure being closed |

Note this is a **refinement of the plan**, which said flatly "deny unowned rather than defaulting them
public".

**There is no honest backfill.** Ownership did not exist when those records were written and their
conversation stamp was never mapped to a user, so no rule can attribute them after the fact; inventing
an owner would be worse than leaving them unowned. `scripts/file_ownership_report.py` prints the size
of the problem instead — totals, per-owner counts, recent-window counts, and optionally the most
recent unowned filenames — so `AGENT_TOKEN_STRICT=1` gets flipped on evidence rather than on hope.

### 5.2 Conversations (`e067055`)

**The hole.** `get_or_create_memory(memory_id)` fetched by bare UUID with no owner check: anyone
holding an id read that transcript. Worse, the *create* half would have indexed over a document it had
just refused to read, replacing the owner's conversation with an empty one.

**Ownership is asserted at the edge.** `rag_pipeline/memory_module.py` gains
`assert_owner(memory_id, *, allow_unowned=None)`, called from `api/server.py` on `/agent/chat`,
`/agent/chat/stream` and both `/agent/conversations/<id>` methods — rather than guarded inside each
read and write. The reason is that specific get-or-create failure: refusing at the door removes the
whole class of mistake instead of patching each door. On `/agent/chat` the assertion is placed
*inside* the identity binding, because `assert_owner` reads the caller from the same `ContextVar`
everything else does and before that line would see nobody and wave every conversation through. On
the streaming path the refusal travels as an SSE `error` frame, since headers are long gone by then.

**A write attributes a conversation only when it has no owner yet**, so a write can never move one
between users — the second half of the guarantee `assert_owner` makes at the door, not a repeat of
it. `update_memory` also stamps `updatedAt`, without which every conversation sorts equal and the
user's list is arbitrary.

**New endpoint: `GET /agent/conversations`.** Returns summaries, never transcripts, via
`list_memories()` — a `term: {owner_id}` query sorted by `updatedAt` desc. The summary is built from
**named keys** rather than spread from `_source`: `_source` in a query is a request, not a guarantee,
and the field it must never leak is `chat_history`, the whole transcript. A test caught exactly that.
Outside token mode the endpoint returns `{"conversations": []}` with 200 rather than an error — a
client that shows a history pane should render it empty, not break.

### 5.3 The conversation, not a transcript of it (`1f880d3`)

**The reason.** `chat_history` is the AGENT's memory: what was asked and answered, used to give the
next turn context. The user's conversation also has the layers on the map, every file uploaded across
the session, a region and a model. Restored from `chat_history` alone it comes back as a text shell
whose answers say "you can see these features on the map" beside an empty map — the transcript lies.

**Structural, and the shape was not invented here.** The client already models it correctly, and
`map-ui-prototype/src/sessionStore.ts` was written server-shaped on purpose ("moving it behind a
per-user endpoint later means swapping the transport, not the record"). So the server **stores that
record** rather than rebuilding it from tool output — which would duplicate the client's
layer-descriptor rules (which geometry is small enough to inline, which layer re-fetches by url) in a
second place where they would drift.

`save_session_snapshot` / `get_session_snapshot` land on `memory_module`, behind
`GET|PUT /agent/conversations/<memory_id>`.

**Client-supplied, therefore treated as data.** `_SNAPSHOT_RESERVED = {owner_id, chat_history,
createdAt, updatedAt, _id}` is stripped before storage: a snapshot cannot claim an `owner_id` and
cannot rewrite `chat_history`. Both have tests, because "hand yourself someone else's conversation by
saying you own it" is the obvious attack. Ownership is never re-derived from the snapshot — it is
asserted by the caller before the write, and the document's own `owner_id` is left untouched.

**Oversize is refused (413), not truncated.** Default 5 MB. A conversation that came back missing half
its layers would look like data loss with no explanation.

`messageCount` / `layerCount` / `fileCount` are written onto the **document**, not into the snapshot,
so a history list can say "12 messages, 3 layers" without fetching twelve messages to count them.

**Two bugs the tests caught, both worth recording:**

- the size cap was a module-level constant frozen at import, so no deployment could change it without
  a restart. It is now `_snapshot_max_bytes()`, read at call time — a limit that needs a restart to
  change is a limit nobody adjusts when a real conversation turns out to sit just over it;
- the test double's `update()` quietly created missing documents, hiding the create path entirely.
  Real OpenSearch raises `NotFoundError` there. `FakeOpenSearch.update` now raises too.

---

## Stage 6 — The browser stops owning the session

Commits `84c60b2`, `4daf2e1`. `map-ui-prototype/src/`.

### 6.1 Refresh once, and say what a refusal means (`84c60b2`)

**Who refreshes: the browser, not the agent.** The agent stays a pure verifier. The client calls the
platform's own refresh endpoint, which validates the refresh cookie and re-mints the `.i-guide.io`
access cookie. Nothing in the client ever sees a token value — both cookies are httpOnly, which is
also why none of this reads `document.cookie`.

**Structural.** New `auth.ts` (`AuthError`, `AuthReason`, `authErrorFrom`, `authMessage`,
`refreshAccessToken`, `withTokenRetry`) and new `auth.check.ts`, wired as `npm run check:auth`.
`agentClient.ts` routes `streamChat`, `uploadFiles`, `listConversations`, `putConversation` and
`getConversation` through `withTokenRetry`, and every agent call now sends
`credentials: 'include'` — same-origin would send the cookie anyway, but a developer running `npm run
dev` against the deployed API is cross-origin, where the default omits it and every request looks
unauthenticated for no visible reason.

**Every retry is bounded at exactly one attempt, and only an expired token retries at all.**
`not_signed_in`, `token_invalid`, `insufficient_role` and `not_your_conversation` are rethrown
untouched: no number of refreshes fixes any of them, and retrying is how a loop against the auth
backend starts. A failed refresh is a sign-in, not another try. `run` is a thunk rather than a
Promise because a retry has to *issue* a new request.

**`check:auth` counts the calls, because none of this is visible in a screenshot — and it caught a
real bug inside the same commit.** The shared in-flight refresh was cleared on a timer, so a 401
arriving after a refresh had finished reused that stale result. A stale `true` is the bad one: it
retries against a cookie that was never actually replaced. It is now cleared the moment the refresh
settles, which still lets a burst of four parallel 401s share a single refresh rather than
stampeding the auth backend.

**Message rewrite.** An auth refusal is not a failed request and no longer reads like one.
`App.tsx`'s catch previously rendered `Request failed: ${e.message}` for everything; an `AuthError`
now renders `authMessage(e)`. Since the role gate starts at contributor, `insufficient_role` is what
most platform accounts will actually hit, so that message names the role required and says signing in
again will not help:

> Your I-GUIDE account is signed in, but does not have access to the agent — it needs the contributor
> role (4) or above […] Signing in again will not change this; ask an I-GUIDE administrator for
> access.

`refresh_url` and `signin_url` come from `/agent/ui-config` rather than the bundle, so one build runs
against either tier. (Superseded by `PLATFORM_TIER` on the server side — Stage 3.2.)

### 6.2 The server owns the history in token mode (`4daf2e1`)

**The measurement.** Two things observed live with the same root cause: signing out left the history
list fully populated, and a signed-out visitor was told "Forbidden: invalid API key" instead of being
asked to sign in.

**Revised during the work: a local filter became a server-owned store.** IndexedDB is
per-**origin**, not per-user. Signing out of the platform does not touch it, so the next person to
open the page gets the previous person's conversations. The first fix — filtering the local store by
owner — *hides* that, and it is the wrong fix: conversations belong to the user, and the whole point
of per-user history is that it follows them to another browser. A local filter cannot do that.

So in token mode `GET`/`PUT /agent/conversations` is the source of truth and IndexedDB is a cache;
`dev` and `demo` identify nobody and keep using it alone, exactly as before.

**The filter is kept anyway**, for a narrower reason than it was first written for: a cache with
someone else's data in it should not serve it. `StoredSession` gains `ownerId`, and
`listSessions(viewer)` / `loadSession(id, viewer)` take the viewer. The rule is deliberately
asymmetric — with no viewer everything is listed (dev and demo unchanged); with a viewer, their own
records **and unowned ones**, so a conversation started before signing in is not orphaned by signing
in. Another user's records are hidden, never deleted: that is their data and their browser too.

`App.tsx` gains `refreshSessions()` as the single path to the history list, plus `tokenMode`,
`viewer` and `viewerRef` state. `viewer` comes from `GET /agent/whoami` — the access cookie is
httpOnly, so the page cannot answer "who am I" itself and has to ask. `restoreSession` tries the local
cache first (a hit avoids a round trip) and falls back to `getConversation` for a conversation opened
on another browser.

**A failed fetch returns `null`, not `[]`.** "Could not ask" and "you have none" are different, and
showing an empty history because the server blinked reads as data loss; `refreshSessions` keeps
whatever is on screen.

**`AGENT_TOKEN_STRICT=0` no longer swallows "who are you".** As written in `69863b8`, `_require_user`
caught every `IdentityError` and returned `None` when non-strict. That dropped a signed-out visitor
through to the API-key gate, to be refused for lacking a credential token mode gives them no way to
enter — the second half of the live bug fixed in Stage 4.3. The flag relaxes **ownership** of records
written before ownership existed, which is what a backfill needs; it never needed to relax sign-in.
The clause is now `if not _token_strict() and _service_key_presented()`: a caller holding the service
key is still the exception, because that is a real credential belonging to something without a
browser.

`TopNav` hides the connection settings in token mode as well as demo — for a different reason, which
the code says out loud: there *is* a credential, it is just not one you paste.

---

## Changes that were revised mid-work

| what was tried first | what it became | where |
|---|---|---|
| expose `map_layer_delivered` to the decider | expose the whole `this_turn` ledger; keep the boolean as an unmissable signal for the one question the prompt asks directly | 1.3 |
| filter the local IndexedDB history by `ownerId` | make the server the source of truth in token mode; keep the filter, but only so a cache does not serve another user's data | 6.2 |
| `AGENT_TOKEN_STRICT=0` relaxes all identity errors | it relaxes record ownership only; a service key is the sole exception for sign-in | 6.2 |
| API-key gate, then identity | identity first; a verified user satisfies the credential check on its own | 4.3 |
| `refresh_url`, `signin_url`, `check_tokens_url` set independently | `PLATFORM_TIER` resolves all three, explicit URLs still win | 3.2 |
| local HS256 verification only (and the plan recorded introspection as rejected) | `AGENT_TOKEN_VERIFY=local\|introspect`; local stays the default, introspect exists for production | 4.2 |
| in-flight refresh cleared on a timer | cleared the moment it settles, so a later 401 does not reuse a stale `true` — caught by `npm run check:auth`, added in the same commit | 6.1 |

---

## Deployment state

| | deployed to the dev VM? |
|---|---|
| `claude/evidence-summary` through `8fe0dcb` (`1044afb`, `99d71ad`, `ae862b5`, `5a10a58`, `8fe0dcb`) | **yes** |
| `claude/evidence-summary` `25c3e1e` (the `this_turn` ledger) | **no** |
| `claude/jwt-identity`, all 10 commits | **no** |

So on the running dev deployment: the evidence summary, the generated capability inventory, the null-
argument wrapper, GeoTIFF routing and the `admin_boundary` recovery are live. The decider is **not**
yet reading this turn's ledger, and none of the identity, ownership or mode work is running — the
deployed server is still configured with the legacy `DEMO_MODE=true`, which `deployment_mode`
deliberately continues to resolve to `demo`.

**Neither branch is merged**, into `prototype` or into each other.

---

## Not verified, and worth knowing before either branch merges

- **`/agent/files/upload` never binds identity.** `upload_agent_files` in `api/server.py` does not
  call `_require_user()`, so `save_uploaded_file`'s `owner_id: current_owner()` always reads `None`
  for uploads. Generated outputs get an owner (they are created inside a chat request, which does
  bind it); uploads do not. That makes `AGENT_TOKEN_STRICT=1` unreachable in practice for uploaded
  files, since they would all be unowned and undownloadable. Not addressed in any of the ten commits.
- **`agent_runtime/session_memory.py` was not given an owner key**, although the plan's Step 3 lists
  it. It is a process-local cache rather than the source of truth, so it does not enforce the
  boundary — but the plan item is open, not done.
- **The plan document was not updated** for the two decisions the branch reversed or refined:
  introspection (recorded as "considered and rejected"; now `AGENT_TOKEN_VERIFY=introspect` exists)
  and unowned records (recorded as "deny unowned"; now `may_read(allow_unowned=...)` lets the two
  callers differ).
- **The backend CORS prerequisite is still someone else's deploy.** The plan states
  `https://agent.i-guide.io` is not in `ALLOWED_DOMAIN_LIST`, so the credentialed refresh XHR in
  `auth.ts` will be blocked by the browser until it is added. Nothing in these commits could change
  that, and `refreshAccessToken` returning `false` on a CORS block is handled — it degrades to a
  sign-in prompt — but the refresh path is untested against the real backend.
- **Production `JWT_ACCESS_TOKEN_NAME` / `JWT_ACCESS_TOKEN_SECRET` are unknown.**
  `platform_endpoints.consistency_warning()` deliberately does not guess prod's cookie name.
- The `1,325` legacy-file figure in `file_store.may_read`'s docstring is carried over from
  pre-existing comments in the same file, not re-measured in this branch.
- **Section 2.4's account of `5a10a58` is superseded by a 17th commit.** `5a10a58` recovered
  only the first inversion — `area` holding a level word, with the place sitting in `name`. A
  re-run of the sweep caught the second and more natural one:

      admin_boundary({'state':'IL','level':'county','name':'Champaign','subdivide':'tracts'})
      ValidationError: area — Field required

  `area` omitted, place in `name`. This dies inside pydantic before any in-body recovery can see
  it, so no amount of the earlier fix could have caught it. A parameter a model reaches for twice
  is not carelessness: `name` was simply the wrong word for a filename. So `area` is no longer
  required, `name` is now a second spelling of the PLACE, and the output filename stem moves to a
  new `output_name`. A caller passing both `area` and `name` keeps the old meaning, so nothing
  that worked before changes.
