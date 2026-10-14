# Design review, October 2026: why the agent keeps getting correct answers wrong

*2026-10-08. Written for the eight-flaws program (architecture stages 41+). Line numbers are
against `prototype` at `4b066f6` (after #87 and #88). `G` is
`agent_runtime/supervisor/graph.py` (6,229 lines), `E` is `agent_runtime/executor_factory.py`,
`SV` is `agent_runtime/sandbox_verify.py`, and `EQ` is `agent_runtime/evidence_quality.py`.
Incident references are to `docs/agent-architecture-changes.md` (`AC`), `docs/DEVLOG.md` (`DL`)
and `AGENTS.md` (`AG`). "Inferred" marks a conclusion read from the code and not reproduced.*

## The short version

The GIS work is mostly right. What goes wrong is everything that happens around it. The
supervisor, the verification layers and the answer writer all need to know **what this turn did
and what each number means**, and **none of them has a record that says so**:

- A tool returns a JSON string with no shared shape (§2.1).
- The sandbox returns a bare number plus a free-text unit word (`SV:477-481`).
- The record of "what happened" exists in seven partial copies, each filtered differently and
  read by a different consumer (§2.3).
- The answer is written from prose cut to 2,000 characters (`G:5499`), audited by a second model
  call against more of the record than the writer saw, and then rewritten by 13 post-processing
  steps that cannot see one another's verdicts (§F7).

So every layer reconstructs meaning by guessing: from unit vocabularies, column names, verb
lists and substring matches. A guess that fails on a new query is patched by adding the new word
to the list.

That is the patch loop the brief describes, and the history confirms it:

| | count |
|---|---|
| incidents recorded since April 2026 | **106** |
| fixed by a list entry or special-case rule | **24**, plus 4 mixed |
| still open | **9** |
| a fix that caused or masked a later failure | **12** |

(Catalogue: §4.) The two 2026-10-08 turns are not unusual. They are the typical case: a correct
answer, judged by layers that had to guess.

**The general fix is one change of representation, applied in four steps:**

1. **A per-turn append-only event log** becomes the only record of what happened (phase 2).
2. **Every number carries `{value, unit, measured_in_crs, source}`** from the tool that made it
   (phase 3).
3. **The answer is written from facts with ids** and checked deterministically against them,
   with one verdict policy (phase 4).
4. **Every source declares its coverage** and travels with what it produced (phase 5).

Each step replaces inference with a recorded fact. That is why a step removes a whole class of
failure instead of one instance of it, and why the outcome stops depending on which words a
particular model chooses.

The review changes the program in five places (§5). The phase order stays. Phase 2 gains
supersession, task pinning and the needs bypass, phase 4 gains the auditor's failure modes, and
the new tool result schema starts in phase 2, not phase 3.

---

## 1. How a turn runs today

```
request ─► triage ─► supervisor ──decide()──► search | analyze | code ─► supervisor ... ─► synthesize
                         ▲  (LLM, sees _distill)        (each a LangGraph agent                 │
                         │                              on its own child thread)                │
                         └────────────── reground (≤1) ◄──────────────────────────────────────────┤
                                                                                                 ▼
  synthesize = LLM answer ─► LLM audit ─► reconcile (rules 1,4,3,2,2′) ─► cut unproducible
            ─► cut travel ─► drop draft mentions ─► [maybe reground] ─► grounding caveat
            ─► image embeds ─► link sanitiser ─► artifact corrections ─► Source: line
            ─► Partial-answer note                                   (G:5998–6118, 5902–5912)
```

- **The decider** sees a distilled JSON summary (`_distill`, `G:1699`). It contains peer
  summaries cut to 220 characters (`G:1724`), `pending_needs` without their reasons (`G:1754`),
  and ledger lines trimmed to 3 rows per tool, 25 rows and 6,000 characters (`G:424`, `432–469`).
- **Each peer** is a `create_agent` loop on a child thread `{thread}::{label}` (`E:862`). Its
  only stop is a step count: `recursion_limit` 60 (`E:1034`), plus #88's repeat middleware.
- **The synthesizer** gets prose: `json.dumps(analysis_results)[:2000]` (`G:5499`) and the code
  peer's own prose answer `[:2000]` (`G:5507`). It does not get this turn's ledger lines; only
  earlier turns' (`G:5997`).
- **The auditor** gets this turn's ledger lines first, then raw results compacted to 8,000
  characters each (`EQ:533–598`). **The auditor sees more of the record than the writer it is
  auditing.**

---

## 2. The root causes, by flaw

Each entry: what the brief said, the mechanism in the code, why it fails on queries and models it
was not written for, what the brief got wrong or missed, and the change.

### F1. Verification guesses what a number is

**Mechanism.** Code declares a result as `IGUIDE_OUTPUTS = {name: {value, unit, min?, max?}}`
(`SV:477–481`). The declaration has no CRS, no source frame, no lineage and no link to an
answer sentence. The unit is free text. At check time the gate holds a bare number and a word,
and has to infer three things from vocabularies and names:

- **Is the word a unit?** `_UNIT_ALIASES`, 113 keys mapping to 14 canonical units, 39 of the keys
  count nouns (`SV:95`). `_KNOWN_UNITS` is a second vocabulary of 25 (`SV:354`).
- **Is the number a count?** `_inferred_count` matches a name pattern,
  `(?:^|_)(?:num|n|nb|count|counts|number|total)(?:_|$)` (`SV:383–398`).
- **Was it measured in metres?** `_METRIC_COLUMN_HINTS` matches 12 substrings in column names
  (`SV:165`). `_is_projected` falls back to `"4326" in text` (`SV:77`). `measured_in` credits
  any projected frame in scope that has a column of the same name (`SV:1036–1053`): a shared
  name is taken as lineage.

A recognised unit word earns PASS "unit {unit}" (`SV:552`). That means only that the word is
known. Nothing checks that the value is in that unit.

**Why it fails on unseen inputs.** Every model has its own vocabulary, and every query counts
something new. The lists can only be completed by meeting each word in a live failure first:

- `km²` (`DL2531`), `records` (`DL3795`), `points` (2026-10-08, `AC6189`), `square_miles`
  (`AC6318`) and `schools` (`AC6470`) were each found that way. `schools` broke stage 36's new
  noun list within hours.
- The two vocabularies already disagree on 9 tokens. Plural `kilometres` is in `_KNOWN_UNITS`
  but not in `_UNIT_ALIASES` (`SV:98`). A run declaring `kilometres` therefore gets UNKNOWN,
  "could not be compared" (`SV:659–663`; inferred).
- The name heuristics also fire the wrong way, both inferred:
  - `"_m"` matches `pop_male` and `"dist"` matches `district_id`. Either makes a WGS84 frame
    "hold a measurement" and blocks the reprojection rescue (`SV:1008`).
  - `total_distance: {value: 12, unit: "leagues"}` passes as a count.

**What the brief missed.**

- **The gate's PASS findings never reach the agent side.** `_read_checks` drops them
  (`code_execution.py:455–457`). So does the `measured_in` evidence that PR #85 added, and
  nothing downstream can use it.
- **`declared_outputs.json` is never compared to the answer.** It goes into the provenance
  manifest only (`artifacts.py:243–249`). The one place numbers are declared with units is
  disconnected from the one place numbers are shown to the user.
- **A gate failure can never be superseded.**
  - `_merge_peer_result` accumulates `tool_results` across runs (`G:1631–1661`).
  - `_gate_failures` collects every `verification` in the context, whatever its order
    (`G:2346–2398`).
  - So a run that failed, was fixed and re-ran correctly still marks the turn as failed, even
    though the tool's own message tells the model to "FIX AND RE-RUN". This is inferred; no
    supersession logic exists. PR #85 made accumulation correct for the record but turned this
    into a latent false alarm.

**Change (phase 3).**

- **Typed outputs.** Every number a tool or script emits is
  `{value, unit, measured_in_crs, source, op}`, built where the number is made, while the
  code still knows it.
- **Real unit parsing.** Units go through `pint`: any unit parses or fails loudly, and a
  dimension check replaces vocabulary membership.
- **Measurement frames as metadata.** The sandbox prologue already has the operation tracker
  (stage 26). It stamps each metric operation's CRS onto its result, so the frame a number is
  stored on stops being evidence.
- **Deletions.** `_UNIT_ALIASES`, `_KNOWN_UNITS`, `_METRIC_COLUMN_HINTS`, `_inferred_count`
  and the twin-frame credit all go.

**Invariant.** A number either carries its unit and measurement frame, or is marked untyped. A
check reads metadata; it never infers from a name.

**Would have prevented:**
- the unit false alarms A7, A8, A11, A13 and A14;
- the twin-frame rule A15;
- the `df`-named-frame false unknowns A5 (`DL2163`);
- the `kilometres` and `leagues` holes above, which have not happened yet.

**Cost.**
- One dependency, `pint`, which is pure Python.
- Each tool's result builder changes: about 40 tools, most through a shared helper (§2.1).
- The sandbox prologue grows by about 30 lines.
- No extra model calls.

**Does not fix.** A number the model types by hand into a script with a wrong unit. `pint`
parses "m" happily for a value that is really in feet. The operation tracker still catches the
common case, because the CRS of the operation is recorded independently of the declaration.

### F2. The answer is written as prose and audited as prose

**Mechanism.**
- **The answer is one LLM call over truncated prose** (§1). Numbers come out as free text with
  no handle back to the tool output that produced them.
- **The audit is a second LLM call** that writes a `claim_ledger` with a "VERBATIM span" per
  claim (`EQ:272–365`). **No code checks that the quoted span exists.** `evidence_quote` is
  read back only at `EQ:727`.
- **`_recompute_verdict` replaces each reason with a generic string** (`EQ:729`). That makes
  reconciliation rule (3), "the reason concedes grounding", effectively dead whenever a ledger
  exists (inferred).
- **Reconciliation then re-adjudicates the LLM's verdict with string rules** (`G:2602–2735`):
  - **Rule (1)** drops an issue if any image exists and the claim contains "has been",
    "available" or "produc" (`G:2292`).
  - **Rule (4)** drops any claim mentioning "on the map" once a layer was delivered, whatever the
    number in it.
  - **Rule (2)** matches only numbers of 3+ digits (`G:2311–2315`). `21.5` yields nothing, and
    `997.93` yields `997`.
- **Patches cover named claim classes:**
  - `_remove_unproducible_claims` has one entry, routing (`G:1008–1013`).
  - `_remove_unrecorded_travel` covers travel words plus a quantity (`G:1405–1444`).
  - `_correct_artifact_claims` covers three fixed cases (`G:2237–2286`).
  - `_with_feature_source` fires only for 3+ list rows from two tool families (`G:2068–2087`).

**Why it fails on unseen inputs.** Claim classes are open-ended: memorised areas ("~998 sq mi",
open in 3 of 4 replays, `AC6749`), populations, dates, speeds, rankings. Each rule covers the
class it was written for, and the next class ships until someone writes a rule. PR #87's own
"not fixed" section names the next one.

The LLM audit is also a sampled judgement:
- The code's own comments record verdicts flipping: "3 out of 3" (`EQ:294`), "5/5 → 4/5"
  (`EQ:657`).
- The re-grounding pass is a second, independent draw, which is how the 10-08 auditor came to
  contradict the answerer.

**Also wrong in the current design.**
- **The answerer and the auditor share premises.**
  - Same model (`G:5550`, `G:6019`).
  - Same ledger text.
  - The auditor reads the peer's own summary as grounding. `_without_peer_prose` removes it for
    rule (2) only, not from the LLM's view (`G:2318`).
  - A memorised figure in a peer summary can therefore be "supported" by quoting that summary.
    That is the exact mechanism of S39 (`AC6679`), and it is closed for one rule only.
- **Some answers are never audited.** Conversational answers with history and no grounding are
  not audited (`G:6012–6020`), and the general-knowledge branch skips every check
  (`G:5933–5948`).

**Change (phase 4).**
- **Facts with ids.** Synthesis writes from a fact set derived from the event log. Each fact is
  a typed value with an id, `{id, value, unit, crs, source, tool_call_id, op}`, and the model is
  asked to cite the fact id for each number it writes.
- **A deterministic number scan.**
  - Every number in the answer must resolve to a fact: by id, or by value within the fact's
    unit, converted with `pint`.
  - An unresolved number is a finding. It is not a regex class.
  - The routing scan (#87) and `_remove_unproducible_claims` (#85) become instances of this one
    pass and are deleted.
- **A demoted LLM audit.** It remains only for qualitative claims (interpretation, causal
  statements), where no deterministic check exists. It never decides about numbers.

**Invariant.** Every number the user sees is either traceable to a recorded fact or marked as
unverified. That holds whatever the claim is about, so no new claim class needs a new rule.

**Would have prevented:**
- the memorised figures B16, B18 and B19 (the open "~998 sq mi");
- B5, a roll-up sentence flagged "high" on a grounded answer;
- B13 and B9, a truncated record read as absence;
- rule (2)'s 3-digit blind spot;
- A9, a wrong population that went undetected. The population would be a fact with a source,
  and the answer's number would be checked against it.

**Cost.**
- The synthesis prompt carries a compact fact table, typically smaller than today's 2,000+2,000
  characters of prose dumps.
- The number scan is local and costs no tokens.
- The LLM audit gets shorter: qualitative claims only.
- Migration: the five reconciliation rules and three removal passes are deleted. Their tests
  become fact-scan tests.

**Does not fix.** A correct number attached to the wrong noun ("31 hospitals" when fact 7 is 31
schools). The scan resolves the value, not the meaning. Fact ids in the answer narrow this,
because the scan can check the noun against the fact's label, but it stays heuristic.

### F3. State is overwritten, not appended

**Mechanism.** `SupervisorState` is a plain `TypedDict` with no reducers (`G:104–134`). The code
says so itself at `G:5759`. Accumulation is hand-written per node and per key. PR #85's
`_merge_peer_result` fixed `analysis_results` and `code_result` (`G:1631–1660`), but:

- Everything else is last-write-wins: `answer`, `audit`, `distilled`, `needs`,
  `evidence_summary` and `grounding_gaps` (table in §2.3).
- **On a re-ground, the draft answer and its audit are never stored.** Only up to 6 claim strings
  go forward (`G:1485`).
- `tool_failures` from an earlier run survive a later success of the same tool. That
  contradicts the "currently failing" meaning that `_repeatedly_failed_tools` gives them
  (`G:4331`; inferred).
- A stale `execution_error` can survive beside `executed=True` (`G:5000`; inferred).

**The brief mis-states the flaw: the deeper problem is not overwriting, it is that there is no
single record.** "What happened this turn" exists in seven places, and each consumer reads a
different one:

| # | record | who reads it |
|---|---|---|
| 1 | `analysis_results` / `code_result` raw `tool_results` (JSON strings) | synthesizer (cut to 2,000 characters), gate walk |
| 2 | `action_rows` (search-node rows only) | ledger writer |
| 3 | `session_memory._ACTIONS` ledger: curated, values cut to 80 characters, written **only** at the end of synthesize (`G:6121`). A turn that raises records nothing | next turn's decider, peers, synthesizer, auditor |
| 4 | peer thread messages in the checkpointer: LRU 500, re-sliced by `PeerSession` | the peer itself |
| 5 | `actions` (decisions, not outcomes) | supervisor |
| 6 | `peer_failures` + `tool_failures` | banners, dead-end detection |
| 7 | trace events | the client, `chat_traces` |

The ledger extractor keeps only 46 argument keys and 33 result keys (`G:360–407`), with 10
per-tool special cases (`G:592–672`). What falls outside those lists is lost, including:
- `execute_code`'s stdout and declared outputs;
- `buffer_layer`'s `total_area_km2` and the CRS it measured in;
- `inundation_at_level`'s `flooded_km2`;
- `geocode_places`'s coordinates;
- `overpass_search`'s failures (no `ok` key, so they render as "0 documents").

**Why it fails on unseen inputs.** Every new tool, result key or node has to be added to the
right lists and the right merge, or it silently vanishes from some consumer's view. Every
consumer disagreement in the history comes from two consumers reading different records:

| incident | what happened | ref |
|---|---|---|
| C3 | `has_evidence=False` on turn 2 | `AG473` |
| C5 | ledger absent at 8+ history items | `AC895` |
| C6 | good `file_id` written on the failed row | `AC896` |
| C8 | the supervisor re-read its own conclusion | `AC910` |
| F4 / F5 | the auditor accused a ledger-correct answer | `AG491`, `AC895` |
| 10-08 | GEOID 17019 flagged as unsupported | `AC6288` |

**Change (phase 2).** One **append-only event log per turn**:

```
{seq, kind: tool_call|tool_result|fact|artifact|plan_update|failure|decision,
 peer, tool, args_key, ok, payload, supersedes?}
```

- **Writers:** the event log is written by the tool layer (a middleware every peer already has,
  `E:_default_middleware`), not by each node.
- **Derived views:** the ledger, the decider's distillation, synthesis inputs, the audit's
  execution context, the banners and the client's `action_rows` all become pure functions of
  the log. The log is checkpointed with the run.
- **Persistence:** the ledger for the next turn is the log's facts, appended as each fact
  arrives. It is no longer extracted at the end, so a turn that raises still keeps what it did.
- **Supersession is explicit.** A re-run that produces the same output name supersedes the
  earlier fact or failure. This closes the latent "fixed run still flagged" issue above.

**Invariant.** There is one record. A consumer can be given less of it, but never a different
version of it. Nothing written to it is ever lost or rewritten.

**Would have prevented:** C3, C5, C6, C7, C8, C14, F4 and F5, the gate-failure supersession
issue, and the "turn that raises records nothing" gap.

**Cost.**
- Migration is the largest of any phase: `_merge_peer_result`, `action_rows`, the ledger
  extractor and `_distill` all move onto the log.
- Runtime cost is negligible: in-memory, checkpointed with the run.

**Does not fix.** It does not decide which facts matter; it only makes them available. The
fact extraction itself is phase 3's typed outputs. Until then, a fact is whatever the tool's JSON
says.

### F4. Loops are bounded by counts, not progress

**Mechanism.** Every bound in the system counts something other than progress:

| bound | where | counts |
|---|---|---|
| `max_steps` = 8 | `G:54` | supervisor invocations |
| `_max_peer_runs` = 3 | `G:65` | runs of a peer |
| `_max_searches` = 2 | `G:57` | searches |
| `recursion_limit` = 60 | `E:1034` | peer super-steps |
| `_peer_error_budget` = 2 | `G:1864` | failures |
| `_REPEAT_STOP_AT` = 4 | `E:1761` | identical asks |
| `_TOOL_FAIL_REPEATS` = 2 | `G:4327` | `ok:false` per tool since its last success |
| `_is_unproductive_repeat` | `G:2134` | same peer back-to-back |

Only `search_empty_streak` looks like progress ("did a search add a new document?"). In unified
mode it resets on any harvest (`G:5804`).

**Why it fails on unseen inputs.** Progress failures take many shapes, and a count catches only
the shape it counts. The code admits all of the following (all inferred):

- **Supervisor A-B-A-B** (analyze, code, analyze, code) never repeats back-to-back. It stops only
  at 3 runs per peer or 8 steps. A `needs`-driven route skips the repeat check entirely
  (`G:5582–5587`).
- **Tool A-B-A-B:** #88's middleware compares only against the latest executed step
  (`E:1810–1841`), so it never matches.
- **Repeated failing calls are never memoised**, by design (`E:1858`). The same failing call
  runs until `recursion_limit`.
- **Varied-but-useless calls** each get a new key. Successes reset the dead-end counter, and
  tools without an `ok` key are never counted (`G:4341`). Only the 60-step recursion limit stops
  them, and that kills the peer: 13 min 5 s on 10-08.
- **The repeat window resets** at every `HumanMessage` (`E:1776`), and so at every retry
  observation and every peer re-run.

The decisive measurement is #88's own: deepseek-v4-flash repeated a call 3/3 times *after being
told it was a repeat* (`AC6519`). A guard that relies on the model reading an observation is a
guard that depends on the model.

**Change (phase 2).**
- **Memoisation.** An identical `(tool, args)` call is answered from the event log, wherever it
  occurred in the turn. That includes failures: a failure is returned again together with the
  observation that it already failed. It costs no tool execution and no new record, and it
  supersedes #88's latest-step window.
- **Progress.** A peer step is productive iff it appended at least one new fact, artifact or
  successful result to the log. Two consecutive unproductive steps end the peer run with a
  stated reason. The same rule applies at the supervisor level: a peer run that added nothing
  ends the turn's attempts at that need.
- **Re-runs.** A re-run is allowed only when some bound tool's declared output kind can produce
  the missing item. This is a capability predicate over the registry
  (`capability_registry.py`) plus each tool's declared output kinds, and it replaces the
  one-entry `_UNPRODUCIBLE_CAPABILITIES` (`G:1008`) and its regexes.

**Invariant.** Every step either adds to the record or counts against a budget of 2. This
holds for any model and any shape of loop.

**Would have prevented:**
- the 10-08 geocode loop D12, ended at step 2 of 27, not by `GraphRecursionError`;
- D5, the 49 SoilGrids searches;
- D11, a gate unknown re-running the analysis;
- E7, the code peer re-fetching the DEM in 16 iterations;
- the open A-B-A-B alternation in both forms.

**Cost.** One comparison per tool call, and no extra model calls. The risk is a legitimately
exploratory step that adds nothing. One unproductive step is always allowed, and the harness
measures whether 2 is too strict.

**Does not fix.** A step that adds a *wrong* fact counts as productive. Correctness is phases
3–4.

### F5. The plan exists only inside the model

**Mechanism.** No plan, goal or subgoal exists in `SupervisorState`. The nearest things are
`needs`, which holds a capability name plus a reason string, and `grounding_gaps`. What reaches
each model:

- **Decider:** counts, flags and trimmed lines (§1). The reason a need was queued is dropped
  before it reaches the decider (`G:1754`).
- **`request_capability` re-runs** carry the generic reason "re-run after needs met"
  (`G:1817`). The actual reason never reaches the re-run peer's task text.
- **The code peer** sees `analysis_results` cut to **1,500 characters** (`G:5302`). After #85's
  merge, that is mostly the summary and the first few tool calls.
- **On a re-run**, the new task is a new `HumanMessage` on the same child thread (`G:4789–4791`).
  Continuity then depends on two things:
  - the checkpointer, which has no thread when no `thread_id` is given (`E:848`) and evicts at
    LRU 500;
  - the context-budget middleware, which keeps **only the last human message plus whatever
    recent tail fits** (`E:1495–1516`). On a retry the last human message is a bare observation,
    so **the original task can be trimmed out of the request**. This was not in the brief.
- **Reasoning** is preserved only for shims that send `reasoning_content`, and only the last
  4,000 characters (`E:223`, `257`). On OpenAI proper there is nothing to preserve
  (`AGENTS.md`, "Which model answers").

**Why it fails on unseen inputs.** A model that keeps its plan in hidden reasoning loses it
whenever the reasoning is not round-tripped: gpt-oss over AnvilGPT (`AG208`), and any future
shim. A model that keeps it in visible content loses it when the budget trims the content. In
both cases the plan's survival depends on the provider and the context size, not on the agent.
Incidents:
- E4: 5 `admin_boundary` calls, 2 identical, after gpt-oss lost its plan;
- E3: the follow-up routed to a peer that had never seen `scale_m`;
- E7: the DEM re-fetched;
- the 10-08 loop, where the model re-derived and re-asked.

**Change (phase 2).**
- **A plan object in state:** `{goal, subgoals: [{id, text, status, needs, produced:
  [fact_ids]}], open_questions}`. The decider creates it on the first step and updates it
  through a structured tool.
- **Rendered every step** into each peer's *system* message. The budget trimmer never trims
  that, so the plan is pinned the way the anchor message is today.
- **Subgoal status is derived from the event log where possible:** a subgoal whose declared
  output kind appeared is done. The model's own update is a hint, not the authority.
- **Pinned task text.** The original task text is pinned beside the plan, closing the trimming
  hole.

**Invariant.** What the turn is trying to do, and what is still missing, are state the agent
owns. They survive any provider, shim, budget trim or peer boundary.

**Would have prevented:**
- E3, E4 and E7;
- the needs-reason loss;
- E1, the decider treating analyze as the answer-writer, which became a subgoal type;
- the trimming hole.

**Cost.**
- About 300–600 tokens per model call for the rendered plan. That is roughly 1–2% of the
  25–30k-token requests measured in the phase 1 sample.
- One extra structured tool for the decider.

**Does not fix.** A bad plan. Making the plan visible makes it inspectable, and the harness can
then score plan quality. Making it good is the model's job.

### F6. Data sources are picked ad hoc and never named

**Mechanism.**
- **No catalog exists.** Source choice for "features of a kind in an area" is left entirely to
  the model, reading tool descriptions (`langchain_granular_tools.py:474–486`,
  `capability_registry.py:112–114`). Nothing records what each source includes or excludes,
  its extent, or its licence. The one comparison of OSM with Chicago's portal is a code comment
  (`langchain_granular_tools.py:407–416`).
- **`source` means four different things across tools:** a service, imagery, a search method,
  or a map-layer origin. The ledger renames it per tool (`G:610–616`).
- **Naming is enforced in one place**, `_with_feature_source`, for 2 tool families and lists of
  3 or more rows (`G:2068–2087`). A number from TIGERweb, 3DEP, a fetched CSV or the KB is never
  source-tagged.

**Why it fails on unseen inputs.** Coverage gaps are invisible until a user knows the answer:

| incident | what happened |
|---|---|
| 10-08 (`AC6528`) | 18 Chicago Public Schools reported, while OSM has 31 schools there |
| G4 | 3DEP answers 200 with NoData outside the US |
| G5 | rs-embed returned 200 while wedged |

**The phase 1 harness already measures it on correct answers.** Source named: deepseek 3/4,
luna 1/4 on the sample (the full baseline is in stage 41). This is **model-dependent behaviour**
that the user experiences as "where did this come from?". Today no code makes it
model-independent.

**Change (phase 5).**
- **A source catalog** (`{id, provider, covers, excludes, extent, licence, vintage,
  failure_modes}`) is consulted by selection and attached to every fact a source produces, as the
  `source` field from phase 3.
- **The answer renders the source deterministically from the facts it cites.** The model is not
  asked to remember it.
- **A silent-success detection rule for every source.** Examples: NoData coverage below a
  threshold, an empty result from a service that should not be empty. This is the general form of
  3DEP's refusal.

**Invariant.** Every fact names its source. Every answer names the sources of the facts it
cites. A source is chosen by declared coverage.

**Would have prevented:** the 10-08 schools source, G4 as a general rule rather than a 3DEP
special case, and B8 (Clay vs gse provenance).

**Cost.** A catalog file to maintain, about 15 sources. That is a list again, but a list of
*sources*, which change on the scale of months, not of *words*, which change every query. The
answer grows by one "Sources:" line.

**Does not fix.** A source that is wrong but complete. Coverage metadata says what a source
claims to contain, not whether it is accurate.

### F7. Warnings never reconcile

**Mechanism.** Thirteen post-processing steps run in sequence (§1). Each appends or prepends its
own text, behind its own `---` separator, without seeing the others' verdicts:

| producer | trigger | "could not check" kept separate from "problem"? |
|---|---|---|
| ⛔ / ⚠️ COULD NOT VERIFY / ℹ️ gate headline (`G:2203–2207`, `2704–2715`) | gate fail / unknown / advisory | **yes**, the only one |
| ⚠️ Grounding check (`G:2209`) | LLM severity `high` | **no** |
| ⚠️ Correction ×3 (`G:2286`) | regex on answer | n/a |
| Routing note (`G:1011`, `1220`, `1418`) | a routing claim was cut | n/a |
| **Source:** line (`G:2082`) | 3+ list rows | n/a |
| ⚠️ Partial answer (`G:2101`) | unresolved peer failure | says "failed" |
| Deterministic answer (`G:5831`) | synthesis raised | explicit |

Specific contradictions in the code (all inferred unless noted):

- **A failed audit looks exactly like a clean one.** When the audit model call fails, it returns
  `severity: "unknown"` (`EQ:643–645`), which raises no flag, so the answer goes out with no
  caveat. **The brief says "could not check" is shown as "detected". The code does both, in
  different producers.** The gate makes unknown loud. The LLM audit makes unknown silent.
- **The ℹ️ headline can mask kept high-severity issues.** When the gate is advisory-only, the
  headline says "Nothing the check could read failed". It is shown under ℹ️ with severity
  `low`, even when the LLM auditor kept high-severity issues (`G:2701`, `2721–2725`).
- **"Could not see" is rewritten as "found nothing".** The auditor is told to report a record as
  "TRUNCATED" (`EQ:328–333`), and `_recompute_verdict` then overwrites the reason.
- **The Source: line is placed by searching for the first `"\n⚠️"`**, so it misses ⛔ and ℹ️
  banners.

**Why it fails on unseen inputs.** Every new check adds a producer, and every pair of producers
is a possible contradiction. Seven producers make 21 pairs, and no code looks at any pair.
Incidents:
- F6: "hallucination … high" stapled under a unit-name note;
- F7: Partial answer after success;
- the 10-08 auditor contradicting the answerer;
- F4: the caveat accusing a ledger-correct follow-up.

**Change (phase 4).**
- **One finding type.** Every check emits findings `{check, severity, kind:
  problem|unverifiable|note, evidence: [fact_ids|event seqs], message}` into the event log.
- **One verdict policy.** A single function maps all findings to **one** user-facing status and
  at most one banner, which lists what was and was not checked.
- **Unknown is never "detected"**, and it is never silent either. "Could not check X" is
  reported as such.
- **Deletions.** The five banner producers are deleted. `_correct_artifact_claims`' three cases
  and `_with_feature_source` become finding producers.

**Invariant.** The user sees exactly one statement about the answer's reliability, derived from
every check that ran. Correct answers whose checks all ran carry no banner.

**Would have prevented:** F4, F6 and F7, both 10-08 false banners, A10's ⛔ on a correct 165 km²,
and the silent audit failure.

**Cost.** Small: the verdict policy is one pure function, and every producer already computes a
verdict. Migration means rewriting the tests that assert banner text.

**Does not fix.** A check that is wrong. One verdict from wrong findings is still wrong, but now
it is wrong in one place, and the harness's banner-on-correct score catches it.

### F8. Nothing re-runs whole tasks after a change

**Mechanism.** The suites test functions. Each of the 106 incidents was found by a person
reading a live answer, and each fix was checked against the turn that exposed it. Twelve fixes
caused or masked a later failure (§4). Five patches shipped on 2026-10-08 alone.

**Change (phase 1, PR #90).** `gis_harness/`: 17 tasks with pinned data and independently
computed answers, scored mechanistically per model through the real API. Each later phase must
show before/after scores on at least two models, and show that no other task regressed.

**Does not fix.** Anything by itself. It also measures only the 17 tasks it has. Generalisation
beyond them is argued by mechanism (this document) and checked by unsolvable variants and new
tasks that phases are not written against.

---

## 3. Flaws the brief did not list

1. **Gate failures are never superseded.** A fixed and re-run script still flags the turn
   (§F1). Belongs in phase 2: the log's `supersedes`.
2. **The context-budget trimmer can drop the original task on a retry** (§F5). Belongs in phase
   2: the pinned plan and task.
3. **The synthesizer sees less of the record than its auditor** (§1). It gets 2,000 characters
   and no current-turn ledger, while the auditor gets the ledger and 8,000 characters per
   result. The writer cannot use facts it was not given, and the auditor then flags it for
   omitting them. Phase 4: both read the same fact set.
4. **Some answers are never audited.** That covers conversational answers and the
   general-knowledge branch (§F2). Phase 4: the number scan runs on every answer, which costs
   nothing.
5. **A composition after a re-grounding pass can loop the turn.** First inferred from the
   exception path (`G:5888–5895`). Phase 2 then reproduced it through a different exit, the
   general-knowledge answer (`G:6262`), which also returns without resetting `reground`. With the
   flag still set, the conditional edge (`G:6153`) sends the graph back to the supervisor on every
   composition. The test reached LangGraph's recursion limit of 10,000. The fix is in phase 2:
   the node wrapper clears the flag on every exit except the one that asks for a pass, rather
   than at each return.
6. **The needs FIFO bypasses both the decider and the repeat guard.** `_dead` silently discards
   requests (`G:5560–5571`). Phase 2: needs become subgoals.
7. **Stale documentation in the code.**
   - `_is_unproductive_repeat`'s docstring still says each peer "overwrites a single result
     slot" (`G:2138`).
   - `E:1071` says the supervisor graph is compiled without a checkpointer.
   - `G:1763` says the synthesizer sees the same record as the decider.

   Each is a trap for the next patch.

---

## 4. Evidence: 106 incidents by class

The full table is Appendix A. The summary:

| class | incidents | list-entry patch fixes | open |
|---|---|---|---|
| A unit / CRS / what-is-this-number | 17 | 5 (+3 mixed) | 2 |
| B prose grounding | 19 | 8 (+1 mixed) | 1 |
| C state lost or overwritten | 14 | 1 | 1 |
| D loops / wasted steps | 15 | 5 | 0 (+ A-B-A-B) |
| E plan / intent lost | 12 | 1 | 0 |
| F contradictory or false warnings | 9 (+10 counted in A/B) | 1 | 0 |
| G source ad hoc / silent success | 11 | 3 | 3 |
| H other | 9 | 0 | 2 |
| **total** | **106** | **24 (+4)** | **9** |

The patch fixes cluster where the brief predicted: the gate's unit tables (A) and the phrase and
regex lists in `graph.py` (B). Classes C and E were mostly fixed generally, but each general fix
added another record or another channel. The ledger is "a patch over the architecture"
(`AG453`), and that is why phase 2 replaces records rather than adding one.

**Twelve fixes caused or masked a later failure.** Three of them:

- **Re-grounding.** The gate (`05fa222b`) later cost the 50 s and 28 s re-runs, wiped the
  first pass's record (S37.1) and stacked layers (S20.1).
- **Stage 36's noun list.** It broke on `schools` within hours (`AC6216`, `6473`).
- **S37.3's audit-flag gate.** Its cut applied only to flagged claims, so travel figures still
  shipped in 2 of 2 replays. Rule (2) had hidden the flag through the peer's own summary
  (`AC6677`).

All twelve share one property: the new rule read a *different* record or a *different*
vocabulary from the rule it interacted with. A single record and typed values remove the
surface such interactions happen on.

---

## 5. Model independence

Which failures come from model behaviour that a better model would hide, and how the new design
makes the outcome independent of it:

| behaviour | models seen | today depends on | after |
|---|---|---|---|
| repeats identical calls even when told | deepseek-v4-flash (3/3 after an observation, `AC6519`); gpt-oss (`AG208`) | the model reading an observation; a count of 4 | memoised by the harness of the loop. The model's reading is irrelevant (phase 2) |
| loses its plan between steps | gpt-oss over a shim; any model whose content is trimmed | provider and shim; context size | plan in state, rendered into the system message (phase 2) |
| unit vocabulary (`square_miles`, `schools`, `records`) | each model has its own | list membership | `pint` parse + dimension (phase 3) |
| quotes memorised figures | all; more often on deepseek in S37/S39 replays | the auditor noticing; regex classes | every number needs a fact (phase 4) |
| names, or does not name, its source | luna 1/4, deepseek 3/4 on the sample | model habit | rendered from cited facts (phase 5) |
| auditor severity drifts between runs | all (`EQ:294`, `657`) | sampling | numbers judged deterministically; LLM audit only for qualitative claims (phase 4) |
| writes "COULD NOT VERIFY"-triggering declarations | all | declaration wording | metadata from the tool, not the declaration (phase 3) |

What a better model will **not** hide, because it is structural:
- the seven records;
- the 2,000-character truncation;
- the trimmed task;
- gate failures never superseded;
- the silent audit failure;
- the pairwise banner contradictions.

These fail identically on every model, and the harness's per-model scores should show them as
shared misses.

---

## 6. What changes in the program

1. **Phase 2 grows** to include:
   - `supersedes` in the log (unsuperseded gate failures);
   - task and plan pinned in the system message (the trimming hole);
   - an explicit reset of the re-ground state on synthesis failure;
   - needs becoming subgoals (the FIFO bypass).

   These are the same state-model change. No new phase is needed.
2. **The tool result schema moves from phase 3 to phase 2,** in its minimal form: `{ok, error?,
   outputs: [typed values], artifacts, source?}`. The event log needs a uniform `ok` and uniform
   outputs to judge progress: `overpass_search` failures have no `ok` key today and would count
   as productive. Phase 3 then adds unit parsing, CRS lineage and the gate rewrite on top.
3. **Phase 4 includes the auditor's failure modes:** the silent unknown, the overwritten
   "TRUNCATED" reason, and the writer seeing less than the auditor.
4. **Phase 5 is unchanged,** but now has a concrete hook: the `source` field already present on
   every typed value from phase 3.
5. **Order unchanged:** harness → state → typed numbers → facts and one verdict → sources. Each
   phase's invariant is a precondition of the next:
   - progress needs a record;
   - typed facts need a record;
   - a number scan needs typed facts;
   - source rendering needs facts.

## 7. Total cost, summarised

| phase | tokens per turn | latency | migration |
|---|---|---|---|
| 2 state | +300–600 per call (plan) | −: loops end at step 2, not 60 | large: ledger, merge, distill onto one log |
| 3 typed numbers | ≈0 | ≈0 | medium: about 40 tool result builders, the gate rewritten, about 300 lines of lists deleted |
| 4 facts + verdict | −: fact table < prose dumps; shorter audit | −: no re-grounding for numeric claims | large: 5 banner producers, 5 reconciliation rules and 3 removal passes deleted |
| 5 sources | +1 line per answer | ≈0 | small: a catalog and a coverage check per source |

The harness reports measured cost per phase (`llm_usage` events). Where this table and the
measurements disagree, the measurements win, and this document should be corrected.

---

## Appendix A. The incident catalogue

How the catalogue was built: from `docs/agent-architecture-changes.md` (AC), `docs/DEVLOG.md`
(DL), `AGENTS.md` (AG) and the commit history, read on 2026-10-08. Line numbers may drift by a
few lines as those files grow.

Fix type:
- **G**: a general mechanism.
- **P**: a list entry or special-case rule.
- **M**: mixed, a general mechanism plus a list entry.
- **O**: still open.

### A. Unit / CRS / "what is this number" (17)

| date | incident | fix | type |
|---|---|---|---|
| 08-12 | An area in degrees² (0.196) exited 0. Rule (2) passed it because the number appeared in stdout (DL1440, 1480) | M6.1/M6.2 `4facd803` | G |
| 08-13 | An EPSG:3435 (US feet) buffer passed, 3.28× too large. The reconciliation walk missed failures 9 times in 19 (DL1909, 1933) | M6.A `0f751dca` | G |
| 08-13 | A `sys.modules` check made every arithmetic run "unverifiable" (DL1962) | M6.A | G |
| 08-13 | Seven NameErrors inside the gate surfaced as `cannot_determine` (DL1957, 2233) | structural test, M6.G | G (after 6 one-offs) |
| 08-13 | A frame named `df` was read as a GeoDataFrame, giving false unknowns (DL2163) | `_GEO_FRAME_HINTS` split | P |
| 08-13 | `reject_all_nan` failed correct runs that had a sparse column (DL2208) | M6.G | G |
| 08-13 | ⛔ on a correct 1,960.34 km²: the 4326 input frame was still bound, and `km²` was not an alias (DL2531) | rule relaxed + 9 areal aliases | M |
| 08-26 | Unit `records`: a correct THEFT count of 27,824 reported as unverified (DL3795) | count aliases + count check | M |
| 08-26 | A wrong population (9,993, from a 49,789-row join, vs 27,824) went undetected (DL3766) | population only recorded | O |
| 10-01 | ⛔ on a correct 165.04 km². The agent re-ran correct code 3 times: 46 steps, 3 download sets, 4 map layers (DL4813) | operation tracker, M8.68 | G |
| 10-08 | `unit:"points"` and `output_crs` "crs" → COULD NOT VERIFY on a correct run (AC6189) | label rule + 10 nouns | M |
| 10-08 | Work inside `def main()` gives a coverage unknown (AC6251) | — | O |
| 10-08 | `mi2` / `square_miles` not recognised (AC6318) | imperial aliases | P |
| 10-08 | Unit `schools` (AC6470) | `_inferred_count` name prefixes | P |
| 10-08 | `distance_m` carried back to 4326 was flagged (AC6482) | credit a projected twin frame | P |
| 10-08 | geocode used the bbox centre: London–Paris 340.0 vs 343.7 km, and nothing flagged it (AC6441) | S37.7 | G |
| 09-10 | 3DEP padding added 466 m per edge and squeezed the DEM 23.5% (AC1005) | `3ac9b762` | G |

### B. Prose grounding (19)

| date | incident | fix | type |
|---|---|---|---|
| 06-25 | A computed result was flagged for lacking a document (AC371) | prompt | G |
| 06-25 | Reasonable framing got a "scary" caveat (AC372) | prompt precision rule | P |
| 08-03 | Fabricated artifact URLs | `5151917d` | G |
| 08-06 | Clean and fabricated answers got the same verdict; an invented figure passed 3 times in 3 (AC446) | ledger before verdict | G |
| 08-07 | A roll-up sentence was rated "high" on a fully grounded answer, 5 times in 5 (DL668) | roll-up class + `_HARD_COUNT_RE` | P |
| 08-18 | The model claimed its code ran after a -11 death (AC418) | structural check | G |
| 08-20 | "Successfully partitioned using max-p" when the tool never completed | `tool_failures` | G |
| 08-31 | "Clay v1.5 … LGND" provenance over a gse raster; the audit passed it (AG349) | `_models_named_in` + `_DENIES_MAP_RE` | P |
| 08-31 | `scale_m=10` answered as "does not specify" (AG508) | `_FACT_PHRASES` | P |
| 09-02 | Bordering counties answered from memory after a gazetteer download (AC929) | audit becomes a gate | G |
| 09-02 | The auditor flagged "pan, zoom, click" (AC936) | `_MAP_CLIENT_AFFORDANCES` | P |
| 09-02 | The auditor was starved: 87,648 characters of record, 2,218 reaching it (AC942) | size-gated elision | G |
| 09-09 | A truncated record was read as absence (AC946) | `29f24c2d` | G |
| 09-04 | A `read_text_file` id was rendered as an output (AC901) | gated on `outputs` | P |
| 10-04 | A tract count of "approximately 100" in 5 of 6 runs; the true count is 48 (AC6035) | `fetch_public_data` | G |
| 10-08 | Memorised "460 km by road / 340 km by Eurostar" (AC6280) | `_UNPRODUCIBLE_CAPABILITIES`, one entry | P |
| 10-08 | The re-run answer restated those figures to disown "the earlier rejected answer" (AC6384) | directive + `_drop_draft_mentions` | P |
| 10-08 | Travel figures shipped in 2 of 2 replays; rule (2) read the peer's summary as evidence (AC6679) | `_without_peer_prose` + `_TRAVEL_MODE_RE` | M |
| 10-08 | "~998 sq mi" in 3 of 4 replays (AC6749) | — | O |

### C. State overwritten or lost (14)

| date | incident | fix | type |
|---|---|---|---|
| 06-25 | A follow-up recalled a cited element instead of the subject (AC352) | role-aware recall | G |
| 08-20 | History dropped inline map layers | `b5195887` | G |
| 08-31 | Turn 2 reported `has_evidence=False` however much turn 1 had done (AG473) | action ledger | G (AG453 calls it "a patch over the architecture") |
| 09-01 | A later turn re-added a layer already on the map | `e3ba7093` | G |
| 09-02 | The ledger was present at 2–7 history items and absent at 8+ (AC895) | own channel | G |
| 09-02 | A fail-then-succeed pair put the good `file_id` on the failed row (AC896) | `86f29229` | G |
| 09-02 | The previous turn's `execute_code` made this turn `executed=True` (AC920) | per-invocation scope | G |
| 09-02 | A failed `admin_boundary` counted as on the map (AC910) | one authority | G |
| 09-03 | A second embedded region overwrote the first (AC985) | region tag, then content digest | P → G |
| 09-09 | ContextVars did not cross threads, so the live path wrote `session=None` (AC1018) | `0fa5d255` | G |
| 09-22 | `chat_memory` was read-only and four days of conversations were lost silently (AC2067) | migration only | O |
| 10-03 | A restored layer was not counted, so a duplicate was drawn (AC1594) | S9.8 | G |
| — | A re-ground stacked a city layer on top of itself (AC4006) | input-keyed layer identity | G |
| 10-08 | The re-grounding pass replaced the first pass: 1 ledger row for a 5-call turn (AC6288) | `_merge_peer_result` | G |

### D. Loops and wasted steps (15)

| date | incident | fix | type |
|---|---|---|---|
| 04-15 | An AnalysisAgent loop had to be killed manually (AC280) | caps | G |
| 05-07 | `load_skill` called twice (AC229) | prompt + per-run sets | P |
| 06-12 | A search repeated with no result | run caps + `_is_unproductive_repeat` | G |
| 04 | Each nested search kept its own empty list, so nothing was deduplicated (AC212) | `d2d23df` | G |
| 08-31 | 49 keyword searches into SoilGrids: 66,275 tokens against a 65,536 window, and one turn reached 199,605 (AG445) | ledger + ceiling | G |
| 09-02 | A ledger note of 88,250 characters against a 6,000-character ceiling (AC899) | `4fcd474` | G |
| 09-02 | A context ceiling of 50,825 tokens when the real window is 922,000 (AC977) | per-model window table | P |
| 09-17 | Explicit `null` rejected: 141 parameters across 62 of 80 tools, and 4 calls to draw one DEM (AC1100) | signature wrapper | G |
| 09-04 | `GraphRecursionError` killed the turn | degrade | G |
| 10-03 | Search bound 14 MCP tools, at about 1,584 input tokens per call, and never called one (AC5231) | `SEARCH_MCP_MODULES` | P |
| 10-08 | A gate unknown re-ran the analysis: about 50 s and 5 extra LLM calls (AC6225) | `00e560f6` | G |
| 10-08 | 27 `geocode_places` calls, 15 of them identical; recursion limit at 60; a 13 min 5 s turn (AC6519) | repeat middleware | G (alternation open) |
| 10-04 | 11 fetches spent guessing TIGERweb layers (AC6089) | layers named in the tool description | P |
| 09-01 | gpt-oss spent 3 calls guessing "Champaign County" (AG543) | `_name_variants` suffix list | P |
| 10-04 | The dead-end check fired after a success (AC426) | count failures since the last success | G |

### E. Plan or intent lost (12)

| date | incident | fix | type |
|---|---|---|---|
| 06-09 | The decider treated analyze as the answer-writer (AC367) | prompt | G |
| 09-01 | With search off the menu, a retrieval question went straight to `done` (AC537) | veto in `supervisor_node` | P |
| 08-31 | A follow-up was routed to a peer that had never seen `scale_m` (AG438, 488) | ledger to synthesis | G |
| 09-01 | gpt-oss's reasoning was dropped by the shim: 5 `admin_boundary` calls, 2 of them identical (AG208) | reasoning-preserving subclass | G |
| 09-02 | The answer said "no map was produced" with a layer on screen (AC898) | `9e1201e3` | G |
| 09-17 | The capability paragraph drifted, so a DEM request searched the KB (AC1066) | generated registry | G |
| 09-17 | The code peer re-fetched the DEM: 266 s and 16 iterations (AC1082) | decider reads the ledger | G |
| 10-03 | The code line promised skills; the container had 0 (AC1131) | `b4981941` | G |
| — | `admin_boundary` was nested under `if input_file_ids:` (AG449) | binding fix | G |
| 09-02 | The capability answer was truncated by 56%; 0 of 6 embedding tools reached the prompt (AC954) | `9e339110` | G |
| 09-02 | With no file tool bound, `os.listdir` reported a scratch script as an artifact (AC1022) | `db843f8` | G |
| 10-08 | Analyze had 43 tools and none that looks up features (AC6535) | `overpass_search` bound | G |

### F. Contradictory or false warnings (9, plus 10 counted in A and B)

| date | incident | fix | type |
|---|---|---|---|
| 08-13 | `cannot_determine` never reached the user: two correct rules deleted it between them (DL2144) | M6.E | G |
| 08-13 | A failed search was reported as "the knowledge base has no matching content" (DL2508) | M6.M | G |
| 08-26 | "Findings were not retained" when there was nothing to check (DL3802) | M8.29 | G |
| 09-01 | The grounding caveat accused a ledger-correct follow-up (AG491) | auditor gets the ledger | G |
| 09-02 | An answer was flagged for a ledger line the auditor could not see (AC895) | `3d9587fa` | G |
| 10-08 | "Hallucination … high severity" over a unit-name finding (AC6333) | ℹ️ note keyed on a message | P |
| 10-08 | "Partial answer" after the code peer delivered (AC6504) | `_unresolved_peer_failures` | G |
| 08-13 | "Extracted from 4 of 4" on a half-failed read (DL2432) | M6.L | G |
| 08-13 | The trace said "no results" for ten hits (DL2262) | M6.H | G |

### G. Source ad hoc, unnamed, or silently succeeding (11)

| date | incident | fix | type |
|---|---|---|---|
| 08-13 | KB search read the local store while a cluster with 4,179 docs was reachable (DL2269) | report the backend | G |
| 08-18 | Geo queries went to the web-search MCP tool | routed to `overpass_search` | P |
| 06 | Semantic search results were presented as curated relationships (AC347) | short-circuits | G |
| 09-10 | 3DEP answered 200 with all NoData outside the US (AG402) | refusal | P |
| 09-08→10-01 | rs-embed returned 500 for 3 weeks while its health check said 200 (AC2551) | service account | O (health check unchanged) |
| — | rs-embed with no model named returns `{"results":[]}` with a 200 (AC2573) | — | O |
| 09-22 | The agent API was wedged for 3 days through 2,092 failed health checks (AG856) | watchdog | G |
| 10-01 | qwen3.6 was listed "Recommended" while returning zero bytes (AG161) | live probe | G |
| 10-04 | `web_fetch` reported the largest quake as M5.5; it was M5.9 (AC6092) | gated fetch | G |
| 10-08 | 18 schools from one district's file, source not named; OSM has 31 (AC6528) | `_with_feature_source` | P |
| 10-08 | Overpass mirrors answered 504/500/504 (AC6640) | PR #89 (merged 10-08) | G |

### H. Other (9)

- A missing tool produced a false answer: 708 grid cells buffered into 4,504 overlapping
  polygons (AC493). 7 of 15 documented selection failures were "the tool was ABSENT" (AC964).
- `fit_zone_model` joins succeeded silently while wrong (AC551–859):
  - a `KeyError` on `area_km2`;
  - a GEOID's leading zero dropped, so 0 zones joined;
  - 25 of 28 zones mispaired by row number.
- Fixes broke in the very next commit: M6.I → M6.J and M6.K → M6.L (DL2455).
- A configuration restart cut off 8 live turns (AC2582).
- `lstrip("mcp_")` mangled tool names (AC2600).
- The first `execute_code` could not open the boundary in all 3 S37 runs. Still open: the
  harness sees it on T01.
- A school 1607.1 m away was narrated as "just outside" a 1609.34 m mile (AC6646). Open.
- Test fixtures encoded the bug they were meant to catch (AC1008, 1018; AG815).
- A saved choice of gpt-5.6-luna broke every later turn (AC516).

### Fixes that caused or masked a later failure (12)

1. M6.N's metric-column rescue did not fire on a scalar, which led to M8.68's ⛔ and 3 re-runs
   (DL4817).
2. M6.2 switched rule (2) off under any gate verdict. That later produced the GEOID 17019 flag
   (AC6327).
3. The re-grounding gate later caused the 50 s and 28 s re-runs, the S37.1 record wipe and the
   S20.1 layer stacking.
4. The dead-end check fired on a dead end that had already been passed (AC426).
5. M6.G's downgrade left the call-site check recording nothing (DL2226).
6. M6.E: two correct severity rules together deleted `cannot_determine` (DL2144).
7. S38.2's first version replayed failed Overpass calls during an outage (AC6565).
8. S37.3 cut only flagged claims, while rule (2) had hidden the flag (AC6677).
9. Stage 36's noun list broke on `schools` within hours (AC6216, 6473).
10. The S39 scan also cuts a grounded "~344 km" that shares a sentence with a memorised figure
    (AC6752).
11. `895fb4a` fixed a layer-id bug caused by its author's own earlier commit (AC993).
12. `3d9587f`'s passing browser check happened to fall inside the only window where the wiring
    worked (AC895).
