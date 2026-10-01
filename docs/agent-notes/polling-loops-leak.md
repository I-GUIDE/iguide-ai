---
name: polling-loops-leak
description: until-loops polling remote logs leak forever when the pattern is wrong or the --since window slides past the match; verify the pattern against a real log line first
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 9b1d4b74-150f-4936-b715-5b9f69c0880d
  modified: 2026-09-01T18:04:23.016Z
---

On 2026-09-01 a task-hygiene check found **four** background `until` loops still cycling from
this session — 20h, 19h, 14h and 1h old — each SSHing to the Jetstream VM every 15–20 s. None
could ever have exited. Their exit conditions:

- `grep -q "turn_ledger_recorded"` — the real log line is `turn ledger: recorded` (spaces and a
  colon). **The pattern was wrong from the first iteration and could never match.**
- `d.get('label')=='synthesize'` on workflow `journal.jsonl` — result lines carry no `label`.
- `docker logs --since 4m | grep -q "Answer composed"` — a *sliding* window: once the turn
  finished, the match scrolled out and never came back.
- `grep -c "Answer composed" | grep -qv "^[01]$"` — needed ≥2 hits in a 5-minute window.

**Why:** the harness reports "moved to background" and later a completion notification for the
*wrapper*, so a loop whose condition never fires looks finished from the transcript while the
shell keeps running. Nothing surfaces it. They are invisible until something goes looking.

**How to apply:** before arming an `until` loop on a log, run the grep ONCE against real output
and confirm it matches — never hand-write the pattern from memory of the log format. Prefer a
non-sliding source (a file, a full log, `--since` wide enough to cover the whole wait) so a
match that already happened still counts. Add a bound (iteration cap or deadline) so a wrong
pattern dies instead of spinning. And periodically check for strays: `ps -eo pid,etime,command
| grep shell-snapshots` lists this session's background shells; map a PID to its task id by
comparing `ps -o lstart=` to the mtimes in the session `tasks/` directory, then `TaskStop` it.
See [[verify-in-web-prototype]].
