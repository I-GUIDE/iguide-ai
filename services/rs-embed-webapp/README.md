# rs-embed web app — tracked deployment snapshot

`server.py` here is the rs-embed FastAPI service the agent talks to via `RS_EMBED_URL`
(`agent_runtime/rs_embed_tools.py`). It is vendored into this repo because **it is not in version
control anywhere else**.

## Why it lives here

Upstream is `cybergis/rs-embed`, where `.gitignore:14` ignores `examples/**`. `server.py` is
therefore untracked there:

```
$ git check-ignore -v examples/webapp/server.py
.gitignore:14:examples/**	examples/webapp/server.py
$ git ls-files --error-unmatch examples/webapp/server.py
error: pathspec ... did not match any file(s) known to git
```

So the deployed service existed only on the VM and in `~/webapp-backups/` — a host rebuild would
have lost it, and no change to it was reviewable. This directory is the fix.

## What is deployed

The snapshot committed as the baseline was taken from the running host:

| | |
|---|---|
| path | `/home/exouser/Documents/rs-embed/examples/webapp/server.py` |
| rs-embed branch | `webapp-zones` at `018fffc` |
| file mtime | 2026-08-31 22:59:34 UTC |
| sha256 | `a43d4e94530017ec79575754a6cb120a75af3a52f0719c808857bd1755a2b5fb` |

The service runs as `uvicorn server:app --app-dir examples/webapp --host 172.17.0.1 --port 8077`.

## This is a snapshot, not a fork

Upstream remains `cybergis/rs-embed`. Changes made here have to be applied to the VM by hand and,
ideally, offered upstream — there is no automation tying the two together. Re-snapshot with:

```bash
scp <vm>:/home/exouser/Documents/rs-embed/examples/webapp/server.py services/rs-embed-webapp/server.py
```

and check the sha against the host before assuming the tree matches what is running.

## Routes it serves

`/api/models` `/api/preview` `/api/embed` `/api/segment` `/api/change` `/api/similarity`
`/api/zones` `/api/heads` `/api/predict` `/api/predict_example` `/api/predict_package`
`/api/download/{name}` `/api/health`

Two constants worth knowing, because agent-side behaviour depends on them:

- `GRID_SAVE_MAX_CELLS = 300 * 300` (`:135`) is now a cell *budget*, not a cliff. A grid larger
  than the budget is decimated by `_grid_stride` and always written; the manifest records
  `grid_stride` and `grid_saved_hw` beside the native `grid_hw`. `EmbedReq.grid_max_cells`
  raises the budget per request (0 = the default). Until 2026-09-09 the grid was *dropped*
  above the cap, which meant `embed_region`'s own default footprint (`buffer_m=2048`, a 4096 m
  square, 410×411 = 168,100 cells) exported no pixels at all.
- `/api/change` embeds whole calendar years — `_temporal(rs, m, f"{y}-01", f"{y}-12")` (`:753`).
  It has no `start`/`end` field, and silently discards the ones the agent tool sends.

## Credential

The service holds a **personal** Google Earth Engine OAuth credential. An OAuth token from an app
in testing status expires after seven days, and it has expired at least once — the service stays up
and `/api/models` keeps returning 200 while every embed fails. Check it before relying on a restart.
