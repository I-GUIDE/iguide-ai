#!/usr/bin/env bash
#
# iguide-agent-watchdog — act on a container health check that Docker itself will not act on.
#
# Why this exists
# ---------------
# On 2026-09-18 agent-api hung: every gunicorn thread was blocked in LLM I/O with no timeout,
# so the container answered nothing for three days while reporting "Up 3 days (unhealthy)"
# with a health-check failing streak of 2,092. Nothing restarted it, because:
#
#   * `restart: unless-stopped` restarts a container that EXITS. A wedged container never exits.
#   * gunicorn's `--timeout 600` kills a worker stuck in a REQUEST, not one stuck in I/O
#     inside a thread (the deployment runs `--threads 4`, one worker).
#   * Docker records the health status and takes no action on it. That is by design; the
#     restarting is somebody else's job. This script is that somebody.
#
# The capture-before-restart rule
# -------------------------------
# Restarting a hung service destroys the only copy of why it hung. Recreating it destroys the
# logs too. So this script, in order: captures a full incident bundle, THEN restarts, and uses
# `docker restart` (keeps the container and its log history) rather than `up --force-recreate`
# (throws both away).
#
# What it deliberately does NOT do
# --------------------------------
# It does not restart on a dependency outage. The container health check is a LIVENESS probe
# ("can this process still answer?"), which is the only question whose answer is "restart me".
# If OpenSearch or the LLM upstream is down, the agent still answers /health and this script
# stays out of the way — restarting cannot fix somebody else's service, and a restart loop
# against an external outage is worse than the outage.
#
# Configuration: /etc/default/iguide-agent-watchdog (see install-watchdog.sh).

set -uo pipefail

# --- defaults, overridable from /etc/default/iguide-agent-watchdog -------------------------

WATCH_CONTAINERS="${WATCH_CONTAINERS:-agent-api mcp-server embedding-server}"

# How long a container must stay unhealthy before we restart it. Generous on purpose: a
# streaming agent turn can occupy a worker thread for minutes, and with `--threads 4` four
# concurrent long turns can make a HEALTHY service look unresponsive. Ten minutes of
# continuous failure is pathological; two minutes is a busy afternoon.
UNHEALTHY_SECONDS="${UNHEALTHY_SECONDS:-600}"

# Flap guard. If restarting did not fix it, restarting again will not either — past this many
# restarts in the window we stop and shout, leaving the service down for a human. A service
# that is down and loud beats one that is restarting every ten minutes and silent.
MAX_RESTARTS="${MAX_RESTARTS:-3}"
RESTART_WINDOW_SECONDS="${RESTART_WINDOW_SECONDS:-3600}"

INCIDENT_ROOT="${INCIDENT_ROOT:-/var/log/iguide-agent/incidents}"
STATE_DIR="${STATE_DIR:-/var/lib/iguide-agent-watchdog}"
KEEP_INCIDENTS="${KEEP_INCIDENTS:-60}"
LOG_TAIL="${LOG_TAIL:-20000}"

# Every docker call is bounded. A watchdog that hangs against a hung daemon is not a watchdog.
DOCKER_TIMEOUT="${DOCKER_TIMEOUT:-30}"

DRY_RUN="${DRY_RUN:-0}"

# --- logging -------------------------------------------------------------------------------
# stdout/stderr land in the journal under this unit; `journalctl -u iguide-agent-watchdog`
# is therefore the full history of every decision this script has ever made.

log()  { printf '%s %s\n'    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
warn() { printf '%s %s\n'    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

d() { timeout "$DOCKER_TIMEOUT" docker "$@"; }

# --- restart budget ------------------------------------------------------------------------

restarts_in_window() {
    local container="$1" file="$STATE_DIR/${1}.restarts" now cutoff
    [ -f "$file" ] || { echo 0; return; }
    now=$(date +%s); cutoff=$(( now - RESTART_WINDOW_SECONDS ))
    awk -v c="$cutoff" '$1 > c' "$file" | wc -l | tr -d ' '
}

record_restart() {
    local container="$1" file="$STATE_DIR/${1}.restarts" now cutoff tmp
    now=$(date +%s); cutoff=$(( now - RESTART_WINDOW_SECONDS * 24 ))
    mkdir -p "$STATE_DIR"
    tmp="$(mktemp)"
    { [ -f "$file" ] && awk -v c="$cutoff" '$1 > c' "$file"; echo "$now"; } > "$tmp" 2>/dev/null
    mv "$tmp" "$file"
}

# --- incident capture ----------------------------------------------------------------------
# Best-effort throughout: a diagnostic that fails must not prevent the rest of the bundle, and
# nothing here may abort the restart. Every command writes its own failure into its own file.

capture_incident() {
    local container="$1" reason="$2" dir="$3"
    mkdir -p "$dir"

    printf 'container: %s\nreason:    %s\ncaptured:  %s\nhost:      %s\n' \
        "$container" "$reason" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname)" > "$dir/00-summary.txt"

    d inspect "$container"                              > "$dir/inspect.json"      2>&1
    d inspect "$container" --format '{{json .State.Health}}' \
        | (python3 -m json.tool 2>/dev/null || cat)     > "$dir/health.json"       2>&1
    d logs --timestamps --tail "$LOG_TAIL" "$container" > "$dir/container-logs.txt" 2>&1
    d top "$container" -eo pid,ppid,etime,stat,cmd      > "$dir/host-process-table.txt" 2>&1
    d stats --no-stream                                 > "$dir/docker-stats.txt"  2>&1

    # Python thread stacks — the one artefact that says WHERE it is stuck rather than THAT it
    # is. py-spy attaches without the target's cooperation, which is the point: a process
    # wedged holding the GIL cannot run its own signal handlers, but can still be read from
    # outside. Absent from older images; recorded as unavailable rather than failing the run.
    {
        if d exec "$container" sh -c 'command -v py-spy' >/dev/null 2>&1; then
            for pid in $(d exec "$container" sh -c 'ls /proc 2>/dev/null' | tr -d '\r' | grep -E '^[0-9]+$'); do
                echo "===== pid $pid ====="
                d exec "$container" py-spy dump --pid "$pid" --nonblocking 2>&1
                echo
            done
        else
            echo "py-spy is not installed in this image — no Python thread stacks captured."
            echo "Rebuild the image (rag_pipeline/Dockerfile installs it) to get stacks next time."
        fi
    } > "$dir/thread-stacks.txt" 2>&1

    {
        echo "=== uptime ==="; uptime
        echo; echo "=== disk ==="; df -h
        echo; echo "=== memory ==="; free -m
        echo; echo "=== load/oom ==="; dmesg --ctime 2>/dev/null | tail -50
    } > "$dir/host.txt" 2>&1

    # The journal keeps this container's output across restarts and recreations, so it reaches
    # further back than `docker logs` on a container that was replaced.
    journalctl CONTAINER_NAME="$container" --since "-6h" --no-pager > "$dir/journal-6h.txt" 2>&1

    log "captured incident bundle: $dir"
}

prune_incidents() {
    [ -d "$INCIDENT_ROOT" ] || return 0
    local count
    count=$(find "$INCIDENT_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
    [ "$count" -le "$KEEP_INCIDENTS" ] && return 0
    find "$INCIDENT_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null \
        | sort -n | head -n $(( count - KEEP_INCIDENTS )) | cut -d' ' -f2- \
        | while read -r old; do rm -rf "$old"; log "pruned old incident bundle: $old"; done
}

# --- the check itself ----------------------------------------------------------------------

check_container() {
    local container="$1"
    local state health streak interval_ns interval unhealthy_for reason budget stamp dir

    state=$(d inspect "$container" --format '{{.State.Status}}' 2>/dev/null) || {
        warn "$container: cannot inspect (does it exist?) — skipping"
        return
    }

    health=$(d inspect "$container" --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' 2>/dev/null)
    streak=$(d inspect "$container" --format '{{if .State.Health}}{{.State.Health.FailingStreak}}{{else}}0{{end}}' 2>/dev/null)
    streak=${streak:-0}

    # An exited container that `restart: unless-stopped` has given up on, or that was stopped
    # by an OOM kill. Docker's restart policy backs off exponentially and eventually stops.
    if [ "$state" = "exited" ] || [ "$state" = "dead" ]; then
        reason="state=$state"
        budget=$(restarts_in_window "$container")
        if [ "$budget" -ge "$MAX_RESTARTS" ]; then
            warn "$container: $reason but $budget restarts already in the last $((RESTART_WINDOW_SECONDS/60))m — NOT restarting, needs a human"
            return
        fi
        stamp=$(date -u +%Y%m%dT%H%M%SZ)
        dir="$INCIDENT_ROOT/${stamp}-${container}-exited"
        warn "$container: $reason — capturing and starting"
        capture_incident "$container" "$reason" "$dir"
        [ "$DRY_RUN" = "1" ] && { log "$container: DRY_RUN=1, not starting"; return; }
        d start "$container" >/dev/null 2>&1 && log "$container: started" || warn "$container: start FAILED"
        record_restart "$container"
        return
    fi

    if [ "$health" != "unhealthy" ]; then
        log "$container: state=$state health=$health — ok"
        return
    fi

    # Elapsed time unhealthy, from the container's own configured interval rather than an
    # assumed 30s, so changing the interval in compose does not silently change the threshold.
    # `{{json ...}}`, not `{{.Config.Healthcheck.Interval}}`: the plain form renders a Go
    # time.Duration through its String method and yields "30s", which is not arithmetic.
    # The json form marshals the underlying int64 nanoseconds on every Docker version.
    interval_ns=$(d inspect "$container" --format '{{json .Config.Healthcheck.Interval}}' 2>/dev/null)
    case "$interval_ns" in
        ''|*[!0-9]*) interval=30 ;;                       # unparseable: assume the compose default
        *)           interval=$(( interval_ns / 1000000000 )) ;;
    esac
    [ "$interval" -lt 1 ] && interval=30
    unhealthy_for=$(( streak * interval ))

    if [ "$unhealthy_for" -lt "$UNHEALTHY_SECONDS" ]; then
        warn "$container: UNHEALTHY for ~${unhealthy_for}s (streak=$streak) — below the ${UNHEALTHY_SECONDS}s threshold, watching"
        return
    fi

    budget=$(restarts_in_window "$container")
    if [ "$budget" -ge "$MAX_RESTARTS" ]; then
        warn "$container: UNHEALTHY for ~${unhealthy_for}s but $budget restarts already in the last $((RESTART_WINDOW_SECONDS/60))m — NOT restarting. Restarting is not fixing it; this needs a human."
        return
    fi

    reason="unhealthy for ~${unhealthy_for}s (failing streak $streak, interval ${interval}s)"
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    dir="$INCIDENT_ROOT/${stamp}-${container}-unhealthy"

    warn "$container: $reason — capturing evidence, then restarting"
    capture_incident "$container" "$reason" "$dir"

    if [ "$DRY_RUN" = "1" ]; then
        log "$container: DRY_RUN=1, not restarting"
        return
    fi

    # `restart`, never `up -d --force-recreate`: recreation deletes the container and its logs
    # go with it. That is exactly how the 2026-09-18 hang lost its own evidence.
    if d restart "$container" >/dev/null 2>&1; then
        record_restart "$container"
        log "$container: restarted (restart $((budget + 1)) of $MAX_RESTARTS in this window)"
        sleep 20
        log "$container: post-restart state=$(d inspect "$container" --format '{{.State.Status}}' 2>/dev/null) health=$(d inspect "$container" --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' 2>/dev/null)"
    else
        record_restart "$container"
        warn "$container: RESTART FAILED"
    fi
}

# --- main ----------------------------------------------------------------------------------

mkdir -p "$INCIDENT_ROOT" "$STATE_DIR"

if ! timeout "$DOCKER_TIMEOUT" docker info >/dev/null 2>&1; then
    warn "docker daemon is not responding — nothing this script can do; leaving it alone"
    exit 0
fi

for c in $WATCH_CONTAINERS; do
    check_container "$c"
done

prune_incidents
