#!/usr/bin/env bash
#
# iguide-rs-embed-watchdog — restart the satellite-embedding service when it stops embedding.
#
# Why this exists
# ---------------
# rs-embed (a systemd unit, uvicorn on 172.17.0.1:8077) reaches a state where it looks healthy
# and cannot embed anything. Twice now (2026-08-20, and 2026-10-19 15:53 to 2026-10-20 01:16
# UTC, nine hours): `GET /api/models` answers 200 in 2 ms, `systemctl` says active, and every
# `POST /api/embed` hangs until the agent's own 600 s timeout. The cause is an outbound Google
# call that never returns: the worker thread that made it is never freed, and once enough are
# parked that way every later request queues behind them. The service now sets an Earth Engine
# deadline (EE_DEADLINE_S in its startup), which should free those threads itself; this
# watchdog is the backstop for whatever else can park a worker, because the last two times
# nobody noticed for nine hours and sixteen hours respectively.
#
# How it decides
# --------------
# A health probe that exercises EMBEDDING, not a status endpoint: the same small precomputed
# request the service answered in 13.7 s right after the last restart. A probe counts as
# failed only when the service did not answer (timeout, connection reset) or answered 5xx.
# A 4xx means the PROBE is wrong for this service version, and that is logged loudly and
# never restarted for: a restart cannot fix the probe.
#
# Two consecutive failures (ten minutes at the timer's five-minute cadence, each probe capped
# at two minutes) restart the unit, after capturing what the process looked like. A restart
# ends any embed genuinely in flight, which the agent reports as an error in seconds; the
# alternative was every caller waiting ten minutes for nothing. Past MAX_RESTARTS in the
# window it stops and leaves the service for a human, as the agent watchdog does.
#
# Configuration: /etc/default/iguide-rs-embed-watchdog (see install-rs-embed-watchdog.sh).

set -uo pipefail

UNIT="${UNIT:-rs-embed}"
RS_EMBED_URL="${RS_EMBED_URL:-http://172.17.0.1:8077}"
# Seconds a probe may take. Measured 13.7 s cold after a restart; ten times that is a hang.
PROBE_TIMEOUT="${PROBE_TIMEOUT:-120}"
# Consecutive failed probes before a restart.
FAILURES_BEFORE_RESTART="${FAILURES_BEFORE_RESTART:-2}"
MAX_RESTARTS="${MAX_RESTARTS:-3}"
RESTART_WINDOW_SECONDS="${RESTART_WINDOW_SECONDS:-3600}"
INCIDENT_ROOT="${INCIDENT_ROOT:-/var/log/iguide-agent/incidents}"
STATE_DIR="${STATE_DIR:-/var/lib/iguide-rs-embed-watchdog}"
KEEP_INCIDENTS="${KEEP_INCIDENTS:-30}"
LOG_TAIL="${LOG_TAIL:-3000}"
DRY_RUN="${DRY_RUN:-0}"
# A small precomputed embedding over Urbana, IL: one tile, no on-the-fly model. Assigned in
# two steps on purpose: inside `${VAR:-default}` the first `}` of the JSON ends the expansion,
# which silently dropped the geometry's closing brace and made the service answer 200 with no
# results.
PROBE_BODY="${PROBE_BODY:-}"
if [ -z "$PROBE_BODY" ]; then
    PROBE_BODY='{"geometry":{"type":"bbox","minlon":-88.25,"minlat":40.10,"maxlon":-88.22,"maxlat":40.12},"start":"2024-06","end":"2024-08","models":["gse"]}'
fi

# To stderr: `probe` is called in a command substitution, and anything it writes to stdout
# becomes part of its verdict.
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

mkdir -p "$STATE_DIR"
FAIL_FILE="$STATE_DIR/consecutive_failures"
RESTARTS_FILE="$STATE_DIR/restarts"

failures() { cat "$FAIL_FILE" 2>/dev/null || echo 0; }
set_failures() { echo "$1" > "$FAIL_FILE"; }

# --- the probe ---------------------------------------------------------------------------

probe() {
    # Prints one word: ok | hang | error | probe-bad | down
    if ! systemctl is-active --quiet "$UNIT"; then
        echo down; return
    fi
    local body code
    body="$(mktemp)"
    code="$(curl -s -m "$PROBE_TIMEOUT" -o "$body" -w '%{http_code}' \
            -H 'Content-Type: application/json' -X POST "$RS_EMBED_URL/api/embed" \
            -d "$PROBE_BODY" 2>/dev/null)"
    local verdict
    if [ "$code" = 200 ] && grep -q '"ok": *true' "$body"; then
        verdict=ok
    elif [ "$code" = 000 ]; then
        verdict=hang
    elif [ "$code" = 200 ] || [ "${code:0:1}" = 4 ]; then
        # Answered, but not with an embedding: the request is wrong for this service version,
        # or its data is missing. The service is alive, and a restart would not change it.
        verdict=probe-bad
        log "probe answered $code without an embedding: $(head -c 300 "$body")"
    else
        verdict=error
        log "probe answered $code: $(head -c 300 "$body")"
    fi
    rm -f "$body"
    echo "$verdict"
}

# --- evidence before the restart ----------------------------------------------------------

capture_incident() {
    local dir="$INCIDENT_ROOT/rs-embed-$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$dir"
    local pid
    pid="$(systemctl show "$UNIT" -p MainPID --value 2>/dev/null)"
    {
        echo "unit: $UNIT  pid: $pid  consecutive failures: $(failures)"
        systemctl status "$UNIT" --no-pager 2>&1 | head -20
    } > "$dir/status.txt"
    if [ -n "$pid" ] && [ "$pid" != 0 ]; then
        ps -o pid,etime,time,nlwp,rss,stat -p "$pid" > "$dir/process.txt" 2>&1
        # Which hosts the parked threads are waiting on: the last incident held ten
        # connections to Google and none to anything else.
        ss -tnp state established 2>/dev/null | grep "pid=$pid," > "$dir/connections.txt"
        local spy
        spy="$(command -v py-spy 2>/dev/null || ls "$(systemctl show "$UNIT" -p WorkingDirectory --value)"/.venv/bin/py-spy 2>/dev/null || true)"
        if [ -n "$spy" ]; then
            timeout 60 "$spy" dump --pid "$pid" --nonblocking > "$dir/stacks.txt" 2>&1
        else
            echo "py-spy is not installed: no thread stacks captured" > "$dir/stacks.txt"
        fi
    fi
    journalctl -u "$UNIT" --no-pager -n "$LOG_TAIL" > "$dir/journal.txt" 2>&1
    # Prune the oldest bundles of this service beyond KEEP_INCIDENTS.
    ls -1d "$INCIDENT_ROOT"/rs-embed-* 2>/dev/null | sort | head -n -"$KEEP_INCIDENTS" \
        | xargs -r rm -rf
    echo "$dir"
}

# --- the restart budget ------------------------------------------------------------------

restarts_in_window() {
    local now cutoff
    now="$(date +%s)"; cutoff=$((now - RESTART_WINDOW_SECONDS))
    [ -f "$RESTARTS_FILE" ] || { echo 0; return; }
    awk -v c="$cutoff" '$1 >= c' "$RESTARTS_FILE" | tee "$RESTARTS_FILE.tmp" | wc -l
    mv -f "$RESTARTS_FILE.tmp" "$RESTARTS_FILE"
}

restart_unit() {
    local n
    n="$(restarts_in_window)"
    if [ "$n" -ge "$MAX_RESTARTS" ]; then
        log "NOT restarting $UNIT: $n restarts in the last ${RESTART_WINDOW_SECONDS}s already. A restart is not fixing it; a human is needed."
        return 1
    fi
    local dir
    dir="$(capture_incident)"
    log "evidence captured in $dir"
    if [ "$DRY_RUN" = 1 ]; then
        log "DRY_RUN: would restart $UNIT now"
        return 0
    fi
    date +%s >> "$RESTARTS_FILE"
    # systemd bounds the stop itself (TimeoutStopSec, 90 s by default) and SIGKILLs past it.
    if ! timeout 180 systemctl restart "$UNIT"; then
        log "systemctl restart $UNIT failed or exceeded 180 s"
        return 1
    fi
    local i
    for i in $(seq 1 24); do
        if curl -s -m 3 -o /dev/null -w '%{http_code}' "$RS_EMBED_URL/api/models" | grep -q '^200$'; then
            log "restarted $UNIT; /api/models answers again after ~$((i * 5))s"
            return 0
        fi
        sleep 5
    done
    log "restarted $UNIT but /api/models has not answered within 120 s"
    return 1
}

# --- main ---------------------------------------------------------------------------------

main() {
    local verdict
    verdict="$(probe)"
    case "$verdict" in
        ok)
            if [ "$(failures)" != 0 ]; then
                log "probe ok again after $(failures) failure(s)"
            fi
            set_failures 0
            ;;
        down)
            # Stopped or failed. systemd's Restart=on-failure covers a crash; an operator's
            # `systemctl stop` is deliberate. Neither is this script's to undo.
            log "$UNIT is not active; leaving it alone"
            set_failures 0
            ;;
        probe-bad)
            log "the probe itself is wrong for this service (see above); fix PROBE_BODY in /etc/default/iguide-rs-embed-watchdog. Not restarting."
            ;;
        hang|error)
            local n
            n=$(( $(failures) + 1 ))
            set_failures "$n"
            log "probe $verdict ($n of $FAILURES_BEFORE_RESTART before a restart)"
            if [ "$n" -ge "$FAILURES_BEFORE_RESTART" ]; then
                if restart_unit; then
                    set_failures 0
                fi
            fi
            ;;
    esac
}

main "$@"
