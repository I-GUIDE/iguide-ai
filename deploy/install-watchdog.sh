#!/usr/bin/env bash
#
# Install the agent watchdog and make container logs outlive their containers.
# Idempotent: safe to re-run after every deploy. Requires root (sudo).
#
# Run from the repository root:   sudo ./deploy/install-watchdog.sh
#
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST=/opt/iguide-agent

[ "$(id -u)" -eq 0 ] || { echo "run with sudo" >&2; exit 1; }

echo "==> installing the watchdog into $DEST"
install -d -m 0755 "$DEST"
install -m 0755 "$SRC/agent-watchdog.sh" "$DEST/agent-watchdog.sh"
install -d -m 0755 /var/log/iguide-agent/incidents /var/lib/iguide-agent-watchdog

# Defaults are written once and never overwritten: an operator who has tuned the thresholds
# on this host keeps their tuning across deploys.
if [ ! -f /etc/default/iguide-agent-watchdog ]; then
    echo "==> writing /etc/default/iguide-agent-watchdog (first install)"
    cat > /etc/default/iguide-agent-watchdog <<'DEFAULTS'
# Which containers the watchdog acts on.
WATCH_CONTAINERS="agent-api mcp-server embedding-server"

# Continuous unhealthy time before a restart. Long on purpose: with gunicorn --threads 4, a
# handful of concurrent streaming turns can make a healthy agent look unresponsive, and
# restarting then would kill live user turns.
UNHEALTHY_SECONDS=600

# Flap guard: past this many restarts in the window, stop and leave it down for a human.
MAX_RESTARTS=3
RESTART_WINDOW_SECONDS=3600

# Incident bundles kept on disk (oldest pruned beyond this).
KEEP_INCIDENTS=60

# Set to 1 to log what it WOULD do without touching any container.
DRY_RUN=0
DEFAULTS
else
    echo "==> keeping existing /etc/default/iguide-agent-watchdog"
fi

echo "==> installing the systemd unit and timer"
install -m 0644 "$SRC/iguide-agent-watchdog.service" /etc/systemd/system/
install -m 0644 "$SRC/iguide-agent-watchdog.timer"   /etc/systemd/system/

# --- persistent logs ---------------------------------------------------------------------
# Container logs must survive the container. The json-file driver stores them under the
# container's own directory, so `docker rm` — and therefore `up -d --force-recreate`, which
# every deploy runs — deletes them. The journald driver hands each line to the host journal
# instead, which is independent of the container's lifetime.

echo "==> configuring journald for persistent, un-rate-limited container logs"
install -d -m 0755 /etc/systemd/journald.conf.d
cat > /etc/systemd/journald.conf.d/iguide.conf <<'JOURNALD'
[Journal]
# Explicit rather than "auto": auto keeps logs only if /var/log/journal already exists, so a
# reinstalled host would silently fall back to a memory-only journal that dies on reboot.
Storage=persistent
# Cap by SIZE. The default (10% of the filesystem, up to 4G) is fine but implicit; pinning it
# means container logging cannot quietly grow into the disk pressure that took the OpenSearch
# cluster read-only at 95%.
SystemMaxUse=3G
SystemKeepFree=5G
JOURNALD

install -d -m 0755 /etc/systemd/system/docker.service.d
cat > /etc/systemd/system/docker.service.d/log-ratelimit.conf <<'RATELIMIT'
[Service]
# Container logs reach the journal through dockerd, so journald's per-unit rate limit
# (10000 messages / 30s by default) applies to ALL containers together. A failing service is
# exactly when logging bursts — and exactly when dropped lines cost the most. Size is already
# bounded by SystemMaxUse, so bound by size, not by rate.
LogRateLimitIntervalSec=0
RATELIMIT

systemctl daemon-reload
systemctl restart systemd-journald

echo "==> enabling the timer"
systemctl enable --now iguide-agent-watchdog.timer

echo
echo "installed."
systemctl list-timers iguide-agent-watchdog.timer --no-pager || true
echo
echo "  decisions:        journalctl -u iguide-agent-watchdog -f"
echo "  container logs:   journalctl CONTAINER_NAME=agent-api -f"
echo "  incidents:        ls /var/log/iguide-agent/incidents/"
echo
echo "NOTE: the journald log driver is set in docker-compose.yml and applies to a container"
echo "      only when it is CREATED. Existing containers keep json-file until recreated by"
echo "      'docker compose up -d --build'."
