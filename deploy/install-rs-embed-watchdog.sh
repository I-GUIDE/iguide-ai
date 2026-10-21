#!/usr/bin/env bash
#
# Install the rs-embed watchdog: a timer that probes a real embedding every five minutes and
# restarts the rs-embed systemd unit when it stops answering. Idempotent; requires root.
#
# Run from the repository root:   sudo ./deploy/install-rs-embed-watchdog.sh
#
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST=/opt/iguide-agent

[ "$(id -u)" -eq 0 ] || { echo "run with sudo" >&2; exit 1; }

echo "==> installing the rs-embed watchdog into $DEST"
install -d -m 0755 "$DEST"
install -m 0755 "$SRC/rs-embed-watchdog.sh" "$DEST/rs-embed-watchdog.sh"
install -d -m 0755 /var/log/iguide-agent/incidents /var/lib/iguide-rs-embed-watchdog

# Defaults are written once and never overwritten, so a host's tuning survives a redeploy.
if [ ! -f /etc/default/iguide-rs-embed-watchdog ]; then
    echo "==> writing /etc/default/iguide-rs-embed-watchdog (first install)"
    cat > /etc/default/iguide-rs-embed-watchdog <<'DEFAULTS'
# The systemd unit to probe and restart, and where it listens.
UNIT=rs-embed
RS_EMBED_URL=http://172.17.0.1:8077

# Seconds one probe may take. The probe embedding answered in 13.7 s right after a restart.
PROBE_TIMEOUT=120

# Consecutive failed probes (one per timer firing) before a restart.
FAILURES_BEFORE_RESTART=2

# Flap guard: past this many restarts in the window, stop and leave it for a human.
MAX_RESTARTS=3
RESTART_WINDOW_SECONDS=3600

# Evidence bundles kept under /var/log/iguide-agent/incidents/rs-embed-*.
KEEP_INCIDENTS=30

# Set to 1 to log what it WOULD do without restarting anything.
DRY_RUN=0

# The probe request. Change it only if the service's /api/embed schema changes; a 4xx answer
# is logged as a bad probe and never restarted for.
# PROBE_BODY='{"geometry":{"type":"bbox","minlon":-88.25,"minlat":40.10,"maxlon":-88.22,"maxlat":40.12},"start":"2024-06","end":"2024-08","models":["gse"]}'
DEFAULTS
else
    echo "==> keeping existing /etc/default/iguide-rs-embed-watchdog"
fi

echo "==> installing the systemd unit and timer"
install -m 0644 "$SRC/iguide-rs-embed-watchdog.service" /etc/systemd/system/
install -m 0644 "$SRC/iguide-rs-embed-watchdog.timer"   /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now iguide-rs-embed-watchdog.timer

echo
echo "installed."
systemctl list-timers iguide-rs-embed-watchdog.timer --no-pager || true
echo
echo "  decisions:   journalctl -u iguide-rs-embed-watchdog -f"
echo "  incidents:   ls /var/log/iguide-agent/incidents/rs-embed-*"
echo "  probe now:   sudo systemctl start iguide-rs-embed-watchdog.service"
