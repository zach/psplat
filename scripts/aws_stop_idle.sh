#!/usr/bin/env bash
# Stop this instance once it has been idle for PSPLAT_IDLE_MINUTES (default 45).
# Installed by aws_bootstrap.sh as a systemd timer that runs every 5 minutes.
#
# Busy means any of: an SSM session, a psplat or gsplat training process, or a
# login shell. The instance's shutdown behaviour is "stop", so this stops
# rather than terminates, and the disk survives. Run by hand with --dry-run to
# see what it would decide.
set -euo pipefail
limit_min="${PSPLAT_IDLE_MINUTES:-45}"
state="/var/tmp/psplat-idle-since"
dry_run=0
[ "${1:-}" = "--dry-run" ] && dry_run=1

busy_reason() {
  pgrep -f ssm-session-worker >/dev/null 2>&1 && { echo "SSM session"; return; }
  # Match the psplat CLI (".../bin/psplat analyze", "python -m psplat") and the
  # trainers, but not this script, whose own path contains "psplat".
  pgrep -f 'bin/psplat( |$)|-m psplat|simple_trainer|msplat' >/dev/null 2>&1 \
    && { echo "psplat or trainer process"; return; }
  [ -n "$(who 2>/dev/null)" ] && { echo "logged-in user"; return; }
  echo ""
}

now="$(date +%s)"
reason="$(busy_reason)"
if [ -n "$reason" ]; then
  rm -f "$state"
  echo "busy: $reason"
  exit 0
fi
[ -f "$state" ] || echo "$now" > "$state"
since="$(cat "$state")"
idle_min=$(( (now - since) / 60 ))
echo "idle for $idle_min min (limit $limit_min)"
if [ "$idle_min" -ge "$limit_min" ]; then
  if [ "$dry_run" = "1" ]; then
    echo "would stop now"
  else
    rm -f "$state"
    logger -t psplat "idle for $idle_min min; stopping"
    shutdown -h now
  fi
fi
