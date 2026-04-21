#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="hermes-dashboard.service"
SERVICE_DST="/etc/systemd/system/$SERVICE_NAME"
REAL_USER="${SUDO_USER:-$(id -un)}"
REAL_HOME="$(getent passwd "$REAL_USER" | cut -d: -f6)"
DASHBOARD_DST="${HERMES_HOME:-$REAL_HOME/.hermes}/hermes-dashboard.py"

run_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

run_root systemctl disable --now "$SERVICE_NAME" 2>/dev/null || true
run_root rm -f "$SERVICE_DST"
run_root systemctl daemon-reload

if [[ -f "$DASHBOARD_DST" ]]; then
  rm -f "$DASHBOARD_DST"
fi

echo "Hermes Agent Dashboard removed."
echo "User data (state.db, sessions) was left untouched."
