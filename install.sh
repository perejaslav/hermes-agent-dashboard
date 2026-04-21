#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_USER="${SUDO_USER:-$(id -un)}"
REAL_HOME="$(getent passwd "$REAL_USER" | cut -d: -f6)"
HERMES_HOME="${HERMES_HOME:-$REAL_HOME/.hermes}"
DASHBOARD_SRC="$SCRIPT_DIR/hermes-dashboard.py"
DASHBOARD_DST="$HERMES_HOME/hermes-dashboard.py"
UNIT_TEMPLATE="$SCRIPT_DIR/hermes-dashboard.service"
SERVICE_DST="/etc/systemd/system/hermes-dashboard.service"

run_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

find_python() {
  if [[ -n "${HERMES_PYTHON:-}" && -x "${HERMES_PYTHON:-}" ]]; then
    printf '%s\n' "$HERMES_PYTHON"
    return 0
  fi

  local candidates=(
    "$HERMES_HOME/hermes-agent/venv/bin/python3"
    "$HOME/.hermes/hermes-agent/venv/bin/python3"
    "/root/.hermes/hermes-agent/venv/bin/python3"
    "$(command -v python3 || true)"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  echo "ERROR: cannot find a Python interpreter for Hermes Agent." >&2
  echo "Set HERMES_PYTHON=/path/to/python3 and retry." >&2
  return 1
}

PYTHON_BIN="$(find_python)"
GROUP_NAME="$(id -gn "$REAL_USER")"

mkdir -p "$HERMES_HOME"
cp "$DASHBOARD_SRC" "$DASHBOARD_DST"
chmod 644 "$DASHBOARD_DST"

TMP_UNIT="$(mktemp)"
sed \
  -e "s|__HERMES_USER__|$REAL_USER|g" \
  -e "s|__HERMES_GROUP__|$GROUP_NAME|g" \
  -e "s|__HERMES_HOME__|$HERMES_HOME|g" \
  -e "s|__HERMES_PYTHON__|$PYTHON_BIN|g" \
  "$UNIT_TEMPLATE" > "$TMP_UNIT"

run_root install -m 644 "$TMP_UNIT" "$SERVICE_DST"
run_root systemctl daemon-reload
run_root systemctl enable --now hermes-dashboard.service
rm -f "$TMP_UNIT"

echo "Hermes Agent Dashboard installed."
echo "Dashboard file: $DASHBOARD_DST"
echo "Service: hermes-dashboard.service"
echo "Port: 8420"
echo "Use: sudo systemctl status hermes-dashboard"
