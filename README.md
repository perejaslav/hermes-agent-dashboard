# Hermes Agent Dashboard

Hermes Agent Dashboard is a small, self-contained dashboard for an existing Hermes Agent installation.
It shows session analytics, token usage, model distribution, tools, delegates, and other live metrics.

## What this repo contains

- `hermes-dashboard.py` — the dashboard server and embedded UI
- `install.sh` — installs the dashboard into an existing Hermes Agent setup
- `uninstall.sh` — removes the installed dashboard files and systemd service
- `hermes-dashboard.service` — systemd unit template used by the installer
- `README.ru.md` — Russian documentation

## Requirements

This dashboard is meant for someone who already has Hermes Agent installed and running locally.
It expects a Hermes-style environment such as:

- `~/.hermes/state.db`
- `~/.hermes/sessions/`
- Hermes Agent Python environment available on the machine

The dashboard does **not** require a separate backend service or database.
It reads the Hermes local state directly.

## Quick install

```bash
git clone https://github.com/perejaslav/hermes-agent-dashboard.git
cd hermes-agent-dashboard
sudo ./install.sh
```

If Hermes is installed in a non-default Python path, provide it explicitly:

```bash
HERMES_PYTHON=/path/to/hermes-agent/venv/bin/python3 sudo ./install.sh
```

## After installation

- service name: `hermes-dashboard`
- port: `8420`
- URL: `http://localhost:8420`

Useful commands:

```bash
sudo systemctl status hermes-dashboard
sudo systemctl restart hermes-dashboard
sudo journalctl -u hermes-dashboard -n 50
```

## Uninstall

```bash
sudo ./uninstall.sh
```

This removes the dashboard file and the systemd unit, but leaves Hermes data (`state.db`, sessions) untouched.

## Notes

- All localization is client-side; no i18n library is needed.
- The dashboard is a single Python file with embedded HTML/CSS/JS.
- It uses Hermes Agent session data from the local machine only.
