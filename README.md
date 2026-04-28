# Hermes Agent Dashboard

A small, self-contained analytics dashboard for [Hermes Agent](https://github.com/nousresearch/hermes-agent). It visualizes session data, token usage, model performance, tool calls, subagent activity, and temporal patterns — all from your local Hermes installation.

![Dashboard Preview](https://raw.githubusercontent.com/perejaslav/hermes-agent-dashboard/main/screenshot.png)

## Features

| Tab | What you see |
|-----|-------------|
| **Overview** | KPI cards, model distribution, top tools, activity by day, token trends, model scorecard |
| **Sessions** | Full session table with search, sorting, pinning, duration ranking, token ranking |
| **Tools** | Tool frequency chart, correlation heatmap, hourly activity, top sessions by tool usage |
| **Subagents** | Delegate tree visualization, goals distribution, toolset usage, hourly subagent activity |
| **Trends** | Period-over-period comparison (current vs previous), delta badges, overlaid daily charts |
| **Analytics** | Activity by hour, platform distribution donut, platform × hour stacked bar chart |

### Built-in capabilities

- **Real-time updates** — WebSocket push or 30-second polling
- **Global search** — filter sessions by model, platform, tool, or filename
- **Session pinning** — pin important sessions (saved in localStorage)
- **Session drill-down** — click any session to see full timeline with roles, tool calls, and tokens
- **Alert banner** — anomaly detection (>2σ) highlights unusual metrics
- **CSV export** — download sessions, duration, or token tables
- **Keyboard shortcuts** — `R` refresh, `/` focus search, `1–5` change period
- **i18n** — Russian / English interface

## Requirements

- Existing Hermes Agent installation
- `~/.hermes/state.db` (SQLite)
- `~/.hermes/sessions/*.jsonl`
- Python 3.11+ with FastAPI + uvicorn

No separate backend or database needed — the dashboard reads Hermes local state directly.

## Quick Install

```bash
git clone https://github.com/perejaslav/hermes-agent-dashboard.git
cd hermes-agent-dashboard
sudo ./install.sh
```

If Hermes is installed in a non-default Python path:

```bash
HERMES_PYTHON=/path/to/hermes-agent/venv/bin/python3 sudo ./install.sh
```

## After Installation

- **Service:** `hermes-dashboard`
- **Port:** `8420`
- **URL:** `http://localhost:8420`

```bash
sudo systemctl status hermes-dashboard
sudo systemctl restart hermes-dashboard
sudo journalctl -u hermes-dashboard -n 50
```

## Uninstall

```bash
sudo ./uninstall.sh
```

Removes the dashboard file and systemd unit. Hermes data (`state.db`, sessions) is untouched.

## Architecture

- **Single file:** `hermes-dashboard.py` (~2,600 lines) — FastAPI backend + inline HTML/CSS/JS
- **Frontend:** vanilla JS + Chart.js 4.x (no build step)
- **Data sources:** SQLite `state.db` + JSONL session files
- **Deployment:** systemd service with auto-restart

## License

MIT
