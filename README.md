# Hermes Agent Dashboard

Hermes Agent Dashboard is a single-file FastAPI + Chart.js dashboard that shows real-time Hermes session analytics.

## What it shows

- session count and activity
- user / assistant / tool / delegate metrics
- token usage: input, output, cache read, total
- token usage by day and by model
- top tools
- sessions sorted by duration and token usage
- agent vs subagent breakdown
- bilingual UI: Russian and English

## Data sources

The dashboard reads data from local Hermes state:

- `~/.hermes/state.db` — token and cost statistics
- `~/.hermes/sessions/*.jsonl` — session transcripts and events
- legacy `sessions.json` fallback where applicable

## Features

- live updates via WebSocket
- REST API at `/api/stats`
- period filter: 1 / 3 / 7 / 14 / 30 days
- dark theme
- localStorage language persistence
- no third-party i18n library

## Run

The dashboard is started as a systemd service:

```bash
sudo systemctl status hermes-dashboard
sudo systemctl restart hermes-dashboard
```

Default port: `8420`

## Project files

- `hermes-dashboard.py` — the dashboard server and embedded UI
- `README.md` — English overview
- `README.ru.md` — Russian overview

## Notes

This dashboard is designed to be simple to maintain: one Python file, no extra dependencies, and all translation handled client-side.
