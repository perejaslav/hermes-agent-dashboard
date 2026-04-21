#!/usr/bin/env python3
"""Hermes Agent Dashboard — real-time session statistics.

Usage:
    python3 hermes-dashboard.py [--port 8420] [--bind 0.0.0.0] [--poll 5]
"""

import argparse
import json
import os
import sqlite3
import time
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

# ── Config ──────────────────────────────────────────────────────────────────

SESSIONS_DIR = Path(os.path.expanduser("~/.hermes/sessions"))
SESSIONS_META = SESSIONS_DIR / "sessions.json"
STATE_DB = Path(os.path.expanduser("~/.hermes/state.db"))
WS_CLIENTS: set[WebSocket] = set()
_last_hash: str = ""


# ── Data collection ─────────────────────────────────────────────────────────

def _scan_jsonl_files(cutoff: datetime) -> list[dict]:
    """Return list of parsed session records from JSONL files newer than cutoff."""
    results = []
    for fpath in sorted(SESSIONS_DIR.glob("*.jsonl")):
        mtime = datetime.fromtimestamp(fpath.stat().st_mtime)
        if mtime < cutoff:
            continue
        session = {
            "file": fpath.name,
            "mtime": mtime,
            "mtime_str": mtime.strftime("%Y-%m-%d %H:%M"),
            "model": "?",
            "platform": "?",
            "user_msgs": 0,
            "assistant_msgs": 0,
            "tool_calls": 0,
            "delegate_calls": 0,
            "tools_used": Counter(),
            "delegate_events": [],
            "duration_min": None,
            "first_ts": None,
            "last_ts": None,
        }
        first_ts = None
        last_ts = None
        with open(fpath, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                role = obj.get("role", "")
                if role == "session_meta":
                    session["model"] = obj.get("model", "?")
                    session["platform"] = obj.get("platform", "?")
                    continue
                ts_str = obj.get("timestamp", "")
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str)
                        if first_ts is None:
                            first_ts = ts
                        last_ts = ts
                    except (ValueError, TypeError):
                        pass
                if role == "user":
                    session["user_msgs"] += 1
                elif role == "assistant":
                    session["assistant_msgs"] += 1
                    tc = obj.get("tool_calls")
                    if isinstance(tc, list):
                        for t in tc:
                            fn = t.get("function", {}) if isinstance(t, dict) else {}
                            name = fn.get("name", "?")
                            session["tool_calls"] += 1
                            session["tools_used"][name] += 1
                            if name == "delegate_task":
                                session["delegate_calls"] += 1
                                args_str = fn.get("arguments", "")
                                goal = ""
                                n_tasks = 0
                                toolsets = []
                                if isinstance(args_str, str):
                                    try:
                                        args_obj = json.loads(args_str)
                                        goal = (args_obj.get("goal") or "")[:100]
                                        tasks = args_obj.get("tasks")
                                        n_tasks = len(tasks) if isinstance(tasks, list) else 0
                                        toolsets = args_obj.get("toolsets") or []
                                    except (json.JSONDecodeError, TypeError):
                                        pass
                                session["delegate_events"].append({
                                    "goal": goal or "(batch)",
                                    "tasks": n_tasks,
                                    "toolsets": toolsets,
                                    "timestamp": (ts_str or "?")[:19],
                                })
        if first_ts and last_ts:
            session["duration_min"] = round((last_ts - first_ts).total_seconds() / 60, 1)
        session["first_ts"] = first_ts
        session["last_ts"] = last_ts
        results.append(session)
    return results


def _load_session_meta() -> dict:
    """Load sessions.json metadata (legacy, used as fallback)."""
    if not SESSIONS_META.exists():
        return {}
    try:
        with open(SESSIONS_META, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _load_token_data_from_db(cutoff: datetime) -> list[dict]:
    """Read token usage from SQLite state.db for sessions newer than cutoff."""
    if not STATE_DB.exists():
        return []
    try:
        conn = sqlite3.connect(str(STATE_DB))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT id, model, input_tokens, output_tokens,
                   cache_read_tokens, cache_write_tokens,
                   reasoning_tokens, estimated_cost_usd,
                   cost_status, started_at, ended_at,
                   message_count, tool_call_count
            FROM sessions
            WHERE started_at >= ?
            ORDER BY started_at DESC
        """, (cutoff.timestamp(),)).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def collect_stats(days: int = 7) -> dict:
    """Build full stats dict for the API."""
    cutoff = datetime.now() - timedelta(days=days)
    sessions = _scan_jsonl_files(cutoff)
    meta = _load_session_meta()

    total_messages = Counter()
    tool_counter = Counter()
    model_counter = Counter()
    platform_counter = Counter()
    day_counter = Counter()
    total_tool_calls = 0
    total_delegates = 0
    all_delegates = []
    total_user_msgs = 0

    for s in sessions:
        total_user_msgs += s["user_msgs"]
        total_messages["user"] += s["user_msgs"]
        total_messages["assistant"] += s["assistant_msgs"]
        total_tool_calls += s["tool_calls"]
        total_delegates += s["delegate_calls"]
        model_counter[s["model"]] += 1
        platform_counter[s["platform"]] += 1
        day_counter[s["mtime_str"][:10]] += 1
        for tool, count in s["tools_used"].items():
            tool_counter[tool] += count
        for ev in s["delegate_events"]:
            ev_copy = dict(ev)
            ev_copy["session"] = s["file"]
            all_delegates.append(ev_copy)

    # Classify sessions: main agent vs subagent-only sessions
    # Subagents in Hermes don't write separate JSONL — they run inside
    # the parent session. So we classify based on delegate_calls presence:
    #   - "main" = session has delegate_calls (used subagents)
    #   - "direct" = session has no delegate_calls (agent worked directly)
    for s in sessions:
        s["agent_type"] = "main_with_subagents" if s["delegate_calls"] > 0 else "direct"

    # Aggregate subagent stats from delegate events
    subagent_tool_counter = Counter()
    subagent_tasks = 0
    for s in sessions:
        for ev in s["delegate_events"]:
            subagent_tasks += max(ev.get("tasks", 1), 1)
            for ts in (ev.get("toolsets") or []):
                subagent_tool_counter[ts] += 1

    # Sort sessions by mtime desc
    sessions.sort(key=lambda x: x["mtime"], reverse=True)

    # Top sessions
    by_duration = sorted(
        [s for s in sessions if s["duration_min"] is not None],
        key=lambda x: x["duration_min"], reverse=True,
    )
    by_tools = sorted(sessions, key=lambda x: x["tool_calls"], reverse=True)

    # Token stats from SQLite state.db (primary source)
    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_cache_write = 0
    total_reasoning = 0
    total_cost = 0.0
    cost_status = "unknown"

    db_sessions = _load_token_data_from_db(cutoff)

    # Token by model aggregation
    token_by_model: dict[str, dict] = {}
    # Token by day aggregation
    token_by_day: dict[str, dict] = {}
    # Top cost sessions
    top_cost_sessions = []

    for db_s in db_sessions:
        inp = db_s.get("input_tokens") or 0
        out = db_s.get("output_tokens") or 0
        cr = db_s.get("cache_read_tokens") or 0
        cw = db_s.get("cache_write_tokens") or 0
        reas = db_s.get("reasoning_tokens") or 0
        cost = db_s.get("estimated_cost_usd") or 0.0
        model = db_s.get("model") or "unknown"
        started = db_s.get("started_at") or 0

        total_input += inp
        total_output += out
        total_cache_read += cr
        total_cache_write += cw
        total_reasoning += reas
        total_cost += cost

        # Track cost status — prefer "tracked" if any session has data
        cs = db_s.get("cost_status") or "unknown"
        if cs not in ("unknown", None) and cost_status == "unknown":
            cost_status = cs

        # Aggregate by model
        if model not in token_by_model:
            token_by_model[model] = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "reasoning": 0, "cost": 0.0, "sessions": 0}
        token_by_model[model]["input"] += inp
        token_by_model[model]["output"] += out
        token_by_model[model]["cache_read"] += cr
        token_by_model[model]["cache_write"] += cw
        token_by_model[model]["reasoning"] += reas
        token_by_model[model]["cost"] += cost
        token_by_model[model]["sessions"] += 1

        # Aggregate by day
        if started:
            day_str = datetime.fromtimestamp(started).strftime("%Y-%m-%d")
        else:
            day_str = "unknown"
        if day_str not in token_by_day:
            token_by_day[day_str] = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "cost": 0.0}
        token_by_day[day_str]["input"] += inp
        token_by_day[day_str]["output"] += out
        token_by_day[day_str]["cache_read"] += cr
        token_by_day[day_str]["cache_write"] += cw
        token_by_day[day_str]["cost"] += cost

    # Top cost sessions
    top_cost_sessions = sorted(
        [
            {
                "session_id": s["id"],
                "model": s.get("model") or "?",
                "input_tokens": s.get("input_tokens") or 0,
                "output_tokens": s.get("output_tokens") or 0,
                "cache_read_tokens": s.get("cache_read_tokens") or 0,
                "estimated_cost_usd": round(s.get("estimated_cost_usd") or 0, 4),
                "started_at": datetime.fromtimestamp(s["started_at"]).strftime("%Y-%m-%d %H:%M") if s.get("started_at") else "?",
            }
            for s in db_sessions if (s.get("estimated_cost_usd") or 0) > 0 or (s.get("input_tokens") or 0) > 0
        ],
        key=lambda x: x["input_tokens"],
        reverse=True,
    )[:15]

    # Build a lookup dict from DB session IDs to token data
    db_token_lookup = {}
    for s in db_sessions:
        db_token_lookup[s["id"]] = {
            "input_tokens": s.get("input_tokens") or 0,
            "output_tokens": s.get("output_tokens") or 0,
            "cache_read_tokens": s.get("cache_read_tokens") or 0,
            "cache_write_tokens": s.get("cache_write_tokens") or 0,
            "estimated_cost_usd": round(s.get("estimated_cost_usd") or 0, 4),
        }

    return {
        "generated_at": datetime.now().isoformat(),
        "period_days": days,
        "summary": {
            "sessions": len(sessions),
            "user_messages": total_user_msgs,
            "assistant_responses": total_messages["assistant"],
            "tool_calls": total_tool_calls,
            "delegate_calls": total_delegates,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cache_read_tokens": total_cache_read,
            "total_cache_write_tokens": total_cache_write,
            "total_reasoning_tokens": total_reasoning,
            "total_tokens": total_input + total_output,
            "estimated_cost_usd": round(total_cost, 4),
            "cost_status": cost_status,
        },
        "by_model": dict(model_counter.most_common()),
        "by_platform": dict(platform_counter.most_common()),
        "by_day": dict(sorted(day_counter.items())),
        "top_tools": dict(tool_counter.most_common(20)),
        "token_by_model": token_by_model,
        "token_by_day": dict(sorted(token_by_day.items())),
        "top_cost_sessions": top_cost_sessions,
        "top_duration": [
            {
                "file": s["file"],
                "model": s["model"],
                "duration_min": s["duration_min"],
                "user_msgs": s["user_msgs"],
                "tool_calls": s["tool_calls"],
                "delegate_calls": s["delegate_calls"],
                "mtime": s["mtime_str"],
                "top_tools": dict(s["tools_used"].most_common(5)),
                "input_tokens": db_token_lookup.get(s["file"].replace(".jsonl", ""), {}).get("input_tokens", 0),
                "output_tokens": db_token_lookup.get(s["file"].replace(".jsonl", ""), {}).get("output_tokens", 0),
            }
            for s in by_duration[:15]
        ],
        "top_tools_sessions": [
            {
                "file": s["file"],
                "model": s["model"],
                "user_msgs": s["user_msgs"],
                "tool_calls": s["tool_calls"],
                "delegate_calls": s["delegate_calls"],
                "mtime": s["mtime_str"],
                "top_tools": dict(s["tools_used"].most_common(5)),
            }
            for s in by_tools[:15] if s["tool_calls"] > 0
        ],
        "delegates": all_delegates,
        "sessions": [
            {
                "file": s["file"],
                "model": s["model"],
                "platform": s["platform"],
                "user_msgs": s["user_msgs"],
                "assistant_msgs": s["assistant_msgs"],
                "tool_calls": s["tool_calls"],
                "delegate_calls": s["delegate_calls"],
                "duration_min": s["duration_min"],
                "mtime": s["mtime_str"],
                "agent_type": s["agent_type"],
                "input_tokens": db_token_lookup.get(s["file"].replace(".jsonl", ""), {}).get("input_tokens", 0),
                "output_tokens": db_token_lookup.get(s["file"].replace(".jsonl", ""), {}).get("output_tokens", 0),
                "cache_read_tokens": db_token_lookup.get(s["file"].replace(".jsonl", ""), {}).get("cache_read_tokens", 0),
                "estimated_cost_usd": db_token_lookup.get(s["file"].replace(".jsonl", ""), {}).get("estimated_cost_usd", 0),
            }
            for s in sessions
        ],
        "agent_breakdown": {
            "total_sessions": len(sessions),
            "direct_sessions": sum(1 for s in sessions if s["agent_type"] == "direct"),
            "sessions_with_subagents": sum(1 for s in sessions if s["agent_type"] == "main_with_subagents"),
            "total_delegate_calls": total_delegates,
            "total_subagent_tasks": subagent_tasks,
            "subagent_toolsets": dict(subagent_tool_counter.most_common()),
        },
    }


# ── File change detection ───────────────────────────────────────────────────

def _dir_hash() -> str:
    """Quick hash of sessions directory state for change detection."""
    parts = []
    for f in sorted(SESSIONS_DIR.glob("*.jsonl")):
        parts.append(f"{f.name}:{f.stat().st_mtime}:{f.stat().st_size}")
    for f in sorted(SESSIONS_DIR.glob("*.json")):
        parts.append(f"{f.name}:{f.stat().st_mtime}:{f.stat().st_size}")
    # Also track state.db for token data changes
    if STATE_DB.exists():
        parts.append(f"state.db:{STATE_DB.stat().st_mtime}:{STATE_DB.stat().st_size}")
    return "|".join(parts[-50:])  # last 50 entries enough


# ── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI(title="Hermes Dashboard", docs_url=None, redoc_url=None)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(DASHBOARD_HTML)


@app.get("/api/stats")
async def api_stats(days: int = 7):
    return collect_stats(days=days)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    WS_CLIENTS.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        WS_CLIENTS.discard(ws)


async def _push_updates(poll_interval: int, days: int):
    """Background task: detect changes and push to WebSocket clients."""
    global _last_hash
    _last_hash = _dir_hash()
    while True:
        await asyncio.sleep(poll_interval)
        current_hash = _dir_hash()
        if current_hash != _last_hash:
            _last_hash = current_hash
            stats = collect_stats(days=days)
            payload = json.dumps(stats, default=str, ensure_ascii=False)
            dead = set()
            for client in WS_CLIENTS:
                try:
                    await client.send_text(payload)
                except Exception:
                    dead.add(client)
            WS_CLIENTS -= dead


@app.on_event("startup")
async def startup():
    import asyncio
    # Will be overridden by lifespan in run()
    pass


# ── HTML Dashboard ──────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hermes Agent Dashboard</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA2NCA2NCI+CiAgPGRlZnM+CiAgICA8bGluZWFyR3JhZGllbnQgaWQ9ImciIHgxPSIwJSIgeTE9IjAlIiB4Mj0iMTAwJSIgeTI9IjEwMCUiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjNmM4Y2ZmIi8+CiAgICAgIDxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iI2E3OGJmYSIvPgogICAgPC9saW5lYXJHcmFkaWVudD4KICAgIDxmaWx0ZXIgaWQ9InMiPgogICAgICA8ZmVEcm9wU2hhZG93IGR4PSIwIiBkeT0iMSIgc3RkRGV2aWF0aW9uPSIyIiBmbG9vZC1jb2xvcj0iIzZjOGNmZiIgZmxvb2Qtb3BhY2l0eT0iMC41Ii8+CiAgICA8L2ZpbHRlcj4KICA8L2RlZnM+CiAgPGNpcmNsZSBjeD0iMzIiIGN5PSIzMiIgcj0iMzAiIGZpbGw9IiMxYTFkMjciIHN0cm9rZT0iIzJlMzM0OCIgc3Ryb2tlLXdpZHRoPSIyIi8+CiAgPHBhdGggZD0iTTM2IDhMMTggMzZoMTJsLTQgMjAgMjAtMjhIMzR6IiBmaWxsPSJ1cmwoI2cpIiBmaWx0ZXI9InVybCgjcykiLz4KPC9zdmc+">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg: #0f1117;
  --surface: #1a1d27;
  --surface2: #242836;
  --border: #2e3348;
  --text: #e1e4ed;
  --text2: #8b90a5;
  --accent: #6c8cff;
  --accent2: #a78bfa;
  --green: #34d399;
  --orange: #fb923c;
  --red: #f87171;
  --blue: #60a5fa;
  --pink: #f472b6;
  --yellow: #fbbf24;
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
  font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', monospace;
  background: var(--bg); color: var(--text);
  line-height: 1.5; font-size: 13px;
}
.container { max-width: 1440px; margin: 0 auto; padding: 20px; }

/* Header */
.header {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 24px; padding-bottom: 16px;
  border-bottom: 1px solid var(--border);
}
.header h1 { font-size: 20px; font-weight: 600; }
.header h1 span { color: var(--accent); }
.status {
  display: flex; align-items: center; gap: 8px;
  font-size: 12px; color: var(--text2);
}
.status-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--green); display: inline-block;
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
.status-dot.offline { background: var(--red); animation: none; }

/* Controls */
.controls {
  display: flex; gap: 12px; margin-bottom: 20px; align-items: center;
}
.controls select, .controls button {
  background: var(--surface2); color: var(--text); border: 1px solid var(--border);
  padding: 6px 12px; border-radius: 6px; font-family: inherit; font-size: 12px;
  cursor: pointer;
}
.controls button:hover { border-color: var(--accent); }
.lang-btn { font-size: 16px; padding: 2px 8px !important; }
.flag-icon { display:inline-block; width:18px; height:12px; vertical-align:middle; border-radius:2px; border:1px solid rgba(255,255,255,.15); box-sizing:border-box; }
.flag-us {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 19 10'%3E%3Crect width='19' height='10' fill='%23fff'/%3E%3Cg fill='%23b22234'%3E%3Crect y='0' width='19' height='0.77'/%3E%3Crect y='1.54' width='19' height='0.77'/%3E%3Crect y='3.08' width='19' height='0.77'/%3E%3Crect y='4.62' width='19' height='0.77'/%3E%3Crect y='6.16' width='19' height='0.77'/%3E%3Crect y='7.69' width='19' height='0.77'/%3E%3Crect y='9.23' width='19' height='0.77'/%3E%3C/g%3E%3Crect width='8' height='5.38' fill='%233C3B6E'/%3E%3Cg fill='%23fff'%3E%3Cpolygon points='0.7,0.5 0.82,0.82 1.16,0.84 0.89,1.05 0.98,1.37 0.7,1.18 0.42,1.37 0.51,1.05 0.24,0.84 0.58,0.82'/%3E%3Cpolygon points='2.1,0.5 2.22,0.82 2.56,0.84 2.29,1.05 2.38,1.37 2.1,1.18 1.82,1.37 1.91,1.05 1.64,0.84 1.98,0.82'/%3E%3Cpolygon points='3.5,0.5 3.62,0.82 3.96,0.84 3.69,1.05 3.78,1.37 3.5,1.18 3.22,1.37 3.31,1.05 3.04,0.84 3.38,0.82'/%3E%3Cpolygon points='4.9,0.5 5.02,0.82 5.36,0.84 5.09,1.05 5.18,1.37 4.9,1.18 4.62,1.37 4.71,1.05 4.44,0.84 4.78,0.82'/%3E%3Cpolygon points='6.3,0.5 6.42,0.82 6.76,0.84 6.49,1.05 6.58,1.37 6.3,1.18 6.02,1.37 6.11,1.05 5.84,0.84 6.18,0.82'/%3E%3Cpolygon points='1.4,1.8 1.52,2.12 1.86,2.14 1.59,2.35 1.68,2.67 1.4,2.48 1.12,2.67 1.21,2.35 0.94,2.14 1.28,2.12'/%3E%3Cpolygon points='2.8,1.8 2.92,2.12 3.26,2.14 2.99,2.35 3.08,2.67 2.8,2.48 2.52,2.67 2.61,2.35 2.34,2.14 2.68,2.12'/%3E%3Cpolygon points='4.2,1.8 4.32,2.12 4.66,2.14 4.39,2.35 4.48,2.67 4.2,2.48 3.92,2.67 4.01,2.35 3.74,2.14 4.08,2.12'/%3E%3Cpolygon points='5.6,1.8 5.72,2.12 6.06,2.14 5.79,2.35 5.88,2.67 5.6,2.48 5.32,2.67 5.41,2.35 5.14,2.14 5.48,2.12'/%3E%3Cpolygon points='7.0,1.8 7.12,2.12 7.46,2.14 7.19,2.35 7.28,2.67 7.0,2.48 6.72,2.67 6.81,2.35 6.54,2.14 6.88,2.12'/%3E%3C/g%3E%3C/svg%3E");
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
}
.flag-ru { background: linear-gradient(to bottom, #fff 0 33.33%, #0c47b7 33.33% 66.66%, #d52b1e 66.66% 100%); }
.last-update { font-size: 11px; color: var(--text2); margin-left: auto; display:flex; align-items:center; gap:8px; }

/* KPI Cards */
.kpi-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px; margin-bottom: 24px;
}
.kpi-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 16px 20px;
  transition: border-color 0.2s;
}
.kpi-card:hover { border-color: var(--accent); }
.kpi-label { font-size: 11px; color: var(--text2); text-transform: uppercase; letter-spacing: 0.5px; }
.kpi-value { font-size: 28px; font-weight: 700; margin-top: 4px; }
.kpi-sub { font-size: 11px; color: var(--text2); margin-top: 2px; }

/* Charts grid */
.charts-grid {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 16px; margin-bottom: 24px;
}
.chart-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 16px;
}
.chart-card.full { grid-column: 1 / -1; }
.chart-title {
  font-size: 13px; font-weight: 600; margin-bottom: 12px;
  color: var(--text);
}

/* Tables */
.table-section {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; margin-bottom: 24px; overflow: hidden;
}
.table-header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 12px 16px; border-bottom: 1px solid var(--border);
}
.table-header h3 { font-size: 13px; font-weight: 600; }
.filter-input {
  background: var(--surface2); color: var(--text); border: 1px solid var(--border);
  padding: 4px 10px; border-radius: 4px; font-family: inherit; font-size: 11px;
  width: 200px;
}
.filter-input::placeholder { color: var(--text2); }
table { width: 100%; border-collapse: collapse; }
th {
  text-align: left; padding: 8px 16px; font-size: 11px;
  color: var(--text2); text-transform: uppercase; letter-spacing: 0.3px;
  background: var(--surface2); border-bottom: 1px solid var(--border);
  cursor: pointer; user-select: none;
  white-space: nowrap;
}
th:hover { color: var(--accent); }
th .sort-arrow { margin-left: 4px; opacity: 0.4; }
th.sorted .sort-arrow { opacity: 1; color: var(--accent); }
td {
  padding: 6px 16px; border-bottom: 1px solid var(--border);
  font-size: 12px; white-space: nowrap;
}
tr:hover td { background: var(--surface2); }
.badge {
  display: inline-block; padding: 2px 8px; border-radius: 4px;
  font-size: 11px; font-weight: 500;
}
.badge-model { background: #1e3a5f; color: var(--blue); }
.badge-platform { background: #1e3f2f; color: var(--green); }
.badge-delegate { background: #3f2f1e; color: var(--orange); }
.badge-subagent { background: #2a1f3f; color: var(--accent2); }
.badge-direct { background: #1e2f1e; color: var(--green); }
.num { font-variant-numeric: tabular-nums; }

/* Delegate events */
.delegate-goal {
  max-width: 400px; overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap;
}

/* Responsive */
@media (max-width: 900px) {
  .charts-grid { grid-template-columns: 1fr; }
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 500px) {
  .kpi-grid { grid-template-columns: 1fr; }
  .container { padding: 12px; }
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>⚡ <span>Hermes</span> Agent Dashboard</h1>
    <div class="status">
      <span class="status-dot" id="ws-dot"></span>
      <span id="ws-label">connecting...</span>
    </div>
  </div>

  <div class="controls">
    <select id="period-select">
      <option value="1">1 день</option>
      <option value="3">3 дня</option>
      <option value="7" selected>7 дней</option>
      <option value="14">14 дней</option>
      <option value="30">30 дней</option>
    </select>
    <button onclick="refresh()">↻ <span data-i18n="btn_refresh">Обновить</span></button>
    <span class="last-update" id="last-update">
      <span id="last-update-text"></span>
      <button class="lang-btn" id="lang-btn" onclick="toggleLang()"></button>
    </span>
  </div>

  <!-- KPI -->
  <div class="kpi-grid" id="kpi-grid"></div>

  <!-- Agent / Subagent breakdown -->
  <div class="table-section" style="margin-bottom:24px;">
    <div class="table-header">
      <h3 data-i18n="sec_agent">🤖 Агент vs Субагенты</h3>
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;padding:16px;">
      <div style="background:var(--surface2);border-radius:8px;padding:14px;">
        <div style="font-size:11px;color:var(--text2);text-transform:uppercase;letter-spacing:.5px;" data-i18n="ab_main">Основной агент (всего сессий)</div>
        <div id="ab-total" style="font-size:24px;font-weight:700;color:var(--accent);margin-top:4px;">—</div>
      </div>
      <div style="background:var(--surface2);border-radius:8px;padding:14px;">
        <div style="font-size:11px;color:var(--text2);text-transform:uppercase;letter-spacing:.5px;" data-i18n="ab_direct">Прямая работа (без субагентов)</div>
        <div id="ab-direct" style="font-size:24px;font-weight:700;color:var(--green);margin-top:4px;">—</div>
      </div>
      <div style="background:var(--surface2);border-radius:8px;padding:14px;">
        <div style="font-size:11px;color:var(--text2);text-transform:uppercase;letter-spacing:.5px;" data-i18n="ab_with_sa">Сессии с субагентами</div>
        <div id="ab-with-sa" style="font-size:24px;font-weight:700;color:var(--accent2);margin-top:4px;">—</div>
      </div>
      <div style="background:var(--surface2);border-radius:8px;padding:14px;">
        <div style="font-size:11px;color:var(--text2);text-transform:uppercase;letter-spacing:.5px;" data-i18n="ab_deleg">Delegate-вызовов</div>
        <div id="ab-deleg" style="font-size:24px;font-weight:700;color:var(--orange);margin-top:4px;">—</div>
      </div>
      <div style="background:var(--surface2);border-radius:8px;padding:14px;">
        <div style="font-size:11px;color:var(--text2);text-transform:uppercase;letter-spacing:.5px;" data-i18n="ab_tasks">Субагент-задач (всего)</div>
        <div id="ab-tasks" style="font-size:24px;font-weight:700;color:var(--pink);margin-top:4px;">—</div>
      </div>
      <div style="background:var(--surface2);border-radius:8px;padding:14px;">
        <div style="font-size:11px;color:var(--text2);text-transform:uppercase;letter-spacing:.5px;" data-i18n="ab_toolsets">Toolsets субагентов</div>
        <div id="ab-ts" style="font-size:13px;font-weight:600;color:var(--text);margin-top:8px;">—</div>
      </div>
    </div>
  </div>

  <!-- Charts -->
  <div class="charts-grid">
    <div class="chart-card">
      <div class="chart-title" data-i18n="sec_models">Распределение по моделям</div>
      <canvas id="chart-models"></canvas>
    </div>
    <div class="chart-card">
      <div class="chart-title" data-i18n="sec_tools">Топ инструментов</div>
      <canvas id="chart-tools"></canvas>
    </div>
    <div class="chart-card full">
      <div class="chart-title" data-i18n="sec_days">Активность по дням (сессии)</div>
      <canvas id="chart-days"></canvas>
    </div>
    <div class="chart-card full">
      <div class="chart-title" data-i18n="sec_token_days">Токены по дням</div>
      <canvas id="chart-token-days"></canvas>
    </div>
    <div class="chart-card full">
      <div class="chart-title" data-i18n="sec_token_models">Токены по модели</div>
      <canvas id="chart-token-models"></canvas>
    </div>
  </div>

  <!-- Sessions table -->
  <div class="table-section">
    <div class="table-header">
      <h3 data-i18n="sec_sessions">📋 Все сессии</h3>
      <input type="text" class="filter-input" id="session-filter" placeholder="Фильтр по модели..." data-i18n-ph="filter_ph">
    </div>
    <div style="overflow-x:auto;">
      <table id="sessions-table">
        <thead>
          <tr>
            <th data-key="mtime" data-i18n="th_date">Дата <span class="sort-arrow">▼</span></th>
            <th data-i18n="th_type">Тип</th>
            <th data-key="model" data-i18n="th_model">Модель <span class="sort-arrow">▼</span></th>
            <th data-key="platform" data-i18n="th_platform">Платформа</th>
            <th data-key="user_msgs">User <span class="sort-arrow">▼</span></th>
            <th data-key="assistant_msgs">Asst <span class="sort-arrow">▼</span></th>
            <th data-key="tool_calls">Tools <span class="sort-arrow">▼</span></th>
            <th data-key="delegate_calls">Deleg <span class="sort-arrow">▼</span></th>
            <th data-key="input_tokens">Input <span class="sort-arrow">▼</span></th>
            <th data-key="output_tokens">Output <span class="sort-arrow">▼</span></th>
            <th data-key="duration_min" data-i18n="th_dur">Длит. <span class="sort-arrow">▼</span></th>
          </tr>
        </thead>
        <tbody id="sessions-tbody"></tbody>
      </table>
    </div>
  </div>

  <!-- Delegates table -->
  <div class="table-section">
    <div class="table-header">
      <h3 data-i18n="sec_delegates">🔀 Delegate-вызовы (субагенты)</h3>
    </div>
    <div style="overflow-x:auto;">
      <table id="delegates-table">
        <thead>
          <tr>
            <th data-i18n="th_time">Время</th>
            <th data-i18n="th_goal">Цель / Задачи</th>
            <th>Tasks</th>
            <th>Toolsets</th>
          </tr>
        </thead>
        <tbody id="delegates-tbody"></tbody>
      </table>
    </div>
  </div>

  <!-- Top sessions by duration -->
  <div class="table-section">
    <div class="table-header">
      <h3 data-i18n="sec_duration">⏱ Топ сессий по длительности</h3>
    </div>
    <div style="overflow-x:auto;">
      <table id="duration-table">
        <thead>
          <tr>
            <th data-i18n="th_dur_min">Длит. (мин)</th>
            <th data-i18n="th_model">Модель</th>
            <th>Tools</th>
            <th>Deleg</th>
            <th>User</th>
            <th>Input</th>
            <th>Output</th>
            <th data-i18n="th_date">Дата</th>
          </tr>
        </thead>
        <tbody id="duration-tbody"></tbody>
      </table>
    </div>
  </div>

  <!-- Top sessions by token usage -->
  <div class="table-section">
    <div class="table-header">
      <h3 data-i18n="sec_tokens">📊 Топ сессий по токенам</h3>
    </div>
    <div style="overflow-x:auto;">
      <table id="token-table">
        <thead>
          <tr>
            <th data-i18n="th_model">Модель</th>
            <th data-i18n="th_input_t">Input токены</th>
            <th data-i18n="th_output_t">Output токены</th>
            <th data-i18n="th_cache">Cache read</th>
            <th data-i18n="th_cost">Стоимость</th>
            <th data-i18n="th_date">Дата</th>
          </tr>
        </thead>
        <tbody id="token-tbody"></tbody>
      </table>
    </div>
  </div>

</div>

<script>
// ── State ──────────────────────────────────────────────────────────────
let stats = null;
let ws = null;
let chartModels = null, chartTools = null, chartDays = null, chartTokenDays = null, chartTokenModels = null;
let sortKey = 'mtime', sortDir = -1;

const COLORS = [
  '#6c8cff','#a78bfa','#34d399','#fb923c','#f87171',
  '#60a5fa','#f472b6','#fbbf24','#2dd4bf','#818cf8',
  '#e879f9','#4ade80','#facc15','#38bdf8','#fb7185',
];

// ── i18n ─────────────────────────────────────────────────────────────
let lang = localStorage.getItem('dash-lang') || 'ru';
function _t(key) {
  const T = {
    ru: {
      period_1:'1 день', period_3:'3 дня', period_7:'7 дней', period_14:'14 дней', period_30:'30 дней',
      btn_refresh:'Обновить', filter_ph:'Фильтр по модели...',
      sec_agent:'🤖 Агент vs Субагенты', sec_models:'Распределение по моделям', sec_tools:'Топ инструментов',
      sec_days:'Активность по дням (сессии)', sec_token_days:'Токены по дням', sec_token_models:'Токены по модели',
      sec_sessions:'📋 Все сессии', sec_delegates:'🔀 Delegate-вызовы (субагенты)',
      sec_duration:'⏱ Топ сессий по длительности', sec_tokens:'📊 Топ сессий по токенам',
      ab_main:'Основной агент (всего сессий)', ab_direct:'Прямая работа (без субагентов)',
      ab_with_sa:'Сессии с субагентами', ab_deleg:'Delegate-вызовов', ab_tasks:'Субагент-задач (всего)',
      ab_toolsets:'Toolsets субагентов',
      th_date:'Дата', th_type:'Тип', th_model:'Модель', th_platform:'Платформа', th_dur:'Длит.',
      th_time:'Время', th_goal:'Цель / Задачи', th_dur_min:'Длит. (мин)',
      th_input_t:'Input токены', th_output_t:'Output токены', th_cache:'Cache read', th_cost:'Стоимость',
      updated:'Обновлено: ',
      kpi_sessions:'Сессии', kpi_user:'User сообщения', kpi_responses:'Ответы',
      kpi_tools:'Tool-вызовы', kpi_deleg:'Delegate-вызовы', kpi_models:'Моделей',
      kpi_input:'Input токены', kpi_output:'Output токены', kpi_cache:'Cache read',
      kpi_total:'Итого токены', kpi_cost:'Стоимость',
      sub_unique:'уникальных', sub_incoming:'входящих', sub_outgoing:'исходящих',
      sub_cache:'из кэша', sub_io:'input + output', sub_included:'вкл. в план',
      badge_sa:'+ субагенты', badge_direct:'прямой', unit_min:' мин',
      no_delegates:'Нет delegate-вызовов за выбранный период',
      no_data:'нет данных', no_token_data:'Нет данных о токенах за выбранный период',
      chart_sessions:'Сессии',
    },
    en: {
      period_1:'1 day', period_3:'3 days', period_7:'7 days', period_14:'14 days', period_30:'30 days',
      btn_refresh:'Refresh', filter_ph:'Filter by model...',
      sec_agent:'🤖 Agent vs Subagents', sec_models:'Distribution by models', sec_tools:'Top tools',
      sec_days:'Activity by day (sessions)', sec_token_days:'Tokens by day', sec_token_models:'Tokens by model',
      sec_sessions:'📋 All sessions', sec_delegates:'🔀 Delegate calls (subagents)',
      sec_duration:'⏱ Top sessions by duration', sec_tokens:'📊 Top sessions by tokens',
      ab_main:'Main agent (total sessions)', ab_direct:'Direct work (no subagents)',
      ab_with_sa:'Sessions with subagents', ab_deleg:'Delegate calls', ab_tasks:'Subagent tasks (total)',
      ab_toolsets:'Subagent toolsets',
      th_date:'Date', th_type:'Type', th_model:'Model', th_platform:'Platform', th_dur:'Dur.',
      th_time:'Time', th_goal:'Goal / Tasks', th_dur_min:'Dur. (min)',
      th_input_t:'Input tokens', th_output_t:'Output tokens', th_cache:'Cache read', th_cost:'Cost',
      updated:'Updated: ',
      kpi_sessions:'Sessions', kpi_user:'User messages', kpi_responses:'Responses',
      kpi_tools:'Tool calls', kpi_deleg:'Delegate calls', kpi_models:'Models',
      kpi_input:'Input tokens', kpi_output:'Output tokens', kpi_cache:'Cache read',
      kpi_total:'Total tokens', kpi_cost:'Cost',
      sub_unique:'unique', sub_incoming:'incoming', sub_outgoing:'outgoing',
      sub_cache:'from cache', sub_io:'input + output', sub_included:'included in plan',
      badge_sa:'+ subagents', badge_direct:'direct', unit_min:' min',
      no_delegates:'No delegate calls in selected period',
      no_data:'no data', no_token_data:'No token data in selected period',
      chart_sessions:'Sessions',
    }
  };
  return (T[lang] && T[lang][key]) || key;
}

function flagHtml(code) {
  return code === 'ru'
    ? '<span class="flag-icon flag-ru" aria-hidden="true"></span>'
    : '<span class="flag-icon flag-us" aria-hidden="true"></span>';
}

function updateLangButton() {
  const btn = document.getElementById('lang-btn');
  btn.innerHTML = flagHtml(lang);
  btn.title = lang === 'ru' ? 'Русский интерфейс' : 'English interface';
}

function toggleLang() {
  lang = lang === 'ru' ? 'en' : 'ru';
  localStorage.setItem('dash-lang', lang);
  updateLangButton();
  document.documentElement.lang = lang;
  applyI18n();
  render();
}

function applyI18n() {
  document.querySelectorAll('[data-i18n]').forEach(el => {
    el.textContent = _t(el.dataset.i18n);
  });
  document.querySelectorAll('[data-i18n-ph]').forEach(el => {
    el.placeholder = _t(el.dataset.i18nPh);
  });
  const ps = document.getElementById('period-select');
  ps.options[0].text = _t('period_1');
  ps.options[1].text = _t('period_3');
  ps.options[2].text = _t('period_7');
  ps.options[3].text = _t('period_14');
  ps.options[4].text = _t('period_30');
}

// ── Fetch ──────────────────────────────────────────────────────────────
async function refresh() {
  const days = document.getElementById('period-select').value;
  try {
    const r = await fetch('/api/stats?days=' + days);
    stats = await r.json();
    render();
  } catch(e) { console.error(e); }
}

// ── WebSocket ──────────────────────────────────────────────────────────
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(proto + '//' + location.host + '/ws');
  ws.onopen = () => {
    document.getElementById('ws-dot').classList.remove('offline');
    document.getElementById('ws-label').textContent = 'live';
  };
  ws.onmessage = (e) => {
    try { stats = JSON.parse(e.data); render(); } catch(err) {}
  };
  ws.onclose = () => {
    document.getElementById('ws-dot').classList.add('offline');
    document.getElementById('ws-label').textContent = 'reconnecting...';
    setTimeout(connectWS, 3000);
  };
}

// ── Render ─────────────────────────────────────────────────────────────
function render() {
  if (!stats) return;
  const s = stats.summary;
  document.getElementById('last-update-text').textContent =
    _t('updated') + new Date(stats.generated_at).toLocaleTimeString(lang === 'ru' ? 'ru-RU' : 'en-US');

  // KPI
  document.getElementById('kpi-grid').innerHTML = [
    kpi(_t('kpi_sessions'), s.sessions, '', '#6c8cff'),
    kpi(_t('kpi_user'), s.user_messages, '', '#a78bfa'),
    kpi(_t('kpi_responses'), s.assistant_responses, '', '#34d399'),
    kpi(_t('kpi_tools'), s.tool_calls, '', '#fb923c'),
    kpi(_t('kpi_deleg'), s.delegate_calls, '', '#f87171'),
    kpi(_t('kpi_models'), Object.keys(stats.by_model).length, _t('sub_unique'), '#60a5fa'),
    kpi(_t('kpi_input'), fmt(s.total_input_tokens), _t('sub_incoming'), '#fbbf24'),
    kpi(_t('kpi_output'), fmt(s.total_output_tokens), _t('sub_outgoing'), '#2dd4bf'),
    kpi(_t('kpi_cache'), fmt(s.total_cache_read_tokens), _t('sub_cache'), '#a78bfa'),
    kpi(_t('kpi_total'), fmt(s.total_tokens), _t('sub_io'), '#6c8cff'),
    kpi(_t('kpi_cost'), s.cost_status === 'included' ? _t('sub_included') : ('$' + s.estimated_cost_usd.toFixed(2)), s.cost_status === 'included' ? '' : s.cost_status, '#f472b6'),
  ].join('');

  renderCharts();
  renderTokenCharts();
  renderSessionsTable();
  renderDelegatesTable();
  renderDurationTable();
  renderTokenTable();
  renderAgentBreakdown();
}

function kpi(label, value, sub, color) {
  return `<div class="kpi-card">
    <div class="kpi-label">${label}</div>
    <div class="kpi-value" style="color:${color}">${value}</div>
    ${sub ? `<div class="kpi-sub">${sub}</div>` : ''}
  </div>`;
}

function fmt(n) {
  if (n >= 1e6) return (n/1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n/1e3).toFixed(1) + 'K';
  return n;
}

// ── Charts ─────────────────────────────────────────────────────────────
function renderCharts() {
  // Models (doughnut)
  const models = stats.by_model;
  const mLabels = Object.keys(models);
  const mData = Object.values(models);
  if (chartModels) chartModels.destroy();
  chartModels = new Chart(document.getElementById('chart-models'), {
    type: 'doughnut',
    data: {
      labels: mLabels,
      datasets: [{ data: mData, backgroundColor: COLORS.slice(0, mLabels.length), borderWidth: 0 }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'right', labels: { color: '#8b90a5', font: { size: 11, family: 'monospace' }, padding: 12 } }
      }
    }
  });

  // Tools (horizontal bar)
  const tools = stats.top_tools;
  const tLabels = Object.keys(tools).reverse();
  const tData = Object.values(tools).reverse();
  if (chartTools) chartTools.destroy();
  chartTools = new Chart(document.getElementById('chart-tools'), {
    type: 'bar',
    data: {
      labels: tLabels,
      datasets: [{ data: tData, backgroundColor: '#6c8cff', borderRadius: 3, barThickness: 18 }]
    },
    options: {
      indexAxis: 'y', responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: '#2e3348' }, ticks: { color: '#8b90a5', font: { size: 10 } } },
        y: { grid: { display: false }, ticks: { color: '#e1e4ed', font: { size: 11, family: 'monospace' } } }
      }
    }
  });

  // Days (line)
  const days = stats.by_day;
  const dLabels = Object.keys(days);
  const dData = Object.values(days);
  if (chartDays) chartDays.destroy();
  chartDays = new Chart(document.getElementById('chart-days'), {
    type: 'bar',
    data: {
      labels: dLabels,
      datasets: [
        {
          label: _t('chart_sessions'), data: dData, backgroundColor: '#6c8cff44',
          borderColor: '#6c8cff', borderWidth: 2, borderRadius: 4, tension: 0.3,
          fill: true,
        }
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: '#2e3348' }, ticks: { color: '#8b90a5', font: { size: 11 } } },
        y: { grid: { color: '#2e3348' }, ticks: { color: '#8b90a5', font: { size: 10 }, stepSize: 1 },
             beginAtZero: true }
      }
    }
  });
}

// ── Sessions table ─────────────────────────────────────────────────────
function renderSessionsTable() {
  const filter = document.getElementById('session-filter').value.toLowerCase();
  let rows = stats.sessions;
  if (filter) rows = rows.filter(r => r.model.toLowerCase().includes(filter));
  rows.sort((a, b) => {
    let va = a[sortKey], vb = b[sortKey];
    if (va == null) va = -Infinity;
    if (vb == null) vb = -Infinity;
    if (typeof va === 'string') return sortDir * va.localeCompare(vb);
    return sortDir * (vb - va);
  });

  document.getElementById('sessions-tbody').innerHTML = rows.map(r => {
    const isSA = r.agent_type === 'main_with_subagents';
    const typeBadge = isSA
      ? '<span class="badge badge-subagent">🔧 ' + _t('badge_sa') + '</span>'
      : '<span class="badge badge-direct">⚡ ' + _t('badge_direct') + '</span>';
    const inp = r.input_tokens || 0;
    const out = r.output_tokens || 0;
    return `<tr>
      <td class="num">${r.mtime}</td>
      <td>${typeBadge}</td>
      <td><span class="badge badge-model">${esc(r.model)}</span></td>
      <td><span class="badge badge-platform">${esc(r.platform)}</span></td>
      <td class="num">${r.user_msgs}</td>
      <td class="num">${r.assistant_msgs}</td>
      <td class="num">${r.tool_calls}</td>
      <td class="num">${r.delegate_calls || 0}</td>
      <td class="num">${inp ? fmt(inp) : '<span style="color:var(--text2)">—</span>'}</td>
      <td class="num">${out ? fmt(out) : '<span style="color:var(--text2)">—</span>'}</td>
      <td class="num">${r.duration_min != null ? r.duration_min + _t('unit_min') : '—'}</td>
    </tr>`;
  }).join('');
}

// ── Delegates table ────────────────────────────────────────────────────
function renderDelegatesTable() {
  const rows = stats.delegates;
  if (!rows.length) {
    document.getElementById('delegates-tbody').innerHTML =
      `<tr><td colspan="4" style="text-align:center;color:var(--text2);padding:20px">${_t('no_delegates')}</td></tr>`;
    return;
  }
  document.getElementById('delegates-tbody').innerHTML = rows.map(r => `
    <tr>
      <td class="num">${r.timestamp}</td>
      <td class="delegate-goal" title="${esc(r.goal)}">${esc(r.goal)}</td>
      <td class="num">${r.tasks || 1}</td>
      <td>${(r.toolsets||[]).map(ts => `<span class="badge badge-delegate">${ts}</span>`).join(' ')}</td>
    </tr>
  `).join('');
}

// ── Duration table ─────────────────────────────────────────────────────
function renderDurationTable() {
  const rows = stats.top_duration;
  document.getElementById('duration-tbody').innerHTML = rows.map(r => `
    <tr>
      <td class="num" style="font-weight:600">${r.duration_min}</td>
      <td><span class="badge badge-model">${esc(r.model)}</span></td>
      <td class="num">${r.tool_calls}</td>
      <td class="num">${r.delegate_calls}</td>
      <td class="num">${r.user_msgs}</td>
      <td class="num">${r.input_tokens ? fmt(r.input_tokens) : '—'}</td>
      <td class="num">${r.output_tokens ? fmt(r.output_tokens) : '—'}</td>
      <td class="num">${r.mtime}</td>
    </tr>
  `).join('');
}

// ── Agent / Subagent breakdown ─────────────────────────────────────────
function renderAgentBreakdown() {
  const ab = stats.agent_breakdown;
  if (!ab) return;
  document.getElementById('ab-total').textContent = ab.total_sessions;
  document.getElementById('ab-direct').textContent = ab.direct_sessions;
  document.getElementById('ab-with-sa').textContent = ab.sessions_with_subagents;
  document.getElementById('ab-deleg').textContent = ab.total_delegate_calls;
  document.getElementById('ab-tasks').textContent = ab.total_subagent_tasks;
  const ts = Object.entries(ab.subagent_toolsets || {});
  document.getElementById('ab-ts').innerHTML = ts.length
    ? ts.map(([k,v]) => `<span class="badge badge-delegate" style="margin:2px">${k}: ${v}</span>`).join('')
    : `<span style="color:var(--text2)">${_t('no_data')}</span>`;
}

// ── Token charts ──────────────────────────────────────────────────
function renderTokenCharts() {
  // Token by day — stacked bar (input, output, cache_read)
  const tbd = stats.token_by_day;
  const tbdKeys = Object.keys(tbd);
  if (chartTokenDays) chartTokenDays.destroy();
  chartTokenDays = new Chart(document.getElementById('chart-token-days'), {
    type: 'bar',
    data: {
      labels: tbdKeys,
      datasets: [
        {
          label: 'Input', data: tbdKeys.map(k => tbd[k].input),
          backgroundColor: '#fbbf2488', borderColor: '#fbbf24', borderWidth: 1, borderRadius: 2,
        },
        {
          label: 'Output', data: tbdKeys.map(k => tbd[k].output),
          backgroundColor: '#2dd4bf88', borderColor: '#2dd4bf', borderWidth: 1, borderRadius: 2,
        },
        {
          label: 'Cache read', data: tbdKeys.map(k => tbd[k].cache_read),
          backgroundColor: '#a78bfa88', borderColor: '#a78bfa', borderWidth: 1, borderRadius: 2,
        },
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: '#8b90a5', font: { size: 11, family: 'monospace' }, padding: 16 } },
        tooltip: {
          callbacks: {
            label: (ctx) => ctx.dataset.label + ': ' + fmt(ctx.raw),
          }
        }
      },
      scales: {
        x: { stacked: true, grid: { color: '#2e3348' }, ticks: { color: '#8b90a5', font: { size: 11 } } },
        y: { stacked: true, grid: { color: '#2e3348' }, ticks: { color: '#8b90a5', font: { size: 10 }, callback: (v) => fmt(v) },
             beginAtZero: true },
      }
    }
  });

  // Token by model — horizontal bar
  const tbm = stats.token_by_model;
  const modelKeys = Object.keys(tbm);
  if (chartTokenModels) chartTokenModels.destroy();
  if (modelKeys.length) {
    chartTokenModels = new Chart(document.getElementById('chart-token-models'), {
      type: 'bar',
      data: {
        labels: modelKeys,
        datasets: [
          {
            label: 'Input', data: modelKeys.map(k => tbm[k].input),
            backgroundColor: '#fbbf2488', borderColor: '#fbbf24', borderWidth: 1, borderRadius: 3, barThickness: 24,
          },
          {
            label: 'Output', data: modelKeys.map(k => tbm[k].output),
            backgroundColor: '#2dd4bf88', borderColor: '#2dd4bf', borderWidth: 1, borderRadius: 3, barThickness: 24,
          },
        ]
      },
      options: {
        indexAxis: 'y', responsive: true,
        plugins: {
          legend: { labels: { color: '#8b90a5', font: { size: 11, family: 'monospace' }, padding: 16 } },
          tooltip: {
            callbacks: {
              afterBody: (items) => {
                const model = items[0].label;
                const d = tbm[model];
                if (!d) return '';
                const lines = [];
                lines.push('Cache read: ' + fmt(d.cache_read));
                lines.push('Cache write: ' + fmt(d.cache_write));
                lines.push('Sessions: ' + d.sessions);
                if (d.cost > 0) lines.push('Cost: $' + d.cost.toFixed(2));
                return lines;
              }
            }
          }
        },
        scales: {
          x: { stacked: true, grid: { color: '#2e3348' }, ticks: { color: '#8b90a5', font: { size: 10 }, callback: (v) => fmt(v) } },
          y: { stacked: true, grid: { display: false }, ticks: { color: '#e1e4ed', font: { size: 12, family: 'monospace' } } },
        }
      }
    });
  }
}

// ── Token table ──────────────────────────────────────────────────
function renderTokenTable() {
  const rows = stats.top_cost_sessions;
  if (!rows || !rows.length) {
    document.getElementById('token-tbody').innerHTML =
      `<tr><td colspan="6" style="text-align:center;color:var(--text2);padding:20px">${_t('no_token_data')}</td></tr>`;
    return;
  }
  document.getElementById('token-tbody').innerHTML = rows.map(r => `
    <tr>
      <td><span class="badge badge-model">${esc(r.model)}</span></td>
      <td class="num">${fmt(r.input_tokens)}</td>
      <td class="num">${fmt(r.output_tokens)}</td>
      <td class="num">${r.cache_read_tokens ? fmt(r.cache_read_tokens) : '—'}</td>
      <td class="num">${r.estimated_cost_usd > 0 ? '$' + r.estimated_cost_usd.toFixed(2) : '<span style="color:var(--text2)">—</span>'}</td>
      <td class="num">${r.started_at}</td>
    </tr>
  `).join('');
}

// ── Utils ──────────────────────────────────────────────────────────────
function esc(s) {
  if (!s) return '';
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Sorting ────────────────────────────────────────────────────────────
document.querySelectorAll('#sessions-table th[data-key]').forEach(th => {
  th.addEventListener('click', () => {
    const key = th.dataset.key;
    if (sortKey === key) sortDir *= -1;
    else { sortKey = key; sortDir = -1; }
    document.querySelectorAll('#sessions-table th').forEach(h => h.classList.remove('sorted'));
    th.classList.add('sorted');
    renderSessionsTable();
  });
});

// ── Filter ─────────────────────────────────────────────────────────────
document.getElementById('session-filter').addEventListener('input', renderSessionsTable);

// ── Period change ──────────────────────────────────────────────────────
document.getElementById('period-select').addEventListener('change', refresh);

// ── Init ───────────────────────────────────────────────────────────────
document.documentElement.lang = lang;
updateLangButton();
applyI18n();
refresh();
connectWS();
setInterval(refresh, 30000);
</script>
</body>
</html>
"""


# ── Main ───────────────────────────────────────────────────────────────────

def run(port: int = 8420, bind: str = "0.0.0.0", poll: int = 5):
    """Start dashboard server."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(_push_updates(poll, 7))
        yield
        task.cancel()

    app.router.lifespan_context = lifespan

    print(f"""
╔══════════════════════════════════════════════╗
║       ⚡ Hermes Agent Dashboard               ║
║       http://{bind}:{port}                    ║
║       Polling every {poll}s                     ║
╚══════════════════════════════════════════════╝
""")
    uvicorn.run(app, host=bind, port=port, log_level="warning", ws_ping_interval=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hermes Agent Dashboard")
    parser.add_argument("--port", type=int, default=8420, help="Port (default: 8420)")
    parser.add_argument("--bind", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--poll", type=int, default=5, help="File poll interval in seconds (default: 5)")
    args = parser.parse_args()
    run(port=args.port, bind=args.bind, poll=args.poll)
