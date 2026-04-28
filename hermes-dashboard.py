#!/usr/bin/env python3
"""Hermes Agent Dashboard — real-time session statistics.

Usage:
    python3 hermes-dashboard.py [--port 8420] [--bind 0.0.0.0] [--poll 5]
"""

import argparse
import json
import os
import re
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


def _safe_json_loads(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def _extract_tool_calls_from_obj(obj: dict) -> list[dict]:
    """Extract normalized tool call records from a JSONL object."""
    calls = []

    def _add_call(name, args=None, meta=None):
        if not name:
            return
        calls.append({"tool": str(name), "arguments": args, "meta": meta or {}})

    tc = obj.get("tool_calls")
    if isinstance(tc, list):
        for t in tc:
            if not isinstance(t, dict):
                continue
            fn = t.get("function", {}) if isinstance(t.get("function", {}), dict) else {}
            _add_call(fn.get("name") or t.get("name"), _safe_json_loads(fn.get("arguments")), t)

    for key in ("content", "message", "tool_use", "tool_uses"):
        val = obj.get(key)
        if isinstance(val, dict):
            val = [val]
        if isinstance(val, list):
            for item in val:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "tool_use" or "tool" in item or "name" in item:
                    _add_call(item.get("tool") or item.get("name"), item.get("arguments") or item.get("input"), item)

    if isinstance(obj.get("content"), str):
        decoded = _safe_json_loads(obj.get("content"))
        if isinstance(decoded, list):
            for item in decoded:
                if isinstance(item, dict) and (item.get("type") == "tool_use" or item.get("tool") or item.get("name")):
                    _add_call(item.get("tool") or item.get("name"), item.get("arguments") or item.get("input"), item)

    return calls


def _parse_iso_datetime(value):
    if not value:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value)
        except Exception:
            return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return None
    return None


def collect_tool_stats(days: int = 7) -> dict:
    """Build tool analytics from session JSONL files."""
    cutoff = datetime.now() - timedelta(days=days)
    tool_counts = Counter()
    tool_latency_sum = defaultdict(float)
    tool_latency_n = Counter()
    tool_errors = Counter()
    tool_arg_keys = defaultdict(Counter)
    tool_hourly = defaultdict(lambda: [0] * 24)
    session_tools = defaultdict(set)
    session_tool_counts = Counter()

    for fpath in sorted(SESSIONS_DIR.glob("*.jsonl")):
        mtime = datetime.fromtimestamp(fpath.stat().st_mtime)
        if mtime < cutoff:
            continue
        session_id = fpath.name
        session_hour = mtime.hour
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    ts = obj.get("timestamp")
                    hour = session_hour
                    if ts:
                        try:
                            hour = datetime.fromisoformat(ts).hour
                        except Exception:
                            pass
                    for call in _extract_tool_calls_from_obj(obj):
                        tool = call["tool"]
                        meta = call.get("meta") or {}
                        tool_counts[tool] += 1
                        tool_hourly[tool][hour] += 1
                        session_tools[session_id].add(tool)
                        session_tool_counts[session_id] += 1

                        latency = meta.get("latency_ms")
                        if latency is None:
                            latency = meta.get("duration_ms")
                        if latency is not None:
                            try:
                                tool_latency_sum[tool] += float(latency)
                                tool_latency_n[tool] += 1
                            except Exception:
                                pass

                        is_error = bool(meta.get("error")) or meta.get("success") is False
                        exit_code = meta.get("exit_code")
                        if exit_code is not None:
                            try:
                                is_error = is_error or int(exit_code) != 0
                            except Exception:
                                pass
                        if is_error:
                            tool_errors[tool] += 1

                        args = call.get("arguments")
                        if isinstance(args, dict):
                            for k in args.keys():
                                tool_arg_keys[tool][k] += 1
        except OSError:
            continue

    tool_frequency = []
    for tool, count in tool_counts.most_common():
        n = tool_latency_n.get(tool, 0)
        tool_frequency.append({
            "tool": tool,
            "count": count,
            "avg_latency_ms": round(tool_latency_sum[tool] / n, 2) if n else None,
            "error_rate": round(tool_errors[tool] / count, 4) if count else 0,
            "top_arguments": [k for k, _ in tool_arg_keys[tool].most_common(5)],
        })

    correlation = []
    sessions_with_tool = {sid: tools for sid, tools in session_tools.items()}
    all_tools = sorted(tool_counts.keys())
    for i, a in enumerate(all_tools):
        for b in all_tools[i + 1:]:
            joint = sum(1 for tools in sessions_with_tool.values() if a in tools and b in tools)
            if not joint:
                continue
            a_sessions = sum(1 for tools in sessions_with_tool.values() if a in tools)
            b_sessions = sum(1 for tools in sessions_with_tool.values() if b in tools)
            denom = ((a_sessions * b_sessions) ** 0.5) or 1.0
            correlation.append({"tool_a": a, "tool_b": b, "joint_sessions": joint, "score": round(joint / denom, 4)})
    correlation.sort(key=lambda x: (-x["joint_sessions"], -x["score"], x["tool_a"], x["tool_b"]))

    top_sessions = []
    for sid, count in sorted(session_tool_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]:
        top_sessions.append({"file": sid, "tool_count": count, "tools": sorted(session_tools[sid])})

    return {"ok": True, "days": days, "tool_frequency": tool_frequency, "correlation": correlation, "hourly": dict(tool_hourly), "top_sessions": top_sessions}


def _truncate(text: str, limit: int = 120) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def collect_subagent_stats(days: int = 7) -> dict:
    """Build delegate/subagent tree and aggregates from session JSONL files."""
    cutoff = datetime.now() - timedelta(days=days)
    stop_words = {
        "the", "and", "for", "with", "that", "this", "from", "have", "your", "you", "are", "was", "were",
        "into", "out", "about", "using", "use", "task", "tasks", "delegate", "subagent", "agent", "please",
        "need", "needed", "make", "build", "create", "fix", "add", "update", "implement", "write", "run",
        "data", "file", "files", "session", "sessions", "goal", "step", "steps", "to", "of", "in", "on",
        "a", "an", "is", "it", "as", "by", "be", "or", "at", "can", "do", "not", "we", "i",
    }

    tree = []
    goal_words = Counter()
    toolsets_distribution = Counter()
    depth_values = []
    total_duration = 0.0
    duration_count = 0
    completed = 0
    total_delegates = 0

    for fpath in sorted(SESSIONS_DIR.glob("*.jsonl")):
        mtime = datetime.fromtimestamp(fpath.stat().st_mtime)
        if mtime < cutoff:
            continue
        session_file = fpath.name
        root = {"session_file": session_file, "children": []}

        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if obj.get("role") != "assistant":
                        continue
                    tc = obj.get("tool_calls")
                    if not isinstance(tc, list):
                        continue
                    for t in tc:
                        fn = t.get("function", {}) if isinstance(t, dict) else {}
                        if fn.get("name") != "delegate_task":
                            continue
                        args_str = fn.get("arguments", "")
                        goal = ""
                        n_tasks = 0
                        toolsets = []
                        if isinstance(args_str, str):
                            try:
                                args_obj = json.loads(args_str)
                                goal = (args_obj.get("goal") or "")[:120]
                                tasks = args_obj.get("tasks")
                                n_tasks = len(tasks) if isinstance(tasks, list) else 0
                                toolsets = args_obj.get("toolsets") or []
                            except (json.JSONDecodeError, TypeError):
                                pass

                        node = {
                            "id": f"{session_file}:{total_delegates}",
                            "goal": goal or "(batch)",
                            "tasks_count": n_tasks,
                            "toolsets": [str(t) for t in toolsets],
                            "duration_ms": None,
                            "status": "completed",
                            "depth": 1,
                            "session_file": session_file,
                            "children": [],
                        }
                        root["children"].append(node)
                        total_delegates += 1
                        completed += 1
                        for ts in node["toolsets"]:
                            toolsets_distribution[ts] += 1
                        for word in re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}", node["goal"].lower()):
                            if word not in stop_words:
                                goal_words[word] += 1
        except OSError:
            continue

        for child in root["children"]:
            depth_values.append(1)

        if root["children"]:
            tree.append(root)

    aggregates = {
        "total_delegates": total_delegates,
        "avg_depth": round(sum(depth_values) / len(depth_values), 2) if depth_values else 0,
        "max_depth": max(depth_values) if depth_values else 0,
        "success_rate": round(completed / total_delegates, 4) if total_delegates else 0,
        "avg_duration_ms": round(total_duration / duration_count, 2) if duration_count else 0,
    }

    return {
        "ok": True,
        "days": days,
        "tree": tree,
        "aggregates": aggregates,
        "top_goals": [{"word": word, "count": count} for word, count in goal_words.most_common(20)],
        "toolsets_distribution": dict(toolsets_distribution),
    }


def collect_stats(days: int = 7) -> dict:
    """Build stats for the API over the last `days` days."""
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

    hour_counter = Counter()
    dow_counter = Counter()
    platform_hour = defaultdict(Counter)
    global_response_time_sum = 0.0
    global_response_time_n = 0

    model_perf_acc = defaultdict(lambda: {"sessions": 0, "duration_sum": 0.0, "messages_sum": 0.0, "tool_calls_sum": 0.0, "tool_results_total": 0, "tool_errors": 0, "response_time_sum": 0.0, "response_time_n": 0})

    for s in sessions:
        total_user_msgs += s["user_msgs"]
        total_messages["user"] += s["user_msgs"]
        total_messages["assistant"] += s["assistant_msgs"]
        total_tool_calls += s["tool_calls"]
        total_delegates += s["delegate_calls"]
        model = s["model"] or "?"
        model_counter[model] += 1
        platform_counter[s["platform"]] += 1
        day_counter[s["mtime_str"][:10]] += 1
        for tool, count in s["tools_used"].items():
            tool_counter[tool] += count
        for ev in s["delegate_events"]:
            ev_copy = dict(ev)
            ev_copy["session"] = s["file"]
            all_delegates.append(ev_copy)

        perf = model_perf_acc[model]
        perf["sessions"] += 1
        perf["duration_sum"] += float(s["duration_min"] or 0.0)
        perf["messages_sum"] += float(s["user_msgs"] + s["assistant_msgs"])
        perf["tool_calls_sum"] += float(s["tool_calls"])

        session_path = SESSIONS_DIR / s["file"]
        if session_path.exists():
            last_user_ts = None
            last_user_ts_global = None
            try:
                with open(session_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        role = obj.get("role")
                        ts_str = obj.get("timestamp", "")
                        if ts_str:
                            try:
                                ts2 = datetime.fromisoformat(ts_str)
                                hour_counter[ts2.strftime("%H")] += 1
                                dow_counter[ts2.strftime("%a")] += 1
                                platform_hour[s["platform"]][ts2.strftime("%H")] += 1
                            except (ValueError, TypeError):
                                pass
                        if role == "tool":
                            meta = obj if isinstance(obj, dict) else {}
                            perf["tool_results_total"] += 1
                            if meta.get("error") or meta.get("is_error") is True:
                                perf["tool_errors"] += 1
                        ts = _parse_iso_datetime(obj.get("timestamp"))
                        if role == "user" and ts:
                            last_user_ts = ts
                            last_user_ts_global = ts
                        elif role == "assistant" and ts and last_user_ts:
                            perf["response_time_sum"] += max((ts - last_user_ts).total_seconds() * 1000.0, 0.0)
                            perf["response_time_n"] += 1
                            last_user_ts = None
                            if last_user_ts_global:
                                global_response_time_sum += max((ts - last_user_ts_global).total_seconds(), 0.0)
                                global_response_time_n += 1
                                last_user_ts_global = None
            except Exception:
                pass

    for s in sessions:
        s["agent_type"] = "main_with_subagents" if s["delegate_calls"] > 0 else "direct"

    subagent_tool_counter = Counter()
    subagent_tasks = 0
    for s in sessions:
        for ev in s["delegate_events"]:
            subagent_tasks += max(ev.get("tasks", 1), 1)
            for ts in (ev.get("toolsets") or []):
                subagent_tool_counter[ts] += 1

    sessions.sort(key=lambda x: x["mtime"], reverse=True)
    by_duration = sorted([s for s in sessions if s["duration_min"] is not None], key=lambda x: x["duration_min"], reverse=True)
    by_tools = sorted(sessions, key=lambda x: x["tool_calls"], reverse=True)

    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_cache_write = 0
    total_reasoning = 0
    total_cost = 0.0
    cost_status = "unknown"

    db_sessions = _load_token_data_from_db(cutoff)
    token_by_model: dict[str, dict] = {}
    token_by_day: dict[str, dict] = {}
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

        cs = db_s.get("cost_status") or "unknown"
        if cs not in ("unknown", None) and cost_status == "unknown":
            cost_status = cs

        if model not in token_by_model:
            token_by_model[model] = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "reasoning": 0, "cost": 0.0, "sessions": 0}
        token_by_model[model]["input"] += inp
        token_by_model[model]["output"] += out
        token_by_model[model]["cache_read"] += cr
        token_by_model[model]["cache_write"] += cw
        token_by_model[model]["reasoning"] += reas
        token_by_model[model]["cost"] += cost
        token_by_model[model]["sessions"] += 1

        day = datetime.fromtimestamp(started).strftime("%Y-%m-%d")
        if day not in token_by_day:
            token_by_day[day] = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "reasoning": 0, "cost": 0.0}
        token_by_day[day]["input"] += inp
        token_by_day[day]["output"] += out
        token_by_day[day]["cache_read"] += cr
        token_by_day[day]["cache_write"] += cw
        token_by_day[day]["reasoning"] += reas
        token_by_day[day]["cost"] += cost

        top_cost_sessions.append({"id": db_s.get("id"), "model": model, "started_at": started, "input_tokens": inp, "output_tokens": out, "cost": cost})

    top_cost_sessions = sorted(top_cost_sessions, key=lambda x: x["cost"], reverse=True)[:10]
    token_trend = []
    for i in range(days - 1, -1, -1):
        day = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        token_trend.append({"day": day, "input": token_by_day.get(day, {}).get("input", 0), "output": token_by_day.get(day, {}).get("output", 0), "cost": token_by_day.get(day, {}).get("cost", 0.0)})

    db_token_lookup = {}
    for s in db_sessions:
        db_token_lookup[s["id"]] = {"input_tokens": s.get("input_tokens") or 0, "output_tokens": s.get("output_tokens") or 0, "cache_read_tokens": s.get("cache_read_tokens") or 0, "estimated_cost_usd": s.get("estimated_cost_usd") or 0}

    model_performance = {}
    for model, perf in model_perf_acc.items():
        db_models = [s for s in db_sessions if (s.get("model") or "?") == model]
        total_input_model = sum(int(s.get("input_tokens") or 0) for s in db_models)
        total_output_model = sum(int(s.get("output_tokens") or 0) for s in db_models)
        total_cost_model = sum(float(s.get("estimated_cost_usd") or 0.0) for s in db_models)
        input_output_ratio = None
        if total_output_model:
            input_output_ratio = round(total_input_model / total_output_model, 3)
        tool_success_rate = None
        if perf["tool_results_total"]:
            tool_success_rate = round((perf["tool_results_total"] - perf["tool_errors"]) / perf["tool_results_total"], 4)
        avg_response_time_ms = None
        if perf["response_time_n"]:
            avg_response_time_ms = round(perf["response_time_sum"] / perf["response_time_n"], 1)
        model_performance[model] = {
            "sessions": perf["sessions"],
            "avg_duration_min": round(perf["duration_sum"] / perf["sessions"], 2) if perf["sessions"] else None,
            "avg_messages": round(perf["messages_sum"] / perf["sessions"], 2) if perf["sessions"] else None,
            "avg_tool_calls": round(perf["tool_calls_sum"] / perf["sessions"], 2) if perf["sessions"] else None,
            "tool_success_rate": tool_success_rate,
            "input_output_ratio": input_output_ratio,
            "total_cost_usd": round(total_cost_model, 4),
            "avg_response_time_ms": avg_response_time_ms,
        }

    for s in sessions:
        sid = s["file"].replace(".jsonl", "")
        if sid in meta:
            s["meta"] = meta[sid]

    return {
        "ok": True,
        "days": days,
        "summary": {"sessions": len(sessions), "users": total_user_msgs, "tool_calls": total_tool_calls, "delegates": total_delegates},
        "delegates": all_delegates,
        "sessions": [{"file": s["file"], "model": s["model"], "platform": s["platform"], "user_msgs": s["user_msgs"], "assistant_msgs": s["assistant_msgs"], "tool_calls": s["tool_calls"], "delegate_calls": s["delegate_calls"], "duration_min": s["duration_min"], "mtime": s["mtime_str"], "agent_type": s["agent_type"], "input_tokens": db_token_lookup.get(s["file"].replace(".jsonl", ""), {}).get("input_tokens", 0), "output_tokens": db_token_lookup.get(s["file"].replace(".jsonl", ""), {}).get("output_tokens", 0), "cache_read_tokens": db_token_lookup.get(s["file"].replace(".jsonl", ""), {}).get("cache_read_tokens", 0), "estimated_cost_usd": db_token_lookup.get(s["file"].replace(".jsonl", ""), {}).get("estimated_cost_usd", 0)} for s in sessions],
        "model_performance": model_performance,
        "by_hour": dict(sorted(hour_counter.items())),
        "by_dow": dict(dow_counter.most_common()),
        "by_platform": dict(platform_counter),
        "platform_hour": {p: dict(sorted(c.items())) for p, c in platform_hour.items()},
        "avg_response_time_sec": round(global_response_time_sum / global_response_time_n, 1) if global_response_time_n else None,
        "agent_breakdown": {"total_sessions": len(sessions), "direct_sessions": sum(1 for s in sessions if s["agent_type"] == "direct"), "sessions_with_subagents": sum(1 for s in sessions if s["agent_type"] == "main_with_subagents"), "total_delegate_calls": total_delegates, "total_subagent_tasks": subagent_tasks, "subagent_toolsets": dict(subagent_tool_counter.most_common())},
    }


def _summarize_trend_stats(stats: dict) -> dict:
    sessions = stats.get("sessions", [])
    summary = stats.get("summary", {})
    return {
        "sessions": summary.get("sessions", len(sessions)),
        "user_msgs": summary.get("users", 0),
        "tool_calls": summary.get("tool_calls", 0),
        "delegates": summary.get("delegates", 0),
        "input_tokens": sum(int(s.get("input_tokens") or 0) for s in sessions),
        "output_tokens": sum(int(s.get("output_tokens") or 0) for s in sessions),
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, variance ** 0.5


def _fmt_num(n: float) -> str:
    if n >= 1_000_000:
        return f"{n:,.0f}"
    if n >= 1_000:
        return f"{n:,.1f}" if n % 1 else f"{n:,.0f}"
    return f"{n:.1f}" if n % 1 else f"{n:.0f}"


def collect_alerts(days: int = 7) -> dict:
    """Detect session and daily anomalies over the last `days` days."""
    stats = collect_stats(days=days)
    sessions = stats.get("sessions", []) or []

    metrics = ["duration_min", "tool_calls", "delegate_calls", "input_tokens", "output_tokens", "user_msgs"]
    metric_values = {metric: [] for metric in metrics}
    for s in sessions:
        for metric in metrics:
            value = s.get(metric)
            if value is None:
                continue
            try:
                metric_values[metric].append(float(value))
            except (TypeError, ValueError):
                continue

    alerts = []
    by_metric = Counter()

    for metric in metrics:
        values = metric_values[metric]
        mean, std = _mean_std(values)
        if std == 0:
            continue
        for s in sessions:
            value = s.get(metric)
            if value is None:
                continue
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                continue
            z = (value_f - mean) / std
            if abs(z) <= 3:
                continue
            severity = "high" if abs(z) > 4 else "medium"
            direction = "above" if z > 0 else "below"
            message = f"Session {metric.replace('_', ' ')} {_fmt_num(value_f)} is {abs(z):.1f}σ {direction} average ({_fmt_num(mean)} ± {_fmt_num(std)})"
            alerts.append({
                "type": "anomaly",
                "severity": severity,
                "metric": metric,
                "session_file": s.get("file"),
                "value": value_f,
                "mean": round(mean, 3),
                "std": round(std, 3),
                "z_score": round(z, 3),
                "message": message,
            })
            by_metric[metric] += 1

    # Daily activity alerts from session dates
    day_counts = Counter()
    for s in sessions:
        day = (s.get("mtime") or "")[:10]
        if day:
            day_counts[day] += 1

    if day_counts:
        daily_values = [float(day_counts.get((datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"), 0)) for i in range(days - 1, -1, -1)]
        daily_mean, daily_std = _mean_std(daily_values)
        if daily_std > 0:
            for day, count in sorted(day_counts.items()):
                z = (count - daily_mean) / daily_std
                if z > 3:
                    alerts.append({
                        "type": "spike_day",
                        "severity": "high" if z > 4 else "medium",
                        "metric": "sessions_per_day",
                        "day": day,
                        "value": count,
                        "mean": round(daily_mean, 3),
                        "std": round(daily_std, 3),
                        "z_score": round(z, 3),
                        "message": f"Spike: {count} sessions on {day} (avg {daily_mean:.1f})",
                    })
                    by_metric["sessions_per_day"] += 1
        if daily_mean > 2:
            for i in range(days - 1, -1, -1):
                day = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                if day_counts.get(day, 0) == 0:
                    alerts.append({
                        "type": "zero_activity_day",
                        "severity": "medium",
                        "metric": "sessions_per_day",
                        "day": day,
                        "value": 0,
                        "mean": round(daily_mean, 3),
                        "std": round(daily_std, 3),
                        "z_score": None,
                        "message": f"No activity on {day}",
                    })
                    by_metric["sessions_per_day"] += 1

    alerts.sort(key=lambda a: (0 if a.get("severity") == "high" else 1, -(abs(a.get("z_score") or 0)), a.get("metric", ""), a.get("session_file", a.get("day", ""))))
    return {"ok": True, "days": days, "alerts": alerts, "summary": {"total_alerts": len(alerts), "by_metric": dict(by_metric)}}


def collect_trends(days: int = 7) -> dict:
    """Compare current period stats against the previous period."""
    current = collect_stats(days=days)
    previous_window = collect_stats(days=days * 2)

    current_summary = _summarize_trend_stats(current)

    current_cutoff = datetime.now() - timedelta(days=days)
    previous_cutoff = datetime.now() - timedelta(days=2 * days)
    current_boundary = current_cutoff.strftime("%Y-%m-%d")
    previous_boundary = previous_cutoff.strftime("%Y-%m-%d")

    previous_sessions = [s for s in previous_window.get("sessions", []) if previous_boundary <= s.get("mtime", "")[:10] < current_boundary]
    previous_summary = {
        "sessions": len(previous_sessions),
        "user_msgs": sum(int(s.get("user_msgs") or 0) for s in previous_sessions),
        "tool_calls": sum(int(s.get("tool_calls") or 0) for s in previous_sessions),
        "delegates": sum(int(s.get("delegate_calls") or 0) for s in previous_sessions),
        "input_tokens": sum(int(s.get("input_tokens") or 0) for s in previous_sessions),
        "output_tokens": sum(int(s.get("output_tokens") or 0) for s in previous_sessions),
    }

    def _count_days(stats: dict, start: str, end: str | None = None) -> Counter:
        counts = Counter()
        for s in stats.get("sessions", []):
            day = s.get("mtime", "")[:10]
            if day >= start and (end is None or day < end):
                counts[day] += 1
        return counts

    current_by_day = _count_days(current, current_boundary)
    previous_by_day = _count_days(previous_window, previous_boundary, current_boundary)

    deltas = {}
    for key, cur_val in current_summary.items():
        prev_val = previous_summary.get(key, 0)
        delta = cur_val - prev_val
        pct = round((delta / prev_val) * 100, 1) if prev_val > 0 else (100 if delta > 0 else 0)
        deltas[key] = {"value": delta, "pct": pct, "trend": "up" if delta > 0 else "down" if delta < 0 else "flat"}

    daily_trend = []
    for i in range(days - 1, -1, -1):
        day = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        previous_day = (datetime.now() - timedelta(days=i + days)).strftime("%Y-%m-%d")
        daily_trend.append({"day": day, "current": current_by_day.get(day, 0), "previous": previous_by_day.get(previous_day, 0)})

    return {"ok": True, "days": days, "current": current_summary, "previous": previous_summary, "deltas": deltas, "daily_trend": daily_trend}


def parse_session_jsonl(file_name: str) -> dict:
    """Parse a single JSONL session into a structured timeline."""
    fpath = SESSIONS_DIR / file_name
    if not fpath.exists():
        return {"ok": False, "error": "not found"}
    turns = []
    meta = {}
    for line in open(fpath, "r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        role = obj.get("role", "")
        if role == "session_meta":
            meta = {k: obj.get(k) for k in ["model", "platform", "provider"]}
            continue
        ts = obj.get("timestamp", "")
        turn = {
            "role": role,
            "timestamp": ts,
            "content_preview": (obj.get("content", "") or "")[:200],
        }
        # Tool calls
        tc = obj.get("tool_calls")
        if isinstance(tc, list) and tc:
            turn["tool_calls"] = []
            for t in tc:
                fn = t.get("function", {}) if isinstance(t, dict) else {}
                turn["tool_calls"].append({
                    "name": fn.get("name", "?"),
                    "arguments_preview": (fn.get("arguments", "") or "")[:300],
                })
        # Tool result
        if role == "tool" and "content" in obj:
            turn["tool_result_preview"] = str(obj["content"])[:200]
        turns.append(turn)
    return {
        "ok": True,
        "file": file_name,
        "meta": meta,
        "turn_count": len(turns),
        "turns": turns,
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


@app.get("/api/tools")
async def api_tools(days: int = 7):
    return collect_tool_stats(days=days)


@app.get("/api/subagents")
async def api_subagents(days: int = 7):
    return collect_subagent_stats(days=days)


@app.get("/api/trends")
async def api_trends(days: int = 7):
    return collect_trends(days=days)


@app.get("/api/alerts")
async def api_alerts(days: int = 7):
    return collect_alerts(days=days)


@app.get("/api/session/{file_id}")
async def api_session(file_id: str):
    # file_id is the UUID without .jsonl extension
    return parse_session_jsonl(file_id + ".jsonl")


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
  display: flex; align-items: center; justify-content: space-between; gap: 16px;
  margin-bottom: 24px; padding-bottom: 16px;
  border-bottom: 1px solid var(--border);
}
.header h1 { font-size: 20px; font-weight: 600; }
.header h1 span { color: var(--accent); }
.header-left { display:flex; align-items:center; gap:16px; min-width:0; flex-wrap:wrap; }
.status {
  display: flex; align-items: center; gap: 8px;
  font-size: 12px; color: var(--text2);
}
.status-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--green); display: inline-block;
  box-shadow: 0 0 0 0 rgba(52,211,153,.4);
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(52,211,153,.4); }
  70% { box-shadow: 0 0 0 8px rgba(52,211,153,0); }
  100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); }
}
.status-dot.offline { background: var(--red); animation: none; }
.alert-wrap { position: relative; }
.alert-badge {
  display:inline-flex; align-items:center; gap:6px;
  padding:6px 10px; border-radius:999px; border:1px solid var(--border);
  background: var(--surface2); color: var(--text2); font-size:12px; cursor:pointer;
  user-select:none; transition: all .15s ease;
}
.alert-badge.has-alerts { color: #ffb454; border-color: rgba(255,180,84,.35); background: rgba(255,180,84,.08); }
.alert-badge:hover { transform: translateY(-1px); }
.alert-panel {
  display:none; position:absolute; right:0; top:calc(100% + 8px); z-index:50;
  width:min(560px, calc(100vw - 32px)); max-height:360px; overflow:auto;
  background: var(--surface); border:1px solid var(--border); border-radius:12px;
  box-shadow: 0 16px 40px rgba(0,0,0,.28); padding:10px;
}
.alert-panel.open { display:block; }
.alert-panel-header { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:4px 6px 10px; border-bottom:1px solid var(--border); margin-bottom:8px; }
.alert-panel-title { font-size:13px; font-weight:600; }
.alert-panel-empty { padding:12px 6px; color: var(--text2); font-size:12px; }
.alert-row {
  display:grid; grid-template-columns: 14px 1fr auto; gap:10px; align-items:start;
  padding:10px 6px; border-bottom:1px solid rgba(255,255,255,.04);
}
.alert-row:last-child { border-bottom:none; }
.alert-severity-dot { width:10px; height:10px; border-radius:50%; margin-top:4px; }
.alert-severity-high { background: var(--red); }
.alert-severity-medium { background: var(--orange); }
.alert-row-main { min-width:0; }
.alert-row-title { font-size:12px; color: var(--text); font-weight:600; margin-bottom:2px; }
.alert-row-message { font-size:12px; color: var(--text2); line-height:1.35; }
.alert-row-meta { margin-top:5px; font-size:11px; color: var(--accent); display:flex; gap:8px; flex-wrap:wrap; }
.alert-session-link { color: var(--accent); text-decoration:none; cursor:pointer; }
.alert-session-link:hover { text-decoration:underline; }
.alert-dismiss { background:transparent; border:none; color:var(--text2); cursor:pointer; font-size:16px; line-height:1; padding:0 4px; }
.alert-dismiss:hover { color: var(--text); }

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

.tab-bar {
  display: flex; gap: 4px; margin-bottom: 20px;
  border-bottom: 1px solid var(--border);
}
.tab-btn {
  background: transparent; color: var(--text2); border: none;
  padding: 8px 16px; font-family: inherit; font-size: 12px;
  cursor: pointer; border-bottom: 2px solid transparent;
  transition: all .15s;
}
.tab-btn:hover { color: var(--text); }
.tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }
.tab-panel { display: none; }
.tab-panel.active { display: block; }

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
.csv-btn {
  background: var(--surface2); color: var(--text2); border: 1px solid var(--border);
  padding: 4px 10px; border-radius: 4px; font-family: inherit; font-size: 11px;
  cursor: pointer; transition: all .15s;
}
.csv-btn:hover { background: var(--surface); color: var(--text); }
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
.scorecard-bar-wrap {
  width: 100%; height: 6px; background: var(--surface2);
  border-radius: 999px; overflow: hidden; margin-top: 4px;
}
.scorecard-bar { height: 100%; background: var(--accent); border-radius: 999px; }

/* Subagent graph */
.subagent-header { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom: 12px; }
.subagent-controls { color: var(--text2); font-size: 12px; }
.subagent-grid { display:grid; grid-template-columns: 1.3fr 1fr; gap: 16px; margin-top: 16px; }
.subagent-tree, .subagent-chart-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
.subagent-tree h3, .subagent-chart-card h3 { margin: 0 0 12px; }
.subagent-tree-wrap { display:flex; flex-direction:column; gap: 8px; }
.subagent-node { border: 1px solid var(--border); border-radius: 8px; background: var(--bg); overflow: hidden; }
.subagent-session { cursor:pointer; padding: 10px 12px; display:flex; align-items:center; justify-content:space-between; gap: 10px; }
.subagent-session summary { list-style:none; }
.subagent-session::-webkit-details-marker { display:none; }
.subagent-session-title { font-weight: 600; }
.subagent-session-meta { color: var(--text2); font-size: 12px; }
.subagent-children { padding: 0 0 4px; }
.subagent-delegate { margin: 8px 8px 0; padding: 10px 12px; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; }
.subagent-delegate-row { display:flex; align-items:flex-start; justify-content:space-between; gap: 12px; }
.subagent-delegate-main { min-width: 0; flex: 1; }
.subagent-delegate-goal { font-weight: 500; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.subagent-delegate-meta { display:flex; flex-wrap:wrap; gap: 6px; margin-top: 6px; }
.subagent-depth { margin-left: 18px; }
.subagent-empty { color: var(--text2); padding: 18px 4px; }
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
.modal-overlay {
  position: fixed; inset: 0; background: rgba(0,0,0,0.7);
  display: none; align-items: center; justify-content: center; z-index: 100;
}
.modal-overlay.active { display: flex; }
.modal {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; width: 90%; max-width: 900px; max-height: 85vh;
  overflow: hidden; display: flex; flex-direction: column;
}
.modal-header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 14px 20px; border-bottom: 1px solid var(--border);
}
.modal-body { padding: 16px 20px; overflow-y: auto; }
.modal-close { background: transparent; color: var(--text2); border: none; font-size: 18px; cursor: pointer; }
.timeline { display: flex; flex-direction: column; gap: 8px; }
.timeline-item {
  display: flex; gap: 12px; padding: 10px 12px;
  background: var(--surface2); border-radius: 8px; border-left: 3px solid var(--border);
}
.timeline-item.user { border-left-color: var(--accent); }
.timeline-item.assistant { border-left-color: var(--green); }
.timeline-item.tool { border-left-color: var(--orange); }
.timeline-time { font-size: 10px; color: var(--text2); white-space: nowrap; min-width: 70px; }
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <div class="header-left">
      <h1>⚡ <span>Hermes</span> Agent Dashboard</h1>
      <div class="status">
        <span class="status-dot" id="ws-dot"></span>
        <span id="ws-label">connecting...</span>
      </div>
      <div class="alert-wrap">
        <button class="alert-badge" id="alert-badge" type="button" onclick="toggleAlertsPanel()">🚨 <span id="alert-count">0 alerts</span></button>
        <div class="alert-panel" id="alert-panel"></div>
      </div>
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

  <div style="display:flex;gap:8px;margin-bottom:16px;">
    <input type="text" id="global-search" placeholder="🔍 Search sessions, models, tools, platforms..."
      style="flex:1;background:var(--surface2);color:var(--text);border:1px solid var(--border);padding:6px 12px;border-radius:6px;font-family:inherit;font-size:12px;">
  </div>

  <div class="tab-bar">
    <button class="tab-btn active" data-tab="overview">Overview</button>
    <button class="tab-btn" data-tab="sessions">Sessions</button>
    <button class="tab-btn" data-tab="tools">Tools</button>
    <button class="tab-btn" data-tab="subagents">Subagents</button>
    <button class="tab-btn" data-tab="trends">Trends</button>
    <button class="tab-btn" data-tab="analytics">Analytics</button>
  </div>

  <div id="tab-overview" class="tab-panel active">
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

    <!-- Model scorecard -->
    <div class="table-section">
      <div class="table-header">
        <h3>🏅 Model Scorecard</h3>
      </div>
      <div style="overflow-x:auto;">
        <table id="scorecard-table">
          <thead>
            <tr>
              <th data-key="model">Model <span class="sort-arrow">▼</span></th>
              <th data-key="sessions">Sessions <span class="sort-arrow">▼</span></th>
              <th data-key="avg_messages">Avg Msgs <span class="sort-arrow">▼</span></th>
              <th data-key="avg_tool_calls">Avg Tools <span class="sort-arrow">▼</span></th>
              <th data-key="tool_success_rate">Success Rate <span class="sort-arrow">▼</span></th>
              <th data-key="input_output_ratio">I/O Ratio <span class="sort-arrow">▼</span></th>
              <th data-key="total_cost_usd">Cost <span class="sort-arrow">▼</span></th>
            </tr>
          </thead>
          <tbody id="scorecard-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <div id="tab-sessions" class="tab-panel">
    <!-- Sessions table -->
    <div class="table-section">
      <div class="table-header">
        <h3 data-i18n="sec_sessions">📋 Все сессии</h3>
        <button onclick="exportSessionsCSV()" class="csv-btn" data-i18n="btn_export">Export CSV</button>
      </div>
      <div style="overflow-x:auto;">
        <table id="sessions-table">
          <thead>
            <tr>
              <th>Pin</th>
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
        <button onclick="exportDurationCSV()" class="csv-btn" data-i18n="btn_export">Export CSV</button>
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
        <button onclick="exportTokensCSV()" class="csv-btn" data-i18n="btn_export">Export CSV</button>
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

  <div id="tab-tools" class="tab-panel">
    <div class="table-section">
      <div class="table-header"><h3>🔧 Tool frequency</h3></div>
      <canvas id="tools-frequency-chart" height="140"></canvas>
    </div>
    <div class="table-section">
      <div class="table-header"><h3>🧩 Tool correlation heatmap</h3></div>
      <div style="overflow-x:auto;" id="tools-heatmap-wrap"></div>
    </div>
    <div class="table-section">
      <div class="table-header"><h3>⏰ Hourly activity</h3></div>
      <canvas id="tools-hourly-chart" height="140"></canvas>
    </div>
    <div class="table-section">
      <div class="table-header"><h3>📋 Top sessions by tool usage</h3></div>
      <div style="overflow-x:auto;">
        <table id="tools-sessions-table">
          <thead>
            <tr>
              <th>File</th>
              <th>Tool count</th>
              <th>Tools used</th>
            </tr>
          </thead>
          <tbody id="tools-sessions-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <div id="tab-subagents" class="tab-panel">
    <div id="subagents-root">
      <div class="subagent-header">
        <div>
          <h2 style="margin:0 0 4px">Subagent graph</h2>
          <div class="subagent-controls">Tree view, KPIs, top goal keywords, and toolset mix</div>
        </div>
      </div>
      <div class="kpi-grid" id="subagents-kpis"></div>
      <div class="subagent-grid">
        <div class="subagent-tree">
          <h3>Tree view</h3>
          <div id="subagents-tree" class="subagent-tree-wrap"></div>
        </div>
        <div class="subagent-chart-card">
          <h3>Toolsets distribution</h3>
          <canvas id="subagents-toolsets-chart" height="220"></canvas>
        </div>
      </div>
      <div class="charts-grid" style="margin-top:16px">
        <div class="chart-card">
          <div class="chart-title"><h3>Top goal keywords</h3></div>
          <canvas id="subagents-goals-chart" height="220"></canvas>
        </div>
      </div>
    </div>
  </div>

  <div id="tab-trends" class="tab-panel">
    <div id="trends-root">
      <div class="subagent-header">
        <div>
          <h2 style="margin:0 0 4px">Trends</h2>
          <div class="subagent-controls">Period-over-period KPI deltas, daily session trend, and metric comparison</div>
        </div>
      </div>
      <div class="kpi-grid" id="trends-kpis"></div>
      <div class="charts-grid">
        <div class="chart-card">
          <div class="chart-title"><h3>Daily sessions: current vs previous period</h3></div>
          <canvas id="trends-daily-chart" height="220"></canvas>
        </div>
        <div class="chart-card">
          <div class="chart-title"><h3>Current vs previous comparison</h3></div>
          <canvas id="trends-bar-chart" height="220"></canvas>
        </div>
      </div>
      <div id="trends-status" style="color:var(--text2)"></div>
    </div>
  </div>

  <div id="tab-analytics" class="tab-panel">
    <div class="charts-grid">
      <div class="chart-card">
        <div class="chart-title">Activity by hour</div>
        <canvas id="chart-hourly"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-title">Platform distribution</div>
        <canvas id="chart-platforms"></canvas>
      </div>
      <div class="chart-card full">
        <div class="chart-title">Platform activity by hour</div>
        <canvas id="chart-platform-hours"></canvas>
      </div>
    </div>
  </div>

</div>

<div class="modal-overlay" id="session-modal">
  <div class="modal">
    <div class="modal-header">
      <h3 id="modal-title">Session</h3>
      <button class="modal-close" onclick="closeSessionModal()">✕</button>
    </div>
    <div class="modal-body" id="modal-body">Loading...</div>
  </div>
</div>

<script>
function showTab(tab) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  const btn = document.querySelector(`.tab-btn[data-tab="${tab}"]`);
  if (btn) btn.classList.add('active');
  const panel = document.getElementById('tab-' + tab);
  if (panel) panel.classList.add('active');
  if (tab === 'tools') renderTools();
  if (tab === 'subagents') renderSubagents();
  if (tab === 'trends') renderTrends();
  if (tab === 'analytics') renderAnalyticsTab();
}

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => showTab(btn.dataset.tab));
});

function alertHash(alert) {
  const raw = `${alert.metric || ''}:${alert.session_file || ''}:${alert.value ?? ''}`;
  try { return btoa(unescape(encodeURIComponent(raw))); } catch(e) { return btoa(raw); }
}

function toggleAlertsPanel(force) {
  alertsPanelOpen = typeof force === 'boolean' ? force : !alertsPanelOpen;
  const panel = document.getElementById('alert-panel');
  if (panel) panel.classList.toggle('open', alertsPanelOpen);
  if (alertsPanelOpen) renderAlerts();
}

function dismissAlert(hash) {
  dismissedAlerts.add(hash);
  localStorage.setItem('dash-dismissed-alerts', JSON.stringify([...dismissedAlerts]));
  renderAlerts();
}

function renderAlerts() {
  const badge = document.getElementById('alert-badge');
  const countEl = document.getElementById('alert-count');
  const panel = document.getElementById('alert-panel');
  if (!badge || !countEl || !panel) return;
  const alerts = (alertsData && alertsData.alerts) ? alertsData.alerts : [];
  const visible = alerts.filter(a => !dismissedAlerts.has(alertHash(a))).slice(0, 10);
  const totalVisible = visible.length;
  countEl.textContent = `${totalVisible} alert${totalVisible === 1 ? '' : 's'}`;
  badge.classList.toggle('has-alerts', totalVisible > 0);
  badge.style.opacity = totalVisible === 0 ? '0.7' : '1';
  panel.innerHTML = `
    <div class="alert-panel-header">
      <div class="alert-panel-title">Anomaly alerts${alertsData ? ` (${alertsData.summary?.total_alerts || 0})` : ''}</div>
      <button class="alert-dismiss" type="button" onclick="toggleAlertsPanel(false)">×</button>
    </div>
    ${visible.length === 0 ? '<div class="alert-panel-empty">No anomalies detected</div>' : visible.map(alert => {
      const hash = alertHash(alert);
      const sevClass = alert.severity === 'high' ? 'alert-severity-high' : 'alert-severity-medium';
      const msg = (alert.message || '').slice(0, 80);
      const session = (alert.session_file || '').replace(/\.jsonl$/, '');
      return `<div class="alert-row">
        <span class="alert-severity-dot ${sevClass}"></span>
        <div class="alert-row-main">
          <div class="alert-row-title">${esc(alert.metric || 'alert')}</div>
          <div class="alert-row-message">${esc(msg)}${(alert.message || '').length > 80 ? '…' : ''}</div>
          <div class="alert-row-meta">
            <a class="alert-session-link" href="#" onclick="openSession(${JSON.stringify(session)}); return false;">${esc(alert.session_file || session)}</a>
          </div>
        </div>
        <button class="alert-dismiss" type="button" title="Dismiss" onclick="dismissAlert('${hash}')">×</button>
      </div>`;
    }).join('')}
  `;
}

function openSession(file) {
  return openSessionModal(file);
}

async function openSessionModal(file) {
  const overlay = document.getElementById('session-modal');
  const body = document.getElementById('modal-body');
  const title = document.getElementById('modal-title');
  overlay.classList.add('active');
  body.innerHTML = '<p style="color:var(--text2)">Loading...</p>';
  try {
    const r = await fetch('/api/session/' + file.replace('.jsonl', ''));
    const data = await r.json();
    if (!data.ok) { body.innerHTML = '<p>Error: ' + esc(data.error) + '</p>'; return; }
    title.textContent = 'Session: ' + esc(file);
    let html = '<div class="timeline">';
    for (const turn of data.turns) {
      const roleClass = turn.role || 'unknown';
      html += `<div class="timeline-item ${esc(roleClass)}">
        <div class="timeline-time">${esc((turn.timestamp || '').slice(11, 19))}</div>
        <div style="flex:1">
          <strong style="text-transform:uppercase;font-size:11px;color:var(--text2)">${esc(turn.role)}</strong>
          <p style="margin:4px 0 0;font-size:12px">${esc(turn.content_preview)}</p>`;
      if (turn.tool_calls) {
        for (const tc of turn.tool_calls) {
          html += `<div style="margin-top:6px;padding:6px;background:var(--bg);border-radius:4px;font-size:11px">
            <code style="color:var(--orange)">${esc(tc.name)}</code>
            <div style="color:var(--text2);margin-top:2px">${esc(tc.arguments_preview)}</div>
          </div>`;
        }
      }
      if (turn.tool_result_preview) {
        html += `<div style="margin-top:4px;font-size:11px;color:var(--green)">→ ${esc(turn.tool_result_preview)}</div>`;
      }
      html += '</div></div>';
    }
    html += '</div>';
    body.innerHTML = html;
  } catch(e) { body.innerHTML = '<p style="color:var(--red)">Failed to load session</p>'; }
}
function closeSessionModal() {
  document.getElementById('session-modal').classList.remove('active');
}
document.getElementById('session-modal').addEventListener('click', (e) => {
  if (e.target.id === 'session-modal') closeSessionModal();
});

// ── State ──────────────────────────────────────────────────────────────
const pinnedSessions = new Set(JSON.parse(localStorage.getItem('dash-pins') || '[]'));
function togglePin(file) {
  if (pinnedSessions.has(file)) pinnedSessions.delete(file);
  else pinnedSessions.add(file);
  localStorage.setItem('dash-pins', JSON.stringify([...pinnedSessions]));
  renderSessionsTable();
}
let stats = null;
let ws = null;
let alertsData = null;
let alertsPanelOpen = false;
let dismissedAlerts = new Set(JSON.parse(localStorage.getItem('dash-dismissed-alerts') || '[]'));
let chartModels = null, chartTools = null, chartDays = null, chartTokenDays = null, chartTokenModels = null, chartHourly = null, chartPlatforms = null, chartPlatformHours = null;
let chartToolsFrequency = null, chartToolsHourly = null, toolsData = null, toolsDataDays = null;
let chartSubagentsGoals = null, chartSubagentsToolsets = null, subagentsData = null, subagentsDataDays = null;
let chartTrendsDaily = null, chartTrendsBar = null, trendsData = null, trendsDataDays = null;
let sortKey = 'date'; let sortDir = -1;
let scorecardSortKey = 'sessions'; let scorecardSortDir = -1;

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
      btn_refresh:'Обновить', filter_ph:'Фильтр по модели...', btn_export:'Export CSV',
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
      btn_refresh:'Refresh', filter_ph:'Filter by model...', btn_export:'Export CSV',
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
async function fetchAlerts(days) {
  try {
    const r = await fetch('/api/alerts?days=' + days);
    alertsData = await r.json();
  } catch(e) {
    console.error(e);
    alertsData = { ok: false, alerts: [], summary: { total_alerts: 0, by_metric: {} } };
  }
  renderAlerts();
}

async function refresh() {
  const days = document.getElementById('period-select').value;
  try {
    const r = await fetch('/api/stats?days=' + days);
    stats = await r.json();
    toolsData = null;
    toolsDataDays = null;
    await fetchAlerts(days);
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
  renderScorecardTable();
  renderAlerts();
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

function renderTools() {
  const days = document.getElementById('period-select').value;
  if (toolsData && toolsDataDays === days) {
    _renderToolsData(toolsData);
    return;
  }
  const wrap = document.getElementById('tab-tools');
  const existing = wrap.querySelector('.tools-loading');
  if (!existing) wrap.insertAdjacentHTML('afterbegin', '<p class="tools-loading" style="color:var(--text2)">Loading tool analytics...</p>');
  fetch('/api/tools?days=' + days)
    .then(r => r.json())
    .then(data => {
      toolsData = data;
      toolsDataDays = days;
      const loading = wrap.querySelector('.tools-loading');
      if (loading) loading.remove();
      _renderToolsData(data);
    })
    .catch(() => {
      const loading = wrap.querySelector('.tools-loading');
      if (loading) loading.remove();
      document.getElementById('tools-heatmap-wrap').innerHTML = '<p style="color:var(--text2)">No tool usage data for this period</p>';
    });
}

function _renderToolsData(data) {
  const noData = !data || !data.ok || !data.tool_frequency || !data.tool_frequency.length;
  if (noData) {
    document.getElementById('tools-heatmap-wrap').innerHTML = '<p style="color:var(--text2)">No tool usage data for this period</p>';
    document.getElementById('tools-sessions-tbody').innerHTML = '';
    if (chartToolsFrequency) chartToolsFrequency.destroy();
    if (chartToolsHourly) chartToolsHourly.destroy();
    return;
  }
  const topFreq = data.tool_frequency.slice(0, 15);
  const maxCount = Math.max(...topFreq.map(x => x.count), 1);
  if (chartToolsFrequency) chartToolsFrequency.destroy();
  chartToolsFrequency = new Chart(document.getElementById('tools-frequency-chart'), {
    type: 'bar',
    data: {
      labels: topFreq.map(x => `${x.tool} (${x.count})`),
      datasets: [{ data: topFreq.map(x => x.count), backgroundColor: topFreq.map(x => `rgba(108,140,255,${0.25 + 0.75 * (x.count / maxCount)})`), borderRadius: 4 }]
    },
    options: {
      indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: {
        x: { beginAtZero: true, ticks: { color: '#8b90a5' }, grid: { color: '#2e3348' } },
        y: { ticks: { color: '#e1e4ed' }, grid: { display: false } }
      }
    }
  });

  const topTools = data.tool_frequency.slice(0, 10).map(x => x.tool);
  const scores = new Map();
  (data.correlation || []).forEach(c => {
    scores.set(`${c.tool_a}||${c.tool_b}`, c.score);
    scores.set(`${c.tool_b}||${c.tool_a}`, c.score);
  });
  let html = '<table><thead><tr><th>Tool</th>' + topTools.map(t => `<th>${esc(t)}</th>`).join('') + '</tr></thead><tbody>';
  topTools.forEach(a => {
    html += `<tr><th>${esc(a)}</th>`;
    topTools.forEach(b => {
      const score = a === b ? 1 : (scores.get(`${a}||${b}`) || 0);
      const alpha = a === b ? 0.95 : Math.max(0.08, score);
      html += `<td style="background:rgba(108,140,255,${alpha});text-align:center;font-variant-numeric:tabular-nums">${score.toFixed(2)}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  document.getElementById('tools-heatmap-wrap').innerHTML = html;

  const hours = [...Array(24)].map((_, i) => String(i).padStart(2, '0'));
  const hourly = data.hourly || {};
  const top5 = topFreq.slice(0, 5).map(x => x.tool);
  const used = new Set(top5);
  const others = hours.map((_, idx) => Object.entries(hourly).reduce((sum, [tool, arr]) => sum + ((used.has(tool) ? 0 : (arr && arr[idx]) || 0)), 0));
  if (chartToolsHourly) chartToolsHourly.destroy();
  chartToolsHourly = new Chart(document.getElementById('tools-hourly-chart'), {
    type: 'line',
    data: { labels: hours, datasets: [
      ...top5.map((tool, i) => ({ label: tool, data: (hourly[tool] || []).slice(0, 24), borderColor: COLORS[i % COLORS.length], backgroundColor: COLORS[i % COLORS.length], tension: 0.25, fill: false })),
      { label: 'others', data: others, borderColor: '#8b90a5', backgroundColor: '#8b90a5', tension: 0.25, fill: false }
    ] },
    options: { responsive: true, plugins: { legend: { labels: { color: '#e1e4ed' } } }, scales: { x: { ticks: { color: '#8b90a5' }, grid: { color: '#2e3348' } }, y: { beginAtZero: true, ticks: { color: '#8b90a5' }, grid: { color: '#2e3348' } } } }
  });

  document.getElementById('tools-sessions-tbody').innerHTML = (data.top_sessions || []).map(r => `<tr><td>${esc(r.file)}</td><td class="num">${r.tool_count}</td><td>${(r.tools || []).map(esc).join(', ')}</td></tr>`).join('') || `<tr><td colspan="3" style="text-align:center;color:var(--text2);padding:20px">No tool usage data for this period</td></tr>`;
}

function _subagentBadge(text, cls='badge-delegate') { return `<span class="badge ${cls}">${esc(text)}</span>`; }
function _subagentDepthLabel(depth) { return `Depth ${depth || 0}`; }
function _subagentNodeHtml(node, depth=0) {
  const hasChildren = Array.isArray(node.children) && node.children.length > 0;
  const sessionLabel = esc(node.session_file || node.id || 'session');
  const childrenHtml = hasChildren ? node.children.map(child => _subagentNodeHtml(child, depth + 1)).join('') : '';
  const toolsetBadges = (node.toolsets || []).map(ts => _subagentBadge(ts)).join(' ');
  const goal = esc(node.goal && node.goal.trim() ? node.goal : '(batch)');
  const tasks = node.tasks_count != null ? node.tasks_count : 0;
  const status = (node.status || 'unknown').toLowerCase();
  const statusCls = status === 'completed' ? 'badge-direct' : (status === 'failed' ? 'badge-subagent' : 'badge-delegate');
  const open = depth === 0 ? ' open' : '';
  return `<details class="subagent-node subagent-depth"${open} style="margin-left:${depth * 18}px">
    <summary class="subagent-session">
      <div>
        <div class="subagent-session-title">${sessionLabel}</div>
        <div class="subagent-session-meta">${_subagentDepthLabel(depth)} • ${hasChildren ? node.children.length + ' delegates' : 'delegate leaf'}</div>
      </div>
      <div class="subagent-session-meta">${hasChildren ? 'collapse/expand' : 'leaf'}</div>
    </summary>
    <div class="subagent-children">
      ${hasChildren ? childrenHtml : ''}
    </div>
  </details>`;
}

function renderSubagents() {
  const days = document.getElementById('period-select').value;
  if (subagentsData && subagentsDataDays === days) { _renderSubagentsData(subagentsData); return; }
  const root = document.getElementById('subagents-root');
  const existing = root.querySelector('.subagents-loading');
  if (!existing) root.insertAdjacentHTML('afterbegin', '<p class="subagents-loading" style="color:var(--text2)">Loading subagent graph...</p>');
  fetch('/api/subagents?days=' + days)
    .then(r => r.json())
    .then(data => {
      subagentsData = data;
      subagentsDataDays = days;
      const loading = root.querySelector('.subagents-loading');
      if (loading) loading.remove();
      _renderSubagentsData(data);
    })
    .catch(() => {
      const loading = root.querySelector('.subagents-loading');
      if (loading) loading.remove();
      document.getElementById('subagents-kpis').innerHTML = '';
      document.getElementById('subagents-tree').innerHTML = '<div class="subagent-empty">No delegate calls for this period</div>';
      if (chartSubagentsGoals) chartSubagentsGoals.destroy();
      if (chartSubagentsToolsets) chartSubagentsToolsets.destroy();
    });
}

function _renderSubagentsData(data) {
  const ok = data && data.ok;
  const ag = (data && data.aggregates) || {};
  const kpis = [
    ['Total Delegates', ag.total_delegates ?? 0, ''],
    ['Success Rate', `${Math.round((ag.success_rate || 0) * 100)}%`, ''],
    ['Max Depth', ag.max_depth ?? 0, ''],
    ['Avg Depth', (ag.avg_depth ?? 0).toFixed(1), ''],
    ['Avg Duration', ag.avg_duration_ms ? (ag.avg_duration_ms >= 1000 ? `${(ag.avg_duration_ms / 1000).toFixed(1)}s` : `${Math.round(ag.avg_duration_ms)}ms`) : '—', ''],
  ];
  document.getElementById('subagents-kpis').innerHTML = kpis.map(([label, value, sub]) => `<div class="kpi-card"><div class="kpi-label">${label}</div><div class="kpi-value">${value}</div><div class="kpi-sub">${sub}</div></div>`).join('');

  const tree = document.getElementById('subagents-tree');
  const roots = (ok && Array.isArray(data.tree)) ? data.tree : [];
  if (!roots.length) {
    tree.innerHTML = '<div class="subagent-empty">No delegate calls for this period</div>';
  } else {
    const renderRoot = (n) => {
      const children = (n.children || []).map(child => renderDelegateNode(child, 1)).join('');
      return `<details class="subagent-node" open><summary class="subagent-session"><div><div class="subagent-session-title">${esc(n.session_file || 'session')}</div><div class="subagent-session-meta">${(n.children || []).length} delegate roots</div></div><div class="subagent-session-meta">session</div></summary><div class="subagent-children">${children}</div></details>`;
    };
    const renderDelegateNode = (node, depth) => {
      const hasChildren = Array.isArray(node.children) && node.children.length > 0;
      const goal = esc((node.goal && node.goal.trim()) ? node.goal : '(batch)');
      const toolsets = (node.toolsets || []).map(ts => `<span class="badge badge-delegate">${esc(ts)}</span>`).join(' ');
      const status = esc(node.status || 'unknown');
      const statusCls = status === 'completed' ? 'badge-direct' : (status === 'failed' ? 'badge-subagent' : 'badge-delegate');
      return `<div class="subagent-delegate" style="margin-left:${depth * 18}px">
        <div class="subagent-delegate-row">
          <div class="subagent-delegate-main">
            <div class="subagent-delegate-goal" title="${goal}">${goal}</div>
            <div class="subagent-delegate-meta">${_subagentBadge(`tasks: ${node.tasks_count || 0}`)} ${_subagentBadge(status, statusCls)} ${toolsets}</div>
          </div>
        </div>
        ${hasChildren ? node.children.map(child => renderDelegateNode(child, depth + 1)).join('') : ''}
      </div>`;
    };
    tree.innerHTML = roots.map(renderRoot).join('');
  }

  const goalWords = (data.top_goals || []).slice(0, 15);
  if (chartSubagentsGoals) chartSubagentsGoals.destroy();
  chartSubagentsGoals = new Chart(document.getElementById('subagents-goals-chart'), {
    type: 'bar',
    data: { labels: goalWords.map(x => x.word), datasets: [{ data: goalWords.map(x => x.count), backgroundColor: '#6c8cffcc', borderRadius: 4 }] },
    options: { indexAxis: 'y', responsive: true, plugins: { legend: { display: false } }, scales: { x: { beginAtZero: true, ticks: { color: '#8b90a5' }, grid: { color: '#2e3348' } }, y: { ticks: { color: '#e1e4ed' }, grid: { display: false } } } }
  });

  const ts = data.toolsets_distribution || {};
  if (chartSubagentsToolsets) chartSubagentsToolsets.destroy();
  chartSubagentsToolsets = new Chart(document.getElementById('subagents-toolsets-chart'), {
    type: 'doughnut',
    data: { labels: Object.keys(ts), datasets: [{ data: Object.values(ts), backgroundColor: COLORS.slice(0, Object.keys(ts).length) }] },
    options: { responsive: true, plugins: { legend: { position: 'bottom', labels: { color: '#e1e4ed' } } } }
  });
}

function renderTrends() {
  const days = document.getElementById('period-select').value;
  const root = document.getElementById('tab-trends');
  const status = document.getElementById('trends-status');
  if (trendsData && trendsDataDays === days) {
    _renderTrendsData(trendsData);
    return;
  }
  if (status) status.textContent = 'Loading trends...';
  fetch('/api/trends?days=' + days)
    .then(r => r.json())
    .then(data => {
      trendsData = data;
      trendsDataDays = days;
      _renderTrendsData(data);
    })
    .catch(() => {
      if (status) status.textContent = 'No trend data for this period';
      document.getElementById('trends-kpis').innerHTML = '';
      if (chartTrendsDaily) chartTrendsDaily.destroy();
      if (chartTrendsBar) chartTrendsBar.destroy();
    });
}

function _renderTrendsData(data) {
  const metrics = [
    { key: 'sessions', label: 'Sessions', color: '#6c8cff' },
    { key: 'user_msgs', label: 'User Messages', color: '#a78bfa' },
    { key: 'tool_calls', label: 'Tool Calls', color: '#fb923c' },
    { key: 'delegates', label: 'Delegates', color: '#f87171' },
    { key: 'input_tokens', label: 'Input Tokens', color: '#fbbf24' },
    { key: 'output_tokens', label: 'Output Tokens', color: '#2dd4bf' },
  ];
  const d = data && data.deltas ? data.deltas : {};
  const current = data && data.current ? data.current : {};
  const deltaCard = (m) => {
    const delta = d[m.key] || {};
    const trend = delta.trend || 'flat';
    const arrow = trend === 'up' ? '↑' : trend === 'down' ? '↓' : '↔';
    const color = trend === 'up' ? 'var(--green)' : trend === 'down' ? 'var(--red)' : 'var(--text2)';
    const pct = Number.isFinite(delta.pct) ? `${delta.pct.toFixed(1)}%` : 'N/A';
    return `<div class="kpi-card" style="border-top:3px solid ${m.color}">
      <div class="kpi-label">${m.label}</div>
      <div class="kpi-value" style="color:${color}">${fmt(current[m.key] ?? 0)}</div>
      <div class="kpi-sub" style="color:${color}">${arrow} ${delta.value ?? 0} (${pct})</div>
    </div>`;
  };
  document.getElementById('trends-kpis').innerHTML = metrics.map(deltaCard).join('');

  const daily = Array.isArray(data.daily_trend) ? data.daily_trend : [];
  const labels = daily.map(x => x.day);
  const currentDaily = daily.map(x => x.current ?? 0);
  const previousDaily = daily.map(x => x.previous ?? 0);
  if (chartTrendsDaily) chartTrendsDaily.destroy();
  chartTrendsDaily = new Chart(document.getElementById('trends-daily-chart'), {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'current', data: currentDaily, borderColor: '#6c8cff', backgroundColor: '#6c8cff33', tension: 0.25, fill: false, pointRadius: 3 },
        { label: 'previous', data: previousDaily, borderColor: '#8b90a5', backgroundColor: '#8b90a533', borderDash: [6, 4], tension: 0.25, fill: false, pointRadius: 3 },
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#e1e4ed' } } },
      scales: {
        x: { ticks: { color: '#8b90a5' }, grid: { color: '#2e3348' } },
        y: { beginAtZero: true, ticks: { color: '#8b90a5', stepSize: 1 }, grid: { color: '#2e3348' } }
      }
    }
  });

  const keys = metrics.map(m => m.key);
  if (chartTrendsBar) chartTrendsBar.destroy();
  chartTrendsBar = new Chart(document.getElementById('trends-bar-chart'), {
    type: 'bar',
    data: {
      labels: metrics.map(m => m.label),
      datasets: [
        { label: 'current', data: keys.map(k => current[k] ?? 0), backgroundColor: '#6c8cffcc', borderRadius: 4 },
        { label: 'previous', data: keys.map(k => (data.previous || {})[k] ?? 0), backgroundColor: '#8b90a566', borderRadius: 4 },
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#e1e4ed' } } },
      scales: {
        x: { ticks: { color: '#8b90a5' }, grid: { display: false } },
        y: { beginAtZero: true, ticks: { color: '#8b90a5' }, grid: { color: '#2e3348' } }
      }
    }
  });
  const status = document.getElementById('trends-status');
  if (status) status.textContent = '';
}

// ── Sessions table ─────────────────────────────────────────────────────
function renderSessionsTable() {
  const search = document.getElementById('global-search').value.toLowerCase();
  let rows = stats.sessions;
  if (search) {
    rows = rows.filter(r =>
      (r.model || '').toLowerCase().includes(search) ||
      (r.platform || '').toLowerCase().includes(search) ||
      (r.file || '').toLowerCase().includes(search)
    );
  }
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
    const isPinned = pinnedSessions.has(r.file);
    const pinCell = `<td><button onclick='togglePin(${JSON.stringify(r.file)})' style="background:transparent;border:none;cursor:pointer;font-size:14px">${isPinned ? '\u2605' : '\u2606'}</button></td>`;
    return `<tr>
      ${pinCell}
      <td class="num" style="cursor:pointer;text-decoration:underline" onclick="openSessionModal('${r.file}')">${r.mtime}</td>
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

function renderScorecardTable() {
  const el = document.getElementById('scorecard-tbody');
  const perf = stats.model_performance || {};
  const rows = Object.entries(perf)
    .filter(([model]) => model !== '?')
    .map(([model, p]) => ({ model, ...(p || {}) }));
  if (!rows.length) {
    el.innerHTML = `<tr><td colspan="7" style="text-align:center;color:var(--text2);padding:20px">No model data available</td></tr>`;
    return;
  }
  const maxSessions = Math.max(...rows.map(r => r.sessions || 0), 1);
  rows.sort((a, b) => {
    let va = a[scorecardSortKey], vb = b[scorecardSortKey];
    if (va == null) va = -Infinity;
    if (vb == null) vb = -Infinity;
    if (typeof va === 'string') return scorecardSortDir * va.localeCompare(vb);
    return scorecardSortDir * (va - vb);
  });
  document.querySelectorAll('#scorecard-table th').forEach(h => h.classList.remove('sorted'));
  const active = document.querySelector(`#scorecard-table th[data-key="${scorecardSortKey}"]`);
  if (active) active.classList.add('sorted');
  el.innerHTML = rows.map(r => {
    const sessions = r.sessions || 0;
    const width = Math.max(3, (sessions / maxSessions) * 100);
    const rate = Number.isFinite(r.tool_success_rate) ? `${(r.tool_success_rate * 100).toFixed(0)}%` : '—';
    const io = Number.isFinite(r.input_output_ratio) ? r.input_output_ratio.toFixed(2) : '—';
    const cost = Number.isFinite(r.total_cost_usd) ? `$${r.total_cost_usd.toFixed(2)}` : '—';
    return `<tr>
      <td><span class="badge badge-model">${esc(r.model)}</span></td>
      <td class="num"><div>${sessions}</div><div class="scorecard-bar-wrap"><div class="scorecard-bar" style="width:${width}%"></div></div></td>
      <td class="num">${Number.isFinite(r.avg_messages) ? r.avg_messages.toFixed(1) : '—'}</td>
      <td class="num">${Number.isFinite(r.avg_tool_calls) ? r.avg_tool_calls.toFixed(1) : '—'}</td>
      <td class="num">${rate}</td>
      <td class="num">${io}</td>
      <td class="num">${cost}</td>
    </tr>`;
  }).join('');
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

// ── Analytics tab charts ────────────────────────────────────────────
function renderAnalyticsTab() {
  if (!stats) return;

  // Hourly bar
  const hours = Object.keys(stats.by_hour || {}).sort();
  if (chartHourly) chartHourly.destroy();
  chartHourly = new Chart(document.getElementById('chart-hourly'), {
    type: 'bar',
    data: { labels: hours, datasets: [{ data: hours.map(h => stats.by_hour[h]), backgroundColor: '#6c8cff', borderRadius: 3 }] },
    options: { responsive: true, plugins: { legend: { display: false } },
      scales: { x: { grid: { display: false }, ticks: { color: '#8b90a5', font: { size: 10 } } },
                y: { grid: { color: '#2e3348' }, ticks: { color: '#8b90a5', font: { size: 10 } }, beginAtZero: true } } }
  });

  // Platforms donut
  const plats = stats.by_platform || {};
  if (chartPlatforms) chartPlatforms.destroy();
  chartPlatforms = new Chart(document.getElementById('chart-platforms'), {
    type: 'doughnut',
    data: { labels: Object.keys(plats), datasets: [{ data: Object.values(plats), backgroundColor: COLORS.slice(0, Object.keys(plats).length), borderWidth: 0 }] },
    options: { responsive: true, plugins: { legend: { position: 'right', labels: { color: '#8b90a5', font: { size: 11 }, padding: 12 } } } }
  });

  // Platform hours stacked
  const ph = stats.platform_hour || {};
  const platNames = Object.keys(ph);
  if (chartPlatformHours) chartPlatformHours.destroy();
  chartPlatformHours = new Chart(document.getElementById('chart-platform-hours'), {
    type: 'bar',
    data: {
      labels: hours,
      datasets: platNames.map((p, i) => ({
        label: p, data: hours.map(h => ph[p][h] || 0),
        backgroundColor: COLORS[i % COLORS.length] + 'aa',
        borderColor: COLORS[i % COLORS.length], borderWidth: 1, borderRadius: 2,
      }))
    },
    options: { responsive: true, plugins: { legend: { labels: { color: '#8b90a5', font: { size: 11 } } } },
      scales: { x: { stacked: true, grid: { color: '#2e3348' }, ticks: { color: '#8b90a5' } },
                y: { stacked: true, grid: { color: '#2e3348' }, ticks: { color: '#8b90a5' }, beginAtZero: true } } }
  });
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

// ── CSV Export ──────────────────────────────────────────────────
function exportCSV(filename, headers, rows) {
  const csv = [headers.join(','), ...rows.map(r => r.map(c => '"' + String(c).replace(/"/g, '""') + '"').join(','))].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

function exportSessionsCSV() {
  const headers = ['File', 'Model', 'Platform', 'UserMsgs', 'AsstMsgs', 'ToolCalls', 'DelegateCalls', 'DurationMin', 'Date'];
  const rows = stats.sessions.map(s => [s.file, s.model, s.platform, s.user_msgs, s.assistant_msgs, s.tool_calls, s.delegate_calls || 0, s.duration_min || '', s.mtime]);
  exportCSV('sessions.csv', headers, rows);
}

function exportDurationCSV() {
  const headers = ['DurationMin', 'Model', 'Tools', 'Deleg', 'User', 'Input', 'Output', 'Date'];
  const rows = (stats.by_duration || []).map(s => [s.duration_min, s.model, s.tool_calls, s.delegate_calls || 0, s.user_msgs, s.input_tokens || 0, s.output_tokens || 0, s.mtime]);
  exportCSV('duration.csv', headers, rows);
}

function exportTokensCSV() {
  const headers = ['Model', 'InputTokens', 'OutputTokens', 'CacheReadTokens', 'CostUSD', 'Date'];
  const rows = (stats.top_cost_sessions || []).map(r => [r.model, r.input_tokens, r.output_tokens, r.cache_read_tokens || 0, r.estimated_cost_usd || 0, r.started_at]);
  exportCSV('tokens.csv', headers, rows);
}

// ── Utils ──────────────────────────────────────────────────
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

document.querySelectorAll('#scorecard-table th[data-key]').forEach(th => {
  th.addEventListener('click', () => {
    const key = th.dataset.key;
    if (scorecardSortKey === key) scorecardSortDir *= -1;
    else { scorecardSortKey = key; scorecardSortDir = key === 'model' ? 1 : -1; }
    renderScorecardTable();
  });
});

// ── Filter ─────────────────────────────────────────────────────────────
document.getElementById('global-search').addEventListener('input', renderSessionsTable);

// ── Period change ──────────────────────────────────────────────────────
document.getElementById('period-select').addEventListener('change', refresh);

// ── Keyboard shortcuts ────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'r' || e.key === 'R') { e.preventDefault(); refresh(); }
  if (e.key === '/') { e.preventDefault(); document.getElementById('global-search').focus(); }
  if (['1','2','3','4','5'].includes(e.key)) {
    const idx = parseInt(e.key) - 1;
    const sel = document.getElementById('period-select');
    if (idx < sel.options.length) { sel.selectedIndex = idx; refresh(); }
  }
});

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
