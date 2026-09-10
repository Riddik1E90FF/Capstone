from __future__ import annotations
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path(__file__).resolve().parents[1]))
LOG_DIR = ROOT / "tracker" / "logs"
RUNTIME_DIR = ROOT / "tracker" / "runtime"
EVENT_LOG = LOG_DIR / "claude_events.jsonl"
STATE_FILE = RUNTIME_DIR / "state.json"
LOCK = Lock()

LOG_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def append_event(event):
    with LOCK:
        with EVENT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

def load_state():
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"sessions": {}, "turns": {}, "tools": {}}

def save_state(state):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)

def short_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def main():
    try:
        data = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        return 0

    event = data.get("hook_event_name", "unknown")
    session_id = data.get("session_id", "unknown")
    prompt_id = data.get("prompt_id")
    mono = time.monotonic()
    timestamp = now_iso()

    state = load_state()
    sessions = state.setdefault("sessions", {})
    turns = state.setdefault("turns", {})
    tools = state.setdefault("tools", {})

    record = {
        "timestamp_utc": timestamp,
        "event": event,
        "session_id": session_id,
        "prompt_id": prompt_id,
        "cwd": data.get("cwd"),
        "agent_id": data.get("agent_id"),
        "agent_type": data.get("agent_type"),
    }

    if event == "SessionStart":
        sessions[session_id] = {
            "start_iso": timestamp,
            "start_mono": mono,
            "source": data.get("source"),
        }
        record["model"] = data.get("model")
        record["effort"] = data.get("effort")

    elif event == "SessionEnd":
        start = sessions.get(session_id)
        if start:
            record["session_duration_ms"] = round((mono - start["start_mono"]) * 1000)
        record["reason"] = data.get("reason")

    elif event == "UserPromptSubmit":
        key = prompt_id or f"{session_id}:{mono}"
        turns[key] = {"start_iso": timestamp, "start_mono": mono}
        prompt = data.get("prompt", "")
        record["prompt_chars"] = len(prompt)
        record["prompt_hash"] = short_hash(prompt) if prompt else None

    elif event in ("Stop", "StopFailure"):
        start = turns.pop(prompt_id, None) if prompt_id else None
        if start:
            record["turn_duration_ms"] = round((mono - start["start_mono"]) * 1000)
            record["turn_start_iso"] = start["start_iso"]
        record["failure"] = event == "StopFailure"

    elif event == "PreToolUse":
        tool_id = data.get("tool_use_id") or f"{session_id}:{mono}"
        tools[tool_id] = {"start_mono": mono, "start_iso": timestamp}
        record["tool_use_id"] = tool_id
        record["tool_name"] = data.get("tool_name")

    elif event in ("PostToolUse", "PostToolUseFailure"):
        record["tool_use_id"] = data.get("tool_use_id")
        record["tool_name"] = data.get("tool_name")
        record["tool_duration_ms"] = data.get("duration_ms")
        record["success"] = event == "PostToolUse"

    elif event in ("SubagentStart", "SubagentStop"):
        record["agent_id"] = data.get("agent_id")
        record["agent_type"] = data.get("agent_type")

    append_event(record)
    save_state(state)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
