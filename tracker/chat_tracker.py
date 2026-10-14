from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACKER_DIR = ROOT / "tracker"
LOG_DIR = TRACKER_DIR / "logs"
RUNTIME_DIR = TRACKER_DIR / "runtime"
CHAT_LOG = LOG_DIR / "claude_chat_sessions.jsonl"
CHAT_STATE = RUNTIME_DIR / "chat_tracker_state.json"
CATEGORY_FILE = RUNTIME_DIR / "current_category.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_KEYWORDS = [
    "claude",
    "claude.ai",
]

SAMPLE_SECONDS = 5.0
DEFAULT_IDLE_MINUTES = 5


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def category():
    return read_json(CATEGORY_FILE, {"category": "Uncategorized"}).get(
        "category", "Uncategorized"
    )


def get_idle_seconds():
    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.c_uint),
            ("dwTime", ctypes.c_uint),
        ]

    info = LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(LASTINPUTINFO)

    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
        return 0.0

    tick = ctypes.windll.kernel32.GetTickCount()
    elapsed_ms = (tick - info.dwTime) & 0xFFFFFFFF
    return elapsed_ms / 1000.0


def get_foreground_title():
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return ""

    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""

    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def get_foreground_process_name():
    """Best-effort process name; failure is harmless."""
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return ""

    pid = ctypes.c_ulong()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if not pid.value:
        return ""

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not handle:
        return ""

    try:
        size = ctypes.c_ulong(1024)
        buf = ctypes.create_unicode_buffer(size.value)
        if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return Path(buf.value).name.lower()
    finally:
        kernel32.CloseHandle(handle)

    return ""


def load_state():
    return read_json(CHAT_STATE, {"active": False})


def save_state(state):
    write_json(CHAT_STATE, state)


def load_keywords():
    config = read_json(
        RUNTIME_DIR / "chat_tracker_config.json",
        {
            "title_keywords": DEFAULT_KEYWORDS,
            "process_names": ["claude.exe"],
            "idle_minutes": DEFAULT_IDLE_MINUTES,
            "sample_seconds": SAMPLE_SECONDS,
        },
    )
    return config


def looks_like_claude(title, process_name, config):
    title_l = (title or "").lower()
    process_l = (process_name or "").lower()

    for name in config.get("process_names", []):
        if process_l == name.lower():
            return True

    for keyword in config.get("title_keywords", []):
        if keyword.lower() in title_l:
            return True

    return False


def append_record(record):
    with CHAT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def flush_segment(state, end_mono, end_iso, reason):
    if state["segment_start_mono"] is None:
        state["segment_start_mono"] = None
        state["segment_active"] = False
        save_state(state)
        return
    duration_ms = max(
        0,
        round((end_mono - state["segment_start_mono"]) * 1000)
        )
    if not state.get("active"):
        return

    duration_ms = max(0, round((end_mono - state["segment_start_mono"]) * 1000))
    if duration_ms <= 0:
        return

    append_record({
        "session_id": state["session_id"],
        "start_utc": state["segment_start_utc"],
        "end_utc": end_iso,
        "duration_ms": duration_ms,
        "category": state.get("category", "Uncategorized"),
        "reason": reason,
        "source": "windows_foreground_activity",
    })


def run(idle_minutes=DEFAULT_IDLE_MINUTES, sample_seconds=SAMPLE_SECONDS):
    if os.name != "nt":
        print("Claude Chat activity tracking currently requires Windows.")
        return 1

    config = load_keywords()
    config["idle_minutes"] = idle_minutes
    config["sample_seconds"] = sample_seconds

    state = load_state()
    if state.get("active"):
        print("Claude Chat tracker already appears to be running.")
        print(f"  Session: {state.get('session_id')}")
        print("If this is stale, run: python tracker\\chat_tracker.py reset")
        return 1

    state = {
        "active": True,
        "session_id": str(uuid.uuid4()),
        "started_utc": now_iso(),
        "segment_start_utc": None,
        "segment_start_mono": None,
        "category": category(),
    }
    save_state(state)

    print("Claude Chat activity tracker running.")
    print("Tracking active foreground use of Claude desktop or a browser tab titled with 'Claude'.")
    print(f"Idle threshold: {idle_minutes} minute(s). Sample interval: {sample_seconds:g}s.")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            title = get_foreground_title()
            process = get_foreground_process_name()
            idle_seconds = get_idle_seconds()

            active_claude = (
                looks_like_claude(title, process, config)
                and idle_seconds < idle_minutes * 60
            )

            mono = time.monotonic()
            timestamp = now_iso()

            if active_claude and not state.get("segment_start_mono"):
                state["segment_start_mono"] = mono
                state["segment_start_utc"] = timestamp
                state["category"] = category()
                save_state(state)

            elif active_claude and state.get("segment_start_mono"):
                # If the user changes the project category while Claude remains active,
                # close the old segment and start a new one under the new category.
                current_category = category()
                if current_category != state.get("category"):
                    flush_segment(state, mono, timestamp, "category_change")
                    state["segment_start_mono"] = mono
                    state["segment_start_utc"] = timestamp
                    state["category"] = current_category
                    save_state(state)

            elif not active_claude and state.get("segment_start_mono"):
                flush_segment(state, mono, timestamp, "inactive_or_idle")
                state["segment_start_mono"] = None
                state["segment_start_utc"] = None
                save_state(state)

            time.sleep(sample_seconds)

    except KeyboardInterrupt:
        mono = time.monotonic()
        timestamp = now_iso()
        flush_segment(state, mono, timestamp, "tracker_stopped")
        state["active"] = False
        state["segment_start_mono"] = None
        state["segment_start_utc"] = None
        save_state(state)
        print("\nClaude Chat tracker stopped.")
        return 0


def status():
    state = load_state()
    print(f"Tracker running: {'YES' if state.get('active') else 'NO'}")
    if state.get("active"):
        print(f"Session: {state.get('session_id')}")
        print(f"Started: {state.get('started_utc')}")
    print(f"Current project category: {category()}")


def reset():
    state = load_state()
    if state.get("active"):
        print("Refusing to reset an active tracker. Stop it with Ctrl+C first.")
        return 1
    save_state({"active": False})
    print("Claude Chat tracker state reset.")


def main():
    ap = argparse.ArgumentParser(description="Track active Claude desktop/web usage on Windows.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="Start foreground activity tracking")
    p.add_argument("--idle-minutes", type=float, default=DEFAULT_IDLE_MINUTES)
    p.add_argument("--sample-seconds", type=float, default=SAMPLE_SECONDS)

    sub.add_parser("status", help="Show tracker state")
    sub.add_parser("reset", help="Reset stale state after tracker has stopped")

    args = ap.parse_args()
    if args.cmd == "run":
        return run(args.idle_minutes, args.sample_seconds)
    if args.cmd == "status":
        return status()
    if args.cmd == "reset":
        return reset()


if __name__ == "__main__":
    raise SystemExit(main() or 0)
