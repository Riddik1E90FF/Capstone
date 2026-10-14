from __future__ import annotations
import argparse
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
HUMAN_LOG = LOG_DIR / "human_sessions.jsonl"
HUMAN_STATE = RUNTIME_DIR / "human_state.json"
CATEGORY_FILE = RUNTIME_DIR / "current_category.json"

DEFAULT_CATEGORIES = [
    "Proposal",
    "Research",
    "Coding",
    "Data / EEG",
    "AI / Model Training",
    "Game Environment",
    "Experiments",
    "Testing / Evaluation",
    "Documentation",
    "Presentation",
    "Meeting",
    "Planning",
    "Other",
]

LOG_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

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
    return read_json(CATEGORY_FILE, {"category": "Uncategorized"}).get("category", "Uncategorized")

def set_category(name):
    write_json(CATEGORY_FILE, {"category": name, "updated_utc": now_iso()})
    print(f"Current project category: {name}")

def append_session(record):
    with HUMAN_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def start(cat=None):
    state = read_json(HUMAN_STATE, {})
    if state.get("active"):
        print("A human timer is already running.")
        print(f"  Category: {state.get('category')}")
        print(f"  Started:  {state.get('start_utc')}")
        return 1
    if cat:
        set_category(cat)
    state = {
        "active": True,
        "session_id": str(uuid.uuid4()),
        "category": category(),
        "start_utc": now_iso(),
        "start_mono": time.monotonic(),
    }
    write_json(HUMAN_STATE, state)
    print(f"Human timer started: {state['category']}")

def stop():
    state = read_json(HUMAN_STATE, {})
    if not state.get("active"):
        print("No human timer is running.")
        return 1
    end = time.monotonic()
    duration_ms = round((end - state["start_mono"]) * 1000)
    record = {
        "session_id": state["session_id"],
        "start_utc": state["start_utc"],
        "end_utc": now_iso(),
        "duration_ms": duration_ms,
        "category": state["category"],
    }
    append_session(record)
    write_json(HUMAN_STATE, {"active": False})
    print(f"Human timer stopped: {duration_ms/3600000:.3f} h ({duration_ms/60000:.1f} min)")
    print(f"Category: {record['category']}")

def status():
    state = read_json(HUMAN_STATE, {})
    print(f"Current category: {category()}")
    if state.get("active"):
        elapsed = (time.monotonic() - state["start_mono"]) / 60
        print(f"Human timer: RUNNING ({elapsed:.1f} min)")
    else:
        print("Human timer: stopped")

def list_categories():
    print("Suggested categories:")
    for x in DEFAULT_CATEGORIES:
        print(f"  - {x}")

def main():
    ap = argparse.ArgumentParser(description="Capstone human-time tracker")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("start", help="Start a human work session")
    p.add_argument("category", nargs="?", help="Category, e.g. Coding")

    p = sub.add_parser("category", help="Set the category used by human and Claude tracking")
    p.add_argument("name")

    sub.add_parser("stop", help="Stop the current human work session")
    sub.add_parser("status", help="Show timer/category status")
    sub.add_parser("categories", help="List suggested categories")

    args = ap.parse_args()
    if args.cmd == "start": return start(args.category)
    if args.cmd == "stop": return stop()
    if args.cmd == "status": return status()
    if args.cmd == "category": return set_category(args.name)
    if args.cmd == "categories": return list_categories()

if __name__ == "__main__":
    raise SystemExit(main() or 0)
