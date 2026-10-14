from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "tracker" / "logs"
CLAUDE_LOG = LOG_DIR / "claude_events.jsonl"
HUMAN_LOG = LOG_DIR / "human_sessions.jsonl"
CHAT_LOG = LOG_DIR / "claude_chat_sessions.jsonl"


def load(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def git_stats():
    try:
        commits = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-list", "--count", "HEAD"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
        latest = subprocess.check_output(
            ["git", "-C", str(ROOT), "log", "-1", "--format=%h|%ad|%s", "--date=iso"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
        return {"commits": int(commits), "latest_commit": latest}
    except Exception:
        return {"commits": None, "latest_commit": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    claude = load(CLAUDE_LOG)
    human = load(HUMAN_LOG)
    chat = load(CHAT_LOG)

    sessions = [r for r in claude if r.get("event") == "SessionEnd"]
    turns = [r for r in claude if r.get("event") in ("Stop", "StopFailure")]
    tools = [r for r in claude if r.get("event") in ("PostToolUse", "PostToolUseFailure")]

    session_ms = sum(r.get("session_duration_ms") or 0 for r in sessions)
    turn_ms = sum(r.get("turn_duration_ms") or 0 for r in turns)
    tool_ms = sum(r.get("tool_duration_ms") or 0 for r in tools)
    human_ms = sum(r.get("duration_ms") or 0 for r in human)
    chat_ms = sum(r.get("duration_ms") or 0 for r in chat)

    tool_counts = defaultdict(int)
    for r in tools:
        tool_counts[r.get("tool_name") or "unknown"] += 1

    claude_by_cat = defaultdict(int)
    for r in turns:
        claude_by_cat[r.get("category") or "Uncategorized"] += r.get("turn_duration_ms") or 0

    chat_by_cat = defaultdict(int)
    for r in chat:
        chat_by_cat[r.get("category") or "Uncategorized"] += r.get("duration_ms") or 0

    human_by_cat = defaultdict(int)
    for r in human:
        human_by_cat[r.get("category") or "Uncategorized"] += r.get("duration_ms") or 0

    result = {
        "human_hours": round(human_ms / 3600000, 3),
        "claude_code_turn_hours": round(turn_ms / 3600000, 3),
        "claude_code_session_hours": round(session_ms / 3600000, 3),
        "claude_code_tool_hours": round(tool_ms / 3600000, 3),
        "claude_chat_active_hours": round(chat_ms / 3600000, 3),
        "total_claude_hours": round((turn_ms + chat_ms) / 3600000, 3),
        "combined_tracked_hours": round((human_ms + turn_ms + chat_ms) / 3600000, 3),
        "completed_sessions": len(sessions),
        "completed_turns": len(turns),
        "completed_tool_calls": len(tools),
        "claude_chat_segments": len(chat),
        "tool_calls_by_name": dict(sorted(tool_counts.items())),
        "human_hours_by_category": {
            k: round(v / 3600000, 3) for k, v in sorted(human_by_cat.items())
        },
        "claude_code_turn_hours_by_category": {
            k: round(v / 3600000, 3) for k, v in sorted(claude_by_cat.items())
        },
        "claude_chat_hours_by_category": {
            k: round(v / 3600000, 3) for k, v in sorted(chat_by_cat.items())
        },
        "git": git_stats(),
        "note": (
            "Claude Code turn wall-clock is the closest available proxy for model work time. "
            "It includes model generation/thinking, tool execution, and latency. Hidden private "
            "reasoning time is not separately exposed. Claude Chat time is active foreground "
            "Windows usage detected while a Claude desktop window or browser tab titled with "
            "'Claude' is foreground and the user is below the configured idle threshold. "
            "It does not measure private model reasoning separately. Human time is based on "
            "explicit start/stop sessions."
        ),
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("\n=== CAPSTONE FULL TIME REPORT ===\n")
    print(f"Human tracked time:        {result['human_hours']:.3f} h")
    print(f"Claude Code turn time:     {result['claude_code_turn_hours']:.3f} h")
    print(f"Claude Chat active time:   {result['claude_chat_active_hours']:.3f} h")
    print(f"Total Claude time:         {result['total_claude_hours']:.3f} h")
    print(f"Combined tracked:          {result['combined_tracked_hours']:.3f} h")
    print(f"Claude Code session time:  {result['claude_code_session_hours']:.3f} h")
    print(f"Claude Code tool time:     {result['claude_code_tool_hours']:.3f} h")

    print(f"\nClaude Code sessions:      {result['completed_sessions']}")
    print(f"Claude Code turns:         {result['completed_turns']}")
    print(f"Claude Code tool calls:    {result['completed_tool_calls']}")
    print(f"Claude Chat segments:      {result['claude_chat_segments']}")

    print("\nHuman time by category:")
    for k, v in result["human_hours_by_category"].items():
        print(f"  {k:24} {v:.3f} h")

    print("\nClaude Code time by category:")
    for k, v in result["claude_code_turn_hours_by_category"].items():
        print(f"  {k:24} {v:.3f} h")

    print("\nClaude Chat time by category:")
    for k, v in result["claude_chat_hours_by_category"].items():
        print(f"  {k:24} {v:.3f} h")

    print("\nTool calls:")
    for name, count in result["tool_calls_by_name"].items():
        print(f"  {name:24} {count}")

    print("\nGit:")
    print(f"  Commits:              {result['git']['commits']}")
    if result["git"]["latest_commit"]:
        print(f"  Latest:               {result['git']['latest_commit']}")

    print("\n" + result["note"])


if __name__ == "__main__":
    main()
