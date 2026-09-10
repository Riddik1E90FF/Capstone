from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "tracker" / "logs" / "claude_events.jsonl"

def load():
    if not LOG.exists():
        return []
    rows = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows = load()
    sessions = [r for r in rows if r.get("event") == "SessionEnd"]
    turns = [r for r in rows if r.get("event") in ("Stop", "StopFailure")]
    tools = [r for r in rows if r.get("event") in ("PostToolUse", "PostToolUseFailure")]

    session_ms = sum(r.get("session_duration_ms") or 0 for r in sessions)
    turn_ms = sum(r.get("turn_duration_ms") or 0 for r in turns)
    tool_ms = sum(r.get("tool_duration_ms") or 0 for r in tools)

    counts = {}
    for r in tools:
        name = r.get("tool_name") or "unknown"
        counts[name] = counts.get(name, 0) + 1

    result = {
        "completed_sessions": len(sessions),
        "completed_turns": len(turns),
        "completed_tool_calls": len(tools),
        "session_wall_clock_hours": round(session_ms / 3600000, 3),
        "agent_turn_wall_clock_hours": round(turn_ms / 3600000, 3),
        "tool_execution_hours": round(tool_ms / 3600000, 3),
        "tool_calls_by_name": dict(sorted(counts.items())),
        "note": "Agent turn wall-clock is the closest available proxy for model work time. It includes model generation/thinking, tool execution, and other latency. Hidden private reasoning time is not separately exposed."
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("\n=== CAPSTONE CLAUDE CODE TIME REPORT ===\n")
    print(f"Completed sessions:    {result['completed_sessions']}")
    print(f"Completed turns:       {result['completed_turns']}")
    print(f"Completed tool calls:  {result['completed_tool_calls']}")
    print(f"Session wall-clock:    {result['session_wall_clock_hours']:.3f} h")
    print(f"Agent turn wall-clock: {result['agent_turn_wall_clock_hours']:.3f} h")
    print(f"Tool execution time:   {result['tool_execution_hours']:.3f} h")
    print("\nTool calls:")
    for name, count in result["tool_calls_by_name"].items():
        print(f"  {name:24} {count}")
    print("\n" + result["note"])

if __name__ == "__main__":
    main()
