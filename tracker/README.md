# Capstone Time Tracker

Automatically records Claude Code activity for this repository.

## Measures
- Claude Code sessions and session wall-clock duration
- User-prompt-to-stop turn duration
- Tool execution duration reported by Claude Code
- Tool names/counts
- Subagent start/stop events
- Failed tool calls
- Model/effort metadata when available

## Does NOT measure
It does not claim to measure Claude's private hidden chain-of-thought. The defensible proxy is **agent turn wall-clock time**, from your submitted prompt until Claude finishes that turn.

## Privacy
The tracker does not store prompt text, tool arguments, tool output, file contents, or the Claude transcript. It stores prompt length and a short hash.

## Report
From the repository root:

```powershell
python tracker/report.py
```

Machine-readable:

```powershell
python tracker/report.py --json
```

Generated logs live in `tracker/logs/` and runtime state in `tracker/runtime/`. Both are ignored by Git.
