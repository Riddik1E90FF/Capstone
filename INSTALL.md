# Capstone Tracker Installation

If your repository does not already have `.claude/settings.json`, copy this package's `.claude/settings.json` and `tracker/` into the repository root.

If `.claude/settings.json` already exists, **do not overwrite it**. Merge the `hooks` entries instead.

Then start Claude Code in the repository and run:

```text
/hooks
```

Verify the tracker hooks appear as Project hooks.

After Claude Code does some work:

```powershell
python tracker/report.py
```

The tracker is logging-only and does not approve, deny, or modify Claude actions.
