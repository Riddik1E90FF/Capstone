"""Shared pytest configuration.

Hypothesis profiles for the property-based tests (tests/test_env_properties.py).

- "default": fast enough to run on every `pytest` invocation.
- "thorough": many more examples, for a pre-merge or overnight run:
      HYPOTHESIS_PROFILE=thorough python -m pytest tests/test_env_properties.py

Failing examples are saved under `.pytest_cache/hypothesis` (already
gitignored) instead of Hypothesis's default `.hypothesis/` at the repo root, so
a shrunk counterexample is replayed first on the next run without leaving an
untracked directory behind.
"""

import os
from pathlib import Path

from hypothesis import HealthCheck, settings
from hypothesis.database import DirectoryBasedExampleDatabase

_DB = DirectoryBasedExampleDatabase(
    Path(__file__).resolve().parent.parent / ".pytest_cache" / "hypothesis"
)

# deadline=None: an episode runs up to 300 steps, and its wall time depends on
# how long the drawn policy survives, not on a defect.  A per-example deadline
# would only add flaky failures.
settings.register_profile(
    "default",
    max_examples=25,
    deadline=None,
    database=_DB,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "thorough",
    parent=settings.get_profile("default"),
    max_examples=400,
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
