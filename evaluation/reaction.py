"""Human reaction time per attack, from the per-step log `scripts/play.py` records (D037).

Definition (D037):

* **Onset** is the simulation tick at which the enemy enters TELEGRAPH.
* **Response** is the first *new* dodge or movement input after the decision
  step in which onset happened: a step whose action is DODGE or a MOVE_*
  action and differs from the previous step's action.  Holding a key through
  onset is not a response; pressing, or changing direction, is.  The onset
  step itself never counts, because its input was chosen before the frame
  showing the telegraph was drawn.
* **Reaction time** = the first tick the response was applied minus the
  onset tick, in simulation time (30 Hz ticks), not wall time.  Inputs are
  read once per decision step (2 ticks, 66.7 ms), which is the resolution.
* **Window**: from onset to the end of the attack's RECOVERY phase.  A
  response after the attack resolved (ACTIVE ended) still counts, and is
  flagged ``before_resolution=False``.
* **Anticipation**: a response under ANTICIPATION_TICKS (100 ms) is too fast
  to be a reaction to the telegraph.  It is recorded, not dropped, and
  excluded from the median.
* **No response**: no qualifying input in the window.  **Censored**: the
  episode ended inside the window with no response yet.  Both are counted
  separately, never dropped.

Block is not a response under this definition.  The raw log keeps every
input, so a definition including it can be computed later as a sensitivity
analysis.

This module only reads environment state.  Nothing here reaches the agent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from combat_env import Action
from combat_env.config import TICK_HZ
from combat_env.env import EnemyPhase

RESPONSE_ACTIONS = frozenset(
    [int(a) for a in range(Action.MOVE_N, Action.MOVE_NW + 1)] + [int(Action.DODGE)])
ANTICIPATION_TICKS: int = 3          # 100 ms at 30 Hz

# One row per env.step, in this order.  `tick`, the counter and the phase
# fields are read after the step; `action` is the input held for it.
STEP_FIELDS = ("tick", "action", "attacks_faced", "phase", "phase_start_tick", "wall")


def log_step(env: Any, action: int, wall: float) -> list:
    """The row to append after ``env.step(action)``."""
    return [int(env.tick), int(action), int(env.stats.attacks_faced),
            int(env.enemy_phase), int(env.enemy_phase_start_tick), round(float(wall), 4)]


@dataclass(frozen=True)
class AttackResponse:
    attack: int                      # 1-based, the attacks_faced value
    onset_tick: int
    status: str                      # response | anticipation | no_response | censored
    response_tick: int | None = None
    response_action: int | None = None
    rt_ticks: int | None = None
    before_resolution: bool | None = None

    @property
    def rt_seconds(self) -> float | None:
        return None if self.rt_ticks is None else self.rt_ticks / TICK_HZ


def attack_responses(steps: Sequence[Sequence]) -> list[AttackResponse]:
    """One AttackResponse per attack faced in an episode's step log.

    Relies on every phase lasting longer than one decision step (telegraph
    >= 3 ticks, recovery >= 9, the gap between attacks >= 16 at 30 Hz), so
    each phase change is seen in the log.  A log that breaks this raises
    rather than producing silently wrong windows.
    """
    out: list[AttackResponse] = []
    prev_tick, prev_action, prev_count = 0, None, 0
    open_: dict[str, Any] | None = None      # the attack whose window is open

    def close(status: str, **kw: Any) -> None:
        nonlocal open_
        out.append(AttackResponse(attack=open_["attack"], onset_tick=open_["onset"],
                                  status=status, **kw))
        open_ = None

    for row in steps:
        tick, action, count, phase, phase_start = (int(v) for v in row[:5])
        start = prev_tick + 1                # first tick this step's input was applied

        if open_ is not None:
            if count != open_["attack"]:
                raise ValueError(f"attack {count} began before attack {open_['attack']}'s "
                                 f"window was seen to close")
            if phase == EnemyPhase.RECOVERY and open_["active_end"] is None:
                open_["active_end"] = phase_start
            # The window closes when the enemy re-enters IDLE, at tick W.
            window_end = phase_start if phase == EnemyPhase.IDLE else None
            if (action in RESPONSE_ACTIONS and action != prev_action
                    and (window_end is None or start < window_end)):
                rt = start - open_["onset"]
                resolved = open_["active_end"]
                close("anticipation" if rt < ANTICIPATION_TICKS else "response",
                      response_tick=start, response_action=action, rt_ticks=rt,
                      before_resolution=resolved is None or start < resolved)
            elif window_end is not None:
                close("no_response")

        if count != prev_count:
            if count != prev_count + 1:
                raise ValueError(f"attacks_faced jumped {prev_count} -> {count} in one step")
            if phase != EnemyPhase.TELEGRAPH:
                raise ValueError(f"attack {count} began but the phase after the step is {phase}")
            open_ = {"attack": count, "onset": phase_start, "active_end": None}

        prev_tick, prev_action, prev_count = tick, action, count

    if open_ is not None:
        close("censored")
    return out


def summarize(responses: Iterable[AttackResponse]) -> dict[str, Any]:
    """Median reaction time over responses, and every outcome counted."""
    responses = list(responses)
    rts = [r.rt_ticks for r in responses if r.status == "response"]
    counts = {s: sum(r.status == s for r in responses)
              for s in ("response", "anticipation", "no_response", "censored")}
    return {
        "median_rt_s": float(np.median(rts)) / TICK_HZ if rts else None,
        "n_attacks": len(responses),
        **{f"n_{s}": n for s, n in counts.items()},
        "n_after_resolution": sum(r.status == "response" and r.before_resolution is False
                                  for r in responses),
    }


def wall_clock() -> float:
    """Monotonic seconds, for the `wall` column (dropped-frame checks)."""
    return time.perf_counter()
