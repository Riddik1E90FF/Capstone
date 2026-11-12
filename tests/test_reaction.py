"""Reaction time per attack (evaluation/reaction.py, D037)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from combat_env import Action, make_env
from combat_env.env import EnemyPhase
from evaluation import reaction as rx

IDLE, TEL, ACT, REC = (int(EnemyPhase.IDLE), int(EnemyPhase.TELEGRAPH),
                       int(EnemyPhase.ACTIVE), int(EnemyPhase.RECOVERY))
I, D, N, E, B = (int(Action.IDLE), int(Action.DODGE), int(Action.MOVE_N),
                 int(Action.MOVE_E), int(Action.BLOCK))


def timeline(actions, onset=11, tel=6, act=4, rec=10, repeat=2):
    """Step log for one attack: onset at tick `onset`, then telegraph, active,
    recovery, idle.  Rows read the state after each step, as log_step does."""
    def state(t):
        if t < onset:
            return 0, IDLE, 0
        edges = [(onset, TEL), (onset + tel, ACT), (onset + tel + act, REC),
                 (onset + tel + act + rec, IDLE)]
        start, phase = max((e for e in edges if e[0] <= t), key=lambda e: e[0])
        return 1, phase, start
    rows = []
    for i, a in enumerate(actions):
        t = (i + 1) * repeat
        count, phase, start = state(t)
        rows.append([t, a, count, phase, start, 0.0])
    return rows


# Default timeline: onset at tick 11 falls in step 6 (ticks 11-12).  Step k
# covers ticks 2k-1 and 2k.  Active starts at 17, recovery at 21, idle at 31.

def test_no_input_is_a_no_response():
    (r,) = rx.attack_responses(timeline([I] * 20))
    assert r.status == "no_response" and r.onset_tick == 11 and r.rt_ticks is None


def test_new_dodge_gives_tick_rt():
    acts = [I] * 8 + [D] + [I] * 11        # step 9 starts at tick 17
    (r,) = rx.attack_responses(timeline(acts))
    assert r.status == "response"
    assert r.response_tick == 17 and r.rt_ticks == 6
    assert r.rt_seconds == pytest.approx(0.2)
    assert r.before_resolution is True


def test_held_movement_is_not_a_response_but_a_direction_change_is():
    acts = [N] * 10 + [E] * 10               # step 11 starts at tick 21
    (r,) = rx.attack_responses(timeline(acts))
    assert r.status == "response" and r.response_tick == 21 and r.response_action == E


def test_holding_one_key_throughout_is_a_no_response():
    (r,) = rx.attack_responses(timeline([N] * 20))
    assert r.status == "no_response"


def test_input_in_the_onset_step_does_not_count():
    acts = [I] * 5 + [D] * 15                # pressed in step 6, held after
    (r,) = rx.attack_responses(timeline(acts))
    assert r.status == "no_response"


def test_re_press_after_release_counts():
    acts = [I] * 5 + [D, I, D] + [I] * 12    # step 8 starts at tick 15
    (r,) = rx.attack_responses(timeline(acts))
    assert r.status == "response" and r.response_tick == 15


def test_too_fast_is_an_anticipation():
    acts = [I] * 6 + [D] + [I] * 13          # step 7 starts at 13: rt 2 ticks
    (r,) = rx.attack_responses(timeline(acts))
    assert r.status == "anticipation" and r.rt_ticks == 2
    assert rx.ANTICIPATION_TICKS == 3


def test_exactly_100ms_is_a_response():
    # onset 12 falls in step 6 (ticks 11-12); step 8 starts at 15: rt 3 ticks.
    (r,) = rx.attack_responses(timeline([I] * 7 + [D] + [I] * 12, onset=12))
    assert r.rt_ticks == 3 and r.status == "response"


def test_late_response_is_flagged_after_resolution():
    acts = [I] * 12 + [D] + [I] * 7          # step 13 starts at 25; active ended 21
    (r,) = rx.attack_responses(timeline(acts))
    assert r.status == "response" and r.before_resolution is False


def test_response_around_the_resolving_tick():
    # Active ends (recovery starts) at tick 21.
    (r,) = rx.attack_responses(timeline([I] * 9 + [D] + [I] * 10))    # starts 19
    assert r.response_tick == 19 and r.before_resolution is True
    (r,) = rx.attack_responses(timeline([I] * 10 + [D] + [I] * 9))    # starts 21
    assert r.response_tick == 21 and r.before_resolution is False


def test_window_closes_when_idle_resumes():
    # Idle re-entered at tick 31, which is where step 16 starts.
    (r,) = rx.attack_responses(timeline([I] * 15 + [D] + [I] * 4))
    assert r.status == "no_response"
    (r,) = rx.attack_responses(timeline([I] * 14 + [D] + [I] * 5))    # starts 29
    assert r.status == "response" and r.response_tick == 29


def test_block_is_not_a_response():
    (r,) = rx.attack_responses(timeline([I] * 8 + [B] + [I] * 11))
    assert r.status == "no_response"


def test_episode_ending_inside_the_window_is_censored():
    (r,) = rx.attack_responses(timeline([I] * 9))
    assert r.status == "censored"


def test_jumped_counter_raises():
    rows = timeline([I] * 5)
    rows[-1][2] = 2
    with pytest.raises(ValueError):
        rx.attack_responses(rows)


def test_new_attack_before_the_window_closed_raises():
    rows = timeline([I] * 12)
    rows[-1][2:5] = [2, TEL, rows[-1][0]]
    with pytest.raises(ValueError):
        rx.attack_responses(rows)


def test_summarize_counts_everything_and_medians_responses():
    rs = [rx.AttackResponse(1, 0, "response", 3, D, 6, True),
          rx.AttackResponse(2, 0, "response", 3, D, 12, False),
          rx.AttackResponse(3, 0, "response", 3, D, 9, True),
          rx.AttackResponse(4, 0, "anticipation", 1, D, 1, True),
          rx.AttackResponse(5, 0, "no_response"),
          rx.AttackResponse(6, 0, "censored")]
    s = rx.summarize(rs)
    assert s["median_rt_s"] == pytest.approx(9 / 30)
    assert (s["n_attacks"], s["n_response"], s["n_anticipation"],
            s["n_no_response"], s["n_censored"], s["n_after_resolution"]) == (6, 3, 1, 1, 1, 1)


def test_summarize_with_no_responses():
    assert rx.summarize([rx.AttackResponse(1, 0, "no_response")])["median_rt_s"] is None


# -- the real environment --------------------------------------------------

@pytest.mark.parametrize("cfg, delay", [(0, 4), (12, 3), (24, 2)])
def test_real_env_scripted_player(cfg, delay):
    """A player who dodges `delay` steps after the telegraph first appears on
    screen.  Every attack is accounted for, and each RT is exactly the tick
    that dodge began minus the onset tick the env reported."""
    env = make_env(cfg, seed=3)
    env.reset(seed=3)
    steps, fired = [], {}
    seen_at, done = None, False
    while not done:
        if env.enemy_phase == EnemyPhase.TELEGRAPH and seen_at is None:
            seen_at = (len(steps), env.stats.attacks_faced, env.enemy_phase_start_tick)
        if env.enemy_phase == EnemyPhase.IDLE:
            seen_at = None
        a = I
        if seen_at is not None and len(steps) == seen_at[0] + delay - 1:
            a = D
            fired[seen_at[1]] = (env.tick + 1, seen_at[2])
        _, _, term, trunc, _ = env.step(a)
        steps.append(rx.log_step(env, a, 0.0))
        done = term or trunc
    rs = rx.attack_responses(steps)
    assert len(rs) == env.stats.attacks_faced > 0
    assert [r.attack for r in rs] == list(range(1, len(rs) + 1))
    checked = 0
    for r in rs:
        if r.attack in fired and r.status != "censored":
            start, onset = fired[r.attack]
            assert r.onset_tick == onset
            assert r.response_tick == start and r.rt_ticks == start - onset
            checked += 1
    assert checked >= len(fired) - 1 > 0


def test_play_script_still_imports():
    spec = importlib.util.spec_from_file_location(
        "play", Path(__file__).resolve().parents[1] / "scripts" / "play.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.attack_responses is rx.attack_responses
