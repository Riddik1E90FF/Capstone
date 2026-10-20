"""Build step 7a: gradient_update and the training loop.  Written from the
agreed interface (dqn/train.py docstrings) before the implementation exists.

Two parts:

1. gradient_update, tested in isolation.  The oracle is the same step-1..6
   calls made by hand on independent copies of the networks and optimiser:
   the result must match bit for bit.

2. train, run end to end against StubEnv, a tiny fake environment with
   known, seeded episode lengths that raises if stepped after an episode
   ends.  Every env call, every buffer add, every gradient update and every
   target sync is recorded in one event list, so the tests can check counts,
   ordering and what was stored.  These tests see composition, not learning:
   whether the loop learns the right Q-values is L2 (the chain MDP, D029).

The recording works by replacing the names `ReplayBuffer`, `gradient_update`
and `hard_sync` inside dqn.train with recording wrappers that call the real
ones.  So train() must use those three through the names dqn/train.py
already imports, which is what the stub file does.

Skipped as a whole until dqn/train.py exists.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pytest

if not (Path(__file__).resolve().parents[1] / "dqn" / "train.py").exists():
    pytest.skip("dqn/train.py not written yet (build step 7a)", allow_module_level=True)

import dqn.train as train_mod  # noqa: E402
from dqn.adam import Adam  # noqa: E402
from dqn.config import DQNConfig  # noqa: E402
from dqn.exploration import linear_epsilon  # noqa: E402
from dqn.network import MLP  # noqa: E402
from dqn.replay import Batch, ReplayBuffer  # noqa: E402
from dqn.runlog import RunLogger, read_jsonl  # noqa: E402
from dqn.seeding import stream_seed  # noqa: E402
from dqn.target import make_target, sync_due  # noqa: E402
from dqn.td import td_loss_and_grad, td_target  # noqa: E402
from dqn.train import TrainResult, gradient_update, train  # noqa: E402

CFG = DQNConfig()
HYPER = dict(lr=CFG.learning_rate, beta1=CFG.adam_beta1, beta2=CFG.adam_beta2, eps=CFG.adam_eps)


# ======================================================================
# Part 1: gradient_update
# ======================================================================

OBS_DIM, N_ACT, B = 3, 2, 16


def _net(seed: int, sizes=(OBS_DIM, 8, N_ACT)) -> MLP:
    return MLP(sizes, np.random.default_rng(seed))


def _batch(seed: int = 0, n: int = B, terminated=None) -> Batch:
    rng = np.random.default_rng(seed)
    if terminated is None:
        terminated = rng.random(n) < 0.3
    return Batch(obs=rng.normal(size=(n, OBS_DIM)),
                 actions=rng.integers(0, N_ACT, size=n).astype(np.int64),
                 rewards=rng.normal(size=n),
                 next_obs=rng.normal(size=(n, OBS_DIM)),
                 terminated=np.asarray(terminated, dtype=bool))


def _setup(online_seed=0, target_seed=1):
    """online and target deliberately DIFFERENT, so using the wrong network
    for the bootstrap changes the answer."""
    online, target = _net(online_seed), _net(target_seed)
    opt = Adam(online.params, **HYPER)
    return online, target, opt


def _by_hand(online: MLP, target: MLP, opt: Adam, batch: Batch, gamma: float):
    """The oracle: the same components, called in the only order the data
    dependencies allow."""
    q, cache = online.forward(batch.obs)
    next_q, _ = target.forward(batch.next_obs)
    y = td_target(batch.rewards, next_q, batch.terminated, gamma)
    loss, grad_q = td_loss_and_grad(q, batch.actions, y)
    q_mean = float(np.mean(q[np.arange(len(batch.actions)), batch.actions]))
    opt.step(online.backward(cache, grad_q))
    return loss, q_mean


def _snapshot(net: MLP) -> dict[str, np.ndarray]:
    return {k: a.copy() for k, a in net.params.items()}


def _clone(online: MLP, target: MLP, opt: Adam):
    o2, t2 = copy.deepcopy(online), copy.deepcopy(target)
    opt2 = Adam(o2.params, **HYPER)
    return o2, t2, opt2


@pytest.mark.parametrize("gamma", [0.99, 0.5, 0.0])
def test_gradient_update_matches_hand_composition(gamma):
    online, target, opt = _setup()
    ref_online, ref_target, ref_opt = _clone(online, target, opt)
    batch = _batch()

    loss, q_mean = gradient_update(online, target, opt, batch, gamma)
    ref_loss, ref_q_mean = _by_hand(ref_online, ref_target, ref_opt, batch, gamma)

    assert loss == ref_loss
    assert q_mean == ref_q_mean
    for k in ref_online.params:
        np.testing.assert_array_equal(online.params[k], ref_online.params[k], err_msg=k)


def test_gradient_update_three_steps_in_a_row_match():
    # Adam's state (t, m, v) carries across calls; so must the online params.
    online, target, opt = _setup()
    ref_online, ref_target, ref_opt = _clone(online, target, opt)
    for s in range(3):
        batch = _batch(seed=s)
        assert gradient_update(online, target, opt, batch, 0.99) == \
            _by_hand(ref_online, ref_target, ref_opt, batch, 0.99)
    for k in ref_online.params:
        np.testing.assert_array_equal(online.params[k], ref_online.params[k], err_msg=k)
    assert opt.t == 3


def test_target_network_is_not_modified():
    online, target, opt = _setup()
    before = _snapshot(target)
    gradient_update(online, target, opt, _batch(), 0.99)
    for k, a in before.items():
        np.testing.assert_array_equal(target.params[k], a, err_msg=k)


def test_online_params_change_in_place():
    # Adam was built on online.params; replacing the arrays (instead of
    # writing into them) would leave Adam updating stale objects.
    online, target, opt = _setup()
    ids = {k: id(a) for k, a in online.params.items()}
    before = _snapshot(online)
    gradient_update(online, target, opt, _batch(), 0.99)
    assert {k: id(a) for k, a in online.params.items()} == ids
    assert any(not np.array_equal(online.params[k], before[k]) for k in before)


def test_exactly_one_optimiser_step():
    online, target, opt = _setup()
    gradient_update(online, target, opt, _batch(), 0.99)
    assert opt.t == 1


def test_bootstrap_uses_target_not_online():
    # Same batch, two different targets: the update must differ.  If the
    # bootstrap read the online net, the target would not matter at all.
    batch = _batch(terminated=np.zeros(B, dtype=bool))
    o1, t1, opt1 = _setup(online_seed=0, target_seed=1)
    o2, t2, opt2 = _setup(online_seed=0, target_seed=2)
    loss1, _ = gradient_update(o1, t1, opt1, batch, 0.99)
    loss2, _ = gradient_update(o2, t2, opt2, batch, 0.99)
    assert loss1 != loss2


def test_terminated_masks_bootstrap():
    # All terminated: the targets are just the rewards, so the target net is
    # irrelevant and two different targets give the same loss.
    batch = _batch(terminated=np.ones(B, dtype=bool))
    o1, t1, opt1 = _setup(online_seed=0, target_seed=1)
    o2, t2, opt2 = _setup(online_seed=0, target_seed=2)
    assert gradient_update(o1, t1, opt1, batch, 0.99) == gradient_update(o2, t2, opt2, batch, 0.99)


def test_q_mean_is_taken_before_the_parameter_change():
    online, target, opt = _setup()
    batch = _batch()
    q, _ = online.forward(batch.obs)
    expected = float(np.mean(q[np.arange(B), batch.actions]))
    _, q_mean = gradient_update(online, target, opt, batch, 0.99)
    assert q_mean == expected


def test_returns_python_floats():
    online, target, opt = _setup()
    loss, q_mean = gradient_update(online, target, opt, _batch(), 0.99)
    assert type(loss) is float and type(q_mean) is float


def test_gradient_update_uses_no_global_rng():
    np.random.seed(5)
    random.seed(5)
    np_before = np.random.get_state()[1].copy()
    py_before = random.getstate()
    online, target, opt = _setup()
    gradient_update(online, target, opt, _batch(), 0.99)
    np.testing.assert_array_equal(np.random.get_state()[1], np_before)
    assert random.getstate() == py_before


# ======================================================================
# Part 2: train, against a recording stub environment
# ======================================================================

S_OBS, S_ACT = 3, 2


class StubEnv:
    """Fake env with seeded episode lengths and outcomes.

    Each reset draws an episode length in [3, 14] and whether it ends by
    termination or truncation, from the env's own RNG.  Every observation
    is unique (a running counter), so the tests can tell exactly which one
    was stored where.  Reset observations have obs[2] == 1.0, step
    observations obs[2] == 0.0.  Stepping a finished episode raises, as
    CombatEnv does.
    """

    def __init__(self, seed, events: list):
        self.events = events
        self.rng = np.random.default_rng(seed)
        self.counter = 0
        self.done = True
        self.k = self.length = 0
        self.ends_by = "terminated"

    def _obs(self, marker: float) -> np.ndarray:
        self.counter += 1
        return np.array([self.counter * 1e-3, self.k * 0.1, marker])

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.length = int(self.rng.integers(3, 15))
        self.ends_by = "terminated" if self.rng.random() < 0.5 else "truncated"
        self.k = 0
        self.done = False
        obs = self._obs(1.0)
        self.events.append(("reset", seed, obs.copy()))
        return obs, {}

    def step(self, action):
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset()")
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) \
                or not 0 <= action < S_ACT:
            raise ValueError(f"bad action {action!r}")
        self.k += 1
        reward = 1.0 if action == 0 else 0.25
        end = self.k == self.length
        terminated = end and self.ends_by == "terminated"
        truncated = end and self.ends_by == "truncated"
        self.done = end
        obs = self._obs(0.0)
        self.events.append(("step", int(action), reward, obs.copy(), terminated, truncated))
        return obs, reward, terminated, truncated, {}


def _cfg(**overrides) -> DQNConfig:
    base = dict(seed=3, total_timesteps=600, learning_starts=64, batch_size=32,
                buffer_size=200, train_frequency=4, target_network_frequency=50,
                update_log_frequency=20, hidden_sizes=(8,), exploration_fraction=0.5)
    base.update(overrides)
    return DQNConfig().with_overrides(base)


class Run:
    """One recorded train() call."""

    def __init__(self, cfg: DQNConfig, run_dir: Path, monkeypatch):
        self.cfg = cfg
        self.events: list = []
        self.env_fn_seeds: list = []
        events = self.events

        class RecordingBuffer(ReplayBuffer):
            def add(self, obs, action, reward, next_obs, terminated):
                super().add(obs, action, reward, next_obs, terminated)
                events.append(("add", np.array(obs, dtype=float), action, reward,
                               np.array(next_obs, dtype=float), terminated))

        real_update, real_sync = train_mod.gradient_update, train_mod.hard_sync

        def recording_update(*args, **kwargs):
            out = real_update(*args, **kwargs)
            events.append(("update", out))
            return out

        def recording_sync(*args, **kwargs):
            out = real_sync(*args, **kwargs)
            events.append(("sync",))
            return out

        monkeypatch.setattr(train_mod, "ReplayBuffer", RecordingBuffer)
        monkeypatch.setattr(train_mod, "gradient_update", recording_update)
        monkeypatch.setattr(train_mod, "hard_sync", recording_sync)

        def env_fn(seed):
            self.env_fn_seeds.append(seed)
            return StubEnv(seed, events)

        with RunLogger(run_dir, cfg) as logger:
            self.result = train(cfg, env_fn=env_fn, obs_dim=S_OBS, n_actions=S_ACT, logger=logger)
        self.run_dir = run_dir
        self.episodes = read_jsonl(run_dir / "episodes.jsonl") if (run_dir / "episodes.jsonl").exists() else []
        self.updates = read_jsonl(run_dir / "updates.jsonl") if (run_dir / "updates.jsonl").exists() else []

    def steps(self):
        return [e for e in self.events if e[0] == "step"]

    def by_step(self):
        """Events grouped by the 0-based env step they happened after.
        Group -1 holds what happened before the first step."""
        groups: dict[int, list] = {-1: []}
        t = -1
        for e in self.events:
            if e[0] == "step":
                t += 1
                groups[t] = []
            groups[t].append(e)
        return groups


@pytest.fixture
def run(tmp_path, monkeypatch):
    return Run(_cfg(), tmp_path / "run", monkeypatch)


def _expected_update_steps(cfg):
    return [t for t in range(cfg.total_timesteps)
            if t > cfg.learning_starts and t % cfg.train_frequency == 0]


def _expected_sync_steps(cfg):
    return [t for t in range(cfg.total_timesteps)
            if sync_due(t, learning_starts=cfg.learning_starts,
                        target_network_frequency=cfg.target_network_frequency)]


# -- environment interaction -------------------------------------------

def test_exactly_total_timesteps_env_steps(run):
    assert len(run.steps()) == run.cfg.total_timesteps


def test_env_created_once_with_env_stream_seed(run):
    assert run.env_fn_seeds == [stream_seed(run.cfg.seed, "env")]


def test_first_reset_seeded_later_resets_not(run):
    resets = [e for e in run.events if e[0] == "reset"]
    assert run.events[0][0] == "reset", "the env must be reset before the first step"
    assert resets[0][1] == stream_seed(run.cfg.seed, "env")
    assert len(resets) > 1, "test needs more than one episode"
    assert all(r[1] is None for r in resets[1:]), (
        "later resets must not pass a seed: the same seed every reset replays one episode")


def test_reset_exactly_once_after_each_episode_end(run):
    # StubEnv raises on a step after the end, so a missing reset fails the
    # run itself.  This catches the other direction: resets mid-episode or twice.
    ev = [e for e in run.events if e[0] in ("reset", "step")]
    for i in range(1, len(ev)):
        if ev[i][0] == "reset":
            prev = ev[i - 1]
            assert prev[0] == "step" and (prev[4] or prev[5]), (
                f"reset at event {i} did not follow an episode end")
    # Resetting after the very last step is allowed but not required.
    ends = sum(1 for e in run.steps() if e[4] or e[5])
    n_resets = sum(e[0] == "reset" for e in ev)
    assert n_resets in (ends, ends + 1)


# -- what gets stored ----------------------------------------------------

def test_one_add_per_env_step(run):
    for t, group in run.by_step().items():
        if t >= 0:
            assert sum(e[0] == "add" for e in group) == 1, f"step {t}"


def test_stored_transitions_are_the_real_ones(run):
    # obs: what the agent acted from (the reset obs at an episode start).
    # next_obs: what step() returned, never a reset observation.
    # terminated: the env's terminated flag alone, never truncated.
    current = None
    for t, group in run.by_step().items():
        if t < 0:
            current = next(e[2] for e in group if e[0] == "reset")
            continue
        step = group[0]
        add = next(e for e in group if e[0] == "add")
        _, action, reward, next_obs, terminated, truncated = step
        _, s_obs, s_action, s_reward, s_next, s_term = add
        np.testing.assert_array_equal(s_obs, current, err_msg=f"step {t}: stored obs")
        assert s_action == action, f"step {t}: stored action"
        assert s_reward == reward, f"step {t}: stored reward"
        np.testing.assert_array_equal(s_next, next_obs, err_msg=f"step {t}: stored next_obs")
        assert s_next[2] == 0.0, f"step {t}: a reset observation was stored as next_obs"
        assert bool(s_term) is terminated, (
            f"step {t}: stored terminated={s_term}, env said terminated={terminated}, "
            f"truncated={truncated}")
        current = next_obs
        # A reset in this group belongs to this step's episode end: the agent
        # acts from the reset observation next.
        for e in group:
            if e[0] == "reset":
                current = e[2]


def test_truncations_are_not_stored_as_terminal(run):
    truncated_steps = [t for t, g in run.by_step().items() if t >= 0 and g[0][5]]
    assert truncated_steps, "test needs at least one truncation"
    for t in truncated_steps:
        add = next(e for e in run.by_step()[t] if e[0] == "add")
        assert not add[5], f"step {t}: truncation stored as terminated"


# -- learning schedule ---------------------------------------------------

def test_updates_happen_exactly_on_gated_steps(run):
    got = [t for t, g in run.by_step().items() if any(e[0] == "update" for e in g)]
    assert got == _expected_update_steps(run.cfg)
    for t, g in run.by_step().items():
        assert sum(e[0] == "update" for e in g) <= 1, f"two updates after step {t}"


def test_syncs_happen_exactly_on_schedule(run):
    got = [t for t, g in run.by_step().items() if any(e[0] == "sync" for e in g)]
    assert got == _expected_sync_steps(run.cfg)


def test_no_update_or_sync_before_learning_starts(run):
    for t, g in run.by_step().items():
        if t <= run.cfg.learning_starts:
            assert not any(e[0] in ("update", "sync") for e in g), f"step {t}"


def test_update_comes_after_storing_and_before_sync(run):
    for t, g in run.by_step().items():
        kinds = [e[0] for e in g]
        if "update" in kinds:
            assert kinds.index("add") < kinds.index("update"), f"step {t}: update before add"
        if "update" in kinds and "sync" in kinds:
            assert kinds.index("update") < kinds.index("sync"), f"step {t}: sync before update"


# -- logs ----------------------------------------------------------------

def _completed_episodes(run):
    """Expected episode records, rebuilt from the env's own event log."""
    out, ret, length = [], 0.0, 0
    for t, e in enumerate(run.steps()):
        ret += e[2]
        length += 1
        if e[4] or e[5]:
            out.append(dict(global_step=t + 1, episode=len(out), length=length, ret=ret,
                            terminated=e[4], truncated=e[5],
                            epsilon=linear_epsilon(t, start_e=run.cfg.start_e,
                                                   end_e=run.cfg.end_e,
                                                   duration=run.cfg.exploration_steps)))
            ret, length = 0.0, 0
    return out


def test_one_episode_record_per_completed_episode(run):
    expected = _completed_episodes(run)
    assert len(run.episodes) == len(expected)
    assert run.result.episodes == len(expected)


def test_episode_records_match_the_env(run):
    for got, exp in zip(run.episodes, _completed_episodes(run)):
        i = exp["episode"]
        assert got["episode"] == i
        assert got["global_step"] == exp["global_step"], (
            f"episode {i}: global_step is env steps completed (0-based index + 1)")
        assert got["length"] == exp["length"], f"episode {i}: length"
        assert got["return"] == pytest.approx(exp["ret"], abs=1e-12), f"episode {i}: return"
        assert got["terminated"] is exp["terminated"], f"episode {i}: terminated"
        assert got["truncated"] is exp["truncated"], f"episode {i}: truncated"
        assert got["epsilon"] == exp["epsilon"], f"episode {i}: epsilon of the final action"


def test_unfinished_last_episode_not_logged(tmp_path, monkeypatch):
    # Pick a total_timesteps that ends mid-episode, then check it is not logged.
    probe = Run(_cfg(), tmp_path / "probe", monkeypatch)
    last_end = max(t for t, e in enumerate(probe.steps()) if e[4] or e[5])
    total = last_end + 3          # episodes are >= 3 steps, so this one is unfinished
    r = Run(_cfg(total_timesteps=total), tmp_path / "r", monkeypatch)
    assert r.episodes[-1]["global_step"] == last_end + 1
    assert r.result.episodes == len(r.episodes)


def test_update_records_on_logged_steps_only(run):
    expected = [t for t in _expected_update_steps(run.cfg) if t % run.cfg.update_log_frequency == 0]
    assert expected, "test config must produce logged updates"
    assert [u["global_step"] for u in run.updates] == [t + 1 for t in expected]


def test_update_records_hold_that_updates_values(run):
    returned = {t: e[1] for t, g in run.by_step().items() for e in g if e[0] == "update"}
    for u in run.updates:
        loss, q_mean = returned[u["global_step"] - 1]
        assert u["loss"] == loss
        assert u["q_mean"] == q_mean


# -- result and reproducibility -----------------------------------------

def test_train_result(run):
    r = run.result
    assert isinstance(r, TrainResult)
    assert r.global_steps == run.cfg.total_timesteps
    assert r.updates == len(_expected_update_steps(run.cfg))
    assert isinstance(r.online, MLP)
    assert r.online.layer_sizes == (S_OBS, *run.cfg.hidden_sizes, S_ACT)
    assert r.online.dtype == np.dtype(run.cfg.dtype)


def test_float32_config_builds_float32_network(tmp_path, monkeypatch):
    r = Run(_cfg(dtype="float32", total_timesteps=200), tmp_path / "r", monkeypatch)
    assert r.result.online.dtype == np.float32


def test_same_config_gives_byte_identical_logs_and_params(tmp_path, monkeypatch):
    a = Run(_cfg(), tmp_path / "a", monkeypatch)
    b = Run(_cfg(), tmp_path / "b", monkeypatch)
    for name in ("episodes.jsonl", "updates.jsonl"):
        assert (a.run_dir / name).read_bytes() == (b.run_dir / name).read_bytes(), name
    for k in a.result.online.params:
        np.testing.assert_array_equal(a.result.online.params[k], b.result.online.params[k], err_msg=k)


def test_different_seed_gives_different_run(tmp_path, monkeypatch):
    a = Run(_cfg(seed=3), tmp_path / "a", monkeypatch)
    b = Run(_cfg(seed=4), tmp_path / "b", monkeypatch)
    assert (a.run_dir / "episodes.jsonl").read_bytes() != (b.run_dir / "episodes.jsonl").read_bytes()


def test_train_uses_no_global_rng(tmp_path, monkeypatch):
    np.random.seed(9)
    random.seed(9)
    np_before = np.random.get_state()[1].copy()
    py_before = random.getstate()
    Run(_cfg(total_timesteps=300), tmp_path / "r", monkeypatch)
    np.testing.assert_array_equal(np.random.get_state()[1], np_before)
    assert random.getstate() == py_before


def test_network_built_from_network_init_stream(tmp_path, monkeypatch):
    # With learning_starts near the end, no update and no sync ever happen,
    # so the online net is exactly its initialisation.  It must equal a
    # network built from the "network_init" stream.
    from dqn.seeding import make_rng
    cfg = _cfg(total_timesteps=100, learning_starts=99)
    r = Run(cfg, tmp_path / "r", monkeypatch)
    fresh = MLP((S_OBS, *cfg.hidden_sizes, S_ACT), make_rng(cfg.seed, "network_init"),
                dtype=np.dtype(cfg.dtype))
    for k in fresh.params:
        np.testing.assert_array_equal(r.result.online.params[k], fresh.params[k], err_msg=k)


# ======================================================================
# Part 3: evaluation and checkpoints wired into train (step 7b)
#
# The evaluation tests need dqn/evaluate.py implemented; the checkpoint
# tests do not.
# ======================================================================

def _ckpt_steps(run):
    from dqn.checkpoint import _DIR_RE
    d = run.run_dir / "checkpoints"
    return sorted(int(_DIR_RE.match(p.name).group(1)) for p in d.iterdir()
                  if _DIR_RE.match(p.name))


def test_final_checkpoint_always_written_and_matches_result(run):
    from dqn.checkpoint import latest_checkpoint, load_checkpoint
    assert _ckpt_steps(run) == [run.cfg.total_timesteps]
    net, meta = load_checkpoint(latest_checkpoint(run.run_dir / "checkpoints"))
    for k in net.params:
        np.testing.assert_array_equal(net.params[k], run.result.online.params[k], err_msg=k)
    assert meta["episode"] == run.result.episodes
    assert meta["updates"] == run.result.updates
    assert meta["config"] == run.cfg.to_dict()


@pytest.mark.parametrize("total, freq, expected", [
    (600, 250, [250, 500, 600]),     # final added because 600 is not a multiple
    (500, 250, [250, 500]),          # final coincides with a periodic one: no duplicate
])
def test_periodic_checkpoints(tmp_path, monkeypatch, total, freq, expected):
    r = Run(_cfg(total_timesteps=total, checkpoint_frequency=freq), tmp_path / "r", monkeypatch)
    assert _ckpt_steps(r) == expected


def test_checkpoints_do_not_change_training(tmp_path, monkeypatch):
    off = Run(_cfg(), tmp_path / "off", monkeypatch)
    on = Run(_cfg(checkpoint_frequency=100), tmp_path / "on", monkeypatch)
    for name in ("episodes.jsonl", "updates.jsonl"):
        assert (off.run_dir / name).read_bytes() == (on.run_dir / name).read_bytes(), name


EVAL = dict(eval_frequency=200, eval_episodes=3)


def test_eval_records_on_schedule(tmp_path, monkeypatch):
    r = Run(_cfg(**EVAL), tmp_path / "r", monkeypatch)
    evals = read_jsonl(r.run_dir / "evals.jsonl")
    assert [e["global_step"] for e in evals] == [200, 400, 600]
    for e in evals:
        assert len(e["returns"]) == 3
        assert e["mean_return"] == pytest.approx(float(np.mean(e["returns"])), abs=1e-12)


def test_eval_does_not_change_training(tmp_path, monkeypatch):
    # The central 7b property: evaluating must not touch the training env,
    # the exploration or replay streams, the networks or the buffer.
    off = Run(_cfg(), tmp_path / "off", monkeypatch)
    on = Run(_cfg(**EVAL), tmp_path / "on", monkeypatch)
    for name in ("episodes.jsonl", "updates.jsonl"):
        assert (off.run_dir / name).read_bytes() == (on.run_dir / name).read_bytes(), name
    for k in off.result.online.params:
        np.testing.assert_array_equal(off.result.online.params[k], on.result.online.params[k],
                                      err_msg=k)


def test_eval_uses_eval_env_seeds_every_time(tmp_path, monkeypatch):
    from dqn.seeding import stream_seeds
    r = Run(_cfg(**EVAL), tmp_path / "r", monkeypatch)
    seeded = [e[1] for e in r.events if e[0] == "reset" and e[1] is not None]
    expected = stream_seeds(r.cfg.seed, "eval_env", 3)
    assert seeded[0] == stream_seed(r.cfg.seed, "env")      # the training env's one seeded reset
    assert seeded[1:] == expected * 3                        # three evaluations, same seeds
    assert stream_seed(r.cfg.seed, "env") not in expected


def test_eval_is_reproducible_from_the_final_checkpoint(tmp_path, monkeypatch):
    # Greedy + seeded env + fixed seeds: re-evaluating the saved network
    # reproduces the last in-training evaluation exactly.
    from dqn.checkpoint import latest_checkpoint, load_checkpoint
    from dqn.evaluate import evaluate
    from dqn.seeding import stream_seeds
    r = Run(_cfg(**EVAL), tmp_path / "r", monkeypatch)
    last = read_jsonl(r.run_dir / "evals.jsonl")[-1]
    net, _ = load_checkpoint(latest_checkpoint(r.run_dir / "checkpoints"))
    again = evaluate(net, lambda s: StubEnv(s, []), stream_seeds(r.cfg.seed, "eval_env", 3))
    assert again == last["returns"]
