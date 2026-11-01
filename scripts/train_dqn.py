"""Train the from-scratch DQN for one seed.

This file is the *plumbing* around the training loop: config resolution, the
environment factory, and the run directory.  The loop itself (build step 7a)
lives in `dqn/train.py`.  Use `--dry-run` to check what a run would use
without training.

Config resolution, later steps winning:

    1. DQNConfig defaults          (CleanRL dqn.py, CartPole-v1; see dqn/config.py)
    2. --config FILE               any subset of fields, as JSON
    3. --seed N                    shorthand for --set seed=N
    4. --set key=value ...         repeatable, any field

Usage:
    python scripts/train_dqn.py --dry-run
    python scripts/train_dqn.py --config configs/dqn/cartpole_cleanrl.json --seed 3
    python scripts/train_dqn.py --set env_id=combat:7 --set total_timesteps=1e6 --dry-run

Environment ids:
    CartPole-v1 (or any Gymnasium id)   dev-only; needs requirements-dev.txt
    combat:<config_id>                  combat_env difficulty cell 0..24

The run directory is runs/<env>__<tag>__seed<N>__<UTC timestamp>/ unless
--run-dir is given.  An existing non-empty directory is never overwritten.
"""

from __future__ import annotations

# Make this script runnable from any directory: put the repo root (this
# file's parent's parent) on sys.path so `combat_env` and `dqn` import
# without PYTHONPATH.  Resolved from __file__, so a git worktree imports
# its own copy rather than another checkout's.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Scripts run as `python scripts/train_dqn.py` get scripts/ on sys.path, not
# the repo root, so the packages next to it would not be importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dqn.config import DQNConfig, parse_override  # noqa: E402
from dqn.runlog import RunLogger  # noqa: E402

# An env factory takes an integer seed and returns a fresh environment with
# reset(seed=...) -> (obs, info) and step(action) -> (obs, reward, terminated,
# truncated, info).  The training loop creates environments only through it,
# so `dqn` never needs to know which library the environment came from.
EnvFactory = Callable[[int | None], Any]


# Distinguishes the configuration-selection stream from every environment
# stream, so the two cannot accidentally coincide for some seed.
_SELECTOR_DOMAIN = 0x6D6978          # "mix"


class _MixtureEnv:
    """Uniform mixture over all 25 configurations, drawn per episode.

    The selection has to happen in ``reset``, not in the factory.  ``train``
    calls ``env_fn`` exactly once per run and then advances episodes with bare
    ``env.reset()`` calls, so a factory that picked a configuration and
    returned a fixed environment would train every episode on one cell and
    produce a difficulty sweep that looked flat for no visible reason.

    This is also the seam the curriculum sampler replaces later: everything
    outside ``_draw_config`` stays the same when uniform sampling becomes
    competence-tracked frontier sampling.  Nothing here is privileged
    information: the configuration never enters the observation, so the agent
    sees only the 18-dim vector and cannot condition on which cell it is in.

    One environment is kept per configuration and reused, so each cell's own
    RNG carries on across the episodes it appears in.  That matches the
    single-configuration path, where ``train`` resets the same environment
    without a seed.
    """

    def __init__(self, seed: int | None) -> None:
        from combat_env import N_CONFIGS

        self._n_configs = N_CONFIGS
        self._rng = self._selector_rng(seed)
        self._envs: dict[int, Any] = {}
        self._env: Any = None
        self._config_id: int | None = None

    @staticmethod
    def _selector_rng(seed: int | None) -> np.random.Generator:
        """A stream of its own, derived from but distinct from the run seed."""
        if seed is None:
            return np.random.default_rng()
        return np.random.default_rng([int(seed), _SELECTOR_DOMAIN])

    @property
    def config_id(self) -> int | None:
        """Which configuration the current episode is running."""
        return self._config_id

    def _draw_config(self) -> int:
        """Uniform over all configurations.  The curriculum sampler goes here."""
        return int(self._rng.integers(self._n_configs))

    def reset(self, seed: int | None = None, **kwargs: Any) -> tuple[Any, dict]:
        from combat_env import make_env

        if seed is not None:
            # A seeded reset restarts the whole mixture deterministically,
            # including the per-configuration environments, so the same seed
            # always replays the same sequence of cells.
            self._rng = self._selector_rng(seed)
            self._envs.clear()

        cid = self._draw_config()
        self._config_id = cid

        env = self._envs.get(cid)
        if env is None:
            cell_seed = int(self._rng.integers(1 << 31))
            env = make_env(cid, seed=cell_seed)
            self._envs[cid] = env
            obs, info = env.reset(seed=cell_seed, **kwargs)
        else:
            # No seed: this cell's RNG carries on, as in the single-config path.
            obs, info = env.reset(**kwargs)

        self._env = env
        info = dict(info)
        info["config_id"] = cid
        return obs, info

    def step(self, action: int):
        if self._env is None:
            raise RuntimeError("reset() must be called before step()")
        return self._env.step(action)

    def __getattr__(self, name: str) -> Any:
        # Anything not defined here belongs to the configuration currently in
        # play.  Guarded on the private names so lookups during __init__,
        # before _env exists, raise AttributeError rather than recursing.
        if name.startswith("_"):
            raise AttributeError(name)
        env = self.__dict__.get("_env")
        if env is None:
            raise AttributeError(
                f"{name!r} is unavailable until reset() has chosen a configuration")
        return getattr(env, name)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"_MixtureEnv(config_id={self._config_id})"


def _mixture_factory() -> EnvFactory:
    """EnvFactory for ``combat:mix``: same (seed) -> env shape as a single cell."""
    return lambda seed: _MixtureEnv(seed)


def make_env_factory(env_id: str) -> tuple[EnvFactory, int, int]:
    """Return (factory, obs_dim, n_actions) for `env_id`."""
    if env_id.startswith("combat:"):
        from combat_env import N_ACTIONS, N_CONFIGS, OBS_DIM, make_env

        if env_id == "combat:mix":
            return _mixture_factory(), OBS_DIM, N_ACTIONS

        try:
            config_id = int(env_id.split(":", 1)[1])
        except ValueError:
            raise SystemExit(f"bad combat env id {env_id!r}; expected combat:<0..{N_CONFIGS - 1}>")
        if not 0 <= config_id < N_CONFIGS:
            raise SystemExit(f"combat config id must be in 0..{N_CONFIGS - 1}, got {config_id}")
        return (lambda seed: make_env(config_id, seed=seed)), OBS_DIM, N_ACTIONS


    # Anything else is treated as a Gymnasium id.  Imported here, not at the
    # top, so combat-env runs do not need Gymnasium installed at all.
    try:
        import gymnasium as gym
    except ImportError:
        raise SystemExit(
            f"env_id {env_id!r} needs Gymnasium, which is a dev-only dependency.\n"
            "Install it with:  pip install -r requirements-dev.txt")

    probe = gym.make(env_id)
    if not isinstance(probe.action_space, gym.spaces.Discrete):
        raise SystemExit(f"{env_id} does not have a discrete action space; DQN needs one")
    obs_shape = probe.observation_space.shape
    if obs_shape is None or len(obs_shape) != 1:
        raise SystemExit(f"{env_id} observation must be a flat vector, got shape {obs_shape}")
    obs_dim, n_actions = int(obs_shape[0]), int(probe.action_space.n)
    probe.close()
    # Gymnasium environments take their seed in reset(), not the constructor,
    # so the seed argument is ignored here.  The loop seeds via reset(seed=...).
    return (lambda seed: gym.make(env_id)), obs_dim, n_actions


def resolve_config(args: argparse.Namespace) -> DQNConfig:
    cfg = DQNConfig.from_json(args.config) if args.config else DQNConfig().validate()
    overrides: dict[str, str] = {}
    if args.seed is not None:
        overrides["seed"] = str(args.seed)
    for item in args.set:
        try:
            key, value = parse_override(item)
        except ValueError as exc:
            raise SystemExit(str(exc))
        overrides[key] = value
    try:
        return cfg.with_overrides(overrides) if overrides else cfg
    except ValueError as exc:
        raise SystemExit(str(exc))


def default_run_dir(cfg: DQNConfig, runs_root: Path, tag: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    env = cfg.env_id.replace(":", "-").replace("/", "-")
    return runs_root / f"{env}__{tag}__seed{cfg.seed}__{stamp}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the from-scratch NumPy DQN for one seed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Fields and defaults: see dqn/config.py.  --dry-run prints them all.")
    p.add_argument("--config", type=Path, help="JSON file with any subset of DQNConfig fields")
    p.add_argument("--seed", type=int, help="master seed (shorthand for --set seed=N)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override one config field; repeatable")
    p.add_argument("--runs-root", type=Path, default=REPO_ROOT / "runs",
                   help="parent directory for run folders (default: runs/, git-ignored)")
    p.add_argument("--tag", default="dqn", help="label inserted into the run folder name")
    p.add_argument("--run-dir", type=Path, help="exact run directory; overrides --runs-root/--tag")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve config and env, print them, create nothing, and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = resolve_config(args)
    env_fn, obs_dim, n_actions = make_env_factory(cfg.env_id)
    run_dir = args.run_dir or default_run_dir(cfg, args.runs_root, args.tag)

    if args.dry_run:
        print(json.dumps(cfg.to_dict(), indent=2))
        print(f"\nenv: {cfg.env_id}  obs_dim={obs_dim}  n_actions={n_actions}")
        print(f"run dir (not created): {run_dir}")
        return 0

    # Imported here so --dry-run works even if the loop is broken.
    from dqn.train import train

    with RunLogger(run_dir, cfg, extra_meta={"obs_dim": obs_dim, "n_actions": n_actions}) as logger:
        print(f"run dir: {run_dir}")
        result = train(cfg, env_fn=env_fn, obs_dim=obs_dim, n_actions=n_actions, logger=logger)
        print(f"done: {result.global_steps} env steps, {result.episodes} episodes, "
              f"{result.updates} updates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
