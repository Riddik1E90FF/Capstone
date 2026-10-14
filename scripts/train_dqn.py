"""Train the from-scratch DQN for one seed.

This file is the *plumbing* around the training loop: config resolution, the
environment factory, and the run directory.  The loop itself (build step 7a)
lives in `dqn/train.py`, which does not exist yet.  Until it does, use
`--dry-run` to check what a run would use.

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

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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


def make_env_factory(env_id: str) -> tuple[EnvFactory, int, int]:
    """Return (factory, obs_dim, n_actions) for `env_id`."""
    if env_id.startswith("combat:"):
        from combat_env import N_ACTIONS, N_CONFIGS, OBS_DIM, make_env

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

    try:
        from dqn.train import train
    except ImportError as exc:
        raise SystemExit(f"dqn.train is not available yet (build step 7a): {exc}")

    with RunLogger(run_dir, cfg, extra_meta={"obs_dim": obs_dim, "n_actions": n_actions}) as logger:
        print(f"run dir: {run_dir}")
        # PROVISIONAL call site.  The real signature is fixed in the step-7a
        # interface spec; update this line to match it then.
        train(cfg, env_fn=env_fn, obs_dim=obs_dim, n_actions=n_actions, logger=logger)
    return 0


if __name__ == "__main__":
    sys.exit(main())
