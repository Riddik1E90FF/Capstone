"""Hyperparameter configuration for the from-scratch DQN.

One frozen dataclass holds every number that affects a training run.  The
fully resolved config is written into every run directory (see `runlog.py`),
so a run can always be reproduced from its own folder.

The defaults are **CleanRL `dqn.py` at commit fe8d8a0, CartPole-v1**.  This is
deliberate, not a placeholder: Gate 1a level 4 (D029) compares this DQN's
learning curve against that script run unmodified, and the comparison is only
meaningful if every choice the reference makes is matched.  Each field's
comment names the CleanRL argument it mirrors.  If you change a default here,
you are no longer running the matched comparison.

Combat-environment hyperparameters are **not** chosen yet.  They will be a
separate config file once Gate 1a passes, and must be identical across all
seven experimental conditions (pre-registration §4).

Loading rules
-------------
* A JSON config file may give any subset of fields.  Anything missing takes the
  default below.  Unknown keys are an error, so a typo such as
  ``"learning_rat"`` fails loudly instead of being silently ignored.
* Command-line overrides (``--set key=value``) are applied after the file.
* `validate()` runs after every construction path.  An invalid config never
  reaches the training loop.
"""

from __future__ import annotations

import json
import typing
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

# Values a string-valued field may take.  Each set has one member today: the
# field exists so that the choice is *recorded* in every run's config.json,
# and so a later change (e.g. Huber loss for the combat env's larger reward
# scale) is a visible config diff rather than a silent code change.
INIT_SCHEMES = frozenset({"torch_default"})
LOSSES = frozenset({"mse"})
DTYPES = frozenset({"float64", "float32"})


@dataclass(frozen=True)
class DQNConfig:
    # ------------------------------------------------------------------
    # Run identity
    # ------------------------------------------------------------------

    # Master seed.  Every random stream in a run (network init, exploration,
    # replay sampling, environment) is derived from this one integer by
    # `seeding.py`, so it is the only number needed to replay a run.
    # CleanRL: --seed (default 1).  D029 uses seeds 1-10.
    seed: int = 1

    # Which environment to train on.  Interpreted by the CLI's env factory,
    # never by the `dqn` package itself (it must not import gymnasium; D029):
    #   "CartPole-v1"  -> gymnasium.make("CartPole-v1")      (dev-only)
    #   "combat:<id>"  -> combat_env.make_env(<id>), id in 0..24
    # CleanRL: --env-id.
    env_id: str = "CartPole-v1"

    # ------------------------------------------------------------------
    # Training budget
    # ------------------------------------------------------------------

    # Environment steps (not updates, not episodes).  The x-axis of every
    # learning curve.  CleanRL: --total-timesteps.
    total_timesteps: int = 500_000

    # No gradient updates and no target syncs happen until global_step
    # exceeds this.  The buffer fills with ~uniform-random transitions first,
    # so the first minibatches are not 128 copies of the same few states.
    # CleanRL: --learning-starts.  Note CleanRL's condition is strictly
    # `global_step > learning_starts`.
    learning_starts: int = 10_000

    # One gradient update every `train_frequency` environment steps.  Sets
    # the replay ratio: 500k steps / 10 = ~49k updates per run.
    # CleanRL: --train-frequency.
    train_frequency: int = 10

    # ------------------------------------------------------------------
    # Q-learning
    # ------------------------------------------------------------------

    # Discount factor in the TD target  y = r + gamma * (1 - terminated) * max_a' Q_target(s', a').
    # CleanRL: --gamma.
    gamma: float = 0.99

    # Hard copy online -> target every this many environment steps (counted
    # in global steps, not updates, and only once learning has started).
    # CleanRL: --target-network-frequency, with --tau 1.0, i.e. a hard copy.
    # Soft (Polyak) updates are not supported, so tau has no field here.
    target_network_frequency: int = 500

    # TD loss.  "mse" = mean over the batch of (Q(s,a) - y)^2, matching
    # torch.nn.functional.mse_loss.  The 1/batch_size mean (not sum) matters
    # for the gradient scale handed to the optimiser.
    loss: str = "mse"

    # ------------------------------------------------------------------
    # Replay buffer
    # ------------------------------------------------------------------

    # Capacity in transitions.  Oldest are overwritten first (circular).
    # CleanRL: --buffer-size.  Small for CartPole; the combat env will need
    # its own value, chosen with curriculum forgetting in mind.
    buffer_size: int = 10_000

    # Transitions per gradient update.  CleanRL: --batch-size.
    batch_size: int = 128

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------

    # Hidden-layer widths of the MLP, ReLU between layers, linear output of
    # size n_actions.  Input size comes from the environment.
    # CleanRL QNetwork: obs -> 120 -> 84 -> n_actions.
    hidden_sizes: tuple[int, ...] = (120, 84)

    # Weight/bias initialisation.  "torch_default" means what nn.Linear does
    # out of the box: weights and biases both ~ U(-1/sqrt(fan_in), +1/sqrt(fan_in)).
    # Matching the reference's init matters for L4: init sets the
    # early-training Q-value scale, which shows up in the curve.
    init_scheme: str = "torch_default"

    # Floating-point precision of parameters, activations and gradients.
    # float64, chosen 2026-10-13: float32 measured 1.6-2.6x faster per update,
    # but compute is not binding (pre-registration §9), and float64 removes
    # silent mixed-dtype bugs.  CombatEnv emits float64 observations, which
    # would quietly upcast a float32 network.  Gradient checks also run in
    # the same dtype as training.  CleanRL trains in float32; at 10 seeds,
    # seed variance swamps any precision effect on the L4 comparison.
    # Switching means re-running Gate 1a levels 1 and 2 in the new dtype.
    dtype: str = "float64"

    # ------------------------------------------------------------------
    # Optimiser (Adam)
    # ------------------------------------------------------------------

    # CleanRL: --learning-rate.  Betas and eps are torch.optim.Adam's
    # defaults, which CleanRL does not override.
    learning_rate: float = 2.5e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    # ------------------------------------------------------------------
    # Exploration (epsilon-greedy, linear schedule)
    # ------------------------------------------------------------------

    # epsilon(t) = max(start_e + (end_e - start_e) * t / duration, end_e),
    # where duration = exploration_fraction * total_timesteps.
    # CleanRL: --start-e, --end-e, --exploration-fraction.
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.5

    # ------------------------------------------------------------------
    # Logging, evaluation, checkpointing
    # ------------------------------------------------------------------

    # Write a training-update log record (loss, mean Q) whenever an update
    # happens on a step divisible by this.  CleanRL logs every 100 global
    # steps; matching it keeps the two loss curves at the same resolution.
    update_log_frequency: int = 100

    # Greedy evaluation every this many env steps.  0 = off.  Gate 1a (D029)
    # uses training-episode returns, as CleanRL logs them, so the default is
    # off.  The combat experiment will need it (pre-registration §8).
    eval_frequency: int = 0
    eval_episodes: int = 10
    # Pre-registration §8: "exploration disabled" for the primary DV.
    eval_epsilon: float = 0.0

    # Save a model checkpoint every this many env steps.  0 = final only.
    # Format: see dqn/README.md, "Checkpoint structure".
    checkpoint_frequency: int = 0

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def exploration_steps(self) -> float:
        """Steps over which epsilon anneals.  CleanRL passes the same float."""
        return self.exploration_fraction * self.total_timesteps

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> "DQNConfig":
        """Raise ValueError on any invalid combination.  Returns self for chaining."""
        errors: list[str] = []

        def need(cond: bool, msg: str) -> None:
            if not cond:
                errors.append(msg)

        for name in ("total_timesteps", "train_frequency", "target_network_frequency",
                     "buffer_size", "batch_size", "update_log_frequency", "eval_episodes"):
            need(getattr(self, name) >= 1, f"{name} must be >= 1")
        for name in ("seed", "learning_starts", "eval_frequency", "checkpoint_frequency"):
            need(getattr(self, name) >= 0, f"{name} must be >= 0")

        need(0.0 <= self.gamma < 1.0, "gamma must be in [0, 1)")
        need(self.batch_size <= self.buffer_size, "batch_size must be <= buffer_size")
        # Updates sample batch_size transitions from what is stored, so at
        # least that many must exist by the first update.
        need(self.learning_starts >= self.batch_size,
             "learning_starts must be >= batch_size (buffer must hold a full batch "
             "before the first update)")
        need(self.learning_starts < self.total_timesteps,
             "learning_starts must be < total_timesteps (otherwise nothing trains)")

        need(len(self.hidden_sizes) >= 1, "hidden_sizes must have at least one layer")
        need(all(h >= 1 for h in self.hidden_sizes), "hidden_sizes entries must be >= 1")
        need(self.init_scheme in INIT_SCHEMES, f"init_scheme must be one of {sorted(INIT_SCHEMES)}")
        need(self.loss in LOSSES, f"loss must be one of {sorted(LOSSES)}")
        need(self.dtype in DTYPES, f"dtype must be one of {sorted(DTYPES)}")

        need(self.learning_rate > 0, "learning_rate must be > 0")
        need(0.0 <= self.adam_beta1 < 1.0, "adam_beta1 must be in [0, 1)")
        need(0.0 <= self.adam_beta2 < 1.0, "adam_beta2 must be in [0, 1)")
        need(self.adam_eps > 0, "adam_eps must be > 0")

        for name in ("start_e", "end_e", "eval_epsilon"):
            need(0.0 <= getattr(self, name) <= 1.0, f"{name} must be in [0, 1]")
        need(self.end_e <= self.start_e, "end_e must be <= start_e (epsilon anneals downward)")
        need(0.0 < self.exploration_fraction <= 1.0, "exploration_fraction must be in (0, 1]")

        if errors:
            raise ValueError("invalid DQNConfig:\n  " + "\n  ".join(errors))
        return self

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["hidden_sizes"] = list(self.hidden_sizes)  # JSON has no tuples
        return d

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DQNConfig":
        """Build from a (possibly partial) dict.  Unknown keys raise."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown DQNConfig key(s): {unknown}")
        hints = _field_types()
        coerced = {k: _coerce(v, hints[k], k) for k, v in data.items()}
        return cls(**coerced).validate()

    @classmethod
    def from_json(cls, path: str | Path) -> "DQNConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def with_overrides(self, overrides: dict[str, Any]) -> "DQNConfig":
        """Return a copy with fields replaced.  Values may be strings (from the
        command line) or already-typed values."""
        known = {f.name for f in fields(self)}
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise ValueError(f"unknown DQNConfig key(s): {unknown}")
        hints = _field_types()
        coerced = {k: _coerce(v, hints[k], k) for k, v in overrides.items()}
        return replace(self, **coerced).validate()


# ----------------------------------------------------------------------
# Type coercion.  JSON gives ints where floats are wanted ("gamma": 1 is not
# a bug) and lists where tuples are wanted; the CLI gives strings for
# everything.  Both are normalised here so the dataclass only ever holds the
# declared types.
# ----------------------------------------------------------------------

def _field_types() -> dict[str, Any]:
    return typing.get_type_hints(DQNConfig)


def _coerce(value: Any, typ: Any, name: str) -> Any:
    try:
        if typ is bool:
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in ("1", "true", "yes", "on"):
                    return True
                if lowered in ("0", "false", "no", "off"):
                    return False
                raise ValueError(value)
            return bool(value)
        if typ is int:
            if isinstance(value, bool):
                raise ValueError("bool is not an int here")
            if isinstance(value, float) and not value.is_integer():
                raise ValueError(f"{value} is not a whole number")
            if isinstance(value, str):
                # Allow "500_000" and "5e5" on the command line.
                as_float = float(value.replace("_", ""))
                if not as_float.is_integer():
                    raise ValueError(f"{value} is not a whole number")
                return int(as_float)
            return int(value)
        if typ is float:
            if isinstance(value, bool):
                raise ValueError("bool is not a float here")
            return float(value)
        if typ is str:
            if not isinstance(value, str):
                raise ValueError(f"expected a string, got {type(value).__name__}")
            return value
        if typing.get_origin(typ) is tuple:
            if isinstance(value, str):
                value = [v for v in value.replace(" ", "").split(",") if v]
            return tuple(int(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bad value for {name!r}: {value!r} ({exc})") from None
    raise TypeError(f"no coercion rule for field {name!r} of type {typ!r}")


def parse_override(text: str) -> tuple[str, str]:
    """Split a ``key=value`` command-line override."""
    if "=" not in text:
        raise ValueError(f"override must look like key=value, got {text!r}")
    key, value = text.split("=", 1)
    return key.strip(), value.strip()
