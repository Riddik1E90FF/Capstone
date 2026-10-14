"""Tests for the DQN scaffolding (build step 0): config, seeding, run logging,
the gradient checker, and the D028/D029 import boundary.

These test plumbing, not learning.  Component tests (steps 1-6) go in their own
files, written from the agreed interface spec before each component exists.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from dqn.config import DQNConfig, parse_override
from dqn.gradcheck import assert_gradient_close, numerical_gradient, relative_error
from dqn.runlog import REQUIRED_KEYS, RunLogger, read_jsonl
from dqn.seeding import STREAMS, make_rng, stream_seed

REPO = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

# CleanRL cleanrl/dqn.py @ fe8d8a0 Args defaults, plus torch.optim.Adam and
# nn.Linear defaults.  If this test fails, the L4 comparison is no longer
# matched (D029): fix the default, not the test.
CLEANRL_DEFAULTS = {
    "seed": 1,
    "env_id": "CartPole-v1",
    "total_timesteps": 500_000,
    "learning_rate": 2.5e-4,
    "buffer_size": 10_000,
    "gamma": 0.99,
    "target_network_frequency": 500,
    "batch_size": 128,
    "start_e": 1.0,
    "end_e": 0.05,
    "exploration_fraction": 0.5,
    "learning_starts": 10_000,
    "train_frequency": 10,
    "hidden_sizes": (120, 84),
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_eps": 1e-8,
    "loss": "mse",
    "init_scheme": "torch_default",
}


def test_defaults_match_cleanrl_reference():
    cfg = DQNConfig()
    for key, expected in CLEANRL_DEFAULTS.items():
        assert getattr(cfg, key) == expected, key
    assert cfg.exploration_steps == 250_000


def test_cartpole_config_file_matches_defaults():
    from_file = DQNConfig.from_json(REPO / "configs" / "dqn" / "cartpole_cleanrl.json")
    assert from_file == DQNConfig()


def test_config_json_roundtrip(tmp_path):
    cfg = DQNConfig(seed=7, hidden_sizes=(64, 64, 32), learning_rate=1e-3)
    cfg.to_json(tmp_path / "c.json")
    assert DQNConfig.from_json(tmp_path / "c.json") == cfg


def test_config_is_frozen():
    with pytest.raises(FrozenInstanceError):
        DQNConfig().seed = 2  # type: ignore[misc]


def test_partial_dict_takes_defaults_and_coerces_types():
    cfg = DQNConfig.from_dict({"gamma": 1 - 1e-2, "learning_rate": 1, "hidden_sizes": [32]})
    assert cfg.learning_rate == 1.0 and isinstance(cfg.learning_rate, float)
    assert cfg.hidden_sizes == (32,)
    assert cfg.batch_size == 128


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="learning_rat"):
        DQNConfig.from_dict({"learning_rat": 1e-3})
    with pytest.raises(ValueError, match="learning_rat"):
        DQNConfig().with_overrides({"learning_rat": "1e-3"})


@pytest.mark.parametrize("text, field, expected", [
    ("total_timesteps=1e6", "total_timesteps", 1_000_000),
    ("total_timesteps=500_000", "total_timesteps", 500_000),
    ("hidden_sizes=128,128", "hidden_sizes", (128, 128)),
    ("learning_rate=3e-4", "learning_rate", 3e-4),
    ("env_id=combat:7", "env_id", "combat:7"),
])
def test_cli_string_overrides(text, field, expected):
    key, value = parse_override(text)
    assert getattr(DQNConfig().with_overrides({key: value}), field) == expected


@pytest.mark.parametrize("overrides", [
    {"batch_size": "2.5"},          # non-integer for an int field
    {"gamma": "1.0"},               # gamma must be < 1
    {"batch_size": "20000"},        # larger than buffer
    {"learning_starts": "64"},      # fewer than one batch before first update
    {"end_e": "0.5", "start_e": "0.1"},
    {"exploration_fraction": "0"},
    {"init_scheme": "xavier"},      # not supported, so not silently accepted
    {"hidden_sizes": ""},
    {"dtype": "float16"},
])
def test_invalid_configs_rejected(overrides):
    with pytest.raises(ValueError):
        DQNConfig().with_overrides(overrides)


def test_bad_override_syntax():
    with pytest.raises(ValueError):
        parse_override("seed")


# ----------------------------------------------------------------------
# Seeding
# ----------------------------------------------------------------------

def test_same_seed_and_name_reproduce():
    a = make_rng(3, "replay").integers(0, 1_000_000, size=20)
    b = make_rng(3, "replay").integers(0, 1_000_000, size=20)
    assert np.array_equal(a, b)


def test_streams_are_distinct():
    draws = {name: tuple(make_rng(3, name).integers(0, 2**31, size=8)) for name in STREAMS}
    assert len(set(draws.values())) == len(STREAMS)
    assert stream_seed(3, "env") != stream_seed(3, "eval_env")
    assert stream_seed(3, "env") != stream_seed(4, "env")


def test_stream_independent_of_other_streams_use():
    # Drawing heavily from one stream must not shift another.
    expected = make_rng(5, "exploration").random(10)
    make_rng(5, "replay").random(100_000)
    assert np.array_equal(make_rng(5, "exploration").random(10), expected)


def test_stream_seed_stable_across_processes():
    # Guards against anyone swapping CRC32 for Python's salted hash().
    code = "from dqn.seeding import stream_seed; print(stream_seed(11, 'env'))"
    out = subprocess.check_output([sys.executable, "-c", code], cwd=REPO, text=True)
    assert int(out) == stream_seed(11, "env")


def test_negative_seed_rejected():
    with pytest.raises(ValueError):
        make_rng(-1, "env")


# ----------------------------------------------------------------------
# Run logging
# ----------------------------------------------------------------------

def _episode(**kw):
    # "return" is a keyword, so it can't be a keyword argument name.
    rec = {"global_step": 10, "episode": 0, "return": 9.0, "length": 10,
           "terminated": True, "truncated": False, "epsilon": 1.0}
    rec.update(kw)
    return rec


def test_logger_writes_layout_and_records(tmp_path):
    run = tmp_path / "run"
    with RunLogger(run, DQNConfig(seed=4)) as log:
        log.log_episode(**_episode(global_step=np.int64(10), epsilon=np.float32(0.5)))
        log.log_update(global_step=200, loss=float("nan"), q_mean=1.5)
    assert DQNConfig.from_json(run / "config.json") == DQNConfig(seed=4)
    meta = json.loads((run / "meta.json").read_text())
    assert {"git_commit", "git_dirty", "numpy", "python"} <= meta.keys()
    eps = read_jsonl(run / "episodes.jsonl")
    assert eps[0]["global_step"] == 10 and eps[0]["epsilon"] == 0.5
    assert np.isnan(read_jsonl(run / "updates.jsonl")[0]["loss"])
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["records"] == {"episodes": 1, "updates": 1, "evals": 0}
    assert (run / "checkpoints").is_dir()


def test_logger_refuses_to_overwrite_a_run(tmp_path):
    run = tmp_path / "run"
    with RunLogger(run, DQNConfig()):
        pass
    with pytest.raises(FileExistsError):
        RunLogger(run, DQNConfig())


@pytest.mark.parametrize("stream", sorted(REQUIRED_KEYS))
def test_logger_enforces_required_keys(tmp_path, stream):
    with RunLogger(tmp_path / "run", DQNConfig()) as log:
        with pytest.raises(KeyError):
            getattr(log, f"log_{stream[:-1]}")(global_step=1)


def test_logger_marks_crashed_runs(tmp_path):
    run = tmp_path / "run"
    with pytest.raises(RuntimeError):
        with RunLogger(run, DQNConfig()):
            raise RuntimeError("boom")
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "crashed" and "boom" in summary["error"]


# ----------------------------------------------------------------------
# Gradient checker (the oracle for build step 1, so it gets tested too)
# ----------------------------------------------------------------------

def test_gradcheck_accepts_correct_gradient():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 4))
    W = rng.normal(size=(4, 2))
    f = lambda: float(np.sum(np.tanh(x @ W) ** 2))  # noqa: E731
    t = np.tanh(x @ W)
    analytic = (2 * t * (1 - t ** 2)) @ W.T
    assert_gradient_close(f, x, analytic)


def test_gradcheck_rejects_scaled_and_sign_flipped_gradient():
    x = np.linspace(-1.0, 1.0, 6)
    f = lambda: float(np.sum(x ** 3))  # noqa: E731
    with pytest.raises(AssertionError, match="ratio 2.0000"):
        assert_gradient_close(f, x, 2 * (3 * x ** 2))
    with pytest.raises(AssertionError):
        assert_gradient_close(f, x, -(3 * x ** 2))


def test_numerical_gradient_restores_input():
    x = np.arange(6, dtype=np.float64).reshape(2, 3)
    before = x.copy()
    numerical_gradient(lambda: float(np.sum(x ** 2)), x)
    assert np.array_equal(x, before)


def test_gradcheck_refuses_float32():
    x = np.ones(3, dtype=np.float32)
    with pytest.raises(TypeError):
        numerical_gradient(lambda: float(np.sum(x)), x)


def test_relative_error_handles_zero_gradients():
    assert relative_error(np.zeros(3), np.zeros(3)) == 0.0


# ----------------------------------------------------------------------
# Import boundary (D028: no RL/DL library in the agent; D029: gymnasium and
# torch are dev-only)
# ----------------------------------------------------------------------

FORBIDDEN = {"torch", "gymnasium", "gym", "stable_baselines3", "tensorflow", "jax", "cleanrl"}


def test_dqn_package_has_no_forbidden_imports_static():
    offenders = []
    for path in sorted((REPO / "dqn").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                if name.split(".")[0] in FORBIDDEN:
                    offenders.append(f"{path.relative_to(REPO)}:{node.lineno} imports {name}")
    assert not offenders, "\n".join(offenders)


def test_importing_dqn_loads_no_forbidden_module():
    # Catches indirect imports the AST scan cannot see.
    code = ("import sys, pkgutil, importlib, dqn\n"
            "for m in pkgutil.walk_packages(dqn.__path__, 'dqn.'): importlib.import_module(m.name)\n"
            f"bad = sorted(m for m in sys.modules if m.split('.')[0] in {sorted(FORBIDDEN)!r})\n"
            "print(','.join(bad))")
    out = subprocess.check_output([sys.executable, "-c", code], cwd=REPO, text=True).strip()
    assert out == "", f"importing dqn loaded: {out}"


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _load_cli():
    spec = importlib.util.spec_from_file_location("train_dqn", REPO / "scripts" / "train_dqn.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_dry_run_resolves_combat_env(tmp_path, capsys):
    cli = _load_cli()
    rc = cli.main(["--dry-run", "--seed", "3", "--set", "env_id=combat:7",
                   "--runs-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "obs_dim=18" in out and "n_actions=12" in out and '"seed": 3' in out
    assert not any(tmp_path.iterdir()), "dry run must not create anything"


def test_cli_rejects_bad_combat_id():
    with pytest.raises(SystemExit):
        _load_cli().main(["--dry-run", "--set", "env_id=combat:25"])
