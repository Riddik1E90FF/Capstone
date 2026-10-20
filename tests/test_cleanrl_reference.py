"""Step 8: the vendored CleanRL reference and its log converter.

The vendored files must stay byte-identical to the pinned upstream commit
(D029: "run unmodified"), and the converter must map CleanRL's TensorBoard
scalars onto our run-folder schema exactly as its docstring says.  Every
Gate 1a calibration number passes through this converter.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from dqn.config import DQNConfig
from dqn.exploration import linear_epsilon
from dqn.runlog import RunLogger, read_jsonl

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "third_party" / "cleanrl"

EXPECTED_SHA256 = {
    "dqn.py": "a0effc7d087264da1644c3c79eb99cc3ab02b214a758ebf6e9c1443b21ad57d3",
    "cleanrl_utils/buffers.py": "b43d54020783ac53f7d23fe7c0ed72aff20e1f377a633d134d66eb9bdbece4b8",
    "cleanrl_utils/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "LICENSE": "c6c269472eecb009754856b5fe1029ee53b99b986f57bf22ad7c747b04799c69",
}


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_cleanrl_reference", REPO / "scripts" / "run_cleanrl_reference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = _load_runner()


@pytest.mark.parametrize("name, digest", sorted(EXPECTED_SHA256.items()))
def test_vendored_files_unmodified(name, digest):
    # Git may rewrite line endings on checkout (core.autocrlf), which would
    # change the bytes.  Normalise CRLF to LF before hashing, so the check
    # sees edits rather than checkout settings.
    data = (VENDOR / name).read_bytes().replace(b"\r\n", b"\n")
    assert runner.hashlib.sha256(data).hexdigest() == digest, f"{name} differs from upstream"


def test_linear_schedule_copy_matches_ours():
    # The converter recomputes epsilon with CleanRL's formula; it must agree
    # with dqn.exploration on every logged step.
    for t in (0, 1, 9_999, 125_000, 249_999, 250_000, 250_001, 499_999):
        assert runner.linear_schedule(1.0, 0.05, 250_000.0, t) == pytest.approx(
            linear_epsilon(t, start_e=1.0, end_e=0.05, duration=250_000.0), abs=1e-15)


SCALARS = {
    "charts/episodic_return": [(14, 15.0), (499, 500.0), (63, 49.0)],   # out of order on purpose
    "charts/episodic_length": [(63, 49.0), (14, 15.0), (499, 500.0)],
    "losses/td_loss": [(10_100, 0.5), (10_200, 0.25)],
    "losses/q_values": [(10_200, 2.0), (10_100, 1.5)],
}


def _convert(tmp_path, scalars=SCALARS):
    cfg = DQNConfig().with_overrides({"seed": 1})
    with RunLogger(tmp_path / "r", cfg) as logger:
        counts = runner.convert(scalars, logger, total_timesteps=500_000, time_limit=500)
    return counts, read_jsonl(tmp_path / "r" / "episodes.jsonl"), \
        read_jsonl(tmp_path / "r" / "updates.jsonl")


def test_convert_episodes(tmp_path):
    counts, episodes, _ = _convert(tmp_path)
    assert counts == {"episodes": 3, "updates": 2}
    assert [e["global_step"] for e in episodes] == [15, 64, 500]      # 0-based + 1, in step order
    assert [e["episode"] for e in episodes] == [0, 1, 2]
    assert [e["return"] for e in episodes] == [15.0, 49.0, 500.0]
    assert [e["length"] for e in episodes] == [15, 49, 500]
    assert [e["truncated"] for e in episodes] == [False, False, True]
    assert [e["terminated"] for e in episodes] == [True, True, False]
    for e in episodes:
        step0 = e["global_step"] - 1
        assert e["epsilon"] == pytest.approx(
            linear_epsilon(step0, start_e=1.0, end_e=0.05, duration=250_000.0), abs=1e-15)


def test_convert_updates(tmp_path):
    _, _, updates = _convert(tmp_path)
    assert updates == [{"global_step": 10_101, "loss": 0.5, "q_mean": 1.5},
                       {"global_step": 10_201, "loss": 0.25, "q_mean": 2.0}]


def test_convert_rejects_misaligned_return_and_length(tmp_path):
    bad = {**SCALARS, "charts/episodic_length": [(14, 15.0), (64, 49.0), (499, 500.0)]}
    with pytest.raises(ValueError):
        _convert(tmp_path, bad)


def test_read_scalars_round_trip(tmp_path):
    tb = pytest.importorskip("torch.utils.tensorboard")
    writer = tb.SummaryWriter(str(tmp_path / "tb"))
    for step, value in [(14, 15.0), (63, 49.0)]:
        writer.add_scalar("charts/episodic_return", value, step)
    writer.close()
    scalars = runner.read_scalars(tmp_path / "tb")
    assert scalars["charts/episodic_return"] == [(14, 15.0), (63, 49.0)]
