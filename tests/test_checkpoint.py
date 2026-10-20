"""Model checkpoints (build step 7b, dqn/checkpoint.py).

The tests named in dqn/README.md, "Checkpoint structure": save -> load gives
bit-identical arrays and identical greedy actions; a corrupted file fails the
hash check; a .tmp directory is ignored; a shape mismatch raises.  Plus
format details: directory naming, metadata fields, no overwrite, no pickle.
"""

from __future__ import annotations

import json
import random
import shutil

import numpy as np
import pytest

from dqn.checkpoint import FORMAT_VERSION, latest_checkpoint, load_checkpoint, save_checkpoint
from dqn.config import DQNConfig
from dqn.exploration import greedy_action
from dqn.network import MLP

CFG = DQNConfig().with_overrides({"hidden_sizes": (8, 6)})
OBS_DIM, N_ACT = 4, 3


def _net(seed=0, dtype="float64") -> MLP:
    return MLP((OBS_DIM, *CFG.hidden_sizes, N_ACT), np.random.default_rng(seed), dtype=dtype)


def _save(tmp_path, net=None, step=500, **kw):
    net = net or _net()
    args = dict(global_step=step, episode=17, updates=40, config=CFG)
    args.update(kw)
    return save_checkpoint(tmp_path / "checkpoints", net, **args)


def test_round_trip_bit_identical(tmp_path):
    net = _net()
    loaded, _ = load_checkpoint(_save(tmp_path, net))
    assert list(loaded.params) == list(net.params)
    for k in net.params:
        assert loaded.params[k].dtype == net.params[k].dtype, k
        np.testing.assert_array_equal(loaded.params[k], net.params[k], err_msg=k)
    assert loaded.layer_sizes == net.layer_sizes


def test_round_trip_same_greedy_actions(tmp_path):
    net = _net()
    loaded, _ = load_checkpoint(_save(tmp_path, net))
    obs = np.random.default_rng(1).normal(size=(200, OBS_DIM))
    q1, _ = net.forward(obs)
    q2, _ = loaded.forward(obs)
    np.testing.assert_array_equal(q1, q2)
    assert [greedy_action(r) for r in q1] == [greedy_action(r) for r in q2]


def test_round_trip_float32(tmp_path):
    net = _net(dtype="float32")
    loaded, meta = load_checkpoint(_save(tmp_path, net))
    assert loaded.dtype == np.float32 and meta["dtype"] == "float32"


def test_loaded_network_does_not_share_memory_with_the_saved_one(tmp_path):
    net = _net()
    loaded, _ = load_checkpoint(_save(tmp_path, net))
    for k in net.params:
        assert not np.shares_memory(loaded.params[k], net.params[k]), k


def test_directory_name_is_zero_padded_step(tmp_path):
    path = _save(tmp_path, step=500)
    assert path.name == "step_000000500"
    assert sorted(p.name for p in path.iterdir()) == ["checkpoint.json", "params.npz"]


def test_metadata_fields(tmp_path):
    net = _net()
    _, meta = load_checkpoint(_save(tmp_path, net, step=1234, episode=9, updates=99))
    assert meta["format_version"] == FORMAT_VERSION
    assert (meta["global_step"], meta["episode"], meta["updates"]) == (1234, 9, 99)
    assert (meta["obs_dim"], meta["n_actions"]) == (OBS_DIM, N_ACT)
    assert meta["param_shapes"] == {k: list(v.shape) for k, v in net.params.items()}
    assert meta["config"] == CFG.to_dict()
    assert len(meta["params_sha256"]) == 64


def test_refuses_to_overwrite(tmp_path):
    _save(tmp_path, step=500)
    with pytest.raises(FileExistsError):
        _save(tmp_path, step=500)


def test_no_tmp_left_after_save(tmp_path):
    path = _save(tmp_path)
    assert not any(p.name.endswith(".tmp") for p in path.parent.iterdir())


def test_leftover_tmp_from_a_crash_is_replaced(tmp_path):
    ckpts = tmp_path / "checkpoints"
    stale = ckpts / "step_000000500.tmp"
    stale.mkdir(parents=True)
    (stale / "params.npz").write_bytes(b"half-written garbage")
    path = _save(tmp_path, step=500)
    load_checkpoint(path)
    assert not stale.exists()


def test_corrupted_params_fail_the_hash_check(tmp_path):
    path = _save(tmp_path)
    data = bytearray((path / "params.npz").read_bytes())
    data[-40] ^= 0xFF
    (path / "params.npz").write_bytes(bytes(data))
    with pytest.raises(ValueError, match="sha256"):
        load_checkpoint(path)


def test_params_from_another_checkpoint_fail_the_hash_check(tmp_path):
    a = _save(tmp_path, _net(0), step=100)
    b = _save(tmp_path, _net(1), step=200)
    shutil.copy(b / "params.npz", a / "params.npz")
    with pytest.raises(ValueError, match="sha256"):
        load_checkpoint(a)


def test_shape_mismatch_raises(tmp_path):
    # Consistent file and hash, but metadata claims a different input size.
    path = _save(tmp_path)
    meta = json.loads((path / "checkpoint.json").read_text())
    meta["obs_dim"] = OBS_DIM + 1
    (path / "checkpoint.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="param_shapes"):
        load_checkpoint(path)


def test_hidden_sizes_mismatch_raises(tmp_path):
    path = _save(tmp_path)
    meta = json.loads((path / "checkpoint.json").read_text())
    meta["config"]["hidden_sizes"] = [8, 7]
    (path / "checkpoint.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="param_shapes"):
        load_checkpoint(path)


def test_unknown_format_version_raises(tmp_path):
    path = _save(tmp_path)
    meta = json.loads((path / "checkpoint.json").read_text())
    meta["format_version"] = 99
    (path / "checkpoint.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="format_version"):
        load_checkpoint(path)


def test_pickled_arrays_are_refused(tmp_path):
    # An object array can only be stored with pickle.  Re-hash so only the
    # pickle check stands between the file and the loader.
    path = _save(tmp_path)
    np.savez(path / "params.npz", W0=np.array([object()], dtype=object))
    meta = json.loads((path / "checkpoint.json").read_text())
    import hashlib
    meta["params_sha256"] = hashlib.sha256((path / "params.npz").read_bytes()).hexdigest()
    (path / "checkpoint.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        load_checkpoint(path)


def test_loading_a_tmp_directory_raises(tmp_path):
    path = _save(tmp_path)
    tmp = path.parent / (path.name + ".tmp")
    shutil.copytree(path, tmp)
    with pytest.raises(ValueError, match="tmp"):
        load_checkpoint(tmp)


def test_latest_checkpoint_numeric_order_and_ignores_tmp(tmp_path):
    for step in (9, 10, 100_000):
        _save(tmp_path, step=step)
    ckpts = tmp_path / "checkpoints"
    (ckpts / "step_999999999.tmp").mkdir()
    (ckpts / "notes").mkdir()
    assert latest_checkpoint(ckpts).name == "step_000100000"


def test_latest_checkpoint_none_when_empty_or_missing(tmp_path):
    assert latest_checkpoint(tmp_path / "nope") is None
    (tmp_path / "empty").mkdir()
    assert latest_checkpoint(tmp_path / "empty") is None


def test_save_and_load_use_no_global_rng(tmp_path):
    np.random.seed(4)
    random.seed(4)
    np_before = np.random.get_state()[1].copy()
    py_before = random.getstate()
    load_checkpoint(_save(tmp_path))
    np.testing.assert_array_equal(np.random.get_state()[1], np_before)
    assert random.getstate() == py_before
