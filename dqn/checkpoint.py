"""Model checkpoints: save and load the online network (build step 7b).

Format: dqn/README.md, "Checkpoint structure", Tier 1.  One directory per
checkpoint, named by the global step it was taken at:

    checkpoints/step_000500000/
        params.npz        online network only, one array per parameter
        checkpoint.json   metadata, hash of params.npz, full config

Writes are atomic: everything goes into `step_XXXXXXXXX.tmp/`, which is
then renamed into place.  A crash mid-save leaves only a `.tmp` directory,
which `latest_checkpoint` and the loader never pick up.  An existing
checkpoint is never overwritten.

Loading fails loudly on anything inconsistent: unknown format version, hash
mismatch, missing or extra arrays, or shapes that disagree with each other,
with obs_dim / n_actions, or with the config's hidden_sizes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .config import DQNConfig
from .network import MLP

FORMAT_VERSION = 1
_DIR_RE = re.compile(r"^step_(\d{9})$")


def _dir_name(global_step: int) -> str:
    return f"step_{global_step:09d}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_checkpoint(checkpoint_dir: str | Path, online: MLP, *, global_step: int,
                    episode: int, updates: int, config: DQNConfig) -> Path:
    """Write `online` to checkpoint_dir/step_{global_step:09d}/ and return
    that path.  `episode` is the number of completed training episodes."""
    if global_step < 0 or global_step > 999_999_999:
        raise ValueError(f"global_step must be in [0, 999999999], got {global_step}")
    checkpoint_dir = Path(checkpoint_dir)
    final = checkpoint_dir / _dir_name(global_step)
    if final.exists():
        raise FileExistsError(f"checkpoint {final} already exists; refusing to overwrite")

    # A leftover .tmp is by definition an interrupted save: discard it.
    tmp = checkpoint_dir / (_dir_name(global_step) + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    np.savez(tmp / "params.npz", **online.params)
    meta: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "global_step": int(global_step),
        "episode": int(episode),
        "updates": int(updates),
        "obs_dim": int(online.layer_sizes[0]),
        "n_actions": int(online.layer_sizes[-1]),
        "dtype": str(online.dtype),
        "param_shapes": {k: list(v.shape) for k, v in online.params.items()},
        "params_sha256": _sha256(tmp / "params.npz"),
        "config": config.to_dict(),
    }
    (tmp / "checkpoint.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    os.replace(tmp, final)
    return final


def load_checkpoint(path: str | Path) -> tuple[MLP, dict[str, Any]]:
    """Load a checkpoint directory.  Returns (network, metadata)."""
    path = Path(path)
    if path.name.endswith(".tmp"):
        raise ValueError(f"{path} is an unfinished save (.tmp), not a checkpoint")
    meta = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))

    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path}: unknown format_version {meta.get('format_version')!r}")
    params_file = path / "params.npz"
    if _sha256(params_file) != meta["params_sha256"]:
        raise ValueError(f"{path}: params.npz does not match its recorded sha256")

    config = DQNConfig.from_dict(meta["config"])
    layer_sizes = (meta["obs_dim"], *config.hidden_sizes, meta["n_actions"])
    expected = {}
    for i, (fan_in, fan_out) in enumerate(zip(layer_sizes, layer_sizes[1:])):
        expected[f"W{i}"] = [fan_in, fan_out]
        expected[f"b{i}"] = [fan_out]
    if meta["param_shapes"] != expected:
        raise ValueError(f"{path}: param_shapes {meta['param_shapes']} disagree with "
                         f"obs_dim, hidden_sizes and n_actions {expected}")

    # allow_pickle=False: weights never need pickle, and unpickling runs code.
    with np.load(params_file, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    if sorted(arrays) != sorted(expected):
        raise ValueError(f"{path}: arrays {sorted(arrays)} != expected {sorted(expected)}")
    for k, shape in expected.items():
        if list(arrays[k].shape) != shape:
            raise ValueError(f"{path}: {k} has shape {arrays[k].shape}, expected {tuple(shape)}")
        if str(arrays[k].dtype) != meta["dtype"]:
            raise ValueError(f"{path}: {k} has dtype {arrays[k].dtype}, expected {meta['dtype']}")

    # MLP's constructor needs an rng; a throwaway one is used and every
    # parameter is then overwritten, so no stream or global RNG is involved.
    net = MLP(layer_sizes, np.random.default_rng(0), dtype=meta["dtype"])
    for k in net.params:
        net.params[k] = np.array(arrays[k], dtype=net.dtype)
    return net, meta


def latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """The finished checkpoint with the highest step, or None.  Unfinished
    `.tmp` directories and anything else in the folder are ignored."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return None
    names = sorted(p.name for p in checkpoint_dir.iterdir()
                   if p.is_dir() and _DIR_RE.match(p.name))
    return checkpoint_dir / names[-1] if names else None
