"""Named, independent random streams derived from one master seed.

Why named streams instead of one global RNG
-------------------------------------------
With a single shared generator, every consumer's draws depend on how many
draws every *other* consumer made before it.  Change the replay batch size and
the exploration sequence changes too; add one evaluation episode and training
diverges.  That makes "same seed, same run" fragile, and it makes A/B
comparisons noisier than they need to be.

Here each consumer gets its own stream, keyed by name:

    rng = make_rng(cfg.seed, "exploration")

Streams with different names are statistically independent, and a stream's
sequence depends only on (master seed, name).  It does not depend on the
order streams are created in or on what other streams exist.

The name is hashed with CRC32, not Python's ``hash()``: ``hash()`` of a string
is salted per process, so it would give a different stream on every run.

Stream names used by this package
---------------------------------
Use these exact names, so that a run's seeding is fully described by the
master seed:

    "network_init"   initial weights of the online network
    "exploration"    epsilon-greedy coin flips and random actions
    "replay"         minibatch index sampling
    "env"            training-environment reset seeds
    "eval_env"       evaluation-environment reset seeds (distinct from
                     training seeds; pre-registration §8)

No stream is needed for the target network: it starts as a copy of the
online network.
"""

from __future__ import annotations

import zlib

import numpy as np

STREAMS = ("network_init", "exploration", "replay", "env", "eval_env")


def _seed_sequence(master_seed: int, name: str) -> np.random.SeedSequence:
    if master_seed < 0:
        raise ValueError(f"master_seed must be >= 0, got {master_seed}")
    if not name:
        raise ValueError("stream name must be non-empty")
    return np.random.SeedSequence([master_seed, zlib.crc32(name.encode("utf-8"))])


def make_rng(master_seed: int, name: str) -> np.random.Generator:
    """A fresh Generator for stream `name`.  Same arguments, same sequence."""
    return np.random.default_rng(_seed_sequence(master_seed, name))


def stream_seed(master_seed: int, name: str) -> int:
    """A plain non-negative int for APIs that want an integer seed, such as
    ``env.reset(seed=...)``.  Stable across processes and platforms."""
    return int(_seed_sequence(master_seed, name).generate_state(1, dtype=np.uint32)[0])
