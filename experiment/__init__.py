"""Experiment-level infrastructure for the curriculum comparison.

One experiment run is one (condition, seed) pair (pre-registration §9).  This
package holds what is shared across conditions and is not part of any
learning algorithm.  It imports only the standard library.

The frontier sampler, the RL algorithm and the EEG difficulty index are
written by the project author and do not live here.
"""

from .manifest import CONDITIONS, RunManifest, build_manifest, read_manifest

__all__ = ["CONDITIONS", "RunManifest", "build_manifest", "read_manifest"]
