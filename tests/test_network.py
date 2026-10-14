"""Build step 1: the MLP.  Written from the agreed interface spec before the
implementation exists (dqn/README.md).

Interface under test (dqn/network.py):

    class MLP:
        layer_sizes: tuple[int, ...]        # (obs_dim, *hidden_sizes, n_actions)
        dtype: np.dtype
        params: dict[str, np.ndarray]       # "W0","b0","W1","b1",...
                                            # W_l: (n_l, n_{l+1}), b_l: (n_{l+1},)

        def __init__(self, layer_sizes, rng, dtype=np.float64): ...
        def forward(self, x) -> tuple[np.ndarray, Cache]: ...
        def backward(self, cache, grad_q) -> dict[str, np.ndarray]: ...

The oracle for backward is finite differences (dqn/gradcheck.py), so these
tests do not encode the backward formulas.  They encode what a correct
gradient must equal.

Skipped as a whole until dqn/network.py exists.  Once the file exists, an
import error inside it fails loudly instead of skipping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dqn.gradcheck import assert_gradient_close

if not (Path(__file__).resolve().parents[1] / "dqn" / "network.py").exists():
    pytest.skip("dqn/network.py not written yet (build step 1)", allow_module_level=True)

from dqn.network import MLP  # noqa: E402

CARTPOLE = (4, 120, 84, 2)
COMBAT = (18, 128, 128, 12)


def _keys(n_layers: int) -> list[str]:
    return [k for i in range(n_layers) for k in (f"W{i}", f"b{i}")]


def _set_params(net: MLP, values: dict[str, list]) -> None:
    for k, v in values.items():
        net.params[k] = np.array(v, dtype=net.dtype)


def _preactivations(net: MLP, x: np.ndarray) -> list[np.ndarray]:
    """Hidden-layer pre-activations, computed independently of the
    implementation.  Used only to keep gradient-check inputs off the ReLU kink."""
    zs, h = [], x
    n = len(net.layer_sizes) - 1
    for i in range(n):
        z = h @ net.params[f"W{i}"] + net.params[f"b{i}"]
        if i < n - 1:
            zs.append(z)
            h = np.maximum(z, 0.0)
    return zs


def _off_kink_input(net: MLP, rng: np.random.Generator, batch: int, margin: float = 1e-3):
    for _ in range(100):
        x = rng.normal(size=(batch, net.layer_sizes[0]))
        if all(np.min(np.abs(z)) > margin for z in _preactivations(net, x)):
            return x
    raise RuntimeError("could not draw an input clear of the ReLU kink")


# ----------------------------------------------------------------------
# Construction and initialisation
# ----------------------------------------------------------------------

@pytest.mark.parametrize("sizes", [CARTPOLE, COMBAT, (3, 2)])
def test_param_keys_and_shapes(sizes):
    net = MLP(sizes, np.random.default_rng(0))
    n = len(sizes) - 1
    assert list(net.params) == _keys(n), "keys must be W0, b0, W1, b1, ... in layer order"
    for i in range(n):
        assert net.params[f"W{i}"].shape == (sizes[i], sizes[i + 1]), "W must be (in, out)"
        assert net.params[f"b{i}"].shape == (sizes[i + 1],)
    assert tuple(net.layer_sizes) == sizes


@pytest.mark.parametrize("bad", [(4,), (), (4, 0, 2), (0, 2), (4, -1, 2)])
def test_invalid_layer_sizes_rejected(bad):
    with pytest.raises(ValueError):
        MLP(bad, np.random.default_rng(0))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_init_is_torch_default_with_specified_draw_order(dtype):
    # Spec: for each layer i in order, W_i then b_i, each drawn as
    #   rng.uniform(-bound, bound, size=shape), bound = 1/sqrt(fan_in),
    #   fan_in = layer_sizes[i], then cast to dtype.
    net = MLP(CARTPOLE, np.random.default_rng(123), dtype=dtype)
    rng = np.random.default_rng(123)
    for i, (fan_in, fan_out) in enumerate(zip(CARTPOLE, CARTPOLE[1:])):
        bound = 1.0 / np.sqrt(fan_in)
        expected_W = rng.uniform(-bound, bound, size=(fan_in, fan_out)).astype(dtype)
        expected_b = rng.uniform(-bound, bound, size=(fan_out,)).astype(dtype)
        np.testing.assert_array_equal(net.params[f"W{i}"], expected_W, err_msg=f"W{i}")
        np.testing.assert_array_equal(net.params[f"b{i}"], expected_b, err_msg=f"b{i}")


def test_init_fan_in_axis():
    # First CartPole layer is 4 -> 120: the correct bound is 1/sqrt(4) = 0.5.
    # Using fan_out by mistake gives 1/sqrt(120) ~= 0.09.  With 480 draws the
    # observed max sits close to 0.5 under the correct bound.
    W0 = MLP(CARTPOLE, np.random.default_rng(0)).params["W0"]
    assert np.max(np.abs(W0)) <= 0.5
    assert np.max(np.abs(W0)) > 0.4, "init bound looks like it used fan_out, not fan_in"


def test_same_rng_seed_same_params_different_seed_different():
    a = MLP(COMBAT, np.random.default_rng(5)).params
    b = MLP(COMBAT, np.random.default_rng(5)).params
    c = MLP(COMBAT, np.random.default_rng(6)).params
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["W0"], c["W0"])


def test_weights_are_not_symmetric():
    # Zero or constant init makes every hidden unit identical forever.
    W1 = MLP(CARTPOLE, np.random.default_rng(0)).params["W1"]
    assert np.unique(W1).size == W1.size


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_dtype_preserved_everywhere(dtype):
    net = MLP((5, 7, 3), np.random.default_rng(0), dtype=dtype)
    assert np.dtype(net.dtype) == np.dtype(dtype)
    assert all(v.dtype == dtype for v in net.params.values())
    # Input deliberately in the *other* precision: forward must cast to the
    # network dtype rather than letting NumPy promote silently.
    other = np.float32 if dtype == np.float64 else np.float64
    x = np.random.default_rng(1).normal(size=(4, 5)).astype(other)
    q, cache = net.forward(x)
    assert q.dtype == dtype
    grads = net.backward(cache, np.ones_like(q))
    assert all(g.dtype == dtype for g in grads.values())


def test_default_dtype_is_float64():
    assert MLP((3, 2), np.random.default_rng(0)).params["W0"].dtype == np.float64


# ----------------------------------------------------------------------
# Forward
# ----------------------------------------------------------------------

def test_forward_matches_hand_computation_and_output_is_linear():
    # Worked by hand:
    #   row 0: x = [1, 2]
    #          z0 = [1*1 + 2*2 + 0.5, 1*(-1) + 2*1 - 3] = [5.5, -2]
    #          h1 = relu(z0) = [5.5, 0]
    #          q  = 5.5*2 + 0*7 - 1 = 10
    #   row 1: x = [-1, 0]
    #          z0 = [-1 + 0.5, 1 - 3] = [-0.5, -2]  -> h1 = [0, 0]
    #          q  = -1      (negative: the output layer must NOT apply ReLU)
    net = MLP((2, 2, 1), np.random.default_rng(0))
    _set_params(net, {"W0": [[1, -1], [2, 1]], "b0": [0.5, -3],
                      "W1": [[2], [7]], "b1": [-1]})
    q, _ = net.forward(np.array([[1.0, 2.0], [-1.0, 0.0]]))
    np.testing.assert_allclose(q, [[10.0], [-1.0]])


@pytest.mark.parametrize("batch", [1, 3, 128])
def test_forward_output_shape(batch):
    net = MLP(CARTPOLE, np.random.default_rng(0))
    q, _ = net.forward(np.zeros((batch, 4)))
    assert q.shape == (batch, 2)


@pytest.mark.parametrize("x", [np.zeros(4), np.zeros((2, 5)), np.zeros((2, 4, 1))])
def test_forward_rejects_bad_input_shape(x):
    net = MLP(CARTPOLE, np.random.default_rng(0))
    with pytest.raises(ValueError):
        net.forward(x)


def test_forward_does_not_mutate_params_or_input():
    net = MLP(COMBAT, np.random.default_rng(0))
    before = {k: v.copy() for k, v in net.params.items()}
    x = np.random.default_rng(1).normal(size=(8, 18))
    x_before = x.copy()
    q, cache = net.forward(x)
    net.backward(cache, np.ones_like(q))
    assert all(np.array_equal(net.params[k], before[k]) for k in before)
    assert np.array_equal(x, x_before)


# ----------------------------------------------------------------------
# Backward: the gradient check is the core test
# ----------------------------------------------------------------------

@pytest.mark.parametrize("sizes", [(5, 7, 6, 3), (4, 9, 2), (3, 2)])
def test_backward_matches_finite_differences(sizes):
    rng = np.random.default_rng(42)
    net = MLP(sizes, rng)
    x = _off_kink_input(net, rng, batch=4)
    G = rng.normal(size=(4, sizes[-1]))   # stands in for dL/dq

    # L = sum(q * G)  =>  dL/dq = G exactly, so backward(cache, G) must equal dL/dparams.
    _, cache = net.forward(x)
    analytic = net.backward(cache, G)
    assert set(analytic) == set(net.params)

    for key in net.params:
        assert analytic[key].shape == net.params[key].shape, key
        assert_gradient_close(lambda: float(np.sum(net.forward(x)[0] * G)),
                              net.params[key], analytic[key], name=key)


def test_backward_sums_over_batch_not_mean():
    # The 1/B belongs in grad_q (build step 2), in exactly one place.  A
    # backward that also averages would make this test fail by a factor of 2.
    rng = np.random.default_rng(3)
    net = MLP((5, 7, 3), rng)
    x = _off_kink_input(net, rng, batch=2)
    G = rng.normal(size=(2, 3))
    both = net.backward(net.forward(x)[1], G)
    first = net.backward(net.forward(x[:1])[1], G[:1])
    second = net.backward(net.forward(x[1:])[1], G[1:])
    for k in both:
        np.testing.assert_allclose(both[k], first[k] + second[k], rtol=1e-12, atol=1e-12,
                                   err_msg=f"{k}: batch gradient must be the SUM of per-sample gradients")


def test_relu_derivative_at_zero_is_zero():
    # Hidden pre-activation is exactly 0: x=1, W0=1, b0=-1.  Torch's
    # convention is relu'(0) = 0, so nothing flows back through that unit.
    # A mask written as (z >= 0) gives 1 here.
    net = MLP((1, 1, 1), np.random.default_rng(0))
    _set_params(net, {"W0": [[1.0]], "b0": [-1.0], "W1": [[1.0]], "b1": [0.0]})
    q, cache = net.forward(np.array([[1.0]]))
    grads = net.backward(cache, np.ones_like(q))
    assert grads["W0"][0, 0] == 0.0 and grads["b0"][0] == 0.0
    assert grads["b1"][0] == 1.0


def test_cache_is_independent_of_later_forward_calls():
    # The point of returning the cache explicitly: a forward call between
    # forward(x1) and backward(cache1) must not change backward's result.
    # Such calls happen in a DQN loop, e.g. action selection or evaluation.
    rng = np.random.default_rng(9)
    net = MLP((5, 7, 6, 3), rng)
    x1, x2 = rng.normal(size=(4, 5)), rng.normal(size=(6, 5))
    G = rng.normal(size=(4, 3))
    expected = net.backward(net.forward(x1)[1], G)

    _, cache1 = net.forward(x1)
    net.forward(x2)
    got = net.backward(cache1, G)
    for k in expected:
        np.testing.assert_array_equal(got[k], expected[k], err_msg=k)


def test_backward_is_repeatable_and_returns_fresh_arrays():
    rng = np.random.default_rng(11)
    net = MLP((5, 7, 3), rng)
    q, cache = net.forward(rng.normal(size=(4, 5)))
    G = rng.normal(size=q.shape)
    g1 = net.backward(cache, G)
    g1_copy = {k: v.copy() for k, v in g1.items()}
    for v in g1.values():
        v[...] = 123.0   # a caller (e.g. the optimiser) scribbling on the grads
    g2 = net.backward(cache, G)
    for k in g2:
        np.testing.assert_array_equal(g2[k], g1_copy[k], err_msg=f"{k}: cache was modified")
        assert not np.shares_memory(g2[k], net.params[k]), f"{k}: gradient aliases a parameter"


def test_backward_rejects_wrong_grad_shape():
    net = MLP((5, 7, 3), np.random.default_rng(0))
    q, cache = net.forward(np.zeros((4, 5)))
    with pytest.raises(ValueError):
        net.backward(cache, np.ones((4, 2)))
    with pytest.raises(ValueError):
        net.backward(cache, np.ones((3, 3)))
