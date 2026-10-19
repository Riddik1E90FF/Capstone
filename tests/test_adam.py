"""Build step 3: Adam.  Written from the agreed interface spec before the
implementation exists (dqn/README.md).

Interface under test (dqn/adam.py):

    class Adam:
        params: dict[str, np.ndarray]   # the caller's dict itself, not a copy
        lr, beta1, beta2, eps: float
        t: int                          # completed steps; 0 before the first
        m: dict[str, np.ndarray]        # raw first moment,  zeros_like each param
        v: dict[str, np.ndarray]        # raw second moment, zeros_like each param

        def __init__(self, params, *, lr, beta1, beta2, eps): ...
        def step(self, grads: dict[str, np.ndarray]) -> None: ...

    One step, elementwise, for every key k (Kingma & Ba 2015, Algorithm 1;
    torch.optim.Adam's default path):
        t      <- t + 1
        m      <- beta1 * m + (1 - beta1) * g
        v      <- beta2 * v + (1 - beta2) * g^2
        m_hat   = m / (1 - beta1^t)
        v_hat   = v / (1 - beta2^t)
        param  <- param - lr * m_hat / (sqrt(v_hat) + eps)

    m and v store the raw (uncorrected) averages.  The bias correction is
    applied to a temporary each step and never written back.  Parameters are
    updated in place: after a step, `params[k]` is the same array object.

    The hyperparameters are keyword-only with no defaults, so the only place
    their values live is DQNConfig.

    Invalid hyperparameters raise ValueError at construction.  A grads dict
    with missing or extra keys, a wrongly shaped array, or a non-finite value
    raises ValueError, and leaves params, m, v and t exactly as they were.
    The grads dict and its arrays are never mutated.

The oracles are hand-worked first steps and torch.optim.Adam fed identical
gradients.  torch is dev-only (D029).  This file imports it; dqn/ never does.
The torch tests skip if torch is not installed; Gate 1a L1 needs them to run.

Skipped as a whole until dqn/adam.py exists.  Once the file exists, an import
error inside it fails loudly instead of skipping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

if not (Path(__file__).resolve().parents[1] / "dqn" / "adam.py").exists():
    pytest.skip("dqn/adam.py not written yet (build step 3)", allow_module_level=True)

from dqn.adam import Adam  # noqa: E402
from dqn.config import DQNConfig  # noqa: E402
from dqn.network import MLP  # noqa: E402
from dqn.td import td_loss_and_grad  # noqa: E402

CFG = DQNConfig()
HYPER = dict(lr=CFG.learning_rate, beta1=CFG.adam_beta1, beta2=CFG.adam_beta2, eps=CFG.adam_eps)
CARTPOLE = (4, 120, 84, 2)


def _arr(values, dtype=np.float64) -> np.ndarray:
    return np.array(values, dtype=dtype)


def _scalar_opt(theta0: float = 0.0, *, lr: float = 0.1, eps: float = 1e-8):
    params = {"w": _arr([theta0])}
    return params, Adam(params, lr=lr, beta1=0.9, beta2=0.999, eps=eps)


def _snapshot(opt: Adam) -> dict:
    return {
        "params": {k: a.copy() for k, a in opt.params.items()},
        "m": {k: a.copy() for k, a in opt.m.items()},
        "v": {k: a.copy() for k, a in opt.v.items()},
        "t": opt.t,
    }


def _assert_unchanged(opt: Adam, snap: dict) -> None:
    assert opt.t == snap["t"]
    for field in ("params", "m", "v"):
        for k, before in snap[field].items():
            np.testing.assert_array_equal(getattr(opt, field)[k], before, err_msg=f"{field}[{k}]")


# ----------------------------------------------------------------------
# Construction and state layout
# ----------------------------------------------------------------------

def test_initial_state():
    net = MLP(CARTPOLE, np.random.default_rng(0))
    opt = Adam(net.params, **HYPER)
    assert opt.t == 0
    assert opt.params is net.params, "must hold the caller's dict, not a copy"
    assert (opt.lr, opt.beta1, opt.beta2, opt.eps) == (2.5e-4, 0.9, 0.999, 1e-8)
    for field in (opt.m, opt.v):
        assert list(field) == list(net.params), "same keys, same order as params"
        for k, a in field.items():
            assert a.shape == net.params[k].shape
            assert a.dtype == net.params[k].dtype
            assert np.all(a == 0.0)


def test_state_arrays_are_all_distinct():
    # dict.fromkeys(params, np.zeros(...)) or m = v = {...} would make every
    # key share one buffer, so updating W0's moments would overwrite b0's.
    net = MLP((3, 4, 2), np.random.default_rng(0))
    opt = Adam(net.params, **HYPER)
    arrays = [*opt.m.values(), *opt.v.values(), *net.params.values()]
    for i, a in enumerate(arrays):
        for b in arrays[i + 1:]:
            assert not np.shares_memory(a, b)


def test_hyperparameters_are_keyword_only_without_defaults():
    params = {"w": _arr([1.0])}
    with pytest.raises(TypeError):
        Adam(params)                                    # no silent defaults
    with pytest.raises(TypeError):
        Adam(params, 2.5e-4, 0.9, 0.999, 1e-8)          # positional order is easy to swap


@pytest.mark.parametrize("override", [
    {"lr": 0.0}, {"lr": -1e-3},
    {"beta1": 1.0}, {"beta1": -0.1},
    {"beta2": 1.0}, {"beta2": -0.1},
    {"eps": 0.0}, {"eps": -1e-8},
], ids=lambda d: "-".join(f"{k}={v}" for k, v in d.items()))
def test_rejects_bad_hyperparameters(override):
    # beta = 1 makes the bias correction 1 - 1^t = 0, a division by zero.
    with pytest.raises(ValueError):
        Adam({"w": _arr([1.0])}, **{**HYPER, **override})


def test_rejects_empty_params():
    with pytest.raises(ValueError):
        Adam({}, **HYPER)


# ----------------------------------------------------------------------
# Hand-worked steps.  lr = 0.1 so the numbers are readable.
# ----------------------------------------------------------------------

def test_first_step_moves_each_element_by_about_lr():
    # t = 1:  m = 0.1 g,  v = 0.001 g^2,  m_hat = g,  v_hat = g^2
    #   update = lr * g / (|g| + eps)  ~=  lr * sign(g), whatever |g| is.
    # Without bias correction it would be lr * 0.1 g / (sqrt(0.001) |g|)
    # ~= 3.16 lr; correcting only m gives ~31.6 lr, only v gives 0.1 lr.
    g = _arr([3.0, -0.002, 50.0])
    params = {"w": np.zeros(3)}
    opt = Adam(params, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)
    opt.step({"w": g})
    assert opt.t == 1
    np.testing.assert_allclose(opt.m["w"], 0.1 * g, rtol=1e-12)
    np.testing.assert_allclose(opt.v["w"], 0.001 * g ** 2, rtol=1e-12)
    np.testing.assert_allclose(params["w"], -0.1 * g / (np.abs(g) + 1e-8), rtol=1e-12)
    np.testing.assert_allclose(params["w"], [-0.1, 0.1, -0.1], rtol=1e-5)


def test_second_step_alternating_gradient():
    # g1 = +1:  theta1 = -lr / (1 + eps)
    # g2 = -1:  m2 = 0.9 * 0.1 - 0.1 = -0.01,         1 - 0.9^2   = 0.19   -> m_hat = -1/19
    #           v2 = 0.999 * 0.001 + 0.001 = 0.001999, 1 - 0.999^2 = 0.001999 -> v_hat = 1
    #           theta2 = theta1 + lr / (19 (1 + eps)) = -(18/19) lr / (1 + eps)
    # Momentum is why step 2 only takes back 1/19 of step 1 despite the sign flip.
    params, opt = _scalar_opt()
    opt.step({"w": _arr([1.0])})
    np.testing.assert_allclose(params["w"], [-0.1 / (1 + 1e-8)], rtol=1e-12)
    opt.step({"w": _arr([-1.0])})
    assert opt.t == 2
    np.testing.assert_allclose(opt.m["w"], [-0.01], rtol=1e-12)
    np.testing.assert_allclose(opt.v["w"], [0.001999], rtol=1e-12)
    np.testing.assert_allclose(params["w"], [-(18 / 19) * 0.1 / (1 + 1e-8)], rtol=1e-12)


def test_stored_moments_are_raw_not_bias_corrected():
    # Writing m_hat back into m makes the next step's average start from an
    # inflated value.  After two steps of g = 1 the raw m is 0.19, not 1.
    params, opt = _scalar_opt()
    opt.step({"w": _arr([1.0])})
    opt.step({"w": _arr([1.0])})
    np.testing.assert_allclose(opt.m["w"], [0.19], rtol=1e-12)
    np.testing.assert_allclose(opt.v["w"], [0.001999], rtol=1e-12)


def test_constant_gradient_gives_constant_step_at_every_t():
    # For a constant g, m_t = (1 - beta1^t) g and v_t = (1 - beta2^t) g^2
    # exactly, so m_hat = g and v_hat = g^2 at every t.  Each step is then
    # lr * g / (|g| + eps).  An off-by-one in t (beta^(t+1), or t incremented
    # after use) breaks this at every step, not only the first.
    g = _arr([2.0, -0.5])
    params = {"w": np.zeros(2)}
    opt = Adam(params, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)
    expected_step = -0.1 * g / (np.abs(g) + 1e-8)
    for t in range(1, 101):
        before = params["w"].copy()
        opt.step({"w": g.copy()})
        assert opt.t == t
        np.testing.assert_allclose(params["w"] - before, expected_step, rtol=1e-9,
                                   err_msg=f"step {t}")


def test_epsilon_is_added_to_sqrt_v_hat():
    # g = eps = 1e-8, first step: sqrt(v_hat) = |g| = 1e-8, so
    #   update = lr * 1e-8 / (1e-8 + 1e-8) = lr / 2.
    # Other placements seen in the wild give different numbers here:
    #   inside the root, sqrt(v_hat + eps):            ~ 1e-4 lr
    #   before bias correction, sqrt(v) + eps (the TF
    #   "efficient" form in Kingma & Ba section 2):    ~ 0.031 lr
    # Only the first matches torch, and so CleanRL (D029).
    params, opt = _scalar_opt(lr=0.1, eps=1e-8)
    opt.step({"w": _arr([1e-8])})
    np.testing.assert_allclose(params["w"], [-0.05], rtol=1e-9)


def test_zero_gradient_on_first_step_is_a_no_op_without_nan():
    # m = v = 0, so the update is lr * 0 / (0 + eps) = 0.  This is the case
    # eps exists for.  errstate turns any 0/0 or sqrt of a negative into an
    # error instead of a silent nan.
    params, opt = _scalar_opt(theta0=0.7)
    with np.errstate(all="raise"):
        opt.step({"w": _arr([0.0])})
    assert opt.t == 1
    assert params["w"][0] == 0.7
    assert opt.m["w"][0] == 0.0 and opt.v["w"][0] == 0.0


def test_momentum_keeps_moving_after_gradient_goes_to_zero():
    # g1 = 1, g2 = 0:  m2 = 0.09, m_hat = 0.09 / 0.19 = 9/19
    #                  v2 = 0.000999, v_hat = 0.000999 / 0.001999
    # The parameter keeps moving downhill on a zero gradient.  In the DQN this
    # is a weight whose gradient goes to zero because its ReLU unit stopped
    # firing for the current batch.
    params, opt = _scalar_opt()
    opt.step({"w": _arr([1.0])})
    theta1 = params["w"].copy()
    opt.step({"w": _arr([0.0])})
    expected = -0.1 * (9 / 19) / (np.sqrt(0.000999 / 0.001999) + 1e-8)
    np.testing.assert_allclose(params["w"] - theta1, [expected], rtol=1e-9)
    assert params["w"][0] < theta1[0]


def test_positive_gradient_decreases_parameter():
    # Descent, not ascent.
    params, opt = _scalar_opt(theta0=1.0)
    opt.step({"w": _arr([0.3])})
    assert params["w"][0] < 1.0


def test_elements_are_independent():
    # Adam is elementwise.  Changing one element's gradient history must not
    # change any other element's trajectory.  A global norm or a v shared
    # across elements fails here.
    rng = np.random.default_rng(3)
    grads = rng.normal(size=(10, 6))
    altered = grads.copy()
    altered[:, 2] *= 100.0

    def run(gs):
        params = {"w": np.zeros(6)}
        opt = Adam(params, **HYPER)
        for g in gs:
            opt.step({"w": g.copy()})
        return params["w"]

    a, b = run(grads), run(altered)
    keep = np.arange(6) != 2
    np.testing.assert_array_equal(a[keep], b[keep])
    assert a[2] != b[2]


def test_invariant_to_gradient_scale():
    # Multiplying every gradient by k > 0 multiplies m by k and sqrt(v) by k,
    # so the step is unchanged (up to eps).  This is why one lr works across
    # layers whose gradients differ by orders of magnitude, and why Adam is
    # forgiving of the TD loss's scale.
    rng = np.random.default_rng(4)
    signs = rng.choice([-1.0, 1.0], size=(20, 5))
    grads = signs * rng.uniform(0.5, 2.0, size=(20, 5))   # |g| >> eps

    def run(scale):
        params = {"w": np.zeros(5)}
        opt = Adam(params, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)
        for g in grads:
            opt.step({"w": scale * g})
        return params["w"]

    np.testing.assert_allclose(run(1000.0), run(1.0), rtol=1e-7)


# ----------------------------------------------------------------------
# In-place update, aliasing, no mutation of inputs
# ----------------------------------------------------------------------

def test_updates_params_in_place_through_the_callers_dict():
    # The network sees the update through its own net.params, and each array
    # keeps its identity.  In-place matters for step 4: an online/target
    # aliasing bug only shows up if the online arrays are written in place.
    net = MLP((3, 4, 2), np.random.default_rng(0))
    ids = {k: id(a) for k, a in net.params.items()}
    before = {k: a.copy() for k, a in net.params.items()}
    opt = Adam(net.params, **HYPER)
    rng = np.random.default_rng(1)
    opt.step({k: rng.normal(size=a.shape) for k, a in net.params.items()})
    for k, a in net.params.items():
        assert id(a) == ids[k], f"{k} was rebound, not updated in place"
        assert not np.array_equal(a, before[k]), f"{k} did not change"


def test_does_not_mutate_or_keep_grads():
    params = {"w": np.zeros(3), "b": np.zeros(2)}
    opt = Adam(params, **HYPER)
    grads = {"w": _arr([1.0, -2.0, 3.0]), "b": _arr([0.5, 0.0])}
    copies = {k: g.copy() for k, g in grads.items()}
    opt.step(grads)
    for k in grads:
        np.testing.assert_array_equal(grads[k], copies[k])
        for state in (opt.m[k], opt.v[k], params[k]):
            assert not np.shares_memory(state, grads[k])


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_dtype_preserved(dtype):
    net = MLP((3, 4, 2), np.random.default_rng(0), dtype=dtype)
    opt = Adam(net.params, **HYPER)
    rng = np.random.default_rng(1)
    for _ in range(3):
        opt.step({k: rng.normal(size=a.shape).astype(dtype) for k, a in net.params.items()})
    for k in net.params:
        assert net.params[k].dtype == dtype
        assert opt.m[k].dtype == dtype
        assert opt.v[k].dtype == dtype


# ----------------------------------------------------------------------
# Input validation on step()
# ----------------------------------------------------------------------

def _two_param_opt():
    params = {"W": np.zeros((3, 2)), "b": np.zeros(2)}
    return params, Adam(params, **HYPER)


@pytest.mark.parametrize("grads", [
    {"W": np.ones((3, 2))},                                          # b missing: never trained
    {"W": np.ones((3, 2)), "b": np.ones(2), "c": np.ones(1)},        # unknown key (typo)
    {"W": np.ones((2, 3)), "b": np.ones(2)},                         # transposed
    {"W": np.ones((3, 2)), "b": np.ones((1, 2))},                    # (1, n) would broadcast
    {"W": np.ones((3, 2)), "b": np.float64(1.0)},                    # scalar would broadcast
    {"W": np.ones((3, 2)), "b": _arr([1.0, np.nan])},                # nan poisons v for good
    {"W": np.ones((3, 2)), "b": _arr([np.inf, 1.0])},
], ids=["missing_key", "extra_key", "transposed", "b_1n", "b_scalar", "nan", "inf"])
def test_step_rejects_bad_grads(grads):
    _, opt = _two_param_opt()
    with pytest.raises(ValueError):
        opt.step(grads)


@pytest.mark.parametrize("bad_b", [np.ones(3), _arr([1.0, np.nan])], ids=["shape", "nan"])
def test_rejected_step_changes_nothing(bad_b):
    # Validate everything before touching anything.  Otherwise a bad entry
    # in "b" is found only after "W" has already moved and t has advanced.
    _, opt = _two_param_opt()
    opt.step({"W": np.ones((3, 2)), "b": np.ones(2)})
    snap = _snapshot(opt)
    with pytest.raises(ValueError):
        opt.step({"W": 5.0 * np.ones((3, 2)), "b": bad_b})
    _assert_unchanged(opt, snap)


# ----------------------------------------------------------------------
# Oracle: torch.optim.Adam on identical gradients (dev-only, D029)
# ----------------------------------------------------------------------

@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def _torch_twin(torch, params: dict, hyper: dict):
    tparams = {k: torch.tensor(a.copy(), dtype=torch.float64, requires_grad=True)
               for k, a in params.items()}
    # foreach=False selects the plain single-tensor reference path.
    topt = torch.optim.Adam(list(tparams.values()), lr=hyper["lr"],
                            betas=(hyper["beta1"], hyper["beta2"]), eps=hyper["eps"],
                            weight_decay=0, amsgrad=False, maximize=False, foreach=False)
    return tparams, topt


def _torch_step(torch, tparams, topt, grads: dict) -> None:
    for k, p in tparams.items():
        p.grad = torch.tensor(grads[k].copy(), dtype=torch.float64)
    topt.step()


def _assert_close_scaled(ours, theirs, tol: float, label: str) -> None:
    # Error measured against the array's largest entry, not element by
    # element.  torch updates m with lerp, m + (1 - beta1)(g - m), which rounds
    # differently from beta1 m + (1 - beta1) g.  Where m crosses zero, a 1e-15
    # absolute difference is a huge relative one without being a bug.
    scale = float(np.max(np.abs(theirs)))
    np.testing.assert_allclose(ours, theirs, rtol=0, atol=tol * scale, err_msg=label)


def _assert_matches_torch(opt: Adam, before: dict, tparams, topt, label: str) -> None:
    for k, p in tparams.items():
        tstate = topt.state[p]
        assert opt.t == int(tstate["step"]), f"{label}: t"
        theirs = p.detach().numpy()
        _assert_close_scaled(opt.m[k], tstate["exp_avg"].numpy(), 1e-12, f"{label}: m[{k}]")
        _assert_close_scaled(opt.v[k], tstate["exp_avg_sq"].numpy(), 1e-12, f"{label}: v[{k}]")
        # The step itself, not only the parameter.  At lr = 2.5e-4 a wrong
        # step can hide inside the parameter's tolerance.
        _assert_close_scaled(opt.params[k] - before[k], theirs - before[k], 1e-9,
                             f"{label}: step on {k}")
        _assert_close_scaled(opt.params[k], theirs, 1e-12, f"{label}: {k}")


@pytest.mark.parametrize("hyper", [
    HYPER,
    dict(lr=1e-2, beta1=0.5, beta2=0.9, eps=1e-6),   # betas and eps not hard-coded
], ids=["config_defaults", "other_hyper"])
def test_matches_torch_on_random_gradients(torch, hyper):
    rng = np.random.default_rng(10)
    net = MLP(CARTPOLE, rng)
    opt = Adam(net.params, **hyper)
    tparams, topt = _torch_twin(torch, net.params, hyper)
    for step in range(1, 201):
        # Gradient scale drifts over four orders of magnitude, as TD-loss
        # gradients do when Q-values grow.
        scale = 10.0 ** rng.uniform(-3, 1)
        grads = {k: scale * rng.normal(size=a.shape) for k, a in net.params.items()}
        before = {k: a.copy() for k, a in net.params.items()}
        opt.step(grads)
        _torch_step(torch, tparams, topt, grads)
        _assert_matches_torch(opt, before, tparams, topt, f"step {step}")


def test_matches_torch_with_sparse_and_zero_gradients(torch):
    # Exact zeros are common in the DQN: a dead ReLU unit gives zero gradient
    # to its whole column of W, and only the taken action's output row gets
    # gradient.  Includes elements that are zero from step 1 and whole steps
    # of zero gradient.
    rng = np.random.default_rng(11)
    params = {"W": rng.normal(size=(6, 5)), "b": rng.normal(size=5)}
    opt = Adam(params, **HYPER)
    tparams, topt = _torch_twin(torch, params, HYPER)
    never = rng.random((6, 5)) < 0.2
    for step in range(1, 101):
        g_w = rng.normal(size=(6, 5)) * (rng.random((6, 5)) < 0.5)
        g_w[never] = 0.0
        g_b = np.zeros(5) if step % 7 == 0 else rng.normal(size=5)
        grads = {"W": g_w, "b": g_b}
        before = {k: a.copy() for k, a in params.items()}
        opt.step(grads)
        _torch_step(torch, tparams, topt, grads)
        _assert_matches_torch(opt, before, tparams, topt, f"step {step}")
    assert np.all(params["W"][never] == tparams["W"].detach().numpy()[never])


def test_matches_torch_on_real_td_gradients(torch):
    # The composition actually used in training: MLP.backward of the TD loss
    # gradient, fed to both optimisers.  Gradients are computed from our
    # network; torch receives a copy of the same gradients.
    rng = np.random.default_rng(12)
    net = MLP(CARTPOLE, rng)
    opt = Adam(net.params, **HYPER)
    tparams, topt = _torch_twin(torch, net.params, HYPER)
    B = CFG.batch_size
    for step in range(1, 51):
        x = rng.normal(size=(B, CARTPOLE[0]))
        a = rng.integers(0, CARTPOLE[-1], size=B)
        y = rng.normal(loc=1.0, size=B)
        q, cache = net.forward(x)
        _, grad_q = td_loss_and_grad(q, a, y)
        grads = net.backward(cache, grad_q)
        before = {k: v.copy() for k, v in net.params.items()}
        opt.step(grads)
        _torch_step(torch, tparams, topt, grads)
        _assert_matches_torch(opt, before, tparams, topt, f"update {step}")


def test_reduces_td_loss_on_a_fixed_batch():
    # Behavioural sanity: repeated steps on one fixed batch drive the loss
    # down.  Not an oracle, but it fails loudly on a sign error even if every
    # other test were somehow miswired.
    rng = np.random.default_rng(13)
    net = MLP((4, 16, 2), rng)
    opt = Adam(net.params, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8)
    x = rng.normal(size=(32, 4))
    a = rng.integers(0, 2, size=32)
    y = rng.normal(size=32)
    first = None
    for _ in range(300):
        q, cache = net.forward(x)
        loss, grad_q = td_loss_and_grad(q, a, y)
        first = loss if first is None else first
        opt.step(net.backward(cache, grad_q))
    assert loss < 0.2 * first
