"""Build step 2: TD target and loss gradient.  Written from the agreed
interface spec before the implementation exists (dqn/README.md).

Interface under test (dqn/td.py):

    def td_target(rewards, next_q_target, terminated, gamma) -> np.ndarray
        rewards:        (B,) float
        next_q_target:  (B, n_actions), target network's output on s'
        terminated:     (B,) bool or 0/1
        returns y:      (B,), dtype of next_q_target
        y_i = r_i + gamma * (1 - terminated_i) * max_a' next_q_target[i, a']

    def td_loss_and_grad(q, actions, targets) -> tuple[float, np.ndarray]
        q:        (B, n_actions), online network's output on s
        actions:  (B,) integer, each in [0, n_actions)
        targets:  (B,) y from td_target
        returns (loss, grad_q), grad_q same shape and dtype as q
        loss = mean_i (q[i, a_i] - y_i)^2
        grad_q[i, a_i] = 2 (q[i, a_i] - y_i) / B, zero elsewhere

There is deliberately no `truncated` argument: a truncated transition is an
ordinary bootstrapped one (D029; dqn/README.md, "Terminated vs truncated").
Invalid shapes, batch-size mismatches, non-integer or out-of-range actions
raise ValueError.  Inputs are never mutated.

The oracles are hand-worked numbers and finite differences (dqn/gradcheck.py).

Skipped as a whole until dqn/td.py exists.  Once the file exists, an import
error inside it fails loudly instead of skipping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dqn.gradcheck import assert_gradient_close

if not (Path(__file__).resolve().parents[1] / "dqn" / "td.py").exists():
    pytest.skip("dqn/td.py not written yet (build step 2)", allow_module_level=True)

from dqn.network import MLP  # noqa: E402
from dqn.td import td_loss_and_grad, td_target  # noqa: E402


def _arr(values, dtype=np.float64) -> np.ndarray:
    return np.array(values, dtype=dtype)


# ----------------------------------------------------------------------
# td_target
# ----------------------------------------------------------------------

def test_target_hand_worked():
    # Row 0: non-terminal, max over s' actions is 5  -> 1 + 0.9 * 5 = 5.5
    # Row 1: non-terminal, max is 3 (action 0)       -> 0.5 + 0.9 * 3 = 3.2
    # Row 2: terminal, next_q ignored                -> -1
    y = td_target(rewards=_arr([1.0, 0.5, -1.0]),
                  next_q_target=_arr([[2.0, 5.0], [3.0, 1.0], [10.0, -4.0]]),
                  terminated=np.array([False, False, True]),
                  gamma=0.9)
    np.testing.assert_allclose(y, [5.5, 3.2, -1.0], rtol=1e-12, atol=1e-12)


def test_terminal_row_ignores_next_q_entirely():
    # s' is absorbing: its value is 0 by definition, whatever the target
    # network says about it.
    y = td_target(_arr([2.0]), _arr([[1e12, -1e12, 7.0]]), np.array([True]), gamma=0.99)
    assert y[0] == 2.0


def test_truncated_transition_bootstraps():
    # A time-limit cut is stored as terminated=False, so it bootstraps like any
    # mid-episode step.  Masking it would give y = r = 1, the wrong answer.
    # CartPole-like numbers: reward 1, s' still worth ~100.
    y = td_target(_arr([1.0]), _arr([[100.0, 99.0]]), np.array([False]), gamma=0.99)
    np.testing.assert_allclose(y, [1.0 + 0.99 * 100.0], rtol=1e-12)
    assert y[0] != 1.0, "truncated transition was masked as if terminal"


def test_target_takes_max_including_all_negative_values():
    # max over s' actions, not 0-floored and not |.|.  An accumulator
    # initialised to 0 would return 0 here.
    y = td_target(_arr([0.0]), _arr([[-3.0, -1.0, -2.0]]), np.array([False]), gamma=0.5)
    np.testing.assert_allclose(y, [-0.5], rtol=1e-12)


def test_gamma_zero_gives_reward():
    r = _arr([0.3, -2.0, 4.0])
    y = td_target(r, _arr([[5.0, 6.0], [7.0, 8.0], [9.0, 1.0]]),
                  np.array([False, True, False]), gamma=0.0)
    np.testing.assert_allclose(y, r, rtol=0, atol=0)


def test_terminated_accepts_bool_and_zero_one():
    r = _arr([1.0, 2.0])
    nq = _arr([[3.0, 4.0], [5.0, 6.0]])
    as_bool = td_target(r, nq, np.array([False, True]), gamma=0.9)
    as_float = td_target(r, nq, _arr([0.0, 1.0]), gamma=0.9)
    as_int = td_target(r, nq, np.array([0, 1]), gamma=0.9)
    np.testing.assert_array_equal(as_bool, as_float)
    np.testing.assert_array_equal(as_bool, as_int)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_target_shape_and_dtype_follow_next_q(dtype):
    y = td_target(_arr([1.0, 2.0, 3.0]), _arr(np.ones((3, 4)), dtype=dtype),
                  np.array([False, True, False]), gamma=0.99)
    assert y.shape == (3,)
    assert y.dtype == dtype


def test_target_does_not_mutate_inputs():
    r = _arr([1.0, 2.0])
    nq = _arr([[3.0, 4.0], [5.0, 6.0]])
    term = np.array([False, True])
    copies = (r.copy(), nq.copy(), term.copy())
    td_target(r, nq, term, gamma=0.9)
    for before, after in zip(copies, (r, nq, term)):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("rewards, next_q, terminated", [
    # (B, 1) rewards would broadcast against (B,) to (B, B) without crashing.
    (np.zeros((3, 1)), np.zeros((3, 2)), np.zeros(3, dtype=bool)),
    (np.zeros(3), np.zeros((3, 2)), np.zeros((3, 1), dtype=bool)),
    (np.zeros(3), np.zeros(3), np.zeros(3, dtype=bool)),            # next_q not 2-D
    (np.zeros(3), np.zeros((4, 2)), np.zeros(3, dtype=bool)),       # batch mismatch
    (np.zeros(3), np.zeros((3, 2)), np.zeros(2, dtype=bool)),       # batch mismatch
], ids=["rewards_B1", "terminated_B1", "next_q_1d", "next_q_batch", "terminated_batch"])
def test_target_rejects_bad_shapes(rewards, next_q, terminated):
    with pytest.raises(ValueError):
        td_target(rewards, next_q, terminated, gamma=0.9)


# ----------------------------------------------------------------------
# td_loss_and_grad
# ----------------------------------------------------------------------

def test_loss_single_transition_hand_worked():
    # Q(s, a=1) = 3, y = 2  ->  loss (3 - 2)^2 = 1,  dL/dq[0, 1] = 2 * 1 / 1 = 2
    loss, grad = td_loss_and_grad(_arr([[1.0, 3.0]]), np.array([1]), _arr([2.0]))
    assert loss == pytest.approx(1.0, rel=1e-12)
    np.testing.assert_allclose(grad, [[0.0, 2.0]], rtol=1e-12, atol=1e-12)


def test_loss_batch_hand_worked():
    # Taken-action Q: [1, -1, 4];  targets [3, 1, 4];  errors [-2, -2, 0]
    # loss = (4 + 4 + 0) / 3 = 8/3
    # grad at taken actions = 2 * error / 3 = [-4/3, -4/3, 0]
    q = _arr([[1.0, 2.0], [0.0, -1.0], [4.0, 4.0]])
    loss, grad = td_loss_and_grad(q, np.array([0, 1, 1]), _arr([3.0, 1.0, 4.0]))
    assert loss == pytest.approx(8.0 / 3.0, rel=1e-12)
    np.testing.assert_allclose(grad, [[-4 / 3, 0.0], [0.0, -4 / 3], [0.0, 0.0]],
                               rtol=1e-12, atol=1e-12)


def test_loss_is_python_float():
    loss, _ = td_loss_and_grad(_arr([[1.0, 2.0]]), np.array([0]), _arr([0.0]))
    assert type(loss) is float


def test_grad_is_zero_off_the_taken_action():
    # Only Q(s, a_taken) is in the loss.  A nonzero entry elsewhere means y
    # was broadcast across all actions, which pulls every action's Q toward
    # the same number and destroys the argmax.
    rng = np.random.default_rng(0)
    B, n = 16, 5
    q = rng.normal(size=(B, n))
    a = rng.integers(0, n, size=B)
    _, grad = td_loss_and_grad(q, a, rng.normal(size=B))
    off = np.ones((B, n), dtype=bool)
    off[np.arange(B), a] = False
    assert np.all(grad[off] == 0.0)
    assert np.all(grad[~off] != 0.0)


def test_grad_matches_finite_differences_wrt_q():
    rng = np.random.default_rng(1)
    B, n = 6, 4
    q = rng.normal(size=(B, n))
    a = rng.integers(0, n, size=B)
    y = rng.normal(size=B)
    _, grad = td_loss_and_grad(q, a, y)
    assert_gradient_close(lambda: td_loss_and_grad(q, a, y)[0], q, grad, name="grad_q")


def test_gradient_step_moves_q_toward_target():
    # Sign check in behavioural terms: descending the gradient reduces every
    # row's |Q(s,a) - y|.  A sign error makes Q run away from the target.
    q = _arr([[5.0, 0.0], [0.0, -3.0]])
    a = np.array([0, 1])
    y = _arr([1.0, 2.0])
    loss, grad = td_loss_and_grad(q, a, y)
    stepped = q - 0.1 * grad
    rows = np.arange(2)
    assert np.all(np.abs(stepped[rows, a] - y) < np.abs(q[rows, a] - y))
    assert td_loss_and_grad(stepped, a, y)[0] < loss


def test_loss_is_mean_not_sum():
    # Duplicating every transition: a mean leaves the loss unchanged and
    # halves each gradient entry (B doubles).  A sum doubles the loss.
    rng = np.random.default_rng(2)
    q = rng.normal(size=(5, 3))
    a = rng.integers(0, 3, size=5)
    y = rng.normal(size=5)
    loss1, grad1 = td_loss_and_grad(q, a, y)
    loss2, grad2 = td_loss_and_grad(np.vstack([q, q]), np.concatenate([a, a]),
                                    np.concatenate([y, y]))
    assert loss2 == pytest.approx(loss1, rel=1e-12)
    np.testing.assert_allclose(grad2[:5], grad1 / 2, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(grad2[5:], grad1 / 2, rtol=1e-12, atol=1e-15)


def test_rows_are_independent_of_each_other():
    # Changing one row's target changes only that row's gradient.
    q = _arr([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    a = np.array([1, 0, 1])
    _, g1 = td_loss_and_grad(q, a, _arr([0.0, 0.0, 0.0]))
    _, g2 = td_loss_and_grad(q, a, _arr([0.0, 9.0, 0.0]))
    np.testing.assert_array_equal(g1[[0, 2]], g2[[0, 2]])
    assert not np.array_equal(g1[1], g2[1])


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_grad_shape_and_dtype_follow_q(dtype):
    q = _arr(np.ones((4, 3)), dtype=dtype)
    _, grad = td_loss_and_grad(q, np.array([0, 1, 2, 0]), _arr(np.zeros(4), dtype=dtype))
    assert grad.shape == q.shape
    assert grad.dtype == dtype


def test_loss_does_not_mutate_inputs_and_returns_fresh_grad():
    q = _arr([[1.0, 2.0], [3.0, 4.0]])
    a = np.array([0, 1])
    y = _arr([0.5, 0.5])
    copies = (q.copy(), a.copy(), y.copy())
    _, g1 = td_loss_and_grad(q, a, y)
    for before, after in zip(copies, (q, a, y)):
        np.testing.assert_array_equal(before, after)
    assert not np.shares_memory(g1, q)
    _, g2 = td_loss_and_grad(q, a, y)
    assert not np.shares_memory(g1, g2)


@pytest.mark.parametrize("q, actions, targets", [
    # (B, 1) targets: q_a - y would broadcast to (B, B) and .mean() would
    # still return a scalar.  The classic silent DQN bug.
    (np.zeros((3, 2)), np.array([0, 1, 0]), np.zeros((3, 1))),
    # -1 is a valid NumPy index (the last column), so it would silently
    # train the wrong action.
    (np.zeros((3, 2)), np.array([0, -1, 0]), np.zeros(3)),
    (np.zeros((3, 2)), np.array([0, 2, 0]), np.zeros(3)),                # == n_actions
    (np.zeros((3, 2)), np.array([0.0, 1.0, 0.0]), np.zeros(3)),          # float actions
    (np.zeros((3, 2)), np.array([[0], [1], [0]]), np.zeros(3)),          # actions (B, 1)
    (np.zeros(3), np.array([0, 1, 0]), np.zeros(3)),                     # q not 2-D
    (np.zeros((3, 2)), np.array([0, 1]), np.zeros(3)),                   # batch mismatch
    (np.zeros((3, 2)), np.array([0, 1, 0]), np.zeros(4)),                # batch mismatch
], ids=["targets_B1", "action_negative", "action_too_large", "actions_float",
        "actions_B1", "q_1d", "actions_batch", "targets_batch"])
def test_loss_rejects_bad_inputs(q, actions, targets):
    with pytest.raises(ValueError):
        td_loss_and_grad(q, actions, targets)


# ----------------------------------------------------------------------
# Composition with the MLP (steps 1 + 2)
# ----------------------------------------------------------------------

def test_loss_through_mlp_matches_finite_differences():
    # L(params) = td_loss(MLP(x), a, y) with y held fixed, as it is in
    # training (y comes from the target network, which is not trained).
    # backward(cache, grad_q) must then equal dL/dparams.
    rng = np.random.default_rng(7)
    net = MLP((4, 9, 3), rng)
    B = 5
    for _ in range(100):
        x = rng.normal(size=(B, 4))
        z0 = x @ net.params["W0"] + net.params["b0"]
        if np.min(np.abs(z0)) > 1e-3:   # keep finite differences off the ReLU kink
            break
    else:
        raise RuntimeError("could not draw an input clear of the ReLU kink")
    a = rng.integers(0, 3, size=B)
    y = rng.normal(size=B)

    q, cache = net.forward(x)
    _, grad_q = td_loss_and_grad(q, a, y)
    analytic = net.backward(cache, grad_q)

    for key in net.params:
        assert_gradient_close(lambda: td_loss_and_grad(net.forward(x)[0], a, y)[0],
                              net.params[key], analytic[key], name=key)
