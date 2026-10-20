"""Build step 7a: the DQN training loop."""

from typing import Any, Callable, NamedTuple

import numpy as np

from dqn.adam import Adam
from dqn.config import DQNConfig
from dqn.exploration import epsilon_greedy, linear_epsilon
from dqn.network import MLP
from dqn.replay import Batch, ReplayBuffer
from dqn.runlog import RunLogger
from dqn.seeding import make_rng, stream_seed
from dqn.target import hard_sync, make_target, sync_due
from dqn.td import td_loss_and_grad, td_target


class TrainResult(NamedTuple):
    online: MLP          # the trained online network (step 7b checkpoints it)
    global_steps: int    # env steps taken; always cfg.total_timesteps
    episodes: int        # completed episodes (the one cut off at the end is not counted)
    updates: int         # gradient updates performed


def gradient_update(online: MLP, target: MLP, opt: Adam, batch: Batch,
                    gamma: float) -> tuple[float, float]:
    """One DQN learning step on one minibatch.

    Moves the online network's Q(s, a) for the stored actions toward the TD
    target, where the target's bootstrap value comes from the frozen target
    network.  Only the online network changes, in place, through `opt`.  The
    target network is read and never modified.

    Returns (loss, q_mean) for the update log: the minibatch TD loss, and the
    mean over the minibatch of the online network's Q-value for the action
    that was actually taken (before this step's parameter change).

    Draws no random numbers: the minibatch was already sampled by the caller.
    """
    q, cache = online.forward(batch.obs)
    next_q, _ = target.forward(batch.next_obs)
    y = td_target(batch.rewards, next_q, batch.terminated, gamma)
    loss, grad_q = td_loss_and_grad(q, batch.actions, y)
    param_grads = online.backward(cache, grad_q)
    opt.step(param_grads)
    rows = np.arange(len(batch.actions))
    q_mean = float(np.mean(q[rows, batch.actions]))
    return float(loss), q_mean


def train(cfg: DQNConfig, *, env_fn: Callable[[int | None], Any], obs_dim: int,
          n_actions: int, logger: RunLogger) -> TrainResult:
    """Train one DQN agent for cfg.total_timesteps environment steps.

    Every hyperparameter comes from `cfg`.  The environment is created only
    through `env_fn(seed)`, so this module never learns which library it came
    from.  Everything written to disk goes through `logger`.

    Randomness and reproducibility:
      * Network init, exploration and replay sampling each use their own
        stream from seeding.py.  No global RNG is touched.
      * The environment's seed comes from the "env" stream.  The env receives
        it on creation and on the first reset only.  Later resets pass no seed,
        so the env's own RNG carries on and episodes differ.
      * The same cfg must produce byte-identical episode and update logs.

    Networks and data:
      * The online network maps obs_dim inputs through cfg.hidden_sizes to
        n_actions outputs, in cfg.dtype.  The target starts as an exact copy.
      * Every env step is stored as one transition.  Its next observation is
        the one step() returned, never an observation from a reset.  The
        buffer is told whether the episode terminated, and is never told
        about truncation.

    Episodes:
      * An episode ends when the env reports terminated or truncated.  Only
        then is the env reset, and the reset observation is what the agent
        acts from next.
      * Each completed episode is logged once (see runlog.py for the fields).
        Its epsilon is the one used for its final action.  An episode still
        running when training ends is not logged.

    Learning:
      * Exploration follows the linear schedule, indexed by the env step.
      * No learning and no target syncs happen until the env step exceeds
        cfg.learning_starts.  After that, one gradient update every
        cfg.train_frequency env steps, and a hard target sync on the schedule
        from step 4.  When both fall on the same step, the update comes first.
      * Update diagnostics are logged on update steps that fall on a multiple
        of cfg.update_log_frequency.

    Step counting:
      * The loop counts env steps from 0.  Every schedule and gate above uses
        that 0-based count, as CleanRL does.
      * The global_step written to the logs is the number of env steps
        completed, so it is one more than the 0-based count of the step being
        logged (runlog.py's definition).

    Returns a TrainResult.  Evaluation and checkpointing are step 7b.
    """
    # -- setup ---------------------------------------------------------
    explore_rng = make_rng(cfg.seed, "exploration")
    env_seed = stream_seed(cfg.seed, "env")

    online = MLP((obs_dim, *cfg.hidden_sizes, n_actions),
                 make_rng(cfg.seed, "network_init"), dtype=cfg.dtype)
    target = make_target(online)
    opt = Adam(online.params, lr=cfg.learning_rate, beta1=cfg.adam_beta1,
               beta2=cfg.adam_beta2, eps=cfg.adam_eps)
    buffer = ReplayBuffer(cfg.buffer_size, obs_dim,
                          rng=make_rng(cfg.seed, "replay"), dtype=cfg.dtype)

    env = env_fn(env_seed)
    obs, _ = env.reset(seed=env_seed)      # the only seeded reset

    episode_return, episode_length = 0.0, 0
    episodes, updates = 0, 0

    # -- loop: global_step is the 0-based env step, as in CleanRL --------
    for global_step in range(cfg.total_timesteps):
        epsilon = linear_epsilon(global_step, start_e=cfg.start_e, end_e=cfg.end_e, duration=cfg.exploration_steps)

        q, _ = online.forward(obs[None])                 # (1, obs_dim) -> (1, n_actions)
        action = epsilon_greedy(q[0], epsilon, explore_rng)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        episode_return += reward
        episode_length += 1

        # s' is what step() returned, even on a truncation.  Only
        # `terminated` reaches the buffer.
        buffer.add(obs, action, reward, next_obs, terminated)

        if terminated or truncated:
            logger.log_episode(**{"return": episode_return},
                               global_step=global_step + 1, episode=episodes,
                               length=episode_length, terminated=bool(terminated),
                               truncated=bool(truncated), epsilon=epsilon)
            episodes += 1
            episode_return, episode_length = 0.0, 0
            obs, _ = env.reset()                         # no seed: env RNG carries on
        else:
            obs = next_obs

        if global_step > cfg.learning_starts and global_step % cfg.train_frequency == 0:
            loss, q_mean = gradient_update(online, target, opt,
                                           buffer.sample(cfg.batch_size), cfg.gamma)
            updates += 1
            if global_step % cfg.update_log_frequency == 0:
                logger.log_update(global_step=global_step + 1, loss=loss, q_mean=q_mean)

        # After the update, so a same-step sync copies the freshly updated net.
        if sync_due(global_step, learning_starts=cfg.learning_starts,
                    target_network_frequency=cfg.target_network_frequency):
            hard_sync(target, online)

    return TrainResult(online=online, global_steps=cfg.total_timesteps,
                       episodes=episodes, updates=updates)
    