import os
import copy
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any
import datetime
import chex
import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import LogWrapper
import hydra
from omegaconf import OmegaConf
import flashbax as fbx

from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.wrappers.baselines import (
    SMAXLogWrapper,
    MPELogWrapper,
    LogWrapper,
    CTRolloutManager,
)
from cheap_talk.src.utils.wandb_process import WandbMultiLogger
from cheap_talk.src.utils.jax_utils import pytree_norm


class ScannedRNN(nn.Module):

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        hidden_size = ins.shape[-1]
        rnn_state = jnp.where(
            resets[:, None],
            self.initialize_carry(hidden_size, *ins.shape[:-1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )


class ActorRNN(nn.Module):
    # Actor network that outputs logits for discrete actions
    action_dim: int
    hidden_dim: int
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, hidden, obs, dones, avail_actions):
        embedding = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        # Output action logits
        action_logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(embedding)

        # Mask unavailable actions
        unavail_actions = 1 - avail_actions
        action_logits = action_logits - (unavail_actions * 1e10)

        return hidden, action_logits


class HyperNetwork(nn.Module):
    """HyperNetwork for generating weights of mixing network."""

    hidden_dim: int
    output_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.output_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(x)
        return x


class MixingNetwork(nn.Module):
    """
    Mixing network for projecting individual agent Q-values into Q_tot. Follows the QMIX implementation.
    """

    embedding_dim: int
    hypernet_hidden_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, q_vals, states):
        n_agents, time_steps, batch_size = q_vals.shape
        q_vals = jnp.transpose(q_vals, (1, 2, 0))  # (time_steps, batch_size, n_agents)

        # hypernetwork
        w_1 = HyperNetwork(
            hidden_dim=self.hypernet_hidden_dim,
            output_dim=self.embedding_dim * n_agents,
            init_scale=self.init_scale,
        )(states)
        b_1 = nn.Dense(
            self.embedding_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(states)
        w_2 = HyperNetwork(
            hidden_dim=self.hypernet_hidden_dim,
            output_dim=self.embedding_dim,
            init_scale=self.init_scale,
        )(states)
        b_2 = HyperNetwork(
            hidden_dim=self.embedding_dim, output_dim=1, init_scale=self.init_scale
        )(states)

        # monotonicity and reshaping
        w_1 = w_1.reshape(time_steps, batch_size, n_agents, self.embedding_dim)
        b_1 = b_1.reshape(time_steps, batch_size, 1, self.embedding_dim)
        w_2 = w_2.reshape(time_steps, batch_size, self.embedding_dim, 1)
        b_2 = b_2.reshape(time_steps, batch_size, 1, 1)

        # mix
        hidden = nn.elu(jnp.matmul(q_vals[:, :, None, :], w_1) + b_1)
        q_tot = jnp.matmul(hidden, w_2) + b_2

        return q_tot.squeeze()  # (time_steps, batch_size)


class CriticRNN(nn.Module):
    # Critic network that takes observations and actions and outputs Q-values
    hidden_dim: int
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, hidden, obs, dones, actions):
        # Concatenate observations and actions
        obs_action = jnp.concatenate([obs, actions], axis=-1)

        embedding = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(obs_action)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        # Output individual Q-values for each agent
        q_value = nn.Dense(
            1,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(embedding)

        return hidden, jnp.squeeze(
            q_value, axis=-1
        )  # Shape: (num_agents, timesteps, batch_size)


def gumbel_softmax(rng, logits, tau=1.0, hard=True):
    """Gumbel Softmax implementation for discrete actions"""
    # Sample from Gumbel distribution
    gumbel_noise = -jnp.log(
        -jnp.log(jax.random.uniform(rng, logits.shape) + 1e-10) + 1e-10
    )
    y = (logits + gumbel_noise) / tau
    y_soft = jax.nn.softmax(y, axis=-1)

    if hard:
        # Straight-through estimator
        y_hard = jax.nn.one_hot(jnp.argmax(y_soft, axis=-1), logits.shape[-1])
        y = jax.lax.stop_gradient(y_hard - y_soft) + y_soft
    else:
        y = y_soft

    return y


@chex.dataclass(frozen=True)
class Timestep:
    obs: dict
    actions: dict
    rewards: dict
    dones: dict
    avail_actions: dict


class CustomTrainState(TrainState):
    target_network_params: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def make_train(config, env):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    # Epsilon scheduler for exploration
    eps_scheduler = optax.linear_schedule(
        init_value=config["EPS_START"],
        end_value=config["EPS_FINISH"],
        transition_steps=config["EPS_DECAY"] * config["NUM_UPDATES"],
    )

    def get_greedy_actions(action_logits, valid_actions):
        """Get greedy actions from logits"""
        unavail_actions = 1 - valid_actions
        action_logits = action_logits - (unavail_actions * 1e10)
        return jnp.argmax(action_logits, axis=-1)

    def eps_greedy_exploration(rng, action_logits, eps, valid_actions):
        """Epsilon-greedy action selection"""
        rng_a, rng_e = jax.random.split(
            rng
        )  # one key for random actions, one for epsilon

        # Get greedy actions
        greedy_actions = get_greedy_actions(action_logits, valid_actions)

        # Get random valid actions
        def get_random_actions(rng, val_action):
            return jax.random.choice(
                rng,
                jnp.arange(val_action.shape[-1]),
                p=val_action * 1.0 / jnp.sum(val_action, axis=-1),
            )

        _rngs = jax.random.split(rng_a, valid_actions.shape[0])
        random_actions = jax.vmap(get_random_actions)(_rngs, valid_actions)

        # Choose between random and greedy based on epsilon
        chosen_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape) < eps,
            random_actions,
            greedy_actions,
        )
        return chosen_actions

    def batchify(x: dict):
        return jnp.stack([x[agent] for agent in env.agents], axis=0)

    def unbatchify(x: jnp.ndarray):
        return {agent: x[i] for i, agent in enumerate(env.agents)}

    def train(rng, exp_id):
        jax.debug.print("Compile Finished. Running...")

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"])
        test_env = CTRolloutManager(
            env, batch_size=config["TEST_NUM_ENVS"]
        )  # batched env for testing (has different batch size)

        # to initalize some variables is necessary to sample a trajectory to know its strucutre
        def _env_sample_step(env_state, unused):
            rng, key_a, key_s = jax.random.split(
                jax.random.PRNGKey(0), 3
            )  # use a dummy rng here
            key_a = jax.random.split(key_a, env.num_agents)
            actions = {
                agent: wrapped_env.batch_sample(key_a[i], agent)
                for i, agent in enumerate(env.agents)
            }
            avail_actions = wrapped_env.get_valid_actions(env_state.env_state)
            obs, env_state, rewards, dones, infos = wrapped_env.batch_step(
                key_s, env_state, actions
            )
            timestep = Timestep(
                obs=obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
                avail_actions=avail_actions,
            )
            return env_state, timestep

        _, _env_state = wrapped_env.batch_reset(rng)
        _, sample_traj = jax.lax.scan(
            _env_sample_step, _env_state, None, config["NUM_STEPS"]
        )
        sample_traj_unbatched = jax.tree.map(
            lambda x: x[:, 0], sample_traj
        )  # remove the NUM_ENV dim

        # INIT NETWORK AND OPTIMIZER
        actor_network = ActorRNN(
            action_dim=wrapped_env.max_action_space,
            hidden_dim=config["HIDDEN_SIZE"],
        )

        critic_network = CriticRNN(
            hidden_dim=config["HIDDEN_SIZE"],
        )

        mixer_network = MixingNetwork(
            config["MIXER_EMBEDDING_DIM"],
            config["MIXER_HYPERNET_HIDDEN_DIM"],
            config["MIXER_INIT_SCALE"],
        )

        def create_agent(rng):
            rng, rng_actor, rng_critic, rng_mixer = jax.random.split(rng, 4)

            # Initialize actor
            init_x = (
                jnp.zeros(
                    (1, 1, wrapped_env.obs_size)
                ),  # (time_step, batch_size, obs_size)
                jnp.zeros((1, 1)),  # (time_step, batch size)
                jnp.ones((1, 1, wrapped_env.max_action_space)),  # avail_actions
            )
            init_hs = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], 1
            )  # (batch_size, hidden_dim)
            actor_params = actor_network.init(rng_actor, init_hs, *init_x)

            # Initialize critic
            critic_init_x = (
                jnp.zeros(
                    (1, 1, wrapped_env.obs_size)
                ),  # (time_step, batch_size, obs_size)
                jnp.zeros((1, 1)),  # (time_step, batch size)
                jnp.zeros((1, 1, wrapped_env.max_action_space)),  # actions (one-hot)
            )
            critic_init_hs = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], 1
            )  # (batch_size, hidden_dim)
            critic_params = critic_network.init(
                rng_critic, critic_init_hs, *critic_init_x
            )

            # Initialize mixer
            init_x = jnp.zeros((len(env.agents), 1, 1))  # q vals: agents, time, batch
            state_size = sample_traj.obs["__all__"].shape[
                -1
            ]  # get the state shape from the buffer
            init_state = jnp.zeros(
                (1, 1, state_size)
            )  # (time_step, batch_size, obs_size)
            mixer_params = mixer_network.init(rng_mixer, init_x, init_state)

            lr_scheduler = optax.linear_schedule(
                init_value=config["LR"],
                end_value=1e-10,
                transition_steps=(config["NUM_EPOCHS"]) * config["NUM_UPDATES"],
            )

            lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=lr),
            )

            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=lr),
            )

            actor_train_state = CustomTrainState.create(
                apply_fn=actor_network.apply,
                params=actor_params,
                target_network_params=actor_params,
                tx=actor_tx,
            )

            critic_train_state = CustomTrainState.create(
                apply_fn=critic_network.apply,
                params={"critic": critic_params, "mixer": mixer_params},
                target_network_params={"critic": critic_params, "mixer": mixer_params},
                tx=critic_tx,
            )

            return actor_train_state, critic_train_state

        rng, _rng = jax.random.split(rng)
        actor_train_state, critic_train_state = create_agent(rng)

        # INIT BUFFER
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config["BUFFER_SIZE"] // config["NUM_ENVS"],
            min_length_time_axis=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_batch_size=config["NUM_ENVS"],
            sample_sequence_length=1,
            period=1,
        )
        buffer_state = buffer.init(sample_traj_unbatched)

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            actor_train_state, critic_train_state, buffer_state, test_state, rng = (
                runner_state
            )

            # SAMPLE PHASE
            def _step_env(carry, _):
                actor_hs, critic_hs, last_obs, last_dones, env_state, rng = carry
                rng, rng_a, rng_s, rng_gumbel = jax.random.split(rng, 4)

                # (num_agents, 1 (dummy time), num_envs, obs_size)
                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]
                avail_actions = wrapped_env.get_valid_actions(env_state.env_state)
                _avail_actions = batchify(avail_actions)[:, np.newaxis]

                # Get action logits from actor
                new_actor_hs, action_logits = jax.vmap(
                    actor_network.apply, in_axes=(None, 0, 0, 0, 0)
                )(  # vmap across the agent dim
                    actor_train_state.params,
                    actor_hs,
                    _obs,
                    _dones,
                    _avail_actions,
                )
                action_logits = action_logits.squeeze(
                    axis=1
                )  # (num_agents, num_envs, num_actions) remove the time dim

                # Generate actions using epsilon-greedy exploration
                eps = eps_scheduler(actor_train_state.n_updates)
                _rngs = jax.random.split(rng_gumbel, env.num_agents)
                actions = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, None, 0))(
                    _rngs, action_logits, eps, batchify(avail_actions)
                )
                actions = unbatchify(actions)

                new_obs, new_env_state, rewards, dones, infos = wrapped_env.batch_step(
                    rng_s, env_state, actions
                )
                timestep = Timestep(
                    obs=last_obs,
                    actions=actions,
                    rewards=jax.tree.map(
                        lambda x: config.get("REW_SCALE", 1) * x, rewards
                    ),
                    dones=last_dones,
                    avail_actions=avail_actions,
                )
                return (new_actor_hs, critic_hs, new_obs, dones, new_env_state, rng), (
                    timestep,
                    infos,
                )

            # step the env (should be a complete rollout)
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_dones = {
                agent: jnp.zeros((config["NUM_ENVS"]), dtype=bool)
                for agent in env.agents + ["__all__"]
            }
            init_actor_hs = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], len(env.agents), config["NUM_ENVS"]
            )
            init_critic_hs = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], len(env.agents), config["NUM_ENVS"]
            )
            expl_state = (
                init_actor_hs,
                init_critic_hs,
                init_obs,
                init_dones,
                env_state,
            )
            rng, _rng = jax.random.split(rng)
            _, (timesteps, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )

            actor_train_state = actor_train_state.replace(
                timesteps=actor_train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            # BUFFER UPDATE
            buffer_traj_batch = jax.tree.map(
                lambda x: jnp.swapaxes(x, 0, 1)[
                    :, np.newaxis
                ],  # put the batch dim first and add a dummy sequence dim
                timesteps,
            )  # (num_envs, 1, time_steps, ...)
            buffer_state = buffer.add(buffer_state, buffer_traj_batch)

            # NETWORKS UPDATE
            def _learn_phase(carry, _):

                actor_train_state, critic_train_state, rng = carry
                rng, _rng = jax.random.split(rng)
                minibatch = buffer.sample(buffer_state, _rng).experience
                minibatch = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        x[:, 0], 0, 1
                    ),  # remove the dummy sequence dim (1) and swap batch and temporal dims
                    minibatch,
                )  # (max_time_steps, batch_size, ...)

                # preprocess network input
                init_actor_hs = ScannedRNN.initialize_carry(
                    config["HIDDEN_SIZE"],
                    len(env.agents),
                    config["BUFFER_BATCH_SIZE"],
                )
                init_critic_hs = ScannedRNN.initialize_carry(
                    config["HIDDEN_SIZE"],
                    len(env.agents),
                    config["BUFFER_BATCH_SIZE"],
                )

                # num_agents, timesteps, batch_size, ...
                _obs = batchify(minibatch.obs)
                _dones = batchify(minibatch.dones)
                _actions = batchify(minibatch.actions)
                _rewards = batchify(minibatch.rewards)
                _avail_actions = batchify(minibatch.avail_actions)

                # Convert discrete actions to one-hot for critic
                _actions_one_hot = jax.nn.one_hot(
                    _actions, wrapped_env.max_action_space
                )

                def _critic_loss_fn(params):
                    # Current Q-values
                    _, q_vals = jax.vmap(
                        critic_network.apply, in_axes=(None, 0, 0, 0, 0)
                    )(
                        params["critic"],
                        init_critic_hs,
                        _obs,
                        _dones,
                        _actions_one_hot,
                    )  # (num_agents, timesteps, batch_size)

                    # Mix current Q-values
                    qmix = mixer_network.apply(
                        params["mixer"], q_vals, minibatch.obs["__all__"]
                    )[
                        :-1
                    ]  # Remove last timestep

                    # Target Q-values
                    # Get next actions from target actor
                    _, next_action_logits = jax.vmap(
                        actor_network.apply, in_axes=(None, 0, 0, 0, 0)
                    )(
                        actor_train_state.target_network_params,
                        init_actor_hs,
                        _obs,
                        _dones,
                        _avail_actions,
                    )

                    # Convert to one-hot using hard Gumbel softmax (deterministic target actions)
                    rng_target = jax.random.PRNGKey(
                        42
                    )  # Fixed seed for deterministic targets
                    _rngs_target = jax.random.split(rng_target, len(env.agents))
                    target_actions_one_hot = jax.vmap(
                        lambda rng, logits: gumbel_softmax(
                            rng, logits, tau=0.1, hard=True
                        ),
                        in_axes=(0, 0),
                    )(_rngs_target, next_action_logits)

                    # Target Q-values
                    _, target_q_next = jax.vmap(
                        critic_network.apply, in_axes=(None, 0, 0, 0, 0)
                    )(
                        critic_train_state.target_network_params["critic"],
                        init_critic_hs,
                        _obs,
                        _dones,
                        target_actions_one_hot,
                    )

                    # Mix target Q-values
                    qmix_next = mixer_network.apply(
                        critic_train_state.target_network_params["mixer"],
                        target_q_next,
                        minibatch.obs["__all__"],
                    )

                    # Compute target values
                    qmix_target = minibatch.rewards["__all__"][:-1] + config[
                        "GAMMA"
                    ] * qmix_next[1:] * (1 - minibatch.dones["__all__"][:-1])

                    # MSE loss
                    loss = jnp.mean((qmix - jax.lax.stop_gradient(qmix_target)) ** 2)
                    return loss, q_vals.mean()

                def _actor_loss_fn(actor_params):
                    # Get action logits from current actor
                    _, action_logits = jax.vmap(
                        actor_network.apply, in_axes=(None, 0, 0, 0, 0)
                    )(
                        actor_params,
                        init_actor_hs,
                        _obs,
                        _dones,
                        _avail_actions,
                    )

                    # Convert to one-hot using soft Gumbel softmax for differentiability
                    rng_actor = jax.random.PRNGKey(123)
                    _rngs_actor = jax.random.split(rng_actor, len(env.agents))
                    actions_soft = jax.vmap(
                        lambda rng, logits: gumbel_softmax(
                            rng, logits, tau=config.get("GUMBEL_TAU", 1.0), hard=True
                        ),
                        in_axes=(0, 0),
                    )(_rngs_actor, action_logits)

                    # Get individual Q-values from critic
                    _, q_vals = jax.vmap(
                        critic_network.apply, in_axes=(None, 0, 0, 0, 0)
                    )(
                        critic_train_state.params["critic"],
                        init_critic_hs,
                        _obs,
                        _dones,
                        actions_soft,
                    )

                    # Mix Q-values using mixer
                    qmix = mixer_network.apply(
                        critic_train_state.params["mixer"],
                        q_vals,
                        minibatch.obs["__all__"],
                    )

                    # Maximize mixed Q-value (negative for minimization)
                    loss = -jnp.mean(qmix)
                    return loss

                # Update critic
                (critic_loss, critic_q_vals), critic_grads = jax.value_and_grad(
                    _critic_loss_fn, has_aux=True
                )(critic_train_state.params)
                critic_train_state = critic_train_state.apply_gradients(
                    grads=critic_grads
                )

                # Update actor
                actor_loss, actor_grads = jax.value_and_grad(
                    _actor_loss_fn, has_aux=False
                )(actor_train_state.params)
                actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)

                # Update grad steps
                actor_train_state = actor_train_state.replace(
                    grad_steps=actor_train_state.grad_steps + 1,
                )
                critic_train_state = critic_train_state.replace(
                    grad_steps=critic_train_state.grad_steps + 1,
                )

                return (actor_train_state, critic_train_state, rng), (
                    actor_loss,
                    critic_loss,
                )

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                buffer.can_sample(buffer_state)
            ) & (  # enough experience in buffer
                actor_train_state.timesteps > config["LEARNING_STARTS"]
            )
            (actor_train_state, critic_train_state, rng), (actor_loss, critic_loss) = (
                jax.lax.cond(
                    is_learn_time,
                    lambda actor_ts, critic_ts, rng: jax.lax.scan(
                        _learn_phase,
                        (actor_ts, critic_ts, rng),
                        None,
                        config["NUM_EPOCHS"],
                    ),
                    lambda actor_ts, critic_ts, rng: (
                        (actor_ts, critic_ts, rng),
                        (
                            jnp.zeros(config["NUM_EPOCHS"]),
                            jnp.zeros(config["NUM_EPOCHS"]),
                        ),
                    ),  # do nothing
                    actor_train_state,
                    critic_train_state,
                    _rng,
                )
            )

            # update target networks
            actor_train_state = jax.lax.cond(
                actor_train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda ts: ts.replace(
                    target_network_params=optax.incremental_update(
                        ts.params,
                        ts.target_network_params,
                        config["TAU"],
                    )
                ),
                lambda ts: ts,
                operand=actor_train_state,
            )

            critic_train_state = jax.lax.cond(
                critic_train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda ts: ts.replace(
                    target_network_params=optax.incremental_update(
                        ts.params,
                        ts.target_network_params,
                        config["TAU"],
                    )
                ),
                lambda ts: ts,
                operand=critic_train_state,
            )

            # UPDATE METRICS
            actor_train_state = actor_train_state.replace(
                n_updates=actor_train_state.n_updates + 1
            )
            critic_train_state = critic_train_state.replace(
                n_updates=critic_train_state.n_updates + 1
            )
            metrics = {
                "env_step": actor_train_state.timesteps,
                "update_steps": actor_train_state.n_updates,
                "grad_steps": actor_train_state.grad_steps,
                "actor_loss": actor_loss.mean(),
                "critic_loss": critic_loss.mean(),
            }
            metrics.update(jax.tree.map(lambda x: x.mean(), infos))

            # update the test metrics
            if config.get("TEST_DURING_TRAINING", True):
                rng, _rng = jax.random.split(rng)
                test_state = jax.lax.cond(
                    actor_train_state.n_updates
                    % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                    == 0,
                    lambda _: get_greedy_metrics(_rng, actor_train_state),
                    lambda _: test_state,
                    operand=None,
                )
                metrics.update({"test_" + k: v for k, v in test_state.items()})

            def callback(exp_id, metrics):
                np_log_dict = {k: np.array(v) for k, v in metrics.items()}
                LOGGER.log(int(exp_id), np_log_dict)

            jax.experimental.io_callback(callback, None, exp_id, metrics)

            runner_state = (
                actor_train_state,
                critic_train_state,
                buffer_state,
                test_state,
                rng,
            )

            return runner_state, None

        def get_greedy_metrics(rng, actor_train_state):
            """Help function to test greedy policy during training"""
            if not config.get("TEST_DURING_TRAINING", True):
                return None

            params = actor_train_state.params

            def _greedy_env_step(step_state, unused):
                params, env_state, last_obs, last_dones, actor_hstate, rng = step_state
                rng, key_s = jax.random.split(rng)
                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]
                avail_actions = test_env.get_valid_actions(env_state.env_state)
                _avail_actions = batchify(avail_actions)[:, np.newaxis]

                actor_hstate, action_logits = jax.vmap(
                    actor_network.apply, in_axes=(None, 0, 0, 0, 0)
                )(
                    params,
                    actor_hstate,
                    _obs,
                    _dones,
                    _avail_actions,
                )
                action_logits = action_logits.squeeze(axis=1)
                actions = get_greedy_actions(action_logits, batchify(avail_actions))
                actions = unbatchify(actions)
                obs, env_state, rewards, dones, infos = test_env.batch_step(
                    key_s, env_state, actions
                )
                step_state = (params, env_state, obs, dones, actor_hstate, rng)
                return step_state, (rewards, dones, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            init_dones = {
                agent: jnp.zeros((config["TEST_NUM_ENVS"]), dtype=bool)
                for agent in env.agents + ["__all__"]
            }
            rng, _rng = jax.random.split(rng)
            actor_hstate = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], len(env.agents), config["TEST_NUM_ENVS"]
            )  # (n_agents*n_envs, hs_size)
            step_state = (
                params,
                env_state,
                init_obs,
                init_dones,
                actor_hstate,
                _rng,
            )
            step_state, (rewards, dones, infos) = jax.lax.scan(
                _greedy_env_step, step_state, None, config["TEST_NUM_STEPS"]
            )
            metrics = jax.tree.map(
                lambda x: jnp.nanmean(
                    jnp.where(
                        infos["returned_episode"],
                        x,
                        jnp.nan,
                    )
                ),
                infos,
            )
            return metrics

        rng, _rng = jax.random.split(rng)
        test_state = get_greedy_metrics(_rng, actor_train_state)

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (
            actor_train_state,
            critic_train_state,
            buffer_state,
            test_state,
            _rng,
        )

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train


def env_from_config(config):
    env_name = config["ENV_NAME"]
    # smax init neeeds a scenario
    if "smax" in env_name.lower():
        config["ENV_KWARGS"]["scenario"] = map_name_to_scenario(config["MAP_NAME"])
        env_name = f"{config['ENV_NAME']}_{config['MAP_NAME']}"
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = SMAXLogWrapper(env)

    return env, env_name


@hydra.main(version_base=None, config_path="./", config_name="config")
def main(config):
    try:
        config = OmegaConf.to_container(config)
        alg_name = config.get("ALG_NAME", "facmac")
        env, env_name = env_from_config(copy.deepcopy(config))

        # WANDB
        job_type = f"FACMAC_{config['MAP_NAME']}"
        group = f"FACMAC_{config['MAP_NAME']}"
        if config["USE_TIMESTAMP"]:
            group += datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
        global LOGGER
        LOGGER = WandbMultiLogger(
            project=config["PROJECT"],
            group=group,
            job_type=job_type,
            config=config,
            mode=config["WANDB_MODE"],
            seed=config["SEED"],
            num_seeds=config["NUM_SEEDS"],
        )

        rng = jax.random.PRNGKey(config["SEED"])
        rng_seeds = jax.random.split(rng, config["NUM_SEEDS"])
        exp_ids = jnp.arange(config["NUM_SEEDS"])

        print("Starting compile...")
        train_vmap = jax.vmap(make_train(config, env))
        train_vjit = jax.jit(train_vmap)
        outs = jax.block_until_ready(train_vjit(rng_seeds, exp_ids))

    finally:
        LOGGER.finish()
        print("Finished training.")


if __name__ == "__main__":
    main()
