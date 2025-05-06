"""
Based on PureJaxRL Implementation of IPPO, with changes to give a centralised critic.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Tuple, Union, Dict
import datetime
from flax.training.train_state import TrainState
import distrax
from jax._src.typing import Array
import hydra
from omegaconf import DictConfig, OmegaConf
from functools import partial
import jaxmarl
from jaxmarl.wrappers.baselines import MPELogWrapper, JaxMARLWrapper
from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State
from cheap_talk.src.jax_experiments.utils.wandb_process import WandbMultiLogger
from cheap_talk.src.jax_experiments.utils.jax_utils import pytree_norm

import wandb
import functools
import matplotlib.pyplot as plt


class MPEWorldStateWrapper(JaxMARLWrapper):

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state(obs)
        return obs, env_state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(key, state, action)
        obs["world_state"] = self.world_state(obs)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def world_state(self, obs):
        """
        For each agent: [agent obs, all other agent obs]
        """

        @partial(jax.vmap, in_axes=(0, None))
        def _roll_obs(aidx, all_obs):
            robs = jnp.roll(all_obs, -aidx, axis=0)
            robs = robs.flatten()
            return robs

        all_obs = jnp.array([obs[agent] for agent in self._env.agents]).flatten()
        all_obs = jnp.expand_dims(all_obs, axis=0).repeat(self._env.num_agents, axis=0)
        return all_obs

    def world_state_size(self):
        spaces = [self._env.observation_space(agent) for agent in self._env.agents]
        return sum([space.shape[-1] for space in spaces])


class ScannedRNN(nn.Module):
    @functools.partial(
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
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi


class CriticRNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(world_state)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        critic = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray


@struct.dataclass
class TrainRunnerState:
    actor_train_state: TrainState
    critic_train_state: TrainState
    env_state: MPEWorldStateWrapper
    obs: dict
    done: jnp.array
    actor_hidden_state: jnp.array
    critic_hidden_state: jnp.array
    actor_hidden_state_k: jnp.array
    update_step: int
    rng: Array


@struct.dataclass
class UpdateRunnerState:
    actor_train_state: TrainState
    critic_train_state: TrainState
    traj_batch: Transition
    advantages: jnp.ndarray
    targets: jnp.ndarray
    actor_hidden_state: jnp.ndarray
    critic_hidden_state: jnp.ndarray
    rng: Array


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )

    env = MPEWorldStateWrapper(env)
    env = MPELogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng, exp_id):
        jax.debug.print("Compile Finished. Running...")

        # INIT NETWORK
        actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)
        critic_network = CriticRNN(config=config)
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        ac_init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        actor_hidden_state_init = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        actor_network_params = actor_network.init(
            _rng_actor, actor_hidden_state_init, ac_init_x
        )

        cr_init_x = (
            jnp.zeros(
                (
                    1,
                    config["NUM_ENVS"],
                    env.world_state_size(),
                )
            ),  #  + env.observation_space(env.agents[0]).shape[0]
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        critic_hidden_state_init = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        critic_network_params = critic_network.init(
            _rng_critic, critic_hidden_state_init, cr_init_x
        )

        if config["ANNEAL_LR"]:
            lr_schedule = linear_schedule
        else:
            lr_schedule = lambda x: config["LR"]

        if config["ACTOR_OPTIMIZER"] == "adam":
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=lr_schedule, eps=1e-5),
            )
        if config["CRITIC_OPTIMIZER"] == "adam":
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=lr_schedule, eps=1e-5),
            )
        if config["ACTOR_OPTIMIZER"] == "rmsprop":
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.rmsprop(learning_rate=lr_schedule, eps=1e-5),
            )
        if config["CRITIC_OPTIMIZER"] == "rmsprop":
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.rmsprop(learning_rate=lr_schedule, eps=1e-5),
            )
        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        actor_hidden_state_init = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )
        critic_hidden_state_init = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )

        # TRAIN LOOP
        def _update_step(train_runner_state, unused):
            # COLLECT TRAJECTORIES
            actor_hidden_state_init = train_runner_state.actor_hidden_state
            critic_hidden_state_init = train_runner_state.critic_hidden_state
            update_step = train_runner_state.update_step

            def _env_step(runner_state, unused):
                actor_train_state = runner_state.actor_train_state
                critic_train_state = runner_state.critic_train_state
                env_state = runner_state.env_state
                last_obs = runner_state.obs
                last_done = runner_state.done  # done is always batched
                actor_hidden_state = runner_state.actor_hidden_state
                critic_hidden_state = runner_state.critic_hidden_state
                rng = runner_state.rng

                # SELECT ACTION
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[None, :],
                    last_done[None, :],
                )

                actor_hidden_state_new, pi = actor_network.apply(
                    actor_train_state.params,
                    actor_hidden_state,
                    ac_in,
                )
                rng, _rng = jax.random.split(rng)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                # VALUE
                # output of wrapper is (num_envs, num_agents, world_state_size)
                # swap axes to (num_agents, num_envs, world_state_size) before reshaping to (num_actors, world_state_size)
                world_state = last_obs["world_state"].swapaxes(0, 1)
                world_state = world_state.reshape((config["NUM_ACTORS"], -1))
                cr_in = (
                    world_state[None, :],
                    last_done[np.newaxis, :],
                )
                critic_hidden_state_new, value = critic_network.apply(
                    critic_train_state.params,
                    critic_hidden_state,
                    cr_in,
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obs, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                info_batched = jax.tree.map(
                    lambda x: x.reshape((config["NUM_ACTORS"])), info
                )
                done_batched = batchify(
                    done, env.agents, config["NUM_ACTORS"]
                ).squeeze()
                transition = Transition(  # batched data
                    global_done=jnp.tile(
                        done["__all__"], env.num_agents
                    ),  # global done for each env
                    done=last_done,
                    action=action.squeeze(),
                    value=value.squeeze(),
                    reward=batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob=log_prob.squeeze(),
                    obs=obs_batch,
                    world_state=world_state,
                    info=info_batched,
                )
                runner_state = runner_state.replace(
                    env_state=env_state,
                    obs=obs,
                    done=done_batched,
                    actor_hidden_state=actor_hidden_state_new,
                    critic_hidden_state=critic_hidden_state_new,
                    rng=rng,
                )
                return runner_state, transition

            train_runner_state, traj_batch = jax.lax.scan(
                _env_step, train_runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            actor_train_state = train_runner_state.actor_train_state
            critic_train_state = train_runner_state.critic_train_state
            last_obs = train_runner_state.obs
            last_done = train_runner_state.done
            critic_hidden_state = train_runner_state.critic_hidden_state
            rng = train_runner_state.rng

            last_world_state = last_obs["world_state"].swapaxes(0, 1)
            last_world_state = last_world_state.reshape((config["NUM_ACTORS"], -1))

            # critic needs extra leading dimension on last_world_state and last_done
            cr_in = (
                last_world_state[None, :],
                last_done[None, :],
            )
            _, last_val = critic_network.apply(
                critic_train_state.params, critic_hidden_state, cr_in
            )
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                gae_and_next_value = (
                    jnp.zeros_like(last_val),
                    last_val,
                )
                _, advantages = jax.lax.scan(
                    _get_advantages,
                    gae_and_next_value,  # carry = gae (initially 0) & last value
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minibatch(carry, minibatch_idx):
                    (
                        actor_train_state,
                        critic_train_state,
                        actor_hidden_state_init,
                        critic_hidden_state_init,
                        obs,
                        done,
                        world_state,
                        value,
                        action,
                        log_prob_k0,
                        advantages,
                        targets,
                        permutation,
                    ) = carry

                    actor_params_k0 = actor_train_state.params
                    actor_opt_state_k0 = actor_train_state.opt_state

                    # Get log_prob_k0_joint
                    log_prob_k0_reshape = log_prob_k0.reshape(
                        -1, env.num_agents, config["NUM_ENVS"]
                    )
                    log_prob_k0_prod = jnp.sum(log_prob_k0_reshape, axis=1)
                    log_prob_k0_joint = jnp.tile(log_prob_k0_prod, (1, env.num_agents))

                    batch = (
                        actor_hidden_state_init,
                        critic_hidden_state_init,
                        obs,
                        done,
                        world_state,
                        value,
                        action,
                        advantages,
                        targets,
                        log_prob_k0,
                        log_prob_k0_joint,
                    )
                    shuffled_batch = jax.tree.map(
                        lambda x: jnp.take(x, permutation, axis=1), batch
                    )
                    shuffled_batch_reshaped = jax.tree.map(
                        lambda x: jnp.reshape(  # reshapes shuffled batch into separate minibatches by adding a dimension after actor dim
                            # e.g. advantages_stack (128,320) -> (128,2,160) if NUM_MINIBATCHES = 2
                            # traj_batch_stack.obs (128,320,127) -> (128,2,160,127)
                            x,
                            list(x.shape[0:1])
                            + [config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        shuffled_batch,
                    )
                    minibatches = jax.tree.map(  # move minibatch dimension to the front
                        lambda x: jnp.moveaxis(x, 1, 0), shuffled_batch_reshaped
                    )

                    minibatch_actor_hidden_state_init = minibatches[0][minibatch_idx]
                    minibatch_critic_hidden_state_init = minibatches[1][minibatch_idx]
                    minibatch_obs = minibatches[2][minibatch_idx]
                    minibatch_done = minibatches[3][minibatch_idx]
                    minibatch_world_state = minibatches[4][minibatch_idx]
                    minibatch_value = minibatches[5][minibatch_idx]
                    minibatch_action = minibatches[6][minibatch_idx]
                    minibatch_advantages = minibatches[7][minibatch_idx]
                    minibatch_targets = minibatches[8][minibatch_idx]
                    minibatch_log_prob_k0 = minibatches[9][minibatch_idx]
                    minibatch_log_prob_k0_joint = minibatches[10][minibatch_idx]

                    def _actor_loss_fn_k1(
                        actor_params,
                        actor_hidden_state_init,
                        obs,
                        done,
                        action,
                        gae,
                        log_prob_k0,
                    ):
                        # RERUN NETWORK
                        _, pi = actor_network.apply(
                            actor_params,
                            actor_hidden_state_init.squeeze(),
                            (
                                obs,
                                done,
                            ),
                        )
                        log_prob = pi.log_prob(action)

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - log_prob_k0
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        actor_loss = loss_actor - config["ENT_COEF"] * entropy

                        return actor_loss, (
                            loss_actor,
                            entropy,
                            ratio,
                            approx_kl,
                            clip_frac,
                        )

                    def _critic_loss_fn(
                        critic_params,
                        critic_hidden_state_init,
                        done,
                        world_state,
                        value_old,
                        targets,
                    ):
                        # RERUN NETWORK
                        _, value = critic_network.apply(
                            critic_params,
                            critic_hidden_state_init.squeeze(),
                            (world_state, done),
                        )

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = value_old + (value - value_old).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss)

                    actor_grad_fn_k1 = jax.value_and_grad(
                        _actor_loss_fn_k1, has_aux=True
                    )
                    actor_loss, actor_grads_k1 = actor_grad_fn_k1(
                        actor_train_state.params,
                        minibatch_actor_hidden_state_init,
                        minibatch_obs,
                        minibatch_done,
                        minibatch_action,
                        minibatch_advantages,
                        minibatch_log_prob_k0,
                    )
                    actor_grad_norm_k1 = pytree_norm(actor_grads_k1)

                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params,
                        minibatch_critic_hidden_state_init,
                        minibatch_done,
                        minibatch_world_state,
                        minibatch_value,
                        minibatch_targets,
                    )
                    critic_grad_norm = pytree_norm(critic_grads)

                    actor_train_state = actor_train_state.apply_gradients(
                        grads=actor_grads_k1
                    )
                    critic_train_state = critic_train_state.apply_gradients(
                        grads=critic_grads
                    )

                    # get k1 probs
                    actor_params_k1 = actor_train_state.params
                    _, pi_k1 = actor_network.apply(
                        actor_params_k1,
                        actor_hidden_state_init.squeeze(),
                        (
                            obs,
                            done,
                        ),
                    )
                    log_prob_k1 = pi_k1.log_prob(action)
                    log_prob_k1_reshape = log_prob_k1.reshape(
                        -1, env.num_agents, config["NUM_ENVS"]
                    )
                    log_prob_k1_prod = jnp.sum(log_prob_k1_reshape, axis=1)
                    log_prob_k1_joint = jnp.tile(log_prob_k1_prod, (1, env.num_agents))

                    batch_k2 = (
                        log_prob_k1,
                        log_prob_k1_joint,
                    )
                    shuffled_batch_k2 = jax.tree.map(
                        lambda x: jnp.take(x, permutation, axis=1), batch_k2
                    )
                    shuffled_batch_reshaped_k2 = jax.tree.map(
                        lambda x: jnp.reshape(  # reshapes shuffled batch into separate minibatches by adding a dimension after actor dim
                            # e.g. advantages_stack (128,320) -> (128,2,160) if NUM_MINIBATCHES = 2
                            # traj_batch_stack.obs (128,320,127) -> (128,2,160,127)
                            x,
                            list(x.shape[0:1])
                            + [config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        shuffled_batch_k2,
                    )
                    minibatches_k2 = (
                        jax.tree.map(  # move minibatch dimension to the front
                            lambda x: jnp.moveaxis(x, 1, 0), shuffled_batch_reshaped_k2
                        )
                    )

                    minibatch_log_prob_k1 = minibatches_k2[0][minibatch_idx]
                    minibatch_log_prob_k1_joint = minibatches_k2[1][minibatch_idx]

                    # reset actor
                    actor_train_state = actor_train_state.replace(
                        params=actor_params_k0,
                        opt_state=actor_opt_state_k0,
                    )

                    def _actor_loss_fn_k2(
                        actor_params,
                        actor_hidden_state_init,
                        obs,
                        done,
                        action,
                        gae,
                        log_prob_k0,
                        log_prob_k0_joint,
                        log_prob_k1,
                        log_prob_k1_joint,
                    ):
                        # RERUN NETWORK
                        _, pi = actor_network.apply(
                            actor_params,
                            actor_hidden_state_init.squeeze(),
                            (
                                obs,
                                done,
                            ),
                        )
                        log_prob = pi.log_prob(action)

                        # CALCULATE ACTOR LOSS
                        logratio_is = (
                            log_prob
                            + log_prob_k1_joint
                            - log_prob_k0_joint
                            - log_prob_k1
                        )

                        ratio_is = jnp.exp(logratio_is)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio_is * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio_is,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # debug
                        approx_kl = ((ratio_is - 1) - logratio_is).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio_is - 1) > config["CLIP_EPS"])

                        actor_loss = loss_actor - config["ENT_COEF"] * entropy

                        return actor_loss, (
                            loss_actor,
                            entropy,
                            ratio_is,
                            approx_kl,
                            clip_frac,
                        )

                    actor_grad_fn_k2 = jax.value_and_grad(
                        _actor_loss_fn_k2, has_aux=True
                    )
                    actor_loss_k2, actor_grads_k2 = actor_grad_fn_k2(
                        actor_train_state.params,
                        minibatch_actor_hidden_state_init,
                        minibatch_obs,
                        minibatch_done,
                        minibatch_action,
                        minibatch_advantages,
                        minibatch_log_prob_k0,
                        minibatch_log_prob_k0_joint,
                        minibatch_log_prob_k1,
                        minibatch_log_prob_k1_joint,
                    )
                    actor_grad_norm_k2 = pytree_norm(actor_grads_k2)

                    actor_train_state = actor_train_state.apply_gradients(
                        grads=actor_grads_k2
                    )

                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "entropy": actor_loss[1][1],
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                    }
                    loss_info_k = {
                        "actor_loss_k": actor_loss_k2[0],
                        "entropy_k": actor_loss_k2[1][1],
                        "ratio_k": actor_loss_k2[1][2],
                        "approx_kl_k": actor_loss_k2[1][3],
                        "clip_frac_k": actor_loss_k2[1][4],
                        "actor_grad_norm_k": actor_grad_norm_k2,
                    }

                    return (
                        actor_train_state,
                        critic_train_state,
                        actor_hidden_state_init,
                        critic_hidden_state_init,
                        obs,
                        done,
                        world_state,
                        value,
                        action,
                        log_prob_k0,
                        advantages,
                        targets,
                        permutation,
                    ), (
                        loss_info,
                        loss_info_k,
                    )

                actor_train_state = update_state.actor_train_state
                critic_train_state = update_state.critic_train_state
                traj_batch = update_state.traj_batch
                advantages = update_state.advantages
                targets = update_state.targets
                actor_hidden_state_init = update_state.actor_hidden_state
                critic_hidden_state_init = update_state.critic_hidden_state
                rng = update_state.rng

                # decompose traj_batch
                obs = traj_batch.obs
                done = traj_batch.done
                world_state = traj_batch.world_state
                value = traj_batch.value
                action = traj_batch.action
                log_prob_k0 = traj_batch.log_prob

                rng, _rng = jax.random.split(rng)

                # add extra leading dimension to hstates
                actor_hidden_state_init = actor_hidden_state_init[None, :]
                critic_hidden_state_init = critic_hidden_state_init[None, :]

                # we permute among the actors (remember, we backprop through the entire episode)
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                # batch is in sequence
                carry = (
                    actor_train_state,
                    critic_train_state,
                    actor_hidden_state_init,
                    critic_hidden_state_init,
                    obs,
                    done,
                    world_state,
                    value,
                    action,
                    log_prob_k0,
                    advantages,
                    targets,
                    permutation,
                )
                train_states, loss_info = jax.lax.scan(
                    _update_minibatch,
                    carry,
                    jnp.arange(config["NUM_MINIBATCHES"]),
                )
                update_state = update_state.replace(
                    actor_train_state=train_states[0],
                    critic_train_state=train_states[1],
                    rng=rng,
                )
                return update_state, loss_info

            initial_update_state = UpdateRunnerState(
                actor_train_state=actor_train_state,
                critic_train_state=critic_train_state,
                traj_batch=traj_batch,
                advantages=advantages,
                targets=targets,
                actor_hidden_state=actor_hidden_state_init,
                critic_hidden_state=critic_hidden_state_init,
                rng=rng,
            )
            final_update_state, (loss_info, loss_info_k) = jax.lax.scan(
                _update_epoch,
                initial_update_state,
                None,
                config["UPDATE_EPOCHS"],
            )
            actor_train_state = final_update_state.actor_train_state
            critic_train_state = final_update_state.critic_train_state
            rng = final_update_state.rng

            loss_info["ratio_0"] = loss_info["ratio"].at[0, 0].get()
            loss_info_k["ratio_0_k"] = loss_info_k["ratio_k"].at[0, 0].get()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            loss_info_k = jax.tree.map(lambda x: x.mean(), loss_info_k)

            # Wandb metrics
            metric = traj_batch.info
            metric = jax.tree.map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)
                ),
                traj_batch.info,
            )
            metric["loss"] = loss_info
            metric["loss_k"] = loss_info_k
            metric["update_step"] = update_step

            def callback(exp_id, metric):
                log_dict = {
                    "returns": metric["returned_episode_returns"][-1, :].mean(),
                    "env_step": metric["update_step"]
                    * config["NUM_ENVS"]
                    * config["NUM_STEPS"],
                    **metric["loss"],
                    **metric["loss_k"],
                }
                np_log_dict = {k: np.array(v) for k, v in log_dict.items()}
                LOGGER.log(int(exp_id), np_log_dict)

            jax.experimental.io_callback(callback, None, exp_id, metric)

            update_step = update_step + 1

            train_runner_state = train_runner_state.replace(
                actor_train_state=actor_train_state,
                critic_train_state=critic_train_state,
                update_step=update_step,
                rng=rng,
            )
            return train_runner_state, metric

        rng, _rng = jax.random.split(rng)

        initial_runner_state = TrainRunnerState(
            actor_train_state=actor_train_state,
            critic_train_state=critic_train_state,
            env_state=env_state,
            obs=obsv,
            done=jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            actor_hidden_state=actor_hidden_state_init,
            critic_hidden_state=critic_hidden_state_init,
            actor_hidden_state_k=actor_hidden_state_init,
            update_step=0,
            rng=_rng,
        )
        final_runner_state, metrics_batch = jax.lax.scan(
            _update_step, initial_runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": final_runner_state}

    return train


@hydra.main(version_base=None, config_path="./", config_name="config_ik2m")
def main(config):
    try:
        config = OmegaConf.to_container(config)

        # WANDB
        group = f"K2MAPPO_{config['ENV_NAME']}"
        job_type = f"K2MAPPO_{config['ENV_NAME']}"
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
        train_vmap = jax.vmap(make_train(config))
        train_vjit = jax.jit(train_vmap)
        out = jax.block_until_ready(train_vjit(rng_seeds, exp_ids))
    finally:
        LOGGER.finish()
        print("Finished training.")


if __name__ == "__main__":
    main()
