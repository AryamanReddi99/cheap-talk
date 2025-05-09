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
from cheap_talk.src.jax_experiments.utils.wandb_process import WandbMultiLogger
from cheap_talk.src.jax_experiments.utils.jax_utils import pytree_norm
import chex
import os
from safetensors.flax import save_file
from flax.traverse_util import flatten_dict

from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import DictConfig, OmegaConf
from functools import partial
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper, JaxMARLWrapper
from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State


import wandb
import functools
import matplotlib.pyplot as plt


class HanabiWorldStateWrapper(JaxMARLWrapper):

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state(obs, env_state)
        return obs, env_state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(key, state, action)
        obs["world_state"] = self.world_state(obs, state)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def world_state(self, obs, state):
        """
        For each agent: [agent obs, own hand]
        """

        return jnp.array([obs[agent] for agent in self._env.agents])
        # hands = state.player_hands.reshape((self._env.num_agents, -1))
        # return jnp.concatenate((all_obs, hands), axis=1)

    @partial(jax.jit, static_argnums=0)
    def world_state_size(self):

        return (
            self._env.observation_space(self._env.agents[0]).n * self._env.num_agents
        )  # + 125 # NOTE hardcoded hand size


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
            self.initialize_carry(ins.shape[0], ins.shape[1]),
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
        obs, dones, avail_actions = x
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
        unavail_actions = 1 - avail_actions
        action_logits = action_logits - (unavail_actions * 1e10)

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

        critic = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
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
    avail_actions: jnp.ndarray


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

    # env = FlattenObservationWrapper(env) # NOTE need a batchify wrapper
    env = HanabiWorldStateWrapper(env)
    env = LogWrapper(env)

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
            jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).n)),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n)),
        )
        ac_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)
        print("ac init x", ac_init_x)
        cr_init_x = (
            jnp.zeros(
                (
                    1,
                    config["NUM_ENVS"],
                    658,
                )
            ),  # NOTE hardcoded
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        critic_network_params = critic_network.init(
            _rng_critic, cr_init_hstate, cr_init_x
        )

        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
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
        ac_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )
        cr_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done, hstates, rng = (
                    runner_state
                )

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_legal_moves)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    avail_actions,
                )
                ac_hstate, pi = actor_network.apply(
                    train_states[0].params, hstates[0], ac_in
                )
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = jax.tree.map(lambda x: x.squeeze(), env_act)

                # VALUE
                # output of wrapper is (num_envs, num_agents, world_state_size)
                # swap axes to (num_agents, num_envs, world_state_size) before reshaping to (num_actors, world_state_size)
                world_state = last_obs["world_state"].swapaxes(0, 1)
                world_state = world_state.reshape((config["NUM_ACTORS"], -1))
                cr_in = (
                    world_state[None, :],
                    last_done[np.newaxis, :],
                )
                cr_hstate, value = critic_network.apply(
                    train_states[1].params, hstates[1], cr_in
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    world_state,
                    info,
                    avail_actions,
                )
                runner_state = (
                    train_states,
                    env_state,
                    obsv,
                    done_batch,
                    (ac_hstate, cr_hstate),
                    rng,
                )
                return runner_state, transition

            initial_hstates = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, last_done, hstates, rng = runner_state

            last_world_state = last_obs["world_state"].swapaxes(0, 1)
            last_world_state = last_world_state.reshape((config["NUM_ACTORS"], -1))
            cr_in = (
                last_world_state[None, :],
                last_done[np.newaxis, :],
            )
            _, last_val = critic_network.apply(
                train_states[1].params, hstates[1], cr_in
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

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
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
                        avail_actions,
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
                        avail_actions,
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
                    minibatch_avail_actions = minibatches[4][minibatch_idx]
                    minibatch_world_state = minibatches[5][minibatch_idx]
                    minibatch_value = minibatches[6][minibatch_idx]
                    minibatch_action = minibatches[7][minibatch_idx]
                    minibatch_advantages = minibatches[8][minibatch_idx]
                    minibatch_targets = minibatches[9][minibatch_idx]
                    minibatch_log_prob_k0 = minibatches[10][minibatch_idx]
                    minibatch_log_prob_k0_joint = minibatches[11][minibatch_idx]

                    def _actor_loss_fn_k1(
                        actor_params,
                        actor_hidden_state_init,
                        obs,
                        done,
                        avail_actions,
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
                                avail_actions,
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
                        minibatch_avail_actions,
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

                    def _k_update(carry_k, unused):
                        (
                            actor_train_state,
                            actor_params_k0,
                            actor_opt_state_k0,
                            actor_hidden_state_init,
                            obs,
                            done,
                            action,
                            minibatch_actor_hidden_state_init,
                            minibatch_obs,
                            minibatch_done,
                            minibatch_action,
                            minibatch_advantages,
                            minibatch_log_prob_k0,
                            minibatch_log_prob_k0_joint,
                            permutation,
                        ) = carry_k

                        # get k probs
                        actor_params_k = actor_train_state.params
                        _, pi_k = actor_network.apply(
                            actor_params_k,
                            actor_hidden_state_init.squeeze(),
                            (
                                obs,
                                done,
                                avail_actions,
                            ),
                        )
                        log_prob_k = pi_k.log_prob(action)
                        log_prob_k_reshape = log_prob_k.reshape(
                            -1, env.num_agents, config["NUM_ENVS"]
                        )
                        log_prob_k_prod = jnp.sum(log_prob_k_reshape, axis=1)
                        log_prob_k_joint = jnp.tile(
                            log_prob_k_prod, (1, env.num_agents)
                        )

                        batch_k = (
                            log_prob_k,
                            log_prob_k_joint,
                        )
                        shuffled_batch_k = jax.tree.map(
                            lambda x: jnp.take(x, permutation, axis=1), batch_k
                        )
                        shuffled_batch_reshaped_k = jax.tree.map(
                            lambda x: jnp.reshape(  # reshapes shuffled batch into separate minibatches by adding a dimension after actor dim
                                # e.g. advantages_stack (128,320) -> (128,2,160) if NUM_MINIBATCHES = 2
                                # traj_batch_stack.obs (128,320,127) -> (128,2,160,127)
                                x,
                                list(x.shape[0:1])
                                + [config["NUM_MINIBATCHES"], -1]
                                + list(x.shape[2:]),
                            ),
                            shuffled_batch_k,
                        )
                        minibatches_k = (
                            jax.tree.map(  # move minibatch dimension to the front
                                lambda x: jnp.moveaxis(x, 1, 0),
                                shuffled_batch_reshaped_k,
                            )
                        )

                        minibatch_log_prob_k = minibatches_k[0][minibatch_idx]
                        minibatch_log_prob_k_joint = minibatches_k[1][minibatch_idx]

                        # reset actor
                        actor_train_state = actor_train_state.replace(
                            params=actor_params_k0,
                            opt_state=actor_opt_state_k0,
                        )

                        def _actor_loss_fn_k(
                            actor_params,
                            actor_hidden_state_init,
                            obs,
                            done,
                            avail_actions,
                            action,
                            gae,
                            log_prob_k0,
                            log_prob_k0_joint,
                            log_prob_k,
                            log_prob_k_joint,
                        ):
                            # RERUN NETWORK
                            _, pi = actor_network.apply(
                                actor_params,
                                actor_hidden_state_init.squeeze(),
                                (
                                    obs,
                                    done,
                                    avail_actions,
                                ),
                            )
                            log_prob = pi.log_prob(action)

                            # CALCULATE ACTOR LOSS
                            logratio_is = (
                                log_prob
                                + log_prob_k_joint
                                - log_prob_k0_joint
                                - log_prob_k
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
                            clip_frac = jnp.mean(
                                jnp.abs(ratio_is - 1) > config["CLIP_EPS"]
                            )

                            actor_loss = loss_actor - config["ENT_COEF"] * entropy

                            return actor_loss, (
                                loss_actor,
                                entropy,
                                ratio_is,
                                approx_kl,
                                clip_frac,
                            )

                        actor_grad_fn_k = jax.value_and_grad(
                            _actor_loss_fn_k, has_aux=True
                        )
                        actor_loss_k, actor_grads_k = actor_grad_fn_k(
                            actor_train_state.params,
                            minibatch_actor_hidden_state_init,
                            minibatch_obs,
                            minibatch_done,
                            minibatch_avail_actions,
                            minibatch_action,
                            minibatch_advantages,
                            minibatch_log_prob_k0,
                            minibatch_log_prob_k0_joint,
                            minibatch_log_prob_k,
                            minibatch_log_prob_k_joint,
                        )

                        actor_grad_norm_k = pytree_norm(actor_grads_k)

                        actor_train_state = actor_train_state.apply_gradients(
                            grads=actor_grads_k
                        )

                        carry_k = (
                            actor_train_state,
                            actor_params_k0,
                            actor_opt_state_k0,
                            actor_hidden_state_init,
                            obs,
                            done,
                            action,
                            minibatch_actor_hidden_state_init,
                            minibatch_obs,
                            minibatch_done,
                            minibatch_action,
                            minibatch_advantages,
                            minibatch_log_prob_k0,
                            minibatch_log_prob_k0_joint,
                            permutation,
                        )

                        loss_info_k = {
                            "actor_loss_k": actor_loss_k[0],
                            "entropy_k": actor_loss_k[1][1],
                            "ratio_k": actor_loss_k[1][2],
                            "approx_kl_k": actor_loss_k[1][3],
                            "clip_frac_k": actor_loss_k[1][4],
                            "actor_grad_norm_k": actor_grad_norm_k,
                        }

                        return carry_k, loss_info_k

                    init_carry_k = (
                        actor_train_state,
                        actor_params_k0,
                        actor_opt_state_k0,
                        actor_hidden_state_init,
                        obs,
                        done,
                        action,
                        minibatch_actor_hidden_state_init,
                        minibatch_obs,
                        minibatch_done,
                        minibatch_action,
                        minibatch_advantages,
                        minibatch_log_prob_k0,
                        minibatch_log_prob_k0_joint,
                        permutation,
                    )

                    final_carry_k, loss_info_k = jax.lax.scan(
                        _k_update, init_carry_k, None, config["K"] - 1
                    )
                    actor_train_state = final_carry_k[0]

                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "entropy": actor_loss[1][1],
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                        "actor_grad_norm": actor_grad_norm_k1,
                    }
                    loss_info_k = {k: v.mean() for k, v in loss_info_k.items()}

                    return (
                        actor_train_state,
                        critic_train_state,
                        actor_hidden_state_init,
                        critic_hidden_state_init,
                        obs,
                        done,
                        avail_actions,
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

                (
                    train_states,
                    init_hstates,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state

                # decompose traj_batch
                obs = traj_batch.obs
                done = traj_batch.done
                avail_actions = traj_batch.avail_actions
                world_state = traj_batch.world_state
                value = traj_batch.value
                action = traj_batch.action
                log_prob_k0 = traj_batch.log_prob

                rng, _rng = jax.random.split(rng)

                # add extra leading dimension to hstates
                actor_hidden_state_init = init_hstates[0][None, :]
                critic_hidden_state_init = init_hstates[1][None, :]

                # we permute among the actors (remember, we backprop through the entire episode)
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                carry = (
                    train_states[0],
                    train_states[1],
                    actor_hidden_state_init,
                    critic_hidden_state_init,
                    obs,
                    done,
                    avail_actions,
                    world_state,
                    value,
                    action,
                    log_prob_k0,
                    advantages,
                    targets,
                    permutation,
                )
                carry_out, loss_info = jax.lax.scan(
                    _update_minibatch, carry, jnp.arange(config["NUM_MINIBATCHES"])
                )
                actor_train_state = carry_out[0]
                critic_train_state = carry_out[1]
                update_state = (
                    (actor_train_state, critic_train_state),
                    jax.tree.map(lambda x: x.squeeze(), init_hstates),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                train_states,
                initial_hstates,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            final_update_state, (loss_info, loss_info_k) = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            loss_info["ratio_0"] = loss_info["ratio"].at[0, 0].get()
            loss_info_k["ratio_0_k"] = loss_info_k["ratio_k"].at[0, 0].get()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            loss_info_k = jax.tree.map(lambda x: x.mean(), loss_info_k)

            train_states = update_state[0]
            metric = traj_batch.info
            metric = jax.tree.map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)
                ),
                traj_batch.info,
            )
            metric["loss"] = loss_info
            metric["loss_k"] = loss_info_k
            metric["update_step"] = update_steps
            rng = final_update_state[-1]

            def callback(exp_id, metrics):
                metric_dict = {
                    "returns": metrics["returned_episode_returns"][-1, :].mean(),
                    "env_step": metrics["update_steps"]
                    * config["NUM_ENVS"]
                    * config["NUM_STEPS"],
                    **metrics["loss"],
                }
                np_log_dict = {k: np.array(v) for k, v in metric_dict.items()}
                LOGGER.log(int(exp_id), np_log_dict)

            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, exp_id, metric)
            update_steps = update_steps + 1
            runner_state = (train_states, env_state, last_obs, last_done, hstates, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            (ac_init_hstate, cr_init_hstate),
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train


@hydra.main(version_base=None, config_path="./", config_name="config")
def main(config):
    try:
        config = OmegaConf.to_container(config)
        # WANDB
        job_type = f"K{config['K']}MAPPO_{config['ENV_NAME']}"
        group = f"K{config['K']}MAPPO_{config['ENV_NAME']}"
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
        outs = jax.block_until_ready(train_vjit(rng_seeds, exp_ids))
    finally:
        LOGGER.finish()
        print("Finished training")
    # save params
    # if config["SAVE_PATH"] is not None:

    #     def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
    #         flattened_dict = flatten_dict(params, sep=",")
    #         save_file(flattened_dict, filename)

    #     params = out["runner_state"][0][0][0].params
    #     save_dir = os.path.join(config["SAVE_PATH"], run.project, run.name)
    #     os.makedirs(save_dir, exist_ok=True)
    #     save_params(params, f"{save_dir}/model.safetensors")
    #     print(f"Parameters of first batch saved in {save_dir}/model.safetensors")

    #     # upload this to wandb as an artifact
    #     artifact = wandb.Artifact(f"{run.name}-checkpoint", type="checkpoint")
    #     artifact.add_file(f"{save_dir}/model.safetensors")
    #     artifact.save()


if __name__ == "__main__":
    main()
