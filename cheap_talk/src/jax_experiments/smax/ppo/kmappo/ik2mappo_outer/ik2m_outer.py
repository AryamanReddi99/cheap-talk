"""
Based on PureJaxRL Implementation of IPPO, with changes to give a centralised critic.
"""

import jax
import datetime
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Dict
from jax._src.typing import Array

import functools
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import OmegaConf
from functools import partial

from jaxmarl.wrappers.baselines import SMAXLogWrapper, JaxMARLWrapper, SMAXLogEnvState
from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX
from cheap_talk.src.jax_experiments.utils.wandb_process import WandbMultiLogger
from cheap_talk.src.jax_experiments.utils.jax_utils import pytree_norm

# env provides variables as dictionaries indexed by agent ids
# e.g. obs, env_state = self._env.reset(key)
# obs = {'agent_0': ..., 'agent_1': ...}
# if vmapped, the shapes of the returned arrays are (num_envs, array_shape)
# e.g. if there are 64 envs and the obs is 127, then obs['ally_0'].shape = (64,127)

# we will call them env format (dictionaries) and agent format (array of shape (num_actors, arr_dim))


class SMAXWorldStateWrapper(JaxMARLWrapper):
    """
    Provides a `"world_state"` observation for the centralised critic.
    world state observation of dimension: (num_agents, world_state_size)
    """

    def __init__(
        self,
        env: HeuristicEnemySMAX,
        obs_with_agent_id=True,
    ):
        super().__init__(env)
        self.obs_with_agent_id = obs_with_agent_id

        # if obs needs agent_id, we add the id to the world state
        if not self.obs_with_agent_id:
            self._world_state_size = self._env.state_size
            self.world_state_fn = self.ws_just_env_state
        else:
            self._world_state_size = self._env.state_size + self._env.num_allies
            self.world_state_fn = self.ws_with_agent_id

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        # add 'world_state' key to obs dict
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state_fn(obs, env_state)
        return obs, env_state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(key, state, action)
        obs["world_state"] = self.world_state_fn(obs, state)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def ws_just_env_state(self, obs, state):
        # return all_obs
        world_state = obs["world_state"]
        world_state = world_state[None].repeat(self._env.num_allies, axis=0)
        return world_state

    @partial(jax.jit, static_argnums=0)
    def ws_with_agent_id(self, obs, state):
        # all_obs = jnp.array([obs[agent] for agent in self._env.agents])
        world_state = obs["world_state"]
        world_state = world_state[None].repeat(self._env.num_allies, axis=0)
        one_hot = jnp.eye(self._env.num_allies)
        return jnp.concatenate((world_state, one_hot), axis=1)

    def world_state_size(self):

        return self._world_state_size


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
        # rnn_state and ins are (num_actors, hidden_dim)
        rnn_state = carry
        ins, resets = x

        # resets go from shape (1, num_actors) to (1, num_actors, 1)
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
    # these are the constructor args
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
        embedding = nn.relu(
            embedding
        )  # embedding has 3d shape (1, num_actors, FC_DIM_SIZE)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)

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


def batchify(x: dict, agent_list, num_actors):
    # no padding?
    # stack adds a new dim across the agents at axis=0
    # e.g. if the obs had dimension (64, 127) for 64 parallel envs and 127-dim obs,
    # the new shape is (num_agents, 64, 127)
    x = jnp.stack([x[a] for a in agent_list])

    # reshape puts all the vectors into a big 2d array ordered by the agents
    # e.g. [[agent_0 env_0],
    #       [agent_0 env_1]
    #       [[agent_1 env_0],
    #       [agent_1 env_1]]
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_agents):
    # reshapes the 2d array into a 3d array indexed by [agent, env, arr_dim]
    x = x.reshape((num_agents, num_envs, -1))
    # back into a dictionary idexed by agent_id
    return {a: x[i] for i, a in enumerate(agent_list)}


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


@struct.dataclass
class TrainRunnerState:
    actor_train_state: TrainState
    critic_train_state: TrainState
    env_state: SMAXLogEnvState
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


def make_train(config):
    # Environment
    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    env = SMAXWorldStateWrapper(env, config["OBS_WITH_AGENT_ID"])
    env = SMAXLogWrapper(env)

    # Config
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

    def train(rng, exp_id):
        jax.debug.print("Compile Finished. Running...")

        def train_setup(rng):
            # Networks
            actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)
            critic_network = CriticRNN(config=config)
            rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
            ac_init_x = (
                jnp.zeros(
                    (
                        1,
                        config["NUM_ENVS"],
                        env.observation_space(env.agents[0]).shape[0],
                    )
                ),
                jnp.zeros((1, config["NUM_ENVS"])),
                jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n)),
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
                ),
                jnp.zeros((1, config["NUM_ENVS"])),
            )
            critic_hidden_state_init = ScannedRNN.initialize_carry(
                config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
            )
            critic_network_params = critic_network.init(
                _rng_critic, critic_hidden_state_init, cr_init_x
            )

            # Optimizers
            if config["ANNEAL_LR"]:

                def lr_schedule(count):
                    frac = (
                        1.0
                        - (
                            count
                            // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
                        )
                        / config["NUM_UPDATES"]
                    )
                    return config["LR"] * frac

            else:

                def lr_schedule(count):
                    return config["LR"]

            if config["ACTOR_OPTIMIZER"] == "adam":
                actor_tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=lr_schedule, eps=1e-5),
                )
            elif config["ACTOR_OPTIMIZER"] == "rmsprop":
                actor_tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.rmsprop(learning_rate=lr_schedule, eps=1e-5),
                )
            if config["CRITIC_OPTIMIZER"] == "adam":
                critic_tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=lr_schedule, eps=1e-5),
                )
            elif config["CRITIC_OPTIMIZER"] == "rmsprop":
                critic_tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.rmsprop(learning_rate=lr_schedule, eps=1e-5),
                )

            # Train states
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

            # Initialise envs, hstates
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
            actor_hidden_state_init = ScannedRNN.initialize_carry(
                config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
            )
            critic_hidden_state_init = ScannedRNN.initialize_carry(
                config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
            )
            return (
                actor_train_state,
                critic_train_state,
                actor_network,
                critic_network,
                obs,
                env_state,
                actor_hidden_state_init,
                critic_hidden_state_init,
            )

        rng, _rng_setup = jax.random.split(rng)
        (
            actor_train_state,
            critic_train_state,
            actor_network,
            critic_network,
            obs,
            env_state,
            actor_hidden_state_init,
            critic_hidden_state_init,
        ) = train_setup(_rng_setup)

        def _train_loop(train_runner_state, unused):

            # save initial settings for k
            actor_params_k0 = train_runner_state.actor_train_state.params
            critic_params_k0 = train_runner_state.critic_train_state.params
            actor_optimizer_k0 = train_runner_state.actor_train_state.opt_state
            env_state_initial = train_runner_state.env_state
            obs_initial = train_runner_state.obs
            done_initial = train_runner_state.done
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
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[None, :],
                    last_done[None, :],
                    avail_actions,
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
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # VALUE
                # world_state -> (num_envs, num_agents, world_state_size)
                # swap axes to (num_agents, num_envs, world_state_size) before reshaping to (num_actors, world_state_size)
                world_state = last_obs["world_state"].swapaxes(0, 1)
                world_state = world_state.reshape((config["NUM_ACTORS"], -1))

                cr_in = (
                    world_state[None, :],
                    last_done[None, :],
                )
                critic_hidden_state_new, value = critic_network.apply(
                    critic_train_state.params,
                    critic_hidden_state,
                    cr_in,
                )

                # Env info (num_envs, num_agents):
                # returned_episode (whether ep finished),
                # returned_episode_lengths (length so far)
                # returned_episode_returns (return so far)
                # returned_won_episode (success so far)

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
                    avail_actions=avail_actions,
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

            # traj_batch is Transition object with each element being a
            # jnp array of shape (config["NUM_STEPS"], config["NUM_ACTORS"], arr_dim)
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

            # shape (num_actors)
            advantages, targets = _calculate_gae(traj_batch, last_val)

            # multiple update epochs
            def _update_epoch_k1(update_state, unused):
                def _update_minibatch_k1(train_states, minibatch):
                    actor_train_state, critic_train_state = train_states
                    (
                        actor_hidden_state_init,
                        critic_hidden_state_init,
                        traj_batch,
                        advantages,
                        targets,
                    ) = minibatch

                    def _actor_loss_fn(
                        actor_params, actor_hidden_state_init, traj_batch, gae
                    ):
                        # RERUN NETWORK
                        _, pi = actor_network.apply(
                            actor_params,
                            actor_hidden_state_init.squeeze(),
                            (
                                traj_batch.obs,
                                traj_batch.done,
                                traj_batch.avail_actions,
                            ),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
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
                        critic_params, critic_hidden_state_init, traj_batch, targets
                    ):
                        # RERUN NETWORK
                        _, value = critic_network.apply(
                            critic_params,
                            critic_hidden_state_init.squeeze(),
                            (traj_batch.world_state, traj_batch.done),
                        )

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params,
                        actor_hidden_state_init,
                        traj_batch,
                        advantages,
                    )
                    actor_grad_norm = pytree_norm(actor_grads)

                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params,
                        critic_hidden_state_init,
                        traj_batch,
                        targets,
                    )
                    critic_grad_norm = pytree_norm(critic_grads)

                    actor_train_state = actor_train_state.apply_gradients(
                        grads=actor_grads
                    )
                    critic_train_state = critic_train_state.apply_gradients(
                        grads=critic_grads
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
                        "actor_grad_norm": actor_grad_norm,
                        "critic_grad_norm": critic_grad_norm,
                    }

                    return (actor_train_state, critic_train_state), loss_info

                actor_train_state = update_state.actor_train_state
                critic_train_state = update_state.critic_train_state
                traj_batch = update_state.traj_batch
                advantages = update_state.advantages
                targets = update_state.targets
                actor_hidden_state_init = update_state.actor_hidden_state
                critic_hidden_state_init = update_state.critic_hidden_state
                rng = update_state.rng

                rng, _rng = jax.random.split(rng)

                # add extra leading dimension to hstates
                actor_hidden_state_init = actor_hidden_state_init[None, :]
                critic_hidden_state_init = critic_hidden_state_init[None, :]

                # batch is in sequence
                # init_hstates has shape (1, NUM_ACTORS, dim)
                batch = (
                    actor_hidden_state_init,
                    critic_hidden_state_init,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                # we permute among the actors (remember, we backprop through the entire episode)
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
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

                train_states, loss_info = jax.lax.scan(
                    _update_minibatch_k1,
                    (actor_train_state, critic_train_state),
                    minibatches,
                )
                update_state = update_state.replace(
                    actor_train_state=train_states[0],
                    critic_train_state=train_states[1],
                    rng=rng,
                )
                return update_state, loss_info

            initial_update_state_k1 = UpdateRunnerState(
                actor_train_state=actor_train_state,
                critic_train_state=critic_train_state,
                traj_batch=traj_batch,
                advantages=advantages,
                targets=targets,
                actor_hidden_state=actor_hidden_state_init,
                critic_hidden_state=critic_hidden_state_init,
                rng=rng,
            )
            final_update_state_k1, loss_info = jax.lax.scan(
                _update_epoch_k1,
                initial_update_state_k1,
                None,
                config["UPDATE_EPOCHS"],
            )

            actor_train_state = final_update_state_k1.actor_train_state
            critic_train_state = final_update_state_k1.critic_train_state

            actor_params_k1 = actor_train_state.params
            actor_train_state = actor_train_state.replace(
                params=actor_params_k0, opt_state=actor_optimizer_k0
            )

            initial_update_state_k2 = UpdateRunnerState(
                actor_train_state=actor_train_state,
                critic_train_state=critic_train_state,
                traj_batch=traj_batch,
                advantages=advantages,
                targets=None,
                actor_hidden_state=actor_hidden_state_init,
                critic_hidden_state=critic_hidden_state_init,
                rng=rng,
            )

            def _update_epoch_k2(update_state, unused):
                def _update_minibatch_k2(train_states, minibatch):
                    actor_train_state, critic_train_state = train_states
                    (
                        actor_hidden_state_init,
                        traj_batch,
                        advantages,
                        log_prob_k0,
                        log_prob_k0_joint,
                        log_prob_k1,
                        log_prob_k1_joint,
                    ) = minibatch

                    def _actor_loss_fn_k2(
                        actor_params,
                        actor_hidden_state_init,
                        traj_batch,
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
                                traj_batch.obs,
                                traj_batch.done,
                                traj_batch.avail_actions,
                            ),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE ACTOR LOSS
                        logratio_is = (
                            log_prob
                            - log_prob_k0
                            + (1 / (env.num_agents - 1))
                            * (
                                log_prob_k1_joint
                                - log_prob_k1
                                - log_prob_k0_joint
                                + log_prob_k0
                            )
                        )
                        # logratio_is = (
                        #     log_prob
                        #     + log_prob_k1_joint
                        #     - log_prob_k1
                        #     - log_prob_k0_joint
                        # )
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
                    actor_loss_k, actor_grads_k = actor_grad_fn_k2(
                        actor_train_state.params,
                        actor_hidden_state_init,
                        traj_batch,
                        advantages,
                        log_prob_k0,
                        log_prob_k0_joint,
                        log_prob_k1,
                        log_prob_k1_joint,
                    )
                    actor_grad_norm_k = pytree_norm(actor_grads_k)

                    actor_train_state = actor_train_state.apply_gradients(
                        grads=actor_grads_k
                    )

                    loss_info = {
                        "actor_loss_k": actor_loss_k[0],
                        "entropy_k": actor_loss_k[1][1],
                        "ratio_k": actor_loss_k[1][2],
                        "approx_kl_k": actor_loss_k[1][3],
                        "clip_frac_k": actor_loss_k[1][4],
                        "actor_grad_norm_k": actor_grad_norm_k,
                    }

                    return (actor_train_state, critic_train_state), loss_info

                actor_train_state = update_state.actor_train_state
                traj_batch = update_state.traj_batch
                advantages = update_state.advantages
                actor_hidden_state_init = update_state.actor_hidden_state
                critic_hidden_state_init = update_state.critic_hidden_state
                rng = update_state.rng

                rng, _rng = jax.random.split(rng)

                # Get log prob k0_joint
                log_prob_k0 = traj_batch.log_prob
                log_prob_k0_reshape = log_prob_k0.reshape(
                    -1, env.num_agents, config["NUM_ENVS"]
                )
                log_prob_k0_prod = jnp.sum(log_prob_k0_reshape, axis=1)
                log_prob_k0_joint = jnp.tile(log_prob_k0_prod, (1, env.num_agents))

                # Get log prob k1 and k1_joint
                _, pi_k1 = actor_network.apply(
                    actor_params_k1,
                    actor_hidden_state_init.squeeze(),
                    (
                        traj_batch.obs,
                        traj_batch.done,
                        traj_batch.avail_actions,
                    ),
                )
                log_prob_k1 = pi_k1.log_prob(traj_batch.action)
                log_prob_k1_reshape = log_prob_k1.reshape(
                    -1, env.num_agents, config["NUM_ENVS"]
                )
                log_prob_k1_prod = jnp.sum(log_prob_k1_reshape, axis=1)
                log_prob_k1_joint = jnp.tile(log_prob_k1_prod, (1, env.num_agents))

                # add extra leading dimension to hstates
                actor_hidden_state_init = actor_hidden_state_init[None, :]
                critic_hidden_state_init = critic_hidden_state_init[None, :]

                # batch is in sequence
                # init_hstates has shape (1, NUM_ACTORS, dim)
                batch = (
                    actor_hidden_state_init,
                    traj_batch,
                    advantages.squeeze(),
                    log_prob_k0,
                    log_prob_k0_joint,
                    log_prob_k1,
                    log_prob_k1_joint,
                )
                # we permute among the actors (remember, we backprop through the entire episode)
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
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

                train_states, loss_info = jax.lax.scan(
                    _update_minibatch_k2,
                    (actor_train_state, critic_train_state),
                    minibatches,
                )
                update_state = update_state.replace(
                    actor_train_state=train_states[0],
                    critic_train_state=train_states[1],
                    rng=rng,
                )
                return update_state, loss_info

            final_update_state_k2, loss_info_k = jax.lax.scan(
                _update_epoch_k2,
                initial_update_state_k2,
                None,
                config["UPDATE_EPOCHS"],
            )

            loss_info["ratio_0"] = loss_info["ratio"].at[0, 0].get()
            loss_info_k["ratio_0_k"] = loss_info_k["ratio_k"].at[0, 0].get()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            loss_info_k = jax.tree.map(lambda x: x.mean(), loss_info_k)

            actor_train_state = final_update_state_k2.actor_train_state
            rng = final_update_state_k2.rng

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
                    "return": metric["returned_episode_returns"][:, :, 0][
                        metric["returned_episode"][:, :, 0]
                    ].mean(),
                    "win_rate": metric["returned_won_episode"][:, :, 0][
                        metric["returned_episode"][:, :, 0]
                    ].mean(),
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

        rng, _train_rng = jax.random.split(rng)

        initial_runner_state = TrainRunnerState(
            actor_train_state=actor_train_state,
            critic_train_state=critic_train_state,
            env_state=env_state,
            obs=obs,
            done=jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            actor_hidden_state=actor_hidden_state_init,
            critic_hidden_state=critic_hidden_state_init,
            actor_hidden_state_k=actor_hidden_state_init,
            update_step=0,
            rng=_train_rng,
        )

        # highest level runner state has 6 elements
        final_runner_state, metrics_batch = jax.lax.scan(
            _train_loop, initial_runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": final_runner_state}

    return train


@hydra.main(version_base=None, config_path="./", config_name="config_ik2m_outer")
def main(config):
    try:
        config = OmegaConf.to_container(config)

        # WANDB
        job_type = f"iK2MAPPO_{config['MAP_NAME']}"
        group = f"iK2MAPPO_{config['MAP_NAME']}"
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
        train_jit = jax.jit(make_train(config))
        out = jax.vmap(train_jit)(rng_seeds, exp_ids)
    finally:
        LOGGER.finish()
        print("Finished training.")


if __name__ == "__main__":
    main()
