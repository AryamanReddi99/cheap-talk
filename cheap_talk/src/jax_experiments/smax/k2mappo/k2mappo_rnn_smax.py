"""
Based on PureJaxRL Implementation of IPPO, with changes to give a centralised critic.
"""

import flax.serialization
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
import flax
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
    actor_train_state_k: TrainState
    critic_train_state_k: TrainState
    env_state: SMAXLogEnvState
    obs: dict
    done: jnp.array
    actor_hidden_state: jnp.array
    critic_hidden_state: jnp.array
    actor_hidden_state_k: jnp.array
    critic_hidden_state_k: jnp.array
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
            def linear_schedule(count):
                frac = (
                    1.0
                    - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
                    / config["NUM_UPDATES"]
                )
                return config["LR"] * frac

            if config["ANNEAL_LR"]:
                actor_tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=linear_schedule, eps=1e-5),
                )
                critic_tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=linear_schedule, eps=1e-5),
                )

                # K optimizers (just for convenience)
                actor_tx_k = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=linear_schedule, eps=1e-5),
                )
                critic_tx_k = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=linear_schedule, eps=1e-5),
                )

            else:
                actor_tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.rmsprop(config["LR"], eps=1e-5),
                )
                critic_tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.rmsprop(config["LR"], eps=1e-5),
                )

                # K
                actor_tx_k = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.rmsprop(config["LR"], eps=1e-5),
                )
                critic_tx_k = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.rmsprop(config["LR"], eps=1e-5),
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

            # K1 networks
            actor_network_k = ActorRNN(env.action_space(env.agents[0]).n, config=config)
            critic_network_k = CriticRNN(config=config)
            rng, _rng_actor_k, _rng_critic_k = jax.random.split(rng, 3)
            actor_network_params_k = actor_network_k.init(
                _rng_actor_k, actor_hidden_state_init, ac_init_x
            )
            critic_network_params_k = critic_network.init(
                _rng_critic_k, critic_hidden_state_init, cr_init_x
            )

            # Train states
            actor_train_state_k = TrainState.create(
                apply_fn=actor_network_k.apply,
                params=actor_network_params_k,
                tx=actor_tx_k,
            )
            critic_train_state_k = TrainState.create(
                apply_fn=critic_network_k.apply,
                params=critic_network_params_k,
                tx=critic_tx_k,
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
                actor_train_state_k,
                critic_train_state_k,
                actor_network_k,
                critic_network_k,
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
            actor_train_state_k,
            critic_train_state_k,
            actor_network_k,
            critic_network_k,
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
            actor_train_state_k = train_runner_state.actor_train_state_k
            critic_train_state_k = train_runner_state.critic_train_state_k
            env_state_initial = train_runner_state.env_state
            obs_initial = train_runner_state.obs
            done_initial = train_runner_state.done
            actor_hidden_state_init = train_runner_state.actor_hidden_state
            critic_hidden_state_init = train_runner_state.critic_hidden_state
            update_step = train_runner_state.update_step

            def _update_step_k1(train_runner_state, unused):
                # save initial hidden states
                actor_hidden_state_init = train_runner_state.actor_hidden_state
                critic_hidden_state_init = train_runner_state.critic_hidden_state

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
                        reward=batchify(
                            reward, env.agents, config["NUM_ACTORS"]
                        ).squeeze(),
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
                        delta = (
                            reward + config["GAMMA"] * next_value * (1 - done) - value
                        )
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
                def _update_epoch(update_state, unused):
                    def _update_minibatch(train_states, minibatch):
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
                            clip_frac = jnp.mean(
                                jnp.abs(ratio - 1) > config["CLIP_EPS"]
                            )

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
                            value_losses_clipped = jnp.square(
                                value_pred_clipped - targets
                            )
                            value_loss = (
                                0.5
                                * jnp.maximum(value_losses, value_losses_clipped).mean()
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

                        critic_grad_fn = jax.value_and_grad(
                            _critic_loss_fn, has_aux=True
                        )
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
                        _update_minibatch,
                        (actor_train_state, critic_train_state),
                        minibatches,
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
                final_update_state, loss_info = jax.lax.scan(
                    _update_epoch, initial_update_state, None, config["UPDATE_EPOCHS"]
                )
                loss_info["ratio_0"] = loss_info["ratio"].at[0, 0].get()
                loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

                actor_train_state = final_update_state.actor_train_state
                critic_train_state = final_update_state.critic_train_state
                rng = final_update_state.rng

                train_runner_state = train_runner_state.replace(
                    actor_train_state=actor_train_state,
                    critic_train_state=critic_train_state,
                    rng=rng,
                )
                return train_runner_state, traj_batch, loss_info

            train_runner_state, traj_batch, loss_info = _update_step_k1(
                train_runner_state, None
            )

            actor_train_state = train_runner_state.actor_train_state
            critic_train_state = train_runner_state.critic_train_state
            rng = train_runner_state.rng

            # Set k networks to k1 state
            actor_train_state_k = actor_train_state_k.replace(
                params=actor_train_state.params
            )
            critic_train_state_k = critic_train_state_k.replace(
                params=critic_train_state.params
            )

            # reset actor & critic to k0
            actor_train_state = actor_train_state.replace(
                params=actor_params_k0, opt_state=actor_optimizer_k0
            )
            critic_params_k1 = critic_train_state.params
            critic_train_state = critic_train_state.replace(params=critic_params_k0)

            def _update_step_k2(train_runner_state_k):
                # save initial hidden states
                actor_hidden_state_init = train_runner_state_k.actor_hidden_state
                critic_hidden_state_init = train_runner_state_k.critic_hidden_state

                def _get_advantages(runner_state_k, agent_k0_idx):
                    def _env_step(runner_state_k, unused):
                        # runner_state is a single object that gets passed by the scan back to _env_step at each loop
                        actor_train_state = runner_state_k.actor_train_state
                        critic_train_state = runner_state_k.critic_train_state
                        actor_train_state_k = runner_state_k.actor_train_state_k
                        env_state = runner_state_k.env_state
                        last_obs = runner_state_k.obs
                        last_done = runner_state_k.done  # done is always batched
                        actor_hidden_state = runner_state_k.actor_hidden_state
                        critic_hidden_state = runner_state_k.critic_hidden_state
                        actor_hidden_state_k = runner_state_k.actor_hidden_state_k
                        rng = runner_state_k.rng

                        # SELECT ACTION
                        avail_actions = jax.vmap(env.get_avail_actions)(
                            env_state.env_state
                        )
                        avail_actions = jax.lax.stop_gradient(
                            batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                        )
                        obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                        ac_in = (
                            obs_batch[None, :],
                            last_done[None, :],
                            avail_actions,
                        )

                        # network.apply just means we use the __call__ method of nn.Module we overwrite in the network child class
                        rng, _rng, _rng_k = jax.random.split(rng, 3)
                        actor_hidden_state_new, pi = actor_network.apply(
                            actor_train_state.params, actor_hidden_state, ac_in
                        )
                        action = pi.sample(seed=_rng)
                        action_reshaped = action.reshape(
                            (env.num_agents, config["NUM_ENVS"], -1)
                        )
                        log_prob = pi.log_prob(action)

                        # k1 actions
                        actor_hidden_state_k_new, pi_k = actor_network_k.apply(
                            actor_train_state_k.params, actor_hidden_state_k, ac_in
                        )
                        action_k = jax.lax.stop_gradient(pi_k.sample(seed=_rng_k))
                        action_k_reshaped = action_k.reshape(
                            (env.num_agents, config["NUM_ENVS"], -1)
                        )
                        action_k_reshaped = action_k_reshaped.at[agent_k0_idx].set(
                            action_reshaped[agent_k0_idx]
                        )
                        env_act_k_mixed = {
                            agent: action_k_reshaped[idx].squeeze()
                            for idx, agent in enumerate(env.agents)
                        }

                        # VALUE
                        # output of wrapper is (num_envs, num_agents, world_state_size)
                        # swap axes to (num_agents, num_envs, world_state_size) before reshaping to (num_actors, world_state_size)
                        world_state = last_obs["world_state"].swapaxes(0, 1)
                        world_state = world_state.reshape((config["NUM_ACTORS"], -1))

                        cr_in = (
                            world_state[None, :],
                            last_done[None, :],
                        )
                        critic_hidden_state_new, value = critic_network.apply(
                            critic_train_state.params, critic_hidden_state, cr_in
                        )

                        # STEP ENV
                        rng, _rng = jax.random.split(rng)
                        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                        obs, env_state, reward, done, info = jax.vmap(
                            env.step, in_axes=(0, 0, 0)
                        )(rng_step, env_state, env_act_k_mixed)
                        info_batched = jax.tree.map(
                            lambda x: x.reshape((config["NUM_ACTORS"])), info
                        )
                        done_batched = batchify(
                            done, env.agents, config["NUM_ACTORS"]
                        ).squeeze()
                        transition = Transition(
                            global_done=jnp.tile(
                                done["__all__"], env.num_agents
                            ),  # global done for each env
                            done=last_done,
                            action=action.squeeze(),
                            value=value.squeeze(),
                            reward=batchify(
                                reward, env.agents, config["NUM_ACTORS"]
                            ).squeeze(),
                            log_prob=log_prob.squeeze(),
                            obs=obs_batch,
                            world_state=world_state,
                            info=info_batched,
                            avail_actions=avail_actions,
                        )
                        runner_state_k = runner_state_k.replace(
                            env_state=env_state,
                            obs=obs,
                            done=done_batched,
                            actor_hidden_state=actor_hidden_state_new,
                            critic_hidden_state=critic_hidden_state_new,
                            actor_hidden_state_k=actor_hidden_state_k_new,
                            rng=rng,
                        )
                        return runner_state_k, transition

                    # trajectory with k0 actions from agent_id and k1 from others
                    runner_state_k, traj_batch_k = jax.lax.scan(
                        _env_step, runner_state_k, None, config["NUM_STEPS"]
                    )

                    # CALCULATE ADVANTAGE
                    critic_train_state = runner_state_k.critic_train_state
                    last_obs = runner_state_k.obs
                    last_done = runner_state_k.done
                    critic_hidden_state = runner_state_k.critic_hidden_state

                    last_world_state = last_obs["world_state"].swapaxes(0, 1)
                    last_world_state = last_world_state.reshape(
                        (config["NUM_ACTORS"], -1)
                    )

                    cr_in = (
                        last_world_state[None, :],
                        last_done[None, :],
                    )
                    _, last_val_k = critic_network.apply(
                        critic_train_state.params, critic_hidden_state, cr_in
                    )
                    last_val_k = last_val_k.squeeze()

                    def _calculate_gae(traj_batch, last_val):
                        def _get_advantages(gae_and_next_value, transition):
                            gae, next_value = gae_and_next_value
                            done, value, reward = (
                                transition.global_done,
                                transition.value,
                                transition.reward,
                            )
                            delta = (
                                reward
                                + config["GAMMA"] * next_value * (1 - done)
                                - value
                            )
                            gae = (
                                delta
                                + config["GAMMA"]
                                * config["GAE_LAMBDA"]
                                * (1 - done)
                                * gae
                            )
                            return (gae, value), gae

                        _, advantages = jax.lax.scan(
                            _get_advantages,
                            (
                                jnp.zeros_like(last_val),
                                last_val,
                            ),  # carry = gae (initially 0) & last value
                            traj_batch,
                            reverse=True,
                            unroll=16,
                        )
                        return advantages

                    # shape (num_actors, timesteps)
                    advantages_k = _calculate_gae(traj_batch_k, last_val_k)

                    return runner_state_k, (advantages_k, traj_batch_k)

                _, (advantages_stack, traj_batch_stack) = jax.vmap(
                    _get_advantages, in_axes=(None, 0)
                )(train_runner_state_k, jnp.arange(env.num_agents))
                # advantages_stack has shape (num_agents, num_steps, num_actors)
                # each element in traj_batch_stack has shape (num_actors, num_steps, num_actors, arr_dim)

                # mask
                agent_k0_idx_outer = jnp.arange(env.num_agents)[:, None, None, None]
                agent_k0_idx_inner = jnp.arange(env.num_agents)[None, None, :, None]
                mask = agent_k0_idx_outer == agent_k0_idx_inner
                loss_mask = jnp.broadcast_to(
                    mask,
                    (*advantages_stack.shape[:2], env.num_agents, config["NUM_ENVS"]),
                )
                loss_mask = loss_mask.reshape((*loss_mask.shape[:2], -1))

                def _update_epoch(update_state, unused):
                    def _update_minibatch(
                        train_states,
                        minibatch,
                    ):
                        actor_train_state, critic_train_state = train_states
                        (
                            actor_hidden_state_init,
                            traj_batch_stack,
                            advantages_stack,
                            loss_mask,
                        ) = minibatch

                        def _actor_total_loss(
                            actor_params, init_hstate, traj_batch, gae, loss_mask
                        ):
                            def _actor_loss(
                                actor_params, init_hstate, traj_batch, gae, loss_mask
                            ):
                                # RERUN NETWORK
                                _, pi = actor_network.apply(
                                    actor_params,
                                    init_hstate.squeeze(),
                                    (
                                        traj_batch.obs,
                                        traj_batch.done,
                                        traj_batch.avail_actions,
                                    ),
                                )
                                log_prob = pi.log_prob(traj_batch.action)

                                # CALCULATE ACTOR LOSS
                                count_mask = jax.lax.stop_gradient(loss_mask.sum())
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
                                loss_actor_all = -jnp.minimum(loss_actor1, loss_actor2)
                                loss_actor_masked = jnp.where(
                                    loss_mask > 0, loss_actor_all, 0
                                )
                                loss_actor = jnp.where(
                                    count_mask > 0,
                                    loss_actor_masked.sum() / count_mask,
                                    0,
                                )
                                entropy_all = pi.entropy()
                                entropy_masked = jnp.where(
                                    loss_mask > 0, entropy_all, 0
                                )
                                entropy = jnp.where(
                                    count_mask > 0,
                                    entropy_masked.sum() / count_mask,
                                    0,
                                )

                                # debug
                                approx_kl = (
                                    ((ratio - 1) - logratio) * loss_mask
                                ).sum() / loss_mask.sum()
                                clip_frac = (
                                    (jnp.abs(ratio - 1) > config["CLIP_EPS"])
                                    * loss_mask
                                ).sum() / loss_mask.sum()

                                actor_loss = loss_actor - config["ENT_COEF"] * entropy

                                return actor_loss, (
                                    loss_actor,
                                    entropy,
                                    ratio,
                                    approx_kl,
                                    clip_frac,
                                )

                            total_actor_loss, loss_info = jax.vmap(
                                _actor_loss, in_axes=(None, None, 0, 0, 0)
                            )(actor_params, init_hstate, traj_batch, gae, loss_mask)
                            total_loss = total_actor_loss.sum()

                            return total_loss, loss_info

                        actor_grad_fn_k = jax.value_and_grad(
                            _actor_total_loss, has_aux=True
                        )
                        actor_loss_k, actor_grads_k = actor_grad_fn_k(
                            actor_train_state.params,
                            actor_hidden_state_init,
                            traj_batch_stack,
                            advantages_stack,
                            loss_mask,
                        )

                        if config["SCALE_ACTOR_GRAD"]:
                            actor_grads_k = jax.tree.map(
                                lambda x: x / env.num_agents, actor_grads_k
                            )

                        actor_train_state = actor_train_state.apply_gradients(
                            grads=actor_grads_k
                        )

                        actor_grad_norm_k = pytree_norm(actor_grads_k)

                        loss_info_k = {
                            "actor_loss_k": actor_loss_k[0],
                            "entropy_k": actor_loss_k[1][1],
                            "ratio_k": actor_loss_k[1][2],
                            "approx_kl_k": actor_loss_k[1][3],
                            "clip_frac_k": actor_loss_k[1][4],
                            "actor_grad_norm_k": actor_grad_norm_k,
                        }

                        return (actor_train_state, critic_train_state), loss_info_k

                    actor_train_state = update_state.actor_train_state
                    critic_train_state = update_state.critic_train_state
                    traj_batch_stack = update_state.traj_batch
                    advantages_stack = update_state.advantages
                    actor_hidden_state_init = update_state.actor_hidden_state
                    critic_hidden_state_init = update_state.critic_hidden_state
                    rng = update_state.rng

                    rng, _rng = jax.random.split(rng)

                    # add two extra leading dimensions to hstates (so the NUM_ACTORS dimension is aligned for minibatching)
                    actor_hidden_state_init = actor_hidden_state_init[None, None, :]
                    critic_hidden_state_init = critic_hidden_state_init[None, None, :]

                    # batch is in sequence
                    # init_hstates has shape (1, 1, NUM_ACTORS, dim)
                    # advantages_stack has shape (num_agents, num_steps, num_actors)
                    # traj_batch_stack items have shape (num_agents, num_steps, num_actors, arr_dim)
                    batch = (
                        actor_hidden_state_init,  # (num_agents,1, NUM_ACTORS, dim)
                        traj_batch_stack,  # (num_agents, num_steps, NUM_ACTORS, arr_dim)
                        advantages_stack,  # (num_agents, num_steps, NUM_ACTORS)
                        loss_mask,  # (num_agents, 1, NUM_ACTORS)
                    )

                    # we permute among the actors (we backprop through the entire episodes)
                    permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
                    shuffled_batch = jax.tree.map(
                        lambda x: jnp.take(x, permutation, axis=2), batch
                    )
                    shuffled_batch_reshaped = jax.tree.map(
                        lambda x: jnp.reshape(  # reshapes shuffled batch into separate minibatches by adding a dimension after actor dim
                            # e.g. advantages_stack (5,128,320) -> (5,128,2,160) if NUM_MINIBATCHES = 2
                            # traj_batch_stack.obs (5,128,320,127) -> (5,128,2,160,127)
                            x,
                            list(x.shape[0:2])
                            + [config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[3:]),
                        ),
                        shuffled_batch,
                    )
                    minibatches = jax.tree.map(  # move minibatch dimension to the front
                        lambda x: jnp.moveaxis(x, 2, 0), shuffled_batch_reshaped
                    )

                    train_states, loss_info = jax.lax.scan(
                        _update_minibatch,
                        (actor_train_state, critic_train_state),
                        minibatches,
                    )
                    update_state = update_state.replace(
                        actor_train_state=train_states[0],
                        critic_train_state=train_states[1],
                        rng=rng,
                    )
                    return update_state, loss_info

                initial_update_state_k = UpdateRunnerState(
                    actor_train_state=train_runner_state_k.actor_train_state,
                    critic_train_state=train_runner_state_k.critic_train_state,
                    traj_batch=traj_batch_stack,
                    advantages=advantages_stack,
                    targets=jnp.zeros_like(advantages_stack),  # not needed
                    actor_hidden_state=actor_hidden_state_init,
                    critic_hidden_state=critic_hidden_state_init,
                    rng=train_runner_state_k.rng,
                )

                final_update_state_k, loss_info_k = jax.lax.scan(
                    _update_epoch, initial_update_state_k, None, config["UPDATE_EPOCHS"]
                )
                loss_info_k["ratio_0_k"] = loss_info_k["ratio_k"].at[0, 0, :].get()
                loss_info_k = jax.tree.map(lambda x: x.mean(), loss_info_k)

                train_runner_state_k = train_runner_state_k.replace(
                    actor_train_state=final_update_state_k.actor_train_state,
                    rng=final_update_state_k.rng,
                )
                return train_runner_state_k, loss_info_k

            rng, _train_rng_k = jax.random.split(rng)
            train_runner_state_k = TrainRunnerState(
                actor_train_state=actor_train_state,
                critic_train_state=critic_train_state,
                actor_train_state_k=actor_train_state_k,
                critic_train_state_k=critic_train_state_k,
                env_state=env_state_initial,
                obs=obs_initial,
                done=done_initial,
                actor_hidden_state=actor_hidden_state_init,
                critic_hidden_state=critic_hidden_state_init,
                actor_hidden_state_k=actor_hidden_state_init,
                critic_hidden_state_k=critic_hidden_state_init,
                update_step=update_step,
                rng=_train_rng_k,
            )
            train_runner_state_k, loss_info_k = _update_step_k2(train_runner_state_k)

            # actor is now k2, need to give critic its update back
            critic_train_state = critic_train_state.replace(params=critic_params_k1)

            actor_train_state = actor_train_state.replace(
                params=train_runner_state_k.actor_train_state.params,
                opt_state=train_runner_state_k.actor_train_state.opt_state,
            )

            # Print to wandb
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
            actor_train_state_k=actor_train_state_k,
            critic_train_state_k=critic_train_state_k,
            env_state=env_state,
            obs=obs,
            done=jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            actor_hidden_state=actor_hidden_state_init,
            critic_hidden_state=critic_hidden_state_init,
            actor_hidden_state_k=actor_hidden_state_init,
            critic_hidden_state_k=critic_hidden_state_init,
            update_step=0,
            rng=_train_rng,
        )

        # highest level runner state has 6 elements
        final_runner_state, metrics_batch = jax.lax.scan(
            _train_loop, initial_runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": final_runner_state}

    return train


@hydra.main(version_base=None, config_path="./", config_name="config")
def main(config):
    try:
        config = OmegaConf.to_container(config)

        # WANDB
        job_type = f"K2MAPPO_K0CR_SCALED_LOSS_{config['MAP_NAME']}"
        group = f"K2MAPPO_K0CR_SCALED_LOSS_{config['MAP_NAME']}"
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
