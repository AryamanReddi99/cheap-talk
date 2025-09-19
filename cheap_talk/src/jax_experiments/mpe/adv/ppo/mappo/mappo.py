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
from omegaconf import OmegaConf
from functools import partial
import jaxmarl
from jaxmarl.wrappers.baselines import MPELogWrapper, JaxMARLWrapper
from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State
from cheap_talk.src.utils.wandb_process import WandbMultiLogger
import functools
from cheap_talk.src.utils.jax_utils import pytree_norm, pytree_diff_norm


class MPEWorldStateWrapper(JaxMARLWrapper):

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        obs["world_state_agent"] = self.world_state_agent(obs)
        obs["world_state_adversary"] = self.world_state_adversary(obs)
        return obs, env_state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(key, state, action)
        obs["world_state_agent"] = self.world_state_agent(obs)
        obs["world_state_adversary"] = self.world_state_adversary(obs)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def world_state_agent(self, obs):
        """
        For each agent: [agent obs, all other agent obs]
        """

        @partial(jax.vmap, in_axes=(0, None))
        def _roll_obs(aidx, all_obs):
            robs = jnp.roll(all_obs, -aidx, axis=0)
            robs = robs.flatten()
            return robs

        all_obs = jnp.array([obs[agent] for agent in self._env.good_agents]).flatten()
        all_obs = jnp.expand_dims(all_obs, axis=0).repeat(
            self._env.num_good_agents, axis=0
        )
        return all_obs

    @partial(jax.jit, static_argnums=0)
    def world_state_adversary(self, obs):
        """
        For each agent: [agent obs, all other agent obs]
        """

        @partial(jax.vmap, in_axes=(0, None))
        def _roll_obs(aidx, all_obs):
            robs = jnp.roll(all_obs, -aidx, axis=0)
            robs = robs.flatten()
            return robs

        all_obs = jnp.array([obs[agent] for agent in self._env.adversaries]).flatten()
        all_obs = jnp.expand_dims(all_obs, axis=0).repeat(
            self._env.num_adversaries, axis=0
        )
        return all_obs

    def world_state_size(self):
        spaces_agent = [
            self._env.observation_space(agent) for agent in self._env.good_agents
        ]
        spaces_adversary = [
            self._env.observation_space(agent) for agent in self._env.adversaries
        ]
        return sum([space.shape[-1] for space in spaces_agent]), sum(
            [space.shape[-1] for space in spaces_adversary]
        )


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
            self.config["fc_dim_size"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["gru_hidden_dim"],
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
            self.config["fc_dim_size"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(world_state)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        critic = nn.Dense(
            self.config["gru_hidden_dim"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    log_prob: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    new_done: jnp.ndarray
    global_done: jnp.ndarray
    value: jnp.ndarray
    info: jnp.ndarray


class RunnerState(NamedTuple):
    train_state_agent_actor: TrainState
    train_state_agent_critic: TrainState
    train_state_adversary_actor: TrainState
    train_state_adversary_critic: TrainState
    obs: Dict[str, jnp.ndarray]
    state: jnp.ndarray
    done_agent: jnp.ndarray
    done_adversary: jnp.ndarray
    hstate_agent_actor: jnp.ndarray
    hstate_adversary_actor: jnp.ndarray
    hstate_agent_critic: jnp.ndarray
    hstate_adversary_critic: jnp.ndarray
    cumulative_return_agent: jnp.ndarray
    cumulative_return_adversary: jnp.ndarray
    update_step: int
    rng: Array


class Updatestate(NamedTuple):
    train_state_agent_actor: TrainState
    train_state_agent_critic: TrainState
    train_state_adversary_actor: TrainState
    train_state_adversary_critic: TrainState
    traj_batch_agent: Transition
    traj_batch_adversary: Transition
    advantages_agent: jnp.ndarray
    advantages_adversary: jnp.ndarray
    targets_agent: jnp.ndarray
    targets_adversary: jnp.ndarray
    hstate_agent_actor: jnp.ndarray
    hstate_adversary_actor: jnp.ndarray
    hstate_agent_critic: jnp.ndarray
    hstate_adversary_critic: jnp.ndarray
    rng: Array


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    env = jaxmarl.make(config["env_name"], **config["env_kwargs"])
    env = MPEWorldStateWrapper(env)
    env = MPELogWrapper(env)
    config["num_actors"] = env.num_agents * config["num_envs"]
    config["num_agents"] = env.num_good_agents * config["num_envs"]
    config["num_adversaries"] = env.num_adversaries * config["num_envs"]
    config["num_updates"] = (
        config["total_timesteps"] // config["num_steps"] // config["num_envs"]
    )
    config["minibatch_size"] = (
        config["num_actors"] * config["num_steps"] // config["num_minibatches"]
    )
    config["clip_eps"] = (
        config["clip_eps"] / env.num_agents
        if config["scale_clip_eps"]
        else config["clip_eps"]
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["num_minibatches"] * config["update_epochs"]))
            / config["num_updates"]
        )
        return config["lr"] * frac

    def train(rng, exp_id):
        jax.debug.print("Compile Finished. Running...")

        # networks
        (
            rng,
            _rng_actor_agent,
            _rng_critic_agent,
            _rng_actor_adversary,
            _rng_critic_adversary,
        ) = jax.random.split(rng, 5)

        # agent actor network
        actor_network_agent = ActorRNN(
            env.action_space(env.good_agents[0]).n, config=config
        )
        ac_init_hstate_agent = ScannedRNN.initialize_carry(
            config["num_envs"], config["gru_hidden_dim"]
        )
        ac_init_x_agent = (
            jnp.zeros(
                (
                    1,
                    config["num_envs"],
                    env.observation_space(env.good_agents[0]).shape[0],
                )
            ),
            jnp.zeros((1, config["num_envs"])),
        )

        actor_network_params_agent = actor_network_agent.init(
            _rng_actor_agent, ac_init_hstate_agent, ac_init_x_agent
        )

        # agent critic network
        critic_network_agent = CriticRNN(config=config)
        cr_init_hstate_agent = ScannedRNN.initialize_carry(
            config["num_envs"], config["gru_hidden_dim"]
        )
        cr_init_x_agent = (
            jnp.zeros(
                (
                    1,
                    config["num_envs"],
                    env.world_state_size()[0],
                )
            ),
            jnp.zeros((1, config["num_envs"])),
        )
        critic_network_params_agent = critic_network_agent.init(
            _rng_critic_agent, cr_init_hstate_agent, cr_init_x_agent
        )

        # adversary actor network
        actor_network_adversary = ActorRNN(
            env.action_space(env.adversaries[0]).n, config=config
        )
        ac_init_hstate_adversary = ScannedRNN.initialize_carry(
            config["num_envs"], config["gru_hidden_dim"]
        )
        ac_init_x_adversary = (
            jnp.zeros(
                (
                    1,
                    config["num_envs"],
                    env.observation_space(env.adversaries[0]).shape[0],
                )
            ),
            jnp.zeros((1, config["num_envs"])),
        )

        actor_network_params_adversary = actor_network_adversary.init(
            _rng_actor_adversary, ac_init_hstate_adversary, ac_init_x_adversary
        )

        # adversary critic network
        critic_network_adversary = CriticRNN(config=config)
        cr_init_hstate_adversary = ScannedRNN.initialize_carry(
            config["num_envs"], config["gru_hidden_dim"]
        )
        cr_init_x_adversary = (
            jnp.zeros(
                (1, config["num_envs"], env.world_state_size()[0]),
            ),
            jnp.zeros((1, config["num_envs"])),
        )
        critic_network_params_adversary = critic_network_adversary.init(
            _rng_critic_adversary, cr_init_hstate_adversary, cr_init_x_adversary
        )

        if config["anneal_lr"]:
            actor_tx_agent = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx_agent = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            actor_tx_adversary = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx_adversary = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )

        actor_train_state_agent = TrainState.create(
            apply_fn=actor_network_agent.apply,
            params=actor_network_params_agent,
            tx=actor_tx_agent,
        )
        critic_train_state_agent = TrainState.create(
            apply_fn=critic_network_agent.apply,
            params=critic_network_params_agent,
            tx=critic_tx_agent,
        )
        actor_train_state_adversary = TrainState.create(
            apply_fn=actor_network_adversary.apply,
            params=actor_network_params_adversary,
            tx=actor_tx_adversary,
        )
        critic_train_state_adversary = TrainState.create(
            apply_fn=critic_network_adversary.apply,
            params=critic_network_params_adversary,
            tx=critic_tx_adversary,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["num_envs"])
        obs, state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            def _env_step(runner_state, unused):
                train_state_agent_actor = runner_state.train_state_agent_actor
                train_state_agent_critic = runner_state.train_state_agent_critic
                train_state_adversary_actor = runner_state.train_state_adversary_actor
                train_state_adversary_critic = runner_state.train_state_adversary_critic
                obs = runner_state.obs
                state = runner_state.state
                done_agent = runner_state.done_agent
                done_adversary = runner_state.done_adversary
                hstate_agent_actor = runner_state.hstate_agent_actor
                hstate_adversary_actor = runner_state.hstate_adversary_actor
                hstate_agent_critic = runner_state.hstate_agent_critic
                hstate_adversary_critic = runner_state.hstate_adversary_critic
                rng = runner_state.rng

                # SELECT ACTION
                rng, _rng_agent, _rng_adversary = jax.random.split(rng, 3)
                obs_agent_batch = batchify(obs, env.good_agents, config["num_agents"])
                ac_in_agent = (
                    obs_agent_batch[np.newaxis, :],
                    done_agent[np.newaxis, :],
                )
                ac_hstate_agent, pi_agent = actor_network_agent.apply(
                    train_state_agent_actor.params, hstate_agent_actor, ac_in_agent
                )
                action_agent = pi_agent.sample(seed=_rng_agent)
                log_prob_agent = pi_agent.log_prob(action_agent)

                obs_adversary_batch = batchify(
                    obs, env.adversaries, config["num_adversaries"]
                )
                ac_in_adversary = (
                    obs_adversary_batch[np.newaxis, :],
                    done_adversary[np.newaxis, :],
                )
                ac_hstate_adversary, pi_adversary = actor_network_adversary.apply(
                    train_state_adversary_actor.params,
                    hstate_adversary_actor,
                    ac_in_adversary,
                )
                action_adversary = pi_adversary.sample(seed=_rng_adversary)
                log_prob_adversary = pi_adversary.log_prob(action_adversary)
                env_act_agent = unbatchify(
                    action_agent,
                    env.good_agents,
                    config["num_envs"],
                    env.num_good_agents,
                )
                env_act_adversary = unbatchify(
                    action_adversary,
                    env.adversaries,
                    config["num_envs"],
                    env.num_adversaries,
                )
                env_act_agent = {k: v.squeeze() for k, v in env_act_agent.items()}
                env_act_adversary = {
                    k: v.squeeze() for k, v in env_act_adversary.items()
                }
                env_act = {**env_act_agent, **env_act_adversary}

                # VALUE
                # output of wrapper is (num_envs, num_agents, world_state_size)
                # swap axes to (num_agents, num_envs, world_state_size) before reshaping to (num_actors, world_state_size)
                world_state_agent = obs["world_state_agent"].swapaxes(0, 1)
                world_state_agent = world_state_agent.reshape(
                    (config["num_agents"], -1)
                )
                world_state_adversary = obs["world_state_adversary"].swapaxes(0, 1)
                world_state_adversary = world_state_adversary.reshape(
                    (config["num_adversaries"], -1)
                )

                cr_in_agent = (
                    world_state_agent[None, :],
                    done_agent[np.newaxis, :],
                )
                cr_hstate_agent, value_agent = critic_network_agent.apply(
                    train_state_agent_critic.params, hstate_agent_critic, cr_in_agent
                )
                cr_in_adversary = (
                    world_state_adversary[None, :],
                    done_adversary[np.newaxis, :],
                )
                cr_hstate_adversary, value_adversary = critic_network_adversary.apply(
                    train_state_adversary_critic.params,
                    hstate_adversary_critic,
                    cr_in_adversary,
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["num_envs"])
                new_obs, new_state, reward, new_done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, state, env_act)
                info = jax.tree.map(lambda x: x.reshape((config["num_actors"])), info)
                new_done_batch_agent = batchify(
                    new_done, env.good_agents, config["num_agents"]
                ).squeeze()
                new_done_batch_adversary = batchify(
                    new_done, env.adversaries, config["num_adversaries"]
                ).squeeze()
                transition_agent = Transition(
                    obs=obs_agent_batch,
                    action=action_agent.squeeze(),
                    log_prob=log_prob_agent.squeeze(),
                    reward=batchify(
                        reward, env.good_agents, config["num_agents"]
                    ).squeeze(),
                    done=done_agent,
                    new_done=new_done_batch_agent,
                    global_new_done=new_done["__all__"],
                    value=value_agent.squeeze(),
                    info=info,
                )
                transition_adversary = Transition(
                    obs=obs_adversary_batch,
                    action=action_adversary.squeeze(),
                    log_prob=log_prob_adversary.squeeze(),
                    reward=batchify(
                        reward, env.adversaries, config["num_adversaries"]
                    ).squeeze(),
                    done=done_adversary,
                    new_done=new_done_batch_adversary,
                    global_new_done=new_done["__all__"],
                    value=value_adversary.squeeze(),
                    info=info,
                )
                runner_state = RunnerState(
                    train_state_agent_actor=train_state_agent_actor,
                    train_state_agent_critic=train_state_agent_critic,
                    train_state_adversary_actor=train_state_adversary_actor,
                    train_state_adversary_critic=train_state_adversary_critic,
                    obs=new_obs,
                    state=new_state,
                    done_agent=done_agent,
                    done_adversary=done_adversary,
                    hstate_agent_actor=ac_hstate_agent,
                    hstate_adversary_actor=ac_hstate_adversary,
                    hstate_agent_critic=cr_hstate_agent,
                    hstate_adversary_critic=cr_hstate_adversary,
                    cumulative_return_agent=runner_state.cumulative_return_agent,
                    cumulative_return_adversary=runner_state.cumulative_return_adversary,
                    update_step=runner_state.update_step,
                    rng=rng,
                )
                return runner_state, (transition_agent, transition_adversary)

            initial_hstate_agent_actor = runner_state.hstate_agent_actor
            initial_hstate_adversary_actor = runner_state.hstate_adversary_actor
            initial_hstate_agent_critic = runner_state.hstate_agent_critic
            initial_hstate_adversary_critic = runner_state.hstate_adversary_critic
            runner_state, (traj_batch_agent, traj_batch_adversary) = jax.lax.scan(
                _env_step, runner_state, None, config["num_steps"]
            )

            # CALCULATE ADVANTAGE
            train_state_agent_actor = runner_state.train_state_agent_actor
            train_state_adversary_actor = runner_state.train_state_adversary_actor
            train_state_agent_critic = runner_state.train_state_agent_critic
            train_state_adversary_critic = runner_state.train_state_adversary_critic
            last_hstate_agent_critic = runner_state.hstate_agent_critic
            last_hstate_adversary_critic = runner_state.hstate_adversary_critic
            last_done_agent = runner_state.done_agent
            last_done_adversary = runner_state.done_adversary
            last_obs = runner_state.obs
            last_state_agent = last_obs["world_state_agent"].swapaxes(0, 1)
            last_state_agent = last_state_agent.reshape((config["num_agents"], -1))

            last_state_adversary = last_obs["world_state_adversary"].swapaxes(0, 1)
            last_state_adversary = last_state_adversary.reshape(
                (config["num_adversaries"], -1)
            )

            cr_in_agent = (
                last_state_agent[None, :],
                last_done_agent[np.newaxis, :],
            )
            _, last_val_agent = critic_network_agent.apply(
                train_state_agent_critic.params, last_hstate_agent_critic, cr_in_agent
            )
            last_val_agent = last_val_agent.squeeze()

            cr_in_adversary = (
                last_state_adversary[None, :],
                last_done_adversary[np.newaxis, :],
            )
            _, last_val_adversary = critic_network_adversary.apply(
                train_state_adversary_critic.params,
                last_hstate_adversary_critic,
                cr_in_adversary,
            )
            last_val_adversary = last_val_adversary.squeeze()

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

            advantages_agent, targets_agent = _calculate_gae(
                traj_batch_agent, last_val_agent
            )
            advantages_adversary, targets_adversary = _calculate_gae(
                traj_batch_adversary, last_val_adversary
            )

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                train_state_agent_actor = update_state.train_state_agent_actor
                train_state_agent_critic = update_state.train_state_agent_critic
                train_state_adversary_actor = update_state.train_state_adversary_actor
                train_state_adversary_critic = update_state.train_state_adversary_critic
                traj_batch_agent = update_state.traj_batch_agent
                traj_batch_adversary = update_state.traj_batch_adversary
                advantages_agent = update_state.advantages_agent
                advantages_adversary = update_state.advantages_adversary
                targets_agent = update_state.targets_agent
                targets_adversary = update_state.targets_adversary
                hstate_agent_actor = update_state.hstate_agent_actor
                hstate_adversary_actor = update_state.hstate_adversary_actor
                hstate_agent_critic = update_state.hstate_agent_critic
                hstate_adversary_critic = update_state.hstate_adversary_critic
                rng = update_state.rng

                rng, _rng_permute_agent, _rng_permute_adversary = jax.random.split(
                    rng, 3
                )
                permutation_agent = jax.random.permutation(
                    _rng_permute_agent, config["num_agents"]
                )
                batch_agent = (
                    traj_batch_agent.obs,
                    traj_batch_agent.action,
                    traj_batch_agent.log_prob,
                    traj_batch_agent.value,
                    traj_batch_agent.done,
                    advantages_agent.squeeze(),
                    targets_agent.squeeze(),
                    hstate_agent_actor.squeeze(),
                    hstate_agent_critic.squeeze(),
                )
                shuffled_batch_agent = jax.tree.map(
                    lambda x: jnp.take(x, permutation_agent, axis=1), batch_agent
                )
                shuffled_batch_split_agent = jax.tree.map(
                    lambda x: jnp.reshape(  # split into minibatches along actor dimension (dim 1)
                        x,
                        [x.shape[0], config["num_minibatches"], -1] + list(x.shape[2:]),
                    ),
                    shuffled_batch_agent,
                )
                minibatches_agent = jax.tree.map(  # swap minibatch and time axis,
                    lambda x: jnp.swapaxes(x, 0, 1),
                    shuffled_batch_split_agent,
                )

                permutation_adversary = jax.random.permutation(
                    _rng_permute_adversary, config["num_adversaries"]
                )
                batch_adversary = (
                    traj_batch_adversary.obs,
                    traj_batch_adversary.action,
                    traj_batch_adversary.log_prob,
                    traj_batch_adversary.value,
                    traj_batch_adversary.done,
                    advantages_adversary.squeeze(),
                    targets_adversary.squeeze(),
                    hstate_adversary_actor.squeeze(),
                    hstate_adversary_critic.squeeze(),
                )
                shuffled_batch_adversary = jax.tree.map(
                    lambda x: jnp.take(x, permutation_adversary, axis=1),
                    batch_adversary,
                )
                shuffled_batch_split_adversary = jax.tree.map(
                    lambda x: jnp.reshape(  # split into minibatches along actor dimension (dim 1)
                        x,
                        [x.shape[0], config["num_minibatches"], -1] + list(x.shape[2:]),
                    ),
                    shuffled_batch_adversary,
                )
                minibatches_adversary = jax.tree.map(  # swap minibatch and time axis,
                    lambda x: jnp.swapaxes(x, 0, 1),
                    shuffled_batch_split_adversary,
                )

                def _update_minibatch(carry, minibatch):
                    (
                        train_state_agent_actor,
                        train_state_agent_critic,
                        train_state_adversary_actor,
                        train_state_adversary_critic,
                    ) = carry
                    minibatch_agent, minibatch_adversary = minibatch
                    (
                        obs_minibatch_agent,
                        action_minibatch_agent,
                        log_prob_minibatch_agent,
                        value_minibatch_agent,
                        done_minibatch_agent,
                        advantages_minibatch_agent,
                        targets_minibatch_agent,
                        hstate_agent_actor_minibatch,
                        hstate_agent_critic_minibatch,
                    ) = minibatch_agent
                    (
                        obs_minibatch_adversary,
                        action_minibatch_adversary,
                        log_prob_minibatch_adversary,
                        value_minibatch_adversary,
                        done_minibatch_adversary,
                        advantages_minibatch_adversary,
                        targets_minibatch_adversary,
                        hstate_adversary_actor_minibatch,
                        hstate_adversary_critic_minibatch,
                    ) = minibatch_adversary

                    def _loss_fn_agent_actor(
                        agent_actor_params,
                        obs_minibatch,
                        action_minibatch,
                        log_prob_minibatch,
                        gae_minibatch,
                        done_minibatch,
                        hstate_minibatch,
                    ):
                        # RERUN NETWORK
                        pi = actor_network_agent.apply(
                            agent_actor_params,
                            hstate_minibatch.squeeze(),
                            (obs_minibatch, done_minibatch),
                        )
                        log_prob = pi.log_prob(action_minibatch)

                        # actor loss
                        logratio = log_prob - log_prob_minibatch
                        ratio = jnp.exp(logratio)
                        gae_normalized = (gae_minibatch - gae_minibatch.mean()) / (
                            gae_minibatch.std() + 1e-8
                        )
                        loss_actor_1 = ratio * gae_normalized
                        loss_actor_2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["clip_eps"],
                                1.0 + config["clip_eps"],
                            )
                            * gae_normalized
                        )
                        loss_actor = -jnp.minimum(loss_actor_1, loss_actor_2).mean()
                        entropy = pi.entropy().mean()

                        # stats
                        approx_kl_backward = ((ratio - 1) - logratio).mean()
                        approx_kl_forward = (ratio * logratio - (ratio - 1)).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["clip_eps"])

                        actor_loss_kl_reg = loss_actor - config["ent_coef"] * entropy

                        return actor_loss_kl_reg, {
                            "actor_loss": loss_actor,
                            "actor_loss_kl_reg": actor_loss_kl_reg,
                            "entropy": entropy,
                            "ratio": ratio,
                            "approx_kl_backward": approx_kl_backward,
                            "approx_kl_forward": approx_kl_forward,
                            "clip_frac": clip_frac,
                            "gae_mean": gae_minibatch.mean(),
                            "gae_std": gae_minibatch.std(),
                            "gae_max": gae_minibatch.max(),
                            "gae_norm_mean": gae_normalized.mean(),
                            "gae_norm_max": gae_normalized.max(),
                            "gae_norm_neg": (gae_normalized < 0).mean(),
                        }

                    def _loss_fn_agent_critic(
                        agent_critic_params,
                        obs_minibatch,
                        done_minibatch,
                        value_minibatch,
                        targets_minibatch,
                        hstate_minibatch,
                    ):
                        # RERUN NETWORK
                        value = critic_network_agent.apply(
                            agent_critic_params,
                            hstate_minibatch.squeeze(),
                            (obs_minibatch, done_minibatch),
                        )
                        value_loss = jnp.square(value - targets_minibatch)
                        value_pred_clipped = value_minibatch + (
                            value - value_minibatch
                        ).clip(-config["clip_eps"], config["clip_eps"])
                        value_loss_clipped = jnp.square(
                            value_pred_clipped - targets_minibatch
                        )
                        value_loss = (
                            0.5 * jnp.maximum(value_loss, value_loss_clipped).mean()
                        )

                        critic_loss = config["vf_coef"] * value_loss
                        return critic_loss, {
                            "critic_loss": critic_loss,
                        }

                    def _loss_fn_adversary_actor(
                        adversary_actor_params,
                        obs_minibatch,
                        action_minibatch,
                        log_prob_minibatch,
                        done_minibatch,
                        gae_minibatch,
                        hstate_minibatch,
                    ):
                        # RERUN NETWORK
                        pi = actor_network_adversary.apply(
                            adversary_actor_params,
                            hstate_minibatch.squeeze(),
                            (obs_minibatch, done_minibatch),
                        )
                        log_prob = pi.log_prob(action_minibatch)

                        # actor loss
                        logratio = log_prob - log_prob_minibatch
                        ratio = jnp.exp(logratio)
                        gae_normalized = (gae_minibatch - gae_minibatch.mean()) / (
                            gae_minibatch.std() + 1e-8
                        )
                        loss_actor_1 = ratio * gae_normalized
                        loss_actor_2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["clip_eps"],
                                1.0 + config["clip_eps"],
                            )
                            * gae_normalized
                        )
                        loss_actor = -jnp.minimum(loss_actor_1, loss_actor_2).mean()
                        entropy = pi.entropy().mean()

                        # stats
                        approx_kl_backward = ((ratio - 1) - logratio).mean()
                        approx_kl_forward = (ratio * logratio - (ratio - 1)).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["clip_eps"])

                        actor_loss_kl_reg = loss_actor - config["ent_coef"] * entropy

                        return actor_loss_kl_reg, {
                            "actor_loss": loss_actor,
                            "actor_loss_kl_reg": actor_loss_kl_reg,
                            "entropy": entropy,
                            "ratio": ratio,
                            "approx_kl_backward": approx_kl_backward,
                            "approx_kl_forward": approx_kl_forward,
                            "clip_frac": clip_frac,
                            "gae_mean": gae_minibatch.mean(),
                            "gae_std": gae_minibatch.std(),
                            "gae_max": gae_minibatch.max(),
                            "gae_norm_mean": gae_normalized.mean(),
                            "gae_norm_max": gae_normalized.max(),
                            "gae_norm_neg": (gae_normalized < 0).mean(),
                        }

                    def _loss_fn_adversary_critic(
                        adversary_critic_params,
                        obs_minibatch,
                        done_minibatch,
                        value_minibatch,
                        targets_minibatch,
                        hstate_minibatch,
                    ):
                        # RERUN NETWORK
                        value = critic_network_adversary.apply(
                            adversary_critic_params,
                            hstate_minibatch.squeeze(),
                            (obs_minibatch, done_minibatch),
                        )
                        value_loss = jnp.square(value - targets_minibatch)
                        value_pred_clipped = value_minibatch + (
                            value - value_minibatch
                        ).clip(-config["clip_eps"], config["clip_eps"])
                        value_loss_clipped = jnp.square(
                            value_pred_clipped - targets_minibatch
                        )
                        value_loss = (
                            0.5 * jnp.maximum(value_loss, value_loss_clipped).mean()
                        )
                        critic_loss = config["vf_coef"] * value_loss
                        return critic_loss, {
                            "critic_loss": critic_loss,
                        }

                    grad_fn_agent_actor = jax.value_and_grad(
                        _loss_fn_agent_actor, has_aux=True
                    )
                    (loss_agent_actor, loss_info_agent_actor), grads_agent_actor = (
                        grad_fn_agent_actor(
                            train_state_agent_actor.params,
                            obs_minibatch_agent,
                            action_minibatch_agent,
                            log_prob_minibatch_agent,
                            done_minibatch_agent,
                            advantages_minibatch_agent,
                            hstate_agent_actor_minibatch,
                        )
                    )
                    grad_fn_agent_critic = jax.value_and_grad(
                        _loss_fn_agent_critic, has_aux=True
                    )
                    (loss_agent_critic, loss_info_agent_critic), grads_agent_critic = (
                        grad_fn_agent_critic(
                            train_state_agent_critic.params,
                            obs_minibatch_agent,
                            done_minibatch_agent,
                            value_minibatch_agent,
                            targets_minibatch_agent,
                            hstate_agent_critic_minibatch,
                        )
                    )
                    grad_fn_adversary_actor = jax.value_and_grad(
                        _loss_fn_adversary_actor, has_aux=True
                    )
                    (
                        loss_adversary_actor,
                        loss_info_adversary_actor,
                    ), grads_adversary_actor = grad_fn_adversary_actor(
                        train_state_adversary_actor.params,
                        obs_minibatch_adversary,
                        action_minibatch_adversary,
                        log_prob_minibatch_adversary,
                        done_minibatch_adversary,
                        advantages_minibatch_adversary,
                        hstate_adversary_actor_minibatch,
                    )
                    grad_fn_adversary_critic = jax.value_and_grad(
                        _loss_fn_adversary_critic, has_aux=True
                    )
                    (
                        loss_adversary_critic,
                        loss_info_adversary_critic,
                    ), grads_adversary_critic = grad_fn_adversary_critic(
                        train_state_adversary_critic.params,
                        obs_minibatch_adversary,
                        done_minibatch_adversary,
                        value_minibatch_adversary,
                        targets_minibatch_adversary,
                        hstate_adversary_critic_minibatch,
                    )
                    updated_train_state_agent_actor = (
                        train_state_agent_actor.apply_gradients(grads=grads_agent_actor)
                    )
                    updated_train_state_agent_critic = (
                        train_state_agent_critic.apply_gradients(
                            grads=grads_agent_critic
                        )
                    )
                    updated_train_state_adversary_actor = (
                        train_state_adversary_actor.apply_gradients(
                            grads=grads_adversary_actor
                        )
                    )
                    updated_train_state_adversary_critic = (
                        train_state_adversary_critic.apply_gradients(
                            grads=grads_adversary_critic
                        )
                    )
                    total_loss = (
                        loss_agent_actor
                        + loss_agent_critic
                        + loss_adversary_actor
                        + loss_adversary_critic
                    )
                    loss_info_agent_actor["grad_norm"] = pytree_norm(grads_agent_actor)
                    loss_info_agent_critic["grad_norm"] = pytree_norm(
                        grads_agent_critic
                    )
                    loss_info_adversary_actor["grad_norm"] = pytree_norm(
                        grads_adversary_actor
                    )
                    loss_info_adversary_critic["grad_norm"] = pytree_norm(
                        grads_adversary_critic
                    )
                    loss_info_agent_actor["distance"] = pytree_diff_norm(
                        updated_train_state_agent_actor.params,
                        train_state_agent_actor.params,
                    )
                    loss_info_agent_critic["distance"] = pytree_diff_norm(
                        updated_train_state_agent_critic.params,
                        train_state_agent_critic.params,
                    )
                    loss_info_adversary_actor["distance"] = pytree_diff_norm(
                        updated_train_state_adversary_actor.params,
                        train_state_adversary_actor.params,
                    )
                    loss_info_adversary_critic["distance"] = pytree_diff_norm(
                        updated_train_state_adversary_critic.params,
                        train_state_adversary_critic.params,
                    )

                    # new ratios
                    pi_new_agent = actor_network_agent.apply(
                        updated_train_state_agent_actor.params,
                        hstate_agent_actor_minibatch.squeeze(),
                        (obs_minibatch_agent, done_minibatch_agent),
                    )
                    log_prob_new_agent = pi_new_agent.log_prob(action_minibatch_agent)
                    ratio_new_agent = jnp.exp(
                        log_prob_new_agent - log_prob_minibatch_agent
                    )
                    loss_info_agent_actor["ratio_new"] = ratio_new_agent
                    pi_new_adversary = actor_network_adversary.apply(
                        updated_train_state_adversary_actor.params,
                        hstate_adversary_actor_minibatch.squeeze(),
                        (obs_minibatch_adversary, done_minibatch_adversary),
                    )
                    log_prob_new_adversary = pi_new_adversary.log_prob(
                        action_minibatch_adversary
                    )
                    ratio_new_adversary = jnp.exp(
                        log_prob_new_adversary - log_prob_minibatch_adversary
                    )
                    loss_info_adversary_actor["ratio_new"] = ratio_new_adversary

                    loss_info_agent_actor = {
                        k + "_agent_actor": v for k, v in loss_info_agent_actor.items()
                    }
                    loss_info_agent_critic = {
                        k + "_agent_critic": v
                        for k, v in loss_info_agent_critic.items()
                    }
                    loss_info_adversary_actor = {
                        k + "_adversary_actor": v
                        for k, v in loss_info_adversary_actor.items()
                    }
                    loss_info_adversary_critic = {
                        k + "_adversary_critic": v
                        for k, v in loss_info_adversary_critic.items()
                    }
                    loss_info = {
                        **loss_info_agent_actor,
                        **loss_info_agent_critic,
                        **loss_info_adversary_actor,
                        **loss_info_adversary_critic,
                    }
                    loss_info["total_loss"] = total_loss

                    return (
                        updated_train_state_agent_actor,
                        updated_train_state_agent_critic,
                        updated_train_state_adversary_actor,
                        updated_train_state_adversary_critic,
                    ), (loss_info)

                (
                    final_train_state_agent_actor,
                    final_train_state_agent_critic,
                    final_train_state_adversary_actor,
                    final_train_state_adversary_critic,
                ), (loss_info) = jax.lax.scan(
                    _update_minibatch,
                    (
                        train_state_agent_actor,
                        train_state_agent_critic,
                        train_state_adversary_actor,
                        train_state_adversary_critic,
                    ),
                    (minibatches_agent, minibatches_adversary),
                )
                update_state = Updatestate(
                    train_state_agent_actor=final_train_state_agent_actor,
                    train_state_agent_critic=final_train_state_agent_critic,
                    train_state_adversary_actor=final_train_state_adversary_actor,
                    train_state_adversary_critic=final_train_state_adversary_critic,
                    traj_batch_agent=traj_batch_agent,
                    traj_batch_adversary=traj_batch_adversary,
                    advantages_agent=advantages_agent,
                    advantages_adversary=advantages_adversary,
                    targets_agent=targets_agent,
                    targets_adversary=targets_adversary,
                    hstate_agent_actor=initial_hstate_agent_actor,
                    hstate_adversary_actor=initial_hstate_adversary_actor,
                    hstate_agent_critic=initial_hstate_agent_critic,
                    hstate_adversary_critic=initial_hstate_adversary_critic,
                    rng=rng,
                )
                return update_state, loss_info

            update_state = Updatestate(
                train_state_agent_actor=train_state_agent_actor,
                train_state_agent_critic=train_state_agent_critic,
                train_state_adversary_actor=train_state_adversary_actor,
                train_state_adversary_critic=train_state_adversary_critic,
                traj_batch_agent=traj_batch_agent,
                traj_batch_adversary=traj_batch_adversary,
                advantages_agent=advantages_agent,
                advantages_adversary=advantages_adversary,
                targets_agent=targets_agent,
                targets_adversary=targets_adversary,
                hstate_agent_actor=initial_hstate_agent_actor,
                hstate_adversary_actor=initial_hstate_adversary_actor,
                hstate_agent_critic=initial_hstate_agent_critic,
                hstate_adversary_critic=initial_hstate_adversary_critic,
                rng=rng,
            )
            final_update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["update_epochs"]
            )
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            metric_agent = traj_batch_agent.info
            metric_agent = jax.tree.map(
                lambda x: x.reshape(
                    (config["num_steps"], config["num_envs"], config["num_agents"]),
                ),
                metric_agent,
            )
            metric_adversary = traj_batch_adversary.info
            metric_adversary = jax.tree.map(
                lambda x: x.reshape(
                    (
                        config["num_steps"],
                        config["num_envs"],
                        config["num_adversaries"],
                    ),
                ),
                metric_adversary,
            )
            metric = {**metric_agent, **metric_adversary}
            metric["loss"] = loss_info
            metric["update_step"] = runner_state.update_step

            def callback(exp_id, metric):
                log_dict = {
                    "returns": metric["returned_episode_returns"][-1, :].mean(),
                    "env_step": metric["update_steps"]
                    * config["num_envs"]
                    * config["num_steps"],
                    **metric["loss"],
                }
                np_log_dict = {k: np.array(v) for k, v in log_dict.items()}
                LOGGER.log(int(exp_id), np_log_dict)

            jax.experimental.io_callback(callback, None, exp_id, metric)

            final_runner_state = RunnerState(
                train_state_agent_actor=final_update_state.train_state_agent_actor,
                train_state_agent_critic=final_update_state.train_state_agent_critic,
                train_state_adversary_actor=final_update_state.train_state_adversary_actor,
                train_state_adversary_critic=final_update_state.train_state_adversary_critic,
                obs=last_obs,
                state=runner_state.state,
                done_agent=last_done_agent,
                done_adversary=last_done_adversary,
                cumulative_return_agent=runner_state.cumulative_return_agent,
                cumulative_return_adversary=runner_state.cumulative_return_adversary,
                update_step=runner_state.update_step + 1,
                rng=final_update_state.rng,
            )
            return final_runner_state, metric

        rng, _rng = jax.random.split(rng)
        initial_runner_state = RunnerState(
            train_state_agent_actor=actor_train_state_agent,
            train_state_agent_critic=critic_train_state_agent,
            train_state_adversary_actor=actor_train_state_adversary,
            train_state_adversary_critic=critic_train_state_adversary,
            obs=obs,
            state=state,
            done_agent=jnp.zeros((config["num_agents"]), dtype=bool),
            done_adversary=jnp.zeros((config["num_adversaries"]), dtype=bool),
            hstate_agent_actor=ac_init_hstate_agent,
            hstate_adversary_actor=ac_init_hstate_adversary,
            hstate_agent_critic=cr_init_hstate_agent,
            hstate_adversary_critic=cr_init_hstate_adversary,
            cumulative_return_agent=jnp.zeros((config["num_agents"]), dtype=float),
            cumulative_return_adversary=jnp.zeros(
                (config["num_adversaries"]), dtype=float
            ),
            update_step=0,
            rng=_rng,
        )
        final_runner_state, log_dict = jax.lax.scan(
            _update_step, initial_runner_state, None, config["num_updates"]
        )
        return final_runner_state, log_dict

    return train


@hydra.main(version_base=None, config_path="./", config_name="config_mappo")
def main(config):
    try:
        # vmap and compile
        config = OmegaConf.to_container(config)
        rng = jax.random.PRNGKey(config["seed"])
        rng_seeds = jax.random.split(rng, config["num_seeds"])
        exp_ids = jnp.arange(config["num_seeds"])

        print("Starting compile...")
        train_vmap = jax.vmap(make_train(config))
        train_vjit = jax.block_until_ready(jax.jit(train_vmap))
        print("Compile finished...")

        # wandb
        job_type = f"{config['job_type']}_{config['env_name']}"
        group = f"{config['env_name']}" + datetime.datetime.now().strftime(
            "_%Y-%m-%d_%H-%M-%S"
        )
        global LOGGER
        LOGGER = WandbMultiLogger(
            project=config["project"],
            group=group,
            job_type=job_type,
            config=config,
            mode=(lambda: "online" if config["wandb_mode"] else "disabled")(),
            seed=config["seed"],
            num_seeds=config["num_seeds"],
        )

        # run
        print("Running...")
        out = jax.block_until_ready(train_vjit(rng_seeds, exp_ids))
    finally:
        LOGGER.finish()
        print("Finished.")


if __name__ == "__main__":
    main()
