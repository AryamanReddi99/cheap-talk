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
from typing import Sequence, NamedTuple, Any, Tuple, Union, Dict
import wandb
import functools
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import DictConfig, OmegaConf
from functools import partial

from jaxmarl.wrappers.baselines import SMAXLogWrapper, JaxMARLWrapper
from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX
from cheap_talk.src.jax_experiments.utils.wandb_process import WandbMultiLogger

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
        rnn_state = carry
        ins, resets = x
        # resets go from shape (1, num_actors) to (1, num_actors, 1)
        # rnn_state and ins are (num_actors, hidden_dim)
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


def make_train(config):
    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = (
        env.num_agents * config["NUM_ENVS"]
    )  # total number of actors across parallel envs
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )  # this is the x-axis on wandb
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )

    env = SMAXWorldStateWrapper(env, config["OBS_WITH_AGENT_ID"])
    env = SMAXLogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng, seed):
        def callback(seed, metric):
            log_dict = {
                "return": metric["returned_episode_returns"][:, :, 0][
                    metric["returned_episode"][:, :, 0]
                ].mean(),
                "win_rate": metric["returned_won_episode"][:, :, 0][
                    metric["returned_episode"][:, :, 0]
                ].mean(),
                "env_step": metric["update_steps"]
                * config["NUM_ENVS"]
                * config["NUM_STEPS"],
                **metric["loss"],
            }
            np_log_dict = {k: np.array(v) for k, v in log_dict.items()}
            LOGGER.log(int(seed), np_log_dict)

        # INIT NETWORK
        actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)
        critic_network = CriticRNN(config=config)
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        ac_init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n)),
        )
        ac_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)
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
            config["NUM_ACTORS"], 128
        )  # 2d shape with (num_actors, hidden dim)
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], 128)

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            #     runner_state = (
            #     (actor_train_state, critic_train_state),  # network train states
            #     env_state,  # env state
            #     obsv,  # observations
            #     jnp.zeros((config["NUM_ACTORS"]), dtype=bool),  # dones
            #     (ac_init_hstate, cr_init_hstate),  # hidden states for actor & critic
            #     _rng,  # rng for the entire training loop
            # )

            def _env_step(runner_state, unused):

                # runner_state is a single object that gets passed by the scan back to _env_step at each loop
                train_states, env_state, last_obs, last_done, hstates, rng = (
                    runner_state
                )

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[
                        np.newaxis, :
                    ],  # add new dim to obs_batch (1, num_actors, arr_dim)
                    last_done[
                        np.newaxis, :
                    ],  # adds new dim, i.e. becomes 2d (1, num_actors)
                    avail_actions,
                )

                # network.apply just means we use the __call__ method of nn.Module we overwrite in the network child class
                ac_hstate, pi = actor_network.apply(
                    train_states[0].params, hstates[0], ac_in
                )
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

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
                    jnp.tile(
                        done["__all__"], env.num_agents
                    ),  # global done for one env
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

            # runner_state has 6 elements
            # traj_batch is list of transitions
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
                    (
                        jnp.zeros_like(last_val),
                        last_val,
                    ),  # carry = gae (initially 0) & last value
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)
            # shape (num_actors)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = (
                        batch_info
                    )

                    def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae):
                        # RERUN NETWORK
                        _, pi = actor_network.apply(
                            actor_params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done, traj_batch.avail_actions),
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
                        critic_params, init_hstate, traj_batch, targets
                    ):
                        # RERUN NETWORK
                        _, value = critic_network.apply(
                            critic_params,
                            init_hstate.squeeze(),
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
                        actor_train_state.params, ac_init_hstate, traj_batch, advantages
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params, cr_init_hstate, traj_batch, targets
                    )

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
                    }

                    return (actor_train_state, critic_train_state), loss_info

                (
                    train_states,
                    init_hstates,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                init_hstates = jax.tree.map(
                    lambda x: jnp.reshape(x, (1, config["NUM_ACTORS"], -1)),
                    init_hstates,
                )

                batch = (
                    init_hstates[0],
                    init_hstates[1],
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                # train_states = (actor_train_state, critic_train_state)
                train_states, loss_info = jax.lax.scan(
                    _update_minbatch, train_states, minibatches
                )
                update_state = (
                    train_states,
                    jax.tree.map(lambda x: x.squeeze(), init_hstates),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info

            # update_state has 6 elements
            update_state = (
                train_states,
                initial_hstates,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            loss_info["ratio_0"] = loss_info["ratio"].at[0, 0].get()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            train_states = update_state[0]
            metric = traj_batch.info
            metric = jax.tree.map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)
                ),
                traj_batch.info,
            )
            metric["loss"] = loss_info
            rng = update_state[-1]

            metric["update_steps"] = update_steps

            # metric items: loss
            jax.experimental.io_callback(callback, None, seed, metric)
            update_steps = update_steps + 1
            runner_state = (train_states, env_state, last_obs, last_done, hstates, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),  # network train states
            env_state,  # env state
            obsv,  # observations (still in env format), shape (num_envs, arr_dim)
            jnp.zeros(
                (config["NUM_ACTORS"]), dtype=bool
            ),  # dones are in shape (num_actors)
            (ac_init_hstate, cr_init_hstate),  # hidden states for actor & critic
            _rng,  # rng for the entire training loop
        )

        # highest level runner state has 6 elements
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
        group = f"MAPPO_{config['MAP_NAME']}"
        if config["USE_TIMESTAMP"]:
            group += datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
        global LOGGER
        LOGGER = WandbMultiLogger(
            project=config["PROJECT"],
            group=group,
            config=config,
            mode=config["WANDB_MODE"],
            num_seeds=config["NUM_SEEDS"],
        )

        rng = jax.random.PRNGKey(config["SEED"])
        rng_seeds = jax.random.split(rng, config["NUM_SEEDS"])
        with jax.disable_jit(False):
            train_jit = jax.jit(make_train(config))
            out = jax.vmap(train_jit)(rng_seeds, jnp.arange(config["NUM_SEEDS"]))
    finally:
        LOGGER.finish()
        print("Finished training.")


if __name__ == "__main__":
    main()
