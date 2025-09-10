import os

# disable randomness
# os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import NamedTuple
import datetime
from flax.training.train_state import TrainState
from jax._src.typing import Array
import hydra
from omegaconf import OmegaConf
import jaxmarl
from jaxmarl.wrappers.baselines import MPELogWrapper
from cheap_talk.src.utils.wandb_process import WandbMultiLogger
from cheap_talk.src.networks.mlp import ActorCriticDiscreteMLP
from cheap_talk.src.utils.jax_utils import pytree_norm


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    log_prob: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    new_done: jnp.ndarray
    global_new_done: jnp.ndarray
    value: jnp.ndarray
    info: jnp.ndarray


class RunnerState(NamedTuple):
    train_state_agent: TrainState
    train_state_adversary: TrainState
    obs_agent: jnp.ndarray
    obs_adversary: jnp.ndarray
    state: jnp.ndarray
    done_agent: jnp.ndarray
    done_adversary: jnp.ndarray
    cumulative_return_agent: jnp.ndarray
    cumulative_return_adversary: jnp.ndarray
    update_step: int
    rng: Array


class Updatestate(NamedTuple):
    train_state_agent: TrainState
    train_state_adversary: TrainState
    traj_batch_agent: Transition
    traj_batch_adversary: Transition
    advantages_agent: jnp.ndarray
    advantages_adversary: jnp.ndarray
    targets_agent: jnp.ndarray
    targets_adversary: jnp.ndarray
    rng: Array


# default jnp array is (time, agent, env, action)
def batchify(x: dict, agent_list, num_actors):  # convert dict to jnp.ndarray
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(
    x: jnp.ndarray, agent_list, num_envs, num_actors
):  # convert jnp.ndarray to dict
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    # env
    env = jaxmarl.make(config["env_name"], **config["env_kwargs"])
    env = MPELogWrapper(env)

    # config
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
            - (count // (config["num_minibatches"] * config["num_updates"]))
            / config["num_updates"]
        )
        return config["lr"] * frac

    def train(rng, exp_id):
        def train_setup(rng):
            # env reset
            rng, _rng_reset = jax.random.split(rng)
            _rng_resets = jax.random.split(_rng_reset, config["num_envs"])
            obs, state = jax.vmap(env.reset, in_axes=(0))(_rng_resets)
            obs_agent_batch = batchify(obs, env.good_agents, config["num_agents"])
            obs_adversary_batch = batchify(
                obs, env.adversaries, config["num_adversaries"]
            )

            if config["anneal_lr"]:
                lr_schedule = linear_schedule
            else:
                lr_schedule = config["lr"]
            network_agent = ActorCriticDiscreteMLP(
                action_dim=env.action_space(env.good_agents[0]).n,
                activation=config["activation"],
            )
            network_adversary = ActorCriticDiscreteMLP(
                action_dim=env.action_space(env.adversaries[0]).n,
                activation=config["activation"],
            )
            rng, _rng_agent, _rng_adversary = jax.random.split(rng, 3)
            init_x_agent = jnp.zeros(env.observation_space(env.good_agents[0]).shape)
            init_x_adversary = jnp.zeros(
                env.observation_space(env.adversaries[0]).shape
            )
            network_params_agent = network_agent.init(_rng_agent, init_x_agent)
            network_params_adversary = network_adversary.init(
                _rng_adversary, init_x_adversary
            )
            if config["optimizer"] == "adam":
                optim_agent = optax.adam
                optim_adversary = optax.adam
            elif config["optimizer"] == "rmsprop":
                optim_agent = optax.rmsprop
                optim_adversary = optax.rmsprop
            elif config["optimizer"] == "sgd":
                optim_agent = optax.sgd
                optim_adversary = optax.sgd
            else:
                raise ValueError(f"Invalid optimizer: {config['optimizer']}")
            tx_agent = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optim_agent(learning_rate=lr_schedule, eps=1e-5),
            )
            tx_adversary = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optim_adversary(learning_rate=lr_schedule, eps=1e-5),
            )

            train_state_agent = TrainState.create(
                apply_fn=network_agent.apply,
                params=network_params_agent,
                tx=tx_agent,
            )
            train_state_adversary = TrainState.create(
                apply_fn=network_adversary.apply,
                params=network_params_adversary,
                tx=tx_adversary,
            )

            return (
                train_state_agent,
                train_state_adversary,
                network_agent,
                network_adversary,
                obs_agent_batch,
                obs_adversary_batch,
                state,
            )

        rng, _rng_setup = jax.random.split(rng)
        (
            train_state_agent,
            train_state_adversary,
            network_agent,
            network_adversary,
            obs_agent_batch,
            obs_adversary_batch,
            state,
        ) = train_setup(_rng_setup)

        # TRAIN LOOP
        def _update_step(carry, unused):
            runner_state, timesteps = carry

            def _env_step(runner_state, unused):
                train_state_agent = runner_state.train_state_agent
                train_state_adversary = runner_state.train_state_adversary
                obs_agent_batch = runner_state.obs_agent
                obs_adversary_batch = runner_state.obs_adversary
                state = runner_state.state
                done_agent = runner_state.done_agent
                done_adversary = runner_state.done_adversary
                rng = runner_state.rng

                # SELECT ACTION
                rng, _rng_agent, _rng_adversary = jax.random.split(rng, 3)
                pi_agent, value_agent = network_agent.apply(
                    train_state_agent.params, obs_agent_batch
                )
                pi_adversary, value_adversary = network_adversary.apply(
                    train_state_adversary.params, obs_adversary_batch
                )
                action_agent = pi_agent.sample(seed=_rng_agent)
                action_adversary = pi_adversary.sample(seed=_rng_adversary)
                log_prob_agent = pi_agent.log_prob(action_agent)
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

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["num_envs"])
                new_obs, new_state, reward, new_done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, state, env_act)
                info = jax.tree.map(lambda x: x.reshape((config["num_actors"])), info)
                new_obs_batch_agent = batchify(
                    new_obs, env.good_agents, config["num_agents"]
                )
                new_obs_batch_adversary = batchify(
                    new_obs, env.adversaries, config["num_adversaries"]
                )
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
                    train_state_agent=train_state_agent,
                    train_state_adversary=train_state_adversary,
                    obs_agent=new_obs_batch_agent,
                    obs_adversary=new_obs_batch_adversary,
                    state=new_state,
                    done_agent=done_agent,
                    done_adversary=done_adversary,
                    cumulative_return_agent=runner_state.cumulative_return_agent,
                    cumulative_return_adversary=runner_state.cumulative_return_adversary,
                    update_step=runner_state.update_step,
                    rng=rng,
                )
                return runner_state, (transition_agent, transition_adversary)

            runner_state, (traj_batch_agent, traj_batch_adversary) = jax.lax.scan(
                _env_step, runner_state, None, config["num_steps"]
            )

            # CALCULATE ADVANTAGE
            train_state_agent = runner_state.train_state_agent
            train_state_adversary = runner_state.train_state_adversary
            last_obs_agent_batch = runner_state.obs_agent
            last_obs_adversary_batch = runner_state.obs_adversary
            last_done_agent = runner_state.done_agent
            last_done_adversary = runner_state.done_adversary
            rng = runner_state.rng

            _, last_val_agent = network_agent.apply(
                train_state_agent.params, last_obs_agent_batch
            )
            _, last_val_adversary = network_adversary.apply(
                train_state_adversary.params, last_obs_adversary_batch
            )
            last_val_agent = last_val_agent.squeeze()
            last_val_adversary = last_val_adversary.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["gamma"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["gamma"] * config["gae_lambda"] * (1 - done) * gae
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
                train_state_agent = update_state.train_state_agent
                train_state_adversary = update_state.train_state_adversary
                traj_batch_agent = update_state.traj_batch_agent
                traj_batch_adversary = update_state.traj_batch_adversary
                advantages_agent = update_state.advantages_agent
                advantages_adversary = update_state.advantages_adversary
                targets_agent = update_state.targets_agent
                targets_adversary = update_state.targets_adversary
                rng = update_state.rng

                # decompose traj_batch
                obs_agent = traj_batch_agent.obs
                obs_adversary = traj_batch_adversary.obs
                value_agent = traj_batch_agent.value
                value_adversary = traj_batch_adversary.value
                action_agent = traj_batch_agent.action
                action_adversary = traj_batch_adversary.action
                log_prob_agent_k0 = traj_batch_agent.log_prob
                log_prob_adversary_k0 = traj_batch_adversary.log_prob

                rng, _rng_permute_agent, _rng_permute_adversary = jax.random.split(
                    rng, 3
                )
                permutation_agent = jax.random.permutation(
                    _rng_permute_agent, config["num_agents"]
                )
                permutation_adversary = jax.random.permutation(
                    _rng_permute_adversary, config["num_adversaries"]
                )

                # batch is in sequence
                carry = (
                    train_state_agent,
                    train_state_adversary,
                    obs_agent,
                    obs_adversary,
                    value_agent,
                    value_adversary,
                    action_agent,
                    action_adversary,
                    log_prob_agent_k0,
                    log_prob_adversary_k0,
                    advantages_agent,
                    advantages_adversary,
                    targets_agent,
                    targets_adversary,
                    permutation_agent,
                    permutation_adversary,
                )

                def _update_minibatch(carry, minibatch_idx):
                    (
                        train_state_agent,
                        train_state_adversary,
                        obs_agent,
                        obs_adversary,
                        value_agent,
                        value_adversary,
                        action_agent,
                        action_adversary,
                        log_prob_agent_k0,
                        log_prob_adversary_k0,
                        advantages_agent,
                        advantages_adversary,
                        targets_agent,
                        targets_adversary,
                        permutation_agent,
                        permutation_adversary,
                    ) = carry

                    agent_params_k0 = train_state_agent.params
                    adversary_params_k0 = train_state_adversary.params

                    # Get log_prob_k0_joint
                    log_prob_k0_all = jnp.concatenate(
                        [log_prob_agent_k0, log_prob_adversary_k0], axis=1
                    )
                    log_prob_k0_all_reshape = log_prob_k0_all.reshape(
                        -1, env.num_agents, config["num_envs"]
                    )
                    log_prob_k0_all_joint = jnp.sum(log_prob_k0_all_reshape, axis=1)
                    log_prob_k0_all_joint_agent = jnp.tile(
                        log_prob_k0_all_joint, (1, env.num_good_agents)
                    )
                    log_prob_k0_all_joint_adversary = jnp.tile(
                        log_prob_k0_all_joint, (1, env.num_adversaries)
                    )

                    batch_agent = (
                        obs_agent,
                        value_agent,
                        action_agent,
                        advantages_agent,
                        targets_agent,
                        log_prob_agent_k0,
                        log_prob_k0_all_joint_agent,
                    )

                    batch_adversary = (
                        obs_adversary,
                        value_adversary,
                        action_adversary,
                        advantages_adversary,
                        targets_adversary,
                        log_prob_adversary_k0,
                        log_prob_k0_all_joint_adversary,
                    )

                    shuffled_batch_agent = jax.tree.map(
                        lambda x: jnp.take(x, permutation_agent, axis=1), batch_agent
                    )
                    shuffled_batch_reshaped_agent = jax.tree.map(
                        lambda x: jnp.reshape(  # reshapes shuffled batch into separate minibatches by adding a dimension after actor dim
                            # e.g. advantages_stack (128,320) -> (128,2,160) if NUM_MINIBATCHES = 2
                            # traj_batch_stack.obs (128,320,127) -> (128,2,160,127)
                            x,
                            list(x.shape[0:1])  # time dimension
                            + [
                                config["num_minibatches"],
                                -1,
                            ]  # minibatch dimension, actor dimension
                            + list(x.shape[2:]),  # rest of the dimensions
                        ),
                        shuffled_batch_agent,
                    )
                    minibatches_agent = (
                        jax.tree.map(  # move minibatch dimension to the front
                            lambda x: jnp.moveaxis(x, 1, 0),
                            shuffled_batch_reshaped_agent,
                        )
                    )

                    shuffled_batch_adversary = jax.tree.map(
                        lambda x: jnp.take(x, permutation_adversary, axis=1),
                        batch_adversary,
                    )
                    shuffled_batch_reshaped_adversary = jax.tree.map(
                        lambda x: jnp.reshape(  # reshapes shuffled batch into separate minibatches by adding a dimension after actor dim
                            x,
                            list(x.shape[0:1])  # time dimension
                            + [
                                config["num_minibatches"],
                                -1,
                            ]  # minibatch dimension, actor dimension
                            + list(x.shape[2:]),  # rest of the dimensions
                        ),
                        shuffled_batch_adversary,
                    )
                    minibatches_adversary = (
                        jax.tree.map(  # move minibatch dimension to the front
                            lambda x: jnp.moveaxis(x, 1, 0),
                            shuffled_batch_reshaped_adversary,
                        )
                    )

                    minibatch_obs_agent = minibatches_agent[0][minibatch_idx]
                    minibatch_obs_adversary = minibatches_adversary[0][minibatch_idx]
                    minibatch_value_agent = minibatches_agent[1][minibatch_idx]
                    minibatch_value_adversary = minibatches_adversary[1][minibatch_idx]
                    minibatch_action_agent = minibatches_agent[2][minibatch_idx]
                    minibatch_action_adversary = minibatches_adversary[2][minibatch_idx]
                    minibatch_advantages_agent = minibatches_agent[3][minibatch_idx]
                    minibatch_advantages_adversary = minibatches_adversary[3][
                        minibatch_idx
                    ]
                    minibatch_targets_agent = minibatches_agent[4][minibatch_idx]
                    minibatch_targets_adversary = minibatches_adversary[4][
                        minibatch_idx
                    ]
                    minibatch_log_prob_agent_k0 = minibatches_agent[5][minibatch_idx]
                    minibatch_log_prob_adversary_k0 = minibatches_adversary[5][
                        minibatch_idx
                    ]
                    minibatch_log_prob_k0_all_joint_agent = minibatches_agent[6][
                        minibatch_idx
                    ]
                    minibatch_log_prob_k0_all_joint_adversary = minibatches_adversary[
                        6
                    ][minibatch_idx]

                    def _loss_fn_k1(
                        params,
                        network,
                        obs,
                        action,
                        gae,
                        value_old,
                        targets,
                        log_prob_k0,
                    ):
                        # RERUN NETWORK
                        pi, value = network.apply(
                            params,
                            obs,
                        )
                        log_prob = pi.log_prob(action)

                        # actor loss
                        logratio = log_prob - log_prob_k0
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor_1 = ratio * gae
                        loss_actor_2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["clip_eps"],
                                1.0 + config["clip_eps"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor_1, loss_actor_2).mean()
                        entropy = pi.entropy().mean()

                        # critic loss
                        value_pred_clipped = value_old + (value - value_old).clip(
                            -config["clip_eps"], config["clip_eps"]
                        )
                        value_loss = jnp.square(value - targets)
                        value_loss_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_loss, value_loss_clipped).mean()
                        )

                        # stats
                        approx_kl_backward = ((ratio - 1) - logratio).mean()
                        approx_kl_forward = (ratio * logratio - (ratio - 1)).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["clip_eps"])

                        total_loss = (
                            loss_actor
                            + config["vf_coef"] * value_loss
                            - config["ent_coef"] * entropy
                        )

                        return total_loss, {
                            "total_loss": total_loss,
                            "value_loss": value_loss,
                            "actor_loss": loss_actor,
                            "entropy": entropy,
                            "ratio": ratio,
                            "approx_kl_backward": approx_kl_backward,
                            "approx_kl_forward": approx_kl_forward,
                            "clip_frac": clip_frac,
                            "gae_mean": gae.mean(),
                            "gae_std": gae.std(),
                            "gae_max": gae.max(),
                        }

                    grad_fn_k1 = jax.value_and_grad(_loss_fn_k1, has_aux=True)
                    (loss_agent_k1, loss_info_agent_k1), grads_agent_k1 = grad_fn_k1(
                        agent_params_k0,
                        network_agent,
                        minibatch_obs_agent,
                        minibatch_action_agent,
                        minibatch_advantages_agent,
                        minibatch_value_agent,
                        minibatch_targets_agent,
                        minibatch_log_prob_agent_k0,
                    )
                    (loss_adversary_k1, loss_info_adversary_k1), grads_adversary_k1 = (
                        grad_fn_k1(
                            adversary_params_k0,
                            network_adversary,
                            minibatch_obs_adversary,
                            minibatch_action_adversary,
                            minibatch_advantages_adversary,
                            minibatch_value_adversary,
                            minibatch_targets_adversary,
                            minibatch_log_prob_adversary_k0,
                        )
                    )
                    updated_train_state_agent = train_state_agent.apply_gradients(
                        grads=grads_agent_k1
                    )
                    updated_train_state_adversary = (
                        train_state_adversary.apply_gradients(grads=grads_adversary_k1)
                    )
                    total_loss_k1 = loss_agent_k1 + loss_adversary_k1
                    loss_info_agent_k1["grad_norm"] = pytree_norm(grads_agent_k1)
                    loss_info_adversary_k1["grad_norm"] = pytree_norm(
                        grads_adversary_k1
                    )
                    loss_info_agent_k1 = {
                        k + "_agent_k1": v for k, v in loss_info_agent_k1.items()
                    }
                    loss_info_adversary_k1 = {
                        k + "_adversary_k1": v
                        for k, v in loss_info_adversary_k1.items()
                    }
                    loss_info_k1 = {**loss_info_agent_k1, **loss_info_adversary_k1}
                    loss_info_k1["total_loss"] = total_loss_k1

                    # get k1 probs
                    agent_params_k1 = updated_train_state_agent.params
                    adversary_params_k1 = updated_train_state_adversary.params
                    pi_k1_agent, _ = network_agent.apply(agent_params_k1, obs_agent)
                    pi_k1_adversary, _ = network_adversary.apply(
                        adversary_params_k1, obs_adversary
                    )
                    log_prob_k1_agent = pi_k1_agent.log_prob(action_agent)
                    log_prob_k1_adversary = pi_k1_adversary.log_prob(action_adversary)
                    log_prob_k1_all = jnp.concatenate(
                        [log_prob_k1_agent, log_prob_k1_adversary], axis=1
                    )
                    log_prob_k1_all_reshape = log_prob_k1_all.reshape(
                        -1, env.num_agents, config["num_envs"]
                    )
                    log_prob_k1_all_joint = jnp.sum(log_prob_k1_all_reshape, axis=1)
                    log_prob_k1_all_joint_agent = jnp.tile(
                        log_prob_k1_all_joint, (1, env.num_good_agents)
                    )
                    log_prob_k1_all_joint_adversary = jnp.tile(
                        log_prob_k1_all_joint, (1, env.num_adversaries)
                    )
                    batch_agent_k2 = (log_prob_k1_agent, log_prob_k1_all_joint_agent)
                    batch_adversary_k2 = (
                        log_prob_k1_adversary,
                        log_prob_k1_all_joint_adversary,
                    )

                    shuffled_batch_agent_k2 = jax.tree.map(
                        lambda x: jnp.take(x, permutation_agent, axis=1), batch_agent_k2
                    )
                    shuffled_batch_adversary_k2 = jax.tree.map(
                        lambda x: jnp.take(x, permutation_adversary, axis=1),
                        batch_adversary_k2,
                    )
                    shuffled_batch_reshaped_agent_k2 = jax.tree.map(
                        lambda x: jnp.reshape(  # reshapes shuffled batch into separate minibatches by adding a dimension after actor dim
                            x,
                            list(x.shape[0:1])  # time dimension
                            + [config["num_minibatches"], -1]
                            + list(x.shape[2:]),
                        ),
                        shuffled_batch_agent_k2,
                    )
                    shuffled_batch_reshaped_adversary_k2 = jax.tree.map(
                        lambda x: jnp.reshape(  # reshapes shuffled batch into separate minibatches by adding a dimension after actor dim
                            x,
                            list(x.shape[0:1])  # time dimension
                            + [config["num_minibatches"], -1]
                            + list(x.shape[2:]),
                        ),
                        shuffled_batch_adversary_k2,
                    )
                    minibatches_agent_k2 = (
                        jax.tree.map(  # move minibatch dimension to the front
                            lambda x: jnp.moveaxis(x, 1, 0),
                            shuffled_batch_reshaped_agent_k2,
                        )
                    )
                    minibatches_adversary_k2 = (
                        jax.tree.map(  # move minibatch dimension to the front
                            lambda x: jnp.moveaxis(x, 1, 0),
                            shuffled_batch_reshaped_adversary_k2,
                        )
                    )

                    minibatch_log_prob_k1_agent = minibatches_agent_k2[0][minibatch_idx]
                    minibatch_log_prob_k1_all_joint_agent = minibatches_agent_k2[1][
                        minibatch_idx
                    ]
                    minibatch_log_prob_k1_adversary = minibatches_adversary_k2[0][
                        minibatch_idx
                    ]
                    minibatch_log_prob_k1_all_joint_adversary = (
                        minibatches_adversary_k2[1][minibatch_idx]
                    )

                    def _loss_fn_k2(
                        params,
                        network,
                        obs,
                        action,
                        gae,
                        value_old,
                        targets,
                        log_prob_k0_joint,
                        log_prob_k1,
                        log_prob_k1_joint,
                    ):
                        # RERUN NETWORK
                        pi, value = network.apply(
                            params,
                            obs,
                        )
                        log_prob = pi.log_prob(action)

                        # actor loss
                        logratio_is = (
                            log_prob
                            + log_prob_k1_joint
                            - log_prob_k0_joint
                            - log_prob_k1
                        )
                        ratio_is = jnp.exp(logratio_is)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor_1 = ratio_is * gae
                        loss_actor_2 = (
                            jnp.clip(
                                ratio_is,
                                1.0 - config["clip_eps"],
                                1.0 + config["clip_eps"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor_1, loss_actor_2).mean()
                        entropy = pi.entropy().mean()

                        # critic loss
                        value_pred_clipped = value_old + (value - value_old).clip(
                            -config["clip_eps"], config["clip_eps"]
                        )
                        value_loss = jnp.square(value - targets)
                        value_loss_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_loss, value_loss_clipped).mean()
                        )

                        # stats
                        approx_kl_backward = ((ratio_is - 1) - logratio_is).mean()
                        approx_kl_forward = (
                            ratio_is * logratio_is - (ratio_is - 1)
                        ).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio_is - 1) > config["clip_eps"])

                        total_loss = (
                            loss_actor
                            + config["vf_coef"] * value_loss
                            - config["ent_coef"] * entropy
                        )

                        return total_loss, {
                            "total_loss": total_loss,
                            "value_loss": value_loss,
                            "actor_loss": loss_actor,
                            "entropy": entropy,
                            "ratio_is": ratio_is,
                            "approx_kl_backward": approx_kl_backward,
                            "approx_kl_forward": approx_kl_forward,
                            "clip_frac": clip_frac,
                            "gae_mean": gae.mean(),
                            "gae_std": gae.std(),
                            "gae_max": gae.max(),
                        }

                    grad_fn_k2 = jax.value_and_grad(_loss_fn_k2, has_aux=True)
                    (loss_agent_k2, loss_info_agent_k2), grads_agent_k2 = grad_fn_k2(
                        agent_params_k0,
                        network_agent,
                        minibatch_obs_agent,
                        minibatch_action_agent,
                        minibatch_advantages_agent,
                        minibatch_value_agent,
                        minibatch_targets_agent,
                        minibatch_log_prob_k0_all_joint_agent,
                        minibatch_log_prob_k1_agent,
                        minibatch_log_prob_k1_all_joint_agent,
                    )
                    (loss_adversary_k2, loss_info_adversary_k2), grads_adversary_k2 = (
                        grad_fn_k2(
                            adversary_params_k0,
                            network_adversary,
                            minibatch_obs_adversary,
                            minibatch_action_adversary,
                            minibatch_advantages_adversary,
                            minibatch_value_adversary,
                            minibatch_targets_adversary,
                            minibatch_log_prob_k0_all_joint_adversary,
                            minibatch_log_prob_k1_adversary,
                            minibatch_log_prob_k1_all_joint_adversary,
                        )
                    )
                    updated_train_state_agent = train_state_agent.apply_gradients(
                        grads=grads_agent_k2
                    )
                    updated_train_state_adversary = (
                        train_state_adversary.apply_gradients(grads=grads_adversary_k2)
                    )
                    total_loss_k2 = loss_agent_k2 + loss_adversary_k2
                    loss_info_agent_k2["grad_norm"] = pytree_norm(grads_agent_k2)
                    loss_info_adversary_k2["grad_norm"] = pytree_norm(
                        grads_adversary_k2
                    )
                    loss_info_agent_k2 = {
                        k + "_agent_k2": v for k, v in loss_info_agent_k2.items()
                    }
                    loss_info_adversary_k2 = {
                        k + "_adversary_k2": v
                        for k, v in loss_info_adversary_k2.items()
                    }
                    loss_info_k2 = {**loss_info_agent_k2, **loss_info_adversary_k2}
                    loss_info_k2["total_loss"] = total_loss_k2

                    return (
                        updated_train_state_agent,
                        updated_train_state_adversary,
                        obs_agent,
                        obs_adversary,
                        value_agent,
                        value_adversary,
                        action_agent,
                        action_adversary,
                        log_prob_agent_k0,
                        log_prob_adversary_k0,
                        advantages_agent,
                        advantages_adversary,
                        targets_agent,
                        targets_adversary,
                        permutation_agent,
                        permutation_adversary,
                    ), (loss_info_k1, loss_info_k2)

                (
                    final_train_state_agent,
                    final_train_state_adversary,
                    obs_agent,
                    obs_adversary,
                    value_agent,
                    value_adversary,
                    action_agent,
                    action_adversary,
                    log_prob_agent_k0,
                    log_prob_adversary_k0,
                    advantages_agent,
                    advantages_adversary,
                    targets_agent,
                    targets_adversary,
                    permutation_agent,
                    permutation_adversary,
                ), (loss_info_k1, loss_info_k2) = jax.lax.scan(
                    _update_minibatch,
                    carry,
                    jnp.arange(config["num_minibatches"]),
                )
                update_state = Updatestate(
                    train_state_agent=final_train_state_agent,
                    train_state_adversary=final_train_state_adversary,
                    traj_batch_agent=traj_batch_agent,
                    traj_batch_adversary=traj_batch_adversary,
                    advantages_agent=advantages_agent,
                    advantages_adversary=advantages_adversary,
                    targets_agent=targets_agent,
                    targets_adversary=targets_adversary,
                    rng=rng,
                )
                return update_state, (loss_info_k1, loss_info_k2)

            update_state = Updatestate(
                train_state_agent=train_state_agent,
                train_state_adversary=train_state_adversary,
                traj_batch_agent=traj_batch_agent,
                traj_batch_adversary=traj_batch_adversary,
                advantages_agent=advantages_agent,
                advantages_adversary=advantages_adversary,
                targets_agent=targets_agent,
                targets_adversary=targets_adversary,
                rng=rng,
            )
            final_update_state, (loss_info_k1, loss_info_k2) = jax.lax.scan(
                _update_epoch, update_state, None, config["num_updates"]
            )
            train_state_agent = final_update_state.train_state_agent
            train_state_adversary = final_update_state.train_state_adversary
            rng = final_update_state[-1]

            loss_info_k1_mean = jax.tree.map(lambda x: x.mean(), loss_info_k1)
            loss_info_k2_mean = jax.tree.map(lambda x: x.mean(), loss_info_k2)

            # log returns
            def _returns(carry_return, inputs):
                reward, done = inputs
                cumulative_return = carry_return + reward
                reset_return = jnp.zeros(reward.shape[1:], dtype=float)
                carry_return = jnp.where(done, reset_return, cumulative_return)
                return carry_return, cumulative_return

            # agent
            reward_agent = traj_batch_agent.reward
            done_agent = traj_batch_agent.new_done
            cumulative_return_agent = runner_state.cumulative_return_agent

            new_cumulative_return_agent, returns = jax.lax.scan(
                _returns,
                cumulative_return_agent,
                (reward_agent, done_agent),
            )
            only_returns = jnp.where(done_agent, returns, 0)
            returns_avg_agent = jnp.where(
                done_agent.sum() > 0, only_returns.sum() / done_agent.sum(), 0.0
            )

            # adversary
            reward_adversary = traj_batch_adversary.reward
            done_adversary = traj_batch_adversary.new_done
            cumulative_return_adversary = runner_state.cumulative_return_adversary

            new_cumulative_return_adversary, returns = jax.lax.scan(
                _returns,
                cumulative_return_adversary,
                (reward_adversary, done_adversary),
            )
            only_returns = jnp.where(done_adversary, returns, 0)
            returns_avg_adversary = jnp.where(
                done_adversary.sum() > 0, only_returns.sum() / done_adversary.sum(), 0.0
            )

            # log episode lengths
            global_new_done = traj_batch_agent.global_new_done

            def _episode_lengths(carry_length, done):
                cumulative_length = carry_length + 1
                reset_length = jnp.zeros(done.shape[1:], dtype=jnp.int32)
                carry_length = jnp.where(done, reset_length, cumulative_length)
                return carry_length, cumulative_length

            # agent
            timesteps, lengths = jax.lax.scan(
                _episode_lengths,
                timesteps,
                global_new_done,
            )
            only_episode_ends = jnp.where(
                global_new_done, lengths, 0
            )  # only lengths at done steps
            episode_length_avg = jnp.where(
                global_new_done.sum() > 0,
                only_episode_ends.sum() / global_new_done.sum(),
                0.0,
            )

            log_dict = {}
            log_dict = {**loss_info_k1_mean, **loss_info_k2_mean}
            log_dict["update_step"] = runner_state.update_step
            log_dict["env_step"] = (
                runner_state.update_step * config["num_envs"] * config["num_steps"]
            )
            log_dict["samples"] = (
                runner_state.update_step * config["num_steps"] * config["num_actors"]
            )
            log_dict["returns_agent"] = returns_avg_agent
            log_dict["returns_adversary"] = returns_avg_adversary
            log_dict["return_total"] = returns_avg_agent + returns_avg_adversary
            log_dict["episode_length"] = episode_length_avg

            if config["log_network_stats"]:
                # log network stats
                network_leaves_agent = jax.tree.leaves(
                    update_state.train_state_agent.params
                )
                network_leaves_adversary = jax.tree.leaves(
                    update_state.train_state_adversary.params
                )
                flat_network_agent = jnp.concatenate(
                    [jnp.ravel(x) for x in network_leaves_agent]
                )
                flat_network_adversary = jnp.concatenate(
                    [jnp.ravel(x) for x in network_leaves_adversary]
                )
                network_l1_agent = jnp.sum(jnp.abs(flat_network_agent))
                network_l2_agent = jnp.linalg.norm(flat_network_agent)
                network_linfty_agent = jnp.max(jnp.abs(flat_network_agent))
                network_mu_agent = jnp.mean(flat_network_agent)
                network_std_agent = jnp.std(flat_network_agent)
                network_max_agent = jnp.max(flat_network_agent)
                network_min_agent = jnp.min(flat_network_agent)
                network_l1_adversary = jnp.sum(jnp.abs(flat_network_adversary))
                network_l2_adversary = jnp.linalg.norm(flat_network_adversary)
                network_linfty_adversary = jnp.max(jnp.abs(flat_network_adversary))
                network_mu_adversary = jnp.mean(flat_network_adversary)
                network_std_adversary = jnp.std(flat_network_adversary)
                network_max_adversary = jnp.max(flat_network_adversary)
                network_min_adversary = jnp.min(flat_network_adversary)

                log_dict["network_l1_agent"] = network_l1_agent
                log_dict["network_l2_agent"] = network_l2_agent
                log_dict["network_linfty_agent"] = network_linfty_agent
                log_dict["network_mu_agent"] = network_mu_agent
                log_dict["network_std_agent"] = network_std_agent
                log_dict["network_max_agent"] = network_max_agent
                log_dict["network_min_agent"] = network_min_agent
                log_dict["network_l1_adversary"] = network_l1_adversary
                log_dict["network_l2_adversary"] = network_l2_adversary
                log_dict["network_linfty_adversary"] = network_linfty_adversary
                log_dict["network_mu_adversary"] = network_mu_adversary
                log_dict["network_std_adversary"] = network_std_adversary
                log_dict["network_max_adversary"] = network_max_adversary
                log_dict["network_min_adversary"] = network_min_adversary

            def callback(exp_id, log_dict):
                np_log_dict = {k: np.array(v) for k, v in log_dict.items()}
                LOGGER.log(int(exp_id), np_log_dict)

            jax.experimental.io_callback(callback, None, exp_id, log_dict)
            runner_state = RunnerState(
                train_state_agent=train_state_agent,
                train_state_adversary=train_state_adversary,
                obs_agent=last_obs_agent_batch,
                obs_adversary=last_obs_adversary_batch,
                state=runner_state.state,
                done_agent=last_done_agent,
                done_adversary=last_done_adversary,
                cumulative_return_agent=new_cumulative_return_agent,
                cumulative_return_adversary=new_cumulative_return_adversary,
                update_step=runner_state.update_step + 1,
                rng=rng,
            )
            return (runner_state, timesteps), log_dict

        rng, _rng = jax.random.split(rng)
        initial_timesteps = jnp.zeros((config["num_envs"]), dtype=int)
        initial_runner_state = RunnerState(
            train_state_agent=train_state_agent,
            train_state_adversary=train_state_adversary,
            obs_agent=obs_agent_batch,
            obs_adversary=obs_adversary_batch,
            state=state,
            done_agent=jnp.zeros((config["num_agents"]), dtype=bool),
            done_adversary=jnp.zeros((config["num_adversaries"]), dtype=bool),
            cumulative_return_agent=jnp.zeros((config["num_agents"]), dtype=float),
            cumulative_return_adversary=jnp.zeros(
                (config["num_adversaries"]), dtype=float
            ),
            update_step=0,
            rng=_rng,
        )
        final_runner_state, log_dict = jax.lax.scan(
            _update_step,
            (initial_runner_state, initial_timesteps),
            None,
            config["num_updates"],
        )
        return final_runner_state, log_dict

    return train


@hydra.main(version_base=None, config_path="./", config_name="config_ik2p_mlp")
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
