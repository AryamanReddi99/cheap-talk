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
from cheap_talk.src.utils.jax_utils import pytree_norm, pytree_diff_norm


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

                rng, _rng_permute_agent, _rng_permute_adversary = jax.random.split(
                    rng, 3
                )
                permutation_agent = jax.random.permutation(
                    _rng_permute_agent, config["num_agents"]
                )
                permutation_adversary = jax.random.permutation(
                    _rng_permute_adversary, config["num_adversaries"]
                )
                batch_agent = (
                    traj_batch_agent.obs,
                    traj_batch_agent.action,
                    traj_batch_agent.log_prob,
                    traj_batch_agent.value,
                    advantages_agent.squeeze(),
                    targets_agent.squeeze(),
                )
                batch_adversary = (
                    traj_batch_adversary.obs,
                    traj_batch_adversary.action,
                    traj_batch_adversary.log_prob,
                    traj_batch_adversary.value,
                    advantages_adversary.squeeze(),
                    targets_adversary.squeeze(),
                )
                shuffled_batch_agent = jax.tree.map(
                    lambda x: jnp.take(x, permutation_agent, axis=1), batch_agent
                )
                shuffled_batch_adversary = jax.tree.map(
                    lambda x: jnp.take(x, permutation_adversary, axis=1),
                    batch_adversary,
                )
                shuffled_batch_split_agent = jax.tree.map(
                    lambda x: jnp.reshape(  # split into minibatches along actor dimension (dim 1)
                        x,
                        [x.shape[0], config["num_minibatches"], -1] + list(x.shape[2:]),
                    ),
                    shuffled_batch_agent,
                )
                shuffled_batch_split_adversary = jax.tree.map(
                    lambda x: jnp.reshape(  # split into minibatches along actor dimension (dim 1)
                        x,
                        [x.shape[0], config["num_minibatches"], -1] + list(x.shape[2:]),
                    ),
                    shuffled_batch_adversary,
                )
                minibatches_agent = jax.tree.map(  # swap minibatch and time axis,
                    lambda x: jnp.swapaxes(x, 0, 1),
                    shuffled_batch_split_agent,
                )
                minibatches_adversary = jax.tree.map(  # swap minibatch and time axis,
                    lambda x: jnp.swapaxes(x, 0, 1),
                    shuffled_batch_split_adversary,
                )

                def _update_minibatch(carry, minibatch):
                    (
                        train_state_agent,
                        train_state_adversary,
                    ) = carry
                    minibatch_agent, minibatch_adversary = minibatch
                    (
                        obs_minibatch_agent,
                        action_minibatch_agent,
                        log_prob_minibatch_agent,
                        value_minibatch_agent,
                        advantages_minibatch_agent,
                        targets_minibatch_agent,
                    ) = minibatch_agent
                    (
                        obs_minibatch_adversary,
                        action_minibatch_adversary,
                        log_prob_minibatch_adversary,
                        value_minibatch_adversary,
                        advantages_minibatch_adversary,
                        targets_minibatch_adversary,
                    ) = minibatch_adversary

                    def _loss_fn_agent(
                        agent_params,
                        obs_minibatch,
                        action_minibatch,
                        log_prob_minibatch,
                        value_minibatch,
                        gae_minibatch,
                        targets_minibatch,
                    ):
                        # RERUN NETWORK
                        pi, value = network_agent.apply(
                            agent_params,
                            obs_minibatch,
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

                        # critic loss
                        value_pred_clipped = value_minibatch + (
                            value - value_minibatch
                        ).clip(-config["clip_eps"], config["clip_eps"])
                        value_loss = 0.5 * jnp.square(value - targets_minibatch)
                        value_loss_clipped = jnp.square(
                            value_pred_clipped - targets_minibatch
                        )
                        # value_loss = (
                        #     0.5 * jnp.maximum(value_loss, value_loss_clipped).mean()
                        # )

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
                            "gae_mean": gae_minibatch.mean(),
                            "gae_std": gae_minibatch.std(),
                            "gae_max": gae_minibatch.max(),
                            "gae_norm_mean": gae_normalized.mean(),
                            "gae_norm_max": gae_normalized.max(),
                            "gae_norm_neg": (gae_normalized < 0).mean(),
                        }

                    def _loss_fn_adversary(
                        adversary_params,
                        obs_minibatch,
                        action_minibatch,
                        log_prob_minibatch,
                        value_minibatch,
                        gae_minibatch,
                        targets_minibatch,
                    ):
                        # RERUN NETWORK
                        pi, value = network_adversary.apply(
                            adversary_params,
                            obs_minibatch,
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

                        # critic loss
                        value_pred_clipped = value_minibatch + (
                            value - value_minibatch
                        ).clip(-config["clip_eps"], config["clip_eps"])
                        value_loss = jnp.square(value - targets_minibatch)
                        value_loss_clipped = jnp.square(
                            value_pred_clipped - targets_minibatch
                        )
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
                            "gae_mean": gae_minibatch.mean(),
                            "gae_std": gae_minibatch.std(),
                            "gae_max": gae_minibatch.max(),
                            "gae_norm_mean": gae_normalized.mean(),
                            "gae_norm_max": gae_normalized.max(),
                            "gae_norm_neg": (gae_normalized < 0).mean(),
                        }

                    grad_fn_agent = jax.value_and_grad(_loss_fn_agent, has_aux=True)
                    (loss_agent, loss_info_agent), grads_agent = grad_fn_agent(
                        train_state_agent.params,
                        obs_minibatch_agent,
                        action_minibatch_agent,
                        log_prob_minibatch_agent,
                        value_minibatch_agent,
                        advantages_minibatch_agent,
                        targets_minibatch_agent,
                    )
                    grad_fn_adversay = jax.value_and_grad(
                        _loss_fn_adversary, has_aux=True
                    )
                    (loss_adversary, loss_info_adversary), grads_adversary = (
                        grad_fn_adversay(
                            train_state_adversary.params,
                            obs_minibatch_adversary,
                            action_minibatch_adversary,
                            log_prob_minibatch_adversary,
                            value_minibatch_adversary,
                            advantages_minibatch_adversary,
                            targets_minibatch_adversary,
                        )
                    )
                    updated_train_state_agent = train_state_agent.apply_gradients(
                        grads=grads_agent
                    )
                    updated_train_state_adversary = (
                        train_state_adversary.apply_gradients(grads=grads_adversary)
                    )
                    total_loss = loss_agent + loss_adversary
                    loss_info_agent["grad_norm"] = pytree_norm(grads_agent)
                    loss_info_adversary["grad_norm"] = pytree_norm(grads_adversary)
                    loss_info_agent["distance"] = pytree_diff_norm(
                        updated_train_state_agent.params, train_state_agent.params
                    )
                    loss_info_adversary["distance"] = pytree_diff_norm(
                        updated_train_state_adversary.params,
                        train_state_adversary.params,
                    )

                    # new ratios
                    pi_new_agent, _ = network_agent.apply(
                        updated_train_state_agent.params,
                        obs_minibatch_agent,
                    )
                    log_prob_new_agent = pi_new_agent.log_prob(action_minibatch_agent)
                    ratio_new_agent = jnp.exp(
                        log_prob_new_agent - log_prob_minibatch_agent
                    )
                    loss_info_agent["ratio_new"] = ratio_new_agent
                    pi_new_adversary, _ = network_adversary.apply(
                        updated_train_state_adversary.params,
                        obs_minibatch_adversary,
                    )
                    log_prob_new_adversary = pi_new_adversary.log_prob(
                        action_minibatch_adversary
                    )
                    ratio_new_adversary = jnp.exp(
                        log_prob_new_adversary - log_prob_minibatch_adversary
                    )
                    loss_info_adversary["ratio_new"] = ratio_new_adversary

                    loss_info_agent = {
                        k + "_agent": v for k, v in loss_info_agent.items()
                    }
                    loss_info_adversary = {
                        k + "_adversary": v for k, v in loss_info_adversary.items()
                    }
                    loss_info = {**loss_info_agent, **loss_info_adversary}
                    loss_info["total_loss"] = total_loss

                    # gae difference
                    loss_info["gae_diff"] = (
                        advantages_minibatch_agent.mean()
                        - advantages_minibatch_adversary.mean()
                    )

                    return (
                        updated_train_state_agent,
                        updated_train_state_adversary,
                    ), (loss_info)

                (
                    final_train_state_agent,
                    final_train_state_adversary,
                ), (loss_info) = jax.lax.scan(
                    _update_minibatch,
                    (
                        train_state_agent,
                        train_state_adversary,
                    ),
                    (minibatches_agent, minibatches_adversary),
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
                return update_state, loss_info

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
            final_update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["num_updates"]
            )
            train_state_agent = final_update_state.train_state_agent
            train_state_adversary = final_update_state.train_state_adversary
            rng = final_update_state[-1]

            loss_info_mean = jax.tree.map(lambda x: x.mean(), loss_info)

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
            log_dict = {**loss_info_mean}
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


@hydra.main(version_base=None, config_path="./", config_name="config_ippo_mlp")
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
