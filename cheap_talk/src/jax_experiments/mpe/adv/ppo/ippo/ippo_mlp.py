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
    global_done: jnp.ndarray
    value: jnp.ndarray
    info: jnp.ndarray


class RunnerState(NamedTuple):
    train_state_agent: TrainState
    train_state_adversary: TrainState
    running_grad_agent: jnp.ndarray  # Running gradient for cosine similarity
    running_grad_adversary: jnp.ndarray  # Running gradient for cosine similarity
    obs_agent: jnp.ndarray
    obs_adversary: jnp.ndarray
    state: jnp.ndarray
    done_agent: jnp.ndarray
    done_adversary: jnp.ndarray
    update_step: int
    rng: Array


class Updatestate(NamedTuple):
    train_state_agent: TrainState
    train_state_adversary: TrainState
    running_grad_agent: jnp.ndarray  # Running gradient for cosine similarity
    running_grad_adversary: jnp.ndarray  # Running gradient for cosine similarity
    running_grad: jnp.ndarray  # Running gradient for cosine similarity
    traj_batch_agent: Transition
    traj_batch_adversary: Transition
    advantages_agent: jnp.ndarray
    advantages_adversary: jnp.ndarray
    targets_agent: jnp.ndarray
    targets_adversary: jnp.ndarray
    rng: Array


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    # env
    env = jaxmarl.make(config["env_name"], **config["env_kwargs"])
    env = MPELogWrapper(env)

    # config
    config["num_actors"] = env.num_agents * config["num_envs"]
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
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng, exp_id):
        def train_setup(rng):
            # env reset
            rng, _rng_reset = jax.random.split(rng)
            _rng_resets = jax.random.split(_rng_reset, config["num_envs"])
            obs, state = jax.vmap(env.reset, in_axes=(0))(_rng_resets)
            obs_agent = {k: obs[k] for k in env.good_agents}
            obs_adversary = {k: obs[k] for k in env.adversaries}

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

            # Initialize running gradient (zero gradient)
            running_grad_agent = jax.tree.map(jnp.zeros_like, network_params_agent)
            running_grad_adversary = jax.tree.map(
                jnp.zeros_like, network_params_adversary
            )

            return (
                train_state_agent,
                train_state_adversary,
                obs_agent,
                obs_adversary,
                state,
                running_grad_agent,
                running_grad_adversary,
            )

        rng, _rng_setup = jax.random.split(rng)
        (
            train_state_agent,
            train_state_adversary,
            network_agent,
            network_adversary,
            obs_agent,
            obs_adversary,
            state,
            running_grad_agent,
            running_grad_adversary,
        ) = train_setup(_rng_setup)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            def _env_step(runner_state, unused):
                train_state_agent = runner_state.train_state_agent
                train_state_adversary = runner_state.train_state_adversary
                obs_agent = runner_state.obs_agent
                obs_adversary = runner_state.obs_adversary
                state = runner_state.state
                done_agent = runner_state.done_agent
                done_adversary = runner_state.done_adversary
                rng = runner_state.rng

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch_agent = batchify(
                    obs_agent, env.good_agents, config["num_actors"]
                )
                obs_batch_adversary = batchify(
                    obs_adversary, env.adversaries, config["num_actors"]
                )
                pi_agent, value_agent = network_agent.apply(
                    train_state_agent.params, obs_batch_agent
                )
                pi_adversary, value_adversary = network_adversary.apply(
                    train_state_adversary.params, obs_batch_adversary
                )
                action_agent = pi_agent.sample(seed=_rng)
                action_adversary = pi_adversary.sample(seed=_rng)
                log_prob_agent = pi_agent.log_prob(action_agent)
                log_prob_adversary = pi_adversary.log_prob(action_adversary)
                env_act_agent = unbatchify(
                    action_agent, env.good_agents, config["num_envs"], env.num_agents
                )
                env_act_adversary = unbatchify(
                    action_adversary,
                    env.adversaries,
                    config["num_envs"],
                    env.num_agents,
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
                obs_batch_agent = batchify(
                    new_obs, env.good_agents, config["num_actors"]
                )
                obs_batch_adversary = batchify(
                    new_obs, env.adversaries, config["num_actors"]
                )
                done_batch_agent = batchify(
                    done_agent, env.good_agents, config["num_actors"]
                ).squeeze()
                done_batch_adversary = batchify(
                    done_adversary, env.adversaries, config["num_actors"]
                ).squeeze()
                transition_agent = Transition(
                    obs=obs_batch_agent,
                    action=action_agent.squeeze(),
                    log_prob=log_prob_agent.squeeze(),
                    reward=batchify(reward, env.agents, config["num_actors"]).squeeze(),
                    done=done_agent,
                    new_done=new_done,
                    global_done=jnp.tile(new_done["__all__"], env.num_agents),
                    value=value_agent.squeeze(),
                    info=info,
                )
                transition_adversary = Transition(
                    obs=obs_batch_adversary,
                    action=action_adversary.squeeze(),
                    log_prob=log_prob_adversary.squeeze(),
                    reward=batchify(
                        reward, env.adversaries, config["num_actors"]
                    ).squeeze(),
                    done=done_adversary,
                    new_done=new_done,
                    global_done=jnp.tile(new_done["__all__"], env.num_agents),
                    value=value_adversary.squeeze(),
                    info=info,
                )
                runner_state = RunnerState(
                    train_state_agent=train_state_agent,
                    train_state_adversary=train_state_adversary,
                    running_grad_agent=runner_state.running_grad_agent,
                    running_grad_adversary=runner_state.running_grad_adversary,
                    obs_agent=obs_batch_agent,
                    obs_adversary=obs_batch_adversary,
                    state=new_state,
                    done_agent=done_batch_agent,
                    done_adversary=done_batch_adversary,
                    update_step=runner_state.update_steps,
                    rng=rng,
                )
                return runner_state, transition_agent, transition_adversary

            runner_state, traj_batch_agent, traj_batch_adversary = jax.lax.scan(
                _env_step, runner_state, None, config["num_steps"]
            )

            # CALCULATE ADVANTAGE
            train_state_agent = runner_state.train_state_agent
            train_state_adversary = runner_state.train_state_adversary
            last_obs_agent = runner_state.obs_agent
            last_obs_adversary = runner_state.obs_adversary
            last_done_agent = runner_state.done_agent
            last_done_adversary = runner_state.done_adversary
            rng = runner_state.rng

            last_obs_batch_agent = batchify(
                last_obs_agent, env.good_agents, config["num_actors"]
            )
            last_obs_batch_adversary = batchify(
                last_obs_adversary, env.adversaries, config["num_actors"]
            )
            _, last_val_agent = network_agent.apply(
                train_state_agent.params, last_obs_batch_agent
            )
            _, last_val_adversary = network_adversary.apply(
                train_state_adversary.params, last_obs_batch_adversary
            )
            last_val = last_val_agent.squeeze()
            last_val_adversary = last_val_adversary.squeeze()
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
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
                running_grad_agent = update_state.running_grad_agent
                running_grad_adversary = update_state.running_grad_adversary
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
                    _rng_permute_agent, config["num_envs"]
                )
                permutation_adversary = jax.random.permutation(
                    _rng_permute_adversary, config["num_envs"]
                )
                batch_agent = (
                    traj_batch_agent,
                    advantages_agent.squeeze(),
                    targets_agent.squeeze(),
                )
                batch_adversary = (
                    traj_batch_adversary,
                    advantages_adversary.squeeze(),
                    targets_adversary.squeeze(),
                )
                shuffled_batch_adversary = jax.tree.map(
                    lambda x: jnp.take(x, permutation_adversary, axis=1),
                    batch_adversary,
                )
                shuffled_batch_agent = jax.tree.map(
                    lambda x: jnp.take(x, permutation_agent, axis=1), batch_agent
                )
                shuffled_batch_split_adversary = jax.tree.map(
                    lambda x: jnp.reshape(  # split into minibatches along actor dimension (dim 1)
                        x,
                        [x.shape[0], config["num_minibatches"], -1] + list(x.shape[2:]),
                    ),
                    shuffled_batch_adversary,
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
                minibatches_adversary = jax.tree.map(  # swap minibatch and time axis,
                    lambda x: jnp.swapaxes(x, 0, 1),
                    shuffled_batch_split_adversary,
                )

                def _update_minibatch(carry, minibatch):
                    (
                        train_state_agent,
                        train_state_adversary,
                        network_agent,
                        network_adversary,
                        running_grad_agent,
                        running_grad_adversary,
                    ) = carry
                    minibatch_agent, minibatch_adversary = minibatch
                    (
                        traj_minibatch_agent,
                        advantages_minibatch_agent,
                        targets_minibatch_agent,
                    ) = minibatch_agent
                    (
                        traj_minibatch_adversary,
                        advantages_minibatch_adversary,
                        targets_minibatch_adversary,
                    ) = minibatch_adversary

                    def _loss_fn(
                        params, traj_minibatch, gae_minibatch, targets_minibatch
                    ):
                        # RERUN NETWORK
                        pi, value = network.apply(
                            params,
                            traj_minibatch.obs,
                        )
                        log_prob = pi.log_prob(traj_minibatch.action)

                        # actor loss
                        logratio = log_prob - traj_minibatch.log_prob
                        ratio = jnp.exp(logratio)
                        gae_minibatch = (gae_minibatch - gae_minibatch.mean()) / (
                            gae_minibatch.std() + 1e-8
                        )
                        loss_actor_1 = ratio * gae_minibatch
                        loss_actor_2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["clip_eps"],
                                1.0 + config["clip_eps"],
                            )
                            * gae_minibatch
                        )
                        loss_actor = -jnp.minimum(loss_actor_1, loss_actor_2).mean()
                        entropy = pi.entropy().mean()

                        # critic loss
                        value_pred_clipped = traj_minibatch.value + (
                            value - traj_minibatch.value
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
                        }

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state_agent.params,
                        traj_minibatch,
                        advantages_minibatch,
                        targets_minibatch,
                    )
                    updated_train_state = train_state.apply_gradients(grads=grads)
                    new_running_grad = grads
                    total_loss[1]["grad_norm"] = pytree_norm(grads)
                    return (updated_train_state, new_running_grad), total_loss

                (
                    final_train_state_agent,
                    final_train_state_adversary,
                    final_running_grad_agent,
                    final_running_grad_adversary,
                ), total_loss_agent = jax.lax.scan(
                    _update_minibatch,
                    (
                        train_state_agent,
                        train_state_adversary,
                        network_agent,
                        network_adversary,
                        running_grad_agent,
                        running_grad_adversary,
                    ),
                    (minibatches_agent, minibatches_adversary),
                )
                update_state = Updatestate(
                    train_state_agent=final_train_state_agent,
                    train_state_adversary=final_train_state_adversary,
                    running_grad_agent=final_running_grad_agent,
                    running_grad_adversary=final_running_grad_adversary,
                    traj_batch_agent=traj_batch_agent,
                    traj_batch_adversary=traj_batch_adversary,
                    advantages_agent=advantages_agent,
                    advantages_adversary=advantages_adversary,
                    targets_agent=targets_agent,
                    targets_adversary=targets_adversary,
                    rng=rng,
                )
                return update_state, total_loss

            update_state = Updatestate(
                train_state_agent=train_state_agent,
                train_state_adversary=train_state_adversary,
                running_grad_agent=runner_state.running_grad_agent,
                running_grad_adversary=runner_state.running_grad_adversary,
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
            metric = traj_batch_agent.info
            metric = jax.tree.map(
                lambda x: x.reshape(
                    (config["num_steps"], config["num_envs"], env.num_agents)
                ),
                traj_batch_agent.info,
            )
            ratio_0 = loss_info[1][3].at[0, 0].get().mean()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            metric["loss"] = {
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "ratio": loss_info[1][3],
                "ratio_0": ratio_0,
                "approx_kl": loss_info[1][4],
                "clip_frac": loss_info[1][5],
            }

            rng = update_state[-1]

            def callback(exp_id, metric):
                log_dict = {
                    # the metrics have an agent dimension, but this is identical
                    # for all agents so index into the 0th item of that dimension.
                    "returns": metric["returned_episode_returns"][:, :, 0][
                        metric["returned_episode"][:, :, 0]
                    ].mean(),
                    "win_rate": metric["returned_won_episode"][:, :, 0][
                        metric["returned_episode"][:, :, 0]
                    ].mean(),
                    "env_step": metric["update_steps"]
                    * config["num_envs"]
                    * config["num_steps"],
                    **metric["loss"],
                }
                np_log_dict = {k: np.array(v) for k, v in log_dict.items()}
                LOGGER.log(int(exp_id), np_log_dict)

            metric["update_steps"] = runner_state.update_step

            jax.experimental.io_callback(callback, None, exp_id, metric)
            runner_state = RunnerState(
                train_state_agent=train_state_agent,
                train_state_adversary=train_state_adversary,
                running_grad=runner_state.running_grad,
                obs_agent=last_obs_agent,
                obs_adversary=last_obs_adversary,
                state=last_done_agent,
                done_agent=last_done_agent,
                done_adversary=last_done_adversary,
                update_step=runner_state.update_step + 1,
                rng=rng,
            )
            return runner_state, metrics_batch

        rng, _rng = jax.random.split(rng)
        initial_runner_state = RunnerState(
            train_state_agent=train_state_agent,
            train_state_adversary=train_state_adversary,
            running_grad_agent=running_grad_agent,
            running_grad_adversary=running_grad_adversary,
            obs_agent=obs_agent,
            obs_adversary=obs_adversary,
            state=state,
            done_agent=jnp.zeros((config["num_actors"]), dtype=bool),
            done_adversary=jnp.zeros((config["num_actors"]), dtype=bool),
            update_step=0,
            rng=_rng,
        )
        final_runner_state, metrics_batch = jax.lax.scan(
            _update_step, initial_runner_state, None, config["num_updates"]
        )
        return final_runner_state, metrics_batch

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
            mode=config["wandb_mode"],
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
