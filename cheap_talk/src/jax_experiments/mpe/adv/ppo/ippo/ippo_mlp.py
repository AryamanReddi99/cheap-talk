import os

# disable randomness
# os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Tuple, Union, Dict
import datetime
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import DictConfig, OmegaConf
from functools import partial
import jaxmarl
from jaxmarl.wrappers.baselines import MPELogWrapper, JaxMARLWrapper
from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State
from cheap_talk.src.utils.wandb_process import WandbMultiLogger
import functools
from cheap_talk.src.networks.mlp import ActorCriticDiscreteMLP


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    log_prob: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    new_done: jnp.ndarray
    value: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


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
        job_type = f"IPPO_{config['job_type']}_{config['env_name']}"
        group = (
            f"{config['beta_1']}_{config['env_name']}"
            + datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
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
