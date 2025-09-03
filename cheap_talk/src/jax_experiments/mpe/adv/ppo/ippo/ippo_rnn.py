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
from cheap_talk.src.jax_experiments.utils.wandb_process import WandbMultiLogger
import functools
