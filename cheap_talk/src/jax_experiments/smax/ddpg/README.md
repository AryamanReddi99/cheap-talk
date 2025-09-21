# DDPG for SMAX Environments

This directory contains a JAX implementation of Deep Deterministic Policy Gradient (DDPG) algorithm adapted for discrete action spaces in SMAX environments.

## Algorithm Overview

DDPG is an actor-critic algorithm that learns a deterministic policy. For discrete action spaces like SMAX, we use the Gumbel Softmax trick to make the discrete action selection differentiable, enabling gradient-based optimization.

### Key Features

- **Actor Network**: RNN-based network that outputs action logits
- **Critic Network**: RNN-based network that estimates Q-values given state-action pairs
- **Gumbel Softmax**: Converts continuous logits to discrete actions while maintaining differentiability
- **Epsilon-Greedy Exploration**: Uses epsilon-greedy action selection for exploration
- **Experience Replay**: Uses a replay buffer for stable learning
- **Target Networks**: Soft updates for stable training

## Files

- `ddpg.py`: Main DDPG implementation
- `config.yaml`: Configuration parameters
- `run_ddpg.sh`: Simple run script

## Usage

### Basic Usage

```bash
# Activate the conda environment
conda activate cheap

# Run with default parameters
python ddpg.py

# Or use the run script
./run_ddpg.sh
```

### Configuration

The algorithm can be configured by modifying `config.yaml`. Key parameters include:

- `GUMBEL_TAU`: Temperature parameter for Gumbel Softmax (default: 1.0)
- `EPS_START`: Starting epsilon for exploration (default: 1.0)
- `EPS_FINISH`: Final epsilon value (default: 0.05)
- `EPS_DECAY`: Percentage of updates over which to decay epsilon (default: 0.1)
- `TAU`: Soft update rate for target networks (default: 0.005)
- `HIDDEN_SIZE`: Size of RNN hidden states (default: 512)
- `LR`: Learning rate (default: 0.00005)

### Environment Settings

- `MAP_NAME`: SMAX map to use (default: "3s_vs_5z")
- `ENV_NAME`: Environment type (default: "HeuristicEnemySMAX")

## Algorithm Details

### Actor Network
- Takes observations and available actions as input
- Outputs action logits that are masked for invalid actions
- Uses GRU cells for handling temporal dependencies

### Critic Network
- Takes observations and actions (one-hot encoded) as input
- Outputs Q-values for the given state-action pairs
- Also uses GRU cells for temporal modeling

### Gumbel Softmax
- Converts continuous logits to discrete probability distributions
- Uses straight-through estimator for hard sampling during execution
- Maintains differentiability for gradient-based learning

### Training Process
1. Collect experience using current policy with epsilon-greedy exploration
2. Sample minibatches from replay buffer
3. Update critic to minimize TD error
4. Update actor to maximize Q-values from critic
5. Soft update target networks

## Performance Notes

- The algorithm uses JAX for efficient computation and JIT compilation
- Experience replay buffer helps stabilize learning
- Target networks reduce correlations in training
- Epsilon-greedy exploration helps with action space exploration

## Troubleshooting

If you encounter issues:

1. Ensure the `cheap` conda environment is activated
2. Check that all dependencies are installed (JAX, Flax, JaxMARL, etc.)
3. Verify the SMAX environment is properly configured
4. Monitor WandB logs if enabled for training diagnostics

## References

- [DDPG Paper](https://arxiv.org/abs/1509.02971)
- [Gumbel Softmax](https://arxiv.org/abs/1611.01144)
- [JaxMARL Framework](https://github.com/FLAIROx/JaxMARL)
