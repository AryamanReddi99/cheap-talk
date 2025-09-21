# FACMAC Implementation

## Overview

This directory contains a JAX implementation of FACMAC (Factorized Multi-Agent Centralised Policy Gradients) for the SMAX environment. FACMAC combines aspects of DDPG and QMIX to enable off-policy multi-agent learning with factorized value functions.

## Key Features

### Algorithm Design
- **Off-policy learning**: Uses experience replay buffer like DDPG
- **Shared networks**: Single actor and critic networks shared across all agents
- **Non-monotonic mixing**: Mixing network weights are not constrained to be positive (unlike QMIX)
- **End-to-end training**: Actor is updated through the mixing network output

### Network Architecture
1. **ActorRNN**: Outputs action logits for discrete actions, takes observations and available actions
2. **CriticRNN**: Outputs individual Q-values for each action, takes only observations  
3. **NonMonotonicMixingNetwork**: Combines individual Q-values into Q_tot, conditioned on global state

### Key Differences from DDPG/QMIX
- **vs DDPG**: Uses discrete actions with Gumbel-Softmax, mixing network for multi-agent coordination
- **vs QMIX**: Non-monotonic mixing weights, off-policy actor-critic learning instead of on-policy DQN

## Files

- `facmac.py`: Main implementation with networks and training loop
- `config.yaml`: Configuration parameters
- `test_facmac.py`: Simple test script to verify implementation
- `README.md`: This documentation

## Usage

### Training
```bash
cd /path/to/facmac/
python facmac.py
```

### Testing
```bash
python test_facmac.py
```

## Configuration

Key parameters in `config.yaml`:

### Learning Parameters
- `LR`: Learning rate (0.00005)
- `GAMMA`: Discount factor (0.99)
- `TAU`: Soft update rate for target networks (0.005)
- `NUM_EPOCHS`: Number of learning epochs per update (8)

### Buffer Parameters
- `BUFFER_SIZE`: Experience replay buffer size (5000)
- `BUFFER_BATCH_SIZE`: Minibatch size for learning (32)
- `LEARNING_STARTS`: Timesteps before learning starts (10000)

### Mixing Network Parameters
- `MIXER_EMBEDDING_DIM`: Hidden dimension for mixing network (64)
- `MIXER_HYPERNET_HIDDEN_DIM`: Hypernetwork hidden dimension (256)
- `MIXER_INIT_SCALE`: Initialization scale for mixing network (0.001)

### Exploration Parameters
- `EPS_START/FINISH/DECAY`: Epsilon-greedy exploration schedule
- `GUMBEL_TAU`: Temperature for Gumbel-Softmax sampling (1.0)

## Implementation Details

### Training Loop
1. **Data Collection**: Agents act in environment using current policy with epsilon-greedy exploration
2. **Buffer Update**: Store transitions in replay buffer
3. **Learning Phase**: Sample minibatches and update networks
   - Critic and mixer updated using TD error
   - Actor updated end-to-end through mixer gradients
4. **Target Update**: Soft update of target networks

### Loss Functions
- **Critic Loss**: MSE between current Q_tot and TD targets
- **Actor Loss**: Negative Q_tot (maximize expected return)

### Key Technical Choices
- **Gumbel-Softmax**: Enables differentiable discrete action sampling for actor gradients
- **Non-monotonic Mixing**: Allows more flexible value function approximation
- **Shared Networks**: Reduces parameter count and enables better generalization
- **RNN Architecture**: Handles partial observability in SMAX environment

## Dependencies

- JAX
- Flax
- Optax
- JaxMARL
- Flashbax
- Hydra

## References

- FACMAC: [Factorized Multi-Agent Centralised Policy Gradients](https://arxiv.org/abs/2003.06709)
- DDPG: [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)  
- QMIX: [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
