import jax
import jax.numpy as jnp
import numpy as np
import jaxmarl
from jaxmarl.wrappers.baselines import MPELogWrapper


def test_mpe_environment():
    """Test script for MPE_simple_adversary_v3 environment with 2 random actions."""

    # Create environment
    env = jaxmarl.make("MPE_simple_adversary_v3")
    env = MPELogWrapper(env)

    print("=== MPE Environment Test ===")
    print(f"Environment: MPE_simple_adversary_v3")
    print(f"Number of agents: {env.num_agents}")
    print(f"Good agents: {env.good_agents}")
    print(f"Adversaries: {env.adversaries}")
    print(f"Action spaces: {[env.action_space(agent) for agent in env.agents]}")
    print(
        f"Observation spaces: {[env.observation_space(agent) for agent in env.agents]}"
    )
    print()

    # Initialize random key
    rng = jax.random.PRNGKey(42)

    # Reset environment
    print("=== RESET ENVIRONMENT ===")
    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng)

    print("Initial observations:")
    for agent_id, obs_value in obs.items():
        print(f"  {agent_id}: {obs_value}")
    print(f"Initial state: {state}")
    print()

    # Take 2 random actions
    for step in range(2):
        print(f"=== STEP {step + 1} ===")

        # Generate random actions for all agents
        rng, action_rng = jax.random.split(rng)
        actions = {}

        for agent_id in env.agents:
            action_space = env.action_space(agent_id)
            if hasattr(action_space, "n"):  # Discrete action space
                action = jax.random.randint(action_rng, (), 0, action_space.n)
            else:  # Continuous action space
                action = jax.random.uniform(
                    action_rng,
                    action_space.shape,
                    minval=action_space.low,
                    maxval=action_space.high,
                )
            actions[agent_id] = action
            action_rng, _ = jax.random.split(action_rng)

        print("Actions taken:")
        for agent_id, action in actions.items():
            print(f"  {agent_id}: {action}")

        # Step environment
        rng, step_rng = jax.random.split(rng)
        new_obs, new_state, rewards, dones, info = env.step(step_rng, state, actions)

        print("New observations:")
        for agent_id, obs_value in new_obs.items():
            print(f"  {agent_id}: {obs_value}")

        print("Rewards:")
        for agent_id, reward in rewards.items():
            print(f"  {agent_id}: {reward}")

        print("Done flags:")
        for agent_id, done in dones.items():
            print(f"  {agent_id}: {done}")

        print("Info:")
        for key, value in info.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")

        print(f"New state: {new_state}")
        print()

        # Update state for next iteration
        state = new_state
        obs = new_obs

        # Check if episode is done
        if dones.get("__all__", False):
            print("Episode finished!")
            break

    print("=== TEST COMPLETED ===")


if __name__ == "__main__":
    test_mpe_environment()
