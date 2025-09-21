#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from facmac import make_train


def test_facmac():
    """Simple test to verify FACMAC implementation works"""

    # Test config
    config = {
        "ALG_NAME": "facmac",
        "TOTAL_TIMESTEPS": 1000,  # Small for testing
        "NUM_ENVS": 4,  # Small for testing
        "NUM_STEPS": 8,  # Small for testing
        "LR": 0.001,
        "LR_LINEAR_DECAY": False,
        "BUFFER_SIZE": 100,  # Small for testing
        "BUFFER_BATCH_SIZE": 8,
        "HIDDEN_SIZE": 32,  # Small for testing
        "MAX_GRAD_NORM": 10,
        "TAU": 0.005,
        "NUM_EPOCHS": 2,  # Small for testing
        "LEARNING_STARTS": 50,  # Small for testing
        "GAMMA": 0.99,
        "TARGET_UPDATE_INTERVAL": 5,
        "REW_SCALE": 1.0,
        "MIXER_EMBEDDING_DIM": 16,  # Small for testing
        "MIXER_HYPERNET_HIDDEN_DIM": 32,  # Small for testing
        "MIXER_INIT_SCALE": 0.001,
        "EPS_START": 1.0,
        "EPS_FINISH": 0.1,
        "EPS_DECAY": 0.5,
        "GUMBEL_TAU": 1.0,
        "ENV_NAME": "HeuristicEnemySMAX",
        "MAP_NAME": "3s_vs_5z",
        "SEED": 42,
        "NUM_SEEDS": 1,
        "ENV_KWARGS": {
            "see_enemy_actions": True,
            "walls_cause_death": True,
            "attack_mode": "closest",
        },
        "PROJECT": "test_facmac",
        "USE_TIMESTAMP": False,
        "WANDB_MODE": "disabled",
        "TEST_DURING_TRAINING": False,
        "TEST_INTERVAL": 1.0,
        "TEST_NUM_STEPS": 8,
        "TEST_NUM_ENVS": 4,
    }

    try:
        # Create train function
        train_fn = make_train(config)
        print("✓ FACMAC train function created successfully")

        # Test with small run
        rng = jax.random.PRNGKey(42)
        result = train_fn(rng, 0)
        print("✓ FACMAC training completed successfully")

        # Check result structure
        assert "runner_state" in result
        assert "metrics" in result
        print("✓ Result structure is correct")

        # Check metrics structure
        metrics = result["metrics"]
        assert "critic_loss" in metrics
        assert "actor_loss" in metrics
        print("✓ Metrics structure is correct")

        print("\n🎉 All tests passed! FACMAC implementation is working correctly.")

        # Print some basic stats
        final_critic_loss = float(metrics["critic_loss"][-1])
        final_actor_loss = float(metrics["actor_loss"][-1])
        print(f"\nFinal losses:")
        print(f"  Critic loss: {final_critic_loss:.6f}")
        print(f"  Actor loss: {final_actor_loss:.6f}")

        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_facmac()
    exit(0 if success else 1)
