import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from environment import UAVPlacementEnv  # Replace with actual path if needed

# --- Configuration ---
TIMESTEPS = 400_000
LOG_DIR = "./ppo_uav_logs/"
MODEL_SAVE_PATH = os.path.join(LOG_DIR, "sac_first_test")

# --- Create environment ---
env = UAVPlacementEnv()
check_env(env)  # Optional: will raise errors if the env is broken

# --- Setup eval environment ---
eval_env = UAVPlacementEnv()

# --- Setup evaluation callback ---
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=LOG_DIR,
    log_path=LOG_DIR,
    eval_freq=5000,
    deterministic=True,
    render=False
)

# --- Create model ---
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    device="cpu",
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
)

# --- Train the model ---
model.learn(
    total_timesteps=TIMESTEPS,
    callback=eval_callback
)

# --- Save final model ---
model.save(MODEL_SAVE_PATH)
print(f"Model saved to: {MODEL_SAVE_PATH}")
