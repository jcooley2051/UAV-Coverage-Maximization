import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from environment import UAVPlacementEnv  # Replace with actual path if needed

# --- Configuration ---
TIMESTEPS = 300_000
LOG_DIR = "./sac_uav_logs/"
MODEL_SAVE_PATH = os.path.join(LOG_DIR, "sac_first_attempt")

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
model = SAC(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    device="cuda",
    learning_rate=3e-4,
    batch_size=256,  # typical SAC batch size
    buffer_size=100_000,  # off-policy requires a replay buffer
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto",
)


# --- Train the model ---
model.learn(
    total_timesteps=TIMESTEPS,
    callback=eval_callback
)

# --- Save final model ---
model.save(MODEL_SAVE_PATH)
print(f"Model saved to: {MODEL_SAVE_PATH}")
