import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from environment import UAVPlacementEnv
from config import SCALE_FACTOR

# --- Config ---
MODEL_PATH = "./ppo_uav_logs/ppo_uav_model_new_reward_updated_again"

# --- Load environment and model ---
env = UAVPlacementEnv()
model = PPO.load(MODEL_PATH)

obs, _ = env.reset()
done = False
step = 0

# Store step data so we can step through it via key press
step_data = {
    "obs": obs,
    "done": done,
    "step": step
}

# --- Plotting ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

def plot_state(user_positions, uav_positions, covered_users, step_count):
    ax.clear()

    covered = user_positions[covered_users]
    print(f"Covered: {len(covered)}")
    uncovered = user_positions[~covered_users]
    print(covered)
    if len(covered) > 0:
        ax.scatter(covered[:, 0], covered[:, 1], covered[:, 2], c='green', label='Covered Users', alpha=0.6)
    if len(uncovered) > 0:
        ax.scatter(uncovered[:, 0], uncovered[:, 1], uncovered[:, 2], c='gray', label='Uncovered Users', alpha=0.3)

    ax.scatter(uav_positions[:, 0], uav_positions[:, 1], uav_positions[:, 2],
               c='red', marker='^', s=120, edgecolors='black', label='UAVs')

    ax.set_title(f"Step {step_count}")
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_zlim(0, 600)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc='upper left')
    plt.tight_layout()
    fig.canvas.draw()

def on_key(event):
    if event.key == "right":
        if step_data["done"]:
            print("Episode finished.")
            return

        action, _states = model.predict(step_data["obs"], deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        step_data["obs"] = obs
        step_data["done"] = terminated or truncated
        step_data["step"] += 1

        uav_pos = env.uav_positions[:env.num_base_stations]
        user_pos = env.user_positions[:env.num_users]
        covered = env.prev_coverage[:env.num_users]

        plot_state(user_pos, uav_pos, covered, step_data["step"])

# Initial plot
uav_pos = env.uav_positions[:env.num_base_stations]
user_pos = env.user_positions[:env.num_users]
covered = env.prev_coverage[:env.num_users]

plot_state(user_pos, uav_pos, covered, step_data["step"])

fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
