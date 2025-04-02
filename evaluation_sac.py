import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from environment import UAVPlacementEnv
from config import SCALE_FACTOR

# --- Config ---
MODEL_PATH = "./sac_uav_logs/sac_first_attempt.zip"
NUM_EPISODES = 100  # Set the number of episodes to run

# --- Load environment and model ---
env = UAVPlacementEnv()
model = SAC.load(MODEL_PATH)

# --- Plotting (optional) ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


def plot_state(user_positions, uav_positions, covered_users, step_count):
    ax.clear()

    # Split users into covered and uncovered based on the boolean mask
    covered = user_positions[covered_users]
    uncovered = user_positions[~covered_users]

    if len(covered) > 0:
        ax.scatter(covered[:, 0], covered[:, 1], covered[:, 2], c='green',
                   label='Covered Users', alpha=0.6)
    if len(uncovered) > 0:
        ax.scatter(uncovered[:, 0], uncovered[:, 1], uncovered[:, 2], c='gray',
                   label='Uncovered Users', alpha=0.3)

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
    plt.show(block=False)
    plt.pause(0.5)


# --- Run Episodes Automatically ---
coverage_changes = []  # To store coverage changes for each episode

for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    # Record initial coverage count (assuming env.prev_coverage is a boolean array)
    initial_coverage = np.sum(env.prev_coverage[:env.num_users])
    done = False
    step = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1

        # (Optional) Update plot during the episode if desired.
        # Uncomment the following lines to see the state update at every step.
        # uav_pos = env.uav_positions[:env.num_base_stations]
        # user_pos = env.user_positions[:env.num_users]
        # covered = env.prev_coverage[:env.num_users]
        # plot_state(user_pos, uav_pos, covered, step)

    # At the end of the episode, record final coverage
    final_coverage = np.sum(env.prev_coverage[:env.num_users])
    coverage_change = final_coverage - initial_coverage
    coverage_changes.append(coverage_change)
    print(f"Episode {episode + 1}: Initial Coverage = {initial_coverage}, "
          f"Final Coverage = {final_coverage}, Change = {coverage_change}")

# Compute and display the average coverage change across episodes
average_change = np.mean(coverage_changes)
print(f"Average coverage change over {NUM_EPISODES} episodes: {average_change}")
