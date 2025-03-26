import copy

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from genetic_placement import get_UAV_starting_positions
from config import UAV_BOUNDS, USER_BOUNDS, MAX_UAVS, MAX_USERS, BEAM_ANGLE, D_MAX, MAX_MOVE_DELTA, SCALE_FACTOR, USER_CLUSTER_BOUNDS, MAX_NUM_CLUSTERS

class UAVPlacementEnv(gym.Env):
    def __init__(self, num_users=100, num_base_stations=5):
        super().__init__()
        self.num_users = num_users
        self.num_base_stations = num_base_stations
        self.max_uavs = MAX_UAVS
        self.max_users = MAX_USERS
        self.user_positions = []
        self.uav_positions = []
        self.coverage_vector = []
        self.beam_halfwidth = BEAM_ANGLE
        self.max_distance = D_MAX
        self.step_count = 0
        self.max_steps = 25
        self.initial_coverage_count = 0

        # Bounds for UAV movement (must match GA bounds)
        self.uav_bounds = UAV_BOUNDS
        self.user_bounds = USER_BOUNDS

        # Action space: movement deltas for max UAVs
        self.action_space = spaces.Box(
            low=-1, high=1,
            shape=(self.max_uavs * 3,), dtype=np.float32
        )

        # Observation space: all users + UAVs padded to max
        obs_dim = (self.max_users + self.max_uavs) * 3 + self.max_users
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomly generate episode parameters from with a range
        dist_type = np.random.choice(["uniform", "clustered", "gradient"])
        #self.num_users = np.random.randint(20, self.max_users + 1)
        #self.num_base_stations = np.random.randint(2, self.max_uavs + 1)
        self.num_users = self.max_users
        self.num_base_stations = self.max_uavs

        # Generate the distribution of users
        self.user_positions = self._generate_users(dist_type)

        # Load GA-initialized UAV layout (can be varied per episode if needed)
        self.uav_positions = get_UAV_starting_positions(self.user_positions, self.num_users, self.num_base_stations).reshape(self.num_base_stations, 3)
        self.step_count = 0
        self.prev_coverage = self._compute_coverage()
        self.initial_coverage_count = np.sum(self.prev_coverage[:self.num_users])
        return self._get_observation(), {}

    def _generate_users(self, dist_type):
        # Generate random static UAV user positions based on the distribution type
        user_positions = np.zeros(shape=(self.num_users, 3))
        if dist_type == "uniform":
            # Place all users in a random uniform distribution within the bounds
            user_positions = np.random.uniform(low=USER_BOUNDS[0], high=USER_BOUNDS[1], size=(self.num_users, 3))
        elif dist_type == "clustered":
            # Define cluster centers for x and y in a subregion to avoid edge effects.
            num_clusters = np.random.randint(1, MAX_NUM_CLUSTERS)
            cluster_centers = np.random.uniform(low=USER_CLUSTER_BOUNDS[0], high=USER_CLUSTER_BOUNDS[1], size=(num_clusters, 3))
            # Place each user around a cluster
            for i in range(self.num_users):
                # Randomly select a cluster center.
                cluster_idx = np.random.randint(0, num_clusters)
                center = cluster_centers[cluster_idx]
                # Sample x and y from a normal distribution around the cluster center.
                user_positions[i, 0] = np.clip(np.random.normal(center[0], 30), 0, 1000)
                user_positions[i, 1] = np.clip(np.random.normal(center[1], 30), 0, 1000)
                user_positions[i, 2] = np.clip(np.random.normal(center[2], 30), 0, 300)
        else:
            a, b = 2, 5 # Shape parameters, adjust to change bias
            user_positions[:, 0] = np.random.beta(a, b, size=self.num_users) * 1000
            user_positions[:, 1] = np.random.beta(a, b, size=self.num_users) * 1000
            user_positions[:, 2] = np.random.uniform(0, 300, size=self.num_users)

        return user_positions

    def _get_observation(self):
        user_pad = np.zeros((self.max_users, 3), dtype=np.float32)
        user_pad[:self.num_users] = self.user_positions
        user_pad = user_pad / SCALE_FACTOR

        uav_pad = np.zeros((self.max_uavs, 3), dtype=np.float32)
        uav_pad[:self.num_base_stations] = self.uav_positions
        uav_pad = uav_pad / SCALE_FACTOR

        coverage_vec = np.zeros(self.max_users, dtype=np.float32)
        coverage_vec[:self.num_users] = self._compute_coverage()[:self.num_users].astype(np.float32)

        return np.concatenate((user_pad.flatten(), uav_pad.flatten(), coverage_vec)).astype(np.float32)

    def _compute_coverage(self):
        covered = np.zeros(self.max_users, dtype=bool)

        for uav in self.uav_positions[:self.num_base_stations]:
            vectors = self.user_positions[:self.num_users] - uav
            distances = np.linalg.norm(vectors, axis=1)
            within_distance = distances <= self.max_distance
            dz = vectors[:, 2]
            horizontal_norm = np.linalg.norm(vectors[:, :2], axis=1)
            angles = np.arctan2(dz, horizontal_norm) + np.pi / 2
            within_angle = angles <= self.beam_halfwidth
            covered[:self.num_users] |= within_distance & within_angle

        return covered

    def step(self, action):
        deltas = action.reshape(self.max_uavs, 3)
        scaled_deltas = deltas * MAX_MOVE_DELTA  # or whatever your real-world range is

        prev_positions = self.uav_positions[:self.num_base_stations].copy()

        self.uav_positions[:self.num_base_stations] += scaled_deltas[:self.num_base_stations]
        self.uav_positions[:self.num_base_stations] = np.clip(
            self.uav_positions[:self.num_base_stations], self.uav_bounds[0], self.uav_bounds[1]
        )

        new_coverage = self._compute_coverage()
        new_covered_count = np.sum(new_coverage[:self.num_users])
        prev_covered_count = np.sum(self.prev_coverage[:self.num_users])

        movement_cost = np.sum(np.linalg.norm(self.uav_positions[:self.num_base_stations] - prev_positions, axis=1))

        alpha = 1.0  # Weight for coverage change.
        beta = 0.1  # Weight for movement penalty.


        reward = alpha * (new_covered_count - self.initial_coverage_count) - beta * movement_cost
        self.prev_coverage = new_coverage.copy()

        self.step_count += 1
        terminated = new_covered_count == self.num_users
        truncated = self.step_count >= self.max_steps

        return self._get_observation(), reward, terminated, truncated, {}

    def render(self):
        pass  # Optional: visualization hook
