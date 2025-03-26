import numpy as np
import pygad
from config import D_MAX, BEAM_ANGLE, USER_BOUNDS, UAV_BOUNDS

import warnings
warnings.filterwarnings("ignore", message="The percentage of genes to mutate.*")

# === Problem Setup ===
# NUM_USERS = 100
# NUM_BASE_STATIONS = 5
# DISTRIBUTION_TYPE = "clustered"
# NUM_CLUSTERS = 5
#
# USER_CLUSTER_BOUNDS = np.array([[100, 100, 50], [900, 900, 250]])
#
#
# user_positions = np.zeros(shape=(NUM_USERS, 3))
# # Generate random static UAV user positions based on the distribution type
# if DISTRIBUTION_TYPE == "uniform":
#     # Place all users in a random uniform distribution within the bounds
#     user_positions = np.random.uniform(low=USER_BOUNDS[0], high=USER_BOUNDS[1], size=(NUM_USERS, 3))
# elif DISTRIBUTION_TYPE == "clustered":
#     # Define cluster centers for x and y in a subregion to avoid edge effects.
#     cluster_centers = np.random.uniform(low=USER_CLUSTER_BOUNDS[0], high=USER_CLUSTER_BOUNDS[1], size=(NUM_CLUSTERS, 3))
#     # Place each user around a cluster
#     for i in range(NUM_USERS):
#         # Randomly select a cluster center.
#         cluster_idx = np.random.randint(0, NUM_CLUSTERS)
#         center = cluster_centers[cluster_idx]
#         # Sample x and y from a normal distribution around the cluster center.
#         user_positions[i, 0] = np.clip(np.random.normal(center[0], 30), 0, 1000)
#         user_positions[i, 1] = np.clip(np.random.normal(center[1], 30), 0, 1000)
#         user_positions[i, 2] = np.clip(np.random.normal(center[2], 30), 0, 300)
# elif DISTRIBUTION_TYPE == "gradient":
#     a, b = 2, 5 # Shape parameters, adjust to change bias
#     user_positions[:, 0] = np.random.beta(a, b, size=NUM_USERS) * 1000
#     user_positions[:, 1] = np.random.beta(a, b, size=NUM_USERS) * 1000
#     user_positions[:, 2] = np.random.uniform(0, 300, size=NUM_USERS)
# else:
#     print("Invalid distribution type")
#     exit()

def make_fitness_function(user_positions, num_users, num_base_stations):
    def coverage_fitness(ga_instance, solution, solution_idx):
        idle_uav_count = 0
        redundant_uav_count = 0
        user_covered = np.zeros(num_users, dtype=bool)

        for j in range(num_base_stations):
            bx, by, bz = solution[3 * j: 3 * j + 3]
            base_pos = np.array([bx, by, bz])
            vectors = user_positions - base_pos
            distances = np.linalg.norm(vectors, axis=1)

            within_distance = distances <= D_MAX
            dz = vectors[:, 2]
            horizontal_norm = np.linalg.norm(vectors[:, :2], axis=1)
            elevation_angles = np.arctan2(dz, horizontal_norm) + np.pi / 2
            within_angle = elevation_angles <= BEAM_ANGLE
            covered_now = within_distance & within_angle

            if not np.any(covered_now):
                idle_uav_count += 1

            newly_covered = covered_now & ~user_covered
            if not np.any(newly_covered):
                redundant_uav_count += 1

            user_covered[newly_covered] = True

        penalty = idle_uav_count * 2 + redundant_uav_count * 1
        fitness = np.sum(user_covered) - penalty

        return fitness
    return coverage_fitness


def get_UAV_starting_positions(user_positions, num_users, num_base_stations):
    fitness_function = make_fitness_function(user_positions, num_users, num_base_stations)
    # === GA Parameters ===
    gene_space = [
                     {"low": UAV_BOUNDS[0][0], "high": UAV_BOUNDS[1][0]},  # x
                     {"low": UAV_BOUNDS[0][1], "high": UAV_BOUNDS[1][1]},  # y
                     {"low": UAV_BOUNDS[0][2], "high": UAV_BOUNDS[1][2]},  # z
                 ] * num_base_stations  # repeat per UAV base station

    ga = pygad.GA(
        num_generations=250,
        num_parents_mating=15,
        fitness_func=fitness_function,
        sol_per_pop=60,
        num_genes=3 * num_base_stations,
        gene_space=gene_space,
        parent_selection_type="rank",
        keep_parents=10,
        crossover_type="uniform",
        mutation_type="adaptive",
        mutation_percent_genes=(30, 10),
        allow_duplicate_genes=True,
        stop_criteria=["saturate_30"]
    )

    NUM_USERS = num_users

    # === Run & Report ===
    ga.run()
    solution, solution_fitness, _ = ga.best_solution()
    return solution
    #print(f"Best solution covers {int(solution_fitness)} users")



