import numpy as np

# Problem Parameters
MAX_USERS = 30
MAX_UAVS = 2
D_MAX = 350
BEAM_ANGLE = np.radians(45)  # half-angle in radians

# Bounds
USER_BOUNDS = np.array([[0, 0, 0], [1000, 1000, 300]])
UAV_BOUNDS = np.array([[0, 0, 300], [1000, 1000, 600]])
USER_CLUSTER_BOUNDS = np.array([[100, 100, 50], [900, 900, 250]])
MAX_NUM_CLUSTERS = MAX_UAVS

# GA Parameters
GA_GENERATIONS = 250
GA_POP_SIZE = 30

# DRL Parameters
MAX_MOVE_DELTA = 5

#normalization scale factor for environment
SCALE_FACTOR = 1000
