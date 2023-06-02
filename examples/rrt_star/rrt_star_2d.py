# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import numpy as np

from src.rrt.rrt_star import RRTStar
from src.search_space.search_space import SearchSpace
from src.utilities.plotting import Plot

def compute_cost(path):
    state_dists = []
    for i in range(len(path) -1):
        dist = 0
        for j in range(2):
            diff = path[i][j] - path[i+1][j]
            dist += diff ** 2
        state_dists.append(np.sqrt(dist))
    total_cost = sum(state_dists)
    return total_cost


def sample_obs_pose(obstacle_range):
    obs_center_pose = np.random.uniform(obstacle_range[:,0], obstacle_range[:,1])
    obs_center_pose = obs_center_pose.astype(int)

    obstacles_pose = []
    for i in range(int(len(obs_center_pose)/2)):
        temp = np.array([])
        for j in range(2):
            temp = np.append(temp, obs_center_pose[2*i+j])
            
        pose_1 = temp - np.array([10,10])
        pose_2 = temp + np.array([10,10])
        poses = np.append(pose_1,pose_2)
        obstacles_pose.append(poses)
    
    return np.array(obstacles_pose)
        



obstacle_range = np.array([(20,40),
                           (20,40),
                           (20,40),
                           (60,80),
                           (60,80),
                           (20,40),
                           (60,80),
                           (60,80)])


X_dimensions = np.array([(0, 100), (0, 100)])  # dimensions of Search Space
# obstacles
Obstacles = np.array([(20, 20, 40, 40), (20, 60, 40, 80), (60, 20, 80, 40), (60, 60, 80, 80)])
x_init = (0, 0)  # starting location
x_goal = (100, 100)  # goal location
# Obstacles = sample_obs_pose(obstacle_range)
# print(Obstacles)
Q = np.array([(1, 1)])  # length of tree edges
r = 1  # length of smallest edge to check for intersection with obstacles
max_samples = 1024  # max number of samples to take before timing out
rewire_count = 32  # optional, number of nearby branches to rewire
prc = 0  # probability of checking for a connection to goal

# create Search Space
X = SearchSpace(X_dimensions, Obstacles)

# create rrt_search
rrt = RRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)
path = rrt.rrt_star()
print(path)
print(len(path))
print(compute_cost(path))
# print(path)
# print(len(path))
# print(path[0])
# plot
plot = Plot("rrt_star_2d")
plot.plot_tree(X, rrt.trees)
if path is not None:
    plot.plot_path(X, path)
plot.plot_obstacles(X, Obstacles)
plot.plot_start(X, x_init)
plot.plot_goal(X, x_goal)
plot.draw(auto_open=True)
