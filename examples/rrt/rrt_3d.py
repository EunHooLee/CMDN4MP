# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import numpy as np

from src.rrt.rrt import RRT
from src.search_space.search_space import SearchSpace
from src.utilities.plotting import Plot
"""
Q[1]: 사실 1이던 2던 4이던 상관없을 듯하다.
prc : Q[1] 내부 루프에서 현재 상태를 일정 확률로 확인한다. (샘플수, 연결여부)

search_space.py 내부 collsion free 부분을 is_in_collision 이랑 엮어놔야 한다.
obstacle free 함수도 변경해야 한다.
즉, 샘플링한 점 뿐 아니라 TREE로 편입하기 위해 near, rand 사이에 장애물이 없는지를 일정 resolution으로 확인해야 한다.

"""

X_dimensions = np.array([(0, 100), (0, 100), (0, 100)])  # dimensions of Search Space
# obstacles
Obstacles = np.array(
    [(20, 20, 20, 40, 40, 40), (20, 20, 60, 40, 40, 80), (20, 60, 20, 40, 80, 40), (60, 60, 20, 80, 80, 40),
     (60, 20, 20, 80, 40, 40), (60, 20, 60, 80, 40, 80), (20, 60, 60, 40, 80, 80), (60, 60, 60, 80, 80, 80)])
x_init = (0, 0, 0)  # starting location
x_goal = (100, 100, 100)  # goal location

Q = np.array([(0.5, 4)])  # length of tree edges

r = 1  # length of smallest edge to check for intersection with obstacles
max_samples = 21024  # max number of samples to take before timing out
prc = 0.1  # probability of checking for a connection to goal

# create Search Space

X = SearchSpace(X_dimensions,Obstacles)

# create rrt_search
rrt = RRT(X, Q, x_init, x_goal, max_samples, r, prc)
path = rrt.rrt_search()

print('path_lehgth: ',len(path))

# plot
plot = Plot("rrt_3d")
plot.plot_tree(X, rrt.trees)
if path is not None:
    plot.plot_path(X, path)
plot.plot_obstacles(X, Obstacles)
plot.plot_start(X, x_init)
plot.plot_goal(X, x_goal)
plot.draw(auto_open=True)
