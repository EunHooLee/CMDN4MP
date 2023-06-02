import numpy as np
import torch
from rtree import index
import pickle
import argparse
from scipy.stats import truncnorm

from src.rrt.rrt_star import RRTStar
# from src.search_space.search_space import SearchSpace
from src.utilities.plotting import Plot
from src.utilities.geometry import es_points_along_line
from src.utilities.obstacle_generation import obstacle_generator
from src.rrt.rrt import RRT
from operator import itemgetter

from src.rrt.heuristics import cost_to_go
from src.rrt.heuristics import segment_cost, path_cost

from MDN.blocks import MixtureDensityNetwork, NoiseType


class SearchSpace(object):
    def __init__(self, dimension_lengths, mdn, O=None):
        """
        Initialize Search Space
        :param dimension_lengths: range of each dimension
        :param O: list of obstacles
        """
        # sanity check
        if len(dimension_lengths) < 2:
            raise Exception("Must have at least 2 dimensions")
        self.dimensions = len(dimension_lengths)  # number of dimensions
        # sanity checks
        if any(len(i) != 2 for i in dimension_lengths):
            raise Exception("Dimensions can only have a start and end")
        if any(i[0] >= i[1] for i in dimension_lengths):
            raise Exception("Dimension start must be less than dimension end")
        self.dimension_lengths = dimension_lengths  # length of each dimension
        p = index.Property()
        p.dimension = self.dimensions


        if O is None:
            self.obs = index.Index(interleaved=True, properties=p)
        else:
            # r-tree representation of obstacles
            # sanity check
            if any(len(o) / 2 != len(dimension_lengths) for o in O):
                raise Exception("Obstacle has incorrect dimension definition")
            if any(o[i] >= o[int(i + len(o) / 2)] for o in O for i in range(int(len(o) / 2))):
                raise Exception("Obstacle start must be less than obstacle end")
            self.obs = index.Index(obstacle_generator(O), interleaved=True, properties=p)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mdn = mdn
        self.c_c = None
        self.goal = None
        self.feat = None
        self.N = 0
        self.num_nn_sample = 0
        self.num_uni_sample = 0
        self.checker = 0
        self.num_stuck = 0

    # 현재 포인트가 장애물 안, 밖인지 판단
    def obstacle_free(self, x):
        """
        Check if a location resides inside of an obstacle
        :param x: location to check
        :return: True if not inside an obstacle, False otherwise
        """
        # self.obs.count(x): 장애물 외부이면 0 (false), 내부이면 1 (true)
        return self.obs.count(x) == 0 
    
    def obstacle_free_sample(self):
        while True:
            x = self.sample()
            if self.obstacle_free(x):
                self.num_uni_sample +=1
                self.c_c = x
                return x

    def sample_free(self):
        """
        Sample a location within X_free
        :return: random location within X_free
        """
        while True:
            if self.N < 5:
                x = self.sample_from_nn()
                self.checker = 1
            else:
                x = self.sample()
                self.N = 0
                self.checker = 0
                self.num_stuck +=1
            if self.obstacle_free(x):
                if self.checker == 1:
                    self.num_nn_sample +=1
                elif self.checker == 0:
                    self.num_uni_sample +=1
                self.c_c = x
                # print(self.num_nn_sample)
                return x
            self.N +=1

    # NEAR, RAND 를 이어주는 직선 내부에 장애물이 있는지 판단
    def collision_free(self, start, end, r):
        """
        Check if a line segment intersects an obstacle
        :param start: starting point of line
        :param end: ending point of line
        :param r: resolution of points to sample along edge when checking for collisions
        :return: True if line segment does not intersect an obstacle, False otherwise
        """
        points = es_points_along_line(start, end, r)
        
        coll_free = all(map(self.obstacle_free, points))
        return coll_free

    def sample(self):
        """
        Return a random location within X
        :return: random location within X (not necessarily X_free)
        """
        x = np.random.uniform(self.dimension_lengths[:, 0], self.dimension_lengths[:, 1])
        
        return tuple(x)
    
    
    
    def planning_scene_update(self, start, goal, feat):
        self.c_c = start 
        self.goal = torch.FloatTensor(goal).to(self.device)
        self.feat = feat 

    def sample_from_nn(self):

        c_c = torch.FloatTensor(self.c_c).to(self.device)
        # print(c_c.dim(), self.goal.dim(), self.feat.dim())
        Inp = torch.cat([c_c, self.goal, self.feat], dim=0).unsqueeze(0)
        log_pi, mu, sigma = self.mdn(Inp)
        log_pi, mu, sigma = log_pi.cpu().detach().numpy().squeeze(), mu.cpu().detach().numpy().squeeze(), sigma.cpu().detach().numpy().squeeze()
        choose_gaussian_idx = np.exp(log_pi).argmax()
        # print(Inp)
        # print(mu)
        
        x = truncnorm.rvs((self.dimension_lengths[:,0] - mu[choose_gaussian_idx,:])/sigma[choose_gaussian_idx,:],(self.dimension_lengths[:,1] - mu[choose_gaussian_idx,:])/sigma[choose_gaussian_idx,:],loc= mu[choose_gaussian_idx,:],scale=sigma[choose_gaussian_idx,:])
        # x = np.random.uniform(self.dimension_lengths[:, 0], self.dimension_lengths[:, 1]) * sigma[choose_gaussian_idx,:] + mu[choose_gaussian_idx,:]
        
        # print(x)
        return tuple(x)

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


def sample_obstacles(obstacle_range):
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

def sample_init():
    x = np.random.randint(0,100,1)
    x_init = np.append(x,np.random.randint(0,5,1))
    
    return tuple(x_init)

def sample_goal():
    x = np.random.randint(0,100,1)
    x_goal = np.append(x,np.random.randint(95,100,1))
    return tuple(x_goal)


def main(args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obstacle_range = np.array([(20,40),
                           (20,40),
                           (20,40),
                           (60,80),
                           (60,80),
                           (20,40),
                           (60,80),
                           (60,80)])
    
    model = MixtureDensityNetwork(
        dim_in=20, # 4 x 5
        dim_out=2,
        n_components=args.n_components,
        hidden_dim=args.hidden_dim,
        noise_type=NoiseType.DIAGONAL,
    ).to(device=DEVICE)
    model.load_weight(f"mdn_weight_{args.num_epochs}_mode_{args.n_components}_hidden_{args.hidden_dim}_batch_{args.batch_size}.th")
    
    with open(args.path + 'examples/rrt_star/data/rrt_star_path_all.pkl','rb') as f:
        path_dict = pickle.load(f)
    # print(path_dict['1']['obstacle'][99])

    X_dimensions = np.array([(0, 100), (0, 100)])
    Obstacles = np.array([(20, 20, 40, 40), (20, 60, 40, 80), (60, 20, 80, 40), (60, 60, 80, 80)])
    Obstacles = sample_obstacles()


    x_init = (0, 0)  # starting location
    x_goal = (100, 100)  # goal location
    x_init = sample_init()
    x_goal = sample_goal()
    

    Q = np.array([(1, 1)])  # length of tree edges
    r = 1  # length of smallest edge to check for intersection with obstacles
    max_samples = 1024  # max number of samples to take before timing out
    rewire_count = 32  # optional, number of nearby branches to rewire
    prc = 0  # probability of checking for a connection to goal

    feat = torch.FloatTensor(np.append(np.array([]),Obstacles)).to(DEVICE)
    X = SearchSpace(X_dimensions,model, Obstacles)
    X.planning_scene_update(x_init, x_goal, feat)
    rrt = RRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)

    path = rrt.rrt_star()
    print(compute_cost(path))
    print(X.num_uni_sample)
    print(X.num_nn_sample)

    plot = Plot("rrt_star_2d")
    plot.plot_tree(X, rrt.trees)
    if path is not None:
        plot.plot_path(X, path)
    plot.plot_obstacles(X, Obstacles)
    plot.plot_start(X, x_init)
    plot.plot_goal(X, x_goal)
    plot.draw(auto_open=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_components', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--path', type=str, default='/hdisk/home/experiment/simple_experiment/rrt-algorithms-develop/')


    args = parser.parse_args()
    print(args)    
    main(args)