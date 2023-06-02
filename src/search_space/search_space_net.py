# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import torch
import numpy as np
from scipy.stats import truncnorm
from rtree import index

from src.utilities.geometry import es_points_along_line
from src.utilities.obstacle_generation import obstacle_generator


class SearchSpaceNet(object):
    def __init__(self, dimension_lengths, model ,O=None):
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
        self.model = model
        self.c_c = None
        self.goal = None
        self.feat = None
        self.N = 0
        self.num_nn_sample = 0
        self.num_uni_sample = 0

        self.num_collision = 0
        if self.num_collision == 5:
            self.num_uni_sample += 1

    


    # 현재 포인트가 장애물 안, 밖인지 판단
    def obstacle_free(self, x):
        """
        Check if a location resides inside of an obstacle
        :param x: location to check
        :return: True if not inside an obstacle, False otherwise
        """
        # self.obs.count(x): 장애물 외부이면 0 (false), 내부이면 1 (true)
        
        return self.obs.count(x) == 0 
    
    # def obstacle_free_sample(self):
    #     while True:
    #         x = self.sample()
    #         if self.obstacle_free(x):
    #             self.num_uni_sample +=1
    #             self.c_c = x
    #             return x

    def sample_free(self):
        """
        Sample a location within X_free
        :return: random location within X_free
        """
        # print(self.num_collision)
        # if self.num_collision > 5:
        #     x = self.sample()
        #     # print(self.num_collision)
        # else:
        #     x = self.sample_from_nn()
        # return x
        
        while True:
            if self.N >= 5 or self.num_collision > 5:
                x = self.sample()
                self.N = 0
            else:
                x = self.sample_from_nn()
            if self.obstacle_free(x):
               
                self.c_c = x
                
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
        log_pi, mu, sigma = self.model(Inp)
        log_pi, mu, sigma = log_pi.cpu().detach().numpy().squeeze(), mu.cpu().detach().numpy().squeeze(), sigma.cpu().detach().numpy().squeeze()
        choose_gaussian_idx = np.exp(log_pi).argmax()
        # print(Inp)
        # print(mu)
        
        x = truncnorm.rvs((self.dimension_lengths[:,0] - mu[choose_gaussian_idx,:])/sigma[choose_gaussian_idx,:],(self.dimension_lengths[:,1] - mu[choose_gaussian_idx,:])/sigma[choose_gaussian_idx,:],loc= mu[choose_gaussian_idx,:],scale=sigma[choose_gaussian_idx,:])
        # x = np.random.uniform(self.dimension_lengths[:, 0], self.dimension_lengths[:, 1]) * sigma[choose_gaussian_idx,:] + mu[choose_gaussian_idx,:]
        
        # print(x)
        return tuple(x)