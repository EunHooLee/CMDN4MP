import numpy as np
import time
import argparse
import pickle

from src.rrt.rrt_star import RRTStar
from src.search_space.search_space import SearchSpace
from src.utilities.plotting import Plot


class PathPlanning(object):
    def __init__(self):
        self.X_dimensions = np.array([(0, 100), (0, 100)])
        self.X_init = (0,0)
        self.X_goal = (100, 100)
        self.Q = np.array([(1,1)])
        self.obstacles = None
        self.max_samples = 1024
        self.rewire_count =32
        self.prc = 0
        self.r = 1
        self.path = '/hdisk/home/experiment/simple_experiment/rrt-algorithms-develop/examples/rrt_star/data/'

        self.obstacle_range = np.array([(20,40),
                                        (20,40),
                                        (20,40),
                                        (60,80),
                                        (60,80),
                                        (20,40),
                                        (60,80),
                                        (60,80)])
        
    def sample_obstacles(self):
        obs_center_pose = np.random.uniform(self.obstacle_range[:,0], self.obstacle_range[:,1])
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

    def sample_init(self):
        x = np.random.randint(0,100,1)
        x_init = np.append(x,np.random.randint(0,5,1))
        
        return tuple(x_init)

    def sample_goal(self):
        x = np.random.randint(0,100,1)
        x_goal = np.append(x,np.random.randint(95,100,1))
        return tuple(x_goal)
    
    def compute_cost(self,path):
        state_dists = []
        for i in range(len(path) -1):
            dist = 0
            for j in range(2):
                diff = path[i][j] - path[i+1][j]
                dist += diff ** 2
            state_dists.append(np.sqrt(dist))
        total_cost = sum(state_dists)
        return total_cost

    def run(self):
        success = False
        path_dict = {}
        # num of envs
        for i in range(20):
            self.obstacles = self.sample_obstacles()
            path_dict[f'{i}'] = {}
            path_dict[f'{i}']['path'] = []
            path_dict[f'{i}']['cost'] = []
            path_dict[f'{i}']['time'] = []
            path_dict[f'{i}']['obstacle'] = []
            # num of init-goal pairs
            feasible_path = 0
            while feasible_path < 100:
                self.x_init = self.sample_init()
                self.x_goal = self.sample_goal()
                self.X = SearchSpace(self.X_dimensions, self.obstacles)
                rrt = RRTStar(
                    self.X,
                    self.Q,
                    self.x_init,
                    self.x_goal,
                    self.max_samples,
                    self.r,
                    self.prc,
                    self.rewire_count
                )
                start_time = time.time()
                path = rrt.rrt_star()
                planning_time = time.time() - start_time
                if path == None:
                    continue

                cost = self.compute_cost(path)

                path_dict[f'{i}']['path'].append(path)
                path_dict[f'{i}']['cost'].append(cost)
                path_dict[f'{i}']['time'].append(planning_time)
                path_dict[f'{i}']['obstacle'].append(self.obstacles)
                feasible_path += 1
            
            with open(self.path + f'rrt_star_path_{i}.pkl','wb') as f:
                pickle.dump(path_dict[f'{i}'],f)
        
        with open(self.path + 'rrt_star_path_all.pkl', 'wb') as f:
            pickle.dump(path_dict,f)
        success = True

        return success

def main():
    plan = PathPlanning()
    success = plan.run()



if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument()

    main()