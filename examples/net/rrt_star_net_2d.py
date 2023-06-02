import argparse
import pickle
import torch
import numpy as np


from src.rrt_net.rrt_star_net import RRTStarNet
from src.search_space.search_space_net import SearchSpaceNet
from src.utilities.plotting import Plot

from MDN.blocks import MixtureDensityNetwork, NoiseType


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

    """
    평가 항목 : cost
    
    - 결과 path 중에 nn 샘플과 uniform 샘플의 개수 
        - 즉, path 중 대부분이 nn에서 샘플링 되었음을 통해 NN이 학습이 잘 됬음을 보인다.
        - 이 때 nn 샘플의 충돌 개수를 보고, 현재 알고리즘의 단점을 어필한다. 
    
    - 결과 path의 cost 
        - 현재 방식과, 제안하는 방식의 cost를 보고, 학습된 내용과 다른 환경에서 제안하는 알고리즘의 강점을 보인다.

    알고리즘 정리:
    1. NN 에서 장애물이 없는 공간에서 x_rand 샘플링 시도 (장애물이 있는 공간에서 샘플링 안함)
    2. x_new 를 찾는다.
    3. x_new 가 충돌일 경우 nn에서 최대 5번까지 샘플링을 시도하고, 5회 이후 uniform에서 샘플링을 수행한다.


    """
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
    model.load_weight(f"model/mdn_weight_{args.num_epochs}_mode_{args.n_components}_hidden_{args.hidden_dim}_batch_{args.batch_size}.th")
    
    # with open(args.path + 'examples/rrt_star/data/rrt_star_path_all.pkl','rb') as f:
    #     path_dict = pickle.load(f)
   

    X_dimensions = np.array([(0, 100), (0, 100)])
    Obstacles = np.array([(20, 20, 40, 40), (20, 60, 40, 80), (60, 20, 80, 40), (60, 60, 80, 80)])
    x_init = (0, 0)  # starting location
    x_goal = (100, 100)  # goal location
    
    # Obstacles = sample_obstacles(obstacle_range)
    # x_init = sample_init()
    # x_goal = sample_goal()
    
    # q[0] 가 x_new 선택
    Q = np.array([(1, 1)])  # length of tree edges
    r = 0.1  # length of smallest edge to check for intersection with obstacles
    max_samples = 21024  # max number of samples to take before timing out
    rewire_count = 32  # optional, number of nearby branches to rewire
    prc = 0.1  # probability of checking for a connection to goal
    feat = torch.FloatTensor(np.append(np.array([]),Obstacles)).to(DEVICE)
    
    X = SearchSpaceNet(X_dimensions,model, Obstacles)
    X.planning_scene_update(x_init, x_goal, feat)

    rrt = RRTStarNet(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)
    path = rrt.rrt_star()
    
    print('cost: ',compute_cost(path))
    print('uni samples: ',X.num_uni_sample)
    print("nn_samples: ",X.num_nn_sample)

    # plot
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