import numpy as np
import torch
from torch.utils import data

import matplotlib.pyplot as plt
import argparse
import pickle
from termcolor import colored
from blocks import MixtureDensityNetwork, NoiseType

class Dataset(data.Dataset):
    def __init__(self):
        self.path = '/hdisk/home/experiment/simple_experiment/rrt-algorithms-develop/examples/rrt_star/data/'
        self.dataset, self.labels = self.load_train_data()

    def __len__(self): return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]
    
    def load_train_data(self):
        with open(self.path + 'rrt_star_path_all.pkl', 'rb') as f:
            path_dict = pickle.load(f)
        
        names = path_dict.keys()
        _dataset = []
        _labels = []
        for i, name in enumerate(names):
            paths = path_dict[name]['path']
            feats = path_dict[name]['obstacle']
            
            for j in range(len(paths)):
                path = paths[j]
                goal = path[-1]
                feat = feats[j]
                # feat = 4x4 
                # Inpu를 위한 append() 후에 Inp : (20, ) 이 된다.
                for k in range(len(path)-1):
                    _data = np.zeros(4)
                    
                    _data[:2] = list(path[k]).copy()
                    _data[2:] = list(goal).copy()
                    Inp = np.append(_data,feat)
                    
                    _dataset.append(Inp)
                    _labels.append(path[k+1])
        
        dataset = np.array(_dataset)
        labels = np.array(_labels)
        
        shuffling = np.random.permutation(len(labels))
        
        dataset = torch.FloatTensor(dataset[shuffling])
        labels = torch.FloatTensor(labels[shuffling])

        return dataset, labels
    
class MixtureDensity(object):
    def __init__(self,args):
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.path = args.path
        self.n_components = args.n_components
        self.hidden_dim = args.hidden_dim
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("DEVICE: ", self.device)

        self.train_dataset = Dataset()
        self.train_loader = data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        
        self.model = MixtureDensityNetwork(
            dim_in=self.train_dataset.dataset.shape[1],
            dim_out=2,
            n_components=self.n_components,
            hidden_dim=self.hidden_dim,
            noise_type=NoiseType.DIAGONAL,
        ).to(device=self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.num_epochs
        )
        
        self.logger = {
            'loss':[],
        }
        
    def run(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            loss_per_epoch = []
            for batch_idx, (data, label) in enumerate(self.train_loader):
                data = data.to(self.device)
                label = label.to(self.device)
                
                self.optimizer.zero_grad()
                log_pi, mu, sigma = self.model(data)
                
                loss = self.model.loss(log_pi, mu, sigma, label)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                loss_per_epoch.append(loss.cpu().detach().numpy())
            self.logger['loss'].append(np.mean(loss_per_epoch))
                
            if epoch % 10 == 0:
                self.model.save_weight(f'mdn_weight_{self.num_epochs}_mode_{self.n_components}_hidden_{self.hidden_dim}_batch_{self.batch_size}.th')
                print(colored(f'LOSS per Epoch: {np.mean(loss_per_epoch)}','yellow'))
        self.model.save_weight(f"mdn_weight_{self.num_epochs}_mode_{self.n_components}_hidden_{self.hidden_dim}_batch_{self.batch_size}.th")

    def save_plot(self):
        y = self.logger['loss']
        x = np.arange(len(y))

        plt.plot(x,y)
        plt.title('Loss')
        plt.xlabel('Loss per epoch')
        plt.ylabel('Epochs')
        plt.savefig(self.path + f'loss_{self.num_epochs}_mode_{self.n_components}_hidden_{self.hidden_dim}_batch_{self.batch_size}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_components', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--path', type=str, default='/hdisk/home/experiment/simple_experiment/rrt-algorithms-develop/')

    args = parser.parse_args()
    print(args)
    mixture_density = MixtureDensity(args)
    mixture_density.run()
    mixture_density.save_plot()