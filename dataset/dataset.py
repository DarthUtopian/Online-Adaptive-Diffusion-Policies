import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import d4rl
import tqdm
import sklearn
import sklearn.datasets
import scipy

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, name, size=1000000):
        assert name in ["swissroll", "8gaussians", "moons", "rings", "checkerboard", "2spirals"]
        self.data_size = size
        self.name = name
        self.datas, self.energy = train_data_gen(name, batch_size=size)
        self.datadim = 2
      
    def __getitem__(self, index):
        return {"a": self.datas[index], "e": self.energy[index]}

    def __add__(self, other):
        raise NotImplementedError

    def __len__(self):
        return self.data_size
    
    def get_formatted_data(self, obs_dim):
        observations = np.zeros((self.data_size, obs_dim))
        return {
            "observations": observations,
            "actions": self.datas,
            "next_observations": observations,
            "rewards": self.energy,
            "terminals": np.ones((self.data_size, 1)),
        }
    
    
def energy_sample(name, beta, sample_per_state=1000, **kwargs):
    data, e = train_data_gen(name, batch_size=1000*sample_per_state)
    idx = np.random.choice(1000*sample_per_state, p=scipy.special.softmax(beta*e).squeeze(), size=sample_per_state, replace=False)
    data = data[idx]
    e = e[idx]
    return data, e
     
    
def train_data_gen(data, batch_size=256):
    print(f"generating {data} data")
    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data, np.sum(data**2, axis=-1,keepdims=True) / 9.0
    
    elif data == "8gaussians":
        scale = 4.
        centers = [
                   (0, 1), 
                   (-1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (-1, 0), 
                   (-1. / np.sqrt(2), -1. / np.sqrt(2)),
                   (0, -1),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)),
                    (1, 0), 
                   (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   ]
        centers = [(scale * x, scale * y) for x, y in centers]
        dataset = []
        idxs = []
        for i in range(batch_size):
            point = np.random.randn(2) * 0.5
            idx = np.random.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            idxs.append(idx)
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset, np.array(idxs, dtype="float32")[:, None] / 7.0
    
    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data, np.sum(data**2, axis=-1,keepdims=True) / 9.0
    
    elif data == "rings":
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25
        
        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        #X = util_shuffle(X)

        center_dist = X[:,0]**2 + X[:,1]**2
        energy = np.zeros_like(center_dist)

        energy[(center_dist >=8.5)] = 0.667 
        energy[(center_dist >=5.0) & (center_dist <8.5)] = 0.333 
        energy[(center_dist >=2.0) & (center_dist <5.0)] = 1.0 
        energy[(center_dist <2.0)] = 0.0

        # Add noise
        X = X + np.random.normal(scale=0.08, size=X.shape)

        return X.astype("float32"), energy[:,None]