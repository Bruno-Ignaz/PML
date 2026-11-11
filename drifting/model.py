import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, noise_dim=32, hidden=256, depth=3):
        super().__init__()
        if min(noise_dim, hidden, depth) < 1:
            raise ValueError('Invalid model hyperparameters')
        
        self.noise_dim = noise_dim
        layers = [nn.Linear(noise_dim, hidden), nn.SiLU()]

        for _ in range(depth-1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]

        layers += [nn.Linear(hidden, 2)]
        self.net = nn.Sequential(*layers)
        
    def forward(self, eps):
        return self.net(eps)
