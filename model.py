import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


"""
Duckietown observation space is (480, 640, 3) and action space is (2,)
1. we grrayscale the image to (480, 640)
2. we stack 4 frames to make the observation space (4, 480, 640)


"""
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            (nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            (nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),  # (4, 64, 7, 7) -> (4 * 64 * 7 * 7)
        )
        self.fc = nn.Sequential(
            (nn.Linear(3136, 512)),
            nn.ReLU(),
        )
        self.actor_mean = (nn.Linear(512, 1))
        self.actor_logstd = nn.Parameter(torch.zeros(1, 1))

        #self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = (nn.Linear(512, 1))

    def get_value(self, x):
        if x.dim() == 3:  # input (C, H, W) instead of (B, C, H, W)
            x = x.unsqueeze(0)
        x = x / 255.0  # Normalize image input
        hidden = self.fc(self.network(x))
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None):
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = x / 255.0  # Normalize

        hidden = self.network(x)
        hidden = self.fc(hidden)
        mean = self.actor_mean(hidden)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()

            #action = torch.tanh(action) 
        #print(f'dim of returns {action.shape} and {dist.log_prob(action).shape} and {dist.entropy().shape} and {self.critic(hidden).shape}')
        return (
            action,
            dist.log_prob(action),
            dist.entropy(),
            self.critic(hidden),
        ) 
