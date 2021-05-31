
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam
from utils import soft_update, hard_update, create_log_gaussian, logsumexp
from custom_nn_modules.Graph_NNs import GraphConvolution, GraphAggregation, MLP
from agents.replay_memory import ReplayMemory
from agents.actor_critic_net import Actor, Critic


# Refines a protein using PPO
class PPOHbondAgent():

    def __init__():
        self.actor = 

        self.critic = 
        return
    
    def ppo():


    