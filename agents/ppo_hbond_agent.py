
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam
from utils import soft_update, hard_update, create_log_gaussian, logsumexp
from custom_nn_modules.Graph_NNs import GraphConvolution, GraphAggregation, MLP
from utilities.replay_memory import ReplayMemory
from agents.actor_critic_net import Actor, Critic

# PPO as implemented in the paper https://arxiv.org/pdf/1707.06347.pdf

# Refines a protein using PPO
class PPOHbondAgent():
    # 
    def __init__():
        # Store Hyperparameters for learning
        self.gamma = hyperparams["discount_rate"]
        self.tau = hyperparams["tau"]
        self.alpha = hyperparams["alpha"]
        self.lr = hyperparams["lr"]
        self.policy_type = hyperparams["policy"]
        self.target_update_interval = hyperparams["target_update_interval"]
        self.automatic_entropy_tuning = hyperparams["automatic_entropy_tuning"]
        self.environment = env
        self.device = hyperparams['device']
        crit_hyp = hyperparams["Critic"]
        act_hyp = hyperparams["Actor"]
        # Store Dimensions  t
        self.node_dim = len(self.environment.prot.atom_chem_features[0])
        self.num_nodes = len(self.environment.prot.atom_chem_features)
        self.edge_dim = 1
        self.input_action_dim = len(self.environment.torsion_ids_to_change)
        self.tensor_shapes = [(self.num_nodes, self.node_dim), (self.num_nodes, self.num_nodes)]
        
        print("We will change", self.input_action_dim, "torsions")
        
        print("Preparing Critic network...")
        self.critic = Critic(crit_hyp["conv_dim"], self.node_dim, self.edge_dim, crit_hyp["z_dim"], crit_hyp["action_dim"], self.input_action_dim, self.tensor_shapes).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        print("Preparing Actor network")
        self.policy = Actor(act_hyp["conv_dim"], self.node_dim, self.edge_dim, self.input_action_dim * 2, self.tensor_shapes, action_space=action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)
        return
    
    # Choose an action depending on the state
    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0) # Unsqueeze batch dim
        action, log_prob, _ = self.sample(state)
        return action

    # Calculates the losses and update the parameters of the network
    def update_parameters(self, buffer):        
        # Sample a batch from the replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # Calculate the ratio
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * adv_targ
        action_loss = -torch.min(surr1, surr2).mean()
        return

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_input=None, critic_input=None, actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/ppo_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/ppo_critic_{}_{}".format(env_name, suffix)
        # Saev as onnx just to store topology
        if suffix == ".onnx" and actor_input != None and critic_input != None:
            torch.onnx.export(self.policy, actor_input, actor_path)
            torch.onnx.export(self.critic, critic_input, critic_path)
        # Save pytorch state dict
        elif suffix != ".onnx":
            torch.save(self.policy.state_dict(), actor_path)
            torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
    

    