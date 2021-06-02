
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam
from utils import soft_update, hard_update, create_log_gaussian, logsumexp
from custom_nn_modules.Graph_NNs import GraphConvolution, GraphAggregation, MLP
from utilities.replay_memory import ReplayMemory
from agents.actor_critic_net import Actor, Critic
from utilities.RolloutBuffer import Rollout

# PPO as implemented in the paper https://arxiv.org/pdf/1707.06347.pdf

# Refines a protein using PPO
class PPOHbondAgent():
    # 
    def __init__(self, env, hyperparams):
        # Store Hyperparameters for learning
        self.gamma = hyperparams["discount_rate"]
        self.tau = hyperparams["tau"]
        self.alpha = hyperparams["alpha"]
        self.lr = hyperparams["lr"]
        self.policy_type = hyperparams["policy"]
        self.target_update_interval = hyperparams["target_update_interval"]
        self.automatic_entropy_tuning = hyperparams["automatic_entropy_tuning"]
        self.environment = env
        self.eps_clip = hyperparams["eps_clip"]
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
        self.policy_old = Actor(act_hyp["conv_dim"], self.node_dim, self.edge_dim, self.input_action_dim * 2, self.tensor_shapes, action_space=action_space).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)
        return
    
    # Choose an action depending on the state
    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0) # Unsqueeze batch dim
        action, log_prob, _ = self.policy.sample(state)
        return action

    # Calculates the losses and update the parameters of the network
    def update_parameters(self, buffer):        
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(buffer.logprobs, dim=0)).detach().to(self.device)

        
        # Optimize policy for some epochs
        for _ in range(100):

            # Evaluating old actions and values
            _, logprobs, mean = self.policy.sample(old_states, old_actions)
            state_values, _ = self.critic(old_states)
            dist_entropy = -torch.sum(logprobs * mean)
            
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * F.mse_loss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.critic_optim.zero_grad()
            self.policy_optim.zero_grad()
            loss.mean().backward()
            self.critic_optim.step()
            self.policy_optim.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        buffer.clear()
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
    

    