import sys
import os
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
from pyrosetta import *
from environments.SingleProtEnv import SingleProtEnv
from agents.replay_memory import ReplayMemory 
import itertools
import argparse
import datetime
import gym
import numpy as np
import itertools
from environments.SingleProtEnv import SingleProtEnv
import torch
from agents.hbond_agent_new import SAC
from torch.utils.tensorboard import SummaryWriter

'''
TODO
1. Set up hbond SAC agent by swapping code 
2. Try different types of action networks
    - First, one network predicts which residue's side chain to change (Vector of length num_residues),
    - The other calculate how much to each side chain angle of that residue (Vector of length max_side_chain_angles (i.e. 4))
    - Second, one network that finds how much to change all the side chains (vector of length num side chains) (maybe use skip connections)
'''


#Inputs
# Define some constants
PDB_DIR = "../data/1BDD/"
OCTREE_DIR = "../Octree"
OUTPUT_DIR = "../data/1BDD/output"
# Define some parameters
pdb_id = "1bdd" # The pdb id of the input protein
dcd_file = ""
pdb_init_file = PDB_DIR + "1bdd.pdb"
pdb_out_file = PDB_DIR + "1cq0_pdb2pqr_charmm-outn.pdb"
psf_file = PDB_DIR + "1bdd_pdb2pqr_charmm.psf"

# Hyperparameters
hyperparameters =  {
    "cuda": False,
    "replay_size": 10000000,
    "seed": 123456,
    "start_steps": 1000, # Num steps until we start sampling
    "batch_size": 256,
    "updates_per_step": 1,
    "num_steps": 10000, # Must match max_time_step in env
    "eval": True,
    "discount_rate": 0.99, # Must match the discount rate in env
    "tau": 0.005,
    "alpha": 0.2,
    "lr": 0.0003,
    "policy": "Gaussian",
    "target_update_interval": 1,
    "env_name": "SingleProtEnv",
    "automatic_entropy_tuning": True,
    "Env": {
        "torsions_to_change": "all",
        "adj_mat_type": "bonded",
        "step_size": 0.5,
        "discount_rate": 0.99,
        "discount_rate_threshold": 100,
        "max_time_step": 10000
    },
    "Actor": {
        "conv_dim":[[4 ,16], 128, [32, 64]],
    },
    "Critic": {
        "conv_dim":[[4 ,16], 128, [32, 64]],
        "z_dim": 8,
        "action_dim": [32, 32]
    }
}

# Extract hyperparams
replay_size = hyperparameters["replay_size"]
seed = hyperparameters["seed"]
start_steps = hyperparameters["start_steps"]
batch_size = hyperparameters["batch_size"]
updates_per_step = hyperparameters["updates_per_step"]
num_steps = hyperparameters["num_steps"]
policy = hyperparameters["policy"]
automatic_entropy_tuning = hyperparameters["automatic_entropy_tuning"]
env_name = hyperparameters["env_name"]
do_eval = hyperparameters["eval"]

# Initialize Rosetta
init()
# Init environ
env = SingleProtEnv(hyperparameters["Env"], seq="AA")
# Set seeds
env.seed(seed)
env.action_space.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
# Init Agent
agent = SAC(env.action_space, env, hyperparameters)

#Tesnorboard
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), env_name,
                                                             policy, "autotune" if automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(replay_size, seed)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > batch_size:
            # Number of updates per step in environment
            for i in range(updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        #print("---------------------")
        print("Action: {} | Reward: {}".format(action, reward))
        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > num_steps:
        break

    #writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if i_episode % 10 == 0 and do_eval is True:
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward


                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes


        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

env.close()