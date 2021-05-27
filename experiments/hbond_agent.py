from environments.SingleProtEnv import SingleProtEnv
import sys
import os
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
from agents.actor_critic_agents.SAC_Hbond import SAC_Hbond
from utilities.data_structures.Config import Config
import gym
from pyrosetta import *
from agents.Trainer import Trainer
from environments.SingleProtEnv import SingleProtEnv

'''
TODO
1. Set up hbond SAC agent by swapping code 
2. Try different types of action networks
    - First, one network predicts which residue's side chain to change (Vector of length num_residues),
    - The other calculate how much to each side chain angle of that residue (Vector of length max_side_chain_angles (i.e. 4))
    - Second, one network that finds how much to change all the side chains (vector of length num side chains) (maybe use skip connections)
'''
# Configs
config = Config()
config.seed = 1
config.num_episodes_to_run = 450
config.file_to_save_data_results = "results/hbond.pkl"
config.file_to_save_results_graph = "results/hbond.png"
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = False
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False
config.debug_mode = False

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
config.environment = SingleProtEnv(config.hyperparameters, pdb_file=pdb_init_file)
# Initialize Rosetta
init()
# Start up the hbond agent
hbond_agent = SAC_Hbond(config)
# Do 200 runs of the agent
for run in range(1, 200):
    game_scores, rolling_scores, time_taken = hbond_agent.run_n_episodes()
    print("Run {} | Game Scores: {} | Rolling Scores: {} | Time taken: {}".format(game_scores, rolling_scores, time_taken))