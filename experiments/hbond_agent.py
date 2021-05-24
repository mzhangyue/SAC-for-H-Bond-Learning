import sys
import os
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
from agents.actor_critic_agents.SAC_Hbond import SAC_Hbond
from utilities.data_structures.Config import Config
import gym
from pyrosetta import *

# Configs
config = Config()
config.seed = 1
config.environment = gym.make("CartPole-v0")
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
# Hyperparameters
config.hyperparameters = {
    "Actor_Critic_Agents":  {

        "learning_rate": 0.005,
        "linear_hidden_units": [20, 10],
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": 5.0,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 1.0,
        "normalise_rewards": True,
        "exploration_worker_difference": 2.0,
        "clip_rewards": False,

        "Actor": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": "Softmax",
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "Critic": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 1000000,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "min_steps_before_learning": 400,
        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0, #for O-H noise
        "theta": 0.15, #for O-H noise
        "sigma": 0.25, #for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 1,
        "learning_updates_per_learning_session": 1,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": False,
        "do_evaluation_iterations": True
    }
}

# Initialize Rosetta
init()

#Inputs
# Define some constants
PDB_DIR = "./data/1BDD/"
OCTREE_DIR = "./Octree"
OUTPUT_DIR = "./data/1BDD/output"
# Define some parameters
pdb_id = "1bdd" # The pdb id of the input protein
dcd_file = ""
pdb_init_file = PDB_DIR + "1bdd.pdb"
pdb_out_file = PDB_DIR + "1cq0_pdb2pqr_charmm-outn.pdb"
psf_file = PDB_DIR + "1bdd_pdb2pqr_charmm.psf"
# Set flagspdb_file = "./data/"

tester = SAC_Hbond(config)
tester.create_features()
