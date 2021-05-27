from utils import tensors_to_flat
from environments.EnvBase import EnvBase
from mol_processors.Protein import Prot
import numpy as np
import torch
import gym
from gym import spaces
from gym.utils import seeding
import math


class SingleProtEnv(gym.Env):
    
    # Initialize the protein environment
    def __init__(self, hyperparameters, pdb_file=None, seq=None, psf_file=None, mol2_file=None, prm_file=None, rtf_file=None, aprm_file=None, pdb_id=None):
        metadata = {
            'render.modes': ['ansi'],
        }
        # Extract hyperparameters
        self.adj_mat_type = hyperparameters["adj_mat_type"]
        self.step_size = hyperparameters["step_size"]
        self.discount_rate = hyperparameters["discount_rate"]
        self.discount_rate_threshold = hyperparameters["discount_rate_threshold"]
        self.max_time_step = hyperparameters["max_time_step"]
        self.time_step = 0
        # Initialize protein
        self.prot = Prot(pdb_file, psf_file, mol2_file, prm_file, rtf_file, aprm_file, pdb_id, seq=seq)
        # Get list of dihedral angles that the agent is allowed to change
        self.torsion_ids_to_change = self.prot.get_torsion_ids(torsion_type=hyperparameters["torsions_to_change"])
        # Get features
        self.features = self.prot.atom_chem_features
        # Get adjacency matrix
        self.bond_adj_mat = self.prot.bond_adj_mat
        # self.update_adj_list()
        # Gym Variables
        self._max_episode_steps = self.max_time_step
        self.min_action = -30
        self.max_action = 30
        self.low_state = -np.finfo(np.float32).max
        self.high_state = np.finfo(np.float32).max
        # Current Score
        self.cur_score = self.prot.get_score()
        # Sets the dimensions of the action space (2 x num_torsions_to_change)
        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(len(self.torsion_ids_to_change),),
            dtype=np.float32
        )
        num_atoms, num_features = np.shape(self.features)
        # Sets the dims of the observation space (num_atoms x (num_features + num_atoms))
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            shape=(num_atoms, num_features + num_atoms),
            dtype=np.float32
        )
        return

    # Gets an adjacency list of atoms. the adjacency list can either be based
    # on the covalent bonds or hydrogen bonds
    def update_adj_mat(self):
        if self.adj_mat_type == "bonded":
            self.bond_adj_mat = self.prot.bond_adj_mat
        else:
            self.bond_adj_mat = self.prot.get_hbonds()
        return self.bond_adj_mat
    
    # Applies action to the environment, transitions to next state, and returns reward
    def apply_action(self, angle_change, save=True):
        angle_change = np.tanh(angle_change)
        if save:
            self.prot.write_to_pdb("./results/pdbs/" + self.prot.pdb_name + str(self.time_step) + ".pdb")
        # Perturb torsion angles by angle change to transition to next state
        self.prot.perturb_torsion_ids(self.torsion_ids_to_change, angle_change)
        self.prot.update_cart_coords()
        if self.adj_mat_type == "hbond_net": # Only hbonds could change
            self.update_adj_mat()
        # Get reward
        #reward = -(self.prot.get_score() - self.cur_score)
        reward = self.get_reward(angle_change)
        # Increment time step
        self.time_step += 1
        return reward

    # Returns the state which consists of
    # 1. Feature matrix (num_atoms x num_features)
    # 2. Adjacency List (num_atoms x num_neighbors_per_atom)
    def get_state(self):
        flat_features = np.ndarray.flatten(self.features)
        flat_adj_mat = self.bond_adj_mat.reshape(-1)
        return np.concatenate((flat_features, flat_adj_mat), axis=None)
    
    # Given the angle_chain in radians gets the reward
    # Computes $r(s_t, a_t) \gets e^{\gamma t/T}[(\sum_{j=1}^M \dot{d}_j^2)/2-E_t]$
    def get_reward(self, angle_change):
        # e^{\gamma t/T}
        exp_term = np.exp(self.discount_rate * self.time_step / self.discount_rate_threshold)
        energy = self.prot.get_score() # E_t
        # \gamma t/T}[(\sum_{j=1}^M \dot{d}_j^2)/2-E_t
        return -energy
        #return exp_term * (np.sum(angle_change ** 2)/2 - energy)

    # Checks if we are in terminal state
    def is_terminal_state(self):
        if self.time_step == self.max_time_step:
            return True
        return False
    
    # Resets the episode by sampling from ramachandran distribution and then
    # backbone dependent rotamers
    def reset(self):
        self.prot.sample_rama_backbone_dihedrals()
        self.prot.sample_bbind_rotamers()
        self.prot.update_cart_coords()
        self.update_adj_mat()
        return self.get_state()
        
    # Returns the next state
    def step(self, action):
        reward = self.apply_action(action)
        state = self.get_state()
        #done = self.is_terminal_state()
        done = False
        return state, reward, done, {}

    # Render hbond net as a string
    def render(self, mode='ansi'):
        # Only allow ansi
        if mode != 'ansi':
            raise Exception('Render mode for now can only be ansi')
        return "Not Implemented"
    
    # Sets a random seed for the gym environment
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
