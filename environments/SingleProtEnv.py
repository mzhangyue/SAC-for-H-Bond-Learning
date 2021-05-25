from environments.EnvBase import EnvBase
from mol_processors.Protein import Prot
import numpy as np
import torch


class SingleProtEnv(EnvBase):
    
    # Initialize the protein environment
    def __init__(self, hyperparameters, pdb_file, psf_file=None, mol2_file=None, prm_file=None, rtf_file=None, aprm_file=None, pdb_id=None):
        EnvBase.__init__(self)
        # Extract hyperparameters
        self.adj_list_type = hyperparameters["adj_list_type"]
        self.step_size = hyperparameters["step_size"]
        self.discount_rate = hyperparameters["discount_rate"]
        self.discount_rate_threshold = hyperparameters["discount_rate_threshold"]
        self.time_step = 0
        # Initialize protein
        self.prot = Prot(pdb_file, psf_file, mol2_file, prm_file, rtf_file, aprm_file, pdb_id)
        # Get list of dihedral angles that the agent is allowed to change
        self.torsion_ids_to_change = self.prot.get_torsion_ids(torsion_type=hyperparameters["torsions_to_change"])
        # Get features
        self.features = self.prot.atom_chem_features
        # Get adjacency list
        self.adj_list = None
        self.update_adj_list()
        return

    # Gets an adjacency list of atoms. the adjacency list can either be based
    # on the covalent bonds or hydrogen bonds
    def update_adj_list(self):
        if self.adj_list_type == "bonded":
            self.adj_list = self.prot.bond_adj_list
        else:
            self.adj_list = self.prot.get_hbonds()
        return self.adj_list
    
    # Applies action to the environment, transitions to next state, and returns reward
    def apply_action(self, angle_change):
        # Convert angle change to degrees
        angle_change_in_deg = torch.deg2rad(angle_change)
        # Perturb torsion angles by angle change to transition to next state
        self.prot.perturb_torsion_ids(self.torsion_ids_to_change, angle_change_in_deg)
        self.prot.update_cart_coords()
        if self.adj_list_type == "hbond_net": # Only hbonds could change
            self.update_adj_list()
        # Get reward
        reward = self.get_reward(angle_change)
        # Increment time step
        self.time_step += 1
        return

    # Returns the state which consists of
    # 1. Feature matrix (num_atoms x num_features)
    # 2. Adjacency List (num_atoms x num_neighbors_per_atom)
    def get_state(self):
        return self.features, self.adj_list
    
    # Given the angle_chain in radians gets the reward
    # Computes $r(s_t, a_t) \gets e^{\gamma t/T}[(\sum_{j=1}^M \dot{d}_j^2)/2-E_t]$
    def get_reward(self, angle_change):
        # e^{\gamma t/T}
        exp_term = torch.exp(self.discount_rate * self.time_step / self.discount_rate_threshold)
        energy = self.prot.get_score() # E_t
        # \gamma t/T}[(\sum_{j=1}^M \dot{d}_j^2)/2-E_t
        return torch.sum(torch.tanh(angle_change) ** 2)/2 - energy
        