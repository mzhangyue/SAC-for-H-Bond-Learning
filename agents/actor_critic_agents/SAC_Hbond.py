import numpy as np
from agents.Base_Agent import Base_Agent
from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
from mol_processors.Protein import Prot
from utils import generate_one_hot_encoding



class SAC_Hbond(SAC_Discrete):

    def __init__(self, config, pdb_file, psf_file=None, mol2_file=None, prm_file=None, rtf_file=None, aprm_file=None, pdb_id=None):
        Base_Agent.__init__(self, config)
        self.prot_agent = Prot(pdb_file, psf_file, mol2_file, prm_file, rtf_file, aprm_file, pdb_id)
        return

    # Atomic Features
    # 1. X, Y, Z
    # 2. Chemical Features
    def create_features(self):
        features = np.concatenate([self.get_cart_coords(), self.generate_chemical_features()])
        return features


