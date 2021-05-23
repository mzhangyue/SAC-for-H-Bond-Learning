import numpy as np
from agents.Base_Agent import Base_Agent
from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
from mol_processors.Protein import Prot
from utils import generate_one_hot_encoding

path_to_pdb = "/home/conradli/SAC-for-H-Bond-Learning/data/1BDD/1bdd_pnon_charmm.pdb"
path_to_psf = "/home/conradli/SAC-for-H-Bond-Learning/data/1BDD/1bdd_pnon_charmm.psf"


class SAC_Hbond(SAC_Discrete):

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.prot_agent = Prot(pdb_file=path_to_pdb, psf_file=path_to_psf)
        return



    # Atomic Features
    # 1. X, Y, Z
    # 2. Chemical Features

    def create_features(self):
        features = np.concatenate([self.get_cart_coords(), self.generate_chemical_features()])
        return features


