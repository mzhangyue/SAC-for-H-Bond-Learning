# Imports
from octree import Protein
from pympler import asizeof, tracker, refbrowser
from memory_profiler import profile
import MDAnalysis as mda
from pyrosetta import *
from pyrosetta.toolbox import *
from pyrosetta.rosetta.core.scoring import ScoreType
from .bat import BAT
import nglview as nv
import numpy as np

# Class that allows you to query data about a specific protein and edit its
# configuration
class Prot:

    # Initializes the Prot Object
    # pdb_file: path to pdb
    # psf_file: path to psf
    def __init__(self, pdbFile=None, psfFile=None, mol2File=None, prmFile=None, rtfFile=None, aprmFile=None, outnFile=None, outoFile=None):
        '''
        MDANALYSIS WAY
        # Load in pdb and psf
        self.prot = mda.Universe(psf_file, psf_file)
        protein_residues = self.prot.select_atoms("protein")
        # Store the Cartesian coordinates
        self.cart = None
        # Create a bond adjacency list
        # Calculates a N x 4 numpy matrix representing the internal coordinates
        #self.intern = BAT(protein_residues) # N x 4 matrix representing the internal coordinates 
        #self.intern.run()
        # Backbone atoms
        '''
        # Load in pdb file
        if pdbFile != None:
            self.pose = pose_from_pdb(pdbFile)
            self.octree_rep = Prot(pdbFile, psfFile, mol2File, prmFile, rtfFile, aprmFile, outnFile, outoFile)
        # Default to alanine dipeptide
        else:
            self.pose = pose_from_sequence("AA", auto_termini=True)
        self.num_residues = self.pose.total_residues()
        self.num_atoms = self.pose.total_atoms()
        # Store secondary structure per atomAAAA
        self.pose.update_sec_struct()
        # Store Cartesian coordinates 
        self.cart_coords = np.zeros((self.num_atoms, 3))
        self.update_cart_coords()
        # Store bonded adjacency list
        self.bond_adj_list = self.get_bond_adj_list()
        self.sfxn = get_score_function(True)
        # Store hbond net adjacency list
        self.hbond_adj_list = {}
        # Generates chemical features
        self.atom_chem_features = self.generate_chemical_features()

        
        
    # Coords can be a pdb file or traj file (dcd, xtc,...)
    def visualize_protein(self, default=None, default_representation=False):
        # Select all atoms associated with a protein
        #protein_residues = self.prot.select_atoms("protein")
        #w = nv.show_mdanalysis(protein_residues, default=default, default_representation=default_representation)
        return 
    
    # Returns the internal coordinates
    # Converts the atomsitic positions to internal coordinates using mdanalysis
    # In interanl coordinates, each atom is defined by
    # 1. Bond Length (defined by 2 atoms)
    # 2. Bond Angle (defined by 3 atoms)
    # 3. Dihedral Angle (defined by 4 atoms)
    # 
    def get_internal_coords(self):
        return None

    # Returns the Cartesian coordinates
    def get_cart_coords(self):
        return self.cart_coords

    # Returns the Cartesian coordinates or the X, Y, Z position of each atom
    def update_cart_coords(self):
        pdb_atom_index = 0
        # Loop through all residues
        for res_index in range(1, len(self.num_residues + 1)):
            residue = self.pose.residue(res_index)
            # Loop through all atoms of each residue
            for atom in residue.atoms():
                xyz = atom.xyz() 
                self.cart_coords[pdb_atom_index, 0] = xyz[0]
                self.cart_coords[pdb_atom_index, 1] = xyz[1]
                self.cart_coords[pdb_atom_index, 2] = xyz[2]
                pdb_atom_index += 1
    
    def update_sec_struct(self):
        self.pose.display_secstruct()

    # The adjacency matrix showing which atoms are bonded to which atoms
    def get_bond_adj_list(self):
        global_atom_index  = 0
        bond_adj_list = {}
        # Loop through all residues
        for res_index in range(1, len(self.num_residues + 1)):
            residue = self.pose.residue(res_index)
            num_res_atoms = residue.num_atoms()
            # Loop through all atoms of each residue
            for res_atom_index in range(1, len(num_res_atoms)):
                neighbors = np.array(residue.bonded_neighbor(res_atom_index))
                bond_adj_list[global_atom_index] = neighbors
                global_atom_index += 1
        return bond_adj_list

    # Retuns the energy of the current configuration
    # Set only_hbond to true if you only want the hbond energy
    def set_score_function(self, only_hbond=False):
        if only_hbond:
            sfxn = ScoreFunction()
            sfxn.set_weight(ScoreType.hbond_lr_bb, 1.0)
            sfxn.set_weight(ScoreType.hbond_sr_bb, 1.0)
            sfxn.set_weight(ScoreType.hbond_bb_sc, 1.0)
            sfxn.set_weight(ScoreType.hbond_sc, 1.0)
        else:
            return get_score_function(True)
        
    # Returns an array of length 3 with the mainchain torsions
    def get_backbone_dihedrals(self, residue_index):
        return self.pose.residue(residue_index).mainchain_torsions()

    # Given a residue index, change the phi-psi angles of that residue
    # The dim of angle_change and cur_dihedrals are both vectors of length 2 
    # new_dihedrals = cur_dihedrals + angle_change 
    def set_backbone_dihedral(self, residue_index, angle_change):
        residue = self.pose.residue(residue_index)
        cur = [self.pose.phi(residue_index),  self.pose.psi(residue_index)]
        new_phi_psi = np.append(cur + np.array(angle_change), self.pose.omega(residue_index))
        residue.mainchain_torsions(Vector1(new_phi_psi))
        return

    # TODO
    def get_rotamer(self, residue):
        return
    
    # TODO
    def get_all_rotamers(self):
        return

    # Sets one of the rotamer angles (i.e. side chain angle) of a specific residues
    # If one of the rotamers is not specified, we assume it remains the same
    def set_rotamer(self, residue, rotamer_angles):
        return
    
    # TODO
    def set_all_rotamers(self, rotamer_angles):
        return

    # Returns the set of hydrogen bonds using rosetta
    def get_hbonds(self, rosetta=True):
        self.hbond_adj_list = {}
        # Loop through all residues
        for res_index in range(1, len(self.num_residues + 1)):
            residue = self.pose.residue(res_index)
            num_res_atoms = residue.num_atoms()
            # Loop through all atoms of each residue
            for atom_index in range(1, len(num_res_atoms)):
                residue = None
        return
                    
    
    # Generates atom features using Rosetta
    def generate_chemical_features(self):
        # Gather atom names
        # Gather atom partial charges
        # Gather whether atom is backbone
        chemical_features = []
        for res_index in range(1, len(self.num_residues + 1)):
            residue = self.pose.residue(res_index)
            num_res_atoms = residue.num_atoms()
            # Loop through all atoms of each residue
            for atom_index in range(1, len(num_res_atoms)):
                charge = residue.atomic_charge(atom_index)
                atom_name = residue.atom_name(atom_index)
                chemical_features.append([atom_name, charge])
        return chemical_features

    # Creates a graph for the current molecule
    # The node features are:
    # 1. A one-hot encoding of the atom name ()
    # 2. X, Y, Z coordinates
    # 3. Partial charge of each atom as determined by CHARMM 
    def get_graph(self):
        return


