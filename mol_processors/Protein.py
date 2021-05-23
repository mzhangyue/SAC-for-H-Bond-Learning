# Imports
from Octree.octree import Protein
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
    def __init__(self, pdbFile=None, psfFile=None, mol2File=None, prmFile=None, rtfFile=None, aprmFile=None, outnFile=None, outoFile=None, pdb_id=None):
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
            print("Loading pdb fie into Roseta...")
            self.pose = pose_from_pdb(pdbFile)
            #self.octree_rep = Protein(pdbFile, psfFile, mol2File, prmFile, rtfFile, aprmFile, outnFile, outoFile)
        elif pdb_id != None:
            print("Loading pdb id into Rosetta")
            self.pose = pose_from_rcsb(pdb_id)
        # Default to alanine dipeptide
        else:
            self.pose = pose_from_sequence("AA", auto_termini=True)
        # Get total atoms and total residues
        self.num_residues = self.pose.total_residue()
        self.num_atoms = self.pose.total_atoms()
        # Store atom ids
        self.atom_ids = self.get_atom_ids()
        # Store secondary structure per atom
        self.update_sec_struct()
        # Store Cartesian coordinates 
        self.cart_coords = np.zeros((self.num_atoms, 3))
        self.update_cart_coords()
        # Store bonded adjacency list
        self.bond_adj_list = self.get_bond_adj_list()
        # Store hbond net adjacency list
        self.hbond_set = None
        #self.get_hbonds()
        # Generates chemical features
        self.atom_chem_features = self.generate_chemical_features()
        # Set the scoring functon
        self.score_function = get_score_function(True)

    # Writes the protein configuration to a PDB file
    def write_to_pdb(self, output_file):
        self.pose.dump_pdb(output_file)
        return

    # Returns a numpy array of atom ids in 
    # Atom ID stores the residue id and atom number: 
    # 1. atom_id.rsd()
    # 2. atom_id.atomno()
    def get_atom_ids(self):
        atom_ids = []
        for res_index in range(1, self.num_residues + 1):
            residue = self.pose.residue(res_index)
            num_res_atoms = residue.natoms()
            # Loop through all atoms of each residue
            for res_atom_index in range(1, num_res_atoms+1):
                atom_ids.append(AtomID(res_atom_index, res_index))
        return atom_ids

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
    
    ################# CARTESIAN COORDINATES #################

    # Returns the Cartesian coordinates
    def get_cart_coords(self):
        return self.cart_coords

    # Returns the Cartesian coordinates or the X, Y, Z position of each atom
    def update_cart_coords(self):
        pdb_atom_index = 0
        # Loop through all residues
        for res_index in range(1, self.num_residues + 1):
            residue = self.pose.residue(res_index)
            num_res_atoms = residue.natoms()
            # Loop through all atoms of each residue
            for res_atom_index in range(1, num_res_atoms+1):
                xyz = residue.xyz(res_atom_index) 
                self.cart_coords[pdb_atom_index, 0] = xyz[0]
                self.cart_coords[pdb_atom_index, 1] = xyz[1]
                self.cart_coords[pdb_atom_index, 2] = xyz[2]
                pdb_atom_index += 1

    ################# SECONDARY STRUCTURES #################

    # Gets the DSSP secondary structure assignment of the residue
    def get_sec_struct(self, residue_index):
        return self.pose.secstruct(residue_index)

    # Updates the scondary structure data
    def update_sec_struct(self):
        self.pose.display_secstruct()

    
    ################# COVALENT BONDS #################

    # The adjacency matrix showing which atoms are bonded to which atoms
    def get_bond_adj_list(self):
        global_atom_index  = 0
        bond_adj_list = {}
        # Loop through all residues
        for res_index in range(1, self.num_residues + 1):
            residue = self.pose.residue(res_index)
            num_res_atoms = residue.natoms()
            # Loop through all atoms of each residue
            for res_atom_index in range(1, num_res_atoms+1):
                neighbors = np.array(residue.bonded_neighbor(res_atom_index))
                bond_adj_list[global_atom_index] = neighbors
                global_atom_index += 1
        return bond_adj_list

    ################# SCORING FUNCTION #################
    
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
    
    # Gets the score of the current protein conformation
    def get_current_score(self):
        return self.score_function(self.pose)
        
    ################# BACKBONE DIHEDRALS (PHI, PSI, OMEGA) #################

    # Returns an array of length 3 with the mainchain torsions
    def get_backbone_dihedrals(self, residue_index):
        return self.pose.residue(residue_index).mainchain_torsions()

    # Given a residue index, change the phi-psi angles of that residue
    # The dim of angle_change and cur_dihedrals are both vectors of length 2 
    # new_dihedrals = cur_dihedrals + angle_change 
    def set_backbone_dihedrals(self, residue_index, angle_change):
        residue = self.pose.residue(residue_index)
        cur = [self.pose.phi(residue_index),  self.pose.psi(residue_index)]
        new_phi_psi = np.append(cur + np.array(angle_change), self.pose.omega(residue_index))
        residue.mainchain_torsions(Vector1(new_phi_psi))
        return
    
    # Sample backbone dihedrals from a ramachanndran distribution
    def sample_backbone_dihedrals():
        return

    ################# SIDECHAIN DIHEDRALS (CHI) ROTAMERS #################

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

    ################# HYDROGEN BONDS #################

    # Returns a dictionary mapping from an accceptor id to donor atom id 
    # used in the hydrogen bonding network
    def get_hbonds(self, rosetta=True):
        hbond_adj_list = {}
        # Generate the hbonds 
        self.hbond_set = self.pose.get_hbonds()
        # Loop through all hbonds 
        for hbond in self.hbond_set.hbonds():
            # Grab acceptor atom and residue index
            acc_res_index = hbond.acc_res()
            acc_atom_index = hbond.acc_atm()
            # Grab donor atom and residue index
            don_res_index = hbond.don_res()
            # For the donor atom, we want its parent not the actual hydrogen
            don_atom_index = self.pose.residue(don_res_index).atom_base(hbond.don_hatm())
            # Make atom id
            acc_atom_id = AtomID(acc_atom_index, acc_res_index) 
            don_atom_id = AtomID(don_atom_index, don_res_index) 
            hbond_adj_list[acc_atom_id] = don_atom_id
        return hbond_adj_list

    def print_hbond_adj_list(self, hbond_adj_list):
        for acc_atom_id in hbond_adj_list:
            don_atom_id = hbond_adj_list[acc_atom_id]
            # Grab acceptor atom info
            acc_res_index = acc_atom_id.rsd()
            acc_atom_index = acc_atom_id.atomno()
            acc_residue = self.pose.residue(acc_res_index)
            acc_res_name = acc_residue.name()
            acc_atom_name = acc_residue.atom_name(acc_atom_index)
            # Grab donor atom info
            don_res_index = don_atom_id.rsd()
            don_atom_index = don_atom_id.atomno()
            don_residue = self.pose.residue(don_res_index)
            don_res_name = don_residue.name()
            don_atom_name = don_residue.atom_name(don_atom_index)
            print("{} with Atom ID ({},{}) donated to a {} with Atom ID({}, {})".format(acc_atom_name, acc_res_index, acc_atom_index, don_atom_name, don_res_index, don_atom_index))


    # Prints the hbond network
    def print_hbond_network(self):
        if self.hbond_set == None:
            raise Exception("You must call get_hbonds to generate the hbond network before printing it")
        self.hbond_set.show(self.pose)
                    
    ################# ATOMWISE CHEMICAL FEATURES #################
    
    # Generates atom features using Rosetta
    # Each atom has the following features in this order
    # 1. The atom element name
    # 2. The atom the lj radius
    # 3. The atomic charge
    # 4. Boolean indicating whether atom could be donor
    # 5. Boolean indicating whether atom could be acceptor
    # 6. Boolean indicating whether atom is on the backbone
    def generate_chemical_features(self):
        chemical_features = []
        for res_index in range(1, self.num_residues + 1):
            residue = self.pose.residue(res_index)
            num_res_atoms = residue.natoms()
            # Loop through all atoms of each residue
            for atom_index in range(1, num_res_atoms + 1):
                #atom_name = residue.atom_name(atom_index)
                atom_type = residue.atom_type(atom_index)
                #hybrid = atom_type.hybridization()
                lj_radius = atom_type.lj_radius()
                is_donor = atom_type.is_donor()
                is_acceptor = atom_type.is_acceptor()
                is_backbone = residue.atom_is_backbone(atom_index)
                atomic_charge = residue.atomic_charge(atom_index)
                element = atom_type.element()
                #charge = residue.atomic_charge(atom_index)
                chemical_features.append([element, lj_radius, atomic_charge, is_donor, is_acceptor, is_backbone])
        return chemical_features


