# Imports
from utils import write_array
from Octree.octree import Protein
from pympler import asizeof, tracker, refbrowser
from memory_profiler import profile
import MDAnalysis as mda
from pyrosetta import *
from pyrosetta.toolbox import *
from pyrosetta.rosetta.core.scoring import ScoreType, Ramachandran, ScoringManager
from pyrosetta.rosetta.protocols.backbone_moves import RandomizeBBByRamaPrePro
from pyrosetta.rosetta.protocols.relax import FastRelax
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
            print("Loading pdb fie into Rosetta...")
            self.pose = pose_from_pdb(pdbFile)
            #self.octree_rep = Protein(pdbFile, psfFile, mol2File, prmFile, rtfFile, aprmFile, outnFile, outoFile)
            self.pdb_name = pdbFile.split(".pdb")[0]
        elif pdb_id != None:
            print("Loading pdb id into Rosetta")
            self.pose = pose_from_rcsb(pdb_id)
            self.pdb_name = pdb_id
        # Default to alanine dipeptide
        else:
            self.pose = pose_from_sequence("AA", auto_termini=True)
        
        # Get total atoms and total residues
        self.num_residues = self.pose.total_residue()
        self.num_atoms = self.pose.total_atoms()
        self.num_virt_atoms = self.get_num_virt_atoms()
        # Store atom ids
        self.atom_ids = self.get_atom_ids()
        # Store secondary structure per atom
        self.update_sec_struct()
        # Store Cartesian coordinates (do not store virtual atoms) 
        self.cart_coords = np.zeros((self.num_atoms-self.num_virt_atoms, 3))
        self.update_cart_coords()
        # Store bonded adjacency list
        self.bond_adj_list = self.get_bond_adj_list()
        # Store hbond set. This only gets filled when self.get_hbonds is called
        self.hbond_set = None
        # Generates chemical features
        self.atom_chem_features = self.generate_chemical_features()
        # Set the scoring functon
        self.score_function = get_score_function(True)

    ################# INPUT/OUTPUT FUNCTIONS #################
    
    # Writes the protein configuration to a PDB file
    def write_to_pdb(self, output_file):
        self.pose.dump_pdb(output_file)
        return
    
    # Writes numpy coords to pdb files
    # Coord_files is a list of coord files of saved npy files
    def np_coords_to_pdbs(self, coord_files, output_dir, coord_file_type= "npy"):
        # Check for valid file type
        if coord_file_type != "npy" or coord_file_type != "text":
            raise Exception("Invalid file type")
        # Loop through all coord files
        for index, coord_file in enumerate(coord_files):
            coords = []
            if coord_file_type == "npy":
                coords = np.load(coord_file)
            elif coord_file_type == "text":
                coords = np.loadtxt(coord_file)
            # Convert each coord to xyz vector
            xyz = []
            for coord in coords:
                xyz.append(numeric.xyzVector_float(coord))
            # Set the coords
            self.pose.batch_set_xyz(Vector1(self.atom_ids), Vector1(xyz))
            output_file_name = self.pdb_name + str(index) + ".pdb"
            self.write_to_pdb(os.path.join(output_dir, output_file_name))
        return 

    # Coords can be a pdb file or traj file (dcd, xtc,...)
    def visualize_protein(self, default=None, default_representation=False):
        # Select all atoms associated with a protein
        #protein_residues = self.prot.select_atoms("protein")
        #w = nv.show_mdanalysis(protein_residues, default=default, default_representation=default_representation)
        return 
    
    # Writes the Cartesian coordinates to disk
    def write_cart_coords(self, output_file, file_type=None):
        write_array(output_file, self.cart_coords, file_type)
        return        
    ################# ATOM IDS #################
    # Get the number of virtual atoms
    def get_num_virt_atoms(self):
        count = 0
        for res_index in range(1, self.num_residues + 1):
            residue = self.pose.residue(res_index)
            num_res_atoms = residue.natoms()
            # Loop through all atoms of each residue
            for res_atom_index in range(1, num_res_atoms+1):
                if(residue.atom_type(res_atom_index).is_virtual()):
                    count += 1
        return count

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
                # skip virtual atoms (i.e. NV on proline)
                if(not residue.atom_type(res_atom_index).is_virtual()):
                    atom_ids.append(AtomID(res_atom_index, res_index))
        return atom_ids

    
    # Returns the internal coordinates
    # Converts the atomsitic positions to internal coordinates using mdanalysis
    # In interanl coordinates, each atom is defined by
    # 1. Bond Length (defined by 2 atoms)
    # 2. Bond Angle (defined by 3 atoms)
    # 3. Dihedral Angle (defined by 4 atoms)
    # TODO
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
        for atom_id in self.atom_ids:
                res_atom_index = atom_id.atomno()
                residue = self.pose.residue(atom_id.rsd())
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
        for atom_id in self.atom_ids:
            neighbor_atom_ids  = self.pose.conformation().bonded_neighbor_all_res(atom_id)
            bond_adj_list[atom_id] = np.array(neighbor_atom_ids)
        return bond_adj_list
    
    def print_bond_adj_list(self):
        for atom_id in self.bond_adj_list:
            # Get atom id data
            bonded_atom_ids = self.bond_adj_list[atom_id]
            for bonded_atom_id in bonded_atom_ids:
                atom_res_index = atom_id.rsd()
                atom_index = atom_id.atomno()
                atom_residue = self.pose.residue(atom_res_index)
                atom_res_name = atom_residue.name()
                atom_name = atom_residue.atom_name(atom_index)
                # Get bonded neighbor's data
                bonded_atom_res_index = bonded_atom_id.rsd()
                bonded_atom_index = bonded_atom_id.atomno()
                bonded_atom_residue = self.pose.residue(bonded_atom_res_index)
                bonded_atom_res_name = bonded_atom_residue.name()
                bonded_atom_name = bonded_atom_residue.atom_name(bonded_atom_index)
                print("{} with Atom ID ({},{}) is bonded to {} with Atom ID({}, {})".format(atom_name, atom_index, atom_res_index, bonded_atom_name, bonded_atom_index, bonded_atom_res_index))
    
    def validate_bond_adj_list(self):
        valid = True
        count = 0
        # Loop through all atom ids
        for atom_id in self.bond_adj_list:
            # Loop through all atom neighbors
            bonded_atom_ids = self.bond_adj_list[atom_id]
            for bonded_atom_id in bonded_atom_ids:
                bonded_atomno = bonded_atom_id.atomno()
                bonded_res = bonded_atom_id.rsd()
                # Find the atom id address with the matching atomno and res
                bond_atom_id_key = None 
                for bonded_key in self.bond_adj_list:
                    if bonded_key.atomno() == bonded_atomno and bonded_key.rsd() == bonded_res:
                        bond_atom_id_key = bonded_key
                        break
                if bond_atom_id_key == None:
                    raise Exception("For some reason, a bonded atom is not in the bond adjacency list")
                atom_id_found = False
                # Check if atom id is in its neighbor's bonded list
                for bonded_bonded_atom_id in self.bond_adj_list[bond_atom_id_key]:
                    if bonded_bonded_atom_id.atomno() == atom_id.atomno() and bonded_bonded_atom_id.rsd() == atom_id.rsd():
                        atom_id_found = True
                        break
                if not atom_id_found:
                    count += 1
                    atom_res_index = atom_id.rsd()
                    atom_index = atom_id.atomno()
                    atom_residue = self.pose.residue(atom_res_index)
                    atom_res_name = atom_residue.name()
                    atom_name = atom_residue.atom_name(atom_index)
                    print(atom_id)
                    print(atom_name, atom_res_name, atom_index)
        return valid

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

    # Sets the phi angle of a residue
    def set_phi_angle(self, residue_index, new_angle):
        self.pose.set_phi(residue_index, new_angle)
        return

    # Sets the psi angle of a residue
    def set_psi_angle(self, residue_index, new_angle):
        self.pose.set_psi(residue_index, new_angle)
        return
    
    # Given a residue index, change the phi-psi angles of that residue
    # The dim of angle_change and cur_dihedrals are both vectors of length 2 
    # new_dihedrals = cur_dihedrals + angle_change 
    def set_backbone_dihedrals(self, residue_index, angle_change):
        residue = self.pose.residue(residue_index)
        #cur = [self.pose.phi(residue_index),  self.pose.psi(residue_index)]
        #new_phi_psi = np.append(cur + np.array(angle_change), self.pose.omega(residue_index))
        #residue.mainchain_torsions(Vector1(new_phi_psi))
        residue.set_phi(self.pose.phi(residue_index) + angle_change[0])
        residue.set_psi(self.pose.psi(residue_index) + angle_change[1])
        return
    
    # Sample backbone dihedrals for a specific residue from a ramachanndran distribution
    # biased if there is a proline 
    def smaple_rama_backbone_dihedrals(self, res_index):
        RandomizeBBByRamaPrePro().apply(self.pose)
        return

    ################# SIDECHAIN DIHEDRALS (CHI) ROTAMERS #################

    # Gets the sidechain angle (i.e. chi) given the residue and angle index
    def get_sidechain_angle(self, residue_index, angle_index):
        return self.pose.chi(angle_index, residue_index)
    
    # Gets all the sidechain angles
    def get_sidechain_angles(self, residue_index):
        return np.array(self.pose.chi(residue_index))

    # Sets one of the rotamer angles (i.e. side chain angle) of a specific residues
    # If one of the rotamers is not specified, we assume it remains the same
    def set_sidechain_angle(self, residue_index, angle_index, new_angle):
        self.pose.set_chi(angle_index, residue_index, new_angle)
        return
    
    # Sets all the sidechain angles of a residue
    def set_sidechain_angles(self, residue_index, angle_change):
        residue = self.pose.residue(residue_index)
        # Store current chi angles
        cur_chis = []
        for chi_index in range(1, residue.nchi()):
            cur_chis.append(self.pose.chi(chi_index, residue_index))
        # Only choose the residue angles that are needed
        res_angle_change = np.array(angle_change[:residue.nchi()])
        # Computer and set new chis
        new_chis = np.array(cur_chis) + res_angle_change
        residue.set_all_chi(Vector1(new_chis))
        return
    
    # Sets all the sidechain angles
    def set_all_sidechain_angles(self, angle_changes):
        return
        
    # Sample rotamers    
    def sample_rotamers(self):
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
            if(acc_residue.atom_type(acc_atom_index).is_virtual() or don_residue.atom_type(don_atom_index).is_virtual() ):
                print("VIRTUAL ATOM DETECTED")
            print("{} with Atom ID ({},{}) donated to a {} with Atom ID({}, {})".format(acc_atom_name, acc_atom_index, acc_res_index, don_atom_name, don_atom_index, don_res_index))

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
        for atom_id in  self.atom_ids:
                atom_index = atom_id.atomno()
                residue = self.pose.residue(atom_id.rsd())
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

    # Relax the protein
    def relax_prot(self):
        relax = FastRelax()
        relax.apply(self.pose)
        return



















