# Standard imports
import numpy as np
import pandas as pd
import os
import sys
import shutil
from collections import defaultdict

# Imports to parse MD simulation files and PDB files
import MDAnalysis as mda
from Bio.PDB import *
from Bio import SeqIO
import Bio.SeqUtils as SeqUtils
from modeller import *
from modeller.automodel import *
#from MDAnalysis.tests.datafiles import PSF, DCD
from utils import str_replace, str_insert, sum_seq_in_dict
from .bat import BAT


# Dictionary of converting three letter residues to 1 letter code 
res_3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

# Converts PDB files to PQR files 
# Missing residues are added with MODELLER
# Missing atoms are added with PDB2PQR 3.1
# Protonation states are assigned with PROPKA3
# Charges and Atom radii are assigned with the AMBER Forcefield FF19SB
# TODO STILL HAVE TO IMPLEMENT
def pdb_to_pqr(pdb):  
    return 

# Converts the atomsitic positions to internal coordinates using mdanalysis
# In interanl coordinates, each atom is defined by
# 1. Bond Length (defined by 2 atoms)
# 2. Bond Angle (defined by 3 atoms)
# 3. Dihedral Angle (defined by 4 atoms)
# 
# Returns a N x 4 numpy matrix representing the internal coordinates 
def pdb_to_intcoords(psf, pdb):
    u = mda.Universe(psf, pdb)
    intern = BAT(selected_residues)
    intern.run()
    return intern

# Grabs coordinates from the a trajectory or pdb file
def get_coords(coord_path, top_path, file_type="dcd", save_pdbs=False, save_np=True, np_file="prot_coords.npy"):
    u = mda.Universe(top_path, coord_path)
    protein = u.select_atoms("protein")
    result = []
    
    # Case if the file is a trajectory file
    if file_type == "dcd":
        # Add the postions of the protein at each timestep to result
        for ts in u.trajectory:
            print(ts)
            result.append(ts.positions)
        # Option to save pdbs of each trajectory
        if (save_pdbs):
            with mda.Writer("protein.pdb", protein.n_atoms) as W:
                for ts in u.trajectory:
                    W.write(protein)
    
    # Case if the file is a single pdb file
    elif file_type == "pdb":
        result.append(u.positions)

    # Return a numpy array of type float64
    result = np.array(result).astype(np.float64)
    np.save(np_file, result)
    return result


# Receives a pdb file and converts it into a graph representation
# Nodes are atoms and bonds are edges
# Returns a Feature Matrix "X" and an Adjacency Matrix "A"
# TODO STILL HAVE TO IMPLEMENT
def pdb_to_graph(pdb, parm, is_pqr=False):
    # Try to parse the pdb file
    try:
        parser = PDBParser(PERMISSIVE=False, is_pqr=is_pqr)
        structure = parser.get_structure ("id", pdb)
    except:
        print ("There was a parsing error for pdb file at ", pdb)
        return None

    if (pqr == False):
        universe = mda.Universe(pdb, parm)
    return X, A


# Downloads a given pdb id from the PDB database to output file name
# Default output is the current directory
def download_pdb (pdb_id, output_dir=None):
    pdb_id = pdb_id.upper() + ".pdb"
    # Default output is current working directory
    if (output_dir == None):
        output_dir = "./"
    output = output_dir + pdb_id.lower()
    # if the output file does not exists, download the pdb file
    if not os.path.isfile(output):
        os.system(f"curl -L https://files.rcsb.org/download/{pdb_id}.gz --output {output}.gz")
        os.system (f"gunzip {output}.gz")
    else:
        print ("The file already exists")
    return 


# Fills in missing residues of pdb file using MODELLER
# The files will be ouputted to your current directory
# If the file path of the pdb file is not specified, a new pdb is downloaded in the current directory
# Params:
# pdb_id (str): The pdb id of the PDB file
# file_path (str): File path of the PDB file (NOTE THAT IT MUST START WITH "pdb_id")
# read_het (bool): Flag to also read in the HETATMS 
def fill_missing_residues(pdb_id, file_path=None,read_het=False):
    # Download the pdb it is not downloaded
    if file_path == None:
        download_pdb (pdb_id)
        file_path = os.getcwd() + "/" + pdb_id.lower() + ".pdb"
    # Check first if there are any missing residues
    head = parse_pdb_header (file_path)
    if not head['has_missing_residues']:
        print ("There are not missing residues")
        return
    # Generate full and missing sequence dictionaries
    full_seq, miss_seq = create_full_and_missing_seq (pdb_id, file_path, head=head, read_het=read_het)
    if (miss_seq == None):
        print ("There are too many consecutive missing residues")
        return
    # Set up MODELLER environment to read heteroatoms
    #log.verbose()
    env = environ()
    env.io.hetatm = read_het
    env.io.atom_files_directory = [os.path.dirname(file_path)]
    # Create a model of pdb id
    m = model(env, file='6gyp')
    aln = alignment(env)
    # Try to alignment the sequence of the pdb_id with existing structures
    aln.append_model(m, align_codes=pdb_id)
    # Write the homologicall modelled sequence to a .seq file
    seq_file = env.io.atom_files_directory[0] + '/' + pdb_id
    aln.write(file=seq_file + ".seq")
    # Read in the MODELLER header (first three lines)
    header = []
    with open (seq_file + ".seq", "r") as f:
        i = 0
        for line in f:
            if i <= 2:
                header.append(line)
            else:
                break
            i += 1
    # Write the align file containing the full sequence and the summed sequences
    align_path = os.path.dirname(file_path) + "/" + pdb_id + "_align.ali"
    f = open (align_path, "w")   
    f.write(sum_seq_in_dict (miss_seq, header=header[0] + header[1] + header[2], pretty=True))
    f.write("\n")
    f.write(sum_seq_in_dict (full_seq, header=header[0] + header[1][:-1] + "_fill\n" + "sequence:::::::::\n",pretty=True))
    f.close()

    # Perform the loop modeling
    a = loopmodel(env, alnfile = align_path,
              knowns = pdb_id, sequence = pdb_id + "_fill")
    a.starting_model= 1
    a.ending_model  = 1
    a.loop.starting_model = 1
    a.loop.ending_model   = 2
    a.loop.md_level       = refine.fast

    a.make()
    return 

# Creates two dictionaries where each key refers to the chain id and each
# value refers to its amino acid sequence. In one dictionary, the full 
# sequence of each chain will be stored and in the other each missing residue
# is represented as a "-" char
# Params:
# pdb_id (str): The pdb id of the PDB file
# file_path (str): File path of the PDB file (NOTE THAT IT MUST START WITH "pdb_id")
# read_het (bool): Flag to also read in the HETATMS  
def create_full_and_missing_seq (pdb_id, file_path, head=None, read_het=False):
    
    # Extract header information if not given
    if head is None:
        head = parse_pdb_header (file_path)
    # Extract chains
    structure = PDBParser().get_structure(pdb_id.upper(), file_path)
    chains = [each.id for each in structure.get_chains()]
    print (chains)
    # Make a dictionary for each chain
    missing_res = defaultdict(list)
    # Fill each chain's list with missing residue indices and missing residue names
    for residue in head['missing_residues']:
        if residue["chain"] in chains:
            seq_num = residue["ssseq"]
            res_code = res_3to1[residue["res_name"]]
            missing_res[residue["chain"]].append({"seq_num": seq_num, "res_code": res_code})
        
    # Collect the full and missing sequences
    full_seq = {}
    miss_seq = {}
    miss_index = 0
    # Grab the original pdb sequence by chain
    for chain in structure[0]:
        orig_seq = ""
        # Concantenate each residue in each chain
        for residue in chain:
            # Check whether the residue is a HETATM record
            if read_het or (residue.id[0].isspace() != read_het):
                orig_seq += res_3to1[residue.resname]
        # Print chain sequence
        print (">Chain", chain.id, "\n", ''.join(orig_seq))
        print ()
        # Try to get missing res list for chain
        miss_indices = missing_res.get (chain.id)
        # Case for no missing residues in chain
        if (miss_indices == None):
            print ("No missing residues for chain ", chain)
            full_seq[chain.id] = orig_seq + "/"
            miss_seq[chain.id] = miss_seq + "/"
        # Case for missing residues
        else:
            print ("There are", len(miss_indices), "missing residues in chain", chain.id)
            # Replace all 
            temp_full = orig_seq
            temp_miss = orig_seq
            # Add in missing residues or gap characters
            for res in miss_indices:
                seq_index = res["seq_num"] - 1
                temp_full = str_insert (temp_full, res["res_code"], seq_index)
                temp_miss = str_insert (temp_miss, "-", seq_index)
            # Add the chain sequence
            full_seq[chain.id] = temp_full + "/"
            miss_seq[chain.id] = temp_miss + "/"
    if not below_max_missing_res (miss_seq):
        return None, None
    return full_seq, miss_seq

# Checks if a sequence does not have a consecutive missing residues larger than
# a threshold
# Params:
# seq (dict): the dictonary of amino acid sequence separated by chain
# threshold (int): the max number of residues allowed
def below_max_missing_res (seq, threshold=100):
    for chain in seq:
        num_consec = 0
        for char in seq[chain]:
            if char == "-":
                num_consec += 1
                if num_consec > threshold:
                    print ("Chain ", chain, "had ", num_consec, "consecutive missing residues")
                    return False
            else:
                num_consec = 0
    return True



# Extract chains from a PDB File and convert them into PDB files
# file_path (str): file path of the PDB
# chains (array of str): chain ids that we want; default behavior is extract all chains
def extract_chains (file_path, chains=None, output_path=None):
    if (output_path == None):
        output_path = os.path.dirname(file_path)
    parser = PDBParser (PERMISSIVE=False)
    pdb_id = os.path.splitext(os.path.split(file)[1])[0]
    structure = parser.get_structure (pdb_id, file_path)
    if chains == None:
        chains = [each.id for each in structure.get_chains()]
    # Extract each chain from the pdb file 
    for chain in chains:
        pdb_chain_file = output_path + "/" + pdb_id + "_" + chain.upper() + ".pdb"
        pdb_io = PDBIO()
        pdb_io.set_structure(structure)
        pdb_io.save ('{}'.format(pdb_chain_file), ChainSelect(chain))
    return


# Class used to help select chains
class ChainSelect (Select):
    def __init__(self, chain):
        self.chain = chain
    def accept_chain(self, chain):
        if chain.get_id() == self.chain:
            return 1
        else:          
            return 0

if __name__ == '__main__':
    file = "/home/conrad/Oct-GP/Learning-Viral-Assembly-Pathways-with-RL-/data/6gyp/6gyp.pdb"
    dcd_file = "/mnt/c/Users/conra/CHARMM PDB Sims/1acb_b_rmin_full.dcd"
    pdb_file = "/mnt/c/Users/conra/CHARMM PDB Sims/1acb_b_rmin.pdb"
    top_file = "/mnt/c/Users/conra/CHARMM PDB Sims/1acb_b_rmin.psf"
    coors = get_coords(pdb_file, top_file, save_np=True, np_file="/home/conradli/Learning-Viral-Assembly-Pathways-with-RL-/data/1acb/1acb_coords_pdb.npy")
    print(coors)
    print(coors.shape)
    output_path="/home/conrad/Oct-GP/Learning-Viral-Assembly-Pathways-with-RL-/data/6gyp/chains"
    #create_full_and_missing_seq ("6gyp", file)
    #below_max_missing_res ("conrad@conrads-desktop:~/Oct-GP/Learning-Viral-Assembly-Pathways-with-RL-/data/6gyp/6gyp_align.ali", threshold=12)
    #full_seq, miss_seq = create_full_and_missing_seq ("6gyp", file, head=None, read_het=False)
    #extract_chains (file)
    print("No compile errors")