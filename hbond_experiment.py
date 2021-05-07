# %%
# Imports
import os
import sys
from MDAnalysis.core.universe import Universe
import torch
import torch.nn as nn
import torch.functional as F
from mol_processors.PDB_processor import download_pdb, get_coords, pdb_to_intcoords

# %%
# Define some constants
PDB_DIR = "./data/pdbs/"
OCTREE_DIR = "./Octree"
# Define some parameters
pdb_id = "1cq0" # The pdb id of the input protein
dcd_file = ""
# Set flags
download_protein = True # Flag to download protein
# %%
# Load in the pdb_id in data dir
if download_protein:
    download_pdb(pdb_id, output_dir=PDB_DIR)
pdb_file = PDB_DIR + pdb_id.lower()
psf_file = PDB_DIR + pdb_id.lower()   
# %%
# Grab internal coordinates
intern = pdb_to_intcoords(input_file)
print(intern.bat)
