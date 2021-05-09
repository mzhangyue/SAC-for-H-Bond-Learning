# %%
# Imports
import os
import sys
from pympler import asizeof, tracker, refbrowser
import gc
# ML Libraries
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.functional as F
from resource import getrusage, RUSAGE_SELF
# Protein Analysis Libaries
import MDAnalysis as mda
# Custom Libraries
from mol_processors.PDB_processor import download_pdb, get_coords, pdb_to_intcoords
%pylab inline
tr = tracker.SummaryTracker()
# %%
# Define some constants
PDB_DIR = "./data/pdbs/"
OCTREE_DIR = "./Octree"
# Define some parameters
pdb_id = "1cq0" # The pdb id of the input protein
dcd_file = ""
pdb_file = "data/1cq0_pnon.pdb"
psf_file = "data/1cq0.psf"
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
intern = pdb_to_intcoords(psf_file, pdb_file)
print(intern.bat)

# %%
# %%

# %%
# Memory statprint("Memory Used: ", getrusage(RUSAGE_SELF).ru_maxrss / 100000.0, "MB")
tr.print_diff()
# %%