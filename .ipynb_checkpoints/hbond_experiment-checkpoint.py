# %%
# Imports
import os
import sys
# ML Libraries
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.functional as F
# Protein Analysis Libaries
import MDAnalysis as mda
import nglview as nv
from nglview.datafiles import PDB, XTC
# Custom Libraries
from mol_processors.PDB_processor import download_pdb, get_coords, pdb_to_intcoords
%pylab inline
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
u = mda.Universe(PDB, XTC)
protein = u.select_atoms('protein')
w = nv.show_mdanalysis(protein)
w
# %%
plt.plot([0, 1, 2], [0, 1, 4])
# %%
