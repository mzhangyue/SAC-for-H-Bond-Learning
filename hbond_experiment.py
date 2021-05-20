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
from mol_processors.Protein import Prot
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
print("Initialization")
#path_prefix = "/home/conradli/SAC-for-H-Bond-Learning/Octree/FromBU/oct-example/"
dir_prefix = "/home/conradli/SAC-for-H-Bond-Learning/data"
file_prefix = "ala_dip_charmm"
path_prefix = os.path.join(dir_prefix, "alanine_dipeptide/", file_prefix)
param_prefix = os.path.join(dir_prefix, "params/")
print(path_prefix)
# File Paths for param and pdb files
pdbFile = bytes(path_prefix + ".pdb", encoding="ascii")
pdbFixedFile = bytes(path_prefix + "-fixed.pdb", encoding="ascii") # Currently not used
mol2File = bytes(path_prefix +".mol2", encoding="ascii")
psfFile = bytes(path_prefix + ".psf", encoding="ascii")
outnFile = bytes(path_prefix + "-outn.pdb", encoding="ascii") # Output for nonbonded list
outoFile = bytes(path_prefix + "-outo.pdb", encoding="ascii") # Output for octree
prmFile = bytes(param_prefix + "parm_new.prm", encoding="ascii")
rtfFile = bytes(param_prefix + "pdbamino_new.rtf", encoding="ascii")
aprmFile = bytes(param_prefix + "atoms.0.0.6.prm.ms.3cap+0.5ace.Hr0rec", encoding="ascii")
prot = Prot(pdbFile, psfFile, mol2File, prmFile, rtfFile, aprmFile, outnFile, outoFile)
print("After min")
# %%

# %%
# %%

# %%
# Memory statprint("Memory Used: ", getrusage(RUSAGE_SELF).ru_maxrss / 100000.0, "MB")
tr.print_diff()
# %%