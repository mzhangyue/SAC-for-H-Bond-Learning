import octree
from octree import Protein
import os

#Prolly can import this
small_molecules = ['2JOF', '1L2Y', '1FME', '2F4K', '2HBA', '2WxC', '1PRB', '2P6J', '1ENH', '2A3D', '1CQ0', '1L2Y', '1ROO', '1T5Q', '1V1D', '1LFC', '8TFV', '1KGM', '1M4F', '1TV0', '2jof', '2f4k', '1ery', '2p6j', '3gb1', '1prb', '1bdd', '1gh1', '1fex', '1fme', '2A3D', '1ubq', '2WxC', '2HBA', '1pou']

# TODO
# 1. Preprocess all the pdb files
# 2. Check if moving the atom column to colunm 78 matters
# 3. Verify that all data files work with the octree

if __name__ == '__main__':
    # File Paths for param and pdb files
    for molecule in small_molecules:
        molecule = molecule.lower()
        path_prefix = os.path.join("/home/jeffreymo572/Mol2_File_Conversion/", molecule)
        pdbFile = bytes(os.path.join(path_prefix, "{}_pnon.pdb".format(molecule)), encoding="ascii")
        pdbFixedFile = bytes(os.path.join(path_prefix, "{}_pnon.pdb".format(molecule)), encoding="ascii")
        mol2File = bytes(os.path.join(path_prefix, "{}_pnon.mol2".format(molecule)), encoding="ascii")
        psfFile = bytes(os.path.join(path_prefix, "{}_pnon.psf".format(molecule)), encoding="ascii")
        outnFile = bytes(os.path.join(path_prefix, "{}_b_rmin-outn.pdb".format(molecule)), encoding="ascii")
        outoFile = bytes(os.path.join(path_prefix, "{}_b_rmin-outo.pdb".format(molecule)), encoding="ascii")
        #Constants
        params_path = "/home/jeffreymo572/SAC-for-H-Bond-Learning/Octree/FromBU/oct-example/params"
        prmFile = bytes(os.path.join(params_path, "parm_new.prm"), encoding="ascii")
        rtfFile = bytes(os.path.join(params_path, "pdbamino_new.rtf"), encoding="ascii")
        aprmFile = bytes(os.path.join(params_path, "atoms.0.0.6.prm.ms.3cap+0.5ace.Hr0rec"), encoding="ascii")
        prot = Protein (pdbFile, psfFile, mol2File, prmFile, rtfFile, aprmFile, outnFile, outoFile)
        #print(octree.octree_test())
