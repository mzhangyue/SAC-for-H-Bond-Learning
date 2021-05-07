import octree
from octree import Protein


# TODO
# 1. Preprocess all the pdb files
# 2. Check if moving the atom column to colunm 78 matters
# 3. Verify that all data files work with the octree

if __name__ == '__main__':
    path_prefix = "/home/conradli/Learning-Viral-Assembly-Pathways-with-RL-/src/Octree/FromBU/oct-example/"
    # File Paths for param and pdb files
    pdbFile = bytes(path_prefix + "examples/1acb_b_rmin.pdb", encoding="ascii")
    pdbFixedFile = bytes(path_prefix + "examples/1acb_b_rmin-fixed.pdb", encoding="ascii")
    mol2File = bytes(path_prefix +"examples/1acb_b_rmin.mol2", encoding="ascii")
    psfFile = bytes(path_prefix + "examples/1acb_b_rmin.psf", encoding="ascii")
    outnFile = bytes(path_prefix + "examples/1acb_b_rmin-outn.pdb", encoding="ascii")
    outoFile = bytes(path_prefix + "examples/1acb_b_rmin-outo.pdb", encoding="ascii")
    prmFile = bytes(path_prefix + "params/parm_new.prm", encoding="ascii")
    rtfFile = bytes(path_prefix + "params/pdbamino_new.rtf", encoding="ascii")
    aprmFile = bytes(path_prefix + "params/atoms.0.0.6.prm.ms.3cap+0.5ace.Hr0rec", encoding="ascii")
    prot = Protein (pdbFile, psfFile, mol2File, prmFile, rtfFile, aprmFile)
    #print(octree.octree_test())