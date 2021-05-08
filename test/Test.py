import os
from prepare import *

#small_molecules = [2JOF, 1L2Y, 1FME, 2F4K, 2F21, 2HBA, 2WxC, 1PRB, 2P6J, 1ENH, 1MIO, 2A3D, 1LMB, 1CQ0, 1L2Y 1ROO, 1T5Q, 1V1D,1LFC, 8TFV, 1KGM, 1M4F, 1TV0, 2jof, 2f4k, 1ery, 2p6j, 3gb1, 2f21, 1prb, 1bdd, 1gh1, 1fex, 1dv0]
small_molecules = ['2JOF']
base_dir = "/home/jeffreymo572/Mol2_File_Conversion"
prepare_protein_dir = os.path.join(base_dir, '/protein_prep/prepare.py')

def download_pdb (pdb, output_dir=None):
    pdb_id = pdb.upper() + ".pdb"
    # Default output is current working directory
    if (output_dir == None):
        output_dir = "./"
    output = os.path.join(output_dir, pdb_id.lower())
    # if the output file does not exists, download the pdb file
    print(output)
    if not os.path.isfile(output):
        os.system("curl -L https://files.rcsb.org/download/{}.gz --output {}.gz".format(pdb_id, output))
        os.system ("gunzip {}.gz".format(output))
    else:
        print("The file already exists")
    return 

def Missing_Resdue(pdb):
    for line in open(pdb):
        list = line.split()
        id = list[0]
        if id == 'REMARK':
            if list[1] == '465':
                return True
    return False

def Prepared_Pdb(output_dir, pdb, chains):
    pdbpqr(output_dir, pdb, chains)

#TODO fix output path

for molecule in small_molecules:
    molecule_path = os.path.join(base_dir, molecule)
    if not os.path.exists(molecule_path):
        os.mkdir(molecule_path)
    download_pdb(molecule, output_dir = molecule_path)
    #Checks for missing residues, skips iteration if there is for now
    if Missing_Resdue("/home/jeffreymo572/Mol2_File_Conversion/2JOF/{}.pdb".format(molecule.lower())):
        print("{}!!!!!".format(molecule))
        continue
    os.system("/home/jeffreymo572/Mol2_File_Conversion/protein_prep/prepare.py /home/jeffreymo572/Mol2_File_Conversion/2JOF/{}.pdb".format(molecule.lower()))

