import os


small_molecules = ['2JOF', '1L2Y', '1FME', '2F4K', '2HBA', '2WxC', '1PRB', '2P6J', '1ENH', '1MIO', '2A3D', '1LMD', '1CQ0', '1L2Y', '1ROO', '1T5Q', '1V1D', '1LFC', '8TFV', '1KGM', '1M4F', '1TV0', '2jof', '2f4k', '1ery', '2p6j', '3gb1', '2f21', '1prb', '1bdd', '1gh1', '1fex', '1dv0', '1fme', '1MIO', '2A3D', '1ubq', '1hp8', '2WxC', '2HBA', '1pou', '1lmb']

#TODO:Make this nicer looking
base_dir = "/home/jeffreymo572/Mol2_File_Conversion"
prepare_protein_dir =  os.path.join(base_dir, "protein_prep/prepare.py")
path_to_dihe = os.path.join(base_dir,"create_psf/create_psf_dihe")
path_to_rtf = "/home/jeffreymo572/SAC-for-H-Bond-Learning/Octree/FromBU/oct-example/params/pdbamino_new.rtf"
missing_residue_molecules = []

def download_pdb (pdb, output_dir=None):
    pdb_id = pdb.upper() + ".pdb"
    # Default output is current working directory
    if (output_dir == None):
        output_dir = "./"
    output = os.path.join(output_dir, pdb_id.lower())
    # if the output file does not exists, download the pdb file
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



for molecule in small_molecules:
    molecule = molecule.upper()
    molecule_path = os.path.join(base_dir, molecule)
    if not os.path.exists(molecule_path):
        os.mkdir(molecule_path)
    
    #Input: molecule ID
    #Output: pdb in molecule's directory
    download_pdb(molecule, output_dir = molecule_path)

    #Checks for missing residues, skips iteration if there is for now
    path_to_pdb = os.path.join(molecule_path, "{}.pdb".format(molecule.lower()))
    if Missing_Resdue(path_to_pdb):
        missing_residue_molecules.append(molecule)
        continue

    #Input: Molecule.pdb
    #Output: Molecule_pnon.pdb
    os.system("{} {}".format(prepare_protein_dir, path_to_pdb))
    pnon_path = "/home/jeffreymo572/Mol2_File_Conversion/{}/{}_pnon.pdb".format(molecule, molecule.lower())

    #Input: Molecule_pnon.pdb
    #Output: Molecule_PSF
    os.system("{} {} {} {}_pnon.psf".format(path_to_dihe, pnon_path, path_to_rtf, molecule_path))

    #TODO: TEMP FIX (manually moves .psf file into molecule_path)
    os.system("mv {}_pnon.psf {}".format(molecule, molecule_path))

    #Input Molecule_pnon.pdb
    #Output: mol2 file
    os.system("obabel {} -O {}_pnon.mol2".format(pnon_path, molecule))

    #TODO: TEMP FIX (manually moves .mol2 file into molecule_path)
    os.system("mv {}_pnon.mol2 {}".format(molecule, molecule_path))

print(missing_residue_molecules)