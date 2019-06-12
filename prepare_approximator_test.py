
import os
import xml.etree.cElementTree as ET
from parse_pdbtm_xml import pdn
import numpy as np
from Bio.PDB import *
from membrane_position_aproximator import approximate_helix_axis, fit_line
import matplotlib.pyplot as plt
import math

"""
TODO:
least square in approximate_helix_axis instead of mean

clustering of helices axis to refine them ? (with position+length in the space ?)

least square between helices axis as approximate membrane normal

position of membrane from mean middle points of helices

"""


pdbtm_dir = "pdbtm_structures"


def main():
    print("parsing pdbtm xml...")
    pdbtm_s, pdbtm_m = parse_pdbtm("pdbtm_reduced.xml")
    print("choosing ids...")
    test_ids = choose_ids(pdbtm_s, 100)
    print("downloading files...")
    download_pdb_files(test_ids)
    print("parsing all pdbs...")
    helix_c_alphas = parse_pdbs(test_ids, pdbtm_s)
    print("approximating membranes...")

    all_axis = []
    for key in helix_c_alphas.keys():
        helix_axis = []
        for helix in helix_c_alphas[key]:
            axis = approximate_helix_axis(helix)
            helix_axis.append(axis)
            # print(axis)
        all_axis.append(helix_axis)
        # print(helix_axis)
        if len(helix_axis) <= 1:
            print("skipping because too few helices:", key)
            continue

        print()
        print(key)
        print(fit_line(helix_axis))
    # plot_angles(all_axis)
        print(pdbtm_m[key])
    #plt.hist(all_axis)
    #plt.show()


def parse_pdbtm(pdbtm_xml):
    """
    (extended from parse_pdbtm_xml.py)
    Returns two dict: structures, membranes
     membranes:
    pdbid -> [("Normal", x, y, z), ("ROWX", x, y, z, t), ("ROWY", x, y, z, t), ("ROWZ", x, y, z, t)]
     structures:
    pdbid -> [(sec struc 1)(sec struc 2)...(ssx)].
    (secundary structure: ( chain_id, helix_sequence, sequence_begin, sequence_end, pdb_begin, pdb_end, type)



    """
    tree = ET.ElementTree(file=pdbtm_xml)
    root = tree.getroot()
    root.tag, root.attrib
    pdbtms_structures = {}
    pdbtms_membranes = {}
    # Parse XML
    for pdbtm in root:
        pdbid = pdbtm.attrib["ID"]
        for child in pdbtm:
            tag = child.tag[23:]
            if tag == "MEMBRANE":
                for child2 in child:
                    tag_membrane = child2.tag[23:]
                    if tag_membrane == "NORMAL":
                        x = child2.attrib["X"]
                        y = child2.attrib["Y"]
                        z = child2.attrib["Z"]
                        pdbtms_membranes[pdbid] = [("NORMAL", x, y, z)]

                    if tag_membrane == "TMATRIX":
                        for row in child2:
                            pdbtms_membranes[pdbid].append((row.tag[23:], row.attrib["X"], row.attrib["Y"], row.attrib["Z"], row.attrib["T"]))


            elif tag == "CHAIN":
                chainid = child.attrib["CHAINID"]
                for child2 in child:
                    tag2 = child2.tag[23:]
                    # extracting sequence
                    if tag2 == "SEQ":
                        seq = child2.text
                        seq = seq.replace("\n", "")
                        seq = seq.replace(" ", "")
                        # sequence could be also included but not necessary at the moment
                        # pdbtms[pdbid].append(seq)
                    # extracting sec structures
                    elif tag2 == "REGION":
                        seq_beg = int(child2.attrib["seq_beg"])
                        seq_end = int(child2.attrib["seq_end"])
                        pdb_beg = int(child2.attrib["pdb_beg"])
                        pdb_end = int(child2.attrib["pdb_end"])
                        type_ = child2.attrib["type"]
                        # TODO: indices correct?
                        if pdbid in pdbtms_structures:
                            pdbtms_structures[pdbid].append(
                                (chainid, seq[seq_beg - 1:seq_end], seq_beg, seq_end, pdb_beg, pdb_end, type_))
                        else:
                            pdbtms_structures[pdbid] =[
                                (chainid, seq[seq_beg - 1:seq_end], seq_beg, seq_end, pdb_beg, pdb_end, type_)]
    return pdbtms_structures, pdbtms_membranes


def choose_ids(pdbtm_sec, number):
    """
    choose number "random" ids from the keys in the pdbtm_sec dictionary
    """
    ids = list(pdbtm_sec.keys())
    ids = sorted(ids)

    np.random.seed(1234)
    chosen_ids = np.random.choice(ids, number)

    return chosen_ids

def download_pdb_files(id_list):
    pdbl = PDBList()
    for i in id_list:
        pdbl.retrieve_pdb_file(i, pdir=pdbtm_dir,
                               file_format="pdb", overwrite=False)


def parse_pdbs(ids, pdbtm_sructs):
    helix_c_alphas = {}

    for pdb_id in ids:
        pdb_file = "pdb" + pdb_id + ".ent"
        filepath = os.path.join(pdbtm_dir, pdb_file)
        if os.path.isfile(filepath):
            parser = PDBParser()
            protein = parser.get_structure('STS', filepath)

            c_alphas = extract_ca_positions(pdb_id, protein, pdbtm_sructs)

            helix_c_alphas[pdb_id] = c_alphas
        else:
            print("File doesn't exist:", filepath)
    return helix_c_alphas


def extract_ca_positions(pdb_id, protein, pdbtm_structs):
    # use only first model
    model = list(protein.get_models())[0]

    chain_dict = {}
    for chain in model.get_chains():
        chain_dict[chain.id] = list(chain.get_residues())

    helix_c_alphas = []

    for struct in pdbtm_structs[pdb_id]:
        # helix_sequence = struct[1]
        pdb_helix_start = struct[4]
        pdb_helix_end = struct[5]

        if struct[6] != "H":
            continue

        if struct[0] in chain_dict:
            c_alphas = []
            for residue in chain_dict[struct[0]]:
                # print(residue.get_full_id())
                resseq = residue.get_full_id()[3][1]
                if resseq < pdb_helix_start:
                    continue
                elif resseq > pdb_helix_end:
                    break
                else:  # tested gives the same aa as the helix sequence from the xml.
                    for atom in residue.get_atoms():
                        if atom.get_name().strip() == "CA":
                            pos = atom.get_vector()
                            c_alphas.append(pos)

            helix_c_alphas.append(c_alphas)

        else:
            pass
            # print("Missing chain in the pdb???", "Chain:", struct[0])

    return helix_c_alphas



def plot_angles(all_helix_vectors):

    angles = []

    for helix_vectors in all_helix_vectors:
        for i in range(len(helix_vectors)):
            for j in range(len(helix_vectors)):
                if i == j:
                    continue
                else:
                    angles.append(math.degrees(angle_between(helix_vectors[i], helix_vectors[j])))

    bins = np.arange(181)
    plt.hist(angles, bins=bins)
    plt.show()


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


if __name__ == "__main__":
    main()
