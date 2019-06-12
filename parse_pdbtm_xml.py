#!/usr/bin/env python
"""
File Name : parse_pdbtm_xml.py
Creation Date : 05-06-2019
Last Modified : Mi 12 Jun 2019 09:48:52 CEST
Author : Luca Deininger
Function of the script :
"""

import xml.etree.cElementTree as ET
from Bio.PDB import *
import os
import random
import string
import warnings
import collections
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from Bio import BiopythonWarning
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
warnings.simplefilter('ignore', BiopythonWarning)


def pdn(dict_):
    """
    print dict nicely
    """
    for k, v in dict_.items():
        print(k, v)


def parse_pdbtm(pdbtm_xml):
    """
    Returns dict with pdbid -> [(membrane_data), (sec struc 1)(sec struc 2)...(ssx)].
    """
    tree = ET.ElementTree(file=pdbtm_xml)
    root = tree.getroot()
    root.tag, root.attrib
    pdbtms = {}
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
                        pdbtms[pdbid] = [("NORMAL", x, y, z)]
                    # could be extended here to get TMATRIX

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
                        pdbtms[pdbid].append(
                            (seq[seq_beg-1:seq_end], seq_beg, seq_end, pdb_beg, pdb_end, type_))
    return pdbtms


def fill_dict_0s(counter_dict):
    """
    For every aa not in counter dict: Add: aa->0.
    """
    for x in aas:
        if x not in counter_dict:
            counter_dict[x] = 0

    return counter_dict


def get_aa_in_helices(pdbtm):
    """
    Extract aa in helices from pdbtm dict.
    """
    seqs = []
    for k, v in pdbtm.items():
        for elem in v[1:]:
            if elem[5] == "H":
                seqs.append(elem[0])

    return seqs


def get_aa_NOT_in_helices(pdbtm):
    """
    Extract aa NOT in helices from pdbtm dict.
    """
    seqs = []
    for k, v in pdbtm.items():
        for elem in v[1:]:
            if elem[5] != "H":
                seqs.append(elem[0])

    return seqs


def pop_non_aas(dict_):
    """
    Somtimes weird aa counts.
    """
    non_aas = ["-", "?", "B", "J", "O", "U", "X", "Z"]
    for aa in non_aas:
        dict_.pop(aa, None)
    return dict_


def count_aa(seqs):
    """
    Count amino acids for each seq in seqs.
    """
    counter_seqs = []
    for seq in seqs:
        counter_seq = collections.Counter(seq)
        counter_seq = fill_dict_0s(counter_seq)
        counter_seq = pop_non_aas(counter_seq)
        counter_seq = collections.OrderedDict(sorted(counter_seq.items()))
        counter_seqs.append(counter_seq)

    return counter_seqs


def dssp_to_dict(dssp_obj):
    """
    dssp returns a weird datastructure -> conversion to dict.
    """
    dssp = collections.OrderedDict()
    for k in list(dssp_obj.keys()):
        dssp[k] = dssp_obj[k]
    return dssp


def parse_dssp(dssp_dict):
    """
    Parses dssp dict and extracts helices.
    """
    helices = []
    # TODO consider chains
    # TODO save more information
    # TODO information regarding where to find this in pdb file (pointer?)
    counter = 0
    prev_entry = ["", "", ""]
    for k, v in dssp_dict.items():

        # starting new helix at start of dict
        if v[2] == "H" and counter == 0:
            new_helix = [v]

        # starting new helix
        elif v[2] == "H" and prev_entry[2] != "H":
            new_helix = [v]

        # appending to current helix
        elif prev_entry[2] == "H" and v[2] == "H":
            new_helix.append(v)

        # close and save new found helix
        elif v[2] != "H" and prev_entry[2] == "H":
            helices.append(new_helix)

        # close helix at end of dict
        elif v[2] == "H" and counter == len(list(dssp_dict.items()))-1:
            new_helix.append(v)
            helices.append(new_helix)
        counter += 1
        prev_entry = v

    return helices


def get_dssp_dict(pdb_dir, pdb):
    """
    Performs dssp for one pdb file and returns dssp dict.
    """

    # parse DSSP to extract single alpha helices not all helices combined
    p = PDBParser()
    structure = p.get_structure("bla", pdb_dir+pdb)

    # Always take first model
    model = structure[0]

    # DSSP to get sec structure of aas
    dssp = dssp_to_dict(DSSP(model, pdb_dir+pdb))

    return dssp


def count_aa_dssp_dict(dssp_helices):
    """
    Returns: absolute counts for each aa in helices
    """

    # extract only aa sequence from each helix
    helices = []
    for helix_entry in dssp_helices:
        helix = []
        for aa_entry in helix_entry:
            helix.append(aa_entry[1])
        helices.append(helix)

    aa_in_helices = count_aa(helices)

    return aa_in_helices


def define_proteinogeneic_aas():
    """
    Defining all one letter code amino acids.
    """
    global aas
    aas = list(string.ascii_uppercase)
    for no_aa in ["B", "J", "O", "U", "X", "Z"]:
        aas.remove(no_aa)


def main():

    define_proteinogeneic_aas()

    # Extracting helices in pdb files
    pdb_dir = "pdb_structures/"
    pdbs = os.listdir(pdb_dir)
    pdb_counts_aa_helices = []
    print("Extracting non tm helices from sampled pdb structures...")
    for pdb in pdbs:
        print(pdb)
        try:
            dssp = get_dssp_dict(pdb_dir, pdb)
            dssp_helices = parse_dssp(dssp)
            pdb_counts_aa_helices += count_aa_dssp_dict(dssp_helices)
        except:
            print("dssp fails:", pdb)
            continue

    # Extracting helices in pdbtm files
    print("Extracting tm helices from pdbtm xml...")
    pdbtms = parse_pdbtm("pdbtm_reduced.xml")

    pdbtm_helices = get_aa_in_helices(pdbtms)
    pdbtm_nontm_ss = get_aa_NOT_in_helices(pdbtms)

    pdbtm_counts_aa_helices = count_aa(pdbtm_helices)
    pdbtm_counts_nontm_ss = count_aa(pdbtm_nontm_ss)

    # Create data for training and testing SVM
    max_helices = 1000  # to make equal numbers of tm /nontm helices

    data_tm = [list(x.values()) for x in pdbtm_counts_aa_helices[:max_helices]]
    data_nontm = [list(x.values())
                  for x in pdb_counts_aa_helices[:max_helices]]
    #data_nontm2 = [list(x.values()) for x in pdbtm_counts_nontm_ss]
    #label_nontm2 = [0 for i in range(len(data_nontm))]
    data = data_tm+data_nontm
    data = np.array(data, dtype=int)

    # labeling of data
    label_tm = [1 for i in range(len(data_tm))]
    label_nontm = [0 for i in range(len(data_nontm))]
    label = label_tm+label_nontm

    print("Number of tm helices:", len(data_tm))
    print("Number of non tm helices:", len(data_nontm))

    # Setting up SVM
    clf = svm.SVC(kernel='linear', C=1.0)

    print("Training and testing of linear SVM...")
    scores = cross_val_score(clf, data, label, cv=5)
    print("5-fold cross validation scores:", scores)
    print("Mean CV score:", sum(scores)/len(scores))

    #clf.fit(data, label)
    #tests = []
    #data = [list(x.values()) for x in pdb_aa_helices]
    # for x in pdb_aa_helices_data[max_helices:]:
    #    tests.append(int(clf.predict(np.array([x]))))
    # print(tests)


if __name__ == "__main__":
    main()
