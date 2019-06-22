#!/usr/bin/env python
"""
File Name : parse_pdbtm_xml.py
Creation Date : 05-06-2019
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
import pickle

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
    Can be used as preprocessing of input of SVM, given sequence/sequences of helices.
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
    counter = 0
    prev_entry = ["", "", ""]
    for k, v in dssp_dict.items():

        # sometimes helices with non-proteinogenic aas -> skip
        if v[2] not in aas:
            counter += 1
            continue

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
        if v[2] == "H" and counter == len(list(dssp_dict.items()))-1:
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

def relative_counts(list_of_list):
    for i in range(len(list_of_list)):
        sum_counts=sum(list_of_list[i])
        for j in range(len(list_of_list[i])):
            try:
                list_of_list[i][j]=(list_of_list[i][j])/sum_counts
            except ZeroDivisionError:
                list_of_list[i][j]=0.0
    return list_of_list


def get_data_and_labels(pdb_dir, pdbtm_file, nr_tm, nr_nontm):
    # Extracting helices in pdb files
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

    data_tm = [list(x.values()) for x in pdbtm_counts_aa_helices[:nr_tm]]
    data_tm=relative_counts(data_tm)

    data_nontm = [list(x.values())
                  for x in pdb_counts_aa_helices[:nr_nontm]]
    data_nontm=relative_counts(data_nontm)

    #data_nontm2 = [list(x.values()) for x in pdbtm_counts_nontm_ss[:nr_nontm]]
    data = data_tm+data_nontm#+data_nontm2
    data = np.array(data, dtype=float)

    # labeling of data
    label_tm = [1 for i in range(len(data_tm))]
    label_nontm = [0 for i in range(len(data_nontm))]
    #label_nontm2 = [0 for i in range(len(data_nontm2))]
    label = label_tm+label_nontm#+label_nontm2

    return data, label


def seqs_to_svm_input(seqs):
    count_dicts = count_aa(seqs)
    counts = [list(count_dict.values()) for count_dict in count_dicts]
    return np.array(counts)


def export_(data, label):
    folder = "serialized/train_SVM_"
    pickle.dump(data, open(folder+"data.p", "wb"))
    pickle.dump(label, open(folder+"label.p", "wb"))


def import_():
    print("Importing serialized data and labels...")
    folder = "serialized/train_SVM_"
    data = pickle.load(open(folder + "data.p", "rb"))
    label = pickle.load(open(folder + "label.p", "rb"))
    return data, label


def fit_SVM(clf, data, label):
    clf.fit(data, label)
    return clf


def cv_SVM(clf, data, label, fold):
    print("Cross-Validation of linear SVM...")
    scores = cross_val_score(clf, data, label, cv=fold)
    print("{}-fold cross validation scores:".format(fold), scores)
    print("Mean CV score:", sum(scores)/len(scores))


def validate(data, label, predictions):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(label)):
        if label[i] == 1 and predictions[i] == 1:
            tp += 1
        elif label[i] == 0 and predictions[i] == 0:
            tn += 1
        elif label[i] == 0 and predictions[i] == 1:
            fp += 1
        elif label[i] == 1 and predictions[i] == 0:
            fn += 1

    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)

    return tpr, fpr


def main():

    define_proteinogeneic_aas()

    pdb_dir = "pdb_structures/"
    pdbtm_file = "pdbtm_reduced.xml"
    nr_tm = 3000
    nr_nontm = 3000

    # get data and label
    data, label = get_data_and_labels(pdb_dir, pdbtm_file, nr_tm, nr_nontm)
    export_(data, label)
    #data, label = import_()

    # setting up SVM
    clf = svm.SVC(kernel='linear', C=1.0, probability=True)
  #  clf2 = svm.SVC(kernel='linear', C=2.0, probability=True)
  #  clf3 = svm.SVC(kernel='linear', C=3.0, probability=True)

    # train SVM
    trained_SVM = fit_SVM(clf, data, label)
#    trained_SVM2 = fit_SVM(clf2, data, label)
#    trained_SVM3 = fit_SVM(clf3, data, label)
#    cv_SVM(clf, data, label, 10)
#    cv_SVM(clf2, data, label, 10)
#    cv_SVM(clf3, data, label, 10)

    # validate SVM
    predictions = trained_SVM.predict(data)
    print(validate(data, label, predictions))
#    predictions = trained_SVM2.predict(data)
#    print(validate(data, label, predictions))
#    predictions = trained_SVM3.predict(data)
#    print(validate(data, label, predictions))

    # save trained SVM to disk
    filename = 'serialized/trained_SVM_rel.sav'
    pickle.dump(trained_SVM, open(filename, 'wb'))


if __name__ == "__main__":
    main()
