#!/usr/bin/env python
"""
File Name : main.py
Creation Date : 13-06-2019
Last Modified : Sa 22 Jun 2019 18:56:06 CEST
Author : Luca Deininger
Function of the script :
"""

# import xml.etree.cElementTree as ET
from Bio.PDB import *
from train_SVM import pdn
import os
import random
import string
import warnings
import collections
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from Bio import BiopythonWarning
import pickle
import copy

from membrane_approximator import approximate_membrane
from test_results import test_result_against_pdbtm


def main():
    define_proteinogenic_aas()

    # mixed pdbs and pdbtms
    #pdb_dir = "50pdb_0pdbtm/"
    #pdb_dir = "0pdb_25pdbtm/"
    pdb_dir = "50pdb_25pdbtm/"
    #pdb_dir = "test_pdbtm/"

    parse_again = False#True

    if parse_again==True:
        helix_seqs, helix_info, helix_c_alphas = parse_pdbs(pdb_dir)
        export_dicts(helix_seqs, helix_info, helix_c_alphas, pdb_dir)
    else:
        helix_seqs, helix_info, helix_c_alphas = import_dicts(pdb_dir)

    # Annotate helices with SVM
    svm_name="trained_SVM.sav"
    #svm_name="trained_SVM_rel.sav"
    #svm_name="trained_SVM_tm_non_alpha_helices_ss_included.sav"
    #svm_name="trained_SVM_lin_c_5.sav"
    trained_svm = pickle.load(open("serialized/"+svm_name, 'rb'))
    helix_svm_annotations = annotate_helices(trained_svm, helix_seqs)

    # Annotate helices with PDBTM
    pdbtm_annotations = annotate_pdbtm(helix_info)

    # just to show results, might be easier to understand structure by this
    for pdb_id in list(helix_seqs.keys()):
        print(pdb_id)

        if not test_annotations(helix_svm_annotations, pdb_id):
            test_result_against_pdbtm(pdb_id, helix_svm_annotations[pdb_id], [0, 0, 0], [0, 0, 0])
            print()
            continue

        # print("Sequences:", helix_seqs[pdb_id])
        # print("Annotations:", helix_svm_annotations[pdb_id])
        # print("Infos:", helix_info[pdb_id])
        # print("C_alphas:", helix_c_alphas[pdb_id])

        mem_axis, mem_position = approximate_membrane(
            pdb_id, helix_c_alphas, helix_svm_annotations)

        print("Approx. membrane axis:", mem_axis)
        print("Approx. membrane pos :", mem_position)
        # pdbid, helix_annotations, membrane_axis, membrane_position
        test_result_against_pdbtm(pdb_id, helix_svm_annotations[pdb_id], mem_axis, mem_position)

        print()

    # Validate SVM predictions
    tp, tn, fp, fn, tpr, fpr = validate(helix_info, pdbtm_annotations, helix_svm_annotations)
    print("TP: ",tp, "\nTN: ", tn, "\nFP: ", fp, "\nFN: ", fn, "\nTPR: ", tpr, "\nFPR: ", fpr)
    print("Correctly classified: ", (tn+tp)/(tp+tn+fp+fn)) 
    print("Incorrectly classified: ", (fp+fn)/(tp+tn+fp+fn)) 


def validate(helix_info, truth, predictions):
    """
    Returns tp, tn, fp, fn, TPR and FPR for svm annotations vs pdbtm annotations.
    """

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for pdbid, truth_annot in truth.items():
        predict_annot = predictions[pdbid]
        for i in range(len(truth_annot)):
            if truth_annot[i][0] == 1 and predict_annot[i][0] == 1:
                tp += 1
            elif truth_annot[i][0] == 0 and predict_annot[i][0] == 0:
                tn += 1
            elif truth_annot[i][0] == 0 and predict_annot[i][0] == 1:
                fp += 1
    #            print(pdbid)
    #            print(truth_annot[i], predict_annot[i])
    #            print(helix_info[pdbid][i][0], helix_info[pdbid][i][len(helix_info[pdbid][i])-1])
            elif truth_annot[i][0] == 1 and predict_annot[i][0] == 0:
                fn += 1
     #           print(pdbid)
     #           print(truth_annot[i], predict_annot[i])
     #           print(helix_info[pdbid][i][0], helix_info[pdbid][i][len(helix_info[pdbid][i])-1])

    try:
        tpr = tp/(tp+fn)
    except ZeroDivisionError:
        tpr="Undef"

    try:
        fpr = fp/(fp+tn)
    except ZeroDivisionError:
        fpr="Undef"

    return tp, tn, fp, fn, tpr, fpr

def annotate_pdbtm(helix_info):
    """
    Returns dict with pdbid -> [[1], [0], [0], [1], [1]] 0 nontm helix, 1 for every tm helix
    """
    pdbtm = parse_pdbtm_xml("pdbtmall.xml")
    pdbtm_annotations = copy.deepcopy(helix_info)

    for pdbid, helices in helix_info.items():
        pdb_annotation = []
        if pdbid in pdbtm:
            # for every helix: check if it is annotated as tm in pdbtm
            for helix in helices:
                start = helix[0][1]
                end = helix[len(helix)-1][1]
                chain=helix[0][0]
                annot = 0
                for cand in pdbtm[pdbid]:
                    start_pdbtm = cand[0][1]
                    end_pdbtm = cand[1][1]
                    chain_pdbtm=cand[0][0]
                    overlap=2

            #        # annotate helix < length 5 as non_tm
            #        if (end-start)<5:
            #            annot=0
            #            break

                    # if different chain -> continue
                    if chain!=chain_pdbtm:
                        continue

                    # no overlap with pdbtm cand -> continue
                    if start < start_pdbtm and end <= start_pdbtm:
                        continue

                    # no overlap with pdbtm cand -> continue
                    elif start >= end_pdbtm and end > end_pdbtm:
                        continue

                    # only annotate helix as tm if at least half of the length of both helices overlap
                    elif start < start_pdbtm:
                        if ((end-start_pdbtm) > ((end-start)/(overlap))) and ((end-start_pdbtm) > ((end_pdbtm-start_pdbtm)/overlap)):
                           # print(pdbid, start, end, start_pdbtm, end_pdbtm)
                            annot=1
                            break
                        else:
                            continue
                    else:
                        if ((end_pdbtm-start) > ((end-start)/(overlap))) and ((end_pdbtm-start)>((end_pdbtm-start_pdbtm)/overlap)):
                           # print(pdbid, start, end, start_pdbtm, end_pdbtm)
                            annot=1
                            break
                        else:
                            continue
      #          print("pdb:", helix[0], "/", helix[len(helix)-1], "pdbtm:", cand[0], "/", cand[1], "annot:", annot)
                pdb_annotation.append([annot])

            pdbtm_annotations[pdbid] = pdb_annotation

        # if pdb file not in pdbtm contained -> annotate all 0s
        else:
            pdbtm_annotations[pdbid] = [[0] for x in helices]

    return pdbtm_annotations


def parse_pdbtm_xml(xml_file):

    tree = ET.parse('pdbtmall.xml')
    root = tree.getroot()
    root.tag, root.attrib
    pdbtms = {}
    # Parse XML
    for pdbtm in root:
        pdbid = pdbtm.attrib["ID"]
        for child in pdbtm:
            tag = child.tag[23:]
            if tag == "CHAIN":
                chainid = child.attrib["CHAINID"]
                for child2 in child:
                    tag2 = child2.tag[23:]
                    # extracting sequence
                    if tag2 == "SEQ":
                        seq = child2.text
                        seq = seq.replace("\n", "")
                        seq = seq.replace(" ", "")
                    # extracting sec structures
                    elif tag2 == "REGION":
                        seq_beg = int(child2.attrib["seq_beg"])
                        seq_end = int(child2.attrib["seq_end"])
                        pdb_beg = int(child2.attrib["pdb_beg"])
                        pdb_end = int(child2.attrib["pdb_end"])
                        type_ = child2.attrib["type"]
                        if type_ != "H":
                            continue

                        if pdbid in pdbtms:
                            pdbtms[pdbid].append(
                                [(chainid, pdb_beg), (chainid, pdb_end)])
                        else:
                            pdbtms[pdbid] = [
                                [(chainid, pdb_beg), (chainid, pdb_end)]]

    return pdbtms


def test_annotations(annotation, pdb_id):
    count_tm_helices = 0
    indices = []
    for i in range(len(annotation[pdb_id])):
        a = annotation[pdb_id][i]
        if a[0] == 1:
            count_tm_helices += 1
            indices.append(i)

    if count_tm_helices <= 1:
        print("Protein probably not trans membrane too few tm helices.")
        return False

    avg_confidence = 0
    for idx in indices:
        avg_confidence += annotation[pdb_id][idx][1]

    avg_confidence = avg_confidence/count_tm_helices

    # TODO find good criteria:
    # if avg_confidence < 0.9 and count_tm_helices < 5:
    #    return false

    return True


def export_dicts(helix_seqs, helix_info, helix_c_alphas, pdb_dir):
    folder = "serialized/main_"+pdb_dir[0:-1]
    pickle.dump(helix_seqs, open(folder+"helix_seqs.p", "wb"))
    pickle.dump(helix_info, open(folder+"helix_info.p", "wb"))
    pickle.dump(helix_c_alphas, open(folder+"helix_c_alphas.p", "wb"))


def import_dicts(pdb_dir):
    folder = "serialized/main_"+pdb_dir[0:-1]
    helix_seqs = pickle.load(open(folder + "helix_seqs.p", "rb"))
    helix_info = pickle.load(open(folder + "helix_info.p", "rb"))
    helix_c_alphas = pickle.load(
        open(folder + "helix_c_alphas.p", "rb"))
    return helix_seqs, helix_info, helix_c_alphas


def parse_pdbs(pdb_dir):
    """
    returns: 3 dictionaries with key value pairs like in the following (all having same basic structure):

    helix_seqs: pdb_id -> [["helix 1 sequence"], ["helix 2 sequence"], ... ["helix x sequence"]]
    helix_info: Contains for every amino acid in the helix the chain_id and res_id:
                pdb_id -> [[(aa1 chain_id, res_id) (aa2 chain_id, res_id)...], [helix 2 etc same],  ... ]
    helix_c_alphas: Same like in prepare_membrane_approximator.py:
                pdb_id -> [[C-alpha1 Vectory xyz, C-alpha2 Vectory xyz...], [helix 2 etc same], ... ]
    """
    pdbs = os.listdir(pdb_dir)

    print("Parsing PDBs...")
    helix_seqs = {}
    helix_info = {}
    helix_c_alphas = {}
    for pdb in pdbs:
        pdb_id = pdb[3:7]
        print(pdb_id)
        try:
            dssp = get_dssp_dict(pdb_dir, pdb)
            curr_helix_seqs, curr_helix_info = parse_dssp(dssp, pdb_dir, pdb)
            curr_helix_c_alphas = get_c_alphas(pdb_dir, pdb, curr_helix_info)

            helix_seqs[pdb_id] = curr_helix_seqs
            helix_info[pdb_id] = curr_helix_info
            helix_c_alphas[pdb_id] = curr_helix_c_alphas
        except:
            print("DSSP fails")
            continue

    return helix_seqs, helix_info, helix_c_alphas


def annotate_helices(svm, helix_seqs):
    """
    Returns: dictionary helix_annotations. For every helix in helix seqs annotate the more probable annotation 0 (NONTM) or 1 (TM) and its probability
    pdb_id -> [[0, prob(0)], [0, prob(0)] [1, prob(1)]... [0, prob(0)]]

    if pdb file doesn't contain helix: pdb_id -> "Error: No helices existent, thus annotation not possible"
    """
    helix_annotations = {}
    for pdb_id, v in helix_seqs.items():
        svm_input = seqs_to_svm_input(v)

        if len(svm_input) == 0:
            helix_annotations[pdb_id] = "Error: No helices existent, thus annotation not possible"
            continue

        predictions = svm.predict_proba(svm_input)
        predictions = [[list(x).index(max(x)), max(x)] for x in predictions]
        helix_annotations[pdb_id] = predictions

    return helix_annotations


def seqs_to_svm_input(seqs):
    count_dicts = count_aa(seqs)
    counts = [list(count_dict.values()) for count_dict in count_dicts]
    return np.array(counts)


def count_aa(seqs):
    """
    Count amino acids for each seq in seqs.
    seqs must be of format: [["seq1"], ["seq2"]...]
    Can be used as preprocessing of input of SVM, given sequence/sequences of helices.
    """
    counter_seqs = []
    for seq in seqs:
        seq = seq[0]
        counter_seq = collections.Counter(seq)
        counter_seq = fill_dict_0s(counter_seq)
        counter_seq = pop_non_aas(counter_seq)
        counter_seq = collections.OrderedDict(sorted(counter_seq.items()))
        counter_seqs.append(counter_seq)

    return counter_seqs


def fill_dict_0s(counter_dict):
    """
    For every aa not in counter dict: Add: aa->0.
    """
    for x in aas:
        if x not in counter_dict:
            counter_dict[x] = 0

    return counter_dict


def get_trained_svm(filename):
    trained_SVM = pickle.load(open(filename, 'rb'))


def parse_dssp(dssp_dict, pdb_dir, pdb_id):
    """
    Parses dssp dict

    """
    helix_seqs = []
    helix_info = []
    counter = 0
    prev_ss_type = ""
    for k, v in dssp_dict.items():
        chain_id = k[0]
        res_id = k[1][1]
        aa = v[1].upper()
        ss_type = v[2]

        # sometimes helices with non-proteinogenic aas -> skip
        if aa not in aas:
            counter += 1
            continue

        # starting new helix at start of dict / if prev ss type was not helix
        if (ss_type == "H" and counter == 0) or (ss_type == "H" and prev_ss_type != "H"):
            new_helix_seq = aa
            new_helix_info = [(chain_id, res_id)]

        # appending to current helix
        elif prev_ss_type == "H" and ss_type == "H":
            new_helix_seq += aa
            new_helix_info.append((chain_id, res_id))

        # close and save new found helix
        elif ss_type != "H" and prev_ss_type == "H":
            helix_seqs.append([new_helix_seq])
            helix_info.append(new_helix_info)

        # close helix at end of dict
        if ss_type == "H" and counter == len(list(dssp_dict.items()))-1:
            helix_seqs.append([new_helix_seq])
            helix_info.append(new_helix_info)

        counter += 1
        prev_ss_type = ss_type

    return helix_seqs, helix_info


def get_c_alphas(pdb_dir, pdb_id, helix_info):
    """
    Returns
    helix_c_alphas: Same like in prepare_membrane_approximator.py:
                pdb_id -> [[C-alpha1 Vectory xyz, C-alpha2 Vectory xyz...], [helix 2 etc same], ... ]

    """

    p = PDBParser()
    structure = p.get_structure('X', pdb_dir+pdb_id)
    helix_c_alpha = []
    for helix in helix_info:
        new_helix_c_alpha = []
        for c_alpha in helix:
            chain_id = c_alpha[0]
            res_id = c_alpha[1]

            atom = structure[0][chain_id][res_id]['CA']
            atom_coords = atom.get_vector()

            new_helix_c_alpha.append(atom_coords)

        helix_c_alpha.append(new_helix_c_alpha)

    return helix_c_alpha


def dssp_to_dict(dssp_obj):
    """
    dssp returns a weird datastructure -> conversion to dict.
    """
    dssp = collections.OrderedDict()
    for k in list(dssp_obj.keys()):
        dssp[k] = dssp_obj[k]
    return dssp


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


def define_proteinogenic_aas():
    """
    Defining all one letter code amino acids.
    """
    global aas
    aas = list(string.ascii_uppercase)
    for no_aa in ["B", "J", "O", "U", "X", "Z"]:
        aas.remove(no_aa)


def pop_non_aas(dict_):
    """
    Somtimes weird aa counts.
    """
    non_aas = ["-", "?", "B", "J", "O", "U", "X", "Z"]
    for aa in non_aas:
        dict_.pop(aa, None)
    return dict_


if __name__ == "__main__":
    main()
