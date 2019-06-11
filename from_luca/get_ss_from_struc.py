#!/usr/bin/env python
"""
File Name : get_ss_from_struc.py
Creation Date : 02-06-2019
Last Modified : Di 04 Jun 2019 11:38:51 CEST
Author : Luca Deininger
Function of the script :
"""

from Bio.PDB import *
import numpy as np
import os
import collections
import warnings
import string
from Bio import BiopythonWarning
import matplotlib.pyplot as plt

warnings.simplefilter('ignore', BiopythonWarning)


def dssp_to_dict(dssp_obj):
    """
    dssp returns a weird datastructure -> conversion to dict.
    """
    dssp = {}
    for k in list(dssp_obj.keys()):
        dssp[k] = dssp_obj[k]
    return dssp


def get_tm_chains(pdbtm_file, pdb):
    """
    Get chains of pdbtm structure containing transmembrane helices.
    """
    pdb_id = pdb[3:7]
    entries = [x for x in pdbtm_file if pdb_id in x]
    chains = [x[5] for x in entries]

    return chains

def get_aa_in_helices(pdb_dir, pdb, chains):
    """
    Returns: relative and absolute counts for each aa in helices present in given chains.
    """
    p = PDBParser()
    structure = p.get_structure("bla", pdb_dir+pdb)

    # Always take first model
    model = structure[0]

    # DSSP to get sec structure of aas
    dssp = dssp_to_dict(DSSP(model, pdb_dir+pdb))

    # filter dssp dict for helix and extract aa
    aa_in_helices = {k: v for k,
                     v in dssp.items() if v[2] == "H" and k[0] in chains}
    # keep only amino acid in dict (much more information like coordinated provided)
    aa_in_helices = [v[1] for k, v in aa_in_helices.items()]

    # count occurrence of amino acids + fill dict with no occurring aas
    aa_in_helices = collections.Counter(aa_in_helices)
    aa_in_helices = fill_dict_0s(aa_in_helices)

    # relative frequencies of aas
    sum_ = sum(aa_in_helices.values())
    if sum_ > 0:
        rel_aa_in_helices = {k: float(v/sum_)
                          for k, v in aa_in_helices.items()}
    else:
        rel_aa_in_helices = {}

    # pop 'aa' X
    aa_in_helices.pop('X', None)
    rel_aa_in_helices.pop('X', None)

    # ordering dict (important for plots later)
    aa_in_helices = collections.OrderedDict(sorted(aa_in_helices.items()))
    rel_aa_in_helices = collections.OrderedDict(sorted(rel_aa_in_helices.items()))

    return aa_in_helices, rel_aa_in_helices


def sum_up_counter(list_counter):
    """
    Provided a list of MULTIPLE Counter dictionaries it returns ONE Counter dictionary
    uniting all counts.
    """
    result = collections.Counter()
    for x in list_counter:
        result += x

    result = collections.OrderedDict(sorted(result.items()))
    return result


def fill_dict_0s(counter_dict):
    """
    For every aa not in counter dict: Add: aa->0.
    """
    for x in aas:
        if x not in counter_dict:
            counter_dict[x] = 0

    return counter_dict


def get_dist_aas(aa_in_helices):
    """
    Returns the distributions of the counts of each amino acid across multiple counter_dicts
    by returning a list of 20 lists.
    """
    result = []
    for aa in aas:
        result.append(
            [v for counter_dict in aa_in_helices for k, v in counter_dict.items() if k[0] == aa])

    return result


def main():

    # defining all one letter code amino acids
    global aas
    aas = list(string.ascii_uppercase)
    for no_aa in ["B", "J", "O", "U", "X", "Z"]:
        aas.remove(no_aa)

    # Get all pdbtm ids + chain
    f = open('pdbtm_all_list.txt', 'r')
    pdbtm_all_ids = f.readlines()
    f.close()

    # directories of sampled pdbs and pdbtms
    pdb_dir = "pdb_structures/"
    pdbtm_dir = "pdbtm_structures/"

    pdbs = os.listdir(pdb_dir)
    pdbtms = os.listdir(pdbtm_dir)

    # Defining list that will contain multiple counter dicts of amino acids in helices
    pdb_aa_helices = []
    pdb_aa_helices_rel = []
    pdbtm_aa_helices = []
    pdbtm_aa_helices_rel = []
    #end_index = 5
    end_index = len(pdbs)

    for pdb in pdbtms[:end_index]:
        print("PDBTM:", pdb)
        chains = get_tm_chains(pdbtm_all_ids, pdb)
        result = get_aa_in_helices(pdbtm_dir, pdb, chains)
        pdbtm_aa_helices.append(result[0])
        pdbtm_aa_helices_rel.append(result[1])

    for pdb in pdbs[:end_index]:
        print("PDB:", pdb)
        try:
            result = get_aa_in_helices(
                pdb_dir, pdb, list(string.ascii_uppercase))
            pdb_aa_helices.append(result[0])
            pdb_aa_helices_rel.append(result[1])
        except:
            print("DSSP fails:", pdb)
            continue

    #print("PDBTM:", pdbtm_aa_helices)
    #print("PDB:", pdb_aa_helices)

    # Boxplots of amino acid distributions
    # Absolute values
    plt.figure(figsize=(12, 6))
    plt.boxplot(get_dist_aas(pdbtm_aa_helices), positions=[
                6+x*3-0.6 for x in range(20)], widths=0.6)
    plt.boxplot(get_dist_aas(pdb_aa_helices), positions=[
                4+x*3+0.6 for x in range(20)], widths=0.6)
    plt.xticks([2+x*3 for x in range(22)], [""]+aas+[""])
    plt.savefig('boxplot_abs.png')

    # Relative values
    plt.figure(figsize=(12, 6))
    plt.boxplot(get_dist_aas(pdbtm_aa_helices_rel), positions=[
                6+x*3-0.6 for x in range(20)], widths=0.6)
    plt.boxplot(get_dist_aas(pdb_aa_helices_rel), positions=[
                4+x*3+0.6 for x in range(20)], widths=0.6)
    plt.xticks([2+x*3 for x in range(22)], [""]+aas+[""])
    plt.savefig('boxplot_rel.png')

    # Barplot: Summing all values up
    pdb_aa_helices = sum_up_counter(pdb_aa_helices)
    pdb_aa_helices_rel = sum_up_counter(pdb_aa_helices_rel)
    pdbtm_aa_helices = sum_up_counter(pdbtm_aa_helices)
    pdbtm_aa_helices_rel = sum_up_counter(pdbtm_aa_helices_rel)

    # Bar plot of frequency of amino acids
    ind = np.arange(20)
    width = 0.35

    # Absolute values
    plt.figure(figsize=(10, 6))
    plt.bar(ind, pdb_aa_helices.values(), width, color='r', label='PDB')
    plt.bar(ind+width, pdbtm_aa_helices.values(),
            width, color='g', label='PDBTM')
    plt.xticks(ind + width / 2, pdb_aa_helices.keys())
    plt.legend(loc='best')
    plt.savefig('barplot_abs.png')

    # Relative values
    plt.figure(figsize=(10, 6))
    plt.bar(ind, pdb_aa_helices_rel.values(),
            width, color='r', label='PDB')
    plt.bar(ind+width, pdbtm_aa_helices_rel.values(),
            width, color='g', label='PDBTM')
    plt.xticks(ind + width / 2, pdb_aa_helices.keys())
    plt.legend(loc='best')
    plt.savefig('barplot_rel.png')


main()
