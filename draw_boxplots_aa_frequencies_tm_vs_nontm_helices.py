#!/usr/bin/env python
"""
File Name : get_ss_from_struc.py
Creation Date : 02-06-2019
Last Modified : Mi 26 Jun 2019 20:36:59 CEST
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
#import matplotlib.pyplot as plt
import pickle

warnings.simplefilter('ignore', BiopythonWarning)
from pylab import plot, show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes

# function for setting the colors of the box plots pairs
def setBoxColors(bp, color_):
    for i in range(len(bp['boxes'])):
        setp(bp['boxes'][i], color=color_)
    for i in range(len(bp['caps'])):
        setp(bp['caps'][i], color=color_)
    for i in range(len(bp['whiskers'])):
        setp(bp['whiskers'][i], color=color_)
    for i in range(len(bp['medians'])):
        setp(bp['medians'][i], color=color_)


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
    rel_aa_in_helices = collections.OrderedDict(
        sorted(rel_aa_in_helices.items()))

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


def get_distribution(aa_in_helices, aas_of_interest):
    """
    Returns the distributions of the counts of each amino acid across multiple counter_dicts
    by returning a list of 20 lists.
    """
    result = []
    for aa in aas_of_interest:
        result.append(
            [v for counter_dict in aa_in_helices for k, v in counter_dict.items() if k[0] == aa])

    return result


def parse_files(pdbs, pdbtms, pdbtm_all_ids, pdb_dir, pdbtm_dir):
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

    return pdb_aa_helices, pdbtm_aa_helices, pdb_aa_helices_rel, pdbtm_aa_helices_rel


def export_(pdb_aa_helices, pdbtm_aa_helices, pdb_aa_helices_rel, pdbtm_aa_helices_rel):
    folder = "serialized/draw_boxplot_"
    pickle.dump(pdb_aa_helices, open(folder+"pdb_aa_helices.p", "wb"))
    pickle.dump(pdbtm_aa_helices, open(folder+"pdbtm_aa_helices.p", "wb"))
    pickle.dump(pdb_aa_helices_rel, open(folder+"pdb_aa_helices_rel.p", "wb"))
    pickle.dump(pdbtm_aa_helices_rel, open(
        folder+"pdbtm_aa_helices_rel.p", "wb"))


def import_():
    print("Importing serialized datastructures...")
    folder = "serialized/draw_boxplot_"
    pdb_aa_helices = pickle.load(open(folder + "pdb_aa_helices.p", "rb"))
    pdbtm_aa_helices = pickle.load(open(folder + "pdbtm_aa_helices.p", "rb"))
    pdb_aa_helices_rel = pickle.load(
        open(folder + "pdb_aa_helices_rel.p", "rb"))
    pdbtm_aa_helices_rel = pickle.load(
        open(folder + "pdbtm_aa_helices_rel.p", "rb"))
    return pdb_aa_helices, pdbtm_aa_helices, pdb_aa_helices_rel, pdbtm_aa_helices_rel


def main():

    parse_again=False # True

    # directories of sampled pdbs and pdbtms
    pdb_dir = "train_pdb_structures/"
    pdbtm_dir = "pdbtm_structures/"

    # defining all one letter code amino acids
    global aas
    aas = list(string.ascii_uppercase)
    for no_aa in ["B", "J", "O", "U", "X", "Z"]:
        aas.remove(no_aa)

    # Get all pdbtm ids + chain
    f = open('data/pdbtm_all_list.txt', 'r')
    pdbtm_all_ids = f.readlines()
    f.close()


    if parse_again:
        pdbs = os.listdir(pdb_dir)
        pdbtms = os.listdir(pdbtm_dir)
        pdb_aa_helices, pdbtm_aa_helices, pdb_aa_helices_rel, pdbtm_aa_helices_rel = parse_files(pdbs, pdbtms, pdbtm_all_ids, pdb_dir, pdbtm_dir)
        export_(pdb_aa_helices, pdbtm_aa_helices, pdb_aa_helices_rel, pdbtm_aa_helices_rel)
    else:
        pdb_aa_helices, pdbtm_aa_helices, pdb_aa_helices_rel, pdbtm_aa_helices_rel=import_()

    # Boxplots of amino acid distributions
    print("Plotting Boxplots...")
   # Relative values without outliers
   # all together
    fig = figure(figsize=(12, 6))
    ax = axes()
    nr_aas=len(aas)
    bp=boxplot(get_distribution(pdbtm_aa_helices_rel, aas), positions=[
                6+x*3-0.6 for x in range(nr_aas)], widths=0.6, showfliers=False)
    setBoxColors(bp, "blue")
    bp=boxplot(get_distribution(pdb_aa_helices_rel, aas), positions=[
                4+x*3+0.6 for x in range(nr_aas)], widths=0.6, showfliers=False)
    setBoxColors(bp, "black")
    ylim(0,0.3)
    ax.set_xticklabels([""]+aas+[""])
    ax.set_xticks([2+x*3 for x in range(nr_aas+2)])

    # draw temporary red and blue lines and use them to create a legend
    hB, = plot([1,1],'k-', color = 'black')
    hR, = plot([1,1],'k-', color = 'blue')
    legend((hB, hR),('NON-TM helices', 'TM helices'))
    hB.set_visible(False)
    hR.set_visible(False)
    savefig('figures/boxplot_rel_all_aas_without_outliers.png')

   # hydrophobic aas
    fig = figure(figsize=(8, 6))
    ax = axes()
    #plt.figure(figsize=(12, 6))
    hydrophobic=["C", "F","G", "I", "L", "V", "W"]
    nr_aas=len(hydrophobic)
    bp=boxplot(get_distribution(pdbtm_aa_helices_rel, hydrophobic), positions=[
                6+x*3-0.6 for x in range(nr_aas)], widths=0.6, showfliers=False)
    setBoxColors(bp, "blue")
    bp=boxplot(get_distribution(pdb_aa_helices_rel, hydrophobic), positions=[
                4+x*3+0.6 for x in range(nr_aas)], widths=0.6, showfliers=False)
    setBoxColors(bp, "black")
    ylim(0,0.3)
    ax.set_xticklabels([""]+hydrophobic+[""])
    ax.set_xticks([2+x*3 for x in range(nr_aas+2)])

    # draw temporary red and blue lines and use them to create a legend
    hB, = plot([1,1],'k-', color = 'black')
    hR, = plot([1,1],'k-', color = 'blue')
    legend((hB, hR),('NON-TM helices', 'TM helices'))
    hB.set_visible(False)
    hR.set_visible(False)
    savefig('figures/boxplot_rel_hydrophobic_without_outliers.png')

    # Plotting hydrophlic amino acids
    fig = figure(figsize=(8, 6))
    ax = axes()
    hydrophilic=["D", "E", "K", "R"]
    nr_aas=len(hydrophilic)
    bp=boxplot(get_distribution(pdbtm_aa_helices_rel, hydrophilic), positions=[
                6+x*3-0.6 for x in range(nr_aas)], widths=0.6, showfliers=False)
    setBoxColors(bp, "blue")
    bp=boxplot(get_distribution(pdb_aa_helices_rel, hydrophilic), positions=[
                4+x*3+0.6 for x in range(nr_aas)], widths=0.6, showfliers=False)
    setBoxColors(bp, "black")
    ylim(0,0.3)
    ax.set_xticklabels([""]+hydrophilic+[""])
    ax.set_xticks([2+x*3 for x in range(nr_aas+2)])

    # draw temporary red and blue lines and use them to create a legend
    hB, = plot([1,1],'k-', color = 'black')
    hR, = plot([1,1],'k-', color = 'blue')
    legend((hB, hR),('NON-TM helices', 'TM helices'))
    hB.set_visible(False)
    hR.set_visible(False)
    savefig('figures/boxplot_rel_hydrophilic_without_outliers.png')


main()
