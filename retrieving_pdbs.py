#!/usr/bin/env python
"""
File Name : retrieving_pdbs.py
Creation Date : 02-06-2019
Last Modified : Mi 26 Jun 2019 20:01:40 CEST
Author : Luca Deininger
Function of the script :
"""

import Bio
from Bio.PDB import PDBList
import numpy as np

# Number of to Number of to be sample pdbs and pdbtms
nr_pdb = 500
nr_pdbtm = 100
out_dir='{}pdb_{}pdbtm'.format(nr_pdb, nr_pdbtm)

# Don't change these parameters
pdbtm_file = "data/pdbtm_all_list.txt"
pdb_file = "data/pdb_all_list_resolution.txt"

# Get all pdbtm ids + chain
f = open(pdbtm_file, 'r')
pdbtm_all_ids = f.readlines()
f.close()

# Reduce pdbtm to pdb id
pdbtm_unique_ids = [x[:4].upper() for x in pdbtm_all_ids]
pdbtm_unique_ids=list(set(pdbtm_unique_ids))

# Get all pdb ids with resolution
f = open(pdb_file, 'r')
pdb_all_ids_reso = f.readlines()
f.close()

# filter pdbs < resolution
max_res = 2.8
pdb_ids = []
for x in pdb_all_ids_reso:
    try:
        curr_resolution = float(x.split("\t")[2].rstrip())
        if curr_resolution < max_res:
            # take only pdbs that are not in PDBTM
            if x[:4] not in pdbtm_unique_ids:
                pdb_ids.append(x[:4])
    except:
        continue

# Sample pdbs and pdbtms
np.random.seed(1996)
sampled_pdbs = np.random.choice(pdb_ids, nr_pdb, replace=False)

np.random.seed(1996)
sampled_pdbtms = np.random.choice(pdbtm_unique_ids, nr_pdbtm, replace=False)

print("Sampled PDBs:", sampled_pdbs)
print("Sampled PDBTMs:", sampled_pdbtms)

pdbl = PDBList()

# retrieve these from pdb database
for i in sampled_pdbs:
    pdbl.retrieve_pdb_file(i, pdir=out_dir,
                           file_format="pdb", overwrite=False)

for i in sampled_pdbtms:
    pdbl.retrieve_pdb_file(i, pdir=out_dir,
                           file_format="pdb", overwrite=False)
