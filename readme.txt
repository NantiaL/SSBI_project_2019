retrieve_pdbs.py:
	- samples 100 pdb and 100 pdbtm structures (always the same, random seed was set)
	- and downloads them
	- saves downloaded files in directories pdb_structures/ and pdbtm_structures/

parse_pdbtm_xml.py :
	- parses pdbtm and pdb files
	- extracts tm (from pdbtm) and nontm (from pdb) helices
	- and builds a linear SVM classifier for the identification of tm helices

get_ss_from_structure.py:
	- was previously used for parsing and creating plots
	- boxplots and barplots contained the distribution of the 20 amino acids in helices of 
	  pdb structures vs. helices of pdbtm_structures
