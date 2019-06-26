main.py:
	- performs annotation and validation of given data
	- uses a command line parser
	- for comfortable usage, use -t 500pdb_100pdbtm; the parsed serialized data is then used for the classifier:
        >> python3 main.py -t 500pdb_100pdbtm
	- else use own directory containing pdb structures
	- outputs validation scores for classifier
	- by default is SVM_abs used

General usage:
usage: main.py [-h] [-d DIRECTORY] [-s {abs,abs_ext,rel}]
               [-t {None,500pdb_100pdbtm,0pdb_500pdbtm}]

Classifier for membrane proteins

optional arguments:
  -h, --help            show this help message and exit
  -d DIRECTORY,         --directory:                    The directory with the pdb files.
  -s ,                  --svm_type {abs,abs_ext,rel}    Type of svm to use. (default: abs)
  -t ,                  --test_data {None,500pdb_100pdbtm,0pdb_500pdbtm}
                                Allows testing with given datasets.(default: None)

The -d option allows the testing of any directory with pdb files in it.
The -s option allows the user to choose from three different svm.
The -t option allows to the user to choose from different provided test datasets see options above.

if the option -t is given the -d option is ignored and the test_dataset is analyzed.  

	

For default usage of classifier the following scripts are not necessary:

retrieve_pdbs.py:
	- samples given number of pdb and pdbtm structures (always the same, random seed was set)
	- and saves them to out_dir
	- and downloads them
	- it is ensured that no sampled pdb is contained in the pdbtm
	- saves downloaded files in directory {1}pdb_{2}pdbtm/ with {1}=nr of pdbs and {2}=nr of pdbtms

train_SVM.py
	- uses by default serialized tm and non tm helices and builds a linear SVM classifier with absolute amino acid counts on it
	- to use custom data, change pdb_dir and parse_again to True
	- pdb_dir contains pdb structures the SVM is trained on
	- to train "abs_ext" or "rel" SVM change svm_type in main() to "abs_ext" or "rel"

draw_boxplots_aa_frequencies_tm_vs_nontm_helices.py
	- uses by default serialized tm and non tm helices to draw boxplots of amino acid frequencies


The rest of the data explained (within ./data):

pdbtmall.xml:
	- xml downloaded from the PDBTM website

pdbtm_all_list.txt:
	- downloaded from PDBTM
	- contains all PDB ids of PDBTM structures
	- used e.g. for retrieving_pdbs.py

pdb_all_list.txt
	- downloaded from PDB
	- contains all PDB ids
	- used e.g. for retrieving_pdbs.py

