from sklearn import svm
from train_SVM import seqs_to_svm_input, fill_dict_0s, define_proteinogeneic_aas
import pickle

# needed
define_proteinogeneic_aas()

# load the trained SVM from disk
filename = 'trained_SVM.sav'
trained_SVM = pickle.load(open(filename, 'rb'))

test_seqs = ["RQKLQNLFINFCLILICLLLICIIV",
             "ARQKLQNLFINFCLILICLLLICIIV", "RQKLQNLFINFCLILICLLLICIIV"]
print("Test sequences (all from pdbtm):", test_seqs)
test_input = seqs_to_svm_input(test_seqs)

# Predict TM/NONTM of given test_seqs (1: TM, 0: NONTM)
print("Prediction:", trained_SVM.predict(test_input))

