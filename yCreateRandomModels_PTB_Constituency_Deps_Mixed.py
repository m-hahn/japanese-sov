import subprocess
import random

from math import exp
import sys

model = sys.argv[1]
if len(sys.argv) > 2:
  prescribedID = sys.argv[2]
else:
  prescribedID = None

assert model in ["REAL_REAL", "RANDOM_BY_TYPE"], model



# /u/scr/mhahn/deps/memory-need-neural-wordforms/search-PTB_yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_OpenVocab_WordForms_Dropout_Constituency_Deps.py_model_829970015_RANDOM_BY_TYPE.txt

dropout1 = 0.25
emb_dim = 150
lstm_dim = 256
layers = 2

# -6.907755278982137
#>>> log(b)
#-2.995732273553991

learning_rate = 2.5
dropout2 = 0.05
batch_size = 16
sequence_length = 20
input_noising = 0.05

for i in range(20):
   command = ["./python27", "yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_OpenVocab_WordForms_Dropout_Constituency_Deps_Mixed.py", "PTB", "PTB", dropout1, emb_dim, lstm_dim, layers, learning_rate, model, dropout2, batch_size,input_noising,  sequence_length]
   if prescribedID is not None:
     command.append(prescribedID)
   command = map(str,command)
   subprocess.call(command)
   if prescribedID is not None:
      break

