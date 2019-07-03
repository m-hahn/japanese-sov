# /u/scr/mhahn/deps/memory-need-neural-wordforms/search-English_yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_OnlyWordForms_BoundedVocab.py_model_350445637_RANDOM_BY_TYPE.txt


import subprocess
import random

from math import exp
import sys

model = sys.argv[1]
if len(sys.argv) > 2:
  prescribedID = sys.argv[2]
else:
  prescribedID = None
model = "GROUND_SOV"
dropout1 = 0.15
emb_dim = 150
lstm_dim = 1024
layers = 2


learning_rate = 0.1
dropout2 = 0.35
batch_size = 2
sequence_length = 20
input_noising = 0.0

language = 'English'

for i in range(100):
   command = ['./python27', 'yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_OnlyWordForms_BoundedVocab_SOV.py', language, language, dropout1, emb_dim, lstm_dim, layers, learning_rate, model, dropout2, batch_size,input_noising,  sequence_length]
   if prescribedID is not None:
     command.append(prescribedID)
   command = map(str,command)
   subprocess.call(command)
   if prescribedID is not None:
      break
quit()
