import os
resus = [x for x in os.listdir("results/ptb/") if x.startswith("estimates-PTB_yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_OpenVocab_WordForms_Dropout_Constituency_Deps_Mixed.py_model_") and x.endswith("_REAL_REAL.txt")]

with open("results/ptb/tradeoff-summary.tsv", "w") as outFile:
 print >> outFile, "\t".join(["Model", "Condition", "Distance", "MI", "UnigramCE"])
 for result in resus:
   ID = result.split("_")[-3]
   try:
     with open("results/ptb/"+result, "r") as inFile:
       next(inFile)
       next(inFile)
       version = [None, None]
       version[0] = [float(x) for x in next(inFile).strip().split(" ")]
       version[1] = [float(x) for x in next(inFile).strip().split(" ")]
       unigramCE = (version[0][0] + version[1][0])/2
       for i in range(2):
         print >> outFile, "\t".join([str(x) for x in [ID, i, 0, 0, unigramCE]])

         for j in range(1, len(version[0])):
            print >> outFile, "\t".join([str(x) for x in [ID, i, j, version[i][j-1]- version[i][j], unigramCE]])
   except StopIteration:
       continue


