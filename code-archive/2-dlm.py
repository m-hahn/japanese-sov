# Computes counterfactual order

#/u/nlp/bin/stake.py -g 11.5g -s run-stats-pretrain2.json "python readDataDistEnglishGPUFree.py"

import random
import sys

from collections import deque

#objectiveName = "gradients_WordSurp"

language = sys.argv[1]
model = "REAL_REAL"
temperature = "Infinity"

assert temperature == "Infinity"

myID = random.randint(0,10000000)

posUni = set() #[ "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"] 

posFine = set() #[ "``", ",", ":", ".", "''", "$", "ADD", "AFX", "CC",  "CD", "DT", "EX", "FW", "GW", "HYPH", "IN", "JJ", "JJR",  "JJS", "-LRB-", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS",  "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "-RRB-", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  "WDT", "WP", "WP$", "WRB", "XX" ]

deps = ["acl", "acl:relcl", "advcl", "advmod", "amod", "appos", "aux", "auxpass", "case", "cc", "ccomp", "compound", "compound:prt", "conj", "conj:preconj", "cop", "csubj", "csubjpass", "dep", "det", "det:predet", "discourse", "dobj", "expl", "foreign", "goeswith", "iobj", "list", "mark", "mwe", "neg", "nmod", "nmod:npmod", "nmod:poss", "nmod:tmod", "nsubj", "nsubjpass", "nummod", "parataxis", "punct", "remnant", "reparandum", "root", "vocative", "xcomp"] 

#deps = ["acl", " advcl", " advmod", " amod", " appos", " aux", " case cc", " ccompclf", " compound", " conj", " cop", " csubjdep", " det", " discourse", " dislocated", " expl", " fixed", " flat", " goeswith", " iobj", " list", " mark", " nmod", " nsubj", " nummod", " obj", " obl", " orphan", " parataxis", " punct", " reparandum", " root", " vocative", " xcomp"]


from math import log, exp
from random import random, shuffle
import os

conll_header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]

my_fileName = __file__.split("/")[-1]

#from corpusIterator_FuncHead_V import CorpusIteratorFuncHead_V
from corpusIterator_V import CorpusIterator_V

originalDistanceWeights = {}


def makeCoarse(x):
   if ":" in x:
      return x[:x.index(":")]
   return x

def initializeOrderTable():
   orderTable = {}
   keys = set()
   vocab = {}
   distanceSum = {}
   distanceCounts = {}
   depsVocab = set()
   for partition in ["train", "dev"]:
     for sentence in CorpusIterator_V(language,partition, shuffleDataSeed=40).iterator():
      for line in sentence:
          vocab[line["word"]] = vocab.get(line["word"], 0) + 1
          line["coarse_dep"] = makeCoarse(line["dep"])
          depsVocab.add(line["coarse_dep"])
          posFine.add(line["posFine"])
          posUni.add(line["posUni"])
  
          if line["coarse_dep"] == "root":
             continue
          posHere = line["posUni"]
          posHead = sentence[line["head"]-1]["posUni"]
          dep = line["coarse_dep"]
          direction = "HD" if line["head"] < line["index"] else "DH"
          key = dep
          keyWithDir = (dep, direction)
          orderTable[keyWithDir] = orderTable.get(keyWithDir, 0) + 1
          keys.add(key)
          distanceCounts[key] = distanceCounts.get(key,0.0) + 1.0
          distanceSum[key] = distanceSum.get(key,0.0) + abs(line["index"] - line["head"])
   #print orderTable
   dhLogits = {}
   for key in keys:
      hd = orderTable.get((key, "HD"), 0) + 1.0
      dh = orderTable.get((key, "DH"), 0) + 1.0
      dhLogit = log(dh) - log(hd)
      dhLogits[key] = dhLogit
      originalDistanceWeights[key] = (distanceSum[key] / distanceCounts[key])
   return dhLogits, vocab, keys, depsVocab

#import torch.distributions
import torch.nn as nn
import torch
from torch.autograd import Variable


# "linearization_logprobability"
def recursivelyLinearize(sentence, position, result, gradients_from_the_left_sum):
   line = sentence[position-1]
   # Loop Invariant: these are the gradients relevant at everything starting at the left end of the domain of the current element
   allGradients = gradients_from_the_left_sum 
   line["length"] = 1
   # there are the gradients of its children
   if "children_DH" in line:
      for child in line["children_DH"]:
         assert child > 0
         if "removed" not in sentence[child-1]:
           length = recursivelyLinearize(sentence, child, result, allGradients)
           line["length"] += length

   result.append(line)

   line["length_postHead"] = 0
   if "children_HD" in line:
      for child in line["children_HD"]:
         assert child > 0
         if "removed" not in sentence[child-1]:
           length = recursivelyLinearize(sentence, child, result, allGradients)
           line["length"] += length
           line["length_postHead"] += length


   return line["length"]

import numpy.random

softmax_layer = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()



def orderChildrenRelative(sentence, remainingChildren, reverseSoftmax):
       logits = [(x, distanceWeights[stoi_deps[sentence[x-1]["dependency_key"]]]) for x in remainingChildren]
       logits = sorted(logits, key=lambda x:x[1], reverse=(not reverseSoftmax))
       childrenLinearized = map(lambda x:x[0], logits)
       return childrenLinearized           



def orderSentence(sentence, dhLogits, printThings):
   global model

   root = None
   logits = [None]*len(sentence)
   logProbabilityGradient = 0
   if model == "REAL_REAL":
      eliminated = []
   for line in sentence:
      line["coarse_dep"] = makeCoarse(line["dep"])
      if line["coarse_dep"] == "root":
          root = line["index"]
          continue
      if line["coarse_dep"].startswith("punct"): # assumes that punctuation does not have non-punctuation dependents!
         if model == "REAL_REAL":
            eliminated.append(line)
         continue


      if line["coarse_dep"] in ["aux", "mark", "case", "neg", "cc"]: #.startswith("punct"): # assumes that punctuation does not have non-punctuation dependents!
         if model == "REAL_REAL":
            eliminated.append(line)
         continue



      key = line["coarse_dep"]
      line["dependency_key"] = key
      dhLogit = dhWeights[stoi_deps[key]]
      dhSampled = (line["index"] < line["head"])

      
     
      direction = "DH" if dhSampled else "HD"
#      if line["coarse_dep"] == "nsubj":
 #        direction = "HD"

      if printThings: 
         print "\t".join(map(str,["ORD", line["index"], (line["word"]+"           ")[:10], ("".join(list(key)) + "         ")[:22], line["head"], dhSampled, direction, str(1/(1+exp(-dhLogits[key])))[:8], (str(distanceWeights[stoi_deps[key]])+"    ")[:8] , str(originalDistanceWeights[key])[:8]    ]  ))

      headIndex = line["head"]-1
      sentence[headIndex]["children"] = (sentence[headIndex].get("children", []) + [line["index"]])
      sentence[headIndex]["children_"+direction] = (sentence[headIndex].get("children_"+direction, []) + [line["index"]])

      head = sentence[headIndex]

   
   if model != "REAL_REAL":
      for line in sentence:
         if "children" in line:
            childrenLinearized = orderChildrenRelative(sentence, line["children"][:], False)
            line["children"] = childrenLinearized
   if model == "REAL_REAL":
       while len(eliminated) > 0:
          line = eliminated[0]
          del eliminated[0]
          if "removed" in line:
             continue
          line["removed"] = True
          if "children" in line:
            assert 0 not in line["children"]
            eliminated = eliminated + [sentence[x-1] for x in line["children"]]

   
   linearized = []
   recursivelyLinearize(sentence, root, linearized, 0)

              

   if model == "REAL_REAL":
      linearizedByRule = linearized
      linearized = filter(lambda x:"removed" not in x, sentence)
#   if False: # for checking correctness in the absence of reordering (e.g. placing nsubj somewhere else)
   if True or printThings or len(linearized) == 0:
   #  print "========================"
   #  print " ".join(map(lambda x:x["word"], sentence))
  #   LBR = " ".join(map(lambda x:x["word"], linearizedByRule))
 #    LR = " ".join(map(lambda x:x["word"], linearized))
   #  print LBR
   #  print LR
     LBR2 = " ".join(map(lambda x:x["posUni"], linearizedByRule))
     LR2 = " ".join(map(lambda x:x["posUni"], linearized)) 
   #  print LBR2
   #  print LR2
     if LBR2 != LR2:
        print "WARNING!!!"

   #  assert LBR == LR
    # assert len(linearized) == len(linearizedByRule)
   linearized = linearizedByRule



   # store new dependency links
   moved = [None] * len(sentence)
   for i, x in enumerate(linearized):
#      print x
      moved[x["index"]-1] = i
      x["reordered_index"] = i+1
 #  print moved
   for i,x in enumerate(linearized):
  #    print x
      if x["head"] == 0: # root
         x["reordered_head"] = 0
      else:
         x["reordered_head"] = 1+moved[x["head"]-1]


   if True or sentence[0]["word"] != "_":
     #print " ".join(map(lambda x:x["word"], linearized))
     for line in linearized:        
        if "children" in line:
           if line["posUni"] == "VERB":
     #         print(line["children"])
              if not (line["coarse_dep"] in ["root", "conj", "dep"] or (line["reordered_index"] - line["reordered_head"]) < 0):
                  print ("WARNING!!!", line["coarse_dep"])

              deps = ([(lambda y: (y["coarse_dep"], y["length"], y["length_postHead"], y["index"] < line["index"]))(sentence[x-1]) for x in line["children"] if "removed" not in sentence[x-1]])
              if "nsubj" in [x[0] for x in deps]:
                 #print(line["index"] - line["head"], line["coarse_dep"])
                 #print(deps)
                 # length between VERB and its head
                 dlVerb = (line["reordered_index"] - line["reordered_head"]) if line["coarse_dep"] != "root" else 0
                 # length between VERB and its preceding dependents including the subject
                 openDependencies = 0
                 lengthAccumulated = 0
                 for child in deps:
                    if not child[3]:
                       break
                    lengthAccumulated += child[2] + 1
                    lengthAccumulated += openDependencies * child[1]
                    openDependencies += 1
                 depsBefore = lengthAccumulated

                 openDependencies = 0
                 lengthAccumulated = 0
                 for child in deps[::-1]:
                    if child[3]:
                       break
                    lengthAccumulated += child[1] - child[2] 
                    lengthAccumulated += openDependencies * child[1]
                    openDependencies += 1
                 depsAfter = lengthAccumulated





                 openDependencies = 0
                 lengthAccumulated = 0
                 for child in deps:
                    if not child[3]:
                       break
                    if child[0] == "nsubj":
                      continue
                    lengthAccumulated += child[2] + 1
                    lengthAccumulated += openDependencies * child[1]
                    openDependencies += 1
                 depsBeforeWithoutSubj = lengthAccumulated


                 # length when placing the subject where it is
                 L_Real = abs(dlVerb) + depsBefore + depsAfter


                 # idealized length 
                 openDependencies = 0
                 lengthAccumulated = 0
                 for child in sorted([x for x in deps if x[3]], key=lambda x:-x[1]):
                    lengthAccumulated += child[2] + 1
                    lengthAccumulated += openDependencies * child[1]
                    openDependencies += 1
                 depsBeforeShortest = lengthAccumulated

                 L_SV_ideal = abs(dlVerb) + depsBeforeShortest + depsAfter


               
                 subjects = [x for x in deps if x[0] == "nsubj"] #, key=lambda x:x[1])
                 
                 openDependencies = 0
                 lengthAccumulated = 0
                 for child in subjects:
                    lengthAccumulated += child[1] - child[2] 
                    lengthAccumulated += openDependencies * child[1]
                    openDependencies += 1
                 depsSubjAfter = lengthAccumulated + sum([x[1] for x in deps if x[3] == False])
                 assert depsSubjAfter > 0

                 if dlVerb == 0:
                   dlVerb_VS = dlVerb
                 elif dlVerb < 0:
                   dlVerb_VS = dlVerb - sum([x[1] for x in subjects])
                   assert dlVerb_VS < 0
                 elif dlVerb > 0:
                   dlVerb_VS = dlVerb - sum([x[1] for x in subjects])
                   assert dlVerb_VS > 0

                 L_VS = abs(dlVerb_VS) + depsBeforeWithoutSubj + depsAfter + depsSubjAfter


                 # TODO there is some bug in the calculation here, ordering SUBJECTS ascendingly or descendingly yields oddly different results, in unexpected direction
                 subjects = sorted([x for x in deps if not x[3] and x[0] != "nsubj"] + [x for x in deps if x[0] == "nsubj"], key=lambda x:x[1])
                 
                 openDependencies = 0
                 lengthAccumulated = 0
                 for child in subjects[::-1]:
                    lengthAccumulated += child[1] - child[2] 
                    lengthAccumulated += openDependencies * child[1]
                    openDependencies += 1
                 depsSubjAfter2 = lengthAccumulated + sum([x[1] for x in deps if x[3] == False])
                 assert depsSubjAfter2 > 0


                 # TODO there is some bug in the calculation here, ordering SUBJECTS ascendingly or descendingly yields oddly different results, in unexpected direction
                 subjects = sorted([x for x in deps if not x[3] and x[0] != "nsubj"] + [x for x in deps if x[0] == "nsubj"], key=lambda x:-x[1])
                 print(subjects)
                 openDependencies = 0
                 lengthAccumulated = 0
                 for child in subjects[::-1]:
                    lengthAccumulated += child[1] - child[2] 
                    lengthAccumulated += openDependencies * child[1]
                    openDependencies += 1
                 depsSubjAfter3 = lengthAccumulated + sum([x[1] for x in deps if x[3] == False])
                 assert depsSubjAfter3 > 0
                 print(depsSubjAfter2, depsSubjAfter3)
                 assert depsSubjAfter2 <= depsSubjAfter3


                 subjects = [x for x in deps if x[0] == "nsubj"] #, key=lambda x:x[1])


                 if dlVerb == 0:
                   dlVerb_VS = dlVerb
                 elif dlVerb < 0:
                   dlVerb_VS = dlVerb - sum([x[1] for x in subjects])
                   assert dlVerb_VS < 0, subjects
                 elif dlVerb > 0:
                   dlVerb_VS = dlVerb - sum([x[1] for x in subjects])
                   assert dlVerb_VS > 0, subjects

                 L_VS2 = abs(dlVerb_VS) + depsBeforeWithoutSubj + depsSubjAfter2




 
#                 print(dlVerb, depsBefore, depsAfter, depsBeforeWithoutSubj, depsBeforeShortest, L_Real, L_VS, L_SV_ideal, depsSubjAfter)
                 print(L_Real, L_VS2)
                 assert L_VS2 > 0
                 global result
                 result[0] += (L_Real - L_VS2)
                 result[1] += 1
                 print(result[0]/float(result[1]))

   return linearized, logits

result = [0,0]


dhLogits, vocab, vocab_deps, depsVocab = initializeOrderTable()

posUni = list(posUni)
itos_pos_uni = posUni
stoi_pos_uni = dict(zip(posUni, range(len(posUni))))

posFine = list(posFine)
itos_pos_ptb = posFine
stoi_pos_ptb = dict(zip(posFine, range(len(posFine))))



itos_pure_deps = sorted(list(depsVocab)) 
stoi_pure_deps = dict(zip(itos_pure_deps, range(len(itos_pure_deps))))
   

itos_deps = sorted(vocab_deps)
stoi_deps = dict(zip(itos_deps, range(len(itos_deps))))

print itos_deps

dhWeights = [0.0] * len(itos_deps)
distanceWeights = [0.0] * len(itos_deps)





words = list(vocab.iteritems())
words = sorted(words, key = lambda x:x[1], reverse=True)
itos = map(lambda x:x[0], words)
stoi = dict(zip(itos, range(len(itos))))
#print stoi
#print itos[5]
#print stoi[itos[5]]

if len(itos) > 6:
   assert stoi[itos[5]] == 5

#print dhLogits

#for sentence in getNextSentence():
#   print orderSentence(sentence, dhLogits)

vocab_size = 50000


batchSize = 1

lr_lm = 0.1


crossEntropy = 10.0

def encodeWord(w):
   return stoi[w]+3 if stoi[w] < vocab_size else 1

#loss = torch.nn.CrossEntropyLoss(reduce=False, ignore_index = 0)



import torch.cuda
import torch.nn.functional




counter = 0


lastDevLoss = None
failedDevRuns = 0
devLosses = [] 


def doForwardPass(current):
       global counter
       global crossEntropy
       global printHere
       global devLosses
       batchOrderedLogits = zip(*map(lambda (y,x):orderSentence(x, dhLogits, y==0 and printHere), zip(range(len(current)),current)))
      
       batchOrdered = batchOrderedLogits[0]
       logits = batchOrderedLogits[1]
   
       lengths = map(len, current)
       # current is already sorted by length
       maxLength = lengths[int(0.8*batchSize)]
       input_words = []
       input_pos_u = []
       input_pos_p = []
       for i in range(maxLength+2):
          input_words.append(map(lambda x: 2 if i == 0 else (encodeWord(x[i-1]["word"]) if i <= len(x) else 0), batchOrdered))
          input_pos_u.append(map(lambda x: 2 if i == 0 else (stoi_pos_uni[x[i-1]["posUni"]]+3 if i <= len(x) else 0), batchOrdered))
          input_pos_p.append(map(lambda x: 2 if i == 0 else (stoi_pos_ptb[x[i-1]["posFine"]]+3 if i <= len(x) else 0), batchOrdered))

       hidden = None #(Variable(torch.FloatTensor().new(2, batchSize, 128).zero_()), Variable(torch.FloatTensor().new(2, batchSize, 128).zero_()))
       loss = 0
       wordNum = 0
       lossWords = 0
       policyGradientLoss = 0
       baselineLoss = 0

       totalQuality = 0.0

       if True:
           totalDepLength = 0
           byType = []
           for i in range(1,len(input_words)): #range(1,maxLength+1): # don't include i==0
              for j in range(batchSize):
                 if input_words[i][j] != 0:
                    if batchOrdered[j][i-1]["head"] == 0:
                       realHead = 0
                    else:
                       realHead = batchOrdered[j][i-1]["reordered_head"] # this starts at 1, so just right for the purposes here
                    if batchOrdered[j][i-1]["coarse_dep"] == "root":
                       continue
                    depLength = abs(i - realHead) # - 1 # to be consistent with Richard's stuff
                    assert depLength >= 1
                    totalDepLength += depLength
                    byType.append((batchOrdered[j][i-1]["coarse_dep"], depLength))
                    #if input_words[i] > 2 and j == 0 and printHere:
                       #print [itos[input_words[i][j]-3], itos_pos_ptb[input_pos_p[i][j]-3] ]
                    wordNum += 1

       if wordNum > 0:
          crossEntropy = 0.99 * crossEntropy + 0.01 * (totalDepLength/wordNum)
       else:
          assert totalDepLength == 0
       numberOfWords = wordNum
       return (totalDepLength, numberOfWords, byType)



assert batchSize == 1

depLengths = []
#while True:
outpath = "/u/scr/mhahn/japanese/"+str(myID)
#with open(outpath, "w") as outFile:
if True:
 #print >> outFile, "\t".join(["Sent", "Length"])
 counter = 0
 if True:
   corpus = CorpusIterator_V(language,"train", shuffleDataSeed=40)
   corpusIterator = corpus.iterator()
   if corpus.length() == 0:
      quit()
   while True:
     try:
        batch = map(lambda x:next(corpusIterator), 10*range(batchSize))
     except StopIteration:
        break
     batch = sorted(batch, key=len)
     partitions = range(10)
     
     for partition in partitions:
        counter += 1
        printHere = (counter % 100 == 0)
        current = batch[partition*batchSize:(partition+1)*batchSize]
 
        depLength = doForwardPass(current)
        depLengths.append(depLength)
  #      print >> outFile, "\t".join(map(str, [counter, depLength[0]]))
        if counter % 100 == 0:
           print  "Average dep length "+str(sum(map(lambda x:x[0], depLengths))/float(len(depLengths)))+" "+str(counter)+" "+str(float(counter)/corpus.length())+"%"
 
 
print(outpath) 