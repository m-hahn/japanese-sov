#/u/nlp/bin/stake.py -g 11.5g -s run-stats-pretrain2.json "python readDataDistEnglishGPUFree.py"
# uses recursivelyLength
# orders children by lenght
# removes functional heads
# does not reorder the subject

import random
import sys

from collections import deque

#objectiveName = "gradients_WordSurp"

language = sys.argv[1]
model = "REAL_REAL"
temperature = "Infinity"

REMOVE_FUNCTIONAL_HEADS = True
REVERSE_SUBJECT = False
USE_FUNCHEAD_VERSIOM = True
SORT_CHILDREN_BY_LENGTH = True
REORDER_SUBJECT_INTERNAL = False
USE_V_VERSION = True
SORT_HD_SUBJECT_LAST = False
SORT_RECURSIVELY_BY_LENGTH = True

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

assert USE_V_VERSION
assert USE_FUNCHEAD_VERSIOM
from corpusIterator_FuncHead_V import CorpusIteratorFuncHead_V

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
     for sentence in CorpusIteratorFuncHead_V(language,partition, shuffleDataSeed=40).iterator():
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
def recursivelyLength(sentence, position, result, gradients_from_the_left_sum):
   line = sentence[position-1]
   allGradients = gradients_from_the_left_sum 
   line["length"] = 1
   # there are the gradients of its children
   if "children_DH" in line:

      for child in line["children_DH"]:
         assert child > 0
         if "removed" not in sentence[child-1]:
           length = recursivelyLength(sentence, child, result, allGradients)
           line["length"] += length


   if "children_HD" in line:
 #     deps = [sentence[x-1]["coarse_dep"] for x in line["children_HD"]]
  #    if "nsubj" in deps and len(set(deps)) > 1:
   #      print(deps)

    #  line["children_HD"] = sorted(line["children_HD"], key=lambda x:0 if sentence[x-1]["coarse_dep"] != "nsubj" else 1)

      for child in line["children_HD"]:
         assert child > 0
         if "removed" not in sentence[child-1]:
           length = recursivelyLength(sentence, child, result, allGradients)
           line["length"] += length



   return line["length"]



# "linearization_logprobability"
def recursivelyLinearize(sentence, position, result, gradients_from_the_left_sum):
   line = sentence[position-1]
   # Loop Invariant: these are the gradients relevant at everything starting at the left end of the domain of the current element
   allGradients = gradients_from_the_left_sum 
   line["length"] = 1
   # there are the gradients of its children
   if "children_DH" in line:
      assert SORT_CHILDREN_BY_LENGTH
      line["children_DH"] = sorted(line["children_DH"], key=lambda x:-sentence[x-1]["length"])

      for child in line["children_DH"]:
         assert child > 0
         if "removed" not in sentence[child-1]:
           length = recursivelyLinearize(sentence, child, result, allGradients)
           line["length"] += length

   result.append(line)

   if "children_HD" in line:
 #     deps = [sentence[x-1]["coarse_dep"] for x in line["children_HD"]]
  #    if "nsubj" in deps and len(set(deps)) > 1:
   #      print(deps)

    #  line["children_HD"] = sorted(line["children_HD"], key=lambda x:0 if sentence[x-1]["coarse_dep"] != "nsubj" else 1)
      assert SORT_CHILDREN_BY_LENGTH
      assert not SORT_HD_SUBJECT_LAST
      line["children_HD"] = sorted(line["children_HD"], key=lambda x:sentence[x-1]["length"])

      for child in line["children_HD"]:
         assert child > 0
         if "removed" not in sentence[child-1]:
           length = recursivelyLinearize(sentence, child, result, allGradients)
           line["length"] += length



   return line["length"]

import numpy.random

softmax_layer = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()






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

      assert REMOVE_FUNCTIONAL_HEADS
      if line["coarse_dep"] in ["aux", "mark", "case", "neg", "cc"]: #.startswith("punct"): # assumes that punctuation does not have non-punctuation dependents!
         if model == "REAL_REAL":
            eliminated.append(line)
         continue



      key = line["coarse_dep"]
      line["dependency_key"] = key
      dhLogit = dhWeights[stoi_deps[key]]
      dhSampled = (line["index"] < line["head"])

      
     
      direction = "DH" if dhSampled else "HD"
      assert not REVERSE_SUBJECT

#      if printThings: 
 #        print "\t".join(map(str,["ORD", line["index"], (line["word"]+"           ")[:10], ("".join(list(key)) + "         ")[:22], line["head"], dhSampled, direction, str(1/(1+exp(-dhLogits[key])))[:8], (str(distanceWeights[stoi_deps[key]])+"    ")[:8] , str(originalDistanceWeights[key])[:8]    ]  ))

      headIndex = line["head"]-1
      sentence[headIndex]["children"] = (sentence[headIndex].get("children", []) + [line["index"]])
      sentence[headIndex]["children_"+direction] = (sentence[headIndex].get("children_"+direction, []) + [line["index"]])

      head = sentence[headIndex]

   
   if model != "REAL_REAL":
      assert False
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

   assert SORT_RECURSIVELY_BY_LENGTH
   recursivelyLength(sentence, root, None, 0)
  
   linearized = []

   recursivelyLinearize(sentence, root, linearized, 0)

   if sentence[0]["word"] != "_":
     #print " ".join(map(lambda x:x["word"], linearized))
     for line in sentence:        
        if "children" in line:
           if line["posUni"] == "VERB":
     #         print(line["children"])
    #          print(line["index"] - line["head"], line["coarse_dep"])
              assert line["coarse_dep"] == "root" or (line["index"] - line["head"]) < 0

   #           print([(lambda y: (y["coarse_dep"], y["length"], y["index"] < line["index"]))(sentence[x-1]) for x in line["children"] if "removed" not in sentence[x-1]])
               

   if model == "REAL_REAL":
      linearizedByRule = linearized
      linearized = filter(lambda x:"removed" not in x, sentence)
   if False: # for checking correctness in the absence of reordering (e.g. placing nsubj somewhere else)
#   if True or printThings or len(linearized) == 0:
   #  print "========================"
   #  print " ".join(map(lambda x:x["word"], sentence))
     LBR = " ".join(map(lambda x:x["word"], linearizedByRule))
     LR = " ".join(map(lambda x:x["word"], linearized))
   #  print LBR
   #  print LR
     LBR2 = " ".join(map(lambda x:x["posUni"], linearizedByRule))
     LR2 = " ".join(map(lambda x:x["posUni"], linearized)) 
   #  print LBR2
   #  print LR2
     if LBR2 != LR2:
        print "WARNING!!!"

     assert LBR == LR
     assert len(linearized) == len(linearizedByRule)
   linearized = linearizedByRule



   # store new dependency links
   moved = [None] * len(sentence)
   for i, x in enumerate(linearized):
#      print x
      moved[x["index"]-1] = i
 #  print moved
   for i,x in enumerate(linearized):
  #    print x
      if x["head"] == 0: # root
         x["reordered_head"] = 0
      else:
         x["reordered_head"] = 1+moved[x["head"]-1]
   return linearized, logits


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
with open(outpath, "w") as outFile:
 print >> outFile, "\t".join(["Sent", "Length"])
 counter = 0
 if True:
   corpus = CorpusIteratorFuncHead_V(language,"train", shuffleDataSeed=40)
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
        print >> outFile, "\t".join(map(str, [counter, depLength[0]]))
        if counter % 100 == 0:
           print  "Average dep length "+str(sum(map(lambda x:x[0], depLengths))/float(len(depLengths)))+" "+str(counter)+" "+str(float(counter)/corpus.length())+"%"
 
 
print(outpath) 
