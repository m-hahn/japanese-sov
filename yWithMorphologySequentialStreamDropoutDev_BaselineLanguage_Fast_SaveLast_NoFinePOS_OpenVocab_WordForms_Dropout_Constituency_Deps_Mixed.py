# Baseline respects constituency structures, and is parameterized w.r.t. dependencies (should be a stroger baseline than the dependecy-only and constituency-only baselines)


# /u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7 yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_OpenVocab_WordForms_Dropout_Constituency_Deps.py PTB PTB 0.1 50 128 2 0.1 RANDOM_BY_TYPE 0.2 4 0.01 20 


#/u/nlp/bin/stake.py -g 11.5g -s run-stats-pretrain2.json "python readDataDistEnglishGPUFree.py"

#Created for confirmatory experiments: full word forms, open vocabulary, word noising, universal POS


# TODO also try other optimizers

import random
import sys

objectiveName = "LM"

language = sys.argv[1]
languageCode = sys.argv[2]
dropout_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 0.33
emb_dim = int(sys.argv[4]) if len(sys.argv) > 4 else 100
rnn_dim = int(sys.argv[5]) if len(sys.argv) > 5 else 512
rnn_layers = int(sys.argv[6]) if len(sys.argv) > 6 else 2
lr_lm = float(sys.argv[7]) if len(sys.argv) > 7 else 0.1
model = sys.argv[8]
input_dropoutRate = float(sys.argv[9]) # 0.33
batchSize = int(sys.argv[10])
replaceWordsProbability = float(sys.argv[11])
horizon = int(sys.argv[12]) if len(sys.argv) > 12 else 20
prescripedID = sys.argv[13] if len(sys.argv)> 13 else None
gpuNumber = sys.argv[14] if len(sys.argv) > 14 else "GPU0"
assert gpuNumber.startswith("GPU")
gpuNumber = int(gpuNumber[3:])

#if len(sys.argv) == 13:
#  del sys.argv[12]
assert len(sys.argv) in [12,13,14, 15]


assert dropout_rate <= 0.5
assert input_dropoutRate <= 0.5

devSurprisalTable = [None] * horizon
if prescripedID is not None:
  myID = int(prescripedID)
else:
  myID = random.randint(0,10000000)


TARGET_DIR = "/u/scr/mhahn/deps/memory-need-neural-wordforms/"

#with open("/juicier/scr120/scr/mhahn/deps/LOG"+language+"_"+__file__+"_model_"+str(myID)+".txt", "w") as outFile:
#    print >> outFile, " ".join(sys.argv)



posUni = set() #[ "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"] 

posFine = set() #[ "``", ",", ":", ".", "''", "$", "ADD", "AFX", "CC",  "CD", "DT", "EX", "FW", "GW", "HYPH", "IN", "JJ", "JJR",  "JJS", "-LRB-", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS",  "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "-RRB-", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  "WDT", "WP", "WP$", "WRB", "XX" ]




#deps = ["acl", " advcl", " advmod", " amod", " appos", " aux", " case cc", " ccompclf", " compound", " conj", " cop", " csubjdep", " det", " discourse", " dislocated", " expl", " fixed", " flat", " goeswith", " iobj", " list", " mark", " nmod", " nsubj", " nummod", " obj", " obl", " orphan", " parataxis", " punct", " reparandum", " root", " vocative", " xcomp"]


from math import log, exp
from random import random, shuffle, randint


from corpusIterator_PTB_Deps import CorpusIterator_PTB

originalDistanceWeights = {}

morphKeyValuePairs = set()

vocab_lemmas = {}

import nltk.tree

corpus_cached = {}
corpus_cached["train"] = CorpusIterator_PTB("PTB", "train")
corpus_cached["dev"] = CorpusIterator_PTB("PTB", "dev")


def descendTree(tree, vocab, posFine, depsVocab):
   label = tree.label()
   for child in tree:
      if type(child) == nltk.tree.Tree:
   #     print((label, child.label()), type(tree))
        key = (label, child.label())
        depsVocab.add(key)
        descendTree(child, vocab, posFine, depsVocab)
      else:
        posFine.add(label)
        word = child.lower()
        if "*-" in word:
           continue
        vocab[word] = vocab.get(word, 0) + 1
    #    print(child)
def initializeOrderTable():
   orderTable = {}
   vocab = {}
   distanceSum = {}
   distanceCounts = {}
   depsVocab = set()
   for partition in ["train", "dev"]:
     for sentenceAndTree in corpus_cached[partition].iterator():
      _, sentence = sentenceAndTree
      #descendTree(sentence, vocab, posFine, depsVocab)

      for line in sentence:
          vocab[line["word"]] = vocab.get(line["word"], 0) + 1
          posFine.add(line["posFine"])
          depsVocab.add(line["dep"])
   return vocab, depsVocab

#import torch.distributions
import torch.nn as nn
import torch
from torch.autograd import Variable

import numpy.random

softmax_layer = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()




def orderSentenceRec(tree, sentence, printThings, linearized, order="mixed"):
   label = tree.label()
#   print("TREE", tree)
   children = [child for child in tree]
 #  print("CHILDREN", children)
   if type(children[0]) != nltk.tree.Tree:
      assert all([type(x) != nltk.tree.Tree for x in children])
      #print(children)
      for c in children:
#        print((label, label in ["'", ":", "``", ",", "''", "#", ".", "-NONE-"] or label[0] == "-" or c.startswith("*-")))
        if label in ["'", ":", "``", ",", "''", "#", ".", "-NONE-"] or label[0] == "-" or "*-" in c:
           continue
        word = sentence[tree.start]["word"] #c.lower(), )
        if word != c.lower().replace("\/","/"):
           print(142, word, c.lower())
        linearized.append({"word" : word, "posFine" : label})
   else:
      assert all([type(x) == nltk.tree.Tree for x in children])
      children = [child for child in children if child.start < child.end] # remove children that consist of gaps or otherwise eliminated tokens

      # find those 

     # 
     # if len(tree.incoming) > 1:
     #    print("INCOMING", [sentence[x]["dep"] for _, x in tree.incoming])


      # find which children seem to be dependents of which other children
      if model != "REAL_REAL": 
        childDeps = [None for _ in children]
        for i in range(len(children)):
           incomingFromOutside = [x for x in tree.incoming if x in children[i].incoming]
           if len(incomingFromOutside) > 0:
              childDeps[i] = sentence[incomingFromOutside[-1][1]]["dep"]
              if len(incomingFromOutside) > 1:
                  print("FROM OUTSIDE", [sentence[incomingFromOutside[x][1]]["dep"] for x in range(len(incomingFromOutside))])
           for j in range(len(children)):
              if i == j:
                 continue
              incomingFromJ = [x for x in children[i].incoming if x in children[j].outgoing]
              if len(incomingFromJ) > 0:
                 if len(incomingFromJ) > 1:
                    duplicateDeps = tuple([sentence[incomingFromJ[x][1]]["dep"] for x in range(len(incomingFromJ))])
                    if not (duplicateDeps == ("obj", "xcomp")):
                       print("INCOMING FROM NEIGHBOR", duplicateDeps)
                 childDeps[i] = sentence[incomingFromJ[-1][1]]["dep"]
        assert None not in childDeps, (childDeps, children)
  
        keys = childDeps
  
        logits = [(x, distanceWeights[stoi_deps[key]]) for x, key in zip(children, keys)]
        logits = sorted(logits, key=lambda x:-x[1])
        childrenLinearized = map(lambda x:x[0], logits)
      else:
        childDeps = [None for _ in children]
        for i in range(len(children)):
           incomingFromOutside = [x for x in tree.incoming if x in children[i].incoming]
           if len(incomingFromOutside) > 0:
              childDeps[i] = sentence[incomingFromOutside[-1][1]]["dep"]
              if len(incomingFromOutside) > 1:
                  print("FROM OUTSIDE", [sentence[incomingFromOutside[x][1]]["dep"] for x in range(len(incomingFromOutside))])
           for j in range(len(children)):
              if i == j:
                 continue
              incomingFromJ = [x for x in children[i].incoming if x in children[j].outgoing]
              if len(incomingFromJ) > 0:
                 if len(incomingFromJ) > 1:
                    duplicateDeps = tuple([sentence[incomingFromJ[x][1]]["dep"] for x in range(len(incomingFromJ))])
                    if not (duplicateDeps == ("obj", "xcomp")):
                       print("INCOMING FROM NEIGHBOR", duplicateDeps)
                 childDeps[i] = sentence[incomingFromJ[-1][1]]["dep"]
        assert None not in childDeps, (childDeps, children)
  
        keys = childDeps
        childrenLinearized = children
        REVERSE_SUBJECT = (order == "VS" or (order == "mixed" and random() > 0.5))
#        print(order, REVERSE_SUBJECT)

        if REVERSE_SUBJECT:
         if "nsubj" in childDeps and len(childDeps) > 1:
           labels = [x.label() for x in children]
           if "NP-SBJ" in str(labels):
              hasReversed = False
              for i in range(len(children)-1):
                 if labels[i].startswith("NP-SBJ") and labels[i+1].startswith("VP"):
                    childrenLinearized[i], childrenLinearized[i+1] = childrenLinearized[i+1], childrenLinearized[i]
                    labels[i], labels[i+1] = labels[i+1], labels[i]
                    hasReversed=True
                 elif labels[i].startswith("NP-SBJ") and labels[i+1].startswith("NP-PRD"):
                    childrenLinearized[i], childrenLinearized[i+1] = childrenLinearized[i+1], childrenLinearized[i]
                    labels[i], labels[i+1] = labels[i+1], labels[i]
                    hasReversed=True
                 elif labels[i].startswith("NP-SBJ") and labels[i+1].startswith("ADJP-PRD"):
                    childrenLinearized[i], childrenLinearized[i+1] = childrenLinearized[i+1], childrenLinearized[i]
                    labels[i], labels[i+1] = labels[i+1], labels[i]
                    hasReversed=True
                 elif i < len(children)-2 and labels[i].startswith("NP-SBJ") and labels[i+2].startswith("VP"):
                    childrenLinearized[i], childrenLinearized[i+1], childrenLinearized[i+2] = childrenLinearized[i+1], childrenLinearized[i+2], childrenLinearized[i]
                    labels[i], labels[i+1], labels[i+2] = labels[i+1], labels[i+2], labels[i]
                    hasReversed=True
                 elif i < len(children)-3 and labels[i].startswith("NP-SBJ") and labels[i+3].startswith("VP"):
                    childrenLinearized[i], childrenLinearized[i+1], childrenLinearized[i+2], childrenLinearized[i+3] = childrenLinearized[i+1], childrenLinearized[i+2], childrenLinearized[i+3], childrenLinearized[i]
                    labels[i], labels[i+1], labels[i+2], labels[i+3] = labels[i+1], labels[i+2], labels[i+3], labels[i]
                    hasReversed=True
                 elif i < len(children)-4 and labels[i].startswith("NP-SBJ") and labels[i+4].startswith("VP"):
                    childrenLinearized[i], childrenLinearized[i+1], childrenLinearized[i+2], childrenLinearized[i+3], childrenLinearized[i+4] = childrenLinearized[i+1], childrenLinearized[i+2], childrenLinearized[i+3], childrenLinearized[i+4], childrenLinearized[i]
                    labels[i], labels[i+1], labels[i+2], labels[i+3], labels[i+4] = labels[i+1], labels[i+2], labels[i+3], labels[i+4], labels[i]
                    hasReversed=True
                 elif i < len(children)-5 and labels[i].startswith("NP-SBJ") and labels[i+4].startswith("VP"):
                    childrenLinearized[i], childrenLinearized[i+1], childrenLinearized[i+2], childrenLinearized[i+3], childrenLinearized[i+4], childrenLinearized[i+5] = childrenLinearized[i+1], childrenLinearized[i+2], childrenLinearized[i+3], childrenLinearized[i+4], childrenLinearized[i+5], childrenLinearized[i]
                    labels[i], labels[i+1], labels[i+2], labels[i+3], labels[i+4], labels[i+5] = labels[i+1], labels[i+2], labels[i+3], labels[i+4], labels[i+5], labels[i]
                    hasReversed=True



              if not hasReversed and not "VP NP-SBJ" in " ".join(labels) and not "VBZ NP-SBJ" in " ".join(labels) and not "VB NP-SBJ" in " ".join(labels):
                 print((childDeps, [x.incoming for x in children], [x.outgoing for x in children], label, [x.label() for x in children]))
   
  #      logits = [(x, distanceWeights[stoi_deps[key]]) for x, key in zip(children, keys)]
 #       logits = sorted(logits, key=lambda x:-x[1])
#        childrenLinearized = map(lambda x:x[0], logits)

#      print(logits)
   
      for child in childrenLinearized:
#        if type(child) == nltk.tree.Tree:
          orderSentenceRec(child, sentence, printThings, linearized, order=order)


def numberSpans(tree, start, sentence):
   if type(tree) != nltk.tree.Tree:
      if tree.startswith("*") or tree == "0":
        return start, ([]), ([])
      else:
        #print("CHILDREN", start, sentence[start].get("children", []))
        outgoing = ([(start, x) for x in sentence[start].get("children", [])])
        #if len(sentence[start].get("children", [])) > 0:
           #print("OUTGOING", outgoing)
           #assert len(outgoing) > 0
#        if sentence[start]["head"] == 0:
#             print("ROOT", start)
        return start+1, ([(sentence[start]["head"]-1, start)]), outgoing
   else:
      tree.start = start
      incoming = ([])
      outgoing = ([])
      for child in tree:
        start, incomingC, outgoingC = numberSpans(child, start, sentence)
        incoming +=  incomingC
        outgoing += outgoingC
      tree.end = start
      #print(incoming, outgoing, tree.start, tree.end)
   #   print(tree.start, tree.end, incoming, [(hd,dep) for hd, dep in incoming if hd < tree.start or hd>= tree.end])
      incoming = ([(hd,dep) for hd, dep in incoming if hd < tree.start or hd>= tree.end])
      outgoing = ([(hd,dep) for hd, dep in outgoing if dep < tree.start or dep>= tree.end])

      tree.incoming = incoming
      tree.outgoing = outgoing
      #print(incoming, outgoing)
      return start, incoming, outgoing

import copy

def orderSentence(tree, printThings, order="mixed"):
   global model
   linearized = []
   tree, sentence = tree
   #tree = copy.deepcopy(tree)
   for i in range(len(sentence)):
      line = sentence[i]
      if line["dep"] == "root":
         continue
      head = line["head"] - 1
      if "children" not in sentence[head]:
        sentence[head]["children"] = []
      sentence[head]["children"].append(i)


   end, incoming, outgoing = numberSpans(tree, 0, sentence)
   assert len(incoming) == 1, incoming
   assert len(outgoing) == 0, outgoing
   if (end != len(sentence)):
      print(tree.leaves())
      print([x["word"] for x in sentence])
   orderSentenceRec(tree, sentence, printThings, linearized, order=order)
   #if printThings:
   #  print("linearized", linearized)
   #for word in linearized:
   #   assert "*-" not in word["word"], word
   return linearized

vocab, depsVocab = initializeOrderTable()


posFine = list(posFine)
itos_pos_fine = posFine
stoi_pos_fine = dict(zip(posFine, range(len(posFine))))



itos_pure_deps = sorted(list(depsVocab)) 
stoi_pure_deps = dict(zip(itos_pure_deps, range(len(itos_pure_deps))))
   

itos_deps = itos_pure_deps
stoi_deps = stoi_pure_deps

#print itos_deps

#dhWeights = [0.0] * len(itos_deps)
distanceWeights = [0.0] * len(itos_deps)

#dhWeights = Variable(torch.FloatTensor([0.0] * len(itos_deps)), requires_grad=True)
#distanceWeights = Variable(torch.FloatTensor([0.0] * len(itos_deps)), requires_grad=True)
#for i, key in enumerate(itos_deps):
#
#   # take from treebank, or randomize
#   dhLogits[key] = 2*(random()-0.5)
#   dhWeights.data[i] = dhLogits[key]
#
#   originalDistanceWeights[key] = random()  
#   distanceWeights.data[i] = originalDistanceWeights[key]

import os

#if model != "RANDOM_MODEL" and model != "REAL" and model != "RANDOM_BY_TYPE":
#   inpModels_path = "/juicier/scr120/scr/mhahn/deps/"+"/manual_output/"
#   models = os.listdir(inpModels_path)
#   models = filter(lambda x:"_"+model+".tsv" in x, models)
#   if len(models) == 0:
#     assert False, "No model exists"
#   if len(models) > 1:
#     assert False, [models, "Multiple models exist"]
#   
#   with open(inpModels_path+models[0], "r") as inFile:
#      data = map(lambda x:x.split("\t"), inFile.read().strip().split("\n"))
#      header = data[0]
#      data = data[1:]
#    
#   for line in data:
#      head = line[header.index("Head")]
#      dependent = line[header.index("Dependent")]
#      dependency = line[header.index("Dependency")]
#      key = (head, dependency, dependent)
#      dhWeights[stoi_deps[key]] = float(line[header.index("DH_Weight")].replace("[", "").replace("]",""))
#      distanceWeights[stoi_deps[key]] = float(line[header.index("DistanceWeight")].replace("[", "").replace("]",""))
#      originalCounter = int(line[header.index("Counter")])
if model == "RANDOM_MODEL":
  for key in range(len(itos_deps)):
     #dhWeights[key] = random() - 0.5
     distanceWeights[key] = random()
  originalCounter = "NA"
elif model == "REAL" or model == "REAL_REAL":
  originalCounter = "NA"
elif model == "RANDOM_BY_TYPE":
  #dhByType = {}
  distByType = {}
  for dep in itos_pure_deps:
 #   dhByType[dep] = random() - 0.5
    distByType[dep] = random()
  for key in range(len(itos_deps)):
#     dhWeights[key] = dhByType[itos_deps[key]]
     distanceWeights[key] = distByType[itos_deps[key]]
  originalCounter = "NA"

lemmas = list(vocab_lemmas.iteritems())
lemmas = sorted(lemmas, key = lambda x:x[1], reverse=True)
#itos_lemmas = map(lambda x:x[0], lemmas)
#stoi_lemmas = dict(zip(itos_lemmas, range(len(itos_lemmas))))

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

vocab_size = 10000
vocab_size = min(len(itos),vocab_size)
#print itos[:vocab_size]
#quit()

# 0 EOS, 1 UNK, 2 BOS
#word_embeddings = torch.nn.Embedding(num_embeddings = vocab_size+3, embedding_dim = emb_dim).cuda()
#pos_u_embeddings = torch.nn.Embedding(num_embeddings = len(posUni)+3, embedding_dim = 10).cuda()
#pos_p_embeddings = torch.nn.Embedding(num_embeddings = len(posFine)+3, embedding_dim=10).cuda()
#morph_embeddings = torch.nn.Embedding(num_embeddings = len(morphKeyValuePairs)+3, embedding_dim=100).cuda()

word_pos_morph_embeddings = torch.nn.Embedding(num_embeddings = len(posFine)+vocab_size+3, embedding_dim=emb_dim).cuda()
print posFine
#print posFine
print morphKeyValuePairs
print itos[:vocab_size]
print "VOCABULARY "+str(len(posFine)+vocab_size+3)
outVocabSize = len(posFine)+vocab_size+3
#assert len(posFine)+vocab_size+3 < 200
#quit()


itos_total = ["EOS", "EOW", "SOS"] + itos_pos_fine + itos[:vocab_size]
assert len(itos_total) == outVocabSize
# could also provide per-word subcategorization frames from the treebank as input???


#baseline = nn.Linear(emb_dim, 1).cuda()

dropout = nn.Dropout(dropout_rate).cuda()

rnn = nn.LSTM(emb_dim, rnn_dim, rnn_layers).cuda()
for name, param in rnn.named_parameters():
  if 'bias' in name:
     nn.init.constant(param, 0.0)
  elif 'weight' in name:
     nn.init.xavier_normal(param)

decoder = nn.Linear(rnn_dim,outVocabSize).cuda()
#pos_fine_decoder = nn.Linear(128,len(posFine)+3).cuda()

components = [rnn, decoder, word_pos_morph_embeddings]
# word_embeddings, pos_u_embeddings, pos_p_embeddings, 
#baseline, 

def parameters():
 for c in components:
   for param in c.parameters():
      yield param
# yield dhWeights
# yield distanceWeights

#for pa in parameters():
#  print pa

initrange = 0.1
#word_embeddings.weight.data.uniform_(-initrange, initrange)
#pos_u_embeddings.weight.data.uniform_(-initrange, initrange)
#pos_p_embeddings.weight.data.uniform_(-initrange, initrange)
#morph_embeddings.weight.data.uniform_(-initrange, initrange)
word_pos_morph_embeddings.weight.data.uniform_(-initrange, initrange)

decoder.bias.data.fill_(0)
decoder.weight.data.uniform_(-initrange, initrange)
#pos_fine_decoder.bias.data.fill_(0)
#pos_fine_decoder.weight.data.uniform_(-initrange, initrange)
#baseline.bias.data.fill_(0)
#baseline.weight.data.uniform_(-initrange, initrange)




crossEntropy = 10.0

#def encodeWord(w):
#   return stoi[w]+3 if stoi[w] < vocab_size else 1

#loss = torch.nn.CrossEntropyLoss(reduce=False, ignore_index = 0)



import torch.cuda
import torch.nn.functional

inputDropout = torch.nn.Dropout2d(p=input_dropoutRate)


counter = 0


lastDevLoss = None
failedDevRuns = 0
devLosses = [] 


lossModule = nn.NLLLoss()
lossModuleTest = nn.NLLLoss(size_average=False, reduce=False, ignore_index=2)

def doForwardPass(input_indices, wordStartIndices, surprisalTable=None, doDropout=True, batchSizeHere=1):
       global counter
       global crossEntropy
       global printHere
       global devLosses
       if printHere:
           print "wordStartIndices"
           print wordStartIndices

       hidden = None #(Variable(torch.FloatTensor().new(2, batchSizeHere, 128).zero_()), Variable(torch.FloatTensor().new(2, batchSizeHere, 128).zero_()))
       loss = 0
       wordNum = 0
       lossWords = 0
       policyGradientLoss = 0
       baselineLoss = 0
       for c in components:
          c.zero_grad()

       totalQuality = 0.0

       if True:
           
           sequenceLength = max(map(len, input_indices))
           for i in range(batchSizeHere):
              input_indices[i] = input_indices[i][:]
              while len(input_indices[i]) < sequenceLength:
                 input_indices[i].append(2)

           inputTensor = Variable(torch.LongTensor(input_indices).transpose(0,1).contiguous()).cuda() # so it will be sequence_length x batchSizeHere
#           print inputTensor
#           quit()

           inputTensorIn = inputTensor[:-1]
           inputTensorOut = inputTensor[1:]

           inputEmbeddings = word_pos_morph_embeddings(inputTensorIn.view(sequenceLength-1, batchSizeHere))
           if doDropout:
              inputEmbeddings = inputDropout(inputEmbeddings)
              if dropout_rate > 0:
                 inputEmbeddings = dropout(inputEmbeddings)
           output, hidden = rnn(inputEmbeddings, hidden)

           # word logits
           if doDropout:
              output = dropout(output)
           word_logits = decoder(output)
           word_logits = word_logits.view((sequenceLength-1)*batchSizeHere, outVocabSize)
           word_softmax = logsoftmax(word_logits)
#           word_softmax = word_softmax.view(-1, batchSizeHere, outVocabSize)

#           lossesWord = [[0]*batchSizeHere for i in range(len(input_indices))]

#           print inputTensorOut.view((sequenceLength-1)*batchSizeHere)
#           quit()
           lossesWord = lossModuleTest(word_softmax, inputTensorOut.view((sequenceLength-1)*batchSizeHere))
           numberOfWordsInvolved = len(lossesWord.view(-1))
           loss = lossesWord.sum()

#           print lossesWord

           if surprisalTable is not None or printHere:           
             lossesCPU = lossesWord.data.cpu().view((sequenceLength-1), batchSizeHere).numpy()
             if printHere:
                for i in range(0,len(input_indices[0])-1): #range(1,maxLength+1): # don't include i==0
#                   for j in range(batchSizeHere):
                         j = 0
                         print (i, itos_total[input_indices[j][i+1]], lossesCPU[i][j])
#                         if "*-" in itos_total[input_indices[j][i+1]]:
 #                            assert False, input_indices[j][i+1]

             if surprisalTable is not None: 
                if printHere:
                   print surprisalTable
                for j in range(batchSizeHere):
                  for r in range(horizon):
                    assert wordStartIndices[j][r]< wordStartIndices[j][r+1]
                    assert wordStartIndices[j][r] < len(lossesWord)+1, (wordStartIndices[j][r],wordStartIndices[j][r+1], len(lossesWord))
                    assert input_indices[j][wordStartIndices[j][r+1]-1] != 2
                    if r == horizon-1:
                      assert wordStartIndices[j][r+1] == len(input_indices[j]) or input_indices[j][wordStartIndices[j][r+1]] == 2
#                    print lossesCPU[wordStartIndices[j][r]:wordStartIndices[j][r+1],j]
 #                   surprisalTable[r] += sum([x.mean() for x in lossesCPU[wordStartIndices[j][r]:wordStartIndices[j][r+1],j]]) #.data.cpu().numpy()[0]
                    surprisalTable[r] += sum(lossesCPU[wordStartIndices[j][r]-1:wordStartIndices[j][r+1]-1,j]) #.data.cpu().numpy()[0]
                   

           wordNum = (len(wordStartIndices[0]) - 1)*batchSizeHere
           assert len(wordStartIndices[0]) == horizon+1, map(len, wordStartIndices)
                    
       if wordNum == 0:
         print input_words
         print batchOrdered
         return 0,0,0,0,0
       if printHere:
         print loss/wordNum
         print lossWords/wordNum
         print ["CROSS ENTROPY", crossEntropy, exp(crossEntropy)]
       crossEntropy = 0.99 * crossEntropy + 0.01 * (loss/wordNum).data.cpu().numpy()
       totalQuality = loss.data.cpu().numpy() # consists of lossesWord + lossesPOS
       numberOfWords = wordNum
#       probabilities = torch.sigmoid(dhWeights)
#       neg_entropy = torch.sum( probabilities * torch.log(probabilities) + (1-probabilities) * torch.log(1-probabilities))

#       policy_related_loss = lr_policy * (entropy_weight * neg_entropy + policyGradientLoss) # lives on CPU
       return loss/numberOfWordsInvolved, None, None, totalQuality, numberOfWords


parameterList = list(parameters())

def  doBackwardPass(loss, baselineLoss, policy_related_loss):
       global lastDevLoss
       global failedDevRuns
       loss.backward()
       if printHere:
         print "BACKWARD 3 "+__file__+" "+language+" "+str(myID)+" "+str(counter)+" "+str(lastDevLoss)+" "+str(failedDevRuns)+"  "+(" ".join(map(str,["Dropout (real)", dropout_rate, "Emb_dim", emb_dim, "rnn_dim", rnn_dim, "rnn_layers", rnn_layers, "MODEL", model])))
         print devLosses
#       torch.nn.utils.clip_grad_norm(parameterList, 5.0, norm_type='inf')
       for param in parameters():
         if param.grad is None:
           print "WARNING: None gradient"
           continue
         param.data.sub_(lr_lm * param.grad.data)



def createStream(corpus):
#    global counter
    global crossEntropy
    global printHere
    global devLosses

    input_indices = [2] # Start of Segment
    wordStartIndices = []
#    sentenceStartIndices = []
    sentCount = 0
    for sentence in corpus:
       sentCount += 1
       #printHere = (sentCount % 10 == 0)
       ordered = orderSentence(sentence,  printHere)
#       sentenceStartIndices.append(len(input_indices))
       for line in ordered+["EOS"]:
          wordStartIndices.append(len(input_indices))
          if line == "EOS":
            input_indices.append(0)
          else:
            
#            if len(itos_pos_fine) > 1:
#               input_indices.append(stoi_pos_fine[line["posFine"]]+3+len(stoi_pos_fine))
            if random() < replaceWordsProbability:
                targetWord = randint(0,vocab_size-1)
# len(posFine)+vocab_size+3
            else:
#                assert "*-" not in line["word"], line
                targetWord = stoi[line["word"]]
            if targetWord >= vocab_size:
               #print(stoi_pos_fine)
               input_indices.append(stoi_pos_fine[line["posFine"]]+3)
            else:
               input_indices.append(targetWord+3+len(stoi_pos_fine))
          if len(wordStartIndices) == horizon:
             yield input_indices, wordStartIndices+[len(input_indices)]
             input_indices = [2] # Start of Segment (makes sure that first word can be predicted from this token)
             wordStartIndices = []



def createStreamContinuous(corpus, order="mixed"):
#    global counter
    global crossEntropy
    global printHere
    global devLosses

    input_indices = [2] # Start of Segment
    wordStartIndices = []
#    sentenceStartIndices = []
    sentCount = 0
    for sentence in corpus:
       sentCount += 1
       if sentCount % 10 == 0:
         print ["DEV SENTENCES", sentCount]

#       if sentCount == 100:
       #printHere = (sentCount % 10 == 0)
       ordered = orderSentence(sentence, printHere, order=order)

#       sentenceStartIndices.append(len(input_indices))
       for line in ordered+["EOS"]:
          wordStartIndices.append(len(input_indices))
          if line == "EOS":
            input_indices.append(0)
          else:
#            if len(itos_pos_fine) > 1:
#               input_indices.append(stoi_pos_fine[line["posFine"]]+3+len(stoi_pos_fine))
            targetWord = stoi[line["word"]]
            if targetWord >= vocab_size:
               input_indices.append(stoi_pos_fine[line["posFine"]]+3)
            else:
               input_indices.append(targetWord+3+len(stoi_pos_fine))
          if len(wordStartIndices) == horizon:
#             print input_indices
#             print wordStartIndices+[len(input_indices)]
             yield input_indices, wordStartIndices+[len(input_indices)]
             if False:
               input_indices = [2] # Start of Segment (makes sure that first word can be predicted from this token)
               wordStartIndices = []
             else:
               input_indices = [2]+input_indices[wordStartIndices[1]:] # Start of Segment (makes sure that first word can be predicted from this token)
               wordStartIndices = [x-wordStartIndices[1]+1 for x in wordStartIndices[1:]]
               assert wordStartIndices[0] == 1




def computeDevLoss(order="mixed"):
   devBatchSize = 512
   global printHere
#   global counter
#   global devSurprisalTable
   global horizon
   devLoss = 0.0
   devWords = 0
#   corpusDev = getNextSentence("dev")
   corpusDev = corpus_cached["dev"].iterator()
   stream = createStreamContinuous(corpusDev, order=order)

   surprisalTable = [0 for _ in range(horizon)]
   devCounter = 0
   devCounterTimesBatchSize = 0
   while True:
#     try:
#        input_indices, wordStartIndices = next(stream)
     try:
        input_indices_list = []
        wordStartIndices_list = []
        for _ in range(devBatchSize):
           input_indices, wordStartIndices = next(stream)
           input_indices_list.append(input_indices)
           wordStartIndices_list.append(wordStartIndices)
     except StopIteration:
        devBatchSize = len(input_indices_list)
#        break
     if devBatchSize == 0:
       break
     devCounter += 1
#     counter += 1
     printHere = (devCounter % 200 == 0)
     _, _, _, newLoss, newWords = doForwardPass(input_indices_list, wordStartIndices_list, surprisalTable = surprisalTable, doDropout=False, batchSizeHere=devBatchSize)
     devLoss += newLoss
     devWords += newWords
     if printHere:
         print "Dev examples "+str(devCounter)
     devCounterTimesBatchSize += devBatchSize
   devSurprisalTableHere = [surp/(devCounterTimesBatchSize) for surp in surprisalTable]
   return devLoss/devWords, devSurprisalTableHere

DEV_PERIOD = 5000
epochCount = 0
corpusBase = corpus_cached["train"]
while failedDevRuns == 0:
  epochCount += 1
  print "Starting new epoch, permuting corpus"
  corpusBase.permute()
#  corpus = getNextSentence("train")
  corpus = corpusBase.iterator()
  stream = createStream(corpus)



  if counter > 5:
#       if counter % DEV_PERIOD == 0:
#          newDevLoss, devSurprisalTableHereSV = computeDevLoss(order="SV")
#          newDevLoss, devSurprisalTableHereVS = computeDevLoss(order="VS")
#          print(devSurprisalTableHereSV)
#          
#          print(devSurprisalTableHereVS)
#          quit()
          newDevLoss, devSurprisalTableHere = computeDevLoss(order="mixed")
#             devLosses.append(
          devLosses.append(newDevLoss)
          print "New dev loss "+str(newDevLoss)+". previous was: "+str(lastDevLoss)
          if newDevLoss > 15 or len(devLosses) > 99:
              print "Abort, training too slow?"
              devLosses.append(newDevLoss+0.001)
              #newDevLoss = 15

          if lastDevLoss is None or newDevLoss < lastDevLoss:
              devSurprisalTable = devSurprisalTableHere
#          if counter == DEV_PERIOD and model != "REAL_REAL":
#             with open(TARGET_DIR+"/model-"+language+"_"+__file__+"_model_"+str(myID)+"_"+model+".txt", "w") as outFile:
#                 print >> outFile, "\t".join(["Key", "DH_Weight", "Distance_Weight"])
#                 for i, key in enumerate(itos_deps):
#                   #dhWeight = dhWeights[i]
#                   distanceWeight = distanceWeights[i]
#                   print >> outFile, "\t".join(map(str,[key, dhWeight, distanceWeight]))

          if newDevLoss > 15 or len(devLosses) > 100:
              print "Abort, training too slow?"
              failedDevRuns = 1
              break


          if lastDevLoss is None or newDevLoss < lastDevLoss:
             lastDevLoss = newDevLoss
             failedDevRuns = 0
          else:
             failedDevRuns += 1
             print "Skip saving, hoping for better model"
             print devLosses
             print "Epoch "+str(epochCount)+" "+str(counter)
             print zip(range(1,horizon+1), devSurprisalTable)


             break

  while True:
       counter += 1
       printHere = (counter % 100 == 0)

       try:
          input_indices_list = []
          wordStartIndices_list = []
          for _ in range(batchSize):
             input_indices, wordStartIndices = next(stream)
             input_indices_list.append(input_indices)
             wordStartIndices_list.append(wordStartIndices)
       except StopIteration:
          break
       loss, baselineLoss, policy_related_loss, _, wordNumInPass = doForwardPass(input_indices_list, wordStartIndices_list, batchSizeHere=batchSize)
       if wordNumInPass > 0:
         doBackwardPass(loss, baselineLoss, policy_related_loss)
       else:
         print "No words, skipped backward"
       if printHere:
          print "Epoch "+str(epochCount)+" "+str(counter)
          print zip(range(1,horizon+1), devSurprisalTable)


newDevLoss, devSurprisalTableHereSV = computeDevLoss(order="SV")
newDevLoss, devSurprisalTableHereVS = computeDevLoss(order="VS")
print(devSurprisalTableHereSV)

print(devSurprisalTableHereVS)


with open("results/ptb"+"/estimates-"+language+"_"+__file__+"_model_"+str(myID)+"_"+model+".txt", "w") as outFile:
    print >> outFile, " ".join(sys.argv)
    print >> outFile, " ".join(map(str,devLosses))
    print >> outFile, " ".join(map(str,devSurprisalTableHereSV))
    print >> outFile, " ".join(map(str,devSurprisalTableHereVS))




# right:
#[6.786592692990107, 5.507791746779186, 5.163098867359887, 5.031761147170272, 4.973789765200808, 4.9426595494569945, 4.92338613443479, 4.910399820412, 4.900841583639424, 4.893672238652172, 4.8884662020162954, 4.884067583085674, 4.881004487975563, 4.87838656113672, 4.876651883799357, 4.8751985482623414, 4.8738616470668426, 4.8727709015718625, 4.872043759776053, 4.871537018921128]
# wrong:
#[6.786592692990107, 5.515482572883915, 5.184529829022917, 5.060820564840738, 5.005406301466529, 4.975996213095824, 4.9577841414726205, 4.9462734136619355, 4.937298680379344, 4.930855762320136, 4.925522574502463, 4.9217107436326115, 4.918733901414627, 4.916623616811106, 4.915051826836932, 4.913446591334558, 4.91226500599, 4.911328594788609, 4.910697741430733, 4.909965576643994]

# another run:
#[6.786155409328393, 5.501257273826284, 5.156584289911885, 5.024320856573909, 4.965475277672772, 4.934414756666834, 4.914249611365661, 4.90195703196484, 4.892793117278371, 4.886134552373278, 4.880749000024465, 4.876225876202303, 4.872963514482095, 4.870623099098129, 4.868600607976441, 4.867191986300277, 4.866134104767418, 4.865227124673688, 4.86458936988604, 4.86413948869967]
#[6.786155409328393, 5.517118954145915, 5.181507091624265, 5.0586127766279345, 5.003393518708782, 4.972750332323443, 4.952938461829628, 4.941119039171158, 4.932391392831568, 4.9252450367752205, 4.919827662314053, 4.9150522900480444, 4.91156855843673, 4.909427334227785, 4.907651790265536, 4.906054842205199, 4.904888625735773, 4.9038211843538635, 4.902975727677161, 4.902287185009879]

# another run
#[6.786361543740107, 5.508351102485558, 5.1602466144976855, 5.029016087294181, 4.971228078129674, 4.94144499738085, 4.9216120501289415, 4.9094417718486785, 4.900191957834044, 4.893063988330241, 4.887502009054508, 4.8830461212271885, 4.879915157057774, 4.87745413036717, 4.875567788504657, 4.874163425667461, 4.873112190666783, 4.872265998720476, 4.871819591952513, 4.871286419808871]
#[6.786361543740107, 5.514468648317764, 5.179411707155184, 5.057936731768672, 5.002236476749933, 4.972250239227317, 4.953227849788711, 4.942573625181055, 4.933531674561547, 4.926797641342947, 4.921387143783101, 4.91710217962165, 4.914083739643623, 4.911802510678476, 4.909978441767425, 4.90862572425537, 4.90755312063586, 4.906618790473061, 4.906026870193533, 4.905363639778371]

