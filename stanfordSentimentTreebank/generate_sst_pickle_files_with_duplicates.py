#!/usr/bin/env python                                                                                                                                         
 
import numpy as np
import cPickle as pickle

sentences = []
with open('SOStr.txt','r') as f:
   for line in f:
        sentences.append(line.rstrip().split("|"))

tree_structures = []
with open('STree.txt','r') as f:
   for line in f:
        tree_structures.append(map(int,line.rstrip().split("|")))
 
data_split = []
with open('datasetSplit.txt','r') as data_split_file:
   next(data_split_file)
   for line in data_split_file:
        data_split.append(int(line.rstrip().split(",")[1]))
 
train_sentences = np.array(sentences)[np.array(data_split) == 1]
train_structures = np.array(tree_structures)[np.array(data_split)==1] 
test_sentences = np.array(sentences)[np.array(data_split) == 2]  
test_structures = np.array(tree_structures)[np.array(data_split)==2] 
dev_sentences = np.array(sentences)[np.array(data_split) == 3]
dev_structures = np.array(tree_structures)[np.array(data_split)==3] 

phrases = {}
with open('dictionary.txt') as f:
   for line in f:
        current_phrase = line.rstrip().split("|")[0].lower()
        current_ID = int(line.rstrip().split("|")[1])
        phrases[current_phrase] = current_ID

sentiments = {}
with open('sentiment_labels.txt') as f:
   next(f)
   for line in f:
      current_ID, current_sentiment = line.rstrip().split("|")
      sentiments[int(current_ID)] = float(current_sentiment)

def nodePhrase(sentence, struct, nodes, current_node):
   current_phrase = np.array(sentence)[struct[:len(sentence)] == current_node] 
   subnodes = nodes[struct[len(sentence):] == current_node]
   for s in subnodes:
        if nodeOrder(struct, nodes, len(sentence), current_node, s):
           current_phrase = np.concatenate([nodePhrase(sentence, struct, nodes, s), current_phrase])
        else:
           current_phrase = np.concatenate([current_phrase,nodePhrase(sentence, struct, nodes, s)])
   return current_phrase
       
def nodeOrder(struct, nodes, sentence_length, node_1, node_2):
   position_1 = np.where(node_1 == struct)[0][0]   
   while position_1 >= sentence_length:
        position_1 = np.where(nodes[position_1 - sentence_length] == struct)[0][0]
   position_2 = np.where(node_2 == struct)[0][0]   
   while position_2 >= sentence_length:
        position_2 = np.where(nodes[position_2 - sentence_length] == struct)[0][0]
   return position_1 > position_2

train_phrase_IDs = []
train_phrases = []
train_sentiments = []
dev_phrase_IDs = []
dev_phrases = []
dev_sentiments = []
test_phrase_IDs = []
test_phrases = []
test_sentiments = []
for i in range(60):
   train_phrase_IDs.append([])
   train_phrases.append([])
   train_sentiments.append([])
   dev_phrase_IDs.append([])
   dev_phrases.append([])
   dev_sentiments.append([])
   test_phrase_IDs.append([])
   test_phrases.append([])
   test_sentiments.append([])

   
for i in range(train_sentences.size):

   current_sentence = train_sentences[i]

   for p in range(len(current_sentence)):
      current_word = current_sentence[p].lower()
      phrase_ID = phrases[current_word]
      train_phrase_IDs[1].append(phrase_ID)
      train_phrases[1].append([current_word])
      train_sentiments[1].append(sentiments[phrase_ID])

   sentence_structure = train_structures[i]
   nodes = np.delete(np.unique(sentence_structure),0)   
   for n in range(len(nodes)):
      phrase = nodePhrase(current_sentence, sentence_structure, nodes, nodes[n])
      phrase_length = len(phrase)
      phrase_ID = phrases[" ".join(phrase).lower()]	
      train_phrase_IDs[phrase_length].append(phrase_ID)
      train_phrases[phrase_length].append(list(phrase))
      train_sentiments[phrase_length].append(sentiments[phrase_ID])


for i in range(dev_sentences.size):

   current_sentence = dev_sentences[i]

   for p in range(len(current_sentence)):
      current_word = current_sentence[p].lower()
      phrase_ID = phrases[current_word]
      dev_phrase_IDs[1].append(phrase_ID)
      dev_phrases[1].append([current_word])
      dev_sentiments[1].append(sentiments[phrase_ID])

   sentence_structure = dev_structures[i]
   nodes = np.delete(np.unique(sentence_structure),0)   
   for n in range(len(nodes)):
      phrase = nodePhrase(current_sentence, sentence_structure, nodes, nodes[n])
      phrase_length = len(phrase)
      phrase_ID = phrases[" ".join(phrase).lower()]	
      dev_phrase_IDs[phrase_length].append(phrase_ID)
      dev_phrases[phrase_length].append(list(phrase))
      dev_sentiments[phrase_length].append(sentiments[phrase_ID])


for i in range(test_sentences.size):

   current_sentence = test_sentences[i]

   for p in range(len(current_sentence)):
      current_word = current_sentence[p].lower()
      phrase_ID = phrases[current_word]
      test_phrase_IDs[1].append(phrase_ID)
      test_phrases[1].append([current_word])
      test_sentiments[1].append(sentiments[phrase_ID])

   sentence_structure = test_structures[i]
   nodes = np.delete(np.unique(sentence_structure),0)   
   for n in range(len(nodes)):
      phrase = nodePhrase(current_sentence, sentence_structure, nodes, nodes[n])
      phrase_length = len(phrase)
      phrase_ID = phrases[" ".join(phrase).lower()]	
      test_phrase_IDs[phrase_length].append(phrase_ID)
      test_phrases[phrase_length].append(list(phrase))
      test_sentiments[phrase_length].append(sentiments[phrase_ID])


with open('orderedTrainPhraseIDsWithDuplicates.pickle', 'wb') as handle:
   pickle.dump(train_phrase_IDs, handle) 

with open('orderedDevPhraseIDsWithDuplicates.pickle', 'wb') as handle:
   pickle.dump(dev_phrase_IDs, handle) 
 
with open('orderedTestPhraseIDsWithDuplicates.pickle', 'wb') as handle:
   pickle.dump(test_phrase_IDs, handle) 

with open('orderedTrainPhrasesWithDuplicates.pickle', 'wb') as handle:
   pickle.dump(train_phrases, handle) 

with open('orderedDevPhrasesWithDuplicates.pickle', 'wb') as handle:
   pickle.dump(dev_phrases, handle) 

with open('orderedTestPhrasesWithDuplicates.pickle', 'wb') as handle:
   pickle.dump(test_phrases, handle) 

with open('orderedTrainSentimentsWithDuplicates.pickle', 'wb') as handle:
   pickle.dump(train_sentiments, handle) 

with open('orderedDevSentimentsWithDuplicates.pickle', 'wb') as handle:
   pickle.dump(dev_sentiments, handle) 

with open('orderedTestSentimentsWithDuplicates.pickle', 'wb') as handle:
   pickle.dump(test_sentiments, handle) 

print 'pickle files generated'

