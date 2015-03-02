#!/usr/bin/env python                                                                                                                                         
 
import numpy as np
import pickle

glove_dataset = 'glove.6B.300d'
glove_file = glove_dataset + '.txt'
glove = open(glove_file)
dictionary = {}
dictionary_size = 0
 
for line in glove:
   dictionary_size += 1
   word_data = line.split(" ") 
   current_word = word_data[0]
   word_data.pop(0)
   current_vector = np.array([float(x) for x in word_data])
   dictionary[current_word] = current_vector/np.linalg.norm(current_vector) 
glove.close()
vector_d = dictionary['the'].size
print 'dictionary generated'

sentences = []
with open('../stanfordSentimentTreebank/SOStr.txt','r') as f:
   for line in f:
        sentences.append(line.rstrip().split("|"))
 
uninitialized_words = []
for i in range(len(sentences)):

   current_sentence = sentences[i]

   for p in range(len(current_sentence)):
      current_word = current_sentence[p].lower()
      if (current_word not in dictionary) and (current_word not in uninitialized_words):
         uninitialized_words.append(current_word)

 
with open(glove_dataset + '_uninitialized_words.pickle', 'wb') as handle:
   pickle.dump(uninitialized_words, handle) 

print 'uninitialized word pickle file generated'
