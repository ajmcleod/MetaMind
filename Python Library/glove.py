#!/usr/bin/env python

import numpy as np
import cPickle as pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import *

class glove:

  def __init__(self, glove_dataset):
    self.alpha_word = 0.1
    self.dictionary = {}
    self.word_gradient_sums = {}

    with open('Glove Data/' + glove_dataset + '.txt','r') as glove_data:
      self.vec_d = len(glove_data.readline().split(" ")) - 1
      glove_data.seek(0)
      for line in glove_data:
        word_data = line.split(" ") 
        current_word = word_data[0]
        word_data.pop(0)
        current_vector = np.array([float(x) for x in word_data])
        self.dictionary[current_word] = current_vector/np.linalg.norm(current_vector) 
        self.word_gradient_sums[current_word] = 1e-8 * np.ones(self.vec_d)

    with open('Glove Data/' + glove_dataset + '_uninitialized_words.pickle','r') as uninitialized:
      uninitialized_words = pickle.load(uninitialized)
      for w in uninitialized_words:
        tempVec = np.random.random([self.vec_d]) - 0.5
        self.dictionary[w] = tempVec/np.linalg.norm(tempVec)  
        self.word_gradient_sums[w] = 1e-8 * np.ones(50)
  
    print 'glove dictionary initialized'

  ############################################################## 

  def concatenatedPhraseVec(self, word_list, normalize = True):
    if isinstance(word_list, basestring):
      vecArray = self.dictionary[word_list.lower()]
    else:
      phrase_length = len(word_list)
      vecArray = np.zeros((phrase_length, self.vec_d))
      for i in range(phrase_length):
        vecArray[i] = self.dictionary[word_list[i].lower()] 
    norm = 1.0
    if normalize == True:
      norm = np.linalg.norm(vecArray.flatten()) 
    return vecArray.flatten() / norm  

  ############################################################## 

  def averagedPhraseVec(self, word_list, normalize = True):
    if isinstance(word_list, basestring):
      vec = self.dictionary[word_list.lower()]
    else:
      phrase_length = len(word_list)
      vec = np.zeros(self.vec_d)
      for i in range(phrase_length):
        vec += self.dictionary[word_list[i].lower()] 
      norm = 1.0
    if normalize == True:
      norm = np.linalg.norm(vec)
    return vec / norm

  ############################################################## 

  def vectorArray(self, phrases, type = 'averaged'):
    if type == 'concatenated':
      vectors = np.vstack([self.concatenatedPhraseVec(x) for x in phrases])
    else:
      vectors = np.vstack([self.averagedPhraseVec(x) for x in phrases])
    return vectors

  ############################################################## 

  def updateDictionary(self, phrases, word_derivatives):
    for i in range(len(phrases)):
      phrase = phrases[i]
      for j in range(len(phrase)): 
        self.dictionary[phrase[j]] = self.dictionary[phrase[j]] + self.alpha_word * word_derivatives[i][j*self.vec_d:(j+1)*self.vec_d] / np.sqrt(self.word_gradient_sums[phrase[j]]) 
        self.word_gradient_sums[phrase[j]] = self.word_gradient_sums[phrase[j]] + np.square(word_derivatives[i][j*self.vec_d:(j+1)*self.vec_d])
    return 0

  ############################################################## 

  def nearestNeighbors(self, phrase_list, num_nearest_neighbors, normalize = True, vector_type = 'avg'):

    assert vector_type == 'concat' or vector_type == 'avg', 'Invalid phrase vector type, will proceed using averaged phrase vector' 

    if vector_type == 'concat':
      phrase_vectors = np.array([concatenatedPhraseVec(phrase, normalize = normalize) for phrase in phrase_list])
    else:
      phrase_vectors = np.array([averagedPhraseVec(phrase, normalize = normalize) for phrase in phrase_list])

    nbrs = NearestNeighbors(n_neighbors = num_nearest_neighbors + 1, algorithm='ball_tree', metric = 'euclidean').fit(phrase_vectors)
    distances, neighbors_list = nbrs.kneighbors(word_vectors)

    return neighbors_list[:,1:]

  ############################################################## 


