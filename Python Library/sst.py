#!/usr/bin/env python

import numpy as np
import cPickle as pickle

class sst():
  def loadTrainingData(self):
    with open('stanfordSentimentTreebank/orderedTrainPhraseIDs.pickle','rb') as handle:
      self.trainingPhraseIDs = pickle.load(handle)
    with open('stanfordSentimentTreebank/orderedTrainPhrases.pickle','rb') as handle:
      self.trainingPhrases = pickle.load(handle)
    with open('stanfordSentimentTreebank/orderedTrainSentiments.pickle','rb') as handle:
      self.trainingSentiments = pickle.load(handle)
    self.trainingSentimentsI = []
    for i in range(60):
      self.trainingSentimentsI.append(map(int,list(np.floor(5 * np.array(self.trainingSentiments[i])) - np.floor(np.array(self.trainingSentiments[i])))))
 
  def loadDevelopmentData(self):
    with open('stanfordSentimentTreebank/orderedDevPhraseIDs.pickle','rb') as handle:
      self.devPhraseIDs = pickle.load(handle)
    with open('stanfordSentimentTreebank/orderedDevPhrases.pickle','rb') as handle:
      self.devPhrases = pickle.load(handle)
    with open('stanfordSentimentTreebank/orderedDevSentiments.pickle','rb') as handle:
      self.devSentiments = pickle.load(handle)
    self.devSentimentsI = []
    for i in range(60):
      self.devSentimentsI.append(map(int,list(np.floor(5 * np.array(self.devSentiments[i])) - np.floor(np.array(self.devSentiments[i])))))

  def loadTestingData(self):
    with open('stanfordSentimentTreebank/orderedTestPhraseIDs.pickle','rb') as handle:
      self.testPhraseIDs = pickle.load(handle)
    with open('stanfordSentimentTreebank/orderedTestPhrases.pickle','rb') as handle:
      self.testPhrases = pickle.load(handle)
    with open('stanfordSentimentTreebank/orderedTestSentiments.pickle','rb') as handle:
         self.testSentiments = pickle.load(handle)
    self.testSentimentsI = []
    for i in range(60):
      self.testSentimentsI.append(map(int,list(np.floor(5 * np.array(self.testSentiments[i])) - np.floor(np.array(self.testSentiments[i])))))

  def loadTrainingDataD(self):
    with open('stanfordSentimentTreebank/orderedTrainPhraseIDsWithDuplicates.pickle','rb') as handle:
      self.trainingPhraseIDsD = pickle.load(handle)
    with open('stanfordSentimentTreebank/orderedTrainPhrasesWithDuplicates.pickle','rb') as handle:
      self.trainingPhrasesD = pickle.load(handle)
    with open('stanfordSentimentTreebank/orderedTrainSentimentsWithDuplicates.pickle','rb') as handle:
      self.trainingSentimentsD = pickle.load(handle)
    self.trainingSentimentsDI = []
    for i in range(60):
      self.trainingSentimentsDI.append(map(int,list(np.floor(5 * np.array(self.trainingSentimentsD[i])) - np.floor(np.array(self.trainingSentimentsD[i])))))

  def loadDevelopmentDataD(self):
    with open('stanfordSentimentTreebank/orderedDevPhraseIDsWithDuplicates.pickle','rb') as handle:
      self.devPhraseIDsD = pickle.load(handle)
    with open('stanfordSentimentTreebank/orderedDevPhrasesWithDuplicates.pickle','rb') as handle:
      self.devPhrasesD = pickle.load(handle)
    with open('stanfordSentimentTreebank/orderedDevSentimentsWithDuplicates.pickle','rb') as handle:
      self.devSentimentsD = pickle.load(handle)
    self.devSentimentsDI = []
    for i in range(60):
      self.devSentimentsDI.append(map(int,list(np.floor(5 * np.array(self.devSentimentsD[i])) - np.floor(np.array(self.devSentimentsD[i])))))

  def loadTestingDataD(self):
    with open('stanfordSentimentTreebank/orderedTestPhraseIDsWithDuplicates.pickle','rb') as handle:
      self.testPhraseIDsD = pickle.load(handle)
    with open('stanfordSentimentTreebank/orderedTestPhrasesWithDuplicates.pickle','rb') as handle:
      self.testPhrasesD = pickle.load(handle)
    with open('stanfordSentimentTreebank/orderedTestSentimentsWithDuplicates.pickle','rb') as handle:
         self.testSentimentsD = pickle.load(handle)
    self.testSentimentsDI = []
    for i in range(60):
      self.testSentimentsDI.append(map(int,list(np.floor(5 * np.array(self.testSentimentsD[i])) - np.floor(np.array(self.testSentimentsD[i])))))

  def fromTrainingSelectN(self, phrase_length, sample_num):
    max_int = len(self.trainingPhraseIDs[phrase_length])
    if max_int > sample_num:
        samples = np.random.randint(max_int, size = sample_num)
        while np.unique(samples).size < sample_num:
            samples = np.concatenate((np.unique(samples),np.random.randint(max_int,size = (sample_num - np.unique(samples).size))))
    else: 
        samples = range(max_int)
    return samples

sst = sst()
