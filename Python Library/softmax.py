#!/usr/bin/env python

import time
import numpy as np
import theano
from theano import tensor as T
from glove import *
from sst import *
from softmax_layer import *

class softmax:

  def __init__(self, training_examples, training_labels, batch_size = 1000, learning_rate = 0.1,
               reg_strength = 1e-3, rms_decay_rate = 0.9, rms_injection_rate = None, dropout_prob = 0.5,
               accuracy_margin = 0.05, marginal_accuracy_tolerance = 10, random_seed = None):

    self.num_features = training_examples.shape[1]
    self.num_classes = np.amax(training_labels) + 1
    num_train_examples = training_examples.shape[0]
    n_train_batches = np.ceil(float(num_train_examples) / batch_size).astype(int)
    self.X_train = theano.shared(np.concatenate((training_examples,np.zeros((batch_size - num_train_examples%batch_size, self.num_features)))))
    self.y_train = theano.shared(np.concatenate((training_labels,np.zeros((batch_size - num_train_examples%batch_size,)).astype(int))))
    if rms_injection_rate == None:
      rms_injection_rate = 1 - rms_decay_rate

    X = T.dmatrix('X')
    y = T.lvector('y')
    index = T.lscalar('index')
    trng = T.shared_randomstreams.RandomStreams(random_seed)

    self.softmaxLayer0 = softmax_layer(inputs = X, training_labels = y, trng = trng, num_features = self.num_features, num_classes = self.num_classes, batch_size = batch_size, dropout_prob = dropout_prob, reg_strength = reg_strength, num_train_examples = training_examples.shape[0], rms_decay_rate = rms_decay_rate, rms_injection_rate = rms_injection_rate)

    train_model = theano.function(inputs = [index], outputs = self.softmaxLayer0.negative_log_likelihood, updates = self.softmaxLayer0.parameter_updates,
                                  givens = {X: self.X_train[index * batch_size: (index + 1) * batch_size],
                                            y: self.y_train[index * batch_size: (index + 1) * batch_size]})

    lowest_cost = np.inf
    current_cost = np.inf
    marginal_iterations = 0
    epoch = 0
    start_time = time.clock()
    done_training = False
    #while (not done_training):	
    for ii in range(1):
      print 'new epoch'
      epoch = epoch + 1
      for batch_index in xrange(n_train_batches):	
        previous_cost = current_cost
        current_cost = train_model(batch_index)
        if current_cost < lowest_cost:
          lowest_cost = current_cost
          best_W = self.softmaxLayer0.W.get_value()
          best_b = self.softmaxLayer0.b.get_value()
        print self.accuracy(self.X_train, self.y_train).eval()

        if np.abs(current_cost - previous_cost) < accuracy_margin: 
          marginal_iterations += 1
          if marginal_iterations == marginal_accuracy_tolerance:
            done_training = True
        else:
          marginal_iterations = 0

      if epoch%100 == 0:
        accuracy_margin *= np.log(10)

    end_time = time.clock()
    self.softmaxLayer0.W = theano.shared(best_W)
    self.softmaxLayer0.b = theano.shared(best_b)
    current_best_accuracy = self.accuracy(self.X_train, self.y_train).eval()
    print 'softmax trained in %i epochs and %f seconds with training accuracy %f %%' % (epoch, end_time - start_time, 100 * current_best_accuracy)

  def accuracy(self, X, y):
    return T.mean(T.eq(T.argmax(T.nnet.softmax(T.dot(X, self.softmaxLayer0.W) + self.softmaxLayer0.b), axis = 1), y))

