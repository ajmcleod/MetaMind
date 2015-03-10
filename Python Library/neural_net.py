#!/usr/bin/env python

import time
import numpy as np
import theano
from theano import tensor as T
from sklearn import metrics
from glove import *
from sst import *
from softmax_layer import *
from fully_connected_layer import *


class neural_net:

  def __init__(self, training_examples, training_labels, num_hidden_neurons, batch_size = 1000, learning_rate = 1e-4,
               reg_strength = 1e-3, rms_decay_rate = 0.9, rms_injection_rate = None, dropout_prob = 0.5,
               use_nesterov_momentum = False, momentum_decay_rate = None, num_epochs = 5, random_seed = None):

    self.num_input_features = training_examples.shape[1]
    self.num_classes        = np.amax(training_labels) + 1
    num_train_examples      = training_examples.shape[0]
    num_train_batches       = np.ceil(float(num_train_examples) / batch_size).astype(int)
    if rms_injection_rate == None:
      rms_injection_rate = 1 - rms_decay_rate

    X                     = T.dmatrix('X')
    y                     = T.lvector('y')
    index                 = T.lscalar('index')
    trng                  = T.shared_randomstreams.RandomStreams(random_seed)
    self.X_train          = theano.shared(training_examples, borrow=True)
    self.y_train          = theano.shared(training_labels, borrow=True)
    training_options = {'learning_rate': learning_rate, 'batch_size': batch_size,
                        'dropout_prob': dropout_prob, 'reg_strength': reg_strength,
                        'rms_decay_rate': rms_decay_rate, 'rms_injection_rate': rms_injection_rate,
                        'use_nesterov_momenum': use_nesterov_momentum, 'momentum_decay_rate': momentum_decay_rate,
                        'reset_gradient_sums': True, 'reset_gradient_velocities': True}

    self.L0 = fully_connected_layer(X_var = X, X_values = self.X_train, trng = trng,
                                    num_output_neurons = num_hidden_neurons[0],
                                    dropout_prob = dropout_prob, batch_size = batch_size)
    self.L1 = fully_connected_layer(X_var = self.L0.masked_output, X_values = self.L0.output,
                                    trng = trng, num_output_neurons = num_hidden_neurons[1],
                                    dropout_prob = dropout_prob, batch_size = batch_size)
    self.L2 = fully_connected_layer(X_var = self.L1.masked_output, X_values = self.L1.output,
                                    trng = trng, num_output_neurons = num_hidden_neurons[2],
                                    dropout_prob = dropout_prob, batch_size = batch_size)
    self.L3 = softmax_layer(X_var = self.L2.masked_output, y_var = y, X_values = self.L2.output,
                            y_values = self.y_train, trng = trng, training_options = training_options)

    self.L0.configure_training_environment(self.L3.training_cost, learning_rate = learning_rate,
                                           reg_strength = reg_strength, rms_decay_rate = rms_decay_rate,
                                           rms_injection_rate = rms_injection_rate,
                                           use_nesterov_momentum = use_nesterov_momentum,
                                           momentum_decay_rate = momentum_decay_rate)
    self.L1.configure_training_environment(self.L3.training_cost, learning_rate = learning_rate,
                                           reg_strength = reg_strength, rms_decay_rate = rms_decay_rate,
                                           rms_injection_rate = rms_injection_rate,
                                           use_nesterov_momentum = use_nesterov_momentum,
                                           momentum_decay_rate = momentum_decay_rate)
    self.L2.configure_training_environment(self.L3.training_cost, learning_rate = learning_rate,
                                           reg_strength = reg_strength, rms_decay_rate = rms_decay_rate,
                                           rms_injection_rate = rms_injection_rate,
                                           use_nesterov_momentum = use_nesterov_momentum,
                                           momentum_decay_rate = momentum_decay_rate)

    parameter_updates = self.L3.parameter_updates
    parameter_updates.extend(self.L2.parameter_updates)
    parameter_updates.extend(self.L1.parameter_updates)
    parameter_updates.extend(self.L0.parameter_updates)
    train_model = theano.function(inputs = [index], outputs = self.L3.training_cost, updates = parameter_updates,
                                  givens = {X: self.X_train[index * batch_size: (index + 1) * batch_size],
                                            y: self.y_train[index * batch_size: (index + 1) * batch_size]})

    lowest_cost = np.inf
    current_cost = np.inf
    epoch = 0
    start_time = time.clock()

    for ii in range(num_epochs):
      epoch = epoch + 1
      print 'starting epoch %i \n' % (ii + 1)
      for batch_index in xrange(num_train_batches):	
        current_cost = train_model(batch_index)
        if current_cost < lowest_cost:
          lowest_cost = current_cost
          best_W0 = self.L0.W.eval()
          best_b0 = self.L0.b.eval()
          best_W1 = self.L1.W.eval()
          best_b1 = self.L1.b.eval()
          best_W2 = self.L2.W.eval()
          best_b2 = self.L2.b.eval()
          best_W3 = self.L3.W.eval()
          best_b3 = self.L3.b.eval()
        print 'negative_log_likelihood: %f' % (current_cost)
        print 'train_accuracy: %f' % (self.L3.train_accuracy.eval())
        print ''

    end_time = time.clock()
    set_parameters = theano.function(inputs = [], outputs = self.L3.train_accuracy,
                                     updates = [(self.L0.W, theano.shared(best_W0)),
                                                (self.L0.b, theano.shared(best_b0)),
                                                (self.L1.W, theano.shared(best_W1)),
                                                (self.L1.b, theano.shared(best_b1)),
                                                (self.L2.W, theano.shared(best_W2)),
                                                (self.L2.b, theano.shared(best_b2)),
                                                (self.L3.W, theano.shared(best_W3)),
                                                (self.L3.b, theano.shared(best_b3))])
    dummy = set_parameters()
    print 'softmax trained in %i epochs and %f seconds with training accuracy %f %%' % \
                               (epoch, end_time - start_time, 100 * self.L3.train_accuracy.eval())

  def accuracy(self, X, y):
    return T.mean(T.eq(T.argmax(T.nnet.softmax(T.dot(X, self.L3.W) + self.L3.b), axis = 1), y))

  def accuracy_np(self, X, y):
    return np.mean(np.equal(np.argmax(T.nnet.softmax(T.dot(X, self.softmaxLayer0.W) + self.softmaxLayer0.b).eval(), axis=1), y))

  def f1(self, X, y):
    return metrics.f1_score(y, T.argmax(T.nnet.softmax(T.dot(X, self.softmaxLayer0.W) + self.softmaxLayer0.b), axis = 1).eval())
