#!/usr/bin/env python

import numpy as np
import theano
from theano import tensor as T
import itertools

class softmax_layer:
  _ids = itertools.count(1)

  def __init__(self, inputs, num_features, num_classes, trng, training_labels = None, learning_rate = 0.1, batch_size = None, num_train_examples = None, reg_strength = None, rms_decay_rate = None, rms_injection_rate = None, dropout_prob = 0.5, W = None, b = None):
    self.layer_id = self._ids.next()

    if W == None:
      self.W = theano.shared(trng.uniform(low = - np.sqrt(6. / (num_features + num_classes)), high = np.sqrt(6. / (num_features + num_classes)), size = (num_features, num_classes)).eval())
    else:
      self.W = theano.shared(W, borrow=True)

    if b == None:
      self.b = theano.shared(np.zeros((num_classes,)))
    else:
      self.b = theano.shared(b, borrow=True)

    self.inputs        = inputs
    self.num_features  = num_features
    self.num_classes   = num_classes
    self.trng          = trng
    self.output        = T.argmax(T.nnet.softmax(T.dot(self.inputs, self.W) + self.b), axis = 1)

    if training_labels != None:
      self.set_training_parameters(training_labels = training_labels, reg_strength = reg_strength, rms_decay_rate = rms_decay_rate, rms_injection_rate = rms_injection_rate, dropout_prob = dropout_prob, learning_rate = learning_rate, batch_size = batch_size, num_train_examples = num_train_examples, reset_gradient_sums = True)

    print 'Softmax %i initialized' % (self.layer_id)

  def set_training_parameters(self, training_labels, reg_strength, rms_decay_rate, rms_injection_rate, dropout_prob = 0.5, learning_rate = 0.1, batch_size = None, num_train_examples = None, reset_gradient_sums = False):

    if reset_gradient_sums:
      self.W_gradient_sums = theano.shared(1e-8 * np.ones((self.num_features, self.num_classes)), borrow=True)
      self.b_gradient_sums = theano.shared(1e-8 * np.ones((self.num_classes,)), borrow=True)

    current_dropout_prob = dropout_prob
    if not isinstance(batch_size, int):
      print 'Variable batch_size needed to set up training environment when dropout_prob > 0. Proceeding with dropout_prob = 0.0 in softmax %i.' % (self.layer_id)
      current_dropout_prob = 0.0

    if current_dropout_prob > 0.0:
      dropout_mask                 = self.trng.binomial(n = 1, p = 1 - current_dropout_prob, size=(batch_size, self.num_features)) / current_dropout_prob
      log_likelihoods              = T.log(T.nnet.softmax(T.dot(dropout_mask[:self.inputs.shape[0]] * self.inputs, self.W) + self.b))
      self.negative_log_likelihood = - T.mean(log_likelihoods[T.arange(training_labels.shape[0]),training_labels])
    else:
      log_likelihoods              = T.log(T.nnet.softmax(T.dot(self.inputs, self.W) + self.b))
      self.negative_log_likelihood = - T.mean(log_likelihoods[T.arange(training_labels.shape[0]),training_labels])

    self.diff=theano.shared(np.zeros((1000,50)))
    g_W = T.grad(cost=self.negative_log_likelihood, wrt=self.W)
    g_b = T.grad(cost=self.negative_log_likelihood, wrt=self.b)
    self.parameter_updates = [(self.W, self.W - learning_rate * g_W / T.sqrt(self.W_gradient_sums + T.sqr(g_W)) - reg_strength * self.W),
                              (self.b, self.b - learning_rate * g_b / T.sqrt(self.b_gradient_sums + T.sqr(g_b)) - reg_strength * self.b),
                              (self.W_gradient_sums, rms_decay_rate * self.W_gradient_sums + rms_injection_rate * T.sqr(g_W)),
                              (self.b_gradient_sums, rms_decay_rate * self.b_gradient_sums + rms_injection_rate * T.sqr(g_b)), (self.diff, dropout_mask)]
    

