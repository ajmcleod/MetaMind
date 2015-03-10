#!/usr/bin/env python

import numpy as np
import theano
from theano import tensor as T
import itertools

class softmax_layer:
  _ids = itertools.count(1)

  def __init__(self, X_var, y_var, X_values, y_values, trng, training_options = None, W = None, b = None):
    self.layer_id = self._ids.next()

    self.X    = X_var
    self.y    = y_var
    self.trng = trng
    self.W    = None
    self.b    = None
    self.initialize_softmax_values(X_values, y_values, W, b)
    self.set_training_parameters(training_options)
    self.train_accuracy = T.mean(T.eq(T.argmax(T.nnet.softmax(T.dot(X_values, self.W) + self.b), axis = 1), y_values))

    print 'Softmax %i initialized' % (self.layer_id)

  #######################################################################################################################

  def set_training_parameters(self, training_options):

    if 'reset_gradient_sums' in training_options and training_options['reset_gradient_sums']:
      self.W_gradient_sums = theano.shared(1e-8 * np.ones((self.num_features, self.num_classes)), borrow=True)
      self.b_gradient_sums = theano.shared(1e-8 * np.ones((self.num_classes,)), borrow=True)
    if 'reset_gradient_velocities' in training_options and training_options['reset_gradient_velocities']:
      self.W_gradient_velocity = theano.shared(np.zeros((self.num_features, self.num_classes)), borrow=True)
      self.b_gradient_velocity = theano.shared(np.zeros((self.num_classes,)), borrow = True)

    learning_rate = training_options['learning_rate']
    dropout_prob  = training_options['dropout_prob']
    batch_size    = training_options['batch_size']
    reg_strength  = training_options['reg_strength']
    rms_decay_rate = training_options['rms_decay_rate']
    rms_injection_rate = training_options['rms_injection_rate']

    if dropout_prob > 0.0:
      dropout_mask                      = self.trng.binomial(n = 1, p = 1 - dropout_prob, size=(batch_size, self.num_features)) / dropout_prob
      masked_log_likelihood             = T.log(T.nnet.softmax(T.dot(dropout_mask[:self.X.shape[0]] * self.X, self.W) + self.b))
    else:
      masked_log_likelihood             = T.log(T.nnet.softmax(T.dot(self.X, self.W) + self.b))
    self.masked_negative_log_likelihood = - T.mean(masked_log_likelihood[T.arange(self.y.shape[0]),self.y])

    g_W = T.grad(cost=self.masked_negative_log_likelihood, wrt=self.W)
    g_b = T.grad(cost=self.masked_negative_log_likelihood, wrt=self.b)

    if 'use_nesterov_momentum' in training_options and training_options['use_nesterov_momentum']:
      if 'momentum_decay_rate' in training_options:
        momentum_decay_rate = training_options['momentum_decay_rate']
      else:
        momentum_decay_rate = 0.9
      W_update = self.W_gradient_velocity * momentum_decay_rate**2 - (1 + momentum_decay_rate) * learning_rate * g_W
      b_update = self.b_gradient_velocity * momentum_decay_rate**2 - (1 + momentum_decay_rate) * learning_rate * g_b
    else:
      W_update = - learning_rate * g_W
      b_update = - learning_rate * g_b

    self.parameter_updates = [(self.W, self.W + W_update / T.sqrt(self.W_gradient_sums + T.sqr(g_W)) - reg_strength * self.W),
                              (self.b, self.b + b_update / T.sqrt(self.b_gradient_sums + T.sqr(g_b)) - reg_strength * self.b),
                              (self.W_gradient_sums, rms_decay_rate * self.W_gradient_sums + rms_injection_rate * T.sqr(W_update / learning_rate)),
                              (self.b_gradient_sums, rms_decay_rate * self.b_gradient_sums + rms_injection_rate * T.sqr(b_update / learning_rate))]

    if 'use_nesterov_momentum' in training_options and training_options['use_nesterov_momentum']:
      self.parameter_updates.append((self.W_gradient_velocity, momentum_decay_rate * self.W_gradient_velocity - learning_rate * g_W))
      self.parameter_updates.append((self.b_gradient_velocity, momentum_decay_rate * self.b_gradient_velocity - learning_rate * g_b))


  #######################################################################################################################

  def initialize_softmax_values(self, X_values, y_values, W, b):

    self.num_features  = X_values.eval().shape[1]
    self.num_classes   = T.max(y_values).eval() + 1

    if self.W == None:
      if W == None:
        self.W = theano.shared(self.trng.uniform(low = - np.sqrt(6. / (self.num_features + self.num_classes)),
                                                 high = np.sqrt(6. / (self.num_features + self.num_classes)),
                                                 size = (self.num_features, self.num_classes)).eval(), borrow=True)
      else:
        self.W = theano.shared(W, borrow=True)

    if self.b == None:
      if b == None:
        self.b = theano.shared(np.zeros((self.num_classes,)), borrow=True)
      else:
        self.b = theano.shared(b, borrow=True)

    self.training_cost = - T.mean(T.log(T.nnet.softmax(T.dot(X_values, self.W) + self.b))[T.arange(y_values.shape[0]), y_values])

  #######################################################################################################################

  def predict(self, X_test):
    return T.argmax(T.nnet.softmax(T.dot(X_test, self.W) + self.b), axis = 1)
