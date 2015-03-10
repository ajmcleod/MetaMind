#!/usr/bin/env python

import numpy as np
import theano
from theano import tensor as T
import itertools

class fully_connected_layer:
  _ids = itertools.count(1)

  def __init__(self, X_var, X_values, num_output_neurons, trng, batch_size, dropout_prob, W = None, b = None):
    self.layer_id = self._ids.next()

    self.W    = None
    self.b    = None
    self.trng = trng
    self.X    = X_var
    self.num_output_neurons = num_output_neurons

    self.initialize_parameters(X_values, W, b)
    self.reset_gradient_sums()
    self.reset_gradient_velocities()

    if dropout_prob > 0.0:
      if batch_size == None:
        batch_size = self.num_training_examples
      self.dropout_mask  = self.trng.binomial(n = 1, p = 1 - dropout_prob, size=(batch_size, self.num_features)) / dropout_prob
      self.masked_output = T.nnet.softplus(T.dot(self.X * self.dropout_mask[:self.X.shape[0]], self.W) + self.b)
    else:
      self.masked_output = T.nnet.softplus(T.dot(self.X, self.W) + self.b)

    print 'Fully connected layer %i initialized' % (self.layer_id)

  #######################################################################################################################

  def configure_training_environment(self, cost_function, learning_rate = 1e-3, reg_strength = 1e-4, rms_decay_rate = 0.9,
                                     rms_injection_rate = None, use_nesterov_momentum = False, momentum_decay_rate = 0.9):

    g_W = T.grad(cost=cost_function, wrt=self.W)
    g_b = T.grad(cost=cost_function, wrt=self.b)

    if use_nesterov_momentum:
      W_update = self.W_gradient_velocity * momentum_decay_rate**2 - (1 + momentum_decay_rate) * learning_rate * g_W
      b_update = self.b_gradient_velocity * momentum_decay_rate**2 - (1 + momentum_decay_rate) * learning_rate * g_b
    else:
      W_update = - learning_rate * g_W
      b_update = - learning_rate * g_b

    self.parameter_updates = [(self.W, self.W + W_update / T.sqrt(self.W_gradient_sums + T.sqr(g_W)) - reg_strength * self.W),
                              (self.b, self.b + b_update / T.sqrt(self.b_gradient_sums + T.sqr(g_b)) - reg_strength * self.b),
                              (self.W_gradient_sums, rms_decay_rate * self.W_gradient_sums + rms_injection_rate * T.sqr(W_update / learning_rate)),
                              (self.b_gradient_sums, rms_decay_rate * self.b_gradient_sums + rms_injection_rate * T.sqr(b_update / learning_rate))]

    if use_nesterov_momentum:
      self.parameter_updates.append((self.W_gradient_velocity, momentum_decay_rate * self.W_gradient_velocity - learning_rate * g_W))
      self.parameter_updates.append((self.b_gradient_velocity, momentum_decay_rate * self.b_gradient_velocity - learning_rate * g_b))

  #######################################################################################################################

  def initialize_parameters(self, X_values, W, b):

    self.num_training_examples, self.num_features  = X_values.eval().shape

    if self.W == None:
      if W == None:
        self.W = theano.shared(self.trng.uniform(low = - np.sqrt(6. / (self.num_features + self.num_output_neurons)),
                                                 high = np.sqrt(6. / (self.num_features + self.num_output_neurons)),
                                                 size = (self.num_features, self.num_output_neurons)).eval(), borrow=True)
      else:
        self.W = theano.shared(W, borrow=True)

    if self.b == None:
      if b == None:
        self.b = theano.shared(np.zeros((self.num_output_neurons,)), borrow=True)
      else:
        self.b = theano.shared(b, borrow=True)

    self.output = T.nnet.softplus(T.dot(X_values, self.W) + self.b)

  #######################################################################################################################

  def predict(self, X_test):
    return T.nnet.softplus(T.dot(X_test, self.W) + self.b)

  #######################################################################################################################

  def reset_gradient_sums(self):
    self.W_gradient_sums = theano.shared(1e-8 * np.ones((self.num_features, self.num_output_neurons)), borrow=True)
    self.b_gradient_sums = theano.shared(1e-8 * np.ones((self.num_output_neurons,)), borrow=True)

  #######################################################################################################################

  def reset_gradient_velocities(self):
    self.W_gradient_velocity = theano.shared(np.zeros((self.num_features, self.num_output_neurons)), borrow=True)
    self.b_gradient_velocity = theano.shared(np.zeros((self.num_output_neurons,)), borrow = True)


