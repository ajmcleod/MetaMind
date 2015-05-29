#!/usr/bin/env python

import numpy as np
import theano
from theano import tensor as T
import itertools
from theano_layer import *

class fully_connected_layer(theano_layer):
  _ids = itertools.count(1)

  def __init__(self, X_var, X_values, input_shape, num_neurons, trng, batch_size, dropout_prob = None, W = None, b = None):
    self.layer_id = self._ids.next()

    self.X    = X_var.flatten(2)
    self.W_shape = (num_neurons, np.prod(input_shape))
    self.X_values = X_values.flatten(2)
    self.num_features = np.prod(input_shape)
    self.num_output_neurons = num_neurons

    self.initialize_parameters(W, b)
    self.reset_gradient_sums()
    self.reset_gradient_velocities()

    self.output = T.dot(self.X_values, self.W.T) + self.b
    self.output_shape = num_neurons

    if dropout_prob > 0.0:
      self.dropout_mask  = trng.binomial(n = 1, p = 1 - dropout_prob, size=(batch_size, self.W_shape[1])) / dropout_prob
      self.masked_output = T.dot(self.X * self.dropout_mask[:self.X.shape[0]], self.W.T) + self.b
    else:
      self.masked_output = T.dot(self.X, self.W.T) + self.b

    print 'Fully Connected Layer %i initialized' % (self.layer_id)
    
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
  #######################################################################################################################

  def predict(self, X_test):
    return T.dot(X_test, self.W) + self.b

  #######################################################################################################################

