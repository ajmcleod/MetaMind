#!/usr/bin/env python

import numpy as np
import theano
from theano import tensor as T
import itertools
from theano_layer import *

class fully_connected_layer(theano_layer):
  _ids = itertools.count(1)

  def __init__(self, parameters, X_full, X_masked, input_shape, num_neurons, W = None, b = None):
    self.layer_id = self._ids.next()

    self.X_full             = X_full.flatten(2)
    self.X_masked           = X_masked.flatten(2)
    self.num_features       = np.prod(input_shape)
    self.W_shape            = (num_neurons, self.num_features)
    self.num_output_neurons = num_neurons

    self.initialize_parameters(W, b)
    self.reset_gradient_sums()
    self.reset_gradient_velocities()

    self.output = T.dot(self.X_full, self.W.T) + self.b
    self.output_shape = num_neurons

    if T.gt(parameters.dropout_prob, 0.0):
      self.dropout_mask  = parameters.trng.binomial(n = 1, p = 1 - parameters.dropout_prob, size=self.X_masked.shape, dtype = 'float32') / parameters.dropout_prob
      self.masked_output = T.dot(self.X_masked * self.dropout_mask, self.W.T) + self.b
    else:
      self.masked_output = T.dot(self.X_masked, self.W.T) + self.b

    print 'Fully Connected Layer %i initialized' % (self.layer_id)
    
  #######################################################################################################################

  def configure_training_environment(self, parameters, cost_function):

    self.g_W = T.grad(cost=cost_function, wrt=self.W)
    self.g_b = T.grad(cost=cost_function, wrt=self.b)

    if parameters.use_nesterov_momentum:
      W_update = self.W_gradient_velocity * parameters.momentum_decay_rate * parameters.momentum_decay_rate - (np.float32(1) + parameters.momentum_decay_rate) * parameters.learning_rate * self.g_W
      b_update = self.b_gradient_velocity * parameters.momentum_decay_rate * parameters.momentum_decay_rate - (np.float32(1) + parameters.momentum_decay_rate) * parameters.learning_rate * self.g_b
    else:
      W_update = - parameters.learning_rate * self.g_W
      b_update = - parameters.learning_rate * self.g_b

    self.parameter_updates = [(self.W, self.W + W_update / T.sqrt(self.W_gradient_sums + T.sqr(self.g_W)) - parameters.reg_strength * self.W),
                              (self.b, self.b + b_update / T.sqrt(self.b_gradient_sums + T.sqr(self.g_b)) - parameters.reg_strength * self.b),
                              (self.W_gradient_sums, parameters.rms_decay_rate * self.W_gradient_sums + parameters.rms_injection_rate * T.sqr(W_update / parameters.learning_rate)),
                              (self.b_gradient_sums, parameters.rms_decay_rate * self.b_gradient_sums + parameters.rms_injection_rate * T.sqr(b_update / parameters.learning_rate))]

    if parameters.use_nesterov_momentum:
      self.parameter_updates.append((self.W_gradient_velocity, parameters.momentum_decay_rate * self.W_gradient_velocity - parameters.learning_rate * self.g_W))
      self.parameter_updates.append((self.b_gradient_velocity, parameters.momentum_decay_rate * self.b_gradient_velocity - parameters.learning_rate * self.g_b))

  #######################################################################################################################
  #######################################################################################################################

  def predict(self, X_test):
    return T.dot(X_test, self.W) + self.b

  #######################################################################################################################

