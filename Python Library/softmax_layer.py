#!/usr/bin/env python

import numpy as np
import theano
from theano import tensor as T
import itertools
from theano_layer import *

class softmax_layer(theano_layer):
  _ids = itertools.count(1)

  def __init__(self, parameters, X_full, X_masked, y_var, input_shape, training_options = None, W = None, b = None):
    self.layer_id = self._ids.next()

    self.X_full       = X_full.flatten(2)
    self.X_masked     = X_masked.flatten(2)
    self.y            = y_var
    self.num_features = np.prod(input_shape)
    self.W_shape      = (parameters.num_classes, np.prod(input_shape))

    self.initialize_parameters(W, b)
    self.reset_gradient_sums()
    self.reset_gradient_velocities()

    if T.gt(parameters.dropout_prob, 0.0):
      self.dropout_mask                 = parameters.trng.binomial(n = 1, p = 1 - parameters.dropout_prob, size=self.X_masked.shape, dtype='float32') / parameters.dropout_prob
      self.masked_log_likelihood        = T.log(T.nnet.softmax(T.dot(self.dropout_mask * self.X_masked, self.W.T) + self.b))
    else:
      self.masked_log_likelihood        = T.log(T.nnet.softmax(T.dot(self.X_masked, self.W.T) + self.b))

    self.masked_negative_log_likelihood = - T.mean(self.masked_log_likelihood[T.arange(self.y.shape[0]), self.y])
    self.training_cost = - T.mean(T.log(T.nnet.softmax(T.dot(self.X_masked, self.W.T) + self.b))[T.arange(self.y.shape[0]), self.y])
#    self.accuracy = T.mean(T.eq(T.argmax(T.nnet.softmax(T.dot(self.X, self.W.T) + self.b), axis = 1), self.y), dtype = 'float32')

    print 'Softmax Layer %i initialized' % (self.layer_id)

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

  def predict(self, X_test):
    return T.argmax(T.nnet.softmax(T.dot(X_test, self.W.T) + self.b), axis = 1)
