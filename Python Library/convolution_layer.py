#!/usr/bin/env python

import numpy as np
import theano
from theano import tensor as T
import itertools
from theano_layer import *

class convolution_layer(theano_layer):
  _ids = itertools.count(1)

  def __init__(self, X_var, X_values, input_shape, stride, depth, trng, batch_size, dropout_prob = None, W = None, b = None):
    self.layer_id = self._ids.next()

    self.X            = X_var
    self.W_shape      = (depth, input_shape[0], stride, stride)
    self.output_shape = (depth, input_shape[1], input_shape[2])
    self.border_shift = (stride - 1) // 2

    self.initialize_parameters(W, b)
    self.reset_gradient_sums()
    self.reset_gradient_velocities()

    convolution = T.nnet.conv.conv2d(input = X_values, filters = self.W,
                                     filter_shape = self.W_shape, subsample = (1,1), border_mode = 'full')[:, :,
                                     self.border_shift: input_shape[1] + self.border_shift, self.border_shift: input_shape[2] + self.border_shift]

    self.output = convolution + self.b.dimshuffle('x', 0, 'x', 'x')

    if dropout_prob > 0.0:
      self.dropout_mask  = trng.binomial(n = 1, p = 1 - dropout_prob, size = np.insert(input_shape, 0, batch_size), dtype = 'float32') / dropout_prob
      self.masked_output = T.nnet.conv.conv2d(input = self.X * self.dropout_mask[:self.X.shape[0]], filters = self.W, filter_shape = self.W_shape, subsample = (1,1), border_mode = 'full')[:, :, self.border_shift: input_shape[1] + self.border_shift, self.border_shift: input_shape[2] + self.border_shift]
    else:
      self.masked_output = T.nnet.conv.conv2d(input = self.X, filters = self.W, subsample = (1,1), border_mode = 'full')[:, :,
                                                self.border_shift: input_shape[1] + self.border_shift, self.border_shift: input_shape[2] + self.border_shift]

    print 'Convolution Layer %i initialized' % (self.layer_id)

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


