#!/usr/bin/env python

import numpy as np
import theano
from theano import tensor as T
import itertools
from theano_layer import *

class convolution_layer(theano_layer):
  _ids = itertools.count(1)

  def __init__(self, parameters, X_full, X_masked, input_shape, stride, depth, pad = True, W = None, b = None):
    self.layer_id = self._ids.next()

    self.W_shape = (depth, input_shape[0], stride, stride)
    self.initialize_parameters(W, b)
    self.reset_gradient_sums()
    self.reset_gradient_velocities()

    if pad != True:

      self.output_shape = (depth, input_shape[1] - stride + 1, input_shape[2] - stride + 1)
      self.output = T.nnet.conv.conv2d(input = X_full, filters = self.W, filter_shape = self.W_shape, subsample = (1,1), border_mode = 'valid') + self.b.dimshuffle('x', 0, 'x', 'x')

      if T.gt(parameters.dropout_prob,0.0):
        self.dropout_mask  = parameters.trng.binomial(n = 1, p = 1 - parameters.dropout_prob, size = X_masked.shape, dtype = 'float32') / parameters.dropout_prob
        self.masked_output = T.nnet.conv.conv2d(input = X_masked * self.dropout_mask[:X_masked.shape[0]], filters = self.W, filter_shape = self.W_shape, subsample = (1,1), border_mode = 'valid') + self.b.dimshuffle('x', 0, 'x', 'x')
      else:
        self.masked_output = T.nnet.conv.conv2d(input = X_masked, filters = self.W, filter_shape = self.W_shape, subsample = (1,1), border_mode = 'valid') + self.b.dimshuffle('x', 0, 'x', 'x')

    else:

      border_shift = (stride - 1) // 2
      self.output_shape = (depth, input_shape[1], input_shape[2])
      self.output = T.nnet.conv.conv2d(input = X_full, filters = self.W, filter_shape = self.W_shape, subsample = (1,1), border_mode = 'full')[:, :, border_shift: input_shape[1] + border_shift, border_shift: input_shape[2] + border_shift] + self.b.dimshuffle('x', 0, 'x', 'x')

      if T.gt(parameters.dropout_prob,0.0):
        self.dropout_mask  = parameters.trng.binomial(n = 1, p = 1 - parameters.dropout_prob, size = X_masked.shape, dtype = 'float32') / parameters.parameters.dropout_prob
        self.masked_output = T.nnet.conv.conv2d(input = X_masked * self.dropout_mask, filters = self.W, filter_shape = self.W_shape, subsample = (1,1), border_mode = 'full')[:, :, border_shift: input_shape[1] + border_shift, border_shift: input_shape[2] + border_shift] + self.b.dimshuffle('x', 0, 'x', 'x')
      else:
        self.masked_output = self.output 


    print 'Convolution Layer %i initialized' % (self.layer_id)

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


