#!/usr/bin/env python

import numpy as np
import theano
from theano import tensor as T
from theano.tensor.signal import downsample
import itertools
from theano_layer import *

class max_pool_layer(theano_layer):
  _ids = itertools.count(1)

  def __init__(self, parameters, X_full, X_masked, input_shape, stride):
    self.layer_id = self._ids.next()

    self.output = T.signal.downsample.max_pool_2d(input = X_full, ds = (stride, stride), ignore_border = False)
    self.output_shape = (input_shape[0], input_shape[1] / stride, input_shape[2] / stride)

    if T.gt(parameters.dropout_prob, 0.0):
      self.dropout_mask  = parameters.trng.binomial(n = 1, p = 1 - parameters.dropout_prob, size = X_masked.shape, dtype = 'float32') / parameters.dropout_prob
      self.masked_output = T.signal.downsample.max_pool_2d(input = X_masked * self.dropout_mask, ds = (stride, stride), ignore_border = False)
    else:
      self.masked_output = T.signal.downsample.max_pool_2d(input = X_masked, ds = (stride, stride), ignore_border = False)

    print 'Maxpool Layer %i initialized' % (self.layer_id)


