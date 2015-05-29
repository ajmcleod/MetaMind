#!/usr/bin/env python

import numpy as np
import theano
from theano import tensor as T
from theano.tensor.signal import downsample
import itertools
from theano_layer import *

class max_pool_layer(theano_layer):
  _ids = itertools.count(1)

  def __init__(self, X_var, X_values, input_shape, stride, trng, batch_size = None, dropout_prob = None):
    self.layer_id = self._ids.next()

    self.output = T.signal.downsample.max_pool_2d(input = X_values, ds = (stride, stride), ignore_border = False)
    self.output_shape = (input_shape[0], input_shape[1] / stride, input_shape[2] / stride)

    if dropout_prob > 0.0:
      if batch_size == None:
        batch_size = self.num_training_examples
      self.dropout_mask  = trng.binomial(n = 1, p = 1 - dropout_prob, size = np.insert(input_shape, 0, batch_size)) / dropout_prob
      self.masked_output = T.signal.downsample.max_pool_2d(input = X_var * self.dropout_mask , ds = (stride, stride), ignore_border = False)
    else:
      self.masked_output = downsample.max_pool_2d(input = X_var, ds = (stride, stride), ignore_border = False)

    print 'Maxpool Layer %i initialized' % (self.layer_id)


