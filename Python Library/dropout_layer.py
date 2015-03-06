#!/usr/bin/env python 

import theano
import theano.tensor as T
import numpy as np

class dropout_layer:

  def __init__(self, inputs, random_seed = None):
    if random_seed == None:
      self.rand = T.shared_randomstreams.RandomStreams(np.random.randint(999999))
    else:
      self.rand = T.shared_randomstreams.RandomStreams(random_seed)
    self.inputs = inputs

  def output(self, dropout_prob):
    print self.rand.binomial(n = 1, p = 1 - dropout_prob, size = (2,3))
    return self.inputs * self.rand.binomial(n = 1, p = 1 - dropout_prob, size = self.inputs.shape) / dropout_prob
