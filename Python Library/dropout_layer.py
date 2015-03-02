#!/usr/bin/env python 

import theano
import theano.tensor as T
import numpy as np

class dropout_layer:

  def __init__(self, input, random_seed = None):
    if random_seed == None:
      self.rand = T.shared_randomstreams.RandomStreams(np.random.randint(999999))
    else:
      self.rand = T.shared_randomstreams.RandomStreams(random_seed)
    self.input = input

  def output(self, dropout):
    return input * self.rand.binomial(n = 1, p = 1 - dropout, size = input.shape)
