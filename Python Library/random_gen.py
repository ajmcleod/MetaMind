#!/usr/bin/env python

import numpy as np
import theano
from theano import tensor as T

class random_gen:

  def __init__(self, batch_size, random_seed = None):

    self.batch_size = batch_size
    self.rand = np.random.RandomState(random_seed)

  def initialize_W(self, num_features, num_classes):
    return self.rand.uniform(low = - np.sqrt(6. / (num_features + num_classes)),
                      high = np.sqrt(6. / (num_features + num_classes)),
                      size = (num_features, num_classes))

  def initialize_b(self, num_classes):
    return self.rand.uniform(low = - np.sqrt(6. / (num_classes)),
                      high = np.sqrt(6. / (num_classes)),
                      size = (num_classes,))

  def generate_mask(self, num_features, dropout_prob):
    return self.rand.binomial(n = 1, p = 1 - dropout_prob, size=(self.batch_size, num_features)).astype(float) / dropout_prob

