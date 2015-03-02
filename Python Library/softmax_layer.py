#!/usr/bin/env python

import numpy as np
import theano
from theano import tensor as T

class softmax:

  def __init__(self, input, num_features, num_classes, W = None, b = None, random_seed = None):

    rand = np.random.RandomState(random_seed)

    if W == None:
      initialized_W = rand.uniform(low = - np.sqrt(6. / (num_features + num_classes)),
                                   high = np.sqrt(6. / (num_features + num_classes)),
                                   size = (num_features, num_classes))
    else:
      initialized_W = W

    if b == None:
      initialized_b = np.zeros((num_classes,))
    else:
      initialized_b = b

    self.num_features = num_features
    self.num_classes = num_classes
    self.W = theano.shared(initialized_W, dtype=theano.config.floatX), borrow=True)
    self.b = theano.shared(initialized_b, dtype=theano.config.floatX), borrow=True)
    self.params = [self.W, self.b]
    self.input = input
    self.output = T.nnet.softmax(T.dot(self.input, self.W) + self.b)


