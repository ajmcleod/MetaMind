#!/usr/bin/env python

import numpy as np
import theano
from theano import tensor as T
import itertools

class theano_layer:


  #######################################################################################################################

  def initialize_parameters(self, W, b):

    if W == None:
      rng = np.random.RandomState()
      self.W = theano.shared(rng.uniform(low = - np.sqrt(6. / np.sum(self.W_shape)),
                                            high = np.sqrt(6. / np.sum(self.W_shape)),
                                            size = self.W_shape).astype(np.float32), borrow=True)
    else:
      self.W = theano.shared(W, borrow=True)

    if b == None:
      self.b = theano.shared(np.zeros((self.W_shape[0])).astype(np.float32), borrow=True)
    else:
      self.b = theano.shared(b, borrow=True)

  #######################################################################################################################

  def reset_gradient_sums(self):
    self.W_gradient_sums = theano.shared(1e-8 * np.ones(self.W_shape), borrow=True)
    self.b_gradient_sums = theano.shared(1e-8 * np.ones((self.W_shape[0],)), borrow=True)

  #######################################################################################################################

  def reset_gradient_velocities(self):
    self.W_gradient_velocity = theano.shared(np.zeros(self.W_shape), borrow=True)
    self.b_gradient_velocity = theano.shared(np.zeros((self.W_shape[0],)), borrow = True)


class input_layer:

  def __init__(self, X_var, X_values, input_shape):

    self.masked_output = X_var
    self.output = X_values
    self.output_shape = input_shape

