#!/usr/bin/env python

import time
import numpy as np
import theano
from theano import tensor as T
from glove import *
from sst import *

class softmax:
  def __init__(self, n_in, n_out):

    self.num_features = n_in
    self.num_classes = n_out
    self.W = theano.shared(value=np.zeros((n_in, n_out),dtype=theano.config.floatX), name='W', borrow=True)
    self.b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)
    self.params = [self.W, self.b]

  def reset(self, n_in, n_out):
    self.num_features = n_in
    self.num_classes = n_out
    self.W = theano.shared(value=np.zeros((n_in, n_out),dtype=theano.config.floatX), name='W', borrow=True)
    self.b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)

  def predict(self, test_set_X):
    y_probabilities = T.nnet.softmax(T.dot(test_set_X, self.W) + self.b)
    return T.argmax(y_probabilities, axis=1)

  def negative_log_likelihood(self, test_examples_X, true_sentiments):
    y_probabilities = T.nnet.softmax(T.dot(test_examples_X, self.W) + self.b)
    return -T.mean(T.log(y_probabilities)[T.arange(test_examples_X.shape[0]), true_sentiments])

  def accuracy(self, test_examples_X, true_sentiments):
    return T.mean(T.eq(self.predict(test_examples_X), true_sentiments))

  def train(self, train_set_X, train_set_y, batch_size = 1000, learning_rate = 0.1, cost_margin = 0.05,
            marginal_cost_tolerance = 10, reg_strength = 1e-3, rms_decay_rate = 0.9, rms_injection_rate = None):

    X = T.matrix('X')
    y = T.lvector('y')
    index = T.lscalar()
    mask = T.matrix('mask')
    cost = self.negative_log_likelihood(X, y)

    n_train_batches = np.ceil(float(train_set_X.get_value(borrow=True).shape[0]) / batch_size).astype(int)
    W_gradient_sums = theano.shared(1e-8 * np.ones_like(self.W.eval()), borrow=True)
    b_gradient_sums = theano.shared(1e-8 * np.ones_like(self.b.eval()), borrow=True)
    if rms_injection_rate == None:
      rms_injection_rate = 1 - rms_decay_rate

    g_W = T.grad(cost=cost, wrt=self.W)
    g_b = T.grad(cost=cost, wrt=self.b)
    parameter_updates = [(W_gradient_sums, rms_decay_rate * W_gradient_sums + rms_injection_rate * T.sqr(g_W)),
                         (b_gradient_sums, rms_decay_rate * b_gradient_sums + rms_injection_rate * T.sqr(g_b)),
                         (self.W, self.W - learning_rate * g_W / T.sqrt(W_gradient_sums + T.sqr(g_W)) - reg_strength * self.W),
                         (self.b, self.b - learning_rate * g_b / T.sqrt(b_gradient_sums + T.sqr(g_b)) - reg_strength * self.b)]
    
    train_model = theano.function(inputs=[index], outputs=cost, updates=parameter_updates,
      givens={X: train_set_X[index * batch_size: (index + 1) * batch_size],
              y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    lowest_cost = np.inf
    current_cost = np.inf
    marginal_iterations = 0
    epoch = 0
    start_time = time.clock()
    done_training = False
    
    while (not done_training):	

      epoch = epoch + 1
      for batch_index in xrange(n_train_batches - 1):	

        previous_cost = current_cost
        current_cost = train_model(batch_index)
        if current_cost < lowest_cost:
          lowest_cost = current_cost
          best_W = self.W.get_value()
          best_b = self.b.get_value()

        print current_cost
        if np.abs(current_cost - previous_cost) < cost_margin:
          marginal_iterations += 1
          if marginal_iterations == marginal_cost_tolerance:
            done_training = True
        else:
          marginal_iterations = 0
          
      if epoch%100 == 0:
        cost_margin *= np.log(10)

    end_time = time.clock()
    self.W = theano.shared(best_W)
    self.b = theano.shared(best_b)
    best_accuracy = self.accuracy(train_set_X, train_set_y).eval()
    print 'softmax trained in %i epochs and %f seconds with training accuracy %f %%' % (epoch, end_time - start_time, best_accuracy * 100)
           

