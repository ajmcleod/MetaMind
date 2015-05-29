#!/usr/bin/env python

import time
import numpy as np
import theano
import os.path
import cPickle as pickle
from theano import tensor as T
from sklearn import metrics
from glove import *
from sst import *
from softmax_layer import *
from fully_connected_layer import *
from convolution_layer import *
from max_pool_layer import *


class conv_net:

  def __init__(self, training_examples, training_labels, filter_parameters, batch_size = 1000, learning_rate = 1e-4,
               reg_strength = 1e-3, rms_decay_rate = 0.9, rms_injection_rate = None, dropout_prob = 0.5,
               load_model = None, use_nesterov_momentum = False, momentum_decay_rate = None, random_seed = None):

    self.num_classes        = np.amax(training_labels) + 1
    self.num_train_batches  = np.ceil(float(training_examples.shape[0]) / batch_size).astype(int)
    if rms_injection_rate == None:
      rms_injection_rate = 1 - rms_decay_rate

    dropout_prob          = np.float32(dropout_prob)
    X                     = T.ftensor4('X')
    y                     = T.lvector('y')
    index                 = T.lscalar('index')
    trng                  = T.shared_randomstreams.RandomStreams(random_seed)
    self.X_train          = theano.shared(training_examples, borrow=True)
    self.y_train          = theano.shared(training_labels , borrow=True)
    training_options = {'learning_rate': learning_rate, 'batch_size': batch_size,
                        'dropout_prob': dropout_prob, 'reg_strength': reg_strength,
                        'rms_decay_rate': rms_decay_rate, 'rms_injection_rate': rms_injection_rate,
                        'use_nesterov_momenum': use_nesterov_momentum, 'momentum_decay_rate': momentum_decay_rate,
                        'reset_gradient_sums': True, 'reset_gradient_velocities': True}

    if load_model != None:
      self.fetch_model_info(load_model)
    else:
      self.training_log = ''
      self.L0_W_init = None
      self.L0_b_init = None
      self.L2_W_init = None
      self.L2_b_init = None
      self.L4_W_init = None
      self.L4_b_init = None
      self.L5_W_init = None
      self.L5_b_init = None

    self.L0 = convolution_layer(X_var = X, X_values = self.X_train, trng = trng, W = self.L0_W_init, b = self.L0_b_init,
                                stride = filter_parameters[0][0], depth = filter_parameters[0][1],
                                dropout_prob = dropout_prob, batch_size = batch_size, input_shape = training_examples.shape[1:])
    self.L1 = max_pool_layer(X_var = self.L0.masked_output, X_values = self.L0.output, stride = filter_parameters[1][0],
                             trng = trng, dropout_prob = dropout_prob, batch_size = batch_size, input_shape = self.L0.output_shape)
    self.L2 = convolution_layer(X_var = T.maximum(self.L1.masked_output, 0), X_values = T.maximum(self.L1.output, 0),
                                trng = trng, stride = filter_parameters[2][0], depth = filter_parameters[2][1], input_shape = self.L1.output_shape,
                                dropout_prob = dropout_prob, batch_size = batch_size, W = self.L2_W_init, b = self.L2_b_init)
    self.L3 = max_pool_layer(X_var = self.L2.masked_output, X_values = self.L2.output, stride = filter_parameters[3][0],
                             trng = trng, dropout_prob = dropout_prob, batch_size = batch_size, input_shape = self.L2.output_shape)
    self.L4 = fully_connected_layer(X_var = T.maximum(self.L3.masked_output, 0), X_values = T.maximum(self.L3.output, 0),
                                    trng = trng, num_neurons = filter_parameters[4][0], W = self.L4_W_init, b = self.L4_b_init,
                                    dropout_prob = dropout_prob, batch_size = batch_size, input_shape = self.L3.output_shape)
    self.L5 = softmax_layer(X_var = T.maximum(self.L4.masked_output, 0), y_var = y, X_values = T.maximum(self.L4.output, 0), input_shape = self.L4.output_shape,
                            y_values = self.y_train, trng = trng, training_options = training_options, W = self.L5_W_init, b = self.L5_b_init)

    self.L0.configure_training_environment(self.L5.training_cost, learning_rate = learning_rate,
                                           reg_strength = reg_strength, rms_decay_rate = rms_decay_rate,
                                           rms_injection_rate = rms_injection_rate,
                                           use_nesterov_momentum = use_nesterov_momentum,
                                           momentum_decay_rate = momentum_decay_rate)
    self.L2.configure_training_environment(self.L5.training_cost, learning_rate = learning_rate,
                                           reg_strength = reg_strength, rms_decay_rate = rms_decay_rate,
                                           rms_injection_rate = rms_injection_rate,
                                           use_nesterov_momentum = use_nesterov_momentum,
                                           momentum_decay_rate = momentum_decay_rate)
    self.L4.configure_training_environment(self.L5.training_cost, learning_rate = learning_rate,
                                           reg_strength = reg_strength, rms_decay_rate = rms_decay_rate,
                                           rms_injection_rate = rms_injection_rate,
                                           use_nesterov_momentum = use_nesterov_momentum,
                                           momentum_decay_rate = momentum_decay_rate)

    parameter_updates = self.L5.parameter_updates
    parameter_updates.extend(self.L4.parameter_updates)
    parameter_updates.extend(self.L2.parameter_updates)
    parameter_updates.extend(self.L0.parameter_updates)

    self.train_model = theano.function(inputs = [index], outputs = [], updates = parameter_updates,
                                  givens = {X: self.X_train[index * batch_size: (index + 1) * batch_size],
                                            y: self.y_train[index * batch_size: (index + 1) * batch_size]}, on_unused_input='ignore')

    self.lowest_cost = np.inf
    self.initialize_best_paramater_set(filter_parameters)
    self.epoch = 0
    print ''

  #######################################################################################################################

  def train(self, num_epochs, compute_accuracy = False):
    start_time = time.clock()

    for ii in range(num_epochs):
      self.epoch += 1
      print 'starting epoch %i \n' % (ii + 1)
      for batch_index in xrange(self.num_train_batches):	
#        current_cost = self.train_model(batch_index)
        self.train_model(batch_index)
        print 'batch complete'
#        if current_cost < self.lowest_cost:
#          self.lowest_cost = current_cost
#          self.record_best_parameters()
#        if compute_accuracy:
#          print 'train_accuracy: %f' % (self.L5.train_accuracy.eval())
#        print 'negative_log_likelihood: %f \n' % (current_cost)

      end_time = time.clock()
#      self.revert_to_best_parameter_set()
      best_train_accuracy = self.L5.train_accuracy.eval()
    print 'softmax trained in %i epochs and %f seconds with training accuracy %f %%' % \
                               (num_epochs, end_time - start_time, 100 * best_train_accuracy)
    self.training_log = self.training_log + 'trained for %i epochs on %i images, new training accuracy %f %% \n' % \
                               (num_epochs, training_examples.shape[0], 100 * best_train_accuracy)

  #######################################################################################################################

  def save_model(self, directory):

    full_path = '/Users/thatscottishkid/Google Drive/Stanford/Machine Learning/cs231n/Trained Models/' + directory
    if os.path.exists(full_path):
      print 'This save directory already exists. Saving model to ' + directory + time.strftime("_%d_%b_%Y_%X", time.localtime())
      full_path = full_path + time.strftime("_%d_%b_%Y_%X", time.localtime())
    os.mkdir(full_path)

    with open(full_path + '/training_log.txt', 'a') as f:
            f.write(self.training_log)

    with open(full_path + '/L0.W', 'wb') as f:
      pickle.dump(self.L0.W.eval(), f)
    with open(full_path + '/L0.b', 'wb') as f:
      pickle.dump(self.L0.b.eval(), f)
    print 'Layer 0 parameters saved'
    with open(full_path + '/L2.W', 'wb') as f:
      pickle.dump(self.L2.W.eval(), f)
    with open(full_path + '/L2.b', 'wb') as f:
      pickle.dump(self.L2.b.eval(), f)
    print 'Layer 2 parameters saved'
    with open(full_path + '/L4.W', 'wb') as f:
      pickle.dump(self.L4.W.eval(), f)
    with open(full_path + '/L4.b', 'wb') as f:
      pickle.dump(self.L4.b.eval(), f)
    print 'Layer 4 parameters saved'
    with open(full_path + '/L5.W', 'wb') as f:
      pickle.dump(self.L5.W.eval(), f)
    with open(full_path + '/L5.b', 'wb') as f:
      pickle.dump(self.L5.b.eval(), f)
    print 'Layer 5 parameters saved'

  #######################################################################################################################

  def fetch_model_info(self, model_dir):

    full_path = '/Users/thatscottishkid/Google Drive/Stanford/Machine Learning/cs231n/Trained Models/' + model_dir

    with open(full_path + '/L0.W', 'rb') as f:
      self.L0_W_init = pickle.load(f)
    with open(full_path + '/L0.b', 'rb') as f:
      self.L0_b_init = pickle.load(f)
    with open(full_path + '/L2.W', 'rb') as f:
      self.L2_W_init = pickle.load(f)
    with open(full_path + '/L2.b', 'rb') as f:
      self.L2_b_init = pickle.load(f)
    with open(full_path + '/L4.W', 'rb') as f:
      self.L4_W_init = pickle.load(f)
    with open(full_path + '/L4.b', 'rb') as f:
      self.L4_b_init = pickle.load(f)
    with open(full_path + '/L5.W', 'rb') as f:
      self.L5_W_init = pickle.load(f)
    with open(full_path + '/L5.b', 'rb') as f:
      self.L5_b_init = pickle.load(f)

    with open(full_path + '/training_log.txt', 'r') as f:
      self.training_log = f.read()

  #######################################################################################################################

  def initialize_best_paramater_set(self, filter_parameters):
    self.best_W0 = theano.shared(np.zeros(self.L0.W.eval().shape))
    self.best_b0 = theano.shared(np.zeros(self.L0.b.eval().shape))
    self.best_W2 = theano.shared(np.zeros(self.L2.W.eval().shape))
    self.best_b2 = theano.shared(np.zeros(self.L2.b.eval().shape))
    self.best_W4 = theano.shared(np.zeros(self.L4.W.eval().shape))
    self.best_b4 = theano.shared(np.zeros(self.L4.b.eval().shape))
    self.best_W5 = theano.shared(np.zeros(self.L5.W.eval().shape))
    self.best_b5 = theano.shared(np.zeros(self.L5.b.eval().shape))
    self.record_best_parameters = theano.function(inputs = [], outputs = [],
                                                  updates = [(self.best_W0, self.L0.W),
                                                             (self.best_b0, self.L0.b),
                                                             (self.best_W2, self.L2.W),
                                                             (self.best_b2, self.L2.b),
                                                             (self.best_W4, self.L4.W),
                                                             (self.best_b4, self.L4.b),
                                                             (self.best_W5, self.L5.W),
                                                             (self.best_b5, self.L5.b)])
    self.revert_to_best_parameter_set = theano.function(inputs = [], outputs = [],
                                                        updates = [(self.L0.W, self.best_W0),
                                                                   (self.L0.b, self.best_b0),
                                                                   (self.L2.W, self.best_W2),
                                                                   (self.L2.b, self.best_b2),
                                                                   (self.L4.W, self.best_W4),
                                                                   (self.L4.b, self.best_b4),
                                                                   (self.L5.W, self.best_W5),
                                                                   (self.L5.b, self.best_b5)])

  def accuracy(self, X, y):
    return T.mean(T.eq(T.argmax(T.nnet.softmax(T.dot(X, self.L3.W) + self.L3.b), axis = 1), y))

  def accuracy_np(self, X, y):
    return np.mean(np.equal(np.argmax(T.nnet.softmax(T.dot(X, self.softmaxLayer0.W) + self.softmaxLayer0.b).eval(), axis=1), y))

  def f1(self, X, y):
    return metrics.f1_score(y, T.argmax(T.nnet.softmax(T.dot(X, self.softmaxLayer0.W) + self.softmaxLayer0.b), axis = 1).eval())

