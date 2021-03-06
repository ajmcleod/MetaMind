#!/usr/bin/env python

import time
import numpy as np
import theano
import os.path
import cPickle as pickle
from theano import tensor as T
from sklearn import metrics
from glove import *
from input_layer import *
from convolution_layer import *
from max_pool_layer import *
from fully_connected_layer import *
from softmax_layer import *


class conv_net:

  #######################################################################################################################

    def initialize_model(self, model_directory = None):

        print ''
        self.layers = [input_layer(self)]
        for ii in range(len(self.network_architecture)):

            layer_info = self.network_architecture[ii]

            if 'activation' in layer_info:
                layer_activation = layer_info['activation']
            else:
                self.network_architecture[ii]['activation'] = None
                layer_activation = None

            if 'dropout_prob' in layer_info:
                dropout_prob = layer_info['dropout_prob']
            else:
                self.network_architecture[ii]['doprout_prob'] = 0.0
                dropout_prob = 0.0

            if layer_activation == 'maxout':
                if  'maxout_depth' in layer_info:
                    maxout_depth = layer_info['maxout_depth']
                else:
                    self.network_architecture[ii]['maxout_depth'] = 2
                    maxout_depth = 2
            else:
                self.network_architecture[ii]['maxout_depth'] = 1
                maxout_depth = 1

            if layer_info['type'] == 'convolution' and 'stride' in layer_info and 'depth' in layer_info and 'pad' in layer_info:
                self.layers.append(convolution_layer(self, previous_layer=self.layers[ii], stride=layer_info['stride'], depth=layer_info['depth'], pad=layer_info['pad'], activation=layer_activation, dropout_prob=dropout_prob, maxout_depth=maxout_depth, model_directory=model_directory))
            elif layer_info['type'] == 'max_pool' and 'stride' in layer_info:
                self.layers.append(max_pool_layer(self, previous_layer=self.layers[ii], stride=layer_info['stride'], dropout_prob=dropout_prob))
            elif layer_info['type'] == 'fully_connected' and 'depth' in layer_info:
                self.layers.append(fully_connected_layer(self, previous_layer=self.layers[ii], depth=layer_info['depth'], activation=layer_activation, dropout_prob=dropout_prob, maxout_depth=maxout_depth, model_directory=model_directory))
            elif layer_info['type'] == 'softmax':
                self.layers.append(softmax_layer(self, previous_layer=self.layers[ii], y_var=self.y, dropout_prob=dropout_prob, model_directory=model_directory))
                if (ii + 1) < len(self.network_architecture):
                    print '\nThe Softmax Layer must be the last layer of the network. No further layers will be added beyond Softmax Layer 1 \n'
                    break
            else:
                print '\nNot enough information to initlizize layer %i \n' % ii 
                sys.exit(0)
        print ''

  #######################################################################################################################

    def set_training_environment(self):

        self.learning_rate         = np.float32(self.learning_rate)
        self.reg_strength          = np.float32(self.reg_strength)
        self.momentum_decay_rate   = np.float32(self.momentum_decay_rate)
        self.rms_decay_rate        = np.float32(self.rms_decay_rate)

        for ii in range(len(self.network_architecture)):
            self.layers[ii+1].configure_outputs(self)
            self.network_architecture[ii]['dropout_prob'] = self.layers[ii+1].dropout_prob

        parameter_updates = []
        for current_layer in self.layers[1:]:
            if current_layer.layer_type in ('convolution','fully_connected','softmax'):
                current_layer.configure_updates(self, cost_function=self.layers[-1].training_cost)
                parameter_updates.extend(current_layer.parameter_updates)

        self.train_batch          = theano.function(inputs = [self.indices], outputs = [], updates = parameter_updates,
                                                    givens = {self.X: T.take(self.X_train, self.indices, axis = 0),
                                                              self.y: T.take(self.y_train, self.indices, axis = 0)})
                           
        self.total_cost_batch     = theano.function(inputs = [self.index], outputs = self.layers[-1].total_cost, updates = None,
                                                    givens = {self.X: self.X_train[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                                                              self.y: self.y_train[self.index * self.batch_size: (self.index + 1) * self.batch_size]})

        self.num_correct_batch    = theano.function(inputs = [self.index], outputs = self.layers[-1].num_correct, updates = None,
                                                    givens = {self.X: self.X_train[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                                                              self.y: self.y_train[self.index * self.batch_size: (self.index + 1) * self.batch_size]})

        self.current_training_parameters = {'learning_rate': self.learning_rate,
                                            'batch_size': self.batch_size,
                                            'reg_strength': self.reg_strength,
                                            'rms_decay_rate': self.rms_decay_rate,
                                            'rms_injection_rate': self.rms_injection_rate,
                                            'use_nesterov_momentum': self.use_nesterov_momentum,
                                            'momentum_decay_rate': self.momentum_decay_rate}

        print 'training environment set with...'
        for key,value in self.current_training_parameters.items():
            print('{} : {}'.format(key, value))
        print ''

        self.training_log = self.training_log + 'training environment set with... \n'
        for key,value in self.current_training_parameters.items():
            self.training_log = self.training_log + '{} : {}'.format(key, value) + '\n'
        for ii in range(len(self.network_architecture)):
            self.training_log = self.training_log + 'layer ' + str(ii+1) + ' dropout_prob : ' + str(self.network_architecture[ii]['dropout_prob']) + '\n'
        self.training_log = self.training_log + '\n'


  #######################################################################################################################

    def train(self, num_batches, compute_accuracy = False):
        start_time = time.clock()
        for ii in range(num_batches):
            self.batch_step += 1
            self.train_batch(np.int32(self.rng.random_integers(0, self.num_train_examples - 1, (self.batch_size))))
        end_time = time.clock()
        self.training_log = self.training_log + 'trained for %i batch steps, new training accuracy %f %% \n' % \
                                             (num_batches, 100 * self.training_accuracy())

  #######################################################################################################################

    def verbose_train(self, num_batches, print_interval = 100):
        self.batch_step = 0
        start_time = time.clock()
        if (self.batch_step % print_interval == 0):
            print '%i batch steps completed \ncurrent training cost: %f \n' % (self.batch_step, self.training_cost())
        for ii in range(num_batches):
            self.train_batch(np.int32(self.rng.random_integers(0, self.num_train_examples - 1, (self.batch_size))))
            self.batch_step += 1
            if (self.batch_step % print_interval == 0):
                print '%i batch steps completed \ncurrent training cost: %f \n' % (self.batch_step, self.training_cost())
        end_time = time.clock()
        print 'softmax trained for %i batches in %f seconds with final training cost %f and final training accuracy %f %% ' % \
                               (num_batches, end_time - start_time, self.training_cost(), 100 * self.training_accuracy())
        self.training_log = self.training_log + 'trained for %i epochs on %i images, new training cost %f and training accuracy %f %% \n \n' % \
                               (num_batches, self.batch_size, self.training_cost(), 100 * self.training_accuracy())

  #######################################################################################################################

    def old_train(self, num_batches, compute_accuracy = False):
        start_time = time.clock()

        for ii in range(num_batches):

            self.batch_step += 1
            if (self.batch_step % 100 == 100):
                print '%i batch steps completed\n \ncurrent training accuracy: %f %% \n' % (self.batch_step, 100 * self.training_accuracy())
            self.train_batch(np.int32(self.rng.random_integers(0, self.num_train_examples - 1, (self.batch_size))))

#        if current_cost < self.lowest_cost:
#          self.lowest_cost = current_cost
#          self.record_best_parameters()
#        if compute_accuracy:
#          print 'train_accuracy: %f' % (self.L5.train_accuracy.eval())
#        print 'negative_log_likelihood: %f \n' % (current_cost)

            end_time = time.clock()
#      self.revert_to_best_parameter_set()
#      best_train_accuracy = self.L5.train_accuracy.eval()
        print 'softmax trained for %i batches in %f seconds with final training accuracy %f %%' % \
                               (num_batches, end_time - start_time, 100 * self.training_accuracy())
        self.training_log = self.training_log + 'trained for %i epochs on %i images, new training accuracy %f %% \n' % \
                               (num_batches, self.num_train_examples, 100 * self.training_accuracy())

  #######################################################################################################################

    def save_model(self, directory):

        full_path = '/farmshare/user_data/ajmcleod/machine_learning/galzoo/trained_models/' + directory 
        if not os.path.exists(full_path):
            print 'Saving model to ' + directory 
        else:
            timestamp = time.strftime("_%d_%b_%Y_%X", time.localtime())
            full_path = full_path + timestamp
            print 'Model with that name already exists. Saving model to ' + directory + timestamp
        os.mkdir(full_path)

        with open(full_path + '/training_log.txt', 'a') as f:
            f.write(self.training_log)

        pickle.dump(self.network_architecture, open(full_path + '/network_architecture.p', 'wb'))

        for ii in range(len(self.layers[1:])):
            if self.layers[ii+1].layer_type != 'max_pool':
                for jj in range(len(self.layers[ii+1].W)):
                    np.savez_compressed(full_path + '/layer_' + str(ii+1) + '_filter_parameters_' + str(jj), W = self.layers[ii+1].W[jj].get_value(), b = self.layers[ii+1].b[jj].get_value())


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

  #######################################################################################################################

    def training_cost(self):
        return float(np.sum([self.total_cost_batch(ii) for ii in np.arange(self.num_train_batches)])) / float(self.num_train_examples)

  #######################################################################################################################

    def training_accuracy(self):
        return float(np.sum([self.num_correct_batch(ii) for ii in np.arange(self.num_train_batches)])) / float(self.num_train_examples)

  #######################################################################################################################

    def accuracy(self, X, y):
        return T.mean(T.eq(T.argmax(T.nnet.softmax(T.dot(X, self.L3.W) + self.L3.b), axis = 1), y))

  #######################################################################################################################

    def accuracy_np(self, X, y):
        return np.mean(np.equal(np.argmax(T.nnet.softmax(T.dot(X, self.softmaxLayer0.W) + self.softmaxLayer0.b).eval(), axis=1), y))
