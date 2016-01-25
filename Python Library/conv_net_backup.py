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

    def __init__(self, training_examples, training_labels, network_architecture, batch_size = 1000, learning_rate = 1e-4,
                 reg_strength = 1e-3, rms_decay_rate = 0.9, momentum_decay_rate = 0.5, rms_injection_rate = None, 
                 load_model = None, use_nesterov_momentum = False, random_seed = None):

        self.network_architecture  = network_architecture
        self.num_classes           = np.amax(training_labels) + 1
        self.num_train_examples    = training_examples.shape[0]
        self.input_shape           = training_examples.shape[1:]
        self.num_train_batches     = np.ceil(float(self.num_train_examples) / batch_size).astype(int)
        self.X_train               = theano.shared(training_examples, borrow=True)
        self.y_train               = theano.shared(training_labels, borrow=True)
        self.trng                  = T.shared_randomstreams.RandomStreams(random_seed)
        self.rng                   = np.random.RandomState(random_seed)
        self.indices               = T.ivector('indices')
        self.index                 = T.iscalar('index')
        self.X                     = T.ftensor4('X')
        self.y                     = T.bvector('y')

        self.batch_size            = batch_size
        self.learning_rate         = np.float32(learning_rate)
        self.reg_strength          = np.float32(reg_strength)
        self.use_nesterov_momentum = use_nesterov_momentum
        self.momentum_decay_rate   = np.float32(momentum_decay_rate)
        self.rms_decay_rate        = np.float32(rms_decay_rate)
        if rms_injection_rate == None:
            self.rms_injection_rate  = np.float32(1.0 - self.rms_decay_rate)
        else: 
            self.rms_injection_rate  = np.float32(rms_injection_rate)

        self.initialize_model()

        self.fetch_model_info(model_dir=load_model)
        self.set_training_parameters()

        self.batch_step = 0

  #######################################################################################################################

    def initialize_model(self):

        print ''
        self.layers = [input_layer(self)]
        for ii in range(len(self.network_architecture)):

            layer_info = self.network_architecture[ii]

            if 'activation' in layer_info:
                layer_activation = layer_info['activation']
            else:
                layer_activation = None

            if 'dropout_prob' in layer_info:
                dropout_prob = layer_info['dropout_prob']
            else:
                dropout_prob = 0.0

            if layer_activation == 'maxout' and 'maxout_depth' in layer_info:
                maxout_depth = layer_info['maxout_depth']
            else:
                maxout_depth = 2

            if layer_info['type'] == 'convolution' and 'stride' in layer_info and 'depth' in layer_info and 'pad' in layer_info:
                self.layers.append(convolution_layer(self, previous_layer=self.layers[ii], stride=layer_info['stride'], depth=layer_info['depth'], pad=layer_info['pad'], activation=layer_activation, dropout_prob=dropout_prob, maxout_depth=maxout_depth))
            elif layer_info['type'] == 'max_pool' and 'stride' in layer_info:
                self.layers.append(max_pool_layer(self, previous_layer=self.layers[ii], stride=layer_info['stride'], dropout_prob=dropout_prob))
            elif layer_info['type'] == 'fully_connected' and 'depth' in layer_info:
                self.layers.append(fully_connected_layer(self, previous_layer=self.layers[ii], depth=layer_info['depth'], activation=layer_activation, dropout_prob=dropout_prob, maxout_depth=maxout_depth))
            elif layer_info['type'] == 'softmax':
                self.layers.append(softmax_layer(self, previous_layer=self.layers[ii], y_var=self.y, dropout_prob=dropout_prob))
                if (ii + 1) < len(self.network_architecture):
                    print '\nThe Softmax Layer must be the last layer of the network. No further layers will be added beyond Softmax Layer 1 \n'
                    break
            else:
                print '\nNot enough information to initlizize layer %i \n' % ii 
                sys.exit(0)
        print ''


  #######################################################################################################################

    def set_training_parameters(self):

        for current_layer in self.layers[1:]:
            current_layer.configure_outputs(self)

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
# set new values in training log...

  #######################################################################################################################

    def train(self, num_batches, compute_accuracy = False):
        start_time = time.clock()
        for ii in range(num_batches):
            self.batch_step += 1
            self.train_batch(np.int32(self.rng.random_integers(0, self.num_train_examples - 1, (self.batch_size))))
        end_time = time.clock()
        self.training_log = self.training_log + 'trained for %i batch steps on %i images, new training accuracy %f %% \n' % \
                                             (num_batches, self.num_train_examples, 100 * self.training_accuracy())

  #######################################################################################################################

    def verbose_train(self, num_batches, print_interval = 100):
        start_time = time.clock()
        if (self.batch_step % print_interval == 0):
            print '%i batch steps completed \ncurrent training cost: %f \n' % (self.batch_step, self.training_cost())
        for ii in range(num_batches):
            self.batch_step += 1
            if (self.batch_step % print_interval == 0):
                print '%i batch steps completed \ncurrent training cost: %f \n' % (self.batch_step, self.training_cost())
            self.train_batch(np.int32(self.rng.random_integers(0, self.num_train_examples - 1, (self.batch_size))))
        end_time = time.clock()
        print 'softmax trained for %i batches in %f seconds with final training cost %f and final training accuracy %f %% ' % \
                               (num_batches, end_time - start_time, self.training_cost(), 100 * self.training_accuracy())
        self.training_log = self.training_log + 'trained for %i epochs on %i images, new training cost %f and training accuracy %f %% \n' % \
                               (num_batches, self.num_train_examples, self.training_cost(), 100 * self.training_accuracy())

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

        timestamp = time.strftime("_%d_%b_%Y_%X", time.localtime())
        full_path = '/farmshare/user_data/ajmcleod/machine_learning/galzoo/trained_models/' + directory + timestamp
        print 'Saving model to ' + directory + timestamp 
        os.mkdir(full_path)

        with open(full_path + '/training_log.txt', 'a') as f:
            f.write(self.training_log)

        for ii in range(len(self.layers[1:])):
            if self.layers[ii+1].layer_type != 'max_pool':
                for jj in range(len(self.layers[ii+1].W)):
                    np.savez_compressed(full_path + '/layer_' + str(ii+1) + '_W_' + str(jj), layer = self.layers[ii+1].W[jj].get_value())
                    np.savez_compressed(full_path + '/layer_' + str(ii+1) + '_b_' + str(jj), layer = self.layers[ii+1].b[jj].get_value())


  #######################################################################################################################

    def fetch_model_info(self, model_dir):

        if model_dir != None:
            full_path = '/farmshare/user_data/ajmcleod/machine_learning/galzoo/trained_models/' + model_dir
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
        else:
            self.training_log = ''


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
