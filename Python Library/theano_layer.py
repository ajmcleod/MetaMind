#!/usr/bin/env python

import numpy as np
import theano
from theano import tensor as T
import itertools

class theano_layer:
    _ids = itertools.count(1)

  #######################################################################################################################

    def initialize_filters(self, parameters, model_directory=None):

        if model_directory:
            filter_parameters = np.load(model_directory + '/layer_' + str(self.layer_id) + '_filter_parameters_0.npz')
            self.W = [theano.shared(filter_parameters['W'])]
            self.b = [theano.shared(filter_parameters['b'])]
        else:
            self.W = [self.generate_W(parameters)]
            self.b = [theano.shared(np.zeros((self.W_shape[0]), dtype = np.float32), borrow=True)]

        if self.activation == 'maxout':
            if model_directory:
                for ii in range(self.maxout_depth - 1):
                    filter_parameters = np.load(model_directory + '/layer_' + str(self.layer_id) + '_filter_parameters_' + str(ii+1) + '.npz')
                    self.W.append(theano.shared(filter_parameters['W']))
                    self.b.append(theano.shared(filter_parameters['b']))
            else:
                for ii in range(self.maxout_depth - 1):
                    self.W.append(self.generate_W(parameters))
                    self.b.append(theano.shared(np.zeros((self.W_shape[0]), dtype = np.float32), borrow=True))


  #######################################################################################################################

    def generate_W(self, parameters):
        return theano.shared(parameters.rng.uniform(low = - np.sqrt(6. / np.sum(self.W_shape)),
                                                   high = np.sqrt(6. / np.sum(self.W_shape)),
                                                   size = self.W_shape).astype(np.float32), borrow=True)

  #######################################################################################################################

    def initialize_gradient_sums(self):
        self.W_gradient_sums = []
        self.b_gradient_sums = []
        for ii in range(len(self.W)):
            self.W_gradient_sums.append(theano.shared(1e-8 * np.ones(self.W_shape, dtype = np.float32), borrow=True))
        for ii in range(len(self.b)):
            self.b_gradient_sums.append(theano.shared(1e-8 * np.ones((self.W_shape[0],), dtype = np.float32), borrow=True))

  #######################################################################################################################

    def initialize_gradient_velocities(self):
        self.W_gradient_velocity = []
        self.b_gradient_velocity = []
        for ii in range(len(self.W)):
            self.W_gradient_velocity.append(theano.shared(np.zeros(self.W_shape, dtype=np.float32), borrow=True))
        for ii in range(len(self.b)):
            self.b_gradient_velocity.append(theano.shared(np.zeros((self.W_shape[0],), dtype=np.float32), borrow=True))

  #######################################################################################################################
 
    def activation_function(self, output_list, masked_output_list):

        if self.activation == 'relu':
            output = T.maximum(output_list[0], 0)
            masked_output = T.maximum(masked_output_list[0], 0)

        elif self.activation == 'maxout':
            current_output_max          = T.maximum(output_list[self.maxout_depth - 2], output_list[self.maxout_depth - 1])
            current_masked_output_max   = T.maximum(masked_output_list[self.maxout_depth - 2], masked_output_list[self.maxout_depth - 1])
            for ii in range(self.maxout_depth - 2):
                current_output_max        = T.maximum(output_list[ii], current_output_max)
                current_masked_output_max = T.maximum(masked_output_list[ii], current_masked_output_max)
            output = current_output_max
            masked_output = current_masked_output_max

        else: 
            output = output_list[0] 
            masked_output = masked_output_list[0]

        return (output, masked_output)

  #######################################################################################################################

    def configure_updates(self, parameters, cost_function):

        self.parameter_updates = []

        for ii in range(len(self.W)):
            g_W = T.grad(cost=cost_function, wrt=self.W[ii])
            if parameters.use_nesterov_momentum:
                W_update = self.W_gradient_velocity[ii] * np.float32(parameters.momentum_decay_rate**2) - (np.float32(1) + parameters.momentum_decay_rate) * parameters.learning_rate * g_W
            else:
                W_update = - parameters.learning_rate * g_W
            self.parameter_updates.append((self.W[ii], self.W[ii] + W_update / T.sqrt(self.W_gradient_sums[ii] + T.sqr(g_W)) - parameters.reg_strength * self.W[ii]))
            self.parameter_updates.append((self.W_gradient_sums[ii], parameters.rms_decay_rate * self.W_gradient_sums[ii] + parameters.rms_injection_rate * T.sqr(W_update / parameters.learning_rate)))
            if parameters.use_nesterov_momentum:
                self.parameter_updates.append((self.W_gradient_velocity[ii], parameters.momentum_decay_rate * self.W_gradient_velocity[ii] - parameters.learning_rate * g_W))

        for ii in range(len(self.b)):
            g_b = T.grad(cost=cost_function, wrt=self.b[ii])
            if parameters.use_nesterov_momentum:
                b_update = self.b_gradient_velocity[ii] * np.float32(parameters.momentum_decay_rate**2) - (np.float32(1) + parameters.momentum_decay_rate) * parameters.learning_rate * g_b
            else:
                b_update = - parameters.learning_rate * g_b
            self.parameter_updates.append((self.b[ii], self.b[ii] + b_update / T.sqrt(self.b_gradient_sums[ii] + T.sqr(g_b)) - parameters.reg_strength * self.b[ii]))
            self.parameter_updates.append((self.b_gradient_sums[ii], parameters.rms_decay_rate * self.b_gradient_sums[ii] + parameters.rms_injection_rate * T.sqr(b_update / parameters.learning_rate)))
            if parameters.use_nesterov_momentum:
                self.parameter_updates.append((self.b_gradient_velocity[ii], parameters.momentum_decay_rate * self.b_gradient_velocity[ii] - parameters.learning_rate * g_b))

