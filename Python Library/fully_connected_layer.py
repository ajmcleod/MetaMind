#!/usr/bin/env python

from theano_layer import *

class fully_connected_layer(theano_layer):

    def __init__(self, parameters, previous_layer, depth, activation = None, dropout_prob = 0.0, maxout_depth = 2, batch_normalize = False, model_directory = None):
        self.layer_id = self._ids.next()
        self.layer_type = 'fully_connected'

        self.previous_layer     = previous_layer
        self.num_features       = np.prod(previous_layer.output_shape)
        self.W_shape            = (depth, self.num_features)
        self.output_shape       = (depth)
        self.activation         = activation
        self.dropout_prob       = dropout_prob
        self.batch_normalize    = batch_normalize

        if self.activation == 'maxout':
            self.maxout_depth  = maxout_depth
        else:
            self.maxout_depth  = 1
        self.initialize_filters(parameters, model_directory)

        self.initialize_gradient_sums()
        self.initialize_gradient_velocities()

        print 'Fully Connected Layer %i initialized' % (self.layer_id)
    
  #######################################################################################################################

    def configure_outputs(self, parameters):

        output_list = []
        masked_output_list = []
        X_full = self.previous_layer.output.flatten(2)
        if self.dropout_prob > 0.0:
            dropout_mask  = parameters.trng.binomial(n = 1, p = 1 - self.dropout_prob, size = self.previous_layer.masked_output.flatten(2).shape, dtype = 'float32') / self.dropout_prob
            X_masked = self.previous_layer.masked_output.flatten(2) * dropout_mask
        else:
            X_masked = self.previous_layer.masked_output.flatten(2)

        for ii in range(self.maxout_depth):
            output_list.append(T.dot(X_full, self.W[ii].T) + self.b[ii])
            masked_output_list.append(T.dot(X_masked, self.W[ii].T) + self.b[ii])

        self.output, self.masked_output = self.activation_function(output_list, masked_output_list)
