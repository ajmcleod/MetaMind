#!/usr/bin/env python

from theano_layer import *

class convolution_layer(theano_layer):

    def __init__(self, parameters, previous_layer, stride, depth, pad = True, activation = None, dropout_prob = 0.0, maxout_depth = 2, batch_normalize = False, model_directory=None):
        self.layer_id = self._ids.next()
        self.layer_type = 'convolution'

        self.previous_layer    = previous_layer
        self.W_shape           = (depth, self.previous_layer.output_shape[0], stride, stride)
        self.depth             = depth
        self.stride            = stride
        self.pad               = pad
        self.activation        = activation
        self.dropout_prob      = dropout_prob
        self.batch_normalize   = batch_normalize

        if self.activation == 'maxout':
            self.maxout_depth  = maxout_depth
        else:
            self.maxout_depth  = 1

        self.initialize_filters(parameters, model_directory)

        if pad:
            self.border_shift  = (stride - 1) // 2
            self.output_shape  = (depth, previous_layer.output_shape[1], previous_layer.output_shape[2])
        else:
            self.output_shape  = (depth, previous_layer.output_shape[1] - stride + 1, previous_layer.output_shape[2] - stride + 1)

        self.initialize_gradient_sums()
        self.initialize_gradient_velocities()

        print 'Convolution Layer %i initialized' % (self.layer_id)

  #######################################################################################################################

    def configure_outputs(self, parameters):

        output_list = []
        masked_output_list = []
        X_full = self.previous_layer.output
        if self.dropout_prob > 0.0:
            dropout_mask = parameters.trng.binomial(n = 1, p = 1 - self.dropout_prob, size = self.previous_layer.masked_output.shape, dtype = 'float32') / self.dropout_prob
            X_masked = self.previous_layer.masked_output * dropout_mask
        else:
            X_masked = self.previous_layer.masked_output

        for ii in range(self.maxout_depth):
            if self.pad:
                output_list.append(T.nnet.conv.conv2d(input = X_full, filters = self.W[ii], filter_shape = self.W_shape, subsample = (1,1), border_mode = 'full')[:, :, self.border_shift: self.previous_layer.output_shape[1] + self.border_shift, self.border_shift: self.previous_layer.output_shape[2] + self.border_shift] + self.b[ii].dimshuffle('x', 0, 'x', 'x'))
                masked_output_list.append(T.nnet.conv.conv2d(input = X_masked, filters = self.W[ii], filter_shape = self.W_shape, subsample = (1,1), border_mode = 'full')[:, :, self.border_shift: self.previous_layer.output_shape[1] + self.border_shift, self.border_shift: self.previous_layer.output_shape[2] + self.border_shift] + self.b[ii].dimshuffle('x', 0, 'x', 'x'))
            else:
                output_list.append(T.nnet.conv.conv2d(input = X_full, filters = self.W[ii], filter_shape = self.W_shape, subsample = (1,1), border_mode = 'valid') + self.b[ii].dimshuffle('x', 0, 'x', 'x'))
                masked_output_list.append(T.nnet.conv.conv2d(input = X_masked, filters = self.W[ii], filter_shape = self.W_shape, subsample = (1,1), border_mode = 'valid') + self.b[ii].dimshuffle('x', 0, 'x', 'x'))

        self.output, self.masked_output = self.activation_function(output_list, masked_output_list)



