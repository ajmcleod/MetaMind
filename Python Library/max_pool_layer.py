#!/usr/bin/env python

from theano.tensor.signal import downsample
from theano_layer import *

class max_pool_layer(theano_layer):

    def __init__(self, parameters, previous_layer, stride, dropout_prob = 0.0, batch_normalize = False):
        self.layer_id = self._ids.next()
        self.layer_type = 'max_pool'

        self.previous_layer  = previous_layer
        self.stride          = stride
        self.dropout_prob    = dropout_prob
        self.batch_normalize = batch_normalize
        self.output_shape    = (previous_layer.output_shape[0], previous_layer.output_shape[1] / stride, previous_layer.output_shape[2] / stride)

        print 'Maxpool Layer %i initialized' % (self.layer_id)

  #######################################################################################################################

    def configure_outputs(self, parameters):

        X_full   = self.previous_layer.output
        if self.dropout_prob > 0.0:
            dropout_mask  = parameters.trng.binomial(n = 1, p = 1 - self.dropout_prob, size = self.previous_layer.masked_output.shape, dtype = 'float32') / self.dropout_prob
            X_masked = self.previous_layer.masked_output * dropout_mask 
        else:
            X_masked = self.previous_layer.masked_output

        self.output = T.signal.downsample.max_pool_2d(input = X_full, ds = (self.stride, self.stride), ignore_border = False)
        self.masked_output = T.signal.downsample.max_pool_2d(input = X_masked, ds = (self.stride, self.stride), ignore_border = False)


