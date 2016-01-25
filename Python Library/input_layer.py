#!/usr/bin/env python

from theano_layer import *

class input_layer(theano_layer):

    def __init__(self, parameters):
        self.layer_type = 'input'

        self.activation    = None
        self.output        = parameters.X
        self.masked_output = parameters.X
        self.output_shape  = parameters.input_shape
