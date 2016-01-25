#!/usr/bin/env python

from theano_layer import *

class softmax_layer(theano_layer):

    def __init__(self, parameters, previous_layer, y_var, dropout_prob, model_directory=None):
        self.layer_id = self._ids.next()
        self.layer_type = 'softmax'

        self.previous_layer = previous_layer
        self.y_var          = y_var
        self.activation     = None
        self.dropout_prob   = dropout_prob
        self.num_features   = np.prod(previous_layer.output_shape)
        self.W_shape        = (parameters.num_classes, self.num_features)
  
        self.initialize_filters(parameters, model_directory)

        self.initialize_gradient_sums()
        self.initialize_gradient_velocities()

        print 'Softmax Layer %i initialized' % (self.layer_id)

  #######################################################################################################################

    def configure_outputs(self, parameters):

        X_full   = self.previous_layer.output.flatten(2)
        if self.dropout_prob > 0.0:
            dropout_mask = parameters.trng.binomial(n = 1, p = 1 - self.dropout_prob, size=self.previous_layer.masked_output.shape, dtype='float32') / self.dropout_prob
            X_masked = self.previous_layer.masked_output.flatten(2) * dropout_mask
        else:
            X_masked = self.previous_layer.masked_output.flatten(2)

        self.masked_log_likelihood = T.log(T.nnet.softmax(T.dot(X_masked, self.W[0].T) + self.b[0]))

        self.training_cost           = - T.mean(self.masked_log_likelihood[T.arange(self.y_var.shape[0]), self.y_var])
        self.total_cost              = - T.sum(T.log(T.nnet.softmax(T.dot(X_full, self.W[0].T) + self.b[0]))[T.arange(self.y_var.shape[0]), self.y_var])
        self.num_correct             = T.sum(T.eq(T.argmax(T.dot(X_full, self.W[0].T) + self.b[0], axis=1), self.y_var))

