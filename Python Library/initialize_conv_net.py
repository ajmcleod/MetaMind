#!/usr/bin/env python

from conv_net import *


class initialize_conv_net(conv_net):

    def __init__(self, training_examples, training_labels, network_architecture, batch_size = 1000, learning_rate = 1e-4,
                 reg_strength = 1e-3, rms_decay_rate = 0.9, momentum_decay_rate = 0.5, rms_injection_rate = None, 
                 load_model = None, use_nesterov_momentum = False, random_seed = None):

        self.training_log          = ''
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
        self.learning_rate         = learning_rate
        self.reg_strength          = reg_strength
        self.use_nesterov_momentum = use_nesterov_momentum
        self.momentum_decay_rate   = momentum_decay_rate
        self.rms_decay_rate        = rms_decay_rate
        if rms_injection_rate == None:
            self.rms_injection_rate  = 1.0 - self.rms_decay_rate
        else: 
            self.rms_injection_rate  = rms_injection_rate

        self.initialize_model(model_directory=None)
        self.set_training_environment()

        self.batch_step = 0

