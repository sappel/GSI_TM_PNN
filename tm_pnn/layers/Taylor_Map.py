import numpy as np

import keras
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Sequential

import tensorflow as tf


class TaylorMap(Layer):
    def __init__(self, output_dim, order=1,
                 weights_regularizer = None,
                 initial_weights = None,
                 **kwargs):
        self.output_dim = output_dim
        self.order = order
        self.initial_weights = initial_weights
        self.weights_regularizer = weights_regularizer

        super(TaylorMap, self).__init__(**kwargs)


    def build(self, input_shape):
        input_dim = input_shape[1]
        self.input_dim = input_dim
        nsize = 1
        self.W = []
        self.nsizes = [nsize]
        for i in range(self.order+1):
            if self.initial_weights is None:
                initial_weight_value = np.zeros((nsize, self.output_dim))
            else:
                initial_weight_value = self.initial_weights[i]
            nsize*=input_dim
            self.nsizes.append(nsize)
            self.W.append(K.variable(initial_weight_value))

        if self.initial_weights is None:
            self.W[1] = (K.variable(np.eye(N=input_dim, M=self.output_dim)))

        self.trainable_weights = self.W

        return


    def call(self, x, mask=None):
        ans = self.W[0]
        tmp = x
        x_vectors = tf.expand_dims(x, -1)
        for i in range(1, self.order+1):
            ans = ans + K.dot(tmp, self.W[i])
            if(i == self.order):
                continue
            xext_vectors = tf.expand_dims(tmp, -1)
            x_extend_matrix = tf.matmul(x_vectors, xext_vectors, adjoint_a=False, adjoint_b=True)
            tmp = tf.reshape(x_extend_matrix, [-1, self.nsizes[i+1]])

        if self.weights_regularizer:
            self.add_loss(self.weights_regularizer(self.W))
        return ans


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)
