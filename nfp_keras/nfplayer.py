#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
#
# Author:   Graphium
# URL:      http://tehutehu.com
# License:  MIT License
# Created:  2016-11-12
#
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializations, regularizers

import numpy as np


class NFPLayer(Layer):
    '''
        NeuralFingerprint implementation with keras

        # Arguments
            N_dim_atom: dimension of atom_feature
            N_dim_bond: dimension of bond_feature
            radius: radius od NFP(default: 4)
            large_weights: if True, NFP layer has big random weight(to compare with ECFP)
    '''
    def __init__(self, output_dim, N_dim_atom, N_dim_bond, radius=4,
                 init='glorot_normal', large_weights=False, H_regularizer=None,
                 W_regularizer=None, **kwargs):
        self.output_dim = output_dim
        self.N_dim_atom = N_dim_atom
        self.N_dim_bond = N_dim_bond
        self.radius = radius

        self.init = initializations.get(init)
        self.H_regularizer = regularizers.get(H_regularizer)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.large_weights = large_weights
        super(NFPLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        h_len = self.N_dim_atom + self.N_dim_bond
        self.H = [self.init((h_len, h_len), name='{}_H{}'.format(self.name, i)) for i in range(self.radius)]
        self.W = [self.init((h_len, self.output_dim), name='{}_W{}'.format(self.name, i)) for i in range(self.radius)]

        if self.large_weights:
            self.H = [h * 1000000 for h in self.H]
            self.W = [w * 1000000 for w in self.W]

        self.regularizers = []
        if self.H_regularizer:
            for h in self.H:
                self.H_regularizer.set_param(h)
            self.regularizers.append(self.H_regularizer)

        if self.W_regularizer:
            for w in self.W:
                self.W_regularizer.set_param(w)
            self.regularizers.append(self.W_regularizer)

        self.trainable_weights = self.H + self.W

    def call(self, x, mask=None):
        atom_num = x[0].astype("int32")[0]
        output = K.variable(np.zeros([self.output_dim]))

        atom_feature_len = atom_num * self.N_dim_atom
        adj_mat_len = atom_num * atom_num
        bond_atom_len = atom_num * atom_num * self.N_dim_bond

        # reshape 1-d vector to 2-d and 3-d matrix
        feature = K.theano.tensor.reshape(x[:, 1:1 + atom_feature_len], [atom_num, self.N_dim_atom])
        adj_mat = K.theano.tensor.reshape(x[:, 1 + atom_feature_len: 1 + atom_feature_len + adj_mat_len], [atom_num, atom_num])
        bond_feature = K.theano.tensor.reshape(x[:, 1 + atom_feature_len + adj_mat_len: 1 + atom_feature_len + adj_mat_len + bond_atom_len], [atom_num, atom_num, self.N_dim_bond])

        for i in range(self.radius):
            bond_ = K.theano.tensor.diagonal(K.dot(adj_mat, bond_feature), 0, 1, 0).T
            atom_ = K.dot(adj_mat, feature[:, :self.N_dim_atom])
            v = K.theano.tensor.concatenate([bond_, atom_], axis=1)
            feature = K.sigmoid(K.dot(v, self.H[i]))
            i = K.softmax(K.dot(feature, self.W[i]))
            output += K.sum(i, axis=0)

        return output.dimshuffle(['x', 0])

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)
